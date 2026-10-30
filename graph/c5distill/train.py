"""C5 student: fine-tune a multilingual encoder on the Claude teacher labels and score it against the S1 bar.

  train.py --task stance|account_type|community_type --data DIR --model PATH --out DIR
           [--unit-cap 20] [--rate UNITS_PER_H] [--preamble-min 10] [--save-model]
           [--sanity --sanity-train 3000 --sanity-eval 1000 --epochs 1]

The rules (splits, hyperparameters, bars) are rules_c5.json; this file implements them.
- stance: 5-way stance head on (parent, child); "hostile" = P(hostile) >= t with t chosen on val. The hostile
  bar (S1 9.7) is kappa vs Sonnet >= 0.6 on the held-out split AND the 95% cluster-bootstrap CI lower bound
  of the student's kappa on the held-out pairs that carry a Haiku label > Haiku-vs-Sonnet kappa on them.
- account_type / community_type: held-out macro-F1 vs the Claude label, 95% CI lower bound >= 0.6.
The held-out ("test") split is read only after training and selection have finished. --sanity removes it
before anything else and scores a pseudo-test carved out of val instead.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c5lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = json.load(open(os.path.join(HERE, "rules_c5.json")))
STANCE = RULES["labels"]["stance"]
ACCT = RULES["labels"]["account_type_classes"]
FILES = {"stance": "c5_pairs.parquet", "account_type": "c5_profiles.parquet", "community_type": "c5_communities.parquet"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(task, data_dir):
    rows = pq.read_table(os.path.join(data_dir, FILES[task])).to_pylist()
    if task == "stance":
        classes = STANCE
        for r in rows:
            r["a"] = ("quote: " if r["kind"] == "quote" else "reply: ") + r["parent_text"]
            r["b"] = r["child_text"]
            r["y"] = classes.index(r["sonnet_stance"])
            r["id"] = r["pair_id"]
    else:
        classes = ACCT if task == "account_type" else sorted({r["label"] for r in rows})
        for i, r in enumerate(rows):
            r["a"], r["b"] = r["text"], None
            r["y"] = classes.index(r["label"])
            r["id"] = r["pid"] if task == "account_type" else str(r["community"])
            r["cluster"] = i
    return rows, classes


def split_rows(rows, A):
    if not A.sanity:
        return ([r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"],
                [r for r in rows if r["split"] == "test"])
    rows = [r for r in rows if r["split"] != "test"]  # the held-out split is never touched in sanity mode
    rng = random.Random(c5lib.SEED)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    half = lambda r: int(hashlib.blake2b(f"c5sanity|{r['cluster']}".encode(), digest_size=4).hexdigest(), 16) % 2
    pv, pt = [r for r in va if half(r) == 0], [r for r in va if half(r) == 1]
    rng.shuffle(tr), rng.shuffle(pv), rng.shuffle(pt)
    return tr[:A.sanity_train], pv[:A.sanity_eval], pt[:A.sanity_eval]


class Student(torch.nn.Module):
    def __init__(self, path, n_out, revision=None):
        super().__init__()
        from transformers import AutoModel
        kw = {"revision": revision} if revision else {}
        self.enc = AutoModel.from_pretrained(path, **kw)
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, n_out)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def batches(rows, bs, tok, max_len, device, shuffle, rng):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    else:
        order.sort(key=lambda i: len(rows[i]["a"]) + len(rows[i]["b"] or ""))
    for s in range(0, len(order), bs):
        chunk = [rows[i] for i in order[s:s + bs]]
        a = [r["a"] for r in chunk]
        b = [r["b"] for r in chunk] if chunk[0]["b"] is not None else None
        enc = tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield ([order[s + j] for j in range(len(chunk))], enc["input_ids"].to(device),
               enc["attention_mask"].to(device), torch.tensor([r["y"] for r in chunk], device=device))


def predict(model, rows, tok, max_len, bs, device, amp):
    model.eval()
    probs = np.zeros((len(rows), model.head.out_features), dtype=np.float32)
    with torch.no_grad():
        for idx, ids, am, _ in batches(rows, bs * 2, tok, max_len, device, False, None):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                lo = model(ids, am)
            probs[idx] = torch.softmax(lo.float(), -1).cpu().numpy()
    return probs


def best_threshold(p_host, y_host):
    best = (-2.0, 0.5)
    for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
        k = c5lib.kappa_cm(c5lib.confusion(y_host, (p_host >= t).astype(int), 2))
        if k > best[0]:
            best = (k, float(t))
    return best[1], best[0]


def val_metric(task, probs, rows, classes):
    y = np.array([r["y"] for r in rows])
    if task == "stance":
        h = classes.index("hostile")
        t, k = best_threshold(probs[:, h], (y == h).astype(int))
        return k, {"hostile_kappa": k, "threshold": t,
                   "stance_kappa": c5lib.kappa_cm(c5lib.confusion(y, probs.argmax(1), len(classes)))}
    m = c5lib.macro_f1_cm(c5lib.confusion(y, probs.argmax(1), len(classes)))
    return m, {"macro_f1": m}


def score_test(task, probs, rows, classes, threshold):
    y = np.array([r["y"] for r in rows])
    cl = np.array([r["cluster"] for r in rows])
    pred = probs.argmax(1)
    out = {}
    if task == "stance":
        h = classes.index("hostile")
        yh, ph = (y == h).astype(int), (probs[:, h] >= threshold).astype(int)
        out["hostile"] = c5lib.report(yh, ph, ["not_hostile", "hostile"], cl)
        out["stance"] = c5lib.report(y, pred, classes, cl)
        hk = np.array([r["haiku_stance"] is not None for r in rows])
        yk = y[hk]
        haiku = np.array([classes.index(r["haiku_stance"]) for r in rows if r["haiku_stance"] is not None])
        sub = {"n": int(hk.sum())}
        if hk.sum():
            sub["student_hostile"] = c5lib.report(yh[hk], ph[hk], ["not_hostile", "hostile"], cl[hk])
            sub["haiku_hostile"] = c5lib.report((yk == h).astype(int), (haiku == h).astype(int),
                                                ["not_hostile", "hostile"], cl[hk])
            sub["student_stance"] = c5lib.report(yk, pred[hk], classes, cl[hk], with_ci=False)
            sub["haiku_stance"] = c5lib.report(yk, haiku, classes, cl[hk], with_ci=False)
        out["haiku_rows"] = sub
        slices = {}
        for key in ["kind", "stratum"]:
            for v in sorted({r[key] for r in rows}):
                m = np.array([r[key] == v for r in rows])
                slices[f"{key}={v}"] = {"n": int(m.sum()), "hostile_kappa": c5lib.kappa_cm(c5lib.confusion(yh[m], ph[m], 2))}
        langs = {}
        for r in rows:
            langs[r["child_langs"]] = langs.get(r["child_langs"], 0) + 1
        for v in sorted(langs, key=langs.get, reverse=True)[:6]:
            m = np.array([r["child_langs"] == v for r in rows])
            slices[f"lang={v or '?'}"] = {"n": int(m.sum()), "hostile_kappa": c5lib.kappa_cm(c5lib.confusion(yh[m], ph[m], 2))}
        out["slices"] = slices
        c1 = out["hostile"]["kappa"] >= 0.6
        c2 = bool(hk.sum()) and sub["student_hostile"]["kappa_ci95"][0] > sub["haiku_hostile"]["kappa"]
        out["bar"] = {"rule": RULES["bars"]["hostile_learnable"], "kappa_ge_0_6": c1,
                      "ci_lower_above_haiku_same_rows": c2,
                      "info_full_split_ci_lower_above_haiku": bool(hk.sum()) and out["hostile"]["kappa_ci95"][0] > sub["haiku_hostile"]["kappa"],
                      "learnable": bool(c1 and c2)}
    else:
        out["type"] = c5lib.report(y, pred, classes, None)
        if task == "account_type":
            sm = np.array([r["sonnet_account_type"] is not None for r in rows])
            sub = {"n": int(sm.sum())}
            if sm.sum():
                ys = np.array([classes.index(r["sonnet_account_type"]) for r in rows if r["sonnet_account_type"]])
                hv = [r["haiku_account_type"] for r in rows if r["sonnet_account_type"]]
                hp = np.array([classes.index(v) if v in classes else classes.index("unclear") for v in hv])
                sub["student_vs_sonnet"] = c5lib.report(ys, pred[sm], classes, None)
                sub["haiku_vs_sonnet"] = c5lib.report(ys, hp, classes, None)
            out["sonnet_rows"] = sub
            langs = {}
            for r in rows:
                langs[r["primary_language"]] = langs.get(r["primary_language"], 0) + 1
            out["slices"] = {}
            for v in sorted(langs, key=langs.get, reverse=True)[:6]:
                m = np.array([r["primary_language"] == v for r in rows])
                out["slices"][f"lang={v}"] = {"n": int(m.sum()), "macro_f1": c5lib.macro_f1_cm(c5lib.confusion(y[m], pred[m], len(classes)))}
        key = "account_type_learnable" if task == "account_type" else "community_type_learnable"
        out["bar"] = {"rule": RULES["bars"][key], "learnable": out["type"]["macro_f1_ci95"][0] >= 0.6}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(FILES))
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit-cap", type=float, default=RULES["units"]["cap"])
    ap.add_argument("--rate", type=float, help="units/hour; default from the assigned GPU name")
    ap.add_argument("--preamble-min", type=float, default=10.0, help="setup minutes already spent before this process")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--sanity-train", type=int, default=3000)
    ap.add_argument("--sanity-eval", type=int, default=1000)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--batch", type=int)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    hp = dict(RULES["training"]["stance" if A.task == "stance" else A.task])
    for k in ["epochs", "max_len", "batch"]:
        if getattr(A, k) is not None:
            hp[k] = getattr(A, k)
    if not A.sanity and any(getattr(A, k) is not None for k in ["epochs", "max_len", "batch"]):
        sys.exit("hyperparameter overrides are for --sanity only; the Colab run uses rules_c5.json")

    seed = RULES["training"]["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dname = torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type
    gpu, rate = c5lib.gpu_rate(dname) if dev.type == "cuda" else ("local", 0.0)
    if A.rate is not None:
        rate = A.rate
    if rate is None:
        sys.exit(f"unknown GPU {dname!r}: pass --rate (units/hour from `colab usage`)")
    guard = c5lib.UnitGuard(A.unit_cap, rate, A.preamble_min)
    amp = dev.type == "cuda"

    rows, classes = load(A.task, A.data)
    tr, va, te = split_rows(rows, A)
    del rows
    print(f"[c5] task={A.task} device={dname} train={len(tr)} val={len(va)} test={len(te)} sanity={A.sanity}", flush=True)

    from transformers import AutoTokenizer
    kw = {"revision": A.revision} if A.revision else {}
    tok = AutoTokenizer.from_pretrained(A.model, **kw)
    model = Student(A.model, len(classes), A.revision).to(dev)
    cnt = np.bincount([r["y"] for r in tr], minlength=len(classes)).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1))
    w = w / (w * cnt).sum() * cnt.sum()
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01, fused=dev.type == "cuda")
    steps = hp["epochs"] * math.ceil(len(tr) / hp["batch"])
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))

    rng = random.Random(seed)
    hist, best, best_state, stopped, step = [], (-9.0, None, None), None, None, 0
    t_train0 = time.time()
    seqs = toks = 0
    eval_secs = 0.0
    for ep in range(hp["epochs"]):
        model.train()
        te0 = time.time()
        nb = math.ceil(len(tr) / hp["batch"])
        for bi, (_, ids, am, y) in enumerate(batches(tr, hp["batch"], tok, hp["max_len"], dev, True, rng)):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                lo = model(ids, am)
            loss = lossf(lo.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            step += 1
            if dev.type == "mps" and step % 25 == 0:
                torch.mps.empty_cache()  # keeps the local smoke run's unified-memory footprint small
            seqs += ids.shape[0]
            toks += int(am.sum())
            if bi % 100 == 99:
                el = time.time() - te0
                rem = el / (bi + 1) * (nb - bi - 1) + max(eval_secs, 60) * 2
                print(f"[c5] ep{ep} {bi + 1}/{nb} loss={loss.item():.4f} {seqs / (time.time() - t_train0):.1f} seq/s "
                      f"units~{guard.spent():.2f}", flush=True)
                if rate and not guard.allows(rem):
                    stopped = f"unit cap: epoch {ep} batch {bi + 1}, {guard.state()}"
                    break
        if stopped:
            print("[c5] STOP", stopped, flush=True)
            break
        ev0 = time.time()
        pv = predict(model, va, tok, hp["max_len"], hp["batch"], dev, amp)
        eval_secs = time.time() - ev0
        m, info = val_metric(A.task, pv, va, classes)
        info.update(epoch=ep, train_secs=round(ev0 - te0, 1), val_secs=round(eval_secs, 1), loss=float(loss.item()))
        hist.append(info)
        print("[c5] val", json.dumps(info), flush=True)
        if m > best[0]:
            best = (m, ep, info)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        ep_secs = time.time() - te0
        if ep + 1 < hp["epochs"] and rate and not guard.allows(ep_secs + eval_secs * 2):
            stopped = f"unit cap before epoch {ep + 1}, {guard.state()}"
            print("[c5] STOP", stopped, flush=True)
            break
    train_secs = time.time() - t_train0
    if best_state is not None:
        model.load_state_dict(best_state)
    threshold = best[2]["threshold"] if (best[2] and A.task == "stance") else 0.5

    # ---- the held-out split is read here, once, by the chosen model ----
    t0 = time.time()
    pt = predict(model, te, tok, hp["max_len"], hp["batch"], dev, amp)
    test = score_test(A.task, pt, te, classes, threshold)
    test_secs = time.time() - t0

    with open(os.path.join(A.out, f"{A.task}_test_predictions.jsonl"), "w") as f:
        for r, p in zip(te, pt):
            f.write(json.dumps({"id": r["id"], "y": classes[r["y"]], "pred": classes[int(p.argmax())],
                                "probs": [round(float(x), 4) for x in p]}) + "\n")
    if A.save_model:
        from safetensors.torch import save_file
        md = os.path.join(A.out, f"{A.task}_model")
        os.makedirs(md, exist_ok=True)
        save_file({k: v.detach().to("cpu", torch.float16).contiguous() for k, v in model.state_dict().items()},
                  os.path.join(md, "student.safetensors"))
        tok.save_pretrained(md)
        model.enc.config.save_pretrained(md)
        json.dump({"classes": classes, "threshold": threshold, "pooling": "mean", "max_len": hp["max_len"]},
                  open(os.path.join(md, "head.json"), "w"))

    import transformers
    res = {"task": A.task, "mode": "sanity (pseudo-test carved from val; held-out split untouched)" if A.sanity else "registered",
           "model": A.model, "revision": A.revision, "classes": classes, "hparams": hp, "device": dname, "gpu": gpu,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()},
           "data_sha256": sha256(os.path.join(A.data, FILES[A.task])), "rules_c5_sha256": sha256(os.path.join(HERE, "rules_c5.json")),
           "rows": {"train": len(tr), "val": len(va), "test": len(te)},
           "val_history": hist, "best_epoch": best[1], "hostile_threshold": threshold if A.task == "stance" else None,
           "stopped": stopped, "timing": {"train_secs": round(train_secs, 1), "test_secs": round(test_secs, 1),
                                          "train_seq_per_s": round(seqs / max(train_secs, 1e-9), 2),
                                          "train_tokens_per_s": round(toks / max(train_secs, 1e-9), 1)},
           "units": guard.state(), "test": test}
    with open(os.path.join(A.out, f"{A.task}_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    head = test.get("hostile") or test.get("type")
    print("[c5] TEST", A.task, json.dumps({"kappa": head["kappa"], "kappa_ci95": head["kappa_ci95"],
                                           "macro_f1": head["macro_f1"], "macro_f1_ci95": head["macro_f1_ci95"],
                                           "bar": test["bar"]}), flush=True)


if __name__ == "__main__":
    main()
