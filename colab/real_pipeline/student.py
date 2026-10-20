"""Ettin-68M dual heads, soft-label BCE, graph-pair Matryoshka InfoNCE."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .common import (HEALTH, MODELS, environment, sha256, validate_input,
                     validate_labeled, validate_pairs, write_json)
from .checkpoints import recover_checkpoint, save_checkpoint
from .contracts import STUDENT_CONTRACT, model_identity
from .language import LANGUAGE_CONTRACT, language_histogram, language_scope


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        model_id, revision = MODELS["student"]
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        config.reference_compile = False
        self.encoder = AutoModel.from_pretrained(model_id, revision=revision, config=config,
                                                 attn_implementation="eager")
        self.health_component_logits = nn.Linear(config.hidden_size, len(HEALTH))
        self.matryoshka_emb = nn.Linear(config.hidden_size, 256)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.health_component_logits(pooled), F.normalize(self.matryoshka_emb(pooled), dim=-1)


def tokenize(tokenizer, texts, device, max_length):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}


def matryoshka_loss(anchor, positive, negative, community_ids, negative_community_ids, temperature=0.07):
    losses = []
    size = anchor.shape[0]
    # Other same-community positives are not negatives. The explicit negative
    # pool gets the same mask to avoid contradictory labels across a batch.
    community = torch.as_tensor(community_ids, device=anchor.device)
    negative_community = torch.as_tensor(negative_community_ids, device=anchor.device)
    diagonal = torch.eye(size, dtype=torch.bool, device=anchor.device)
    false_positive_negatives = community[:, None].eq(community[None, :]) & ~diagonal
    false_explicit_negatives = community[:, None].eq(negative_community[None, :])
    mask = torch.cat((false_positive_negatives, false_explicit_negatives), dim=1)
    for dim in (256, 128, 64):
        a = F.normalize(anchor[:, :dim], dim=-1)
        p = F.normalize(positive[:, :dim], dim=-1)
        n = F.normalize(negative[:, :dim], dim=-1)
        scores = (a @ torch.cat((p, n), dim=0).T) / temperature
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        losses.append(F.cross_entropy(scores, torch.arange(size, device=anchor.device)))
    return torch.stack(losses).mean()


def train(args, rows):
    torch.set_num_threads(args.threads)
    torch.manual_seed(1729)
    random.seed(1729)
    np.random.seed(1729)
    seal = validate_input(args, rows)
    manifest = validate_labeled(args.input, rows, seal, args.split, args.mode)
    language_audit = language_histogram(rows)
    input_rows = len(rows)
    # Null/unscored values are absence of model evidence, never zero-valued targets.
    rows = [row for row in rows if language_scope(row)["in_scope"]]
    if not rows:
        raise ValueError("No declared-English scored rows are available for training")
    for row in rows:
        for label in HEALTH:
            value = row[f"teacher_{label}"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid real teacher target")
    if args.objective == "health" and args.pairs:
        raise ValueError("Health-only objective must not silently consume a social pair bank")
    pairs, pair_manifest = validate_pairs(args.pairs, args.holdout, args.export_manifest) if args.pairs else ([], None)
    pair_input_rows = len(pairs)
    pairs = [row for row in pairs if all(language_scope(row, prefix)["in_scope"]
                                        for prefix in ("anchor", "positive", "negative"))]
    pair_scope = {"input_rows": pair_input_rows, "eligible_declared_english_rows": len(pairs),
                  "excluded_missing_mixed_or_unsupported_language_rows": pair_input_rows - len(pairs)}
    if args.mode != "smoke" and args.objective == "joint" and not pairs:
        raise ValueError("Full dual-head training requires verified v1 graph pairs")
    if args.mode == "smoke" and pairs:
        permitted = {r["uri"] for r in rows}
        pairs = [r for r in pairs if all(r[f"{p}_uri"] in permitted for p in ("anchor", "positive", "negative"))]
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    software = environment()
    run_config = {"student": list(MODELS["student"]), "student_provenance": model_identity("student"),
                  "student_contract": STUDENT_CONTRACT, "language_contract": LANGUAGE_CONTRACT,
                  "objective": args.objective, "input_rows": input_rows, "language_audit": language_audit,
                  "pair_language_scope": pair_scope,
                  "software_identity": {key: software[key] for key in ("packages", "implementation_source_sha256")},
                  "bank_sha256": sha256(args.input),
                  "holdout_sha256": sha256(args.holdout), "pairs": pair_manifest,
                  "mode": args.mode, "rows": len(rows), "max_length": args.max_length,
                  "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                  "social_weight": args.social_weight, "steps": args.steps}
    run_path = destination / "run.json"
    if run_path.exists():
        if not args.resume:
            raise ValueError("Existing student run requires explicit --resume; choose a new directory for a new run")
        if json.loads(run_path.read_text())["config"] != run_config:
            raise ValueError("Existing student configuration differs")
    elif any(destination.iterdir()):
        raise ValueError("Student output directory is nonempty without run identity")
    recovery = None
    if args.resume:
        recovery = recover_checkpoint(
            destination, config=run_config, requested=args.resume,
            loader=lambda path: torch.load(path, map_location="cpu", weights_only=False),
        )
        checkpoint = recovery.payload
    if not run_path.exists():
        write_json(run_path, {"config": run_config, "environment": environment(),
                   "teacher_provenance": manifest["teacher_provenance"], "thresholds_locked": False})
    tokenizer = AutoTokenizer.from_pretrained(MODELS["student"][0], revision=MODELS["student"][1])
    model = Student().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start = 0
    if args.resume:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(args.device)
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if args.device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if args.device == "mps" and checkpoint.get("mps_rng") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        start = recovery.step
    # Fixed real batch measures fitting progress only. It is never called an eval.
    probe = rows if args.mode == "smoke" else rows[:min(32, len(rows))]
    probe_inputs = tokenize(tokenizer, [r["text"] for r in probe], args.device, args.max_length)
    probe_targets = torch.tensor([[r[f"teacher_{label}"] for label in HEALTH] for r in probe], device=args.device)

    def probe_loss():
        model.eval()
        with torch.no_grad():
            logits, embeddings = model(**probe_inputs)
            value = float(F.binary_cross_entropy_with_logits(logits, probe_targets))
        return value, list(logits.shape), list(embeddings.shape)

    initial, logit_shape, embedding_shape = probe_loss()
    losses = []
    started = time.monotonic()
    for step in range(start, args.steps):
        model.train()
        selected = random.sample(rows, min(args.batch_size, len(rows)))
        inputs = tokenize(tokenizer, [r["text"] for r in selected], args.device, args.max_length)
        target = torch.tensor([[r[f"teacher_{label}"] for label in HEALTH] for r in selected], device=args.device)
        logits, _ = model(**inputs)
        health_loss = F.binary_cross_entropy_with_logits(logits, target)
        social_loss = torch.tensor(0.0, device=args.device)
        if pairs:
            triplets = random.sample(pairs, min(args.batch_size, len(pairs)))
            embeddings = []
            for prefix in ("anchor", "positive", "negative"):
                encoded = tokenize(tokenizer, [r[f"{prefix}_text"] for r in triplets], args.device, args.max_length)
                embeddings.append(model(**encoded)[1])
            social_loss = matryoshka_loss(*embeddings, [r["community_id"] for r in triplets],
                                          [r["negative_community_id"] for r in triplets])
        loss = health_loss + args.social_weight * social_loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {"step": step + 1, "loss": float(loss.detach()), "health_loss": float(health_loss.detach()),
                  "social_loss": float(social_loss.detach()) if pairs else None}
        losses.append(record)
        print(json.dumps(record), flush=True)
        with (destination / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save_checkpoint(destination / f"checkpoint-{step+1:07d}.pt",
                            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1,
                             "config": run_config, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                             "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None,
                             "mps_rng": torch.mps.get_rng_state() if args.device == "mps" else None},
                            step=step + 1, config=run_config, serializer=torch.save)
    final, _, _ = probe_loss()
    receipt = {"mode": args.mode, "input_rows": input_rows, "rows": len(rows),
               "language_audit": language_audit, "pair_language_scope": pair_scope,
               "student_contract": STUDENT_CONTRACT, "health_component_logits_shape": logit_shape,
               "matryoshka_emb_shape": embedding_shape, "initial_training_probe_bce": initial,
               "final_training_probe_bce": final, "training_probe_loss_decreased": final < initial,
               "social_training_pairs_available": len(pairs), "social_head_trained": bool(pairs),
               "completed_steps_this_invocation": len(losses), "seconds": time.monotonic() - started,
               "checkpoint_recovery": recovery.info if recovery else None,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "F1": None, "credits": None, "human_reviewed_quality": False}
    receipt_path = destination / "training_receipt.json"
    if receipt_path.exists():
        write_json(destination / f"training_resume_receipt-{time.time_ns()}.json", receipt)
    else:
        write_json(receipt_path, receipt)
    tokenizer.save_pretrained(destination / "tokenizer")
    export_onnx(model.cpu().eval(), tokenizer, probe, destination, args.max_length, args.threads,
                checkpoint_path=destination / f"checkpoint-{args.steps:07d}.pt")
    if args.mode == "smoke" and losses and not final < initial:
        raise RuntimeError("Measured smoke training probe loss did not decrease; see receipt")


def export_onnx(model, tokenizer, rows, destination, max_length, threads, *, checkpoint_path):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    destination = Path(destination)
    export_identity = {"checkpoint_sha256": sha256(checkpoint_path), "student_contract": STUDENT_CONTRACT,
                       "max_length": max_length, "language_contract": LANGUAGE_CONTRACT,
                       "tokenizer": list(MODELS["student"]), "software_identity": environment()}
    example = tokenize(tokenizer, [r["text"] for r in rows[:2]], "cpu", max_length)
    previous_receipt = destination / "onnx_receipt.json"
    if previous_receipt.exists():
        prior = json.loads(previous_receipt.read_text())
        if prior.get("export_identity") == export_identity and all(item["numerical_parity_passed"] and sha256(destination / item["path"]) == item["sha256"] for item in prior["artifacts"]):
            return
    stale = [path for path in destination.glob("*.onnx")]
    if previous_receipt.exists():
        stale.append(previous_receipt)
    if stale:
        archived = destination / f"export-previous-{time.time_ns()}"
        archived.mkdir()
        for path in stale:
            path.rename(archived / path.name)
    fp32 = destination / "ettin_components_v3_social_fp32.onnx"
    int8 = destination / "ettin_components_v3_social_int8.onnx"
    torch.onnx.export(model, (example["input_ids"], example["attention_mask"]), fp32,
                      input_names=["input_ids", "attention_mask"], output_names=["health_component_logits", "matryoshka_emb"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                                    "health_component_logits": {0: "batch"}, "matryoshka_emb": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(fp32))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"], per_channel=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    parity = []
    for path in (fp32, int8):
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        cases = []
        selections = [(rows[start:start + 8], max_length, None) for start in range(0, len(rows), 8)]
        selections += [(rows[:1], min(64, max_length), "max_length"), (rows[-2:], max_length, "max_length")]
        for selected, length, padding in selections:
            if padding:
                encoded = tokenizer([r["text"] for r in selected], padding=padding, truncation=True,
                                    max_length=length, return_tensors="pt")
                encoded = {key: encoded[key] for key in ("input_ids", "attention_mask")}
            else:
                encoded = tokenize(tokenizer, [r["text"] for r in selected], "cpu", length)
            with torch.no_grad():
                reference = [x.numpy() for x in model(**encoded)]
            actual = session.run(None, {k: v.numpy() for k, v in encoded.items()})
            cases.append({"batch": len(selected), "sequence": encoded['input_ids'].shape[1],
                          "uris": [r["uri"] for r in selected], "padding": padding or "longest_in_batch",
                          "health_logits_max_abs_error": float(np.max(np.abs(actual[0] - reference[0]))),
                          "health_probability_max_abs_error": float(np.max(np.abs(1/(1+np.exp(-actual[0])) - 1/(1+np.exp(-reference[0]))))),
                          "embedding_max_abs_error": float(np.max(np.abs(actual[1] - reference[1]))),
                          "embedding_min_cosine": float(np.min(np.sum(actual[1] * reference[1], axis=1) /
                                                               (np.linalg.norm(actual[1], axis=1)*np.linalg.norm(reference[1], axis=1))))})
        passed = all(c["health_probability_max_abs_error"] <= (0.03 if path == int8 else 1e-4)
                     and c["embedding_min_cosine"] >= (0.98 if path == int8 else 0.9999) for c in cases)
        parity.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                       "cases": cases, "numerical_parity_passed": passed})
    receipt = {"artifacts": parity, "export_identity": export_identity,
               "qualification": "numerical diagnostic only; independent evaluation contract is separate",
               "historical_max_error_gate_preserved": True,
               "int8_target_bytes_approx": [65_000_000, 70_000_000],
               "target_is_expectation_not_measurement": True, "human_quality_validation": False, "execution_environment": environment(),
               "quantization": "onnxruntime.quantization.quantize_dynamic QInt8 MatMul+Gather per_channel=True",
               "unique_real_posts_checked": len({r["uri"] for r in rows}),
               "output_contract": {"health_component_logits": list(HEALTH), "matryoshka_emb": 256,
                                   "prefixes": "L2-normalize each 128/64 prefix after slicing"}}
    write_json(destination / "onnx_receipt.json", receipt)
    if not all(item["numerical_parity_passed"] for item in parity):
        raise RuntimeError("ONNX numerical parity failed; inspect onnx_receipt.json")
