"""Assemble the S7 receipt from what run_colab.sh downloaded and logged.

  receipt.py RUN_DIR        (stdlib only; writes RUN_DIR/receipt.json and prints the S7 success checks)

Reads RUN_DIR/prep_receipt.json, RUN_DIR/sessions.tsv and RUN_DIR/colab-results/: usage_{before,after}.txt,
sessions_{before,after}.txt, s7_results_<GPU>.tar (bench.json, job_receipt.json), s7_rt_*.tar (roundtrip.json),
roundtrip_result.txt, units_projected_by_driver.txt. Computes units per 10M posts per GPU with the arithmetic, and
S12's projection (plan row S12: slab ~44.9M posts, band 27-63M, plus 2.04M bios, x 1.5 inefficiency).
"""
import io
import json
import os
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s7lib  # noqa: E402

S12_SLAB, S12_SLAB_BAND, S12_BIOS = 44.9e6, (27e6, 63e6), 2.04e6


def tar_json(path, member):
    if not path.exists():
        return None
    try:
        with tarfile.open(path) as tf:
            return json.load(io.TextIOWrapper(tf.extractfile(member)))
    except (KeyError, tarfile.ReadError, json.JSONDecodeError):
        return None


def text(p):
    return p.read_text() if p.exists() else ""


def main(run):
    run = pathlib.Path(run)
    res = run / "colab-results"
    rec = {"step": "S7", "prep": json.loads(text(run / "prep_receipt.json") or "{}").get("counts"),
           "usage_before": s7lib.parse_usage(text(res / "usage_before.txt")),
           "usage_after": s7lib.parse_usage(text(res / "usage_after.txt")),
           "sessions_tsv": text(run / "sessions.tsv").splitlines(),
           "sessions_after": text(res / "sessions_after.txt").strip().splitlines()[-20:],
           "units_projected_by_driver": text(res / "units_projected_by_driver.txt").strip() or None, "gpus": {}}
    b0, b1 = rec["usage_before"]["balance"], rec["usage_after"]["balance"]
    rec["balance_delta"] = round(b0 - b1, 3) if b0 is not None and b1 is not None else None
    rec["balance_note"] = "account-wide: includes any other session running meanwhile (e.g. C5); see the driver's projection"
    checks = {}
    for gpu in ["L4", "A100"]:
        tp = res / f"s7_results_{gpu}.tar"
        bench, job = tar_json(tp, "bench.json"), tar_json(tp, "job_receipt.json")
        if not bench and not job:
            continue
        g = {"tar_sha256": s7lib.sha256_file(tp) if tp.exists() else None, "job": {k: (job or {}).get(k) for k in
             ["gpu_name", "rate", "installed", "preinstalled", "model", "skipped", "fatal", "units_est_total"]}}
        if bench:
            g["versions"] = bench.get("versions")
            g["batch_chosen"] = bench.get("batch_chosen")
            g["tokens"] = bench.get("tokens")
            g["parity"] = bench.get("parity")
            rate = (job or {}).get("rate") or s7lib.RATES[gpu]
            for k in ["random", "slab"]:
                r = bench.get(k) or {}
                if "posts_per_s" in r:
                    u, arith = s7lib.units_per_10m(r["posts_per_s"], rate)
                    g[k] = {"posts_per_s": r["posts_per_s"], "tokens_per_post_with_prompt": bench["tokens"][k]["tokens_per_post_with_prompt"],
                            "units_per_10m": round(u, 3), "arithmetic": arith, "nan_or_inf": r["nan_rows_full"] + r["nan_rows_stored"] + r["inf_rows"]}
            s = g.get("slab")
            if s:
                def proj(n):
                    return n / s["posts_per_s"] / 3600 * rate * 1.5
                g["s12_projection"] = {
                    "slab_units": round(proj(S12_SLAB), 2),
                    "slab_arith": f"{S12_SLAB:,.0f} / {s['posts_per_s']:,.0f} posts/s / 3600 x {rate:g} x 1.5 = {proj(S12_SLAB):.2f}",
                    "slab_band_units": [round(proj(x), 2) for x in S12_SLAB_BAND],
                    "bios_units": round(proj(S12_BIOS), 2), "wall_hours_slab": round(S12_SLAB / s["posts_per_s"] / 3600, 2)}
            checks[f"{gpu}_no_nan"] = bench.get("nan_or_inf_rows_total") == 0
            if bench.get("parity"):
                checks[f"{gpu}_parity_median_ge_0_99"] = bool(bench["parity"]["bar_median_ge_0_99_on_random"])
        rec["gpus"][gpu] = g
    rts = sorted(res.glob("s7_rt_*.tar"))
    rec["roundtrip"] = {"result": text(res / "roundtrip_result.txt").strip() or None,
                        "attempts": {p.stem: tar_json(p, "roundtrip.json") for p in rts}}
    ok = [a for a in rec["roundtrip"]["attempts"].values() if a and a.get("ok")]
    checks["roundtrip_read_and_write_ok"] = bool(ok)
    rec["roundtrip"]["oauth_scopes"] = (ok[0] if ok else next((a for a in rec["roundtrip"]["attempts"].values() if a), {})
                                        ).get("credential", {}).get("scopes")
    if not ok:
        rec["roundtrip"]["fallback"] = ("§5.1 fallback (Josh runs the auth cell in the Colab UI) was tried and also failed: "
                                        "S12 needs another data path (e.g. colab upload/download) via a fresh ask"
                                        if (res / "roundtrip_fallback.log").exists() else "round trip not completed")
    costs = {g: v["slab"]["units_per_10m"] for g, v in rec["gpus"].items() if "slab" in v}
    rec["cheaper_gpu_for_s12"] = min(costs, key=costs.get) if costs else None
    sess_after = " ".join(rec["sessions_after"])
    checks["own_sessions_absent_after"] = all(s.split("\t")[1] not in sess_after for s in rec["sessions_tsv"] if s.count("\t") >= 3)
    rec["checks"] = checks
    json.dump(rec, open(run / "receipt.json", "w"), indent=1)
    print(s7lib.dumps({"checks": checks, "cheaper_gpu_for_s12": rec["cheaper_gpu_for_s12"],
                       "per_gpu": {g: {k: v.get(k) for k in ["random", "slab", "s12_projection"]} for g, v in rec["gpus"].items()},
                       "roundtrip": {k: rec["roundtrip"][k] for k in ["result", "oauth_scopes"]}}))


if __name__ == "__main__":
    main(sys.argv[1])
