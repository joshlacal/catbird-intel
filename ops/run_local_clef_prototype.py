#!/usr/bin/env python3
"""Local M4 Max prototype runner for Cloudflare/clef-flash (9B) + Political_DEBATE_large_v1.0.

Executes a 64-post stratified pilot smoke test on Apple Silicon (`mps`) over the
prepared 2,000-post pilot sample (`catbird-intel/eval/anthropic_teacher_pilot_v1`),
scoring every post with:
1. `mlburnham/Political_DEBATE_large_v1.0` (pinned rev `526ef82f0bfd4c39de6bbcecfbf9ef1fb6efbd68`)
   for Teacher-1 politicalness entailment probability (`[0, 1]`).
2. `Cloudflare/clef-flash` (pinned rev `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`, `Qwen3.5-9B` +
   `JointSchemaHead`, Apache-2.0) via `/v1/systemone` (`build_systemone_clef_request` +
   `parse_systemone_clef_response`) in a single non-autoregressive prefill pass (`output_tokens = 0`).

Writes a machine-verified JSON receipt and a human-readable Markdown inspection report.
Never makes paid remote API calls or touches sealed `catbird-intel/colab/` files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
from pathlib import Path
import platform
import sqlite3
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_harness import (  # noqa: E402
    COMPONENT_ORDER,
    HarnessConfig,
    assert_no_credentials,
    build_systemone_clef_request,
    canonical_sha256,
    parse_systemone_clef_response,
    scrub_environment_snapshot,
)

CLEF_FLASH_REPO = "Cloudflare/clef-flash"
CLEF_FLASH_REVISION = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"

POLITICAL_DEBATE_REPO = "mlburnham/Political_DEBATE_large_v1.0"
POLITICAL_DEBATE_REVISION = "1a3aff1ecb97ad93a6de598f7414b1707de7e7c3"
POLITICAL_HYPOTHESIS = "This text is about politics, government, elections, or public policy."

STRATIFIED_64_QUOTAS: dict[str, int] = {
    "natural_probability": 24,
    "post_type_community_balanced": 16,
    "editorial_boundary": 10,
    "moderation_enrichment": 10,
    "robustness_drift": 4,
}


def _release_torch_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mps_allocated_mb() -> float:
    if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
        return round(torch.mps.current_allocated_memory() / (1024.0 * 1024.0), 2)
    return 0.0


def _mps_driver_mb() -> float:
    if torch.backends.mps.is_available() and hasattr(torch.mps, "driver_allocated_memory"):
        return round(torch.mps.driver_allocated_memory() / (1024.0 * 1024.0), 2)
    return 0.0


DEFAULT_HYDRATION_PATH = REPO_ROOT / "eval" / "pilot_holdout_parent_quote_hydration_20261021.jsonl"
DEFAULT_BANK_PATH = Path("/Users/joshlacalamito/catbird-lake-local/staging/catbird_v3_export_v1/train_150k_v1.jsonl")


def select_stratified_pilot_slice(
    db_path: Path,
    *,
    quotas: dict[str, int] = STRATIFIED_64_QUOTAS,
    hydration_path: Path | None = DEFAULT_HYDRATION_PATH,
    bank_path: Path | None = DEFAULT_BANK_PATH,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select a deterministic 64-post stratified slice from the 2,000 prepared pilot posts."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT uri, cid, author_did, post_type, sampling_stratum, sampling_method,
                   inclusion_probability, selection_priority_key, has_local_parent_or_quote_text,
                   moderation_join_json, raw_row_json
            FROM posts
            WHERE split = 'train' AND sampling_stratum IS NOT NULL
            ORDER BY sampling_stratum ASC, selection_priority_key ASC, uri ASC
            """
        ).fetchall()

        all_corpus_rows = conn.execute(
            "SELECT uri, raw_row_json FROM posts"
        ).fetchall()
    finally:
        conn.close()

    corpus_by_uri: dict[str, dict[str, Any]] = {}
    if hydration_path is not None and hydration_path.exists():
        with hydration_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                h_row = json.loads(line)
                h_uri = str(h_row.get("uri") or "").strip()
                if h_uri:
                    corpus_by_uri[h_uri] = h_row

    for r in all_corpus_rows:
        corpus_by_uri[str(r["uri"])] = json.loads(r["raw_row_json"])

    needed_bank_uris: set[str] = set()
    for r in rows:
        raw_peek = json.loads(r["raw_row_json"])
        for ref_key in ("reply_parent_uri", "quote_uri", "repost_subject_uri"):
            ref_u = str(raw_peek.get(ref_key) or "").strip()
            if ref_u and ref_u not in corpus_by_uri:
                needed_bank_uris.add(ref_u)

    if needed_bank_uris and bank_path is not None and bank_path.exists():
        with bank_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                b_row = json.loads(line)
                b_uri = str(b_row.get("uri") or "").strip()
                if b_uri in needed_bank_uris:
                    corpus_by_uri[b_uri] = b_row
                    needed_bank_uris.discard(b_uri)
                    if not needed_bank_uris:
                        break

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        raw = json.loads(r["raw_row_json"])
        raw["sampling_stratum"] = r["sampling_stratum"]
        raw["sampling_method"] = r["sampling_method"]
        raw["inclusion_probability"] = r["inclusion_probability"]
        raw["selection_priority_key"] = r["selection_priority_key"]
        parent_uri = str(raw.get("reply_parent_uri") or "").strip()
        quote_uri = str(raw.get("quote_uri") or "").strip()
        has_parent_txt = bool(
            parent_uri
            and parent_uri in corpus_by_uri
            and isinstance(corpus_by_uri[parent_uri].get("text"), str)
            and corpus_by_uri[parent_uri]["text"].strip()
        )
        has_quote_txt = bool(
            quote_uri
            and quote_uri in corpus_by_uri
            and isinstance(corpus_by_uri[quote_uri].get("text"), str)
            and corpus_by_uri[quote_uri]["text"].strip()
        )
        raw["has_local_parent_or_quote_text"] = bool(
            r["has_local_parent_or_quote_text"] or has_parent_txt or has_quote_txt
        )
        raw["moderation_join"] = json.loads(r["moderation_join_json"])
        by_stratum[str(r["sampling_stratum"])].append(raw)

    selected: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        pool = by_stratum.get(stratum, [])
        if len(pool) < quota:
            raise RuntimeError(
                f"Stratum {stratum!r} has only {len(pool)} prepared rows, need {quota}"
            )
        selected.extend(pool[:quota])

    return selected, corpus_by_uri


def run_political_debate_teacher(
    rows: list[dict[str, Any]],
    *,
    device: str = "mps",
    batch_size: int = 8,
    max_length: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run cached `mlburnham/Political_DEBATE_large_v1.0` on the selected rows."""
    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        POLITICAL_DEBATE_REPO,
        revision=POLITICAL_DEBATE_REVISION,
        local_files_only=True,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            POLITICAL_DEBATE_REPO,
            revision=POLITICAL_DEBATE_REVISION,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    load_sec = time.perf_counter() - t_load0

    labels = {label.lower(): int(idx) for idx, label in model.config.id2label.items()}
    if "entailment" not in labels or model.config.num_labels != 2:
        raise ValueError(f"Expected binary entailment labels for Political DEBATE, got {labels}")
    entailment_id = labels["entailment"]

    results: list[dict[str, Any]] = []
    t_inf0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [str(r["text"]) for r in batch],
                [POLITICAL_HYPOTHESIS] * len(batch),
                padding=True,
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits.float().cpu()
            probs = logits.softmax(-1)[:, entailment_id]
            for prob, raw_logit in zip(probs, logits):
                results.append(
                    {
                        "political_debate_prob": round(float(prob), 6),
                        "political_debate_logits": [round(float(x), 6) for x in raw_logit.tolist()],
                    }
                )
    if device == "mps":
        torch.mps.synchronize()
    inf_sec = time.perf_counter() - t_inf0
    peak_alloc_mb = _mps_allocated_mb()
    peak_driver_mb = _mps_driver_mb()

    del model
    del tokenizer
    _release_torch_memory()

    perf = {
        "model_id": POLITICAL_DEBATE_REPO,
        "revision": POLITICAL_DEBATE_REVISION,
        "device": device,
        "batch_size": batch_size,
        "load_seconds": round(load_sec, 4),
        "inference_seconds": round(inf_sec, 4),
        "ms_per_post": round((inf_sec * 1000.0) / max(1, len(rows)), 2),
        "posts_per_second": round(len(rows) / max(1e-9, inf_sec), 3),
        "mps_allocated_mb": peak_alloc_mb,
        "mps_driver_mb": peak_driver_mb,
    }
    return results, perf


def _load_clef_joint_schema_module(snapshot_dir: Path) -> Any:
    module_path = snapshot_dir / "joint_schema_model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"joint_schema_model.py not found in {snapshot_dir}")
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clef_joint_schema_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_clef_flash_teacher(
    rows: list[dict[str, Any]],
    *,
    config: HarnessConfig,
    corpus_by_uri: dict[str, dict[str, Any]],
    device: str = "mps",
    dtype_str: str = "bfloat16",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load `Cloudflare/clef-flash` (9B) on M4 Max and score each post via `/v1/systemone`."""
    from huggingface_hub import snapshot_download

    t_dl0 = time.perf_counter()
    snapshot_dir = Path(
        snapshot_download(
            repo_id=CLEF_FLASH_REPO,
            revision=CLEF_FLASH_REVISION,
        )
    )
    dl_sec = time.perf_counter() - t_dl0

    clef_mod = _load_clef_joint_schema_module(snapshot_dir)
    dtype = getattr(torch, dtype_str)

    t_load0 = time.perf_counter()
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot_dir,
        dtype=dtype,
        device_map={"": str(device)},
        local_files_only=True,
    )
    backbone.config.use_cache = False
    head_config = json.loads((snapshot_dir / "joint_head_config.json").read_text())
    head = clef_mod.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(snapshot_dir / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=dtype)
    try:
        processor = AutoProcessor.from_pretrained(snapshot_dir, local_files_only=True)
    except (ValueError, ImportError):
        tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

        class _TextOnlyProcessor:
            def __init__(self, tokenizer: Any) -> None:
                self.tokenizer = tokenizer

        processor = _TextOnlyProcessor(tok)

    model = clef_mod.ClefModel(backbone, head).eval()
    if device == "mps":
        torch.mps.synchronize()
    load_sec = time.perf_counter() - t_load0

    param_count = sum(p.numel() for p in model.parameters())
    head_param_count = sum(p.numel() for p in model.head.parameters())
    load_alloc_mb = _mps_allocated_mb()
    load_driver_mb = _mps_driver_mb()

    # Warmup pass on first request so MPS shader compilation is measured separately from steady-state latency
    first_req = build_systemone_clef_request(
        rows[0],
        config=config,
        model_name=CLEF_FLASH_REPO,
        observation_mode="text_only",
        corpus_by_uri=corpus_by_uri,
    )
    t_warm0 = time.perf_counter()
    _ = clef_mod.systemone(model, processor, first_req)
    if device == "mps":
        torch.mps.synchronize()
    warmup_sec = time.perf_counter() - t_warm0

    outputs: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    input_tokens_list: list[int] = []

    t_inf0 = time.perf_counter()
    for idx, row in enumerate(rows):
        obs_mode = "context_enriched" if row.get("has_local_parent_or_quote_text") else "text_only"
        req = build_systemone_clef_request(
            row,
            config=config,
            model_name=CLEF_FLASH_REPO,
            observation_mode=obs_mode,
            corpus_by_uri=corpus_by_uri,
        )
        t_step0 = time.perf_counter()
        raw_resp = clef_mod.systemone(model, processor, req)
        if device == "mps":
            torch.mps.synchronize()
        step_ms = (time.perf_counter() - t_step0) * 1000.0
        latencies_ms.append(step_ms)

        parsed = parse_systemone_clef_response(raw_resp)
        parsed["observation_mode"] = obs_mode
        parsed["latency_ms"] = round(step_ms, 2)
        input_tokens_list.append(int(parsed["usage"].get("input_tokens", 0)))
        outputs.append(parsed)

        if (idx + 1) % 8 == 0 or (idx + 1) == len(rows):
            print(
                f"[clef-flash] Scored {idx + 1}/{len(rows)} posts "
                f"(last={step_ms:.1f} ms, mean={sum(latencies_ms)/len(latencies_ms):.1f} ms, "
                f"mps_alloc={_mps_allocated_mb():.0f} MB)",
                flush=True,
            )

    total_inf_sec = time.perf_counter() - t_inf0
    peak_alloc_mb = _mps_allocated_mb()
    peak_driver_mb = _mps_driver_mb()

    del model
    del processor
    _release_torch_memory()

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    p50_ms = sorted_lat[n // 2] if n else 0.0
    p90_ms = sorted_lat[min(n - 1, int(n * 0.90))] if n else 0.0
    p99_ms = sorted_lat[min(n - 1, int(n * 0.99))] if n else 0.0

    perf = {
        "model_id": CLEF_FLASH_REPO,
        "revision": CLEF_FLASH_REVISION,
        "license": "Apache-2.0",
        "device": device,
        "dtype": dtype_str,
        "total_parameters": param_count,
        "joint_schema_head_parameters": head_param_count,
        "snapshot_check_seconds": round(dl_sec, 4),
        "model_load_seconds": round(load_sec, 4),
        "mps_shader_warmup_seconds": round(warmup_sec, 4),
        "steady_state_inference_seconds": round(total_inf_sec, 4),
        "mean_ms_per_post": round(sum(latencies_ms) / max(1, n), 2),
        "p50_ms_per_post": round(p50_ms, 2),
        "p90_ms_per_post": round(p90_ms, 2),
        "p99_ms_per_post": round(p99_ms, 2),
        "posts_per_second": round(n / max(1e-9, total_inf_sec), 3),
        "projected_minutes_for_2000_pilot_posts": round((2000.0 * (total_inf_sec / max(1, n))) / 60.0, 2),
        "mean_input_tokens": round(sum(input_tokens_list) / max(1, len(input_tokens_list)), 1),
        "total_output_tokens": 0,
        "mps_load_allocated_mb": load_alloc_mb,
        "mps_load_driver_mb": load_driver_mb,
        "mps_peak_allocated_mb": peak_alloc_mb,
        "mps_peak_driver_mb": peak_driver_mb,
    }
    return outputs, perf


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx < 1e-12 or deny < 1e-12:
        return 0.0
    return round(num / (denx * deny), 4)


def _spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    return _pearson_corr(_ranks(xs), _ranks(ys))


def compute_prototype_summary(
    rows: list[dict[str, Any]],
    pol_debate_out: list[dict[str, Any]],
    clef_out: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute ordinal distributions, calibration statistics, and cross-teacher agreement."""
    combined_records: list[dict[str, Any]] = []
    comp_ordinal_counts: dict[str, Counter[str]] = {c: Counter() for c in COMPONENT_ORDER}
    comp_confidences: dict[str, list[float]] = {c: [] for c in COMPONENT_ORDER}
    comp_entropies: dict[str, list[float]] = {c: [] for c in COMPONENT_ORDER}
    comp_continuous: dict[str, list[float]] = {c: [] for c in COMPONENT_ORDER}
    comp_noul_probs: dict[str, list[float]] = {c: [] for c in COMPONENT_ORDER}

    topic_counts: Counter[str] = Counter()
    stance_counts: Counter[str] = Counter()
    missing_context_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()

    pol_clef_vals: list[float] = []
    pol_debate_vals: list[float] = []
    pol_regex_vals: list[float] = []
    pol_binary_agree_050 = 0
    pol_binary_total = 0

    for row, pd_res, cf_res in zip(rows, pol_debate_out, clef_out):
        stratum = str(row.get("sampling_stratum") or "unknown")
        stratum_counts[stratum] += 1
        topic_counts[cf_res["topic"]] += 1
        stance_counts[cf_res["stance_ambiguity"]] += 1
        missing_context_counts[cf_res["missing_context_kind"]] += 1

        for comp in COMPONENT_ORDER:
            c_info = cf_res["components"][comp]
            ord_str = str(c_info["ordinal_score"])
            comp_ordinal_counts[comp][ord_str] += 1
            comp_confidences[comp].append(float(c_info["confidence"]))
            comp_entropies[comp].append(float(c_info["normalized_entropy"]))
            comp_continuous[comp].append(float(c_info["continuous_0_1"]))
            comp_noul_probs[comp].append(float(c_info["insufficient_noul_prob"]))

        cf_pol_info = cf_res["components"]["politicalness"]
        pd_prob = float(pd_res["political_debate_prob"])
        reg_pol = float(row.get("regex_baseline_politicalness") or 0.0)

        if not cf_pol_info["insufficient_evidence"]:
            cf_pol = float(cf_pol_info["continuous_0_1"])
            pol_clef_vals.append(cf_pol)
            pol_debate_vals.append(pd_prob)
            pol_regex_vals.append(reg_pol)
            pol_binary_total += 1
            if (cf_pol >= 0.50) == (pd_prob >= 0.50):
                pol_binary_agree_050 += 1

        combined_records.append(
            {
                "uri": row["uri"],
                "cid": row.get("cid"),
                "post_type": row.get("post_type"),
                "sampling_stratum": stratum,
                "observation_mode": cf_res["observation_mode"],
                "text": row["text"],
                "regex_baseline": {
                    "politicalness": round(float(row.get("regex_baseline_politicalness") or 0.0), 4),
                    "heat": round(float(row.get("regex_baseline_heat") or 0.0), 4),
                    "engagement_bait": round(float(row.get("regex_baseline_engagement_bait") or 0.0), 4),
                    "spam": round(float(row.get("regex_baseline_spam") or 0.0), 4),
                },
                "political_debate_teacher": pd_res,
                "clef_flash": cf_res,
            }
        )

    component_summaries: dict[str, Any] = {}
    for comp in COMPONENT_ORDER:
        counts = comp_ordinal_counts[comp]
        confs = comp_confidences[comp]
        ents = comp_entropies[comp]
        conts = comp_continuous[comp]
        nouls = comp_noul_probs[comp]
        component_summaries[comp] = {
            "ordinal_distribution": {
                "0": counts.get("0", 0),
                "1": counts.get("1", 0),
                "2": counts.get("2", 0),
                "3": counts.get("3", 0),
                "insufficient": counts.get("insufficient", 0),
            },
            "nonzero_or_insufficient_count": sum(
                v for k, v in counts.items() if k != "0"
            ),
            "mean_continuous_0_1": round(sum(conts) / max(1, len(conts)), 4),
            "max_continuous_0_1": round(max(conts) if conts else 0.0, 4),
            "mean_confidence": round(sum(confs) / max(1, len(confs)), 4),
            "min_confidence": round(min(confs) if confs else 0.0, 4),
            "mean_normalized_entropy": round(sum(ents) / max(1, len(ents)), 4),
            "mean_insufficient_noul_prob": round(sum(nouls) / max(1, len(nouls)), 4),
            "max_insufficient_noul_prob": round(max(nouls) if nouls else 0.0, 4),
        }

    # Check separation between critical and outraged (calm criticism vs angry outrage)
    critical_gt_outraged = 0
    outraged_gt_critical = 0
    for rec in combined_records:
        c_comp = rec["clef_flash"]["components"]["critical"]
        o_comp = rec["clef_flash"]["components"]["outraged"]
        if c_comp["continuous_0_1"] - o_comp["continuous_0_1"] >= 0.15:
            critical_gt_outraged += 1
        elif o_comp["continuous_0_1"] - c_comp["continuous_0_1"] >= 0.15:
            outraged_gt_critical += 1

    agreement = {
        "scorable_politicalness_pairs": pol_binary_total,
        "clef_vs_political_debate_binary_agreement_at_0_50": round(
            pol_binary_agree_050 / max(1, pol_binary_total), 4
        ),
        "clef_vs_political_debate_pearson_r": _pearson_corr(pol_clef_vals, pol_debate_vals),
        "clef_vs_political_debate_spearman_rho": _spearman_corr(pol_clef_vals, pol_debate_vals),
        "clef_vs_regex_baseline_pearson_r": _pearson_corr(pol_clef_vals, pol_regex_vals),
        "political_debate_vs_regex_baseline_pearson_r": _pearson_corr(pol_debate_vals, pol_regex_vals),
        "critical_exceeds_outraged_by_0_15_count": critical_gt_outraged,
        "outraged_exceeds_critical_by_0_15_count": outraged_gt_critical,
    }

    scorable_post_count = sum(
        1
        for rec in combined_records
        if all(
            not rec["clef_flash"]["components"][c]["insufficient_evidence"]
            for c in COMPONENT_ORDER
        )
    )
    context_enriched_post_count = sum(
        1 for rec in combined_records if rec["observation_mode"] == "context_enriched"
    )

    summary = {
        "total_posts_scored": len(combined_records),
        "scorable_post_count": scorable_post_count,
        "scorable_post_rate": round(scorable_post_count / max(1, len(combined_records)), 4),
        "context_enriched_post_count": context_enriched_post_count,
        "stratum_counts": dict(stratum_counts),
        "component_summaries": component_summaries,
        "topic_distribution": dict(topic_counts.most_common()),
        "stance_ambiguity_distribution": dict(stance_counts.most_common()),
        "missing_context_kind_distribution": dict(missing_context_counts.most_common()),
        "cross_teacher_agreement": agreement,
    }
    return summary, combined_records


def render_markdown_report(receipt: dict[str, Any]) -> str:
    """Render a concise, cold-reader-friendly Markdown report with side-by-side post inspections."""
    perf_clef = receipt["clef_flash_performance"]
    perf_pd = receipt["political_debate_performance"]
    summary = receipt["summary"]
    agr = summary["cross_teacher_agreement"]
    records = receipt["records"]

    lines: list[str] = [
        "# Local M4 Max Prototype Report — `Cloudflare/clef-flash` (9B) + `Political_DEBATE_large_v1.0`",
        "",
        f"**Date:** `{receipt['created_at']}`  ",
        f"**Host:** `{receipt['environment']['platform']}` (`mps`)  ",
        f"**Prepared Run Directory:** `catbird-intel/eval/anthropic_teacher_pilot_v1` (`2,000` frozen pilot posts)  ",
        f"**Hydration Context:** `catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl` (`{summary['context_enriched_post_count']}/{summary['total_posts_scored']}` context-enriched posts in slice)  ",
        f"**Scorable Post Retention:** **`{summary['scorable_post_count']}/{summary['total_posts_scored']}` (`{summary['scorable_post_rate']*100:.1f}%`)** (up from `5/64 = 7.8%` prior to parent/quote hydration and `SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA` option-key disambiguation)  ",
        f"**Prototype Slice:** `64` deterministic stratified posts (`0` paid API calls, `$0.00` spend)",
        "",
        "## 1. Executive Summary",
        "",
        "We executed the 64-post stratified local M4 Max prototype comparing Cloudflare's newly released open-weight **System One** bounded-decision model (`Cloudflare/clef-flash`, `Qwen3.5-9B` + `JointSchemaHead`, Apache-2.0) against cached Teacher-1 (`mlburnham/Political_DEBATE_large_v1.0`) and the legacy `regex_baseline_*` columns across all five v3 health components (`politicalness`, `outraged`, `critical`, `engagement_bait`, `spam`).",
        "",
        "### Throughput & Memory on M4 Max (`mps`, `bfloat16`)",
        "",
        "| Model | Role | Parameters | Steady-State Latency | Throughput | Peak MPS Allocated | Projected 2,000-Post Pilot Time |",
        "|---|---|---:|---:|---:|---:|---:|",
        (
            f"| `Cloudflare/clef-flash` (`{perf_clef['revision'][:10]}`) | All 5 v3 components + `insufficient` + topic/context/stance (`14` simultaneous questions, `0` output tokens) | "
            f"`{perf_clef['total_parameters']:,}` | `{perf_clef['mean_ms_per_post']:.1f} ms/post` (p50 `{perf_clef['p50_ms_per_post']:.1f} ms`, p90 `{perf_clef['p90_ms_per_post']:.1f} ms`) | "
            f"`{perf_clef['posts_per_second']:.2f} posts/s` | `{perf_clef['mps_peak_allocated_mb']:.0f} MB` (driver `{perf_clef['mps_peak_driver_mb']:.0f} MB`) | "
            f"`{perf_clef['projected_minutes_for_2000_pilot_posts']:.1f} min` (`$0`) |"
        ),
        (
            f"| `mlburnham/Political_DEBATE_large_v1.0` (`{perf_pd['revision'][:10]}`) | Teacher-1 binary `politicalness` entailment | "
            f"`~435M` | `{perf_pd['ms_per_post']:.1f} ms/post` | "
            f"`{perf_pd['posts_per_second']:.2f} posts/s` | `{perf_pd['mps_allocated_mb']:.0f} MB` | "
            f"`{round((2000.0 / max(1e-6, perf_pd['posts_per_second'])) / 60.0, 2):.1f} min` (`$0`) |"
        ),
        "",
        "## 2. Ordinal Score & Brier-Calibrated Confidence Distributions (`64` Stratified Posts)",
        "",
        "| Component | Score `0` | Score `1` | Score `2` | Score `3` | `insufficient` | Mean Continuous `[0,1]` | Mean Confidence | Mean Norm Entropy | Mean `noul` (`insufficient`) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for comp in COMPONENT_ORDER:
        cs = summary["component_summaries"][comp]
        od = cs["ordinal_distribution"]
        lines.append(
            f"| `{comp}` | {od['0']} | {od['1']} | {od['2']} | {od['3']} | {od['insufficient']} | "
            f"`{cs['mean_continuous_0_1']:.4f}` | `{cs['mean_confidence']:.4f}` | "
            f"`{cs['mean_normalized_entropy']:.4f}` | `{cs['mean_insufficient_noul_prob']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Cross-Teacher Agreement & Construct Separation",
            "",
            f"- **`Clef-Flash` vs. `Political_DEBATE_large_v1.0` (`politicalness`):**",
            f"  - Binary agreement at `>= 0.50`: **`{agr['clef_vs_political_debate_binary_agreement_at_0_50']*100:.1f}%`** (`{agr['scorable_politicalness_pairs']}` scorable posts)",
            f"  - Pearson correlation ($r$): **`{agr['clef_vs_political_debate_pearson_r']}`**",
            f"  - Spearman rank correlation ($\\rho$): **`{agr['clef_vs_political_debate_spearman_rho']}`**",
            f"- **Comparison against legacy `regex_baseline_politicalness`:**",
            f"  - `Clef-Flash` vs. `regex_baseline`: Pearson $r =$ `{agr['clef_vs_regex_baseline_pearson_r']}`",
            f"  - `Political_DEBATE` vs. `regex_baseline`: Pearson $r =$ `{agr['political_debate_vs_regex_baseline_pearson_r']}`",
            f"- **Decomposed `critical` vs. `outraged` separation:**",
            f"  - Posts where `critical` exceeds `outraged` by $\\ge 0.15$ (calm/substantive critique without outrage): **`{agr['critical_exceeds_outraged_by_0_15_count']}`**",
            f"  - Posts where `outraged` exceeds `critical` by $\\ge 0.15$: **`{agr['outraged_exceeds_critical_by_0_15_count']}`**",
            "",
            "### Topic & Context Routing Distributions",
            "",
            f"- **Topics (`topic`):** `{json.dumps(summary['topic_distribution'])}`",
            f"- **Stance ambiguity (`stance_ambiguity`):** `{json.dumps(summary['stance_ambiguity_distribution'])}`",
            f"- **Missing context kind (`missing_context_kind`):** `{json.dumps(summary['missing_context_kind_distribution'])}`",
            "",
            "## 4. Side-by-Side Post Inspection Examples",
            "",
        ]
    )

    # Select interesting examples to showcase:
    # 1. High political agreement (both Clef & Political DEBATE high)
    # 2. Calm criticism / non-political critique or critical > outraged
    # 3. High outrage / heated post
    # 4. Engagement bait or spam / promo signal
    # 5. Disagreement between Clef-Flash and Political_DEBATE (routing candidate)
    showcase: list[tuple[str, dict[str, Any]]] = []
    seen_uris: set[str] = set()

    def _pick(label: str, key_fn: Any) -> None:
        candidates = [r for r in records if r["uri"] not in seen_uris]
        if not candidates:
            return
        best = max(candidates, key=key_fn)
        seen_uris.add(best["uri"])
        showcase.append((label, best))

    _pick(
        "High Political Agreement (Clef-Flash + Political_DEBATE)",
        lambda r: min(
            r["clef_flash"]["components"]["politicalness"]["continuous_0_1"],
            r["political_debate_teacher"]["political_debate_prob"],
        ),
    )
    _pick(
        "Teacher Disagreement Candidate (|Clef-Flash - Political_DEBATE| for Claude Escalation)",
        lambda r: abs(
            r["clef_flash"]["components"]["politicalness"]["continuous_0_1"]
            - r["political_debate_teacher"]["political_debate_prob"]
        ),
    )
    _pick(
        "Construct Separation (`critical` > `outraged`)",
        lambda r: (
            r["clef_flash"]["components"]["critical"]["continuous_0_1"]
            - r["clef_flash"]["components"]["outraged"]["continuous_0_1"]
        ),
    )
    _pick(
        "High Outrage (`outraged` Signal)",
        lambda r: r["clef_flash"]["components"]["outraged"]["continuous_0_1"],
    )
    _pick(
        "Engagement Bait Signal (`engagement_bait`)",
        lambda r: r["clef_flash"]["components"]["engagement_bait"]["continuous_0_1"],
    )
    _pick(
        "Spam / Promotional Signal (`spam`)",
        lambda r: r["clef_flash"]["components"]["spam"]["continuous_0_1"],
    )
    _pick(
        "Context-Dependent / Missing Context (`context_dependent` / `noul`)",
        lambda r: r["clef_flash"]["context_dependent_noul_prob"],
    )

    for idx, (title, rec) in enumerate(showcase, 1):
        cf = rec["clef_flash"]
        comps = cf["components"]
        pd_prob = rec["political_debate_teacher"]["political_debate_prob"]
        reg = rec["regex_baseline"]
        text_preview = rec["text"].replace("\n", " ").strip()
        if len(text_preview) > 280:
            text_preview = text_preview[:277] + "..."
        lines.extend(
            [
                f"### Example {idx}: {title}",
                f"- **URI:** `{rec['uri']}` | **Stratum:** `{rec['sampling_stratum']}` | **Type:** `{rec['post_type']}` (`{cf['observation_mode']}`)",
                f"- **Text:** > {text_preview}",
                f"- **Politicalness Comparison:** `Clef-Flash` = **`{comps['politicalness']['continuous_0_1']:.3f}`** (ord `{comps['politicalness']['ordinal_score']}`, conf `{comps['politicalness']['confidence']:.2f}`) | `Political_DEBATE` = **`{pd_prob:.3f}`** | `regex_baseline` = `{reg['politicalness']:.3f}`",
                f"- **Clef-Flash Decomposed Health:** `outraged` = **`{comps['outraged']['continuous_0_1']:.3f}`** (ord `{comps['outraged']['ordinal_score']}`) | `critical` = **`{comps['critical']['continuous_0_1']:.3f}`** (ord `{comps['critical']['ordinal_score']}`) | `engagement_bait` = **`{comps['engagement_bait']['continuous_0_1']:.3f}`** (ord `{comps['engagement_bait']['ordinal_score']}`) | `spam` = **`{comps['spam']['continuous_0_1']:.3f}`** (ord `{comps['spam']['ordinal_score']}`)",
                f"- **Metadata:** `topic={cf['topic']}` (`{cf['topic_confidence']:.2f}`), `stance={cf['stance_ambiguity']}`, `context_dep_noul={cf['context_dependent_noul_prob']:.2f}` (`missing={cf['missing_context_kind']}`)",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. Next-Step Recommendation for the 2,000-Post Pilot & $200 Budget",
            "",
            "1. **Run `Cloudflare/clef-flash` + `Political_DEBATE_large_v1.0` locally over all 2,000 prepared pilot posts (`$0` API cost):**",
            f"   - At `{perf_clef['mean_ms_per_post']:.1f} ms/post` (`{perf_clef['posts_per_second']:.2f} posts/s`), scoring all 2,000 pilot posts on M4 Max takes **`~{perf_clef['projected_minutes_for_2000_pilot_posts']:.1f} minutes`** locally with zero token spend.",
            "2. **Use Brier-calibrated entropy & teacher disagreement to route Claude 4.5 / 5.5 Batch escalation:**",
            "   - Posts where `Clef-Flash` and `Political_DEBATE` agree with high confidence (`confidence >= 0.80`, `|delta| < 0.15`) are already well-characterized.",
            "   - Route the high-entropy (`normalized_entropy >= 0.55`), context-ambiguous (`noul >= 0.35`), and cross-teacher disagreement posts (`|Clef - Political_DEBATE| >= 0.25`), plus the mandatory `15%` uniform random audit floor, to the paid Anthropic Batch tiers (`Haiku` / `Sonnet` / `Opus`).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 64-post M4 Max local Clef-Flash + Political DEBATE prototype.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "anthropic_teacher_pilot_v1",
        help="Prepared teacher harness run directory containing state.sqlite3",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Torch device (default: mps)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "eval" / "clef_flash_local_prototype_20261021.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "eval" / "clef_flash_local_prototype_20261021.md",
    )
    args = parser.parse_args()

    db_path = args.run_dir / "state.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"Prepared state.sqlite3 not found at {db_path}")

    config = HarnessConfig.load()
    print(f"[prototype] Selecting 64 stratified pilot posts from {db_path}...", flush=True)
    rows, corpus_by_uri = select_stratified_pilot_slice(db_path)
    print(f"[prototype] Selected {len(rows)} posts across strata: {STRATIFIED_64_QUOTAS}", flush=True)

    print(f"[prototype] Running Teacher-1 ({POLITICAL_DEBATE_REPO}) on {args.device}...", flush=True)
    pol_debate_out, perf_pd = run_political_debate_teacher(rows, device=args.device)
    print(
        f"[prototype] Political DEBATE complete: {perf_pd['ms_per_post']} ms/post "
        f"({perf_pd['posts_per_second']} posts/s)",
        flush=True,
    )

    print(f"[prototype] Running Cloudflare/clef-flash ({CLEF_FLASH_REPO}) on {args.device}...", flush=True)
    clef_out, perf_clef = run_clef_flash_teacher(
        rows,
        config=config,
        corpus_by_uri=corpus_by_uri,
        device=args.device,
        dtype_str="bfloat16",
    )

    summary, combined_records = compute_prototype_summary(rows, pol_debate_out, clef_out)

    receipt: dict[str, Any] = {
        "schema": "catbird-clef-flash-local-prototype-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "paid_api_calls_made": 0,
        "corpus_text_transmitted": False,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "scrubbed_env": scrub_environment_snapshot(),
        },
        "slice_selection": {
            "total_selected": len(rows),
            "quotas": STRATIFIED_64_QUOTAS,
            "selected_uris_sha256": canonical_sha256([r["uri"] for r in rows]),
        },
        "political_debate_performance": perf_pd,
        "clef_flash_performance": perf_clef,
        "summary": summary,
        "records": combined_records,
    }
    assert_no_credentials(receipt, "clef_flash_prototype_receipt")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_report = render_markdown_report(receipt)
    args.output_md.write_text(md_report, encoding="utf-8")
    print(f"[prototype] Wrote JSON receipt to {args.output_json}", flush=True)
    print(f"[prototype] Wrote Markdown report to {args.output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
