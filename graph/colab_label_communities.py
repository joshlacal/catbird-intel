#!/usr/bin/env python3
"""Colab LLM community labeling preparation, 1-shard benchmark gate, checkpointed
runner, and evaluation receipt generator (Phase 1 step 4 / Master Plan §4a).

Hard rules (Handoff §3.4 & Master Plan §4a):
  - Preparing shards and dry-run verification spend 0 Colab units and make 0 API calls.
  - Live GPU/API execution is gated behind explicit `--approve-spend` and a 1-shard
    benchmark receipt confirming projected cost fits the per-job unit cap.
  - Checkpoints a manifest after every shard so preempted Colab sessions resume cleanly.
  - Produces a sealed evaluation receipt recording input hashes, command, and output hashes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time
import urllib.request
from typing import Any


SYSTEM_PROMPT = (
    "You are labeling Bluesky social-graph communities discovered via producer-producer "
    "cosine similarity and Leiden clustering. Given a community's top TF-IDF profile terms, "
    "enriched handle TLDs, and sample top producers (handle, display name, bio), return a "
    "JSON object with keys: "
    '{"label": "<concise 2-5 word community name>", '
    '"description": "<one-sentence summary of who belongs to this community>", '
    '"primary_language": "<ISO 639-1 code or multi>", '
    '"confidence": "<high|medium|low>", '
    '"evidence_terms": ["<up to 5 grounding terms or handles>"]}.'
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_user_prompt(row: dict[str, Any]) -> str:
    lines = [
        f"Community ID: {row['community']} (Scene {row.get('scene_id', 0)}, Domain {row.get('domain_id', 0)})",
        f"Producer count: {row.get('size', 0)}",
        f"Profile TF-IDF terms: {', '.join(row.get('top_terms', [])) or 'none'}",
        f"Enriched handle TLDs: {', '.join(row.get('enriched_tlds', [])) or 'none'}",
        "Sample top producers:",
    ]
    for p in row.get("top_producers", [])[:8]:
        handle = p.get("handle") or p.get("did", "")
        name = p.get("name", "")
        bio = (p.get("bio") or "").replace("\n", " ").strip()
        lines.append(f"  - @{handle} ({name}): {bio}")
    return "\n".join(lines)


def prepare_shards(
    prompts_path: pathlib.Path,
    out_dir: pathlib.Path,
    shard_size: int = 50,
    hide_from_recs_path: pathlib.Path | None = None,
    budget_units_cap: float = 20.0,
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    hide_dids: set[str] = set()
    if hide_from_recs_path and hide_from_recs_path.is_file():
        hide_dids = {
            line.strip()
            for line in hide_from_recs_path.read_text().splitlines()
            if line.strip()
        }

    raw_rows = [
        json.loads(line)
        for line in prompts_path.read_text().splitlines()
        if line.strip()
    ]
    filtered_producers_removed = 0
    prepared_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        orig_prods = r.get("top_producers", [])
        clean_prods = [p for p in orig_prods if p.get("did") not in hide_dids]
        filtered_producers_removed += len(orig_prods) - len(clean_prods)
        r_clean = {**r, "top_producers": clean_prods}
        prepared_rows.append(
            {
                "community": int(r_clean["community"]),
                "scene_id": int(r_clean.get("scene_id", 0)),
                "domain_id": int(r_clean.get("domain_id", 0)),
                "size": int(r_clean.get("size", 0)),
                "heuristic_label": r_clean.get("heuristic_label", ""),
                "top_terms": r_clean.get("top_terms", []),
                "enriched_tlds": r_clean.get("enriched_tlds", []),
                "top_producers": clean_prods,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_user_prompt(r_clean),
            }
        )

    shard_entries = []
    for idx in range(0, len(prepared_rows), max(shard_size, 1)):
        chunk = prepared_rows[idx : idx + max(shard_size, 1)]
        shard_id = f"shard-{idx // max(shard_size, 1):03d}"
        shard_file = shards_dir / f"{shard_id}.jsonl"
        with shard_file.open("w") as fh:
            for item in chunk:
                fh.write(json.dumps(item) + "\n")
        shard_entries.append(
            {
                "shard_id": shard_id,
                "path": f"shards/{shard_id}.jsonl",
                "rows": len(chunk),
                "first_community": chunk[0]["community"] if chunk else None,
                "last_community": chunk[-1]["community"] if chunk else None,
                "sha256": sha256_file(shard_file),
            }
        )

    manifest = {
        "schema": "catbird.colab.community_labeling_manifest.v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_prompts_path": str(prompts_path.resolve()),
        "source_prompts_sha256": sha256_file(prompts_path),
        "hide_from_recs_path": str(hide_from_recs_path.resolve()) if hide_from_recs_path and hide_from_recs_path.is_file() else None,
        "hide_from_recs_sha256": sha256_file(hide_from_recs_path) if hide_from_recs_path and hide_from_recs_path.is_file() else None,
        "hide_from_recs_producers_removed": filtered_producers_removed,
        "model_id": model_id,
        "total_communities": len(prepared_rows),
        "shard_size": shard_size,
        "total_shards": len(shard_entries),
        "gate_contract": {
            "governing_plan_section": "2026-10-19-catbird-community-intelligence-master-plan.md#4a",
            "requires_explicit_josh_approval": True,
            "budget_units_cap": budget_units_cap,
            "steps": [
                "1. Benchmark one shard (shard-000) and record throughput and units per shard.",
                "2. Project full cost across all shards and confirm projected_units <= budget_units_cap.",
                "3. Run remaining shards with per-shard manifest checkpointing.",
                "4. Verify output completeness and save evaluation_receipt.json.",
            ],
        },
        "shards": shard_entries,
    }
    manifest_path = out_dir / "colab_labeling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _call_openai_compatible_endpoint(
    endpoint: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {
        "label": str(parsed.get("label", "")).strip(),
        "description": str(parsed.get("description", "")).strip(),
        "primary_language": str(parsed.get("primary_language", "en")).strip(),
        "confidence": str(parsed.get("confidence", "medium")).strip(),
        "evidence_terms": list(parsed.get("evidence_terms", []))[:5],
    }


def benchmark_one_shard(
    out_dir: pathlib.Path,
    shard_id: str = "shard-000",
    units_per_hour: float = 5.36,
    approve_spend: bool = False,
    endpoint: str | None = None,
    dry_run_local: bool = False,
) -> dict[str, Any]:
    manifest_path = out_dir / "colab_labeling_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard_meta = next((s for s in manifest["shards"] if s["shard_id"] == shard_id), None)
    if shard_meta is None:
        raise SystemExit(f"Shard {shard_id} not found in {manifest_path}")

    shard_path = out_dir / shard_meta["path"]
    if sha256_file(shard_path) != shard_meta["sha256"]:
        raise SystemExit(f"Shard checksum mismatch for {shard_path}")

    if not dry_run_local and not approve_spend:
        raise SystemExit(
            "Refusing to run live GPU/API benchmark without --approve-spend "
            "(Handoff §3.4: Do not spend Colab units or make paid API calls without Josh's explicit per-run OK)."
        )

    rows = [json.loads(line) for line in shard_path.read_text().splitlines() if line.strip()]
    t0 = time.monotonic()
    outputs = []
    for r in rows:
        if dry_run_local:
            # Deterministic offline structural rehearsal (0 compute units, 0 network calls)
            terms = [t for t in r.get("top_terms", [])[:3] if t]
            label = " / ".join(t.title() for t in terms) if terms else r.get("heuristic_label", f"Community #{r['community']}")
            outputs.append(
                {
                    "community": r["community"],
                    "label": label,
                    "description": f"Dry-run rehearsal label from profile terms ({', '.join(terms) or 'none'}).",
                    "primary_language": "en",
                    "confidence": "dry_run_rehearsal",
                    "evidence_terms": terms,
                    "label_source": "dry_run_rehearsal_not_llm",
                }
            )
        else:
            if not endpoint:
                raise SystemExit("--endpoint is required for live benchmark execution")
            res = _call_openai_compatible_endpoint(
                endpoint=endpoint,
                model_id=manifest["model_id"],
                system_prompt=r["system_prompt"],
                user_prompt=r["user_prompt"],
            )
            outputs.append(
                {
                    "community": r["community"],
                    **res,
                    "label_source": f"llm:{manifest['model_id']}",
                }
            )

    elapsed_s = max(time.monotonic() - t0, 1e-6)
    units_per_shard = 0.0 if dry_run_local else (elapsed_s / 3600.0) * units_per_hour
    projected_total_units = units_per_shard * manifest["total_shards"]
    cap = float(manifest["gate_contract"]["budget_units_cap"])

    receipt = {
        "schema": "catbird.colab.community_labeling_benchmark_receipt.v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "dry_run_rehearsal" if dry_run_local else "live_gpu_benchmark",
        "manifest_sha256": sha256_file(manifest_path),
        "shard_id": shard_id,
        "shard_sha256": shard_meta["sha256"],
        "rows_benchmarked": len(outputs),
        "elapsed_seconds": round(elapsed_s, 4),
        "communities_per_second": round(len(outputs) / elapsed_s, 2),
        "assumed_units_per_hour": units_per_hour,
        "observed_units_per_shard": round(units_per_shard, 6),
        "total_shards": manifest["total_shards"],
        "projected_total_units": round(projected_total_units, 4),
        "budget_units_cap": cap,
        "fits_budget": projected_total_units <= cap,
        "qualifies_for_live_production_labels": not dry_run_local,
    }
    (out_dir / "benchmark_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def run_all_shards(
    out_dir: pathlib.Path,
    approve_spend: bool = False,
    endpoint: str | None = None,
    dry_run_local: bool = False,
) -> dict[str, Any]:
    manifest_path = out_dir / "colab_labeling_manifest.json"
    bench_path = out_dir / "benchmark_receipt.json"
    if not bench_path.is_file():
        raise SystemExit("Gate violation: benchmark_receipt.json missing. Run benchmark-shard first.")
    bench = json.loads(bench_path.read_text())
    if not bench.get("fits_budget", False):
        raise SystemExit(
            f"Gate violation: projected_total_units ({bench.get('projected_total_units')}) "
            f"exceeds cap ({bench.get('budget_units_cap')})."
        )
    if not dry_run_local and not approve_spend:
        raise SystemExit("Refusing to run live shards without --approve-spend.")

    manifest = json.loads(manifest_path.read_text())
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    valid_sids = {s["shard_id"] for s in manifest["shards"]}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["manifest_sha256"] = manifest_hash
        checkpoint["completed_shards"] = {
            sid: v for sid, v in checkpoint.get("completed_shards", {}).items() if sid in valid_sids
        }
    else:
        checkpoint = {
            "schema": "catbird.colab.community_labeling_checkpoint.v1",
            "manifest_sha256": manifest_hash,
            "completed_shards": {},
        }

    for shard_meta in manifest["shards"]:
        sid = shard_meta["shard_id"]
        out_file = outputs_dir / f"labels-{sid}.jsonl"
        prev = checkpoint["completed_shards"].get(sid)
        if (
            prev
            and prev.get("input_sha256") == shard_meta["sha256"]
            and out_file.is_file()
            and sha256_file(out_file) == prev.get("output_sha256")
        ):
            continue

        shard_path = out_dir / shard_meta["path"]
        if sha256_file(shard_path) != shard_meta["sha256"]:
            raise SystemExit(f"Input shard checksum mismatch: {shard_path}")

        rows = [json.loads(line) for line in shard_path.read_text().splitlines() if line.strip()]
        out_rows = []
        for r in rows:
            if dry_run_local:
                terms = [t for t in r.get("top_terms", [])[:3] if t]
                label = " / ".join(t.title() for t in terms) if terms else r.get("heuristic_label", f"Community #{r['community']}")
                out_rows.append(
                    {
                        "community": r["community"],
                        "scene_id": r["scene_id"],
                        "domain_id": r["domain_id"],
                        "size": r["size"],
                        "label": label,
                        "description": f"Dry-run rehearsal label from profile terms ({', '.join(terms) or 'none'}).",
                        "primary_language": "en",
                        "confidence": "dry_run_rehearsal",
                        "evidence_terms": terms,
                        "label_source": "dry_run_rehearsal_not_llm",
                    }
                )
            else:
                if not endpoint:
                    raise SystemExit("--endpoint is required for live execution")
                res = _call_openai_compatible_endpoint(
                    endpoint=endpoint,
                    model_id=manifest["model_id"],
                    system_prompt=r["system_prompt"],
                    user_prompt=r["user_prompt"],
                )
                out_rows.append(
                    {
                        "community": r["community"],
                        "scene_id": r["scene_id"],
                        "domain_id": r["domain_id"],
                        "size": r["size"],
                        **res,
                        "label_source": f"llm:{manifest['model_id']}",
                    }
                )

        with out_file.open("w") as fh:
            for item in out_rows:
                fh.write(json.dumps(item) + "\n")

        checkpoint["completed_shards"][sid] = {
            "shard_id": sid,
            "input_sha256": shard_meta["sha256"],
            "output_path": f"outputs/labels-{sid}.jsonl",
            "output_sha256": sha256_file(out_file),
            "rows": len(out_rows),
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    return checkpoint


def verify_and_write_receipt(out_dir: pathlib.Path) -> dict[str, Any]:
    manifest_path = out_dir / "colab_labeling_manifest.json"
    bench_path = out_dir / "benchmark_receipt.json"
    checkpoint_path = out_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bench = json.loads(bench_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())

    all_labels: list[dict[str, Any]] = []
    for shard_meta in manifest["shards"]:
        sid = shard_meta["shard_id"]
        done_meta = checkpoint["completed_shards"].get(sid)
        if not done_meta:
            raise SystemExit(f"Shard {sid} missing from checkpoint_manifest.json")
        out_file = out_dir / done_meta["output_path"]
        if sha256_file(out_file) != done_meta["output_sha256"]:
            raise SystemExit(f"Output shard checksum mismatch for {out_file}")
        rows = [json.loads(line) for line in out_file.read_text().splitlines() if line.strip()]
        if len(rows) != shard_meta["rows"]:
            raise SystemExit(f"Row count mismatch in {out_file}: expected {shard_meta['rows']}, got {len(rows)}")
        all_labels.extend(rows)

    merged_path = out_dir / "communities_llm_labels.jsonl"
    with merged_path.open("w") as fh:
        for r in all_labels:
            fh.write(json.dumps(r) + "\n")

    non_empty_labels = sum(1 for r in all_labels if r.get("label", "").strip())
    is_live_llm = all(str(r.get("label_source", "")).startswith("llm:") for r in all_labels)
    receipt = {
        "schema": "catbird.colab.community_labeling_evaluation_receipt.v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_path": str(pathlib.Path(__file__).resolve()),
        "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "argv": sys.argv,
        "manifest_sha256": sha256_file(manifest_path),
        "benchmark_receipt_sha256": sha256_file(bench_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "merged_output_path": str(merged_path.resolve()),
        "merged_output_sha256": sha256_file(merged_path),
        "communities_expected": manifest["total_communities"],
        "communities_labeled": len(all_labels),
        "non_empty_labels": non_empty_labels,
        "hide_from_recs_producers_removed": manifest.get("hide_from_recs_producers_removed", 0),
        "live_llm_qualified": is_live_llm,
        "status": "qualified_llm_labels" if is_live_llm and non_empty_labels == manifest["total_communities"] else "prepared_dry_run_only",
    }
    (out_dir / "evaluation_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "benchmark-shard", "run", "verify-receipt"])
    ap.add_argument("--prompts", default="/Users/joshlacalamito/catbird-lake-local/communities-cand1-hubdemote/eval-cand1/colab_community_labeling_prompts.jsonl")
    ap.add_argument("--out-dir", default="/Users/joshlacalamito/catbird-lake-local/communities-cand1-hubdemote/eval-cand1/colab-labeling")
    ap.add_argument("--hide-from-recs", default="/Users/joshlacalamito/catbird-lake-local/communities-cand1-hubdemote/hide_from_recs.tsv")
    ap.add_argument("--shard-size", type=int, default=50)
    ap.add_argument("--budget-units-cap", type=float, default=20.0)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--shard-id", default="shard-000")
    ap.add_argument("--units-per-hour", type=float, default=5.36)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--approve-spend", action="store_true")
    ap.add_argument("--dry-run-local", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if args.action == "prepare":
        res = prepare_shards(
            prompts_path=pathlib.Path(args.prompts),
            out_dir=out_dir,
            shard_size=args.shard_size,
            hide_from_recs_path=pathlib.Path(args.hide_from_recs) if args.hide_from_recs else None,
            budget_units_cap=args.budget_units_cap,
            model_id=args.model_id,
        )
    elif args.action == "benchmark-shard":
        res = benchmark_one_shard(
            out_dir=out_dir,
            shard_id=args.shard_id,
            units_per_hour=args.units_per_hour,
            approve_spend=args.approve_spend,
            endpoint=args.endpoint,
            dry_run_local=args.dry_run_local,
        )
    elif args.action == "run":
        res = run_all_shards(
            out_dir=out_dir,
            approve_spend=args.approve_spend,
            endpoint=args.endpoint,
            dry_run_local=args.dry_run_local,
        )
    else:
        res = verify_and_write_receipt(out_dir=out_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
