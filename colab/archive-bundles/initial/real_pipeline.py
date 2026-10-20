#!/usr/bin/env python3
"""Offline teacher inference, Ettin training, and honest artifact verification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from real_pipeline.common import local_guard, read_rows, require_disjoint, validate_posts, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("label", "merge", "train", "review-sheet"):
        item = sub.add_parser(command)
        item.add_argument("--input", required=True, type=Path)
        item.add_argument("--output", required=True, type=Path)
        item.add_argument("--mode", choices=("smoke", "colab"), default="colab")
        item.add_argument("--device", default="cuda")
        item.add_argument("--threads", type=int, default=2)
        item.add_argument("--holdout", type=Path)
        item.add_argument("--split", choices=("train", "holdout"), default="train")
        if command == "label":
            item.add_argument("--teacher", choices=("political", "qwen", "nli"), required=True)
            item.add_argument("--batch-size", type=int, default=4)
            item.add_argument("--max-length", type=int, default=512)
        if command == "merge":
            item.add_argument("--political", type=Path, required=True)
            item.add_argument("--behavior", type=Path, required=True)
            item.add_argument("--behavior-teacher", choices=("qwen", "nli"), required=True)
        if command == "train":
            item.add_argument("--pairs", type=Path)
            item.add_argument("--resume", type=Path)
            item.add_argument("--steps", type=int, default=5000)
            item.add_argument("--save-every", type=int, default=250)
            item.add_argument("--batch-size", type=int, default=32)
            item.add_argument("--learning-rate", type=float, default=2e-5)
            item.add_argument("--social-weight", type=float, default=1.0)
            item.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    rows = read_rows(args.input, limit=32 if args.mode == "smoke" else None)
    if args.mode == "smoke":
        if args.split == "holdout":
            parser.error("Local smoke cannot label or inspect held-out examples")
        rows = rows[:32]
        if args.command == "train" and args.steps > 12:
            parser.error("Local smoke is limited to at most 12 optimizer steps on the same 32 posts")
    local_guard(args.mode, rows, args.device)
    validate_posts(rows, allow_teachers=args.command in ("train", "review-sheet"))
    if args.split == "train":
        if any(row.get("author_split_bucket") == 0 for row in rows):
            parser.error("Reserved holdout author bucket cannot enter training")
        if not args.holdout:
            parser.error("Training-bank operations require --holdout to enforce author isolation")
        require_disjoint(rows, read_rows(args.holdout))
    elif args.command == "train":
        parser.error("Held-out posts can never train the student")
    import torch
    torch.set_num_threads(args.threads)
    if args.command == "label":
        from real_pipeline.teachers import label_stage
        label_stage(args, rows)
    elif args.command == "merge":
        from real_pipeline.teachers import merge_labels
        merge_labels(args, rows)
    elif args.command == "train":
        from real_pipeline.student import train
        train(args, rows)
    elif args.command == "review-sheet":
        # No labels/thresholds/F1 invented. Human fields stay null until reviewed.
        fields = ("politicalness", "outraged", "critical", "engagement_bait", "spam")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            for row in rows:
                row.update({f"human_{name}": None for name in fields})
                row.update(human_reviewed=False, reviewer=None, review_notes="")
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        write_json(str(args.output) + ".status.json", {"rows": len(rows), "human_reviewed": False,
                   "thresholds_locked": False, "F1": None})


if __name__ == "__main__":
    main()
