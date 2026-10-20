#!/usr/bin/env python3
"""Offline teacher inference, Ettin training, and honest artifact verification."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from real_pipeline.common import create_review_sheet, local_guard, read_rows, validate_input


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
        item.add_argument("--export-manifest", type=Path, help="Audited v1 seal; defaults beside the canonical holdout")
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
            item.add_argument("--objective", choices=("health", "joint"), default="joint",
                              help="Health-only is explicit; joint full runs require in-scope graph pairs")
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
    if args.command in ("label", "train"):
        local_guard(args.mode, rows, args.device)
    validate_input(args, rows)
    if args.command == "review-sheet":
        create_review_sheet(args.output, rows)
        return
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
if __name__ == "__main__":
    main()
