#!/usr/bin/env python3
"""Upload explicit repair artifacts and verify bytes/checksum/Drive metadata.

Does not discover or upload old banks, live graph directories, or model outputs.
Existing same-name files with different bytes fail instead of silently replacing
or creating an ambiguous duplicate. Choose a new versioned name in that case.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib

from quarantine_invalidated_drive_v1 import DriveQuarantine, MODEL_NAMES, BANK_NAMES, NOTEBOOK_NAME
from sync_lakehouse_and_training_banks import LAKE_FOLDERS, validate_health_bank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destination", choices=("root", "training-banks", "eval", "colab-v1"),
                    required=True)
    ap.add_argument("--file", type=pathlib.Path, action="append", required=True)
    ap.add_argument("--subdirectory", default="",
                    help="Optional relative package directory inside colab-v1")
    ap.add_argument("--receipt", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1]
                    / "eval/drive_uploads_v1.jsonl")
    args = ap.parse_args()
    drive = DriveQuarantine()
    parent = (LAKE_FOLDERS[args.destination] if args.destination in LAKE_FOLDERS else
              drive.up.ensure_subfolder(LAKE_FOLDERS["root"], args.destination))
    if args.subdirectory:
        parts = pathlib.PurePosixPath(args.subdirectory).parts
        if args.destination != "colab-v1" or any(p in {"/", ".", ".."} for p in parts):
            raise ValueError("Subdirectories must remain inside colab-v1")
        for part in parts:
            parent = drive.up.ensure_subfolder(parent, part)
    for path in args.file:
        path = path.resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty file: {path}")
        if any(part in {"archive", "out-v0"} or part.startswith("_quarantine")
               for part in path.parts):
            raise ValueError(f"Refusing quarantined input: {path}")
        if path.name in MODEL_NAMES | BANK_NAMES | {NOTEBOOK_NAME}:
            raise ValueError(f"Refusing invalidated artifact name: {path.name}")
        if path.name == "phase3_health_training_150k_v1_unlabeled.parquet":
            validate_health_bank(path)
        duplicates = [f for f in drive.children(parent) if f["name"] == path.name]
        if len(duplicates) > 1:
            raise ValueError(f"Ambiguous existing name: {path.name}")
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        sha256 = hashlib.sha256(data).hexdigest()
        if duplicates and (duplicates[0].get("md5Checksum") != md5
                           or int(duplicates[0].get("size", 0)) != len(data)):
            raise ValueError(f"Existing bytes differ: {path.name}; choose a new version")
        result = drive.up.upload_file(path, parent, {f["name"]: f for f in duplicates})
        metadata = drive.metadata(result["id"])
        assert metadata["md5Checksum"] == md5
        assert int(metadata["size"]) == len(data)
        assert parent in metadata["parents"]
        receipt = {"verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "local_path": str(path.resolve()), "sha256": sha256,
                   "destination": args.destination, "upload": result, "metadata": metadata}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("a") as out:
            out.write(json.dumps(receipt, sort_keys=True) + "\n")
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
