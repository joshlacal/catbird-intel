#!/usr/bin/env python3
"""Move the audited v0/stub artifacts into Drive quarantine; never delete.

Default is a read-only inventory. --apply moves only the named artifacts,
verifies parent/checksum readback and journals each completed move for recovery.
OAuth is obtained through DriveUploader and never written into the receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import urllib.parse
import urllib.request

from sync_lakehouse_and_training_banks import DriveUploader, LAKE_FOLDERS

MODEL_NAMES = {
    "phase3_health_student_150k_int8.onnx",
    "phase5_social_student_matryoshka_int8.onnx",
    "phase5_benchmark_report.json",
    "socially_trained_student_256d_int8.json",
    "phase6_eval_report.json",
    "phase1_eval_report.json",
    "known_for.parquet",
    "communities_labeled.parquet",
    "phase3_150k_eval_report.json",
    "phase3_health_student_150k_weights.json",
    "tier3_lakehouse_manifest.json",
    "communities.parquet",
    "cluster_stats.json",
    "colab_community_labeling_prompts.jsonl",
    "producers.tsv",
    "edges.bin",
    "sim.bin",
    "stats.tsv",
}
BANK_NAMES = {
    "phase3_health_training_150k.parquet",
    # Its v0 community IDs are also invalid; prevent accidental training reuse.
    "phase5_social_encoder_pairs_500k.parquet",
}
NOTEBOOK_NAME = "Catbird_Phase3_Phase5_Colab_GPU_Trainer.ipynb"
FIELDS = "id,name,mimeType,parents,size,md5Checksum,webViewLink"


class DriveQuarantine:
    def __init__(self):
        self.up = DriveUploader()

    def request(self, path, params=None, method="GET", body=None):
        url = "https://www.googleapis.com/drive/v3/" + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.up.token(),
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)

    def metadata(self, fid):
        return self.request("files/" + fid, {"fields": FIELDS})

    def children(self, fid):
        result, page = [], None
        while True:
            params = {"q": f"'{fid}' in parents and trashed=false",
                      "fields": f"nextPageToken,files({FIELDS})", "pageSize": 1000}
            if page:
                params["pageToken"] = page
            data = self.request("files", params)
            result.extend(data.get("files", []))
            page = data.get("nextPageToken")
            if not page:
                return result

    def inventory(self):
        targets = []
        for key, names in [("models", MODEL_NAMES), ("training-banks", BANK_NAMES),
                           ("root", {NOTEBOOK_NAME})]:
            for item in self.children(LAKE_FOLDERS[key]):
                if item["name"] in names:
                    targets.append({"source": key, "before": item})
        vectors = self.metadata(LAKE_FOLDERS["tier3-vectors"])
        if LAKE_FOLDERS["root"] in vectors.get("parents", []):
            targets.append({"source": "root", "before": vectors,
                            "children": self.children(vectors["id"])})
        return targets

    def move(self, item, target):
        before = self.metadata(item["before"]["id"])
        source = LAKE_FOLDERS[item["source"]]
        if source not in before.get("parents", []):
            raise RuntimeError(f"Source parent changed for {before['id']}")
        self.request("files/" + before["id"],
                     {"addParents": target, "removeParents": source, "fields": FIELDS},
                     method="PATCH", body={})
        after = self.metadata(before["id"])
        if target not in after.get("parents", []) or source in after.get("parents", []):
            raise RuntimeError(f"Move readback failed for {before['id']}")
        for key in ("size", "md5Checksum"):
            if before.get(key) != after.get(key):
                raise RuntimeError(f"Content metadata changed for {before['id']}")
        return {**item, "before": before, "after": after}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--receipt", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1]
                    / "eval/drive_quarantine_20261020.jsonl")
    args = ap.parse_args()
    drive = DriveQuarantine()
    targets = drive.inventory()
    print(json.dumps({"apply": args.apply, "targets": targets}, indent=2), flush=True)
    if not args.apply:
        return
    target = drive.up.ensure_subfolder(LAKE_FOLDERS["root"], "_quarantine_invalidated_20261019")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    for item in targets:
        result = drive.move(item, target)
        result["verified_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        result["quarantine_id"] = target
        with args.receipt.open("a") as out:
            out.write(json.dumps(result, sort_keys=True) + "\n")
        print(f"verified move: {result['after']['name']} ({result['after']['id']})", flush=True)
    remaining = drive.inventory()
    if remaining:
        raise RuntimeError("Quarantine verification found source artifacts still present")
    print(json.dumps({"status": "verified", "quarantine": drive.metadata(target),
                      "remaining_source_targets": 0, "moved_this_run": len(targets)}), flush=True)


if __name__ == "__main__":
    main()
