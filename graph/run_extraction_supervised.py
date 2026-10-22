#!/usr/bin/env python3
"""Supervisor for the 3-range GCE graph extraction on catbird-extract-20261022.

Enforces:
1. Runs three processes without --once:
   - a: --min-index 3671 --max-index 4580 --work /data/extract-a
   - b: --min-index 4581 --max-index 5490 --work /data/extract-b
   - c: --min-index 5491 --max-index 6413 --work /data/extract-c
2. Hard rule: If a segment fails verification (archive_graph_extract.py logs
   "deferring_segment_until_drivefs_flush" or exits non-zero), immediately stop
   that process (SIGTERM process group), record and print the full error text
   (so we can distinguish a corrupt segment/receipt mismatch from a temporary
   network blip before any restart), write halted_on_failure.json, and halt that
   range without skipping the segment.
3. When a process logs {"msg": "waiting"}, verify its ledger.json contains every
   index in [min_index, max_index]. If complete, terminate that process cleanly
   and generate manifest.json (listing each file in events/, its row_count,
   bytes, and sha256) + upload events/, ledger.json, and manifest.json to
   gs://catbird-436600-jetstream-buffer-20261020/derived/graph-events-v1/{a,b,c}/.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time

import pyarrow.parquet as pq

GCS_PREFIX = "gs://catbird-436600-jetstream-buffer-20261020/sources/gce-selfhost-001/buffer-admission-20261020"
DERIVED_PREFIX = "gs://catbird-436600-jetstream-buffer-20261020/derived/graph-events-v1"
DECODER = "/opt/backfill/decode_segments.mjs"
EXTRACTOR = "/opt/catbird-intel/graph/archive_graph_extract.py"
STATUS_PATH = pathlib.Path("/data/extract_supervisor_status.json")

RANGES = {
    "a": {"min_index": 3671, "max_index": 4580, "work": pathlib.Path("/data/extract-a")},
    "b": {"min_index": 4581, "max_index": 5490, "work": pathlib.Path("/data/extract-b")},
    "c": {"min_index": 5491, "max_index": 6413, "work": pathlib.Path("/data/extract-c")},
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ledger_range(work: pathlib.Path, min_idx: int, max_idx: int) -> tuple[bool, dict]:
    ledger_path = work / "ledger.json"
    if not ledger_path.exists():
        return False, {"error": "ledger.json missing"}
    ledger = json.loads(ledger_path.read_text())
    segments = ledger.get("segments", {})
    indices = [int(v["index"]) for v in segments.values()]
    expected = set(range(min_idx, max_idx + 1))
    got = set(indices)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    dups = len(indices) - len(got)
    ok = (not missing) and (not extra) and (dups == 0)
    return ok, {
        "expected_count": len(expected),
        "actual_count": len(indices),
        "missing_count": len(missing),
        "missing_sample": missing[:10],
        "extra_sample": extra[:10],
        "duplicate_count": dups,
    }


def build_manifest_and_upload(slot: str, cfg: dict) -> dict:
    work: pathlib.Path = cfg["work"]
    events_dir = work / "events"
    ledger_path = work / "ledger.json"
    files_meta = []
    total_rows = 0
    total_bytes = 0
    for f in sorted(events_dir.glob("*.parquet")):
        meta = pq.read_metadata(f)
        rows = int(meta.num_rows)
        size = int(f.stat().st_size)
        sha = sha256_file(f)
        files_meta.append({
            "file": f"events/{f.name}",
            "row_count": rows,
            "bytes": size,
            "sha256": sha,
        })
        total_rows += rows
        total_bytes += size

    ledger_sha = sha256_file(ledger_path)
    ledger_bytes = int(ledger_path.stat().st_size)
    ok, ledger_check = verify_ledger_range(work, cfg["min_index"], cfg["max_index"])
    if not ok:
        raise RuntimeError(f"Ledger range check failed for {slot}: {ledger_check}")

    manifest = {
        "schema": "catbird.graph.extracted-events-manifest.v1",
        "slot": slot,
        "min_index": cfg["min_index"],
        "max_index": cfg["max_index"],
        "segment_count": ledger_check["actual_count"],
        "parquet_file_count": len(files_meta),
        "total_rows": total_rows,
        "total_parquet_bytes": total_bytes,
        "ledger": {
            "file": "ledger.json",
            "bytes": ledger_bytes,
            "sha256": ledger_sha,
            "segment_count": ledger_check["actual_count"],
        },
        "files": files_meta,
        "created_at": time.strftime("%FT%TZ", time.gmtime()),
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    dest = f"{DERIVED_PREFIX}/{slot}"
    env = {**os.environ, "CLOUDSDK_CONFIG": str(work / ".gcloud")}
    subprocess.run(["gcloud", "storage", "cp", "-r", str(events_dir), f"{dest}/"], check=True, env=env)
    subprocess.run(["gcloud", "storage", "cp", str(ledger_path), str(manifest_path), f"{dest}/"], check=True, env=env)
    return manifest


def prepare_slot_gcloud_config(work: pathlib.Path) -> str:
    slot_gcloud = work / ".gcloud"
    slot_gcloud.mkdir(parents=True, exist_ok=True)
    default_gcloud = pathlib.Path.home() / ".config" / "gcloud"
    if default_gcloud.exists() and not (slot_gcloud / "configurations").exists():
        shutil.copytree(default_gcloud, slot_gcloud, dirs_exist_ok=True)
    return str(slot_gcloud)


class WorkerThread(threading.Thread):
    def __init__(self, slot: str, cfg: dict, status_lock: threading.Lock, shared_status: dict):
        super().__init__(daemon=True)
        self.slot = slot
        self.cfg = cfg
        self.lock = status_lock
        self.shared = shared_status
        self.proc: subprocess.Popen | None = None

    def update(self, **kwargs):
        with self.lock:
            self.shared[self.slot].update(kwargs)
            self.shared[self.slot]["updated_at"] = time.strftime("%FT%TZ", time.gmtime())
            tmp = STATUS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.shared, indent=2) + "\n")
            os.replace(tmp, STATUS_PATH)

    def record_halt(self, halt_file: pathlib.Path, reason: str, error_text: str, extra: dict | None = None):
        err_info = {
            "ts": time.strftime("%FT%TZ", time.gmtime()),
            "slot": self.slot,
            "reason": reason,
            "error_text": error_text,
        }
        if extra:
            err_info.update(extra)
        halt_file.write_text(json.dumps(err_info, indent=2) + "\n")
        print(json.dumps({"msg": "process_halted_on_failure", **err_info}), flush=True)
        self.update(state="halted_on_failure", error_text=error_text, error=err_info)

    def run(self):
        work: pathlib.Path = self.cfg["work"]
        halt_file = work / "halted_on_failure.json"
        while True:
            if halt_file.exists():
                prior = json.loads(halt_file.read_text())
                err_text = prior.get("error_text") or json.dumps(prior)
                print(
                    json.dumps({
                        "ts": time.strftime("%FT%TZ", time.gmtime()),
                        "msg": "refusing_restart_until_halt_inspected",
                        "slot": self.slot,
                        "error_text": err_text,
                        "halt": prior,
                    }),
                    flush=True,
                )
                self.update(state="halted_on_failure", error_text=err_text, error=prior)
                while halt_file.exists():
                    time.sleep(5)
                print(
                    json.dumps({
                        "ts": time.strftime("%FT%TZ", time.gmtime()),
                        "msg": "halt_cleared_resuming_slot",
                        "slot": self.slot,
                        "prior_error_text": err_text,
                    }),
                    flush=True,
                )

            ok, check = verify_ledger_range(work, self.cfg["min_index"], self.cfg["max_index"])
            if ok and (work / "manifest.json").exists():
                self.update(state="completed", ledger_check=check)
                return

            gcloud_cfg = prepare_slot_gcloud_config(work)
            env = {**os.environ, "CLOUDSDK_CONFIG": gcloud_cfg}
            log_path = work / "extract.log"
            cmd = [
                "python3", EXTRACTOR,
                "--gcs-prefix", GCS_PREFIX,
                "--min-index", str(self.cfg["min_index"]),
                "--max-index", str(self.cfg["max_index"]),
                "--decoder", DECODER,
                "--work", str(work),
                "--batch", "12",
                "--parallel", "4",
            ]
            self.update(state="running", cmd=cmd, started_at=time.strftime("%FT%TZ", time.gmtime()), error=None, error_text=None)
            recent_lines: collections.deque[str] = collections.deque(maxlen=30)
            halted_in_loop = False
            with log_path.open("a", buffering=1) as log_fh:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                    env=env,
                )
                self.update(pid=self.proc.pid)
                assert self.proc.stdout is not None
                for raw_line in self.proc.stdout:
                    log_fh.write(raw_line)
                    line = raw_line.strip()
                    if not line:
                        continue
                    recent_lines.append(line)
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if ev.get("msg") == "deferring_segment_until_drivefs_flush":
                        # Hard rule: If a segment fails verification, stop that process and report. Don't skip it.
                        try:
                            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        seg_name = ev.get("segment", "<unknown>")
                        err_text = f"segment={seg_name}: {ev.get('error', '')}"
                        self.record_halt(
                            halt_file,
                            reason="segment_verification_failed",
                            error_text=err_text,
                            extra={"segment": seg_name, "event": ev, "recent_log": list(recent_lines)},
                        )
                        halted_in_loop = True
                        break

                    if "done_total" in ev:
                        self.update(
                            done_total=ev["done_total"],
                            last_segments=ev.get("segments"),
                            last_graph_rows=ev.get("graph_rows"),
                            download_mb_s=ev.get("download_mb_s"),
                            decode_s=ev.get("decode_s"),
                        )

                    if ev.get("msg") == "waiting":
                        try:
                            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        ok, check = verify_ledger_range(work, self.cfg["min_index"], self.cfg["max_index"])
                        if not ok:
                            err_text = f"Process entered waiting state before range [{self.cfg['min_index']}, {self.cfg['max_index']}] completed: {json.dumps(check)}"
                            self.record_halt(
                                halt_file,
                                reason="waiting_before_range_complete",
                                error_text=err_text,
                                extra={"event": ev, "ledger_check": check, "recent_log": list(recent_lines)},
                            )
                            halted_in_loop = True
                            break
                        self.update(state="uploading_derived", ledger_check=check)
                        try:
                            manifest = build_manifest_and_upload(self.slot, self.cfg)
                            self.update(
                                state="completed",
                                ledger_check=check,
                                total_rows=manifest["total_rows"],
                                parquet_file_count=manifest["parquet_file_count"],
                                total_parquet_bytes=manifest["total_parquet_bytes"],
                            )
                            return
                        except Exception as exc:
                            self.record_halt(
                                halt_file,
                                reason="upload_or_manifest_failed",
                                error_text=str(exc),
                                extra={"recent_log": list(recent_lines)},
                            )
                            halted_in_loop = True
                            break

                rc = self.proc.wait()
                if not halted_in_loop and rc != 0:
                    tail_text = "\n".join(recent_lines)
                    self.record_halt(
                        halt_file,
                        reason=f"nonzero_exit_{rc}",
                        error_text=f"exit_code={rc}: {tail_text}",
                        extra={"exit_code": rc, "recent_log": list(recent_lines)},
                    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clear-halt",
        choices=list(RANGES),
        action="append",
        default=[],
        help="Report and clear halted_on_failure.json for a slot after confirming a transient network error",
    )
    ap.add_argument(
        "--clear-only",
        action="store_true",
        help="Only report and clear the requested --clear-halt slot(s) and exit (allowing the running supervisor thread to resume)",
    )
    args = ap.parse_args()

    for slot in args.clear_halt:
        halt_file = RANGES[slot]["work"] / "halted_on_failure.json"
        if halt_file.exists():
            prior = json.loads(halt_file.read_text())
            err_text = prior.get("error_text") or prior.get("error", {}).get("event", {}).get("error") or json.dumps(prior)
            print(
                json.dumps({
                    "ts": time.strftime("%FT%TZ", time.gmtime()),
                    "msg": "clearing_inspected_halt_before_restart",
                    "slot": slot,
                    "error_text": err_text,
                    "prior_halt": prior,
                }),
                flush=True,
            )
            archived = RANGES[slot]["work"] / f"halted_on_failure.{int(time.time())}.cleared.json"
            os.replace(halt_file, archived)

    if args.clear_only:
        return 0

    lock = threading.Lock()
    shared = {
        slot: {
            "min_index": cfg["min_index"],
            "max_index": cfg["max_index"],
            "expected_segments": cfg["max_index"] - cfg["min_index"] + 1,
            "state": "initializing",
        }
        for slot, cfg in RANGES.items()
    }
    threads = [WorkerThread(slot, cfg, lock, shared) for slot, cfg in RANGES.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
