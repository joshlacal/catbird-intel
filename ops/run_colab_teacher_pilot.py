#!/usr/bin/env python3
"""Prepare or explicitly launch a bounded, isolated 32-real-post Colab A100 pilot.

Default operation is local preparation only. --execute requires --approved-budget;
that flag records the caller's prior human approval, it does not grant approval.
Never allocates from --help, --self-test, or preparation. No runtime reuse, Drive
mount, browser attachment, auto-buy, retry-allocation or kernel restart exists.

Example (local preparation, network reads only for dependency resolution):
  real-model-smoke-venv/bin/python ops/run_colab_teacher_pilot.py \
      --output ~/catbird-lake-local/colab-pilot-prepared
After explicit human approval, root can execute that reviewed preparation:
  python ops/run_colab_teacher_pilot.py --execute \
      --prepared ~/catbird-lake-local/colab-pilot-prepared --approved-budget
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
MODEL_REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
MAX_SECONDS = 30 * 60
MAX_UNITS = 25.0
POLL_SECONDS = 15
USAGE_TIMEOUT = 10
STOP_RESERVE_SECONDS = 120
WORK_RESERVE_SECONDS = 240


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def redact(text):
    # CLI transport exceptions may contain authenticated proxy URLs. Keep
    # diagnostics without persisting/printing their query credentials.
    return re.sub(r"https?://[^\s\"']+", "[URL redacted]", str(text))


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclasses.dataclass(frozen=True)
class Usage:
    balance: float
    rate_per_hour: float
    assignments: int


def parse_usage(text):
    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    patterns = (r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)\s*compute units",
                r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)/hr",
                r"Active assignments:\s*([0-9]+)")
    matches = [re.search(pattern, clean) for pattern in patterns]
    if not all(matches):
        raise ValueError("Colab usage is unavailable or its CLI format changed")
    result = Usage(float(matches[0][1]), float(matches[1][1]), int(matches[2][1]))
    if not all(math.isfinite(x) and x >= 0 for x in (result.balance, result.rate_per_hour)):
        raise ValueError("Invalid Colab usage values")
    return result


def budget_reason(initial, latest, elapsed, integrated_units, max_seconds, max_units, owned):
    # Account-wide readings conservatively include other sessions. The displayed
    # balance has only two decimals, so reserve an additional 0.02 units.
    spent = max(0.0, initial.balance - latest.balance, integrated_units)
    reserve = max(1.0, latest.rate_per_hour * (STOP_RESERVE_SECONDS + POLL_SECONDS + USAGE_TIMEOUT) / 3600) + 0.02
    if elapsed >= max_seconds - STOP_RESERVE_SECONDS:
        return "time_reserve_reached"
    if spent + reserve >= max_units:
        return "compute_unit_reserve_reached"
    if latest.balance <= reserve:
        return "existing_balance_reserve_reached"
    if owned and latest.rate_per_hour <= 0:
        return "owned_gpu_usage_rate_unavailable"
    return None


def verify_file_inventory(directory, inventory):
    root = Path(directory).resolve()
    for name, expected in inventory.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError(f"Unsafe or missing prepared file: {name}")
        if digest(path) != expected:
            raise ValueError(f"Prepared file changed after review: {name}")


def runtime_from_logs(directory):
    """Recover observed runtime data even when remote artifact salvage failed."""
    observed = None
    for log in sorted(Path(directory).glob("*-bootstrap.log")):
        for line in log.read_text(errors="replace").split("\n"):
            for marker in ("PILOT_RUNTIME_OBSERVED=", "PILOT_BOOTSTRAP_RECEIPT="):
                if marker in line:
                    try:
                        observed = json.loads(line.split(marker, 1)[1])
                    except ValueError:
                        pass
    return observed


def prepare(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Preparation directory already exists; choose a new directory")
    output.mkdir(parents=True, mode=0o700)
    source = args.source.expanduser().resolve()
    bank = args.bank.expanduser().resolve()
    holdout = args.holdout.expanduser().resolve()
    export_manifest = holdout.parent / "phase3_v1_export_manifest.json"
    manifest = json.loads((source / "source_manifest.json").read_text())
    verify_file_inventory(source, manifest["files"])
    notebook = source / manifest["notebook"]["name"]
    if digest(notebook) != manifest["notebook"]["sha256"]:
        raise ValueError("Reviewed notebook digest mismatch")
    bank_manifest = json.loads(bank.with_suffix(".manifest.json").read_text())
    bank_hash = digest(bank)
    if bank_hash != bank_manifest["sha256"]:
        raise ValueError("Sealed real bank digest mismatch")
    sys.path.insert(0, str(source))
    from real_pipeline.common import load_sealed_export, validate_posts
    seal, holdout_rows = load_sealed_export(holdout, export_manifest)
    if bank_hash != seal["bank"]["sha256"]:
        raise ValueError("Pilot bank differs from the audited export seal")
    expected_holdout = seal["holdout"]["sha256"]
    if digest(holdout) != expected_holdout:
        raise ValueError("Sealed holdout digest mismatch")
    rows = next(pq.ParquetFile(bank).iter_batches(batch_size=32, use_threads=False)).to_pylist()
    validate_posts(rows)
    if len(rows) != 32 or len({r["uri"] for r in rows}) != 32:
        raise ValueError("Pilot requires exactly the first 32 distinct real bank posts")
    for row in rows:
        if not row["uri"].startswith("at://did:") or not row["author_did"].startswith("did:") or not row["text"].strip():
            raise ValueError("Missing original post URI/author/text")
        if row.get("author_split_bucket") not in range(1, 20):
            raise ValueError("Reserved holdout bucket leaked into pilot")
        if any(key.startswith("teacher_") for key in row):
            raise ValueError("Pilot source must be unlabeled, never regex teacher aliases")
    holdout_authors = sorted({r["author_did"] for r in holdout_rows})
    if {r["author_did"] for r in rows} & set(holdout_authors):
        raise ValueError("Pilot/holdout author overlap")
    data = output / "payload"
    (data / "source").mkdir(parents=True)
    for relative in manifest["files"]:
        target = data / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    shutil.copyfile(source / "source_manifest.json", data / "source/source_manifest.json")
    shutil.copyfile(notebook, data / "source" / notebook.name)
    inputs = data / "inputs"
    inputs.mkdir()
    for path in (bank, bank.with_suffix(".manifest.json"), holdout,
                 holdout.parent / (holdout.name + ".sha256"), export_manifest):
        shutil.copyfile(path, inputs / path.name)
    pq.write_table(pa.Table.from_pylist(rows), data / "pilot32.parquet", compression="zstd")
    locks = {}
    resolution = {}
    (data / "dependency-locks").mkdir()
    # Resolve wheels without installing CUDA packages locally. Bootstrap chooses
    # the exact lock for the observed interpreter, never a guessed Colab default.
    for version in ("3.10", "3.11", "3.12", "3.13"):
        relative = f"dependency-locks/python-{version}.lock.txt"
        lock = data / relative
        command = [args.uv, "pip", "compile", str(data / "source/requirements-colab.txt"),
                   "--python-version", version, "--python-platform", "x86_64-manylinux_2_28",
                   "--only-binary", ":all:", "--generate-hashes", "--no-strip-extras",
                   "--output-file", str(lock)]
        with (output / f"dependency-resolution-{version}.log").open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                resolution[version] = {"returncode": result.returncode}
            except subprocess.TimeoutExpired:
                resolution[version] = {"returncode": None, "error": "resolution_timeout"}
        if resolution[version]["returncode"] == 0 and lock.is_file():
            locks[version] = {"path": relative, "sha256": digest(lock)}
        else:
            if lock.exists():
                shutil.move(str(lock), output / f"unresolved-python-{version}.lock.txt")
            if version != "3.13":
                write_json(output / "dependency-resolution.json", resolution)
                raise RuntimeError(f"Required Python {version} wheel lock could not resolve; inspect its retained log")
    write_json(output / "dependency-resolution.json", resolution)
    files = {str(path.relative_to(data)): digest(path) for path in sorted(data.rglob("*")) if path.is_file()}
    archive = output / "pilot_payload.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            bundle.write(data / relative, relative)
    name = "catbird-intel-qwen32-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    receipt = {"schema": "catbird-colab-teacher-pilot-v1", "prepared_at": utc(),
               "status": "prepared_only_no_gpu_allocated", "session_name": name,
               "wrapper_sha256": digest(__file__), "source_manifest_sha256": digest(source / "source_manifest.json"),
               "bank_sha256": bank_hash, "bank_path": str(bank), "rows": 32,
               "selected_uris": [r["uri"] for r in rows], "holdout_sha256": expected_holdout,
               "holdout_author_count": len(holdout_authors), "holdout_author_overlap": 0,
               "holdout_text_uploaded": True, "full_canonical_bank_uploaded": True,
               "selection_audit_only": "pilot32.parquet", "export_manifest_sha256": digest(export_manifest),
               "payload_files": files, "payload_sha256": digest(archive),
               "teacher": MODEL, "teacher_revision": MODEL_REVISION,
               "include_political": args.include_political, "python_targets": sorted(locks),
               "dependency_locks": locks, "dependency_resolution": resolution,
               "dependency_platform": "Linux x86_64 manylinux_2_28", "gpu": "A100",
               "max_seconds": args.max_seconds, "max_compute_units": args.max_units,
               "stop_reserve_seconds": STOP_RESERVE_SECONDS, "work_reserve_seconds": WORK_RESERVE_SECONDS,
               "account_wide_budget_accounting": True, "dependency_resolution_only_not_gpu_install_validation": True}
    write_json(output / "preparation.json", receipt)
    print(json.dumps({"status": receipt["status"], "directory": str(output), "rows": 32,
                      "session_name": name, "payload_sha256": receipt["payload_sha256"],
                      "max_seconds": args.max_seconds, "max_compute_units": args.max_units}, indent=2))
    return receipt


class Pilot:
    def __init__(self, args, directory, prepared):
        self.args, self.directory, self.prepared = args, directory, prepared
        self.session = prepared["session_name"]
        self.config = directory / "private-session-state.json"
        self.base = [args.colab, "--config", str(self.config)]
        self.logs = directory / "execution"
        self.logs.mkdir(mode=0o700)
        self.cancel = threading.Event()
        self.monitor_done = threading.Event()
        self.stop_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.start = None
        self.initial = self.latest = None
        self.last_usage_time = None
        self.integrated_units = 0.0
        self.reason = None
        self.stop_confirmed = False
        self.allocation_attempted = False
        self.counter = 0

    def event(self, kind, **fields):
        with self.log_lock:
            with (self.logs / "events.jsonl").open("a") as handle:
                handle.write(json.dumps({"at": utc(), "event": kind, **fields}, default=str) + "\n")
        print(json.dumps({"event": kind, **fields}, default=str), flush=True)

    def owned(self):
        if not self.config.exists():
            return False
        for attempt in range(4):
            try:
                state = json.loads(self.config.read_text())
                break
            except json.JSONDecodeError:
                if attempt == 3:
                    raise RuntimeError("Isolated session state is incomplete/unreadable")
                time.sleep(0.05)
        if set(state) - {self.session}:
            raise RuntimeError("Isolated state unexpectedly contains another session; refusing to operate")
        return self.session in state

    def elapsed(self):
        return time.monotonic() - self.start if self.start is not None else 0

    def usage(self):
        result = subprocess.run(self.base + ["usage"], capture_output=True, text=True, timeout=USAGE_TIMEOUT, check=True)
        value = parse_usage(result.stdout)
        self.event("usage", **dataclasses.asdict(value), elapsed_seconds=self.elapsed())
        return value

    def stop(self):
        with self.stop_lock:
            if self.stop_confirmed:
                return
            if not self.config.exists():
                self.event("ALLOCATION_OUTCOME_UNKNOWN_NO_AUTO_RETRY" if self.allocation_attempted
                           else "stop_no_registered_owned_session",
                           session=self.session, stop_confirmed=False,
                           next_action="Inspect account sessions before any new allocation; never stop an unidentified session")
                return
            try:
                has_owned = self.owned()
            except Exception as exc:
                # Still use only the unique explicit name and isolated registry.
                self.event("owned_state_unreadable_before_stop", error=redact(exc))
                has_owned = True
            if not has_owned:
                self.event("stop_owned_session_already_absent")
                return
            for attempt in range(1, 4):
                try:
                    result = subprocess.run(self.base + ["stop", "--session", self.session], capture_output=True,
                                            text=True, timeout=25)
                    self.event("stop_attempt", attempt=attempt, returncode=result.returncode,
                               output=redact(result.stdout[-500:]), error=redact(result.stderr[-500:]))
                    if result.returncode == 0 and not self.owned():
                        self.stop_confirmed = True
                        return
                except Exception as exc:
                    self.event("stop_attempt_failed", attempt=attempt, error=str(exc))
            self.event("STOP_UNCONFIRMED_REQUIRES_ATTENTION", session=self.session,
                       isolated_config=str(self.config))

    def monitor(self):
        while not self.monitor_done.wait(POLL_SECONDS):
            try:
                now = time.monotonic()
                latest = self.usage()
                if self.last_usage_time is not None:
                    self.integrated_units += max(self.latest.rate_per_hour, latest.rate_per_hour) * (now - self.last_usage_time) / 3600
                self.latest = latest
                self.last_usage_time = now
                reason = budget_reason(self.initial, latest, self.elapsed(), self.integrated_units,
                                       self.prepared["max_seconds"], self.prepared["max_compute_units"], self.owned())
                if reason:
                    raise RuntimeError(reason)
            except Exception as exc:
                self.reason = str(exc)
                self.cancel.set()
                self.event("watchdog_stop", reason=self.reason)
                self.stop()
                return

    def call(self, name, command, timeout):
        if self.cancel.is_set():
            raise RuntimeError(f"Pilot cancelled: {self.reason}")
        self.counter += 1
        log = self.logs / f"{self.counter:02d}-{name}.log"
        self.event("command_start", name=name, timeout_seconds=timeout)
        with log.open("w") as output:
            process = subprocess.Popen(self.base + command, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            began = time.monotonic()
            while process.poll() is None:
                if self.cancel.wait(0.25) or time.monotonic() - began > timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    raise RuntimeError(f"{name} interrupted or exceeded its timeout; see {log}")
            if process.returncode:
                raise RuntimeError(f"{name} failed with status {process.returncode}; see {log}")
        # Raw CLI errors can include signed URLs; redact them before exposing logs.
        log.write_text(redact(log.read_text(errors="replace")))
        self.event("command_finished", name=name, log=str(log))
        return log

    def execute_code(self, name, code, timeout):
        path = self.logs / f"{name}.py"
        path.write_text(code)
        return self.call(name, ["exec", "--session", self.session, "--file", str(path),
                               "--timeout", str(timeout)], timeout + 10)

    def work_remaining(self):
        remaining = self.prepared["max_seconds"] - self.elapsed() - WORK_RESERVE_SECONDS
        if remaining < 45:
            raise RuntimeError("Insufficient reserved time for another work stage")
        return int(remaining)


def bootstrap_code(prepared, root_override=None):
    root = root_override or "/content/" + prepared["session_name"]
    # All values are Python literals, not shell interpolation.
    return f'''import hashlib, json, pathlib, platform, subprocess, sys, zipfile
root = pathlib.Path({root!r})
archive = pathlib.Path({(root + '.zip')!r})
receipt = {{'status': 'runtime_observed', 'python': sys.version,
    'python_version': '.'.join(map(str, sys.version_info[:2])),
    'python_executable': sys.executable, 'platform': platform.system(),
    'machine': platform.machine(), 'gpu': None, 'gpu_probe_returncode': None,
    'payload_sha256': {prepared['payload_sha256']!r},
    'source_manifest_sha256': {prepared['source_manifest_sha256']!r},
    'prepared_python_versions': {sorted(prepared['dependency_locks'])!r}}}
try:
    gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    receipt.update(gpu=gpu_result.stdout.strip(), gpu_probe_returncode=gpu_result.returncode, gpu_probe_error=gpu_result.stderr.strip())
except Exception as exc:
    receipt['gpu_probe_error'] = repr(exc)
# Emit before every gate, so local diagnostics survive a bootstrap/collection failure.
print('PILOT_RUNTIME_OBSERVED=' + json.dumps(receipt), flush=True)
created_root = False
try:
    assert not root.exists(), 'Dedicated pilot folder already exists'
    root.mkdir()
    created_root = True
    (root / 'runtime_before.json').write_text(json.dumps(receipt, indent=2))
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == {prepared['payload_sha256']!r}
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            assert (root / item.filename).resolve().is_relative_to(root.resolve()), 'Unsafe archive path'
        bundle.extractall(root)
    expected = {prepared['payload_files']!r}
    for relative, value in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == value, relative
    assert receipt['platform'] == 'Linux' and receipt['machine'] == 'x86_64', 'Dependency locks target Linux x86_64'
    locks = {prepared['dependency_locks']!r}
    assert receipt['python_version'] in locks, 'No prepared wheel lock for observed Python ' + receipt['python_version']
    selected = locks[receipt['python_version']]
    assert hashlib.sha256((root / selected['path']).read_bytes()).hexdigest() == selected['sha256'], 'Selected dependency lock changed'
    receipt['selected_dependency_lock'] = selected
    assert receipt['gpu_probe_returncode'] == 0, 'GPU probe failed: ' + str(receipt.get('gpu_probe_error'))
    assert 'A100' in receipt['gpu'], 'Unapproved GPU variant: ' + str(receipt['gpu'])
    assert int(receipt['gpu'].split(',')[1].strip().split()[0]) >= 38000, 'Insufficient GPU memory'
    receipt['status'] = 'bootstrap_verified'
except BaseException as exc:
    receipt.update(status='bootstrap_failed', error=repr(exc))
finally:
    if created_root:
        (root / 'runtime_before.json').write_text(json.dumps(receipt, indent=2))
    print('PILOT_BOOTSTRAP_RECEIPT=' + json.dumps(receipt), flush=True)
if receipt['status'] == 'bootstrap_verified':
    print('PILOT_BOOTSTRAP_VERIFIED', flush=True)
'''


def install_code(prepared, timeout):
    root = "/content/" + prepared["session_name"]
    return f'''import hashlib, json, pathlib, subprocess, sys, time
root = pathlib.Path({root!r})
started = time.monotonic()
runtime = json.loads((root / 'runtime_before.json').read_text())
receipt = {{'returncode': None, 'python': sys.version, 'runtime': runtime, 'status': 'install_running'}}
try:
    assert runtime['status'] == 'bootstrap_verified', 'Bootstrap did not pass'
    version = '.'.join(map(str, sys.version_info[:2]))
    locks = {prepared['dependency_locks']!r}
    assert version == runtime['python_version'] and version in locks, 'Interpreter changed after bootstrap'
    selected = locks[version]
    assert selected == runtime['selected_dependency_lock'], 'Selected lock identity changed'
    assert hashlib.sha256((root / selected['path']).read_bytes()).hexdigest() == selected['sha256'], 'Selected lock bytes changed'
    with (root / 'install.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes', '-r', str(root / selected['path'])], stdout=log, stderr=subprocess.STDOUT, timeout={timeout})
    receipt['returncode'] = result.returncode
    assert result.returncode == 0, 'Dependency install failed; install.log retained'
    receipt['status'] = 'install_verified'
except BaseException as exc:
    receipt.update(status='install_failed', error=repr(exc))
finally:
    receipt['seconds'] = time.monotonic() - started
    (root / 'install_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('PILOT_INSTALL_RECEIPT=' + json.dumps(receipt), flush=True)
if receipt['status'] == 'install_verified':
    print('PILOT_INSTALL_VERIFIED', flush=True)
'''


def inference_code(prepared, timeout):
    root = "/content/" + prepared["session_name"]
    teachers = ["qwen", "political"] if prepared["include_political"] else ["qwen"]
    return f'''import hashlib, importlib.metadata, json, os, pathlib, signal, subprocess, sys, time, traceback
root = pathlib.Path({root!r})
started = time.monotonic()
receipt = {{'status': 'running', 'rows': 32, 'teachers': {teachers!r}, 'teacher_model': {MODEL!r}, 'teacher_revision': {MODEL_REVISION!r}, 'credit_use': None,
    'runtime': json.loads((root / 'runtime_before.json').read_text())}}
try:
    for teacher in {teachers!r}:
        remaining = {timeout} - (time.monotonic() - started)
        assert remaining > 20, 'Pilot inference time reserve reached'
        command = [sys.executable, '-u', str(root / 'source/real_pipeline.py'), 'label', '--input', str(root / 'inputs/phase3_health_training_150k_v1_unlabeled.parquet'), '--holdout', str(root / 'inputs/phase3_holdout_600_unlabeled.jsonl'), '--export-manifest', str(root / 'inputs/phase3_v1_export_manifest.json'), '--mode', 'smoke', '--device', 'cuda', '--teacher', teacher, '--batch-size', '4', '--max-length', '1024' if teacher == 'qwen' else '512', '--output', str(root / teacher)]
        with (root / (teacher + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                raise
        assert code == 0, f'{{teacher}} inference failed, see its log'
        manifest = json.loads((root / teacher / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['identity']['rows'] == 32
        assert manifest['identity']['model'] == ([{MODEL!r}, {MODEL_REVISION!r}] if teacher == 'qwen' else ['mlburnham/Political_DEBATE_large_v1.0', '1a3aff1ecb97ad93a6de598f7414b1707de7e7c3'])
    receipt['status'] = 'teacher_inference_complete'
except BaseException as exc:
    receipt['status'] = 'failed'
    receipt['error'] = repr(exc)
    (root / 'failure_traceback.txt').write_text(traceback.format_exc())
finally:
    receipt['seconds'] = time.monotonic() - started
    receipt['versions'] = {{name: importlib.metadata.version(name) for name in ['torch', 'vllm', 'transformers', 'pyarrow']}}
    (root / 'pilot_result.json').write_text(json.dumps(receipt, indent=2))
print('PILOT_INFERENCE_FINISHED', json.dumps(receipt))
'''


def collect_code(prepared):
    root = "/content/" + prepared["session_name"]
    return f'''import hashlib, json, pathlib, subprocess, zipfile
root = pathlib.Path({root!r})
result = root / 'pilot_outputs.zip'
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    after = {{'returncode': gpu.returncode, 'gpu': gpu.stdout.strip(), 'error': gpu.stderr.strip()}}
except Exception as exc:
    after = {{'returncode': None, 'gpu': None, 'error': repr(exc)}}
(root / 'runtime_after.json').write_text(json.dumps(after))
files = [p for p in root.rglob('*') if p.is_file() and (p.parent == root or p.relative_to(root).parts[0] in ('qwen', 'political')) and p.name not in ('pilot32.parquet', 'pilot_requirements.lock.txt', 'pilot_outputs.zip', 'output_manifest.json')]
manifest = {{str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(root / 'output_manifest.json').write_text(json.dumps(manifest, indent=2))
with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for p in files + [root / 'output_manifest.json']:
        bundle.write(p, str(p.relative_to(root)))
print('PILOT_OUTPUT_SHA256=' + hashlib.sha256(result.read_bytes()).hexdigest())
'''


def verify_outputs(directory, prepared, expected_digest):
    archive = directory / "pilot_outputs.zip"
    if digest(archive) != expected_digest:
        raise ValueError("Downloaded pilot archive digest mismatch")
    target = directory / "downloaded"
    target.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError("Unsafe downloaded archive path")
        bundle.extractall(target)
    manifest = json.loads((target / "output_manifest.json").read_text())
    verify_file_inventory(target, manifest)
    result = json.loads((target / "pilot_result.json").read_text()) if (target / "pilot_result.json").exists() else {"status": "failed_before_inference"}
    for name, key in (("runtime_before.json", "runtime"), ("install_receipt.json", "installation")):
        if (target / name).exists():
            result[key] = json.loads((target / name).read_text())
    if result["status"] == "teacher_inference_complete":
        for teacher in result["teachers"]:
            stage = json.loads((target / teacher / "manifest.json").read_text())
            if stage["identity"]["input_sha256"] != prepared["bank_sha256"]:
                raise ValueError("Teacher output is not from the reviewed canonical input")
            verify_file_inventory(target / teacher, stage["files"])
            labels = []
            for name in stage["files"]:
                labels += [json.loads(line) for line in (target / teacher / name).read_text().split("\n") if line]
            if [r["uri"] for r in labels] != prepared["selected_uris"]:
                raise ValueError("Teacher output URI order/set mismatch")
            if teacher == "qwen":
                for row in labels:
                    for metric in ("outraged", "critical", "engagement_bait", "spam"):
                        score = row[f"teacher_{metric}"]
                        logits = row[f"teacher_{metric}_digit_logprobs"]
                        if not math.isfinite(score) or not 0 <= score <= 1 or len(logits) != 10 or not all(math.isfinite(x) for x in logits):
                            raise ValueError("Invalid Qwen score or missing actual digit logprobs")
    return result


def execute(args):
    if not args.approved_budget:
        raise ValueError("Execution requires prior explicit human approval and --approved-budget")
    directory = args.prepared.expanduser().resolve()
    prepared = json.loads((directory / "preparation.json").read_text())
    if prepared["wrapper_sha256"] != digest(__file__):
        raise ValueError("Wrapper changed after preparation; prepare/review a fresh directory")
    if prepared["max_seconds"] > MAX_SECONDS or prepared["max_compute_units"] > MAX_UNITS:
        raise ValueError("Prepared budget exceeds the authorized 30-minute/25-unit ceiling")
    verify_file_inventory(directory / "payload", prepared["payload_files"])
    if digest(directory / "pilot_payload.zip") != prepared["payload_sha256"]:
        raise ValueError("Prepared upload archive changed")
    if (directory / "execution").exists() or (directory / "private-session-state.json").exists():
        raise ValueError("This preparation was already attempted; never reuse or reallocate it")
    os.umask(0o077)
    pilot = Pilot(args, directory, prepared)
    thread = None
    allocated = False
    allocation_attempted = False
    outputs = None
    failure = None
    try:
        pilot.initial = pilot.latest = pilot.usage()
        if pilot.initial.balance < prepared["max_compute_units"]:
            raise RuntimeError("Existing compute-unit balance does not cover the approved ceiling; no purchase allowed")
        pilot.start = pilot.last_usage_time = time.monotonic()
        thread = threading.Thread(target=pilot.monitor, name="colab-pilot-budget-watch", daemon=True)
        thread.start()
        allocation_attempted = True
        pilot.allocation_attempted = True
        pilot.call("allocate", ["new", "--session", pilot.session, "--gpu", "A100"], 180)
        allocated = pilot.owned()
        if not allocated:
            raise RuntimeError("CLI allocation did not register the isolated owned session")
        pilot.call("upload", ["upload", "--session", pilot.session, str(directory / "pilot_payload.zip"),
                               "content/" + pilot.session + ".zip"], 90)
        boot_log = pilot.execute_code("bootstrap", bootstrap_code(prepared), 90)
        if "PILOT_BOOTSTRAP_VERIFIED" not in boot_log.read_text():
            raise RuntimeError("Remote bootstrap did not verify GPU/input identities")
        install_timeout = min(900, pilot.work_remaining())
        install_log = pilot.execute_code("install", install_code(prepared, install_timeout), install_timeout + 5)
        if "PILOT_INSTALL_VERIFIED" not in install_log.read_text():
            raise RuntimeError("Pinned dependency installation failed")
        infer_timeout = pilot.work_remaining()
        inference_log = pilot.execute_code("inference", inference_code(prepared, infer_timeout), infer_timeout + 10)
        if "PILOT_INFERENCE_FINISHED" not in inference_log.read_text():
            raise RuntimeError("Remote inference did not write its completion/failure receipt")
    except BaseException as exc:
        failure = repr(exc)
        pilot.event("pilot_failure", error=failure)
    finally:
        # Bounded best-effort salvage before stopping. Budget/telemetry failure
        # takes precedence: the watchdog stops immediately and local evidence
        # records why outputs could not be recovered.
        try:
            can_recover = allocated and pilot.owned() and not pilot.cancel.is_set()
        except Exception as exc:
            can_recover = False
            pilot.event("output_recovery_state_unavailable", error=redact(exc))
        if can_recover:
            try:
                collect_log = pilot.execute_code("collect", collect_code(prepared), 35)
                match = re.search(r"PILOT_OUTPUT_SHA256=([a-f0-9]{64})", collect_log.read_text())
                if not match:
                    raise RuntimeError("Remote output archive was not sealed")
                pilot.call("download", ["download", "--session", pilot.session,
                                        "content/" + pilot.session + "/pilot_outputs.zip",
                                        str(directory / "pilot_outputs.zip")], 35)
                outputs = verify_outputs(directory, prepared, match[1])
                pilot.event("output_verified", status=outputs["status"], archive_sha256=match[1])
            except BaseException as exc:
                failure = failure or repr(exc)
                pilot.event("output_recovery_failed", error=repr(exc))
        try:
            pilot.stop()
        except BaseException as exc:
            failure = failure or repr(exc)
            pilot.event("STOP_UNCONFIRMED_REQUIRES_ATTENTION", error=redact(exc), session=pilot.session)
        pilot.monitor_done.set()
        if thread:
            thread.join(timeout=USAGE_TIMEOUT + 1)
        try:
            after = pilot.usage()
        except Exception as exc:
            after = None
            pilot.event("final_usage_unavailable", error=str(exc))
        receipt = {"schema": "catbird-colab-teacher-pilot-execution-v1", "session": pilot.session,
                   "allocation_attempted": allocation_attempted, "allocated": allocated, "stop_confirmed": pilot.stop_confirmed,
                   "allocation_outcome_uncertain": allocation_attempted and not allocated and not pilot.stop_confirmed,
                   "automatic_allocation_retry_allowed": False,
                   "elapsed_seconds": pilot.elapsed(), "usage_before": dataclasses.asdict(pilot.initial) if pilot.initial else None,
                   "usage_after": dataclasses.asdict(after) if after else None,
                   "account_balance_delta": max(0, pilot.initial.balance - after.balance) if pilot.initial and after else None,
                   "integrated_account_rate_units": pilot.integrated_units,
                   "budget_accounting_scope": "account-wide; may include other sessions; conservative, not exact pilot billing",
                   "watchdog_reason": pilot.reason, "error": failure, "outputs": outputs,
                   "runtime": (outputs or {}).get("runtime") or runtime_from_logs(pilot.logs),
                   "prepared_sha256": digest(directory / "preparation.json"), "wrapper_sha256": digest(__file__),
                   "status": "teacher_pilot_complete" if outputs and outputs.get("status") == "teacher_inference_complete" and pilot.stop_confirmed and not failure else "failed_or_incomplete"}
        for log in pilot.logs.glob("*.log"):
            log.write_text(redact(log.read_text(errors="replace")))
        write_json(directory / "execution_receipt.json", receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    if receipt["status"] != "teacher_pilot_complete":
        raise RuntimeError("Pilot incomplete; see execution_receipt.json and retained logs")


def self_test():
    # Pure accounting fixtures, not model examples, labels, throughput or bills.
    initial = Usage(100, 0, 0)
    assert parse_usage("Current balance: 100.00 compute units\nUsage rate: 12.00/hr\nActive assignments: 1") == Usage(100, 12, 1)
    assert budget_reason(initial, Usage(99, 12, 1), 10, 1, 1800, 25, True) is None
    assert budget_reason(initial, Usage(76, 12, 1), 10, 24, 1800, 25, True) == "compute_unit_reserve_reached"
    assert budget_reason(initial, Usage(99, 12, 1), 1680, 1, 1800, 25, True) == "time_reserve_reached"
    assert budget_reason(initial, Usage(99, 0, 1), 10, 1, 1800, 25, True) == "owned_gpu_usage_rate_unavailable"
    try:
        parse_usage("unavailable")
    except ValueError:
        pass
    else:
        raise AssertionError("Unparseable usage must fail closed")
    # Execute the generated bootstrap against temporary metadata-only archives,
    # mocked Python/GPU observations and no network/process launches.
    import contextlib
    import io
    import platform
    import tempfile
    from unittest.mock import patch
    cases = 0
    for minor, gpu_ok in ((10, True), (11, True), (12, True), (13, True), (14, True), (12, False)):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "test-runtime"
            archive = Path(str(root) + ".zip")
            locks = {}
            inventory = {}
            with zipfile.ZipFile(archive, "w") as bundle:
                for version in ("3.10", "3.11", "3.12", "3.13"):
                    name = f"dependency-locks/python-{version}.lock.txt"
                    contents = f"# Structural lock fixture for {version}\n".encode()
                    value = hashlib.sha256(contents).hexdigest()
                    locks[version] = {"path": name, "sha256": value}
                    inventory[name] = value
                    bundle.writestr(name, contents)
            prepared = {"session_name": "test-runtime", "payload_sha256": digest(archive),
                        "source_manifest_sha256": "0" * 64, "payload_files": inventory,
                        "dependency_locks": locks}
            output = io.StringIO()
            gpu = subprocess.CompletedProcess([], 0 if gpu_ok else 1, "NVIDIA A100-SXM4-40GB, 40960 MiB, structural-driver", "" if gpu_ok else "structural probe failure")
            with patch.object(sys, "version_info", (3, minor, 0, "final", 0)), \
                    patch.object(sys, "version", f"3.{minor}.0 structural metadata fixture"), \
                    patch.object(platform, "system", return_value="Linux"), \
                    patch.object(platform, "machine", return_value="x86_64"), \
                    patch.object(subprocess, "run", return_value=gpu), \
                    contextlib.redirect_stdout(output):
                exec(compile(bootstrap_code(prepared, str(root)), "<bootstrap-self-test>", "exec"), {})
            receipt = json.loads((root / "runtime_before.json").read_text())
            assert receipt["python_version"] == f"3.{minor}" and receipt["gpu"] == gpu.stdout
            if minor != 14 and gpu_ok:
                assert receipt["status"] == "bootstrap_verified"
                assert receipt["selected_dependency_lock"] == locks[f"3.{minor}"]
            else:
                assert receipt["status"] == "bootstrap_failed" and receipt["error"]
                assert "PILOT_BOOTSTRAP_VERIFIED" not in output.getvalue()
            (directory / "03-bootstrap.log").write_text(output.getvalue())
            assert runtime_from_logs(directory) == receipt
            cases += 1
    print(json.dumps({"self_test": "passed", "scope": "budget/parser and mocked runtime-lock selection",
                      "runtime_gate_cases": cases, "gpu_allocated": False, "model_inference": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New local preparation directory; default mode only prepares")
    parser.add_argument("--prepared", type=Path, help="Previously prepared and reviewed directory to execute once")
    parser.add_argument("--execute", action="store_true", help="Allocate the reviewed dedicated A100 pilot only after approval")
    parser.add_argument("--approved-budget", action="store_true", help="Caller attests prior explicit human approval; not an approval mechanism")
    parser.add_argument("--self-test", action="store_true", help="Pure parser/budget tests; no network or GPU")
    parser.add_argument("--bank", type=Path, default=Path.home() / "catbird-lake-local/drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet")
    parser.add_argument("--holdout", type=Path, default=REPO / "eval/phase3_holdout_600_unlabeled.jsonl")
    parser.add_argument("--source", type=Path, default=REPO / "colab")
    parser.add_argument("--include-political", action="store_true", help="Also label the same32 with Political DEBATE inside the same cap")
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--max-units", type=float, default=MAX_UNITS)
    parser.add_argument("--colab", default="colab")
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not WORK_RESERVE_SECONDS + 60 <= args.max_seconds <= MAX_SECONDS or not 1.1 < args.max_units <= MAX_UNITS:
        parser.error("Budgets must be bounded by30minutes/25existing units and leave a stop reserve")
    if args.execute:
        if not args.prepared:
            parser.error("--execute requires --prepared")
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}; stopping only the owned pilot")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        execute(args)
    else:
        if not args.output:
            parser.error("Preparation requires --output with a new local directory")
        prepare(args)


if __name__ == "__main__":
    main()
