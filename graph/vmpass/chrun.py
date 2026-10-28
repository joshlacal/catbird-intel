"""Run a clickhouse-local script and measure wall time and peak RSS.

Peak RSS is sampled with `ps` every 0.5 s over the clickhouse process tree, so
the S8 projection gets a per-step memory figure on both Linux and macOS.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import threading
import time

CLICKHOUSE = os.environ.get("VMPASS_CLICKHOUSE", shutil.which("clickhouse") or "clickhouse")


class StepError(RuntimeError):
    pass


def _rss_kb(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return int(out or 0)
    except (ValueError, OSError):
        return 0


def run_script(sql: str, workdir: pathlib.Path, name: str, timeout: float | None = None) -> dict:
    """Write `sql` to workdir/sql/<name>.sql and run it with clickhouse local in a
    per-step scratch --path. Returns {seconds, peak_rss_bytes, stdout}."""
    workdir = pathlib.Path(workdir).resolve()
    sqldir = workdir / "sql"
    sqldir.mkdir(parents=True, exist_ok=True)
    script = sqldir / f"{name}.sql"
    script.write_text(sql)
    chpath = workdir / "chtmp" / name
    if chpath.exists():
        shutil.rmtree(chpath)
    chpath.mkdir(parents=True)
    cmd = [CLICKHOUSE, "local", "--path", str(chpath), "--queries-file", str(script)]
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(workdir))
    peak = [0]
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            peak[0] = max(peak[0], _rss_kb(proc.pid))
            done.wait(0.5)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise StepError(f"{name}: timeout after {timeout}s")
    finally:
        done.set()
        th.join()
    secs = time.monotonic() - t0
    shutil.rmtree(chpath, ignore_errors=True)
    if proc.returncode != 0:
        raise StepError(f"{name}: clickhouse exited {proc.returncode}\n{stderr[-4000:]}")
    return {"seconds": round(secs, 3), "peak_rss_bytes": peak[0] * 1024, "stdout": stdout}


def query(sql: str, workdir: pathlib.Path, name: str = "q") -> str:
    return run_script(sql, workdir, name)["stdout"]
