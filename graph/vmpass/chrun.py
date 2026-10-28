"""Run clickhouse-local scripts and measure wall time, CPU and peak RSS.

Every script runs with `--progress err`; its stderr goes to work/logs/<name>.stderr.
A sampler thread polls the process every 0.5 s (peak RSS via `ps`) and, every
`heartbeat_s`, calls `heartbeat(name, info)` with elapsed time, RSS, CPU% and
ClickHouse's own latest progress line (rows/bytes read and rates), so an overrun
still shows where time went.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time

CLICKHOUSE = os.environ.get("VMPASS_CLICKHOUSE", shutil.which("clickhouse") or "clickhouse")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class StepError(RuntimeError):
    pass


def _ps(pid: int) -> tuple[int, float]:
    try:
        out = subprocess.run(["ps", "-o", "rss=,%cpu=", "-p", str(pid)], capture_output=True, text=True).stdout.split()
        return int(out[0]), float(out[1])
    except (ValueError, OSError, IndexError):
        return 0, 0.0


def last_progress(path: pathlib.Path) -> str:
    """ClickHouse's latest 'Progress: ...' line from a stderr file (ANSI stripped)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            n = fh.tell()
            fh.seek(max(0, n - 8192))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    segs = [s for s in re.split(r"[\r\n]", _ANSI.sub("", tail)) if "Progress:" in s]
    return segs[-1].split("Progress:", 1)[1].strip() if segs else ""


def _errors(path: pathlib.Path) -> str:
    try:
        text = _ANSI.sub("", path.read_text(errors="replace"))
    except OSError:
        return ""
    return "\n".join(s for s in re.split(r"[\r\n]", text) if s.strip() and "Progress:" not in s)[-4000:]


def run_script(sql: str, workdir: pathlib.Path, name: str, timeout: float | None = None,
               heartbeat=None, heartbeat_s: float = 120, max_threads: int | None = None) -> dict:
    """Write `sql` to workdir/sql/<name>.sql and run it with clickhouse local in a
    per-step scratch --path. Returns {seconds, peak_rss_bytes, cpu_seconds, stdout}."""
    workdir = pathlib.Path(workdir).resolve()
    sqldir, logdir = workdir / "sql", workdir / "logs"
    sqldir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    script = sqldir / f"{name}.sql"
    script.write_text((f"SET max_threads = {int(max_threads)};\n" if max_threads else "") + sql)
    chpath = workdir / "chtmp" / name
    if chpath.exists():
        shutil.rmtree(chpath)
    chpath.mkdir(parents=True)
    errpath = logdir / f"{name}.stderr"
    cmd = [CLICKHOUSE, "local", "--path", str(chpath), "--progress", "err", "--queries-file", str(script)]
    t0 = time.monotonic()
    with open(errpath, "wb") as errfh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfh, text=True, cwd=str(workdir))
        peak = [0]
        cpu_samples: list[float] = []
        done = threading.Event()

        def sample() -> None:
            last_hb = time.monotonic()
            while not done.is_set():
                rss, cpu = _ps(proc.pid)
                peak[0] = max(peak[0], rss)
                cpu_samples.append(cpu)
                if heartbeat and time.monotonic() - last_hb >= heartbeat_s:
                    last_hb = time.monotonic()
                    try:
                        heartbeat(name, {"elapsed_s": round(last_hb - t0), "rss_gb": round(rss / 2**20, 1),
                                         "cpu_pct": cpu, "clickhouse_progress": last_progress(errpath)})
                    except Exception:  # noqa: BLE001 - a heartbeat must never kill the step
                        pass
                done.wait(0.5)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            raise StepError(f"{name}: timeout after {timeout}s")
        finally:
            done.set()
            th.join()
    secs = time.monotonic() - t0
    shutil.rmtree(chpath, ignore_errors=True)
    if proc.returncode != 0:
        raise StepError(f"{name}: clickhouse exited {proc.returncode}\n{_errors(errpath)}")
    mean_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    return {"seconds": round(secs, 3), "peak_rss_bytes": peak[0] * 1024, "mean_cpu_pct": round(mean_cpu, 1),
            "last_progress": last_progress(errpath), "stdout": stdout}


def run_many(jobs: list[tuple[str, str]], workdir: pathlib.Path, parallel: int, max_threads: int,
             on_done=None, heartbeat=None, heartbeat_s: float = 120) -> dict:
    """Run independent scripts (name, sql) `parallel` at a time, each limited to
    `max_threads`; on_done(name, stats) after each finishes. Returns {name: stats}."""
    results: dict = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(run_script, sql, workdir, name, None, heartbeat, heartbeat_s, max_threads): name for name, sql in jobs}
        for fut in cf.as_completed(futs):
            name = futs[fut]
            results[name] = fut.result()
            if on_done:
                on_done(name, results[name])
    return results


def query(sql: str, workdir: pathlib.Path, name: str = "q") -> str:
    return run_script(sql, workdir, name)["stdout"]
