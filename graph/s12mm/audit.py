"""No-image-bytes-on-disk audit for S12m (D9 item 4: embed in memory only).

`start_marker` records a marker file, disk usage and swap state right before any fetch. `finish` lists every
regular file created or modified after the marker (on the VM: the whole root filesystem, `find / -xdev -newer`,
excluding /proc, /sys, /dev, /run; on the Mac rehearsal: the work dir and $TMPDIR), checks each one's header
for JPEG/PNG/GIF/WebP, scans non-output files' full content for those signatures, and validates every output
parquet structurally (exact schema, 512-byte unit-norm fp16 vectors). It also requires that no swap device is
active (process memory holding image bytes could otherwise be paged to disk) and reports the disk-usage diff.
"""
from __future__ import annotations

import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time

import mmlib as L

EXCLUDE_ROOTS = ("/proc", "/sys", "/dev", "/run")


def swap_active() -> bool | None:
    try:
        lines = pathlib.Path("/proc/swaps").read_text().splitlines()
        return len(lines) > 1
    except OSError:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
            return "used = 0.00M" not in out   # macOS always has dynamic swap; report whether it is used
        return None


def start_marker(work):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    m = work / ".fetch_marker"
    m.write_text(str(time.time()))
    time.sleep(1.1)   # mtime resolution: anything written after this is strictly newer
    return {"path": str(m), "t": m.stat().st_mtime, "du_used": shutil.disk_usage(work).used,
            "swap_active_at_start": swap_active()}


def changed_files(work, marker):
    t = marker["t"]
    if platform.system() == "Linux" and os.geteuid() == 0:
        prune = []
        for r in EXCLUDE_ROOTS:
            prune += (["-o"] if prune else []) + ["-path", r]
        cmd = ["find", "/", "-xdev", "(", *prune, ")", "-prune", "-o", "-type", "f", "-newer", marker["path"], "-print"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        return [p for p in out.splitlines() if p]
    roots = {pathlib.Path(work), pathlib.Path(tempfile.gettempdir())}
    res = []
    for r in roots:
        for p in r.rglob("*"):
            try:
                if p.is_file() and p.stat().st_mtime > t:
                    res.append(str(p))
            except OSError:
                pass
    return res


def finish(work, marker, extra_outputs=()):
    files = [f for f in changed_files(work, marker) if f != marker["path"]]
    outputs = [f for f in files if f.endswith(".parquet") and "/out/" in f] + list(extra_outputs)
    tables = [L.audit_output_table(f) for f in outputs if pathlib.Path(f).exists()]
    # Full-content scans cover files this job could have written (work dir, temp dirs, home, our logs); system
    # files elsewhere (journal, package caches) get the header check only, since random binary content matches
    # a 3-byte JPEG marker by chance. A whole image file anywhere is caught by its header.
    full = tuple(str(pathlib.Path(x).resolve()) for x in (work, tempfile.gettempdir(), "/tmp", "/root", "/home",
                                                           "/var/log/s12mm.log"))
    inputs = tuple(str((pathlib.Path(work) / d).resolve()) + "/" for d in ("in", "sidecar", "bench_in"))
    header_only = [f for f in files if not str(pathlib.Path(f).resolve()).startswith(full)
                   or str(pathlib.Path(f).resolve()).startswith(inputs)]   # downloaded text inputs: header check
    scan = L.audit_files(files, structured=frozenset(outputs) | frozenset(header_only))
    scan["header_only_files"] = len(header_only)
    swap = swap_active()
    linux = platform.system() == "Linux"
    rep = {"scope": "root filesystem" if linux and os.geteuid() == 0 else "work dir + TMPDIR",
           "files_changed": len(files), "listing_head": sorted(files)[:200], "scan": scan,
           "output_tables": tables, "swap_active": swap, "swap_active_at_start": marker["swap_active_at_start"],
           "du_used_delta_bytes": shutil.disk_usage(work).used - marker["du_used"]}
    rep["ok"] = scan["ok"] and all(t["ok"] for t in tables) and (swap is not True or not linux)
    return rep
