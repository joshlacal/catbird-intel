"""Object-store access for S12m on top of graph/vmpass/store.py (local dir for rehearsals, GCS on the VM), plus
listing, and a guard that this job writes only under its own vectors/ and intermediates/ prefixes."""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "vmpass"))   # appended: vmpass/projection.py must not shadow ours
from store import GcsStore, LocalStore, open_store  # noqa: E402,F401

PLAN30 = "gs://catbird-436600-plan-30d"
PLAN60 = "gs://catbird-436600-plan-60d"


def list_keys(store, prefix="", suffix=""):
    """Keys (relative to the store root) that start with `prefix` (a directory part plus an optional name
    prefix, e.g. "_ledger/u-") and end with `suffix`, sorted."""
    d = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
    if isinstance(store, LocalStore):
        root = store.root / d
        keys = [str(p.relative_to(store.root)) for p in root.rglob("*") if p.is_file()] if root.exists() else []
    else:
        store.ops["list"] += 1
        url = f"{store.url}/{d}" if d else store.url
        r = subprocess.run(["gcloud", "storage", "ls", "-r", url], capture_output=True, text=True)
        if r.returncode != 0:
            if "matched no objects" in r.stderr:
                return []
            raise RuntimeError(f"gcloud storage ls failed: {r.stderr[-400:]}")
        keys = [ln[len(store.url) + 1:] for ln in r.stdout.splitlines()
                if ln.startswith(store.url + "/") and not ln.endswith(":") and not ln.endswith("/")]
    return sorted(k for k in keys if k.startswith(prefix) and k.endswith(suffix))


def allowed_write(url: str) -> bool:
    """The job may write only to plan-60d/vectors/s12mm/ and plan-30d/intermediates/ (or a local rehearsal dir)."""
    if not url.startswith("gs://"):
        return True
    return url.startswith(f"{PLAN60}/vectors/s12mm/") or url.startswith(f"{PLAN30}/intermediates/")


def write_text(store, key, text, tmpdir):
    """write_text that carries the store's customTime on GCS (plan-60d's lifecycle keys on daysSinceCustomTime,
    so an object written without it would never expire). Goes through a small local temp file."""
    if isinstance(store, LocalStore) or not getattr(store, "custom_time", None):
        if not isinstance(store, LocalStore) and store.url.startswith(PLAN60):
            raise RuntimeError("refusing a plan-60d write without customTime")
        return store.write_text(key, text)
    p = pathlib.Path(tmpdir) / ("_w_" + key.replace("/", "_"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    store.put(p, key)
    p.unlink()
