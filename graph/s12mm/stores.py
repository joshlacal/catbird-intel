"""Object-store access for S12m: one small interface, two backends.

Every job path (job.py run/bench) talks to a store through the same six calls:

    put(local, key)   get(key, local)   exists(key)   read_text(key)   write_text(key, text)   list(prefix, suffix)

Keys are relative, '/'-separated paths. Backends:
  * LocalDirStore: a directory. Used on Colab (the run's output root is /content/s12mm/out/<batch>, filled by the
    job and taken off the VM by `colab download`), in every Mac rehearsal and in the tests. Writes are atomic
    (temp file + rename in the same directory), so a reader never sees half a ledger marker or unit file, and a
    key that would escape the root ('..', absolute) is refused.
  * GcsStore (graph/vmpass/store.py, imported only when a gs:// URL is opened): the GCE path; listing goes
    through `gcloud storage ls`. plan-60d writes must carry customTime (write_text below).
`allowed_write` keeps a GCE job inside its own vectors/ and intermediates/ prefixes; local paths are always allowed.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PLAN30 = "gs://catbird-436600-plan-30d"
PLAN60 = "gs://catbird-436600-plan-60d"


def _vmpass():
    if str(HERE.parent / "vmpass") not in sys.path:
        sys.path.append(str(HERE.parent / "vmpass"))   # appended: vmpass/projection.py must not shadow ours
    import store as vm_store
    return vm_store


class LocalDirStore:
    """A directory as an object store (see the module docstring)."""

    def __init__(self, root, custom_time=None):
        self.root = pathlib.Path(root)
        self.url = str(self.root)
        self.custom_time = None          # a directory has no lifecycle; accepted for open_store symmetry
        self.ops = {"put": 0, "get": 0, "list": 0}

    def _path(self, key: str) -> pathlib.Path:
        k = pathlib.PurePosixPath(key)
        if not key or k.is_absolute() or ".." in k.parts or "\\" in key:
            raise ValueError(f"refusing store key {key!r}")
        return self.root.joinpath(*k.parts)

    def _atomic(self, dst: pathlib.Path, write):
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=dst.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                write(fh)
            os.replace(tmp, dst)
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise

    def put(self, local, key: str) -> None:
        with open(local, "rb") as src:
            self._atomic(self._path(key), lambda fh: shutil.copyfileobj(src, fh, 1 << 22))
        self.ops["put"] += 1

    def get(self, key: str, local) -> None:
        local = pathlib.Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), local)
        self.ops["get"] += 1

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_text(self, key: str) -> str:
        self.ops["get"] += 1
        return self._path(key).read_text()

    def write_text(self, key: str, text: str) -> None:
        self._atomic(self._path(key), lambda fh: fh.write(text.encode()))
        self.ops["put"] += 1

    def list(self, prefix: str = "", suffix: str = "") -> list[str]:
        """Keys starting with `prefix` (a directory part plus an optional name prefix, e.g. "_ledger/u-") and
        ending with `suffix`, sorted. Temp files of in-flight writes are never listed."""
        self.ops["list"] += 1
        d = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
        base = self._path(d) if d else self.root
        if not base.exists():
            return []
        keys = [p.relative_to(self.root).as_posix() for p in base.rglob("*")
                if p.is_file() and not p.name.startswith(".tmp-")]
        return sorted(k for k in keys if k.startswith(prefix) and k.endswith(suffix))


LocalStore = LocalDirStore   # the name the tests and older call sites use


def __getattr__(name):
    if name == "GcsStore":
        return _gcs_class()
    raise AttributeError(name)


_GCS = None


def _gcs_class():
    """vmpass's GcsStore plus a `list` method, built on first use so Colab never needs vmpass or gcloud."""
    global _GCS
    if _GCS is None:
        base = _vmpass().GcsStore

        class GcsStore(base):
            def list(self, prefix: str = "", suffix: str = "") -> list[str]:
                self.ops["list"] += 1
                d = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
                url = f"{self.url}/{d}" if d else self.url
                r = subprocess.run(["gcloud", "storage", "ls", "-r", url], capture_output=True, text=True)
                if r.returncode != 0:
                    if "matched no objects" in r.stderr:
                        return []
                    raise RuntimeError(f"gcloud storage ls failed: {r.stderr[-400:]}")
                keys = [ln[len(self.url) + 1:] for ln in r.stdout.splitlines()
                        if ln.startswith(self.url + "/") and not ln.endswith(":") and not ln.endswith("/")]
                return sorted(k for k in keys if k.startswith(prefix) and k.endswith(suffix))
        _GCS = GcsStore
    return _GCS


def open_store(url: str, custom_time: str | None = None):
    return _gcs_class()(url, custom_time) if url.startswith("gs://") else LocalDirStore(url)


def list_keys(store, prefix="", suffix=""):
    return store.list(prefix, suffix)


def allowed_write(url: str) -> bool:
    """A GCE job may write only to plan-60d/vectors/s12mm/ and plan-30d/intermediates/; a local directory (Colab,
    rehearsal) is always allowed."""
    if not url.startswith("gs://"):
        return True
    return url.startswith(f"{PLAN60}/vectors/s12mm/") or url.startswith(f"{PLAN30}/intermediates/")


def write_text(store, key, text, tmpdir):
    """write_text that carries the store's customTime on GCS (plan-60d's lifecycle keys on daysSinceCustomTime,
    so an object written without it would never expire). Goes through a small local temp file."""
    if isinstance(store, LocalDirStore) or not getattr(store, "custom_time", None):
        if not isinstance(store, LocalDirStore) and store.url.startswith(PLAN60):
            raise RuntimeError("refusing a plan-60d write without customTime")
        return store.write_text(key, text)
    p = pathlib.Path(tmpdir) / ("_w_" + key.replace("/", "_"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    store.put(p, key)
    p.unlink()
