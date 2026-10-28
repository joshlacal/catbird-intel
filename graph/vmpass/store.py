"""Object stores for pass outputs: a local directory (tests, dry runs) or GCS.

Keys are bucket-relative paths. The pass writes every finished step's files and
a ledger marker; a restarted VM reads the markers to skip finished steps and
downloads only the files a later step needs.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class LocalStore:
    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.url = str(self.root)
        self.ops = {"put": 0, "get": 0, "list": 0}

    def put(self, local: pathlib.Path, key: str) -> None:
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)
        self.ops["put"] += 1

    def put_files(self, pairs: list[tuple[pathlib.Path, str]]) -> None:
        for local, key in pairs:
            self.put(local, key)

    def get_files(self, pairs: list[tuple[str, pathlib.Path]]) -> None:
        for key, local in pairs:
            self.get(key, local)

    def get(self, key: str, local: pathlib.Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / key, local)
        self.ops["get"] += 1

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def read_text(self, key: str) -> str:
        self.ops["get"] += 1
        return (self.root / key).read_text()

    def write_text(self, key: str, text: str) -> None:
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        self.ops["put"] += 1


class GcsStore:
    """gs://bucket[/prefix] through the gcloud CLI (present on every GCE image)."""

    def __init__(self, url: str, custom_time: str | None = None):
        self.url = url.rstrip("/")
        self.ops = {"put": 0, "get": 0, "list": 0}
        self.custom_time = custom_time  # plan-60d: lifecycle keys on daysSinceCustomTime >= 60

    def _run(self, *args: str, input_text: str | None = None) -> str:
        r = subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True, input=input_text)
        if r.returncode != 0:
            raise RuntimeError(f"gcloud storage {' '.join(args[:2])} failed: {r.stderr[-800:]}")
        return r.stdout

    def put(self, local: pathlib.Path, key: str) -> None:
        extra = [f"--custom-time={self.custom_time}"] if self.custom_time else []
        self._run("cp", "--quiet", *extra, str(local), f"{self.url}/{key}")
        self.ops["put"] += 1

    @staticmethod
    def _by_dir(pairs):
        groups: dict = {}
        for a, b in pairs:
            ka = str(pathlib.PurePosixPath(str(a)).parent)
            kb = str(pathlib.PurePosixPath(str(b)).parent)
            groups.setdefault((ka, kb), []).append((a, b))
        return groups

    def put_files(self, pairs: list[tuple[pathlib.Path, str]]) -> None:
        """One `gcloud storage cp` per (local dir, remote dir) instead of one per file."""
        extra = [f"--custom-time={self.custom_time}"] if self.custom_time else []
        for (_ld, rd), grp in self._by_dir(pairs).items():
            for i in range(0, len(grp), 200):
                chunk = grp[i:i + 200]
                if all(pathlib.PurePosixPath(k).name == pathlib.Path(lp).name for lp, k in chunk):
                    self._run("cp", "--quiet", *extra, *[str(lp) for lp, _ in chunk], f"{self.url}/{rd}/")
                else:
                    for lp, k in chunk:
                        self._run("cp", "--quiet", *extra, str(lp), f"{self.url}/{k}")
                self.ops["put"] += len(chunk)

    def get_files(self, pairs: list[tuple[str, pathlib.Path]]) -> None:
        for (_rd, ld), grp in self._by_dir(pairs).items():
            pathlib.Path(ld).mkdir(parents=True, exist_ok=True)
            for i in range(0, len(grp), 200):
                chunk = grp[i:i + 200]
                if all(pathlib.PurePosixPath(k).name == pathlib.Path(lp).name for k, lp in chunk):
                    self._run("cp", "--quiet", *[f"{self.url}/{k}" for k, _ in chunk], f"{ld}/")
                else:
                    for k, lp in chunk:
                        self._run("cp", "--quiet", f"{self.url}/{k}", str(lp))
                self.ops["get"] += len(chunk)

    def get(self, key: str, local: pathlib.Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._run("cp", "--quiet", f"{self.url}/{key}", str(local))
        self.ops["get"] += 1

    def exists(self, key: str) -> bool:
        self.ops["list"] += 1
        r = subprocess.run(["gcloud", "storage", "ls", f"{self.url}/{key}"], capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())

    def read_text(self, key: str) -> str:
        self.ops["get"] += 1
        return self._run("cat", f"{self.url}/{key}")

    def write_text(self, key: str, text: str) -> None:
        self._run("cp", "-", f"{self.url}/{key}", input_text=text)
        self.ops["put"] += 1


def open_store(url: str, custom_time: str | None = None):
    return GcsStore(url, custom_time) if url.startswith("gs://") else LocalStore(url)


class Ledger:
    """Done-markers for finished steps, kept in the 30-day store under
    intermediates/<run>/_ledger/. A marker lists the step's files (relative to the
    local work dir) with sizes and sha256, so a fresh VM can fetch them."""

    def __init__(self, store, run_id: str):
        self.store = store
        self.prefix = f"intermediates/{run_id}"

    def key(self, step: str) -> str:
        return f"{self.prefix}/_ledger/{step}.json"

    def done(self, step: str) -> dict | None:
        k = self.key(step)
        if not self.store.exists(k):
            return None
        return json.loads(self.store.read_text(k))

    def mark(self, step: str, record: dict) -> None:
        self.store.write_text(self.key(step), json.dumps(record, indent=1, sort_keys=True))
