#!/usr/bin/env python3
"""Populate and synchronize CatbirdCommunity/catbird-lake/ on Google Drive.

Uploads and verifies:
1. hot-export/day=YYYY-MM-DD/ from netcup:/var/lib/catbird-intel/export/
2. graph-parquet/ from M4 Max local decoded archive batches (/Users/joshlacalamito/catbird-lake-local/archive-graph/events/)
3. training-banks/: only explicitly named, validated v1 unlabeled banks; run
   export_training_banks_v1.py first. Regex ingest scores are baseline columns,
   never teacher targets. V1 social pairs require their graph provenance manifest.
4. models/: no implicit upload. Invalidated v0/stub artifacts are quarantined;
   publish real model bundles separately after measured validation.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import pathlib
import subprocess
import threading
import time
import urllib.parse
import urllib.request

GWS = "/usr/local/bin/gws"
LAKE_FOLDERS = {
    "root": "19n0W_MnOGeaxisZqsO5_wjmSq9rp_Bvy",
    "hot-export": "13tCh7MmqAnA7iUQN1Vd9mg2HMNrGRdRb",
    "graph-parquet": "1_HkePwouXp1NUFc6O5fSed0tNcRE4ILf",
    "training-banks": "1oY4s8S_Og34IX7PDbgHcV7L1wuElpP4n",
    "models": "12WuLSAT8xYg3dCHicdt-mH9llTB-kkh_",
    "tier3-vectors": "1DmISwmDkoIAAN8RDIGH6Ryn8v3BZEWLr",
}


class DriveUploader:
    def __init__(self) -> None:
        r = subprocess.run([GWS, "auth", "export", "--unmasked"], capture_output=True, text=True, check=True)
        cfg = json.loads(r.stdout[r.stdout.find("{") :])
        self.client_id = cfg["client_id"]
        self.client_secret = cfg["client_secret"]
        self.refresh_token = cfg["refresh_token"]
        self._token = ""
        self._exp = 0.0
        self._lock = threading.Lock()
        self._folder_cache: dict[str, str] = {}

    def token(self) -> str:
        with self._lock:
            now = time.time()
            if self._token and now < self._exp - 60:
                return self._token
            body = urllib.parse.urlencode(
                {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                }
            ).encode()
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            self._token = data["access_token"]
            self._exp = now + float(data.get("expires_in", 3600))
            return self._token

    def ensure_subfolder(self, parent_id: str, name: str) -> str:
        key = f"{parent_id}/{name}"
        with self._lock:
            if key in self._folder_cache:
                return self._folder_cache[key]
        q = (
            f"'{parent_id}' in parents and name = '{name}' "
            f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        params = urllib.parse.urlencode({"q": q, "fields": "files(id,name)"})
        req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{params}",
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            files = json.loads(resp.read().decode()).get("files", [])
        if len(files) > 1:
            raise RuntimeError(f"Ambiguous Drive folder: {name}")
        if files:
            fid = files[0]["id"]
            with self._lock:
                self._folder_cache[key] = fid
            return fid
        meta = json.dumps(
            {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
        ).encode()
        req = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files?fields=id,name",
            data=meta,
            headers={
                "Authorization": f"Bearer {self.token()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            fid = json.loads(resp.read().decode())["id"]
        with self._lock:
            self._folder_cache[key] = fid
        return fid

    def list_existing(self, parent_id: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        page_token = None
        while True:
            q = f"'{parent_id}' in parents and trashed = false"
            params_dict = {
                "q": q,
                "fields": "nextPageToken,files(id,name,size,md5Checksum)",
                "pageSize": "1000",
            }
            if page_token:
                params_dict["pageToken"] = page_token
            params = urllib.parse.urlencode(params_dict)
            req = urllib.request.Request(
                f"https://www.googleapis.com/drive/v3/files?{params}",
                headers={"Authorization": f"Bearer {self.token()}"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            for f in data.get("files", []):
                if f["name"] in out:
                    raise RuntimeError(f"Duplicate Drive name in {parent_id}: {f['name']}")
                out[f["name"]] = f
            page_token = data.get("nextPageToken")
            if not page_token:
                return out

    def upload_file(
        self,
        local_path: pathlib.Path,
        parent_id: str,
        existing_map: dict[str, dict] | None = None,
        remote_name: str | None = None,
    ) -> dict:
        name = remote_name or local_path.name
        size = local_path.stat().st_size
        if size == 0:
            raise RuntimeError(f"Refusing to upload 0-byte file {local_path}")
        md5 = hashlib.md5()
        with local_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                md5.update(chunk)
        expected_md5 = md5.hexdigest()

        if existing_map is None:
            existing_map = self.list_existing(parent_id)
        if existing_map and name in existing_map:
            ex = existing_map[name]
            if int(ex.get("size") or -1) == size and ex.get("md5Checksum") == expected_md5:
                return {"status": "cached", "id": ex["id"], "name": name, "bytes": size}
            raise RuntimeError(f"Existing Drive bytes differ for {name}; use a new versioned name")

        mime = "application/json" if name.endswith(".json") else "application/octet-stream"
        meta = json.dumps({"name": name, "parents": [parent_id]}).encode()
        init_req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum",
            data=meta,
            headers={
                "Authorization": f"Bearer {self.token()}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": mime,
                "X-Upload-Content-Length": str(size),
            },
            method="POST",
        )
        with urllib.request.urlopen(init_req, timeout=30) as resp:
            upload_url = resp.headers["Location"]

        with local_path.open("rb") as fh:
            payload = fh.read()
        put_req = urllib.request.Request(
            upload_url,
            data=payload,
            headers={"Content-Length": str(size), "Content-Type": mime},
            method="PUT",
        )
        with urllib.request.urlopen(put_req, timeout=300) as resp:
            uploaded = json.loads(resp.read().decode())
        if int(uploaded.get("size") or -1) != size or uploaded.get("md5Checksum") != expected_md5:
            raise RuntimeError(f"Checksum mismatch uploading {name}")
        existing_map[name] = uploaded
        return {"status": "uploaded", "id": uploaded["id"], "name": name, "bytes": size}


def validate_health_bank(path: pathlib.Path) -> None:
    """Require the sealed export receipt and actual reserved-author isolation."""
    import pyarrow.parquet as pq
    receipt = json.loads(path.with_suffix(".manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise RuntimeError("Unlabeled bank differs from its export receipt")
    schema = pq.read_schema(path)
    if any(name.startswith("teacher_") for name in schema.names):
        raise RuntimeError("Unlabeled bank contains forbidden teacher aliases")
    required = {"regex_baseline_" + name for name in
                ("politicalness", "heat", "engagement_bait", "spam")}
    if not required.issubset(schema.names):
        raise RuntimeError("Unlabeled bank is missing explicit regex baseline columns")
    data = pq.read_table(path, columns=["author_did", "uri", "author_split_bucket"])
    if data.num_rows != receipt["rows"] or data.num_rows == 0:
        raise RuntimeError("Unlabeled bank row count differs from its receipt")
    if 0 in set(data["author_split_bucket"].to_pylist()):
        raise RuntimeError("Reserved holdout author bucket leaked into bank")
    if len(set(data["uri"].to_pylist())) != data.num_rows:
        raise RuntimeError("Duplicate post URI in bank")
    eval_dir = pathlib.Path(__file__).resolve().parents[1] / "eval"
    manifest = json.loads((eval_dir / "phase3_v1_export_manifest.json").read_text())
    holdout = eval_dir / "phase3_holdout_600_unlabeled.jsonl"
    if hashlib.sha256(holdout.read_bytes()).hexdigest() != manifest["holdout"]["sha256"]:
        raise RuntimeError("Holdout differs from its sealed receipt")
    if manifest["bank"]["sha256"] != receipt["sha256"]:
        raise RuntimeError("Bank is not the holdout's audited training generation")
    with holdout.open() as fh:
        held_authors = {json.loads(line)["author_did"] for line in fh if line.strip()}
    if held_authors & set(data["author_did"].to_pylist()):
        raise RuntimeError("Holdout authors overlap training")


def main() -> None:
    up = DriveUploader()
    staging = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/drive-sync-staging")
    staging.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "hot_export": [],
        "models": [],
        "training_banks": [],
        "graph_parquet_count": 0,
    }

    # 1. Sync netcup hot-export (/var/lib/catbird-intel/export/2026-10-19)
    hot_local = staging / "hot-export" / "2026-10-19"
    hot_local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", "netcup:/var/lib/catbird-intel/export/2026-10-19/", str(hot_local) + "/"],
        check=True,
    )
    day_folder = up.ensure_subfolder(LAKE_FOLDERS["hot-export"], "day=2026-10-19")
    existing_hot = up.list_existing(day_folder)
    for f in sorted(hot_local.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            res = up.upload_file(f, day_folder, existing_hot)
            summary["hot_export"].append(res)
    print(f"[1/4] Synced {len(summary['hot_export'])} hot-export files for day=2026-10-19", flush=True)

    # 2. Never rediscover old banks via a wildcard or regenerate v0 labels.
    tb_local = staging / "training-banks"
    p3_file = tb_local / "phase3_health_training_150k_v1_unlabeled.parquet"
    if not p3_file.exists():
        raise RuntimeError("Run ops/export_training_banks_v1.py before syncing clean banks")
    validate_health_bank(p3_file)
    existing_tb = up.list_existing(LAKE_FOLDERS["training-banks"])
    summary["training_banks"].append(
        up.upload_file(p3_file, LAKE_FOLDERS["training-banks"], existing_tb))
    print("[2/4] Synced validated v1 unlabeled health bank; v0 social bank excluded", flush=True)

    # 3. Explicitly empty until a real, measured model bundle is ready. In
    # particular do not glob an in-progress graph output directory or upload
    # the old Phase 1, Phase 5, Phase 6 reports or hashed/regex student artifacts.
    model_candidates: list[pathlib.Path] = []
    print("[3/4] Model upload deferred to explicit validated release bundle", flush=True)

    # 4. Sync local decoded graph Parquet batches (/Users/joshlacalamito/catbird-lake-local/archive-graph/events/events-*.parquet)
    local_graph_dir = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/archive-graph/events")
    existing_gp = up.list_existing(LAKE_FOLDERS["graph-parquet"])
    gp_files = sorted(local_graph_dir.glob("events-*.parquet"))
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda gp: up.upload_file(gp, LAKE_FOLDERS["graph-parquet"], existing_gp), gp_files))
    summary["graph_parquet_count"] = len(gp_files)
    print(f"[4/4] Synced {len(gp_files)} decoded events-*.parquet files to catbird-lake/graph-parquet/", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
