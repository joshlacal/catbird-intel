#!/usr/bin/env python3
"""Populate and synchronize CatbirdCommunity/catbird-lake/ on Google Drive.

Uploads and verifies:
1. hot-export/day=YYYY-MM-DD/ from netcup:/var/lib/catbird-intel/export/
2. graph-parquet/ from M4 Max local decoded archive batches (/Users/joshlacalamito/catbird-lake-local/archive-graph/events/)
3. training-banks/ exported from netcup ClickHouse (priority=10, max_threads=2):
   - phase3_health_training_150k.parquet (150k stratified Bluesky posts with health & community signals)
   - phase5_social_encoder_pairs_500k.parquet (500k co-engaged / community-labeled posts from feed.posts + catbird.known_for)
4. models/ (KnownFor Parquet, communities Parquet, top-500 labels, socially_trained_student_256d_int8.json, catbird_jev_int8.onnx, evaluation reports)
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
                if int(f.get("size") or 0) > 0:
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

        if existing_map and name in existing_map:
            ex = existing_map[name]
            if int(ex.get("size") or -1) == size and ex.get("md5Checksum") == expected_md5:
                return {"status": "cached", "id": ex["id"], "name": name, "bytes": size}

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
        return {"status": "uploaded", "id": uploaded["id"], "name": name, "bytes": size}


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

    # 2. Export Phase 3 (150k posts) & Phase 5 (500k co-engagement / community post bank) from netcup ClickHouse via stdin
    tb_local = staging / "training-banks"
    tb_local.mkdir(parents=True, exist_ok=True)
    p3_file = tb_local / "phase3_health_training_150k.parquet"
    if not p3_file.exists() or p3_file.stat().st_size < 1_000_000:
        q_p3 = """
        SELECT
            concat('at://', p.did, '/app.bsky.feed.post/', p.rkey) AS uri,
            p.did AS author_did,
            p.text AS text,
            p.langs AS langs,
            p.reply_root != '' AS is_reply,
            p.quote_uri != '' AS is_quote,
            p.politicalness AS teacher_politicalness,
            p.heat AS teacher_heat,
            p.engagement_bait AS teacher_engagement_bait,
            p.spam AS teacher_spam,
            kf.top_comm AS author_primary_community,
            c.label AS community_label,
            c.seed_subculture AS seed_subculture
        FROM catbird.posts AS p
        LEFT JOIN (
            SELECT did, argMax(community, score) AS top_comm
            FROM catbird.known_for
            GROUP BY did
        ) AS kf ON p.did = kf.did
        LEFT JOIN catbird.communities AS c ON kf.top_comm = c.community
        WHERE length(p.text) >= 20 AND p.is_deleted = 0
        ORDER BY cityHash64(p.did, p.rkey)
        LIMIT 150000
        FORMAT Parquet
        SETTINGS output_format_parquet_compression_method = 'zstd', max_threads = 2, priority = 10
        """
        with p3_file.open("wb") as fh:
            subprocess.run(["ssh", "netcup", "clickhouse-client"], input=q_p3.encode(), stdout=fh, check=True)

    p5_file = tb_local / "phase5_social_encoder_pairs_500k.parquet"
    if not p5_file.exists() or p5_file.stat().st_size < 5_000_000:
        q_p5 = """
        SELECT
            fp.uri AS uri,
            fp.author AS author_did,
            fp.created AS created_at,
            fp.text AS text,
            kf.top_comm AS community_id,
            kf.max_score AS community_weight,
            c.label AS community_label,
            c.scene_id AS scene_id,
            c.domain_id AS domain_id,
            c.seed_subculture AS seed_subculture
        FROM feed.posts AS fp
        INNER JOIN (
            SELECT did, argMax(community, score) AS top_comm, max(score) AS max_score
            FROM catbird.known_for
            GROUP BY did
        ) AS kf ON fp.author = kf.did
        LEFT JOIN catbird.communities AS c ON kf.top_comm = c.community
        WHERE length(fp.text) BETWEEN 25 AND 600
        ORDER BY cityHash64(fp.uri)
        LIMIT 500000
        FORMAT Parquet
        SETTINGS output_format_parquet_compression_method = 'zstd', max_threads = 2, priority = 10
        """
        with p5_file.open("wb") as fh:
            subprocess.run(["ssh", "netcup", "clickhouse-client"], input=q_p5.encode(), stdout=fh, check=True)

    existing_tb = up.list_existing(LAKE_FOLDERS["training-banks"])
    for f in sorted(tb_local.glob("*.parquet")):
        res = up.upload_file(f, LAKE_FOLDERS["training-banks"], existing_tb)
        summary["training_banks"].append(res)
    print(f"[2/4] Synced {len(summary['training_banks'])} training bank Parquet files", flush=True)

    # 3. Sync models & evaluation artifacts to catbird-lake/models/
    repo_root = pathlib.Path("/Users/joshlacalamito/Developer/Catbird+Petrel")
    graph_out = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/graph-build/out")
    model_candidates = [
        *sorted(graph_out.glob("*")),
        repo_root / "catbird-intel/graph/out/phase1_eval_report.json",
        repo_root / "catbird-intel/colab/out/phase5_benchmark_report.json",
        repo_root / "catbird-intel/colab/out/socially_trained_student_256d_int8.json",
        repo_root / "catbird-intel/ops/out/phase6_eval_report.json",
        repo_root / "scripts/onboarding/models/catbird_jev_int8.onnx",
        repo_root / "scripts/onboarding/catbird_seed_dataset.json",
    ]
    existing_models = up.list_existing(LAKE_FOLDERS["models"])
    seen_names: set[str] = set()
    for mc in model_candidates:
        if mc.is_file() and mc.name not in seen_names:
            seen_names.add(mc.name)
            res = up.upload_file(mc, LAKE_FOLDERS["models"], existing_models)
            summary["models"].append(res)
    print(f"[3/4] Synced {len(summary['models'])} model & eval artifacts", flush=True)

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
