#!/usr/bin/env python3
"""Synthetic, offline tests for full retained-cache remote readback."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.parse

from gcs_drive_drain import Conflict, FOLDER
from gcs_drive_retained_overflow import BUCKET, DRIVE_ROOT
from gcs_drive_retained_cache_remote import (
    LiveReadOnlyVerifier, SafeDriveHTTP, SafeOAuthRefresh,
)


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "md5": hashlib.md5(data, usedforsecurity=False).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeBlob:
    def __init__(self, body, generation, metadata=None):
        self.body = body
        self.generation = generation
        self.size = len(body)
        self.metadata = metadata or {}
        self.md5_hash = base64.b64encode(bytes.fromhex(digest(body)["md5"])).decode()
        self.calls = []

    def download_as_bytes(self, **kwargs):
        self.calls.append(("mapping", kwargs))
        if kwargs.get("if_generation_match") != self.generation or kwargs.get("retry") is not None:
            raise AssertionError("mapping read was not generation pinned with retries disabled")
        return self.body

    def open(self, mode, **kwargs):
        self.calls.append(("media", kwargs))
        if (mode != "rb" or kwargs.get("if_generation_match") != self.generation
                or kwargs.get("retry") is not None):
            raise AssertionError("media read was not generation pinned with retries disabled")
        return io.BytesIO(self.body)


class FakeBucket:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def get_blob(self, name, *, generation, retry, timeout):
        self.calls.append(("metadata", name, generation, retry, timeout))
        return self.objects.get((name, generation))

    def blob(self, name, *, generation):
        self.calls.append(("body", name, generation))
        return self.objects[(name, generation)]


class FakeAuth:
    def __init__(self):
        self.calls = 0

    def token(self):
        self.calls += 1
        return "synthetic_token"


class FakeDriveHTTP:
    def __init__(self, clock, objects):
        self.clock, self.objects = clock, objects
        self.calls = []

    def open(self, url, token):
        if token != "synthetic_token":
            raise AssertionError("wrong synthetic token")
        self.calls.append((url, self.clock.now))
        key = urllib.parse.urlsplit(url).path
        if urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("alt") == ["media"]:
            key += ":media"
        return self.objects[key]()


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.payload = b"synthetic-payload\x00" * 73
        self.receipt = b'{"synthetic":true}'
        self.mapping_bytes = b'{"synthetic":"mapping"}'
        self.folder_id = "folderSynthetic1234"
        self.ids = {"folder": self.folder_id, "payload": "payloadSynthetic1234",
                    "receipt": "receiptSynthetic1234"}
        self.manifest = {"account_id": "permissionSynthetic1234",
                         "folder_name": "catbird-overflow-synthetic",
                         "index_object": "indexes/drive-retained-overflow/v1/synthetic.json"}
        self.local = {"eligible_for_approval": False,
                      "exclude_reason": "remote_readback_unavailable",
                      "account_id": self.manifest["account_id"],
                      "mapping_bucket": BUCKET,
                      "mapping_object": self.manifest["index_object"],
                      "mapping_sha256": digest(self.mapping_bytes)["sha256"],
                      "drive_root": DRIVE_ROOT, "drive_ids": self.ids, "source": {}, "stage": {}}
        for kind, body, generation in (("payload", self.payload, 11),
                                       ("receipt", self.receipt, 12)):
            name = f"sources/synthetic/{kind}"
            self.manifest[kind] = {"object": name, "generation": generation,
                                   "bytes": len(body), "sha256": digest(body)["sha256"]}
            self.local["source"][kind] = dict(self.manifest[kind])
            suffix = "" if kind == "payload" else ".receipt.json"
            self.local["stage"][kind] = {"path": "/synthetic/seg_0000000001.jss" + suffix,
                                          **digest(body)}
        self.clock = FakeClock()
        self.auth = FakeAuth()
        mapping = FakeBlob(self.mapping_bytes, 13, {
            "kind": "drive_retained_overflow_index", "sha256": digest(self.mapping_bytes)["sha256"]})
        objects = {(self.manifest["index_object"], 13): mapping}
        for kind, body in (("payload", self.payload), ("receipt", self.receipt)):
            ref = self.manifest[kind]
            objects[(ref["object"], ref["generation"])] = FakeBlob(
                body, ref["generation"], {"sha256": ref["sha256"]})
        self.bucket = FakeBucket(objects)
        responses = {
            "/drive/v3/about": lambda: Response(json.dumps({"user": {
                "permissionId": self.manifest["account_id"]}}).encode()),
            "/drive/v3/files/" + DRIVE_ROOT: lambda: Response(json.dumps({
                "id": DRIVE_ROOT, "mimeType": FOLDER, "trashed": False}).encode()),
            "/drive/v3/files/" + self.folder_id: lambda: Response(json.dumps({
                "id": self.folder_id, "name": self.manifest["folder_name"],
                "mimeType": FOLDER, "parents": [DRIVE_ROOT], "trashed": False}).encode()),
        }
        for kind, body in (("payload", self.payload), ("receipt", self.receipt)):
            file_id = self.ids[kind]
            meta = {"id": file_id, "name": Path(self.local["stage"][kind]["path"]).name,
                    "parents": [self.folder_id], "trashed": False,
                    "mimeType": "application/octet-stream" if kind == "payload" else "application/json",
                    "size": str(len(body)), "md5Checksum": digest(body)["md5"]}
            responses["/drive/v3/files/" + file_id] = lambda meta=meta: Response(json.dumps(meta).encode())
            responses["/drive/v3/files/" + file_id + ":media"] = (
                lambda body=body: Response(body, headers={"Content-Length": str(len(body))}))
        self.http = FakeDriveHTTP(self.clock, responses)

    def verifier(self):
        return LiveReadOnlyVerifier(
            local=self.local, manifest=self.manifest, index_generation=13,
            oauth_file=Path("/synthetic/oauth.json"), bucket=self.bucket,
            drive_auth=self.auth, drive_http=self.http,
            mono=self.clock.monotonic, sleep=self.clock.sleep)

    def full_readback(self, verifier):
        verifier.verify_account(self.manifest["account_id"])
        self.assertEqual(verifier.mapping(BUCKET, self.manifest["index_object"]),
                         (13, self.mapping_bytes))
        for kind in ("payload", "receipt"):
            ref = self.manifest[kind]
            self.assertEqual(verifier.gcs_digest(ref["object"], ref["generation"]),
                             digest(self.payload if kind == "payload" else self.receipt))
            observed = verifier.drive_digest(self.ids[kind])
            self.assertEqual({key: observed[key] for key in ("bytes", "sha256", "md5")},
                             digest(self.payload if kind == "payload" else self.receipt))
            self.assertIs(observed["media_readback"], True)

    def test_both_full_media_and_pacing(self):
        verifier = self.verifier()
        self.full_readback(verifier)
        # Matches independent real retained metadata: payload octet-stream,
        # receipt application/json, despite the resumable-start header.
        receipt_meta = self.http.objects["/drive/v3/files/" + self.ids["receipt"]]()
        self.assertEqual(json.loads(receipt_meta.read())["mimeType"], "application/json")
        self.assertEqual(self.auth.calls, 1)
        self.assertEqual(verifier.drive_calls, 7)
        self.assertEqual(verifier.gcs_calls, 6)
        self.assertEqual(len(self.http.calls), 7)
        times = [time for _url, time in self.http.calls]
        self.assertTrue(all(b - a >= 0.599999 for a, b in zip(times, times[1:])))
        self.assertEqual(len([url for url, _ in self.http.calls if "alt=media" in url]), 2)

    def test_pause_before_first_http_open_does_not_compress_next_get(self):
        original_open = self.http.open
        def delayed_first_open(url, token):
            if not self.http.calls:
                self.clock.now += 10.0
            return original_open(url, token)
        self.http.open = delayed_first_open
        self.verifier().verify_account(self.manifest["account_id"])
        times = [time for _url, time in self.http.calls]
        self.assertEqual(len(times), 3)
        self.assertTrue(all(b - a >= 0.599999 for a, b in zip(times, times[1:])))

    def test_receipt_media_mismatch_fails(self):
        media = "/drive/v3/files/" + self.ids["receipt"] + ":media"
        self.http.objects[media] = lambda: Response(b"wrong", headers={"Content-Length": str(len(self.receipt))})
        with self.assertRaises(Conflict):
            self.full_readback(self.verifier())

    def test_receipt_gcs_media_mismatch_fails(self):
        ref = self.manifest["receipt"]
        blob = self.bucket.objects[(ref["object"], ref["generation"])]
        blob.body = b"!" + blob.body[1:]
        with self.assertRaises(Conflict):
            self.full_readback(self.verifier())

    def test_drive_early_eof_and_oversize_fail(self):
        key = "/drive/v3/files/" + self.ids["receipt"] + ":media"
        for body in (self.receipt[:-1], self.receipt + b"x"):
            with self.subTest(body=body):
                self.setUp()
                self.http.objects[key] = lambda body=body: Response(body)
                verifier = self.verifier()
                verifier.verify_account(self.manifest["account_id"])
                with self.assertRaises(Conflict):
                    verifier.drive_digest(self.ids["receipt"])

    def test_wrong_generation_fails_before_media(self):
        verifier = self.verifier()
        verifier.verify_account(self.manifest["account_id"])
        with self.assertRaises(Conflict):
            verifier.gcs_digest(self.manifest["payload"]["object"], 999)
        self.assertFalse(any(key == "body" for key, *_ in self.bucket.calls))

    def test_wrong_account_fails_before_mapping(self):
        self.http.objects["/drive/v3/about"] = lambda: Response(json.dumps({
            "user": {"permissionId": "differentPermission123"}}).encode())
        verifier = self.verifier()
        with self.assertRaises(Conflict):
            verifier.verify_account(self.local["account_id"])
        self.assertEqual(self.bucket.calls, [])

    def test_drive_403_429_and_timeout_do_not_retry(self):
        for error in (lambda: Response(b"{}", status=403),
                      lambda: Response(b"{}", status=429),
                      lambda: (_ for _ in ()).throw(TimeoutError("synthetic"))):
            with self.subTest(error=error):
                self.setUp()
                self.http.objects["/drive/v3/about"] = error
                with self.assertRaises(Conflict):
                    self.verifier().verify_account(self.manifest["account_id"])
                self.assertEqual(len(self.http.calls), 1)

    def test_mismatched_drive_mime_and_encoding_fail(self):
        verifier = self.verifier()
        verifier.verify_account(self.manifest["account_id"])
        file_id = self.ids["receipt"]
        key = "/drive/v3/files/" + file_id
        self.http.objects[key] = lambda: Response(json.dumps({
            "id": file_id, "name": Path(self.local["stage"]["receipt"]["path"]).name,
            "parents": [self.folder_id], "trashed": False, "mimeType": "application/octet-stream",
            "size": str(len(self.receipt)), "md5Checksum": digest(self.receipt)["md5"]}).encode())
        with self.assertRaises(Conflict):
            verifier.drive_digest(file_id)
        self.setUp()
        verifier = self.verifier()
        verifier.verify_account(self.manifest["account_id"])
        self.http.objects["/drive/v3/files/" + file_id + ":media"] = (
            lambda: Response(self.receipt, headers={"Content-Encoding": "gzip"}))
        with self.assertRaises(Conflict):
            verifier.drive_digest(file_id)

    def test_fixed_host_and_redirect_fail_without_retry(self):
        http = SafeDriveHTTP()
        with self.assertRaises(Conflict):
            http.open("https://evil.example/drive/v3/about", "synthetic_token")
        class RedirectOpener:
            def open(self, *_args, **_kwargs):
                raise urllib.error.HTTPError("https://www.googleapis.com/drive/v3/about", 302,
                                             "redirect", {}, None)
        http.opener = RedirectOpener()
        with self.assertRaises(Conflict):
            http.open("https://www.googleapis.com/drive/v3/about", "synthetic_token")

    def test_existing_oauth_refresh_does_not_follow_redirect(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oauth.json"
            path.write_text(json.dumps({"client_id": "synthetic_client",
                                        "client_secret": "synthetic_secret",
                                        "refresh_token": "synthetic_refresh"}))
            os.chmod(path, 0o600)
            auth = SafeOAuthRefresh(path)
            class RedirectOpener:
                def open(self, *_args, **_kwargs):
                    raise urllib.error.HTTPError("https://oauth2.googleapis.com/token", 302,
                                                 "redirect", {}, None)
            auth.opener = RedirectOpener()
            with self.assertRaises(Conflict):
                auth.token()

    def test_world_readable_oauth_file_fails_before_http(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oauth.json"
            path.write_text(json.dumps({"client_id": "synthetic_client",
                                        "client_secret": "synthetic_secret",
                                        "refresh_token": "synthetic_refresh"}))
            os.chmod(path, 0o644)
            auth = SafeOAuthRefresh(path)
            class NoNetwork:
                def open(self, *_args, **_kwargs):
                    raise AssertionError("world-readable grant triggered network")
            auth.opener = NoNetwork()
            with self.assertRaises(Conflict):
                auth.token()


if __name__ == "__main__":
    unittest.main()
