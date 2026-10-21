"""Synthetic-only recovery adapter tests; no credentials or network access."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import urllib.parse
from unittest import mock

import receipt_recovery_live_verifier as live


def digests(raw):
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "md5": hashlib.md5(raw, usedforsecurity=False).hexdigest()}


class Clock:
    def __init__(self):
        self.wall = 2000.0
        self.tick = 0.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.tick

    def sleep(self, seconds):
        self.wall += seconds
        self.tick += seconds


class Raw(io.BytesIO):
    def __init__(self, body, advance=None):
        super().__init__(body)
        self.advance = advance
        self.reads = []

    def read(self, count, *, decode_content=False):
        if decode_content is not False:
            raise AssertionError("automatic decompression is forbidden")
        self.reads.append(count)
        if self.advance:
            self.advance()
        return super().read(count)


class Response:
    def __init__(self, body=b"", *, status=200, headers=None, history=None, advance=None):
        self.raw = Raw(body, advance=advance)
        self.status_code = status
        self.headers = headers or {}
        self.history = history or []
        self.closed = False

    def close(self):
        self.closed = True
        self.raw.close()


class HTTP:
    def __init__(self, clock):
        self.clock = clock
        self.routes = {}
        self.calls = []
        self.responses = []
        self.closed = False

    @staticmethod
    def key(url):
        split = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(split.query)
        return (split.netloc, split.path, query.get("alt", [""])[0])

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, copy.deepcopy(kwargs), self.clock.tick))
        if kwargs.get("allow_redirects") is not False or kwargs.get("stream") is not True:
            raise AssertionError("all requests must be streamed with redirects disabled")
        if not isinstance(kwargs.get("timeout"), tuple) or not all(0 < n <= 30 for n in kwargs["timeout"]):
            raise AssertionError("unbounded connect/read timeout")
        result = self.routes[self.key(url)]()
        self.responses.append(result)
        return result

    def close(self):
        self.closed = True


class Auth:
    def __init__(self):
        self.calls = 0

    def token(self):
        self.calls += 1
        return "synthetic-token"


class RecoveryVerifierTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.http = HTTP(self.clock)
        self.gcs = HTTP(self.clock)
        self.auth = Auth()
        self.bodies = {"payload": b"synthetic-active-payload\x00" * 101,
                       "receipt": b'{"source":"synthetic-active-receipt"}'}
        self.ids = {"folder": "folderSynthetic0123", "payload": "payloadSynthetic0123",
                    "receipt": "receiptSynthetic0123"}
        self.names = {"payload": "seg_0000000001.jss", "receipt": "seg_0000000001.jss.receipt.json"}
        self.pins = {kind: digests(raw) for kind, raw in self.bodies.items()}
        self.manifest = {"account_id": "permission-synthetic", "drive_root": "rootSynthetic0123",
                         "bucket": "synthetic-recovery-bucket", "folder_name": "retained-active-synthetic"}
        for kind, generation in (("payload", 111), ("receipt", 112)):
            self.manifest[kind] = {"object": "sources/synthetic/" + kind + "/a b?x=1",
                                   "generation": generation, "bytes": len(self.bodies[kind]),
                                   "sha256": self.pins[kind]["sha256"]}
        self._json_route(self.http, "https://www.googleapis.com/drive/v3/about",
                         {"user": {"permissionId": self.manifest["account_id"]}})
        self._json_route(self.http, self._drive(self.manifest["drive_root"]), {
            "id": self.manifest["drive_root"], "mimeType": live.FOLDER_MIME, "trashed": False})
        self._json_route(self.http, self._drive(self.ids["folder"]), {
            "id": self.ids["folder"], "name": self.manifest["folder_name"],
            "parents": [self.manifest["drive_root"]], "mimeType": live.FOLDER_MIME, "trashed": False})
        for kind in ("payload", "receipt"):
            wanted = self.manifest[kind]
            self._json_route(self.http, self._drive(self.ids[kind]), self._drive_meta(kind))
            self.http.routes[HTTP.key(self._drive(self.ids[kind]) + "?alt=media")] = (
                lambda kind=kind: Response(self.bodies[kind], headers={"Content-Length": str(len(self.bodies[kind]))}))
            self._json_route(self.gcs, self._gcs(kind), self._gcs_meta(kind))
            self.gcs.routes[HTTP.key(self._gcs(kind, media=True) + "?alt=media")] = (
                lambda kind=kind: Response(self.bodies[kind], headers={
                    "Content-Length": str(len(self.bodies[kind])),
                    "x-goog-generation": str(self.manifest[kind]["generation"])}))
        self.verifier = self.make()
        self.addCleanup(self.verifier.close)

    def make(self, **overrides):
        params = dict(manifest=self.manifest, entry_digests=self.pins, drive_ids=self.ids,
                      payload_name=self.names["payload"], receipt_name=self.names["receipt"],
                      oauth_file=Path("/nonexistent/synthetic/oauth.json"),
                      deadline_epoch=self.clock.wall + 240, gcs_http=self.gcs,
                      drive_http=self.http, drive_auth=self.auth, clock=self.clock.time,
                      mono=self.clock.monotonic, sleep=self.clock.sleep)
        params.update(overrides)
        return live.ReceiptRecoveryLiveVerifier(**params)

    def _drive(self, file_id):
        return "https://www.googleapis.com/drive/v3/files/" + file_id

    def _gcs(self, kind, media=False):
        path = "/storage/v1/b/" + self.manifest["bucket"] + "/o/"
        path += urllib.parse.quote(self.manifest[kind]["object"], safe="")
        return "https://storage.googleapis.com" + ("/download" if media else "") + path

    def _drive_meta(self, kind):
        return {"id": self.ids[kind], "name": self.names[kind], "parents": [self.ids["folder"]],
                "trashed": False, "mimeType": "application/octet-stream" if kind == "payload" else "application/json",
                "size": str(len(self.bodies[kind])), "md5Checksum": self.pins[kind]["md5"]}

    def _gcs_meta(self, kind):
        wanted = self.manifest[kind]
        return {"bucket": self.manifest["bucket"], "name": wanted["object"],
                "generation": str(wanted["generation"]), "size": str(wanted["bytes"]),
                "md5Hash": base64.b64encode(bytes.fromhex(self.pins[kind]["md5"])).decode(),
                "metadata": {"sha256": wanted["sha256"]}}

    @staticmethod
    def _json_route(http, url, value):
        http.routes[HTTP.key(url)] = lambda: Response(json.dumps(value).encode())

    def identity(self, verifier=None):
        verifier = verifier or self.verifier
        verifier.verify_account(self.manifest["account_id"])
        verifier.verify_folder(self.ids["folder"], self.manifest["folder_name"], self.manifest["drive_root"])

    def absence(self, verifier=None):
        return (verifier or self.verifier).verify_exact_receipt_absent(
            account_id=self.manifest["account_id"], root_id=self.manifest["drive_root"],
            folder_id=self.ids["folder"], receipt_id=self.ids["receipt"])

    def test_controller_contract_streams_both_exact_sources_and_drive_files(self):
        self.identity()
        for kind in ("payload", "receipt"):
            wanted = self.manifest[kind]
            gcs = self.verifier.gcs_digest(wanted["object"], wanted["generation"])
            drive = self.verifier.drive_digest(self.ids[kind])
            for proof in (gcs, drive):
                self.assertEqual(proof["bytes"], wanted["bytes"])
                self.assertEqual(proof["sha256"], wanted["sha256"])
                self.assertEqual(proof["md5"], self.pins[kind]["md5"])
            self.assertEqual((gcs["object"], gcs["generation"]), (wanted["object"], wanted["generation"]))
            self.assertEqual(drive["id"], self.ids[kind])
            self.assertEqual(drive["account_id"], self.manifest["account_id"])
            self.assertEqual(drive["parent_id"], self.ids["folder"])
            self.assertIs(drive["trashed"], False)
            self.assertIs(drive["media_readback"], True)
        self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual((self.verifier.drive_gets, self.verifier.gcs_gets), (8, 4))
        self.assertEqual(self.auth.calls, 1)
        for method, url, args, _tick in self.gcs.calls:
            self.assertEqual(method, "GET")
            self.assertNotIn("data", args)
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            self.assertEqual(query["generation"], query["ifGenerationMatch"])
            self.assertGreater(args["max_allowed_time"], 0)
        times = [row[3] for row in self.http.calls]
        self.assertTrue(all(b - a >= 0.5999999 for a, b in zip(times, times[1:])))
        self.assertTrue(all(r.closed for r in self.http.responses + self.gcs.responses))

    def test_exact_absence_requires_fresh_identity_then_one_404(self):
        self.http.routes[HTTP.key(self._drive(self.ids["receipt"]))] = lambda: Response(status=404)
        value = self.absence()
        self.assertEqual(value, {"account_id": self.manifest["account_id"],
                               "root_id": self.manifest["drive_root"], "folder_id": self.ids["folder"],
                               "receipt_id": self.ids["receipt"], "http_status": 404,
                               "observed_at": self.clock.wall})
        self.assertEqual(len(self.http.calls), 4)
        self.assertEqual(len(self.gcs.calls), 0)
        self.assertTrue(all(row[0] == "GET" for row in self.http.calls))

    def test_full_expired_wrapper_sequence_fits_explicit_20_get_bound(self):
        # Root wrapper proves identity before opening authority; current
        # controller repeats it before its expired-session branch.
        self.identity()
        self.identity()
        receipt_key = HTTP.key(self._drive(self.ids["receipt"]))
        self.http.routes[receipt_key] = lambda: Response(status=404)
        self.assertEqual(self.absence()["http_status"], 404)
        self.assertEqual(self.verifier.drive_gets, 10)
        # The controller's separately charged recovery transport handles the
        # permitted receipt write. This verifier only observes the later file.
        self._json_route(self.http, self._drive(self.ids["receipt"]), self._drive_meta("receipt"))
        self.identity()
        for kind in ("payload", "receipt"):
            source = self.manifest[kind]
            self.verifier.gcs_digest(source["object"], source["generation"])
            self.verifier.drive_digest(self.ids[kind])
        self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual((self.verifier.drive_gets, self.verifier.gcs_gets), (18, 4))
        self.assertEqual(live.MAX_DRIVE_GETS, 20)
        self.assertTrue(all(row[0] == "GET" for row in self.http.calls + self.gcs.calls))
        self.assertTrue(all(response.closed for response in self.http.responses + self.gcs.responses))

    def test_absence_rejects_200_403_429_500_without_retry(self):
        key = HTTP.key(self._drive(self.ids["receipt"]))
        for status in (200, 403, 429, 500):
            with self.subTest(status=status):
                self.http.calls.clear()
                self.http.routes[key] = lambda status=status: Response(status=status)
                verifier = self.make()
                with verifier, self.assertRaises(live.Conflict):
                    self.absence(verifier)
                self.assertEqual(len(self.http.calls), 4)

    def test_absence_wrong_pins_fail_before_auth_or_http(self):
        with self.assertRaises(live.Conflict):
            self.verifier.verify_exact_receipt_absent(
                "wrong-account", self.manifest["drive_root"], self.ids["folder"], self.ids["receipt"])
        self.assertEqual((self.auth.calls, len(self.http.calls)), (0, 0))

    def test_wrong_account_or_folder_fails_before_media(self):
        self._json_route(self.http, "https://www.googleapis.com/drive/v3/about",
                         {"user": {"permissionId": "other"}})
        with self.assertRaisesRegex(live.Conflict, "permission"):
            self.identity()
        self.assertEqual(len(self.http.calls), 1)
        self.assertEqual(len(self.gcs.calls), 0)

    def test_wrong_root_or_parent_is_rejected(self):
        self._json_route(self.http, self._drive(self.ids["folder"]), {
            "id": self.ids["folder"], "name": self.manifest["folder_name"],
            "parents": ["other-root"], "mimeType": live.FOLDER_MIME, "trashed": False})
        with self.assertRaisesRegex(live.Conflict, "folder metadata"):
            self.identity()

    def test_generation_and_metadata_mismatch_stop_before_media(self):
        self.identity()
        wanted = self.manifest["payload"]
        with self.assertRaisesRegex(live.Conflict, "generation input"):
            self.verifier.gcs_digest(wanted["object"], wanted["generation"] + 1)
        self.assertEqual(len(self.gcs.calls), 0)
        meta = self._gcs_meta("payload")
        meta["generation"] = "999"
        self._json_route(self.gcs, self._gcs("payload"), meta)
        with self.assertRaisesRegex(live.Conflict, "generation metadata"):
            self.verifier.gcs_digest(wanted["object"], wanted["generation"])
        self.assertEqual(len(self.gcs.calls), 1)

    def test_role_specific_mime_rejects_swapped_types(self):
        for kind, wrong in (("payload", "application/json"), ("receipt", "application/octet-stream")):
            with self.subTest(kind=kind):
                verifier = self.make()
                self.identity(verifier)
                meta = self._drive_meta(kind)
                meta["mimeType"] = wrong
                self._json_route(self.http, self._drive(self.ids[kind]), meta)
                with verifier, self.assertRaisesRegex(live.Conflict, "Drive file metadata"):
                    verifier.drive_digest(self.ids[kind])

    def test_receipt_corruption_is_detected_after_full_stream(self):
        self.identity()
        key = HTTP.key(self._drive(self.ids["receipt"]) + "?alt=media")
        self.http.routes[key] = lambda: Response(b"x" * len(self.bodies["receipt"]))
        with self.assertRaisesRegex(live.Conflict, "full media digest"):
            self.verifier.drive_digest(self.ids["receipt"])
        self.assertTrue(self.http.responses[-1].closed)

    def test_gcs_corruption_is_detected_after_full_stream(self):
        self.identity()
        key = HTTP.key(self._gcs("payload", media=True) + "?alt=media")
        self.gcs.routes[key] = lambda: Response(b"x" * len(self.bodies["payload"]))
        wanted = self.manifest["payload"]
        with self.assertRaisesRegex(live.Conflict, "full media digest"):
            self.verifier.gcs_digest(wanted["object"], wanted["generation"])
        self.assertTrue(self.gcs.responses[-1].closed)

    def test_media_short_oversized_partial_encoded_and_generation_refused(self):
        cases = [(b"short", {}), (self.bodies["payload"] + b"x", {}),
                 (self.bodies["payload"], {"Content-Range": "bytes 0-10/11"}),
                 (self.bodies["payload"], {"Content-Encoding": "gzip"}),
                 (self.bodies["payload"], {"x-goog-generation": "999"})]
        for body, headers in cases:
            with self.subTest(headers=headers, count=len(body)):
                verifier = self.make()
                self.identity(verifier)
                key = HTTP.key(self._gcs("payload", media=True) + "?alt=media")
                self.gcs.routes[key] = lambda body=body, headers=headers: Response(body, headers=headers)
                wanted = self.manifest["payload"]
                with verifier, self.assertRaises(live.Conflict):
                    verifier.gcs_digest(wanted["object"], wanted["generation"])
                self.assertTrue(self.gcs.responses[-1].closed)

    def test_redirects_and_http_failures_are_never_retried(self):
        for status in (301, 302, 307, 401, 403, 404, 429, 500):
            with self.subTest(status=status):
                self.http.calls.clear()
                self.http.routes[HTTP.key("https://www.googleapis.com/drive/v3/about")] = (
                    lambda status=status: Response(status=status))
                with self.make() as verifier, self.assertRaises(live.Conflict):
                    verifier.verify_account(self.manifest["account_id"])
                self.assertEqual(len(self.http.calls), 1)
                self.assertTrue(self.http.responses[-1].closed)

    def test_transport_exception_never_retries(self):
        def failed():
            raise TimeoutError("synthetic socket timeout")
        self.http.routes[HTTP.key("https://www.googleapis.com/drive/v3/about")] = failed
        with self.assertRaisesRegex(live.Conflict, "read failed"):
            self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual(len(self.http.calls), 1)

    def test_json_bound_and_duplicate_keys_are_refused(self):
        for body in (b"x" * (live.MAX_JSON_BYTES + 1), b'{"user":{},"user":{}}'):
            self.http.routes[HTTP.key("https://www.googleapis.com/drive/v3/about")] = lambda body=body: Response(body)
            with self.make() as verifier, self.assertRaises(live.Conflict):
                verifier.verify_account(self.manifest["account_id"])
            self.assertTrue(self.http.responses[-1].closed)

    def test_backward_wall_cannot_extend_shared_monotonic_deadline(self):
        self.clock.wall -= 3600
        self.clock.tick += 241
        with self.assertRaisesRegex(live.Conflict, "shared deadline"):
            self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual(self.http.calls, [])

    def test_forward_wall_expiry_stops_before_token_or_read(self):
        self.clock.wall += 241
        with self.assertRaises(live.Conflict):
            self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual((self.auth.calls, len(self.http.calls)), (0, 0))

    def test_token_delay_cannot_escape_shared_deadline(self):
        def slow_token():
            self.clock.sleep(241)
            return "synthetic-token"
        self.auth.token = slow_token
        with self.assertRaises(live.Conflict):
            self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual(len(self.http.calls), 0)

    def test_media_stream_time_bound_closes_response(self):
        self.identity()
        key = HTTP.key(self._drive(self.ids["receipt"]) + "?alt=media")
        self.http.routes[key] = lambda: Response(self.bodies["receipt"], advance=lambda: self.clock.sleep(91))
        with self.assertRaisesRegex(live.Conflict, "stream deadline"):
            self.verifier.drive_digest(self.ids["receipt"])
        self.assertTrue(self.http.responses[-1].closed)

    def test_deadline_is_shared_across_reads_and_timeout_is_clamped(self):
        with self.make(deadline_epoch=self.clock.wall + 2) as verifier:
            verifier.verify_account(self.manifest["account_id"])
            self.assertTrue(all(n <= 2 for n in self.http.calls[-1][2]["timeout"]))
            self.clock.sleep(1.8)
            with self.assertRaisesRegex(live.Conflict, "pacing cannot fit"):
                # The previous completion is made recent to expose the pacing gate.
                verifier.last_drive_completion = self.clock.tick
                verifier.verify_account(self.manifest["account_id"])

    def test_pacing_is_from_completion_and_never_compresses_after_delay(self):
        original = self.http.request
        def delayed(method, url, **kwargs):
            if not self.http.calls:
                self.clock.sleep(10)
            return original(method, url, **kwargs)
        self.http.request = delayed
        self.identity()
        times = [row[3] for row in self.http.calls]
        self.assertTrue(all(b - a >= 0.5999999 for a, b in zip(times, times[1:])))

    def test_fixed_hosts_reject_before_credentials(self):
        for url in ("http://www.googleapis.com/drive/v3/about", "https://evil.test/drive/v3/about",
                    "https://www.googleapis.com.evil.test/drive/v3/about"):
            with self.assertRaises(live.Conflict), self.verifier._get("drive", url):
                self.fail("outside host must not open")
        self.assertEqual((self.auth.calls, len(self.http.calls)), (0, 0))

    def test_call_bounds_stop_before_extra_dispatch(self):
        self.verifier.drive_gets = live.MAX_DRIVE_GETS
        with self.assertRaisesRegex(live.Conflict, "read count"):
            self.verifier.verify_account(self.manifest["account_id"])
        self.assertEqual(self.http.calls, [])

    def test_constructor_binds_original_inputs_and_rejects_incomplete_digest(self):
        changed = copy.deepcopy(self.pins)
        changed["receipt"]["md5"] = "bad"
        with self.assertRaises(live.Conflict):
            self.make(entry_digests=changed)
        self.manifest["account_id"] = "changed-after-construction"
        self.verifier.verify_account("permission-synthetic")
        with self.assertRaises(live.Conflict):
            self.verifier.verify_account("changed-after-construction")

    def test_existing_oauth_refresh_is_only_post_and_has_no_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-oauth.json"
            path.write_text(json.dumps({"client_id": "synthetic-client", "client_secret": "synthetic-secret",
                                        "refresh_token": "synthetic-refresh"}))
            path.chmod(0o600)
            self._json_route(self.http, "https://oauth2.googleapis.com/token", {"access_token": "synthetic-token"})
            with self.make(oauth_file=path, drive_auth=None) as verifier:
                self.identity(verifier)
            posts = [call for call in self.http.calls if call[0] == "POST"]
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0][1], "https://oauth2.googleapis.com/token")
            fields = urllib.parse.parse_qs(posts[0][2]["data"].decode())
            self.assertEqual(set(fields), {"client_id", "client_secret", "refresh_token", "grant_type"})
            self.assertEqual(fields["grant_type"], ["refresh_token"])
            self.assertTrue(all(response.closed for response in self.http.responses))

    def test_adc_transport_dispatches_no_duplicate_or_redirected_request(self):
        url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
        auth_http = HTTP(self.clock)
        auth_http.routes[HTTP.key(url)] = lambda: Response(b"synthetic-project")
        with mock.patch.object(live, "_plain_session", return_value=auth_http):
            request = live._BoundedADCRequest(self.verifier)
        try:
            value = request(url)
            self.assertEqual(value.data, b"synthetic-project")
            with self.assertRaisesRegex(live.Conflict, "bounded allowlist"):
                request(url)
            with self.assertRaises(live.Conflict):
                request("https://evil.test/token", method="POST", body=b"synthetic")
            self.assertEqual(len(auth_http.calls), 1)
        finally:
            request.close()
        self.assertTrue(auth_http.closed)

    def test_production_gcs_factory_uses_adc_and_zero_refresh_retries(self):
        created = {}
        google = types.ModuleType("google")
        auth = types.ModuleType("google.auth")
        transport = types.ModuleType("google.auth.transport")
        requests_transport = types.ModuleType("google.auth.transport.requests")
        requests = types.ModuleType("requests")
        adapters = types.ModuleType("requests.adapters")
        class Session:
            def __init__(self, credentials, **kwargs):
                created.update(kwargs)
                self.credentials = credentials
                self.mounts = []
                self.closed = False
            def mount(self, prefix, adapter):
                self.mounts.append((prefix, adapter))
            def close(self):
                self.closed = True
        auth.default = lambda **kwargs: ("synthetic-existing-adc", "synthetic-project")
        requests_transport.AuthorizedSession = Session
        adapters.HTTPAdapter = lambda **kwargs: kwargs
        google.auth = auth
        modules = {"google": google, "google.auth": auth, "google.auth.transport": transport,
                   "google.auth.transport.requests": requests_transport,
                   "requests": requests, "requests.adapters": adapters}
        verifier = self.make(gcs_http=None)
        with mock.patch.dict("sys.modules", modules), mock.patch.object(live, "_plain_session", return_value=HTTP(self.clock)):
            session = verifier._session("gcs")
            self.assertEqual(created["max_refresh_attempts"], 0)
            self.assertLessEqual(created["refresh_timeout"], 30)
            self.assertFalse(session.trust_env)
            self.assertEqual(session.mounts, [("https://", {"max_retries": 0})])
            verifier.close()
            self.assertTrue(session.closed)

    def test_closed_adapter_cannot_read_again(self):
        self.verifier.close()
        with self.assertRaisesRegex(live.Conflict, "closed"):
            self.verifier.verify_account(self.manifest["account_id"])


if __name__ == "__main__":
    unittest.main()
