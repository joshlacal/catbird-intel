"""Synthetic receipt recovery transport tests; no credentials or cloud calls."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

from gcs_drive_drain import Conflict
from gcs_drive_retained_receipt_transport import (
    DRIVE_UPLOAD_URL, ReceiptRecoveryTransport,
)


SESSION = ("https://www.googleapis.com/upload/drive/v3/files"
           "?uploadType=resumable&upload_id=synthetic-session-secret")
SESSION_SHA = hashlib.sha256(SESSION.encode()).hexdigest()
FILE_ID = "syntheticPinnedReceiptId_123456"
FOLDER_ID = "syntheticPinnedFolderId_123456"
BODY = b"synthetic receipt bytes only"


class Response:
    def __init__(self, status, body=b"", headers=None, url=SESSION):
        self.status = status
        self.body = io.BytesIO(body)
        self.headers = headers or {}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.body.close()

    def read(self, size=-1):
        return self.body.read(size)

    def geturl(self):
        return self.url


def error(code, headers=None):
    return urllib.error.HTTPError(SESSION, code, "synthetic", headers or {},
                                  io.BytesIO(b""))


class RequestFake:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, method, data, headers, *, timeout, kind):
        self.calls.append((url, method, data, headers, timeout, kind))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class ReceiptTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.receipt = Path(self.tmp.name) / "original.receipt.json"
        self.receipt.write_bytes(BODY)
        self.sha = hashlib.sha256(BODY).hexdigest()
        self.md5 = hashlib.md5(BODY).hexdigest()

    def test_probe_complete_exact_id_is_one_empty_authorized_request(self):
        request = RequestFake(Response(200, json.dumps({"id": FILE_ID}).encode()))
        result = ReceiptRecoveryTransport(request).probe(SESSION, len(BODY), FILE_ID,
                                                         SESSION_SHA)
        self.assertEqual(result, {"state": "complete", "acknowledged": len(BODY),
                                  "metadata": {"id": FILE_ID}, "http_status": 200})
        self.assertEqual(len(request.calls), 1)
        self.assertEqual(request.calls[0][2], b"")
        self.assertEqual(request.calls[0][3]["Content-Range"], f"bytes */{len(BODY)}")
        self.assertEqual(request.calls[0][5], "resumable_status")

    def test_probe_308_range_and_empty_range_stop_before_media(self):
        for headers, offset in (({"Range": "bytes=0-5"}, 6), ({}, 0)):
            request = RequestFake(error(308, headers))
            result = ReceiptRecoveryTransport(request).probe(SESSION, len(BODY), FILE_ID,
                                                             SESSION_SHA)
            self.assertEqual((result["state"], result["acknowledged"]), ("partial", offset))
            self.assertEqual(len(request.calls), 1)
        for headers in ({"Range": f"bytes=0-{len(BODY)-1}"},
                        {"Range": "bytes=5-9"}, {"Range": "bytes=0-9999"}):
            with self.assertRaises(Conflict):
                ReceiptRecoveryTransport(RequestFake(error(308, headers))).probe(
                    SESSION, len(BODY), FILE_ID, SESSION_SHA)

    def test_probe_404_is_expired_session_not_absent_file_proof(self):
        for code in (404, 410):
            request = RequestFake(error(code))
            result = ReceiptRecoveryTransport(request).probe(SESSION, len(BODY), FILE_ID,
                                                             SESSION_SHA)
            self.assertEqual(result, {"state": "expired", "acknowledged": 0,
                                      "http_status": code})
            self.assertEqual(len(request.calls), 1)
        for code in (403, 429, 500, 503):
            with self.assertRaisesRegex(Conflict, "uncertain"):
                ReceiptRecoveryTransport(RequestFake(error(code))).probe(
                    SESSION, len(BODY), FILE_ID, SESSION_SHA)

    def test_transport_failure_does_not_report_private_session(self):
        request = RequestFake(TimeoutError("synthetic URL " + SESSION))
        with self.assertRaises(Conflict) as caught:
            ReceiptRecoveryTransport(request).probe(SESSION, len(BODY), FILE_ID,
                                                    SESSION_SHA)
        self.assertEqual(len(request.calls), 1)
        self.assertNotIn("upload_id", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_session_host_and_redirect_are_rejected_without_secret_in_error(self):
        request = RequestFake(Response(200, json.dumps({"id": FILE_ID}).encode()))
        with self.assertRaises(Conflict) as caught:
            ReceiptRecoveryTransport(request).probe(
                SESSION.replace("www.googleapis.com", "evil.example"), len(BODY), FILE_ID,
                SESSION_SHA)
        self.assertEqual(request.calls, [])
        self.assertNotIn("upload_id", str(caught.exception))
        request = RequestFake(Response(200, json.dumps({"id": FILE_ID}).encode(),
                                       url="https://evil.example/redirect"))
        with self.assertRaisesRegex(Conflict, "redirected"):
            ReceiptRecoveryTransport(request).probe(SESSION, len(BODY), FILE_ID,
                                                    SESSION_SHA)
        wrong = RequestFake(Response(200, json.dumps({"id": FILE_ID}).encode()))
        with self.assertRaisesRegex(Conflict, "fingerprint"):
            ReceiptRecoveryTransport(wrong).probe(SESSION, len(BODY), FILE_ID,
                                                  "0" * 64)
        self.assertEqual(wrong.calls, [])

    def test_resume_sends_only_reviewed_suffix_once(self):
        offset = 7
        request = RequestFake(Response(201, json.dumps({"id": FILE_ID}).encode()))
        result = ReceiptRecoveryTransport(request).resume(
            SESSION, self.receipt, offset, len(BODY), FILE_ID, self.sha, self.md5,
            SESSION_SHA)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(request.calls), 1)
        self.assertEqual(request.calls[0][2], BODY[offset:])
        self.assertEqual(request.calls[0][3]["Content-Range"],
                         f"bytes {offset}-{len(BODY)-1}/{len(BODY)}")
        self.assertEqual(request.calls[0][5], "media_chunk")

    def test_resume_308_stops_without_automatic_retry(self):
        request = RequestFake(error(308, {"Range": "bytes=0-8"}))
        result = ReceiptRecoveryTransport(request).resume(
            SESSION, self.receipt, 7, len(BODY), FILE_ID, self.sha, self.md5,
            SESSION_SHA)
        self.assertEqual((result["state"], result["acknowledged"]), ("partial", 9))
        self.assertEqual(len(request.calls), 1)
        with self.assertRaisesRegex(Conflict, "uncertain"):
            ReceiptRecoveryTransport(RequestFake(error(500))).resume(
                SESSION, self.receipt, 7, len(BODY), FILE_ID, self.sha, self.md5,
                SESSION_SHA)

    def test_resume_rejects_modified_or_symlinked_source_before_request(self):
        request = RequestFake(Response(200, json.dumps({"id": FILE_ID}).encode()))
        with self.assertRaisesRegex(Conflict, "bytes changed"):
            ReceiptRecoveryTransport(request).resume(
                SESSION, self.receipt, 0, len(BODY), FILE_ID, self.sha, "0" * 32,
                SESSION_SHA)
        self.assertEqual(request.calls, [])
        self.receipt.write_bytes(BODY + b"changed")
        with self.assertRaises(Conflict):
            ReceiptRecoveryTransport(request).resume(
                SESSION, self.receipt, 0, len(BODY), FILE_ID, self.sha, self.md5,
                SESSION_SHA)
        self.assertEqual(request.calls, [])
        link = Path(self.tmp.name) / "link.receipt.json"
        link.symlink_to(self.receipt)
        with self.assertRaises(Conflict):
            ReceiptRecoveryTransport(request).resume(
                SESSION, link, 0, len(BODY), FILE_ID, self.sha, self.md5,
                SESSION_SHA)
        self.assertEqual(request.calls, [])

    def test_restart_uses_same_pinned_id_and_returns_only_private_session(self):
        request = RequestFake(Response(200, headers={"Location": SESSION},
                                       url=DRIVE_UPLOAD_URL))
        session = ReceiptRecoveryTransport(request).restart_same_id(
            FILE_ID, FOLDER_ID, self.receipt.name, len(BODY))
        self.assertEqual(session, SESSION)
        self.assertEqual(len(request.calls), 1)
        url, method, body, headers, _timeout, kind = request.calls[0]
        self.assertEqual((url, method, kind), (DRIVE_UPLOAD_URL, "POST", "resumable_start"))
        self.assertEqual(json.loads(body), {"id": FILE_ID, "name": self.receipt.name,
                                             "parents": [FOLDER_ID]})
        self.assertEqual(headers["X-Upload-Content-Length"], str(len(BODY)))
        self.assertEqual(headers["Content-Length"], str(len(body)))

    def test_restart_409_and_untrusted_location_fail_closed(self):
        with self.assertRaisesRegex(Conflict, "uncertain"):
            ReceiptRecoveryTransport(RequestFake(error(409))).restart_same_id(
                FILE_ID, FOLDER_ID, self.receipt.name, len(BODY))
        request = RequestFake(Response(200, headers={"Location":
                                       SESSION.replace("www.googleapis.com", "evil.example")},
                                       url=DRIVE_UPLOAD_URL))
        with self.assertRaises(Conflict):
            ReceiptRecoveryTransport(request).restart_same_id(
                FILE_ID, FOLDER_ID, self.receipt.name, len(BODY))
