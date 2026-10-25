"""Localhost-only HTTP server for the labeling page (stdlib; binds 127.0.0.1)."""

from __future__ import annotations

import json
import mimetypes
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import PACKAGE_DIR, SESSIONS_DIR
from .axes import COMPONENT_AXES
from .service import NotesStore, available_pools, new_session, session_payload
from .store import Session, list_sessions

STATIC_DIR = PACKAGE_DIR / "static"
MAX_JSON_BYTES = 64 * 1024


class LabelingHandler(BaseHTTPRequestHandler):
    sessions_root: Path = SESSIONS_DIR
    notes: NotesStore = NotesStore()

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per error only
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            raise ValueError("Request body too large")
        return self.rfile.read(length)

    def _json_body(self) -> dict[str, Any]:
        raw = self._body(MAX_JSON_BYTES)
        return json.loads(raw or b"{}")

    def _session(self, session_id: str) -> Session:
        return Session.open(session_id, root=self.sessions_root)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        # Reject cross-origin writes: only the page this server serves may post.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            self._json({"error": "cross-origin request refused"}, HTTPStatus.FORBIDDEN)
            return
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if method == "GET" and (not parts or parts == ["index.html"]):
                return self._static("index.html")
            if method == "GET" and parts[0] == "static" and len(parts) == 2:
                return self._static(parts[1])
            if parts[0] != "api":
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return self._api(method, parts[1:], query)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._json({"error": f"not found: {exc}"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"server error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), f"{ctype}; charset=utf-8" if ctype.startswith("text/") else ctype)

    def _api(self, method: str, parts: list[str], query: dict[str, str]) -> None:
        if method == "GET" and parts == ["home"]:
            return self._json(
                {
                    "sessions": list_sessions(self.sessions_root),
                    "pools": available_pools(),
                    "axes": list(COMPONENT_AXES),
                }
            )
        if method == "POST" and parts == ["sessions"]:
            body = self._json_body()
            session = new_session(
                kind=body["kind"],
                axis=body.get("axis"),
                pool=body.get("pool", "trap_pairs_v1"),
                subset=body.get("subset") or None,
                root=self.sessions_root,
            )
            return self._json({"session_id": session.id})
        if method == "GET" and parts == ["notes"]:
            return self._json({"notes": self.notes.notes(query.get("session"))})
        if len(parts) < 2 or parts[0] != "session":
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        session = self._session(parts[1])
        action = parts[2] if len(parts) > 2 else ""
        if method == "GET" and action == "":
            return self._json(session_payload(session))
        if method == "POST" and action == "label":
            body = self._json_body()
            row = session.label(body["key"], body["value"], ms_to_decide=body["ms_to_decide"], note_ids=body.get("note_ids"))
            return self._json({"row": row, "cursor": session.state().cursor})
        if method == "POST" and action == "view":
            body = self._json_body()
            row = session.view(body["key"], ms_on_card=body["ms_on_card"], note_ids=body.get("note_ids"))
            return self._json({"row": row, "cursor": session.state().cursor})
        if method == "POST" and action == "undo":
            row = session.undo()
            state = session.state()
            effective = {key: r.get("value") for key, r in state.effective.items()}
            return self._json({"row": row, "cursor": state.cursor, "effective": effective})
        if method == "POST" and action == "note":
            key = query["key"]
            if key not in session.order:
                raise KeyError(f"{key} is not in session {session.id}")
            uri = session.meta.get("uri_by_key", {}).get(key, key)
            row = self.notes.add(
                session_id=session.id,
                key=key,
                uri=uri,
                mime=self.headers.get("Content-Type", ""),
                audio=self._body(20 * 1024 * 1024),
                duration_ms=int(query.get("duration_ms", "0")),
            )
            return self._json({"note_id": row["note_id"]})
        return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def serve(port: int = 8765, *, sessions_root: Path | None = None) -> None:
    handler = LabelingHandler
    if sessions_root is not None:
        handler = type("ScratchHandler", (LabelingHandler,), {
            "sessions_root": sessions_root, "notes": NotesStore(sessions_root / "notes")})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Labeling tool on http://127.0.0.1:{port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
