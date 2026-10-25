"""Session creation, card payloads and local voice notes, shared by the server and the CLI."""

from __future__ import annotations

import json
import re
import secrets
import threading
from pathlib import Path
from typing import Any

from . import INTEL_DIR, LOCAL_DIR, SESSIONS_DIR
from .axes import NARRATE_KIND, TRAP_AXIS, axis_spec, load_rubric
from .pools import POOLS, load_context_index, load_pool, make_card
from .store import Session, _locked_append, _read_rows, utc_now

NOTES_DIR = LOCAL_DIR / "notes"
NOTE_ID_RE = re.compile(r"^n-[0-9a-f]{12}$")
AUDIO_EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav"}
MAX_NOTE_BYTES = 20 * 1024 * 1024


def new_session(
    *,
    kind: str,
    axis: str | None,
    pool: str,
    subset: str | None = None,
    seed: int | None = None,
    root: Path = SESSIONS_DIR,
) -> Session:
    if kind == "trap":
        axis, pool = TRAP_AXIS, "trap_pairs_v1"
    elif kind == NARRATE_KIND:
        axis = NARRATE_KIND
    rows, pool_sha = load_pool(pool, subset=subset)
    _, context_hashes = load_context_index()
    _, rubric_sha = load_rubric()
    if kind == "trap":
        keys = [r["pair_id"] for r in rows]
        uri_by_key = {r["pair_id"]: r["uri"] for r in rows}
    else:
        keys = [r["uri"] for r in rows]
        uri_by_key = None
    return Session.create(
        kind=kind,
        axis=str(axis),
        pool=pool,
        subset=subset,
        item_keys=keys,
        pool_sha256=pool_sha,
        context_sha256=context_hashes,
        rubric_sha256=rubric_sha,
        seed=seed,
        uri_by_key=uri_by_key,
        root=root,
    )


def session_payload(session: Session) -> dict[str, Any]:
    """Everything the page needs to resume a session exactly where it stopped."""
    rows, pool_sha = load_pool(session.meta["pool"], subset=session.meta.get("subset"))
    if pool_sha != session.meta["pool_sha256"]:
        raise ValueError(f"Pool {session.meta['pool']} changed since session {session.id} was created")
    index, _ = load_context_index()
    key_field = "pair_id" if session.kind == "trap" else "uri"
    by_key = {r[key_field]: r for r in rows}
    cards = [make_card(by_key[key], index) | {"key": key} for key in session.order]
    rubric, _ = load_rubric()
    state = session.state()
    payload: dict[str, Any] = {
        "session": {k: v for k, v in session.meta.items() if k not in ("order", "uri_by_key")},
        "cards": cards,
        "effective": {key: row.get("value") for key, row in state.effective.items()},
        "cursor": state.cursor,
    }
    if session.kind != NARRATE_KIND:
        payload["axis_spec"] = axis_spec(session.axis, rubric)
    return payload


def available_pools() -> list[dict[str, Any]]:
    out = []
    for name, spec in POOLS.items():
        if not spec.path.exists():
            out.append({"pool": name, "built": False})
            continue
        rows, _ = load_pool(name)
        subsets = sorted({r["subset"] for r in rows if r.get("subset")})
        out.append({"pool": name, "built": True, "n": len(rows), "subsets": subsets})
    return out


class NotesStore:
    """Push-to-talk notes: audio files plus an append-only notes.jsonl, all under labeling/local/."""

    def __init__(self, root: Path = NOTES_DIR, transcriber: Any = "auto"):
        self.root = root
        self.index_path = root / "notes.jsonl"
        self._transcriber = transcriber

    def _transcribe_fn(self) -> Any:
        if self._transcriber != "auto":
            return self._transcriber
        try:
            from .transcribe import transcribe_file
        except ImportError:
            return None
        return transcribe_file

    def add(self, *, session_id: str, key: str, uri: str, mime: str, audio: bytes, duration_ms: int) -> dict[str, Any]:
        if not audio or len(audio) > MAX_NOTE_BYTES:
            raise ValueError("Note audio is empty or too large")
        ext = AUDIO_EXTENSIONS.get(mime.split(";")[0].strip())
        if ext is None:
            raise ValueError(f"Unsupported audio type {mime!r}")
        note_id = f"n-{secrets.token_hex(6)}"
        audio_path = self.root / session_id / f"{note_id}.{ext}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio)
        row = {
            "op": "note",
            "note_id": note_id,
            "session_id": session_id,
            "key": key,
            "uri": uri,
            "audio_path": str(audio_path.relative_to(self.root)),
            "mime": mime,
            "bytes": len(audio),
            "duration_ms": max(0, int(duration_ms)),
            "ts": utc_now(),
        }
        self._append(row)
        fn = self._transcribe_fn()
        if fn is not None:
            threading.Thread(target=self._transcribe, args=(fn, note_id, audio_path), daemon=True).start()
        return row

    def _transcribe(self, fn: Any, note_id: str, audio_path: Path) -> None:
        try:
            result = fn(audio_path)
            self._append({"op": "transcript", "note_id": note_id, "ts": utc_now(), **result})
        except Exception as exc:  # noqa: BLE001 - notes must never block labeling
            self._append({"op": "transcript_error", "note_id": note_id, "ts": utc_now(), "error": str(exc)[:500]})

    def _append(self, row: dict[str, Any]) -> None:
        with _locked_append(self.index_path) as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def notes(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        with self.index_path.open(encoding="utf-8") as fh:
            rows = _read_rows(fh)
        notes = {r["note_id"]: dict(r) for r in rows if r["op"] == "note"}
        for row in rows:
            if row["op"] in ("transcript", "transcript_error") and row["note_id"] in notes:
                notes[row["note_id"]]["transcript"] = row.get("text")
                notes[row["note_id"]]["transcript_error"] = row.get("error")
                notes[row["note_id"]]["engine"] = row.get("engine")
        out = list(notes.values())
        return [n for n in out if session_id is None or n["session_id"] == session_id]


def relpath(path: Path) -> str:
    return str(path.relative_to(INTEL_DIR)) if path.is_relative_to(INTEL_DIR) else str(path)
