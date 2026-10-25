"""Append-only labeling sessions: one session.json plus one labels JSONL per session.

Rows are never rewritten. Undo appends an `undo` row naming the label row it
revokes; relabeling a card appends a new `label` row that supersedes the old one.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import SESSIONS_DIR
from .axes import COMPONENT_AXES, NARRATE_KIND, TRAP_AXIS, normalize_value

LABEL_SCHEMA = "catbird-gold-label-v1"
SESSION_SCHEMA = "catbird-gold-label-session-v1"
TOOL_VERSION = "1.0.0"
SESSION_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-z_]+-[0-9a-f]{6}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def seeded_order(uris: list[str], seed: int) -> list[str]:
    order = sorted(uris)
    random.Random(seed).shuffle(order)
    return order


@contextmanager
def _locked_append(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_rows(fh: Any) -> list[dict[str, Any]]:
    fh.seek(0)
    rows = []
    for line_no, line in enumerate(fh, 1):
        if not line.strip():
            continue
        if not line.endswith("\n"):
            raise ValueError(f"{fh.name}:{line_no} is a torn write; inspect before continuing")
        rows.append(json.loads(line))
    return rows


@dataclass
class SessionState:
    effective: dict[str, dict[str, Any]]  # item key -> live label row
    cursor: int  # first order position without an effective label (len(order) when done)
    n_label_rows: int
    n_undo_rows: int


class Session:
    def __init__(self, directory: Path):
        self.dir = directory
        self.meta: dict[str, Any] = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        self.id: str = self.meta["session_id"]
        self.axis: str = self.meta["axis"]
        self.kind: str = self.meta["kind"]
        self.order: list[str] = self.meta["order"]
        self._positions = {key: i for i, key in enumerate(self.order)}

    @property
    def labels_path(self) -> Path:
        return self.dir / f"labels-{self.axis}-{self.id}.jsonl"

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        axis: str,
        pool: str,
        subset: str | None,
        item_keys: list[str],
        pool_sha256: str,
        context_sha256: dict[str, str],
        rubric_sha256: str,
        seed: int | None = None,
        uri_by_key: dict[str, str] | None = None,
        root: Path = SESSIONS_DIR,
    ) -> "Session":
        if kind == "axis" and axis not in COMPONENT_AXES:
            raise ValueError(f"Axis session needs one of {COMPONENT_AXES}, got {axis!r}")
        if kind == "trap" and axis != TRAP_AXIS:
            raise ValueError("Trap sessions use axis 'trap'")
        if kind == NARRATE_KIND and axis != NARRATE_KIND:
            raise ValueError("Narration sessions use axis 'narrate'")
        if kind not in ("axis", "trap", NARRATE_KIND):
            raise ValueError(f"Unknown session kind {kind!r}")
        if len(set(item_keys)) != len(item_keys) or not item_keys:
            raise ValueError("Session items must be non-empty and unique")
        seed = secrets.randbits(32) if seed is None else seed
        now = datetime.now(timezone.utc)
        session_id = f"{now:%Y%m%dT%H%M%SZ}-{axis}-{secrets.token_hex(3)}"
        directory = root / session_id
        directory.mkdir(parents=True, exist_ok=False)
        meta = {
            "schema": SESSION_SCHEMA,
            "tool_version": TOOL_VERSION,
            "session_id": session_id,
            "created_at": utc_now(),
            "kind": kind,
            "axis": axis,
            "pool": pool,
            "subset": subset,
            "pool_sha256": pool_sha256,
            "context_sha256": context_sha256,
            "rubric_sha256": rubric_sha256,
            "shuffle": "sorted(uris) then random.Random(seed).shuffle",
            "seed": seed,
            "order": seeded_order(item_keys, seed),
        }
        if uri_by_key:
            meta["uri_by_key"] = uri_by_key
        (directory / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return cls(directory)

    @classmethod
    def open(cls, session_id: str, root: Path = SESSIONS_DIR) -> "Session":
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"Malformed session id {session_id!r}")
        directory = root / session_id
        if not (directory / "session.json").exists():
            raise FileNotFoundError(session_id)
        return cls(directory)

    def rows(self) -> list[dict[str, Any]]:
        if not self.labels_path.exists():
            return []
        with self.labels_path.open(encoding="utf-8") as fh:
            return _read_rows(fh)

    def state(self, rows: list[dict[str, Any]] | None = None) -> SessionState:
        rows = self.rows() if rows is None else rows
        revoked = {r["undoes_row_id"] for r in rows if r["op"] == "undo"}
        effective: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["op"] in ("label", "view") and row["row_id"] not in revoked:
                effective[row["key"]] = row
        cursor = next((i for i, key in enumerate(self.order) if key not in effective), len(self.order))
        return SessionState(
            effective=effective,
            cursor=cursor,
            n_label_rows=sum(r["op"] in ("label", "view") for r in rows),
            n_undo_rows=len(revoked),
        )

    def _base_row(self, op: str, row_id: int, key: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema": LABEL_SCHEMA,
            "op": op,
            "row_id": row_id,
            "key": key,
            "uri": self.meta.get("uri_by_key", {}).get(key, key),
            "axis": self.axis,
            "ts": utc_now(),
            "session_id": self.id,
            "position": self._positions[key],
        }
        if self.kind == "trap":
            row["pair_id"] = key
        return row

    def label(self, key: str, value: Any, *, ms_to_decide: int, note_ids: list[str] | None = None) -> dict[str, Any]:
        if self.kind == NARRATE_KIND:
            raise ValueError("Narration sessions record views, not labels")
        if key not in self._positions:
            raise KeyError(f"{key} is not in session {self.id}")
        stored = normalize_value(self.axis, value)
        with _locked_append(self.labels_path) as fh:
            row = self._base_row("label", len(_read_rows(fh)) + 1, key)
            row.update(value=stored, ms_to_decide=max(0, int(ms_to_decide)), note_ids=list(note_ids or []))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def view(self, key: str, *, ms_on_card: int, note_ids: list[str] | None = None) -> dict[str, Any]:
        """Narration pass: one row per card left, with time spent and notes recorded."""
        if self.kind != NARRATE_KIND:
            raise ValueError("Only narration sessions record views")
        if key not in self._positions:
            raise KeyError(f"{key} is not in session {self.id}")
        with _locked_append(self.labels_path) as fh:
            row = self._base_row("view", len(_read_rows(fh)) + 1, key)
            row.update(ms_on_card=max(0, int(ms_on_card)), note_ids=list(note_ids or []))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def undo(self) -> dict[str, Any] | None:
        """Revoke the most recent live label row; repeated undo walks further back."""
        with _locked_append(self.labels_path) as fh:
            rows = _read_rows(fh)
            revoked = {r["undoes_row_id"] for r in rows if r["op"] == "undo"}
            target = next(
                (r for r in reversed(rows) if r["op"] in ("label", "view") and r["row_id"] not in revoked),
                None,
            )
            if target is None:
                return None
            row = self._base_row("undo", len(rows) + 1, target["key"])
            row["undoes_row_id"] = target["row_id"]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


def list_sessions(root: Path = SESSIONS_DIR) -> list[dict[str, Any]]:
    out = []
    if not root.exists():
        return out
    for directory in sorted(root.iterdir()):
        if not (directory / "session.json").exists():
            continue
        session = Session(directory)
        state = session.state()
        out.append(
            {
                "session_id": session.id,
                "kind": session.kind,
                "axis": session.axis,
                "pool": session.meta["pool"],
                "subset": session.meta.get("subset"),
                "n_items": len(session.order),
                "n_done": len(state.effective),
                "created_at": session.meta["created_at"],
            }
        )
    return out
