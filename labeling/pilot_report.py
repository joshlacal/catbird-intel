"""Pilot report: time per post with keys (all five axes) versus free narration on the same 50 posts."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from . import SESSIONS_DIR
from .axes import COMPONENT_AXES, NARRATE_KIND
from .pools import sha256_file
from .service import NotesStore
from .store import Session


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "status": "unmeasured"}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_s": round(statistics.fmean(values) / 1000, 2),
        "median_s": round(statistics.median(values) / 1000, 2),
        "p90_s": round(ordered[int(0.9 * (len(ordered) - 1))] / 1000, 2),
    }


def pilot_report(*, pool: str = "extra_v1", subset: str = "pilot50", root: Path = SESSIONS_DIR,
                 notes: NotesStore | None = None) -> dict[str, Any]:
    notes = notes or NotesStore()
    sessions = [Session(d) for d in sorted(root.iterdir()) if (d / "session.json").exists()] if root.exists() else []
    sessions = [s for s in sessions if s.meta["pool"] == pool and s.meta.get("subset") == subset]

    per_axis: dict[str, dict[str, Any]] = {}
    keys_ms_by_uri: dict[str, dict[str, int]] = {}
    narrate_ms_by_uri: dict[str, int] = {}
    receipts = []
    for session in sessions:
        state = session.state()
        receipts.append({"session_id": session.id, "axis": session.axis,
                         "labels_sha256": sha256_file(session.labels_path) if session.labels_path.exists() else None,
                         "pool_sha256": session.meta["pool_sha256"], "seed": session.meta["seed"]})
        if session.kind == NARRATE_KIND:
            for key, row in state.effective.items():
                narrate_ms_by_uri[key] = narrate_ms_by_uri.get(key, 0) + row["ms_on_card"]
            continue
        if session.axis not in COMPONENT_AXES:
            continue
        decisions = [row["ms_to_decide"] for row in state.effective.values()]
        per_axis[session.axis] = {"session_id": session.id, "labeled": len(state.effective),
                                  "of": len(session.order), "per_decision": _summary(decisions)}
        for key, row in state.effective.items():
            keys_ms_by_uri.setdefault(key, {})[session.axis] = row["ms_to_decide"]

    complete = [sum(axes.values()) for axes in keys_ms_by_uri.values() if set(axes) == set(COMPONENT_AXES)]
    session_ids = {s.id for s in sessions}
    pilot_notes = [n for n in notes.notes() if n["session_id"] in session_ids]
    narrate_ids = {s.id for s in sessions if s.kind == NARRATE_KIND}
    return {
        "schema": "catbird-gold-label-pilot-report-v1",
        "pool": pool,
        "subset": subset,
        "command": "python3 -m labeling pilot-report",
        "sessions": receipts,
        "keys_mode": {
            "axes_covered": sorted(per_axis),
            "per_axis": per_axis,
            "per_post_all_five_axes": _summary(complete),
        },
        "narration_mode": {"per_post": _summary(list(narrate_ms_by_uri.values()))},
        "notes": {
            "narration_notes": sum(n["session_id"] in narrate_ids for n in pilot_notes),
            "key_pass_notes": sum(n["session_id"] not in narrate_ids for n in pilot_notes),
            "transcribed": sum(bool(n.get("transcript")) for n in pilot_notes),
            "transcript_errors": sum(bool(n.get("transcript_error")) for n in pilot_notes),
        },
        "new_distinctions_from_narration": "unmeasured: count after reading transcripts (labeling/local/notes/notes.jsonl)",
    }
