"""Labeling axes, built from the frozen teacher rubric so humans and models read the same definitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import INTEL_DIR

RUBRIC_PATH = INTEL_DIR / "teacher_harness" / "config" / "rubric_v1.json"

ORDINAL_AXES = ("politicalness", "outraged", "critical")
# Per the 2026-10-23 handoff, engagement_bait and spam are labeled 0/1 by humans.
# Model 0-3 scores are binarized at BINARY_MODEL_THRESHOLD when graded (rubric 2 = "clear").
BINARY_AXES = ("engagement_bait", "spam")
COMPONENT_AXES = ORDINAL_AXES + BINARY_AXES
TRAP_AXIS = "trap"
NARRATE_KIND = "narrate"
UNSURE = "?"
BINARY_MODEL_THRESHOLD = 2

BINARY_KEY_TEXT = {
    "engagement_bait": {
        "0": "No manipulation cue (rubric 0, or only a mild/borderline hook = rubric 1).",
        "1": "Clear or dominant manipulation cue to harvest replies/reposts/likes/follows (rubric 2-3).",
    },
    "spam": {
        "0": "No spam, or only borderline self-promotion without deceptive/repetitive cues (rubric 0-1).",
        "1": "Clear unsolicited commercial solicitation, scam, phishing or template spam (rubric 2-3).",
    },
}


def load_rubric(path: Path = RUBRIC_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def allowed_values(axis: str) -> tuple[Any, ...]:
    if axis in ORDINAL_AXES:
        return (0, 1, 2, 3, UNSURE)
    if axis in BINARY_AXES:
        return (0, 1, UNSURE)
    if axis == TRAP_AXIS:
        return (0, 1, 2, UNSURE)
    raise ValueError(f"Unknown axis {axis!r}")


def axis_spec(axis: str, rubric: dict[str, Any]) -> dict[str, Any]:
    """Definition, key legend and boundary rules shown on screen for one pass."""
    if axis == TRAP_AXIS:
        return {
            "axis": axis,
            "definition": "Which community is this post actually about or speaking from?",
            "keys": [
                {"key": "1", "value": 1, "text": "Candidate 1"},
                {"key": "2", "value": 2, "text": "Candidate 2"},
                {"key": "0", "value": 0, "text": "Neither"},
                {"key": "?", "value": UNSURE, "text": "Can't tell"},
            ],
            "boundary_rules": [],
        }
    spec = rubric["components"][axis]
    anchors = spec["anchors"]
    if axis in BINARY_AXES:
        keys = [{"key": k, "value": int(k), "text": BINARY_KEY_TEXT[axis][k]} for k in ("0", "1")]
    else:
        keys = [{"key": k, "value": int(k), "text": anchors[k]} for k in ("0", "1", "2", "3")]
    keys.append({"key": "?", "value": UNSURE, "text": rubric["ordinal_scale"]["insufficient"]})
    return {
        "axis": axis,
        "definition": spec["definition"],
        "keys": keys,
        "anchors": anchors,
        "boundary_rules": spec.get("boundary_rules", []),
    }


def normalize_value(axis: str, value: Any) -> Any:
    """Coerce a submitted value ("2", 2, "?") to its stored form or raise ValueError."""
    if value == UNSURE:
        return UNSURE
    if isinstance(value, bool):
        raise ValueError(f"Invalid value {value!r} for {axis}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value {value!r} for {axis}") from None
    if str(number) != str(value).strip() or number not in allowed_values(axis):
        raise ValueError(f"Invalid value {value!r} for {axis}")
    return number
