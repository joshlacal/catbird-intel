"""Local human gold-labeling tool for Catbird timeline-health axes.

Gold labels are Josh's keystrokes only: nothing in this package reads model
output into a labeling session. See
docs/execution-plans/2026-10-23-human-gold-labels-and-claude-teacher-handoff.md.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
INTEL_DIR = PACKAGE_DIR.parent
SESSIONS_DIR = PACKAGE_DIR / "sessions"
POOLS_DIR = PACKAGE_DIR / "pools"
# Voice audio and transcripts stay on this Mac and out of version control.
LOCAL_DIR = PACKAGE_DIR / "local"
