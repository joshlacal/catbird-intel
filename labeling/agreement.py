"""Agreement metrics between human gold labels and model (Claude teacher) labels.

Stdlib only. Covers:

* loading append-only ``labels-<axis>-<session_id>.jsonl`` gold files as
  written by ``labeling/store.py`` (undo resolution, cross-session
  latest-``ts`` resolution, intra-rater repeat consistency; every row is
  keyed by ``key`` -- the post uri, or the pair_id for trap sessions;
  narration (``axis == "narrate"``) files and ``view`` rows are not labels
  and are ignored);
* Cohen's kappa (unweighted, linear, quadratic) and confusion matrices;
* per-axis comparison with ``?`` / ``insufficient`` exclusion, binary
  thresholding and per-stratum breakdown;
* a deterministic item-resampling bootstrap CI for kappa.

Kappa degenerate-case conventions (see :func:`cohen_kappa`):

* no pairs -> ``nan``;
* expected disagreement == 0 (expected agreement == 1, i.e. both raters put
  every item in one and the same category) -> ``1.0`` when the two raters are
  identical on every item, otherwise ``nan``. This differs from sklearn,
  which returns ``nan`` for the identical single-category case (0/0). For
  valid weights the non-identical branch cannot occur, it is kept as a guard.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Hashable, Iterable, Mapping, Sequence

GOLD_SCHEMA = "catbird-gold-label-v1"

ORDINAL_AXES = ("politicalness", "outraged", "critical")
BINARY_AXES = ("engagement_bait", "spam")
TRAP_AXIS = "trap"
NARRATE_AXIS = "narrate"
ALL_AXES = ORDINAL_AXES + BINARY_AXES

GOLD_UNSURE = "?"
MODEL_INSUFFICIENT = "insufficient"

ORDINAL_CATEGORIES = (0, 1, 2, 3)
BINARY_CATEGORIES = (0, 1)
TRAP_CATEGORIES = (0, 1, 2)

DEFAULT_BINARY_THRESHOLD = 2
UNSTRATIFIED = "unstratified"

KappaWeights = str | None  # None | "linear" | "quadratic"


# ---------------------------------------------------------------------------
# Gold loading
# ---------------------------------------------------------------------------


class GoldFormatError(ValueError):
    """Raised when a gold label file violates the catbird-gold-label-v1 contract."""


@dataclass(frozen=True)
class GoldLabel:
    """One effective gold label after undo and cross-session resolution."""

    key: str  # uri for content axes, pair_id for the trap axis
    axis: str
    value: int | str
    session_id: str
    ts: str
    row_id: int
    path: str


@dataclass(frozen=True)
class RepeatSummary:
    """Intra-rater consistency for one axis: keys labelled in 2+ sessions."""

    n_repeated: int
    n_disagreed: int
    disagreed_keys: tuple[str, ...]


@dataclass
class GoldSet:
    """Resolved gold labels plus the receipts needed to reproduce them."""

    labels: dict[str, dict[str, GoldLabel]] = field(default_factory=dict)
    repeats: dict[str, RepeatSummary] = field(default_factory=dict)
    file_sha256: dict[str, str] = field(default_factory=dict)
    ignored_files: list[str] = field(default_factory=list)  # narration sessions

    def values(self, axis: str) -> dict[str, int | str]:
        """Return ``{key: value}`` for one axis (empty if the axis is absent)."""
        return {key: label.value for key, label in self.labels.get(axis, {}).items()}


def allowed_gold_values(axis: str) -> tuple[int | str, ...]:
    """Valid gold values for an axis (including the ``?`` unsure marker)."""
    if axis in ORDINAL_AXES:
        return ORDINAL_CATEGORIES + (GOLD_UNSURE,)
    if axis in BINARY_AXES:
        return BINARY_CATEGORIES + (GOLD_UNSURE,)
    if axis == TRAP_AXIS:
        return TRAP_CATEGORIES + (GOLD_UNSURE,)
    raise GoldFormatError(f"unknown axis {axis!r}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_ts(ts: Any) -> datetime:
    if not isinstance(ts, str):
        raise GoldFormatError(f"ts must be an ISO8601 string, got {ts!r}")
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GoldFormatError(f"unparseable ts {ts!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                raise GoldFormatError(f"{path}:{line_no}: torn write (no trailing newline)")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldFormatError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise GoldFormatError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def _row_key(row: Mapping[str, Any], where: str) -> str:
    key = row.get("key")
    if not isinstance(key, str) or not key:
        raise GoldFormatError(f"{where}: missing key")
    if row.get("axis") == TRAP_AXIS and row.get("op") == "label" and row.get("pair_id", key) != key:
        raise GoldFormatError(f"{where}: trap key {key!r} != pair_id {row.get('pair_id')!r}")
    return key


def resolve_session_file(path: str | Path) -> tuple[str, str, dict[str, GoldLabel]]:
    """Resolve one session file to ``(axis, session_id, {key: GoldLabel})``.

    Rows are resolved by ``key``. The effective label for a key is the last
    ``label`` row not revoked by an ``undo`` row. Undo rows must reference an
    earlier ``label`` or ``view`` row in the same file with the same key.
    ``view`` rows (narration) are never labels. A narration file
    (``axis == "narrate"``) is validated structurally and returns no labels.
    """
    path = Path(path)
    rows = _read_jsonl(path)
    labels_by_row: dict[int, dict[str, Any]] = {}
    undo_targets: dict[int, str] = {}  # label and view rows: row_id -> key
    revoked: set[int] = set()
    last_row_id: int | None = None
    axis: str | None = None
    session_id: str | None = None

    for row in rows:
        row_id = row.get("row_id")
        where = f"{path}: row_id={row_id!r}"
        if row.get("schema") != GOLD_SCHEMA:
            raise GoldFormatError(f"{where}: schema must be {GOLD_SCHEMA!r}")
        if not _is_int(row_id):
            raise GoldFormatError(f"{where}: row_id must be an int")
        if last_row_id is not None and row_id <= last_row_id:
            raise GoldFormatError(f"{where}: row_id not strictly increasing")
        last_row_id = row_id

        row_axis = row.get("axis")
        if not isinstance(row_axis, str):
            raise GoldFormatError(f"{where}: missing axis")
        allowed = () if row_axis == NARRATE_AXIS else allowed_gold_values(row_axis)
        if axis is None:
            axis = row_axis
        elif row_axis != axis:
            raise GoldFormatError(f"{where}: mixed axes {axis!r} and {row_axis!r} in one file")
        row_session = row.get("session_id")
        if not isinstance(row_session, str) or not row_session:
            raise GoldFormatError(f"{where}: missing session_id")
        if session_id is None:
            session_id = row_session
        elif row_session != session_id:
            raise GoldFormatError(f"{where}: mixed session_ids in one file")

        op = row.get("op")
        key = _row_key(row, where)
        if op == "label":
            if row_axis == NARRATE_AXIS:
                raise GoldFormatError(f"{where}: narration sessions cannot contain label rows")
            value = row.get("value")
            if isinstance(value, bool) or value not in allowed:
                raise GoldFormatError(f"{where}: value {value!r} not allowed for axis {row_axis!r}")
            _parse_ts(row.get("ts"))
            labels_by_row[row_id] = row
            undo_targets[row_id] = key
        elif op == "view":
            undo_targets[row_id] = key
        elif op == "undo":
            target = row.get("undoes_row_id")
            if not _is_int(target) or target not in undo_targets:
                raise GoldFormatError(f"{where}: undo references unknown row {target!r}")
            if undo_targets[target] != key:
                raise GoldFormatError(f"{where}: undo key {key!r} != undone row key {undo_targets[target]!r}")
            if target in revoked:
                raise GoldFormatError(f"{where}: row {target} already undone")
            revoked.add(target)
        else:
            raise GoldFormatError(f"{where}: unknown op {op!r}")

    if axis is None or session_id is None:
        raise GoldFormatError(f"{path}: no rows")

    effective: dict[str, GoldLabel] = {}
    for row_id in sorted(labels_by_row):
        if row_id in revoked:
            continue
        row = labels_by_row[row_id]
        key = _row_key(row, str(path))
        effective[key] = GoldLabel(
            key=key,
            axis=axis,
            value=row["value"],
            session_id=session_id,
            ts=row["ts"],
            row_id=row_id,
            path=str(path),
        )
    return axis, session_id, effective


def load_gold(paths: Iterable[str | Path]) -> GoldSet:
    """Load and resolve gold label files.

    Per (key, axis): within a session, last un-undone label wins; across
    sessions, the label with the latest ``ts`` wins (ties broken by
    session_id). Keys labelled in 2+ sessions are counted in ``repeats`` with
    the number whose session values are not all equal (``?`` counts as a
    value). Each label file's SHA-256 is recorded for receipts; narration
    files are skipped and listed in ``ignored_files``.
    """
    gold = GoldSet()
    per_axis: dict[str, dict[str, list[GoldLabel]]] = defaultdict(lambda: defaultdict(list))
    seen_sessions: dict[tuple[str, str], str] = {}

    for raw_path in sorted(str(p) for p in paths):
        axis, session_id, effective = resolve_session_file(raw_path)
        if axis == NARRATE_AXIS:
            gold.ignored_files.append(raw_path)
            continue
        gold.file_sha256[raw_path] = file_sha256(raw_path)
        previous = seen_sessions.get((axis, session_id))
        if previous is not None:
            raise GoldFormatError(
                f"{raw_path}: axis {axis!r} session {session_id!r} already loaded from {previous}"
            )
        seen_sessions[(axis, session_id)] = raw_path
        for key, label in effective.items():
            per_axis[axis][key].append(label)

    for axis in sorted(per_axis):
        resolved: dict[str, GoldLabel] = {}
        n_repeated = 0
        disagreed: list[str] = []
        for key in sorted(per_axis[axis]):
            candidates = per_axis[axis][key]
            resolved[key] = max(candidates, key=lambda lab: (_parse_ts(lab.ts), lab.session_id))
            if len(candidates) > 1:
                n_repeated += 1
                if len({repr(lab.value) for lab in candidates}) > 1:
                    disagreed.append(key)
        gold.labels[axis] = resolved
        gold.repeats[axis] = RepeatSummary(n_repeated, len(disagreed), tuple(disagreed))
    return gold


# ---------------------------------------------------------------------------
# Kappa and confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(
    pairs: Iterable[tuple[Hashable, Hashable]], categories: Sequence[Hashable]
) -> list[list[int]]:
    """Counts with rows = gold (first element), cols = model (second element)."""
    index = {cat: i for i, cat in enumerate(categories)}
    if len(index) != len(categories):
        raise ValueError("categories must be unique")
    matrix = [[0] * len(categories) for _ in categories]
    for gold_value, model_value in pairs:
        if gold_value not in index or model_value not in index:
            raise ValueError(f"pair {(gold_value, model_value)!r} outside categories {list(categories)!r}")
        matrix[index[gold_value]][index[model_value]] += 1
    return matrix


def disagreement_weights(k: int, weights: KappaWeights = None) -> list[list[float]]:
    """k x k disagreement weights: 0 on the diagonal, 1 at maximum distance."""
    if weights not in (None, "linear", "quadratic"):
        raise ValueError(f"weights must be None, 'linear' or 'quadratic', got {weights!r}")
    span = max(k - 1, 1)
    out: list[list[float]] = []
    for i in range(k):
        row: list[float] = []
        for j in range(k):
            if weights is None:
                row.append(0.0 if i == j else 1.0)
            elif weights == "linear":
                row.append(abs(i - j) / span)
            else:
                row.append(((i - j) / span) ** 2)
        out.append(row)
    return out


def kappa_from_matrix(matrix: Sequence[Sequence[int]], weights: KappaWeights = None) -> float:
    """Cohen's kappa from a square confusion matrix (see module docstring for edge cases)."""
    k = len(matrix)
    n = sum(sum(row) for row in matrix)
    if n == 0:
        return float("nan")
    w = disagreement_weights(k, weights)
    row_tot = [sum(matrix[i]) for i in range(k)]
    col_tot = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    observed = sum(w[i][j] * matrix[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w[i][j] * row_tot[i] * col_tot[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        identical = all(matrix[i][j] == 0 for i in range(k) for j in range(k) if i != j)
        return 1.0 if identical else float("nan")
    return 1.0 - observed / expected


def cohen_kappa(
    pairs: Iterable[tuple[Hashable, Hashable]],
    categories: Sequence[Hashable],
    weights: KappaWeights = None,
) -> float:
    """Cohen's kappa over (gold, model) pairs.

    ``categories`` fixes the ordering used for linear/quadratic weights, so
    pass ordinal categories in ascending order. Returns ``nan`` for no pairs;
    returns ``1.0`` when expected agreement is 1 and the raters are identical
    (sklearn would return ``nan`` there), ``nan`` otherwise in that case.
    """
    return kappa_from_matrix(confusion_matrix(pairs, categories), weights)


def exact_agreement(pairs: Sequence[tuple[Hashable, Hashable]]) -> float:
    """Fraction of pairs where gold == model; ``nan`` for no pairs."""
    if not pairs:
        return float("nan")
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default 'linear' method)."""
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_kappa_ci(
    pairs: Sequence[tuple[Hashable, Hashable]],
    categories: Sequence[Hashable],
    weights: KappaWeights = None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap CI for kappa, resampling items with replacement.

    Deterministic for a given ``seed`` (``random.Random(seed)``). Resamples
    whose kappa is ``nan`` are dropped and counted in ``n_nan``.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    pairs = list(pairs)
    point = cohen_kappa(pairs, categories, weights)
    result: dict[str, Any] = {
        "kappa": point,
        "lower": float("nan"),
        "upper": float("nan"),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_nan": 0,
        "seed": seed,
    }
    if not pairs:
        result["n_nan"] = n_resamples
        return result
    rng = random.Random(seed)
    n = len(pairs)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        value = cohen_kappa(sample, categories, weights)
        if math.isnan(value):
            result["n_nan"] += 1
        else:
            stats.append(value)
    stats.sort()
    alpha = (1 - confidence) / 2
    result["lower"] = _percentile(stats, alpha)
    result["upper"] = _percentile(stats, 1 - alpha)
    return result


# ---------------------------------------------------------------------------
# Per-axis comparison
# ---------------------------------------------------------------------------


def axis_kind(axis: str) -> str:
    """``"ordinal"``, ``"binary"`` or ``"nominal"`` (trap)."""
    if axis in ORDINAL_AXES:
        return "ordinal"
    if axis in BINARY_AXES:
        return "binary"
    if axis == TRAP_AXIS:
        return "nominal"
    raise ValueError(f"unknown axis {axis!r}")


def binarize(score: int, threshold: int = DEFAULT_BINARY_THRESHOLD) -> int:
    """Map a 0-3 model score to 0/1: ``score >= threshold`` -> 1."""
    return 1 if score >= threshold else 0


def _check_model_value(value: Any, uri: str) -> None:
    if value == MODEL_INSUFFICIENT:
        return
    if not _is_int(value) or value not in ORDINAL_CATEGORIES:
        raise ValueError(f"model value {value!r} for {uri!r} must be 0-3 or {MODEL_INSUFFICIENT!r}")


def _metrics(pairs: list[tuple[int, int]], categories: Sequence[int], weights: KappaWeights) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "kappa": cohen_kappa(pairs, categories, weights),
        "exact_agreement": exact_agreement(pairs),
    }


def compare_axis(
    gold_by_uri: Mapping[str, int | str],
    model_by_uri: Mapping[str, int | str],
    axis: str,
    *,
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    strata_by_uri: Mapping[str, str] | None = None,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compare gold and model labels on one axis.

    The item universe is the gold keys. Model-only keys are counted in
    ``n_model_only`` and otherwise ignored. Items where gold is ``?`` or the
    model said ``insufficient`` are excluded from kappa/agreement and counted
    separately; ``unsure_cooccurrence`` is a 2x2 over items present on both
    sides. Kappa is quadratic-weighted for ordinal axes, unweighted for binary
    axes (model scores binarized with ``score >= binary_threshold``) and for
    the nominal trap axis. Set ``bootstrap_resamples > 0`` to add a CI.
    """
    kind = axis_kind(axis)
    if kind == "ordinal":
        categories: tuple[int, ...] = ORDINAL_CATEGORIES
        weights: KappaWeights = "quadratic"
    elif kind == "binary":
        categories, weights = BINARY_CATEGORIES, None
    else:
        categories, weights = TRAP_CATEGORIES, None
    allowed_gold = allowed_gold_values(axis)

    pairs: list[tuple[int, int]] = []
    pairs_by_stratum: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cooccurrence = {"both": 0, "gold_unsure_only": 0, "model_insufficient_only": 0, "neither": 0}
    n_gold_unsure = n_model_insufficient = n_missing_model = 0

    for uri in sorted(gold_by_uri):
        gold_value = gold_by_uri[uri]
        if isinstance(gold_value, bool) or gold_value not in allowed_gold:
            raise ValueError(f"gold value {gold_value!r} for {uri!r} not allowed on axis {axis!r}")
        if uri not in model_by_uri:
            n_missing_model += 1
            continue
        model_value = model_by_uri[uri]
        _check_model_value(model_value, uri)
        gold_unsure = gold_value == GOLD_UNSURE
        model_insufficient = model_value == MODEL_INSUFFICIENT
        n_gold_unsure += gold_unsure
        n_model_insufficient += model_insufficient
        if gold_unsure and model_insufficient:
            cooccurrence["both"] += 1
        elif gold_unsure:
            cooccurrence["gold_unsure_only"] += 1
        elif model_insufficient:
            cooccurrence["model_insufficient_only"] += 1
        else:
            cooccurrence["neither"] += 1
        if gold_unsure or model_insufficient:
            continue
        if kind == "binary":
            model_value = binarize(model_value, binary_threshold)
        elif model_value not in categories:
            raise ValueError(f"model value {model_value!r} for {uri!r} outside {categories!r}")
        pair = (gold_value, model_value)
        pairs.append(pair)
        stratum = UNSTRATIFIED if strata_by_uri is None else strata_by_uri.get(uri, UNSTRATIFIED)
        pairs_by_stratum[stratum].append(pair)

    result: dict[str, Any] = {
        "axis": axis,
        "kind": kind,
        "kappa_weights": weights or "unweighted",
        "categories": list(categories),
        "binary_threshold": binary_threshold if kind == "binary" else None,
        "n_gold": len(gold_by_uri),
        "n_compared": len(pairs),
        "n_gold_unsure": n_gold_unsure,
        "n_model_insufficient": n_model_insufficient,
        "n_missing_model": n_missing_model,
        "n_model_only": sum(1 for uri in model_by_uri if uri not in gold_by_uri),
        "kappa": cohen_kappa(pairs, categories, weights),
        "exact_agreement": exact_agreement(pairs),
        "confusion_matrix": confusion_matrix(pairs, categories),
        "unsure_cooccurrence": cooccurrence,
        "strata": {
            name: _metrics(stratum_pairs, categories, weights)
            for name, stratum_pairs in sorted(pairs_by_stratum.items())
        },
    }
    if bootstrap_resamples > 0:
        result["kappa_ci"] = bootstrap_kappa_ci(
            pairs, categories, weights, n_resamples=bootstrap_resamples, seed=bootstrap_seed
        )
    return result
