"""Durable, format-independent checkpoints for the single-writer student run.

``serializer(payload, path)`` must finish and close its writes before returning;
``loader(path)`` returns a mapping containing ``step`` and ``config``. Torch can
be supplied by the caller, but this module and its tests use only the stdlib.

A checkpoint becomes recoverable only after its integrity sidecar is durable.
Interrupted files are retained and moved into ``recovery/`` on the next resume.
The directory must support fsync and atomic rename on the same filesystem.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "catbird-checkpoint-integrity-v1"
_NAME = re.compile(r"checkpoint-(\d+)\.pt")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class CheckpointError(ValueError):
    """A checkpoint cannot safely be used."""


class ConfigurationMismatchError(CheckpointError):
    """An intact checkpoint belongs to a different run configuration."""


class NoValidCheckpointError(CheckpointError):
    """No verified checkpoint is available; ``info`` describes preserved files."""

    def __init__(self, message: str, info: dict[str, Any]):
        super().__init__(message)
        self.info = info


@dataclass(frozen=True)
class RecoveryResult:
    path: Path
    payload: Mapping[str, Any]
    step: int
    info: dict[str, Any]


def config_sha256(config: Any) -> str:
    """Hash JSON configuration identity without whitespace or key-order drift."""
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def integrity_path(path: str | Path) -> Path:
    return Path(str(path) + ".integrity.json")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".partial",
    )
    os.close(descriptor)
    return Path(name)


def _atomic_write(destination: Path, contents: bytes) -> None:
    temporary = _temporary(destination)
    with temporary.open("wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def atomic_write(path: str | Path, contents: bytes, *, overwrite: bool = True) -> None:
    """Publish bytes after file fsync, then fsync the containing directory.

    Requires one writer and atomic same-filesystem rename. Failed writes leave
    their hidden ``.partial`` file for explicit recovery; they are not deleted.
    This local-filesystem contract does not qualify mounted Drive behavior.
    """
    destination = Path(path)
    if destination.is_symlink() or (not overwrite and destination.exists()):
        raise FileExistsError(f"Refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, contents)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_step(step: Any) -> None:
    if type(step) is not int or step < 0:
        raise CheckpointError("Checkpoint step must be a nonnegative integer")


def _payload_identity(payload: Any, step: int, expected_config_hash: str) -> None:
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint payload must be a mapping")
    if type(payload.get("step")) is not int or payload["step"] != step:
        raise CheckpointError("Checkpoint payload and integrity step differ")
    if "config" not in payload:
        raise CheckpointError("Checkpoint payload has no configuration identity")
    if config_sha256(payload["config"]) != expected_config_hash:
        raise ConfigurationMismatchError("Checkpoint payload configuration differs")


def save_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    step: int,
    config: Any,
    serializer: Callable[[Mapping[str, Any], Path], Any],
) -> dict[str, Any]:
    """Publish new checkpoint bytes, then their integrity commit marker.

    Existing checkpoint or sidecar paths are never overwritten. Failures leave
    partial artifacts in place so recovery can archive them without data loss.
    ``path`` must use the run's ``checkpoint-<step>.pt`` naming convention.
    """
    path = Path(path)
    _validate_step(step)
    match = _NAME.fullmatch(path.name)
    if match is None or int(match.group(1)) != step:
        raise CheckpointError("Checkpoint filename and step differ")
    expected_hash = config_sha256(config)
    _payload_identity(payload, step, expected_hash)
    sidecar = integrity_path(path)
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint artifacts: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    serializer(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    size, digest = _fingerprint(temporary)
    metadata = {
        "schema": SCHEMA,
        "checkpoint": path.name,
        "step": step,
        "size_bytes": size,
        "sha256": digest,
        "config_sha256": expected_hash,
    }
    # Single writer: publishing the marker last prevents loading a torn write.
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    _atomic_write(sidecar, _json_bytes(metadata))
    return metadata


def _verify(path: Path, expected_config_hash: str) -> dict[str, Any]:
    sidecar = integrity_path(path)
    if path.is_symlink() or sidecar.is_symlink():
        raise CheckpointError("Checkpoint artifacts must not be symlinks")
    if not path.is_file() or not sidecar.is_file():
        raise CheckpointError("Checkpoint bytes or integrity sidecar are missing")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CheckpointError("Unreadable checkpoint integrity sidecar") from error
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise CheckpointError("Unsupported checkpoint integrity schema")
    step = metadata.get("step")
    _validate_step(step)
    match = _NAME.fullmatch(path.name)
    if match is None or step != int(match.group(1)) or metadata.get("checkpoint") != path.name:
        raise CheckpointError("Checkpoint filename and integrity identity differ")
    if type(metadata.get("size_bytes")) is not int or metadata["size_bytes"] < 0:
        raise CheckpointError("Invalid checkpoint size in sidecar")
    if any(
        not isinstance(metadata.get(key), str) or not _DIGEST.fullmatch(metadata[key])
        for key in ("sha256", "config_sha256")
    ):
        raise CheckpointError("Invalid checkpoint digest in sidecar")
    size, digest = _fingerprint(path)
    if size != metadata["size_bytes"] or digest != metadata["sha256"]:
        raise CheckpointError("Checkpoint size or SHA256 does not match sidecar")
    # Config identity is authoritative only after the checkpoint is verified.
    if metadata["config_sha256"] != expected_config_hash:
        raise ConfigurationMismatchError(f"Intact checkpoint configuration differs: {path.name}")
    return metadata


def _read_steps(path: Path, restored_step: int) -> tuple[bytes, dict[str, Any]]:
    """Keep a chronological, contiguous prefix; never fabricate loss records."""
    records: list[dict[str, Any]] = []
    seen: dict[int, dict[str, Any]] = {}
    duplicates = 0
    stopped_reason = None
    if path.is_symlink():
        raise CheckpointError("Step log must not be a symlink")
    if path.exists():
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    stopped_reason = f"invalid_json_at_line_{line_number}"
                    break
                step = record.get("step") if isinstance(record, dict) else None
                if type(step) is not int or step < 1:
                    stopped_reason = f"invalid_step_at_line_{line_number}"
                    break
                if step > restored_step:
                    stopped_reason = f"after_restored_step_at_line_{line_number}"
                    break
                if step in seen:
                    if record != seen[step]:
                        stopped_reason = f"conflicting_duplicate_at_line_{line_number}"
                        break
                    duplicates += 1
                    continue
                if step != len(records) + 1:
                    stopped_reason = f"noncontiguous_step_at_line_{line_number}"
                    break
                records.append(record)
                seen[step] = record
    encoded = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
    ).encode("utf-8")
    return encoded, {
        "log_last_step": len(records),
        "log_complete_through_checkpoint": len(records) == restored_step,
        "log_duplicate_records_removed": duplicates,
        "log_stopped_reason": stopped_reason,
        "log_existed": path.exists(),
    }


def _archive_copy(source: Path, destination: Path) -> None:
    """Durably preserve a log before replacing it, without a missing-log window."""
    temporary = _temporary(destination)
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def recover_checkpoint(
    directory: str | Path,
    *,
    config: Any,
    loader: Callable[[Path], Mapping[str, Any]],
    requested: str | Path | None = None,
    steps_path: str | Path | None = None,
) -> RecoveryResult:
    """Restore the newest verified checkpoint and repair its append-only log.

    ``requested`` identifies the user's resume file; it may be damaged or absent,
    but must belong to this directory. The newest valid checkpoint always wins,
    including when the request names an older file. An intact requested file or
    inspected candidate with a different configuration is a hard failure before
    any mutation. Unverified legacy checkpoints are never loaded.

    Invalid candidates newer than the selected checkpoint, orphan sidecars, and
    partial writes are moved to a unique recovery directory. The original log is
    copied there before atomic replacement with its contiguous prefix through
    the restored step. Missing historical metrics are reported, never invented.
    Recovery requires exclusive access to the run directory.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    expected_hash = config_sha256(config)
    steps = Path(steps_path) if steps_path is not None else directory / "steps.jsonl"
    if steps.parent.resolve() != directory or steps.name != "steps.jsonl":
        raise CheckpointError("Step log must be steps.jsonl in the checkpoint directory")
    requested_path = None
    if requested is not None:
        requested_path = Path(requested)
        if requested_path.parent.resolve() != directory or not _NAME.fullmatch(requested_path.name):
            raise CheckpointError("Requested checkpoint must belong to the run directory")
        requested_path = directory / requested_path.name

    candidates = {path for path in directory.glob("checkpoint-*.pt") if _NAME.fullmatch(path.name)}
    if requested_path is not None:
        candidates.add(requested_path)
    invalid: dict[Path, str] = {}
    verified: dict[Path, dict[str, Any]] = {}
    # Inspect explicit identity even when a newer matching checkpoint exists.
    if requested_path is not None:
        try:
            verified[requested_path] = _verify(requested_path, expected_hash)
        except ConfigurationMismatchError:
            raise
        except (CheckpointError, OSError) as error:
            invalid[requested_path] = str(error)

    chosen = None
    payload = None
    metadata = None
    for candidate in sorted(candidates, key=lambda item: (int(_NAME.fullmatch(item.name).group(1)), item.name), reverse=True):
        if candidate in invalid:
            continue
        try:
            current = verified.get(candidate) or _verify(candidate, expected_hash)
            loaded = loader(candidate)
            _payload_identity(loaded, current["step"], expected_hash)
        except ConfigurationMismatchError:
            raise
        except Exception as error:
            invalid[candidate] = f"{type(error).__name__}: {error}"
            continue
        chosen, payload, metadata = candidate, loaded, current
        break

    artifacts: dict[Path, str] = {}
    for path, reason in invalid.items():
        for artifact in (path, integrity_path(path)):
            if artifact.exists() or artifact.is_symlink():
                artifacts[artifact] = reason
    for sidecar in directory.glob("checkpoint-*.pt.integrity.json"):
        checkpoint = Path(str(sidecar).removesuffix(".integrity.json"))
        if _NAME.fullmatch(checkpoint.name) and not checkpoint.exists():
            artifacts[sidecar] = "Orphan integrity sidecar"
    for partial in directory.glob(".*.partial"):
        if partial.name.startswith((".checkpoint-", ".steps.jsonl.")):
            artifacts[partial] = "Interrupted atomic write"

    info: dict[str, Any] = {
        "schema": "catbird-checkpoint-recovery-v1",
        "requested": str(requested_path) if requested_path is not None else None,
        "selected": str(chosen) if chosen is not None else None,
        "restored_step": metadata["step"] if metadata is not None else None,
        "config_sha256": expected_hash,
        "fell_back": chosen is not None and any(
            int(_NAME.fullmatch(path.name).group(1)) > metadata["step"] for path in invalid
        ),
        "invalid_checkpoints": [
            {"path": str(path), "reason": reason} for path, reason in invalid.items()
        ],
        "archived": [],
        "recovery_directory": None,
    }
    # Read/validate everything that can refuse recovery before changing files.
    repaired_log = None
    if chosen is not None:
        repaired_log, log_info = _read_steps(steps, metadata["step"])
        info.update(log_info)
    if artifacts or chosen is not None:
        recovery_root = directory / "recovery"
        recovery_root.mkdir(exist_ok=True)
        recovery = recovery_root / f"{time.time_ns()}-{uuid.uuid4().hex[:12]}"
        recovery.mkdir()
        _fsync_directory(recovery_root)
        _fsync_directory(directory)
        info["recovery_directory"] = str(recovery)
        for source, reason in sorted(artifacts.items()):
            target = recovery / source.name
            os.replace(source, target)
            _fsync_directory(recovery)
            _fsync_directory(directory)
            info["archived"].append({"source": str(source), "path": str(target), "reason": reason})
        if chosen is not None:
            if steps.exists():
                archived_log = recovery / steps.name
                _archive_copy(steps, archived_log)
                info["archived"].append({
                    "source": str(steps), "path": str(archived_log), "reason": "Original step log",
                })
            _atomic_write(steps, repaired_log)
        _atomic_write(recovery / "recovery.json", _json_bytes(info))
    if chosen is None:
        raise NoValidCheckpointError("No checkpoint has verified integrity and matching configuration", info)
    return RecoveryResult(chosen, payload, metadata["step"], info)
