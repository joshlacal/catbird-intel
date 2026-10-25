"""Fully local speech-to-text for the labeling tool.

Wraps the on-device Swift CLI in ``labeling/transcriber`` (Apple SpeechTranscriber,
falling back to SFSpeechRecognizer with on-device recognition required). Audio and
transcripts never leave the Mac: ffmpeg runs locally, and the binary refuses any
recognition path that could use Apple's servers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

TRANSCRIBER_DIR = Path(__file__).resolve().parent / "transcriber"
BINARY_PATH = TRANSCRIBER_DIR / ".build" / "transcribe"
BUILD_COMMAND = f"{TRANSCRIBER_DIR / 'build.sh'}"
FFMPEG_CANDIDATES = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
DEFAULT_TIMEOUT_SECONDS = 120.0


class TranscriptionUnavailable(RuntimeError):
    """Raised when local transcription cannot run (missing binary/ffmpeg, engine error)."""


def _ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        if os.access(candidate, os.X_OK):
            return candidate
    raise TranscriptionUnavailable("ffmpeg not found (expected at /opt/homebrew/bin/ffmpeg)")


def _is_16k_mono_pcm_wav(path: Path) -> bool:
    """Cheap RIFF header check: PCM (format 1), 1 channel, 16000 Hz."""
    try:
        with path.open("rb") as handle:
            header = handle.read(36)
    except OSError:
        return False
    if len(header) < 36 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        return False
    if header[12:16] != b"fmt ":
        return False
    audio_format = int.from_bytes(header[20:22], "little")
    channels = int.from_bytes(header[22:24], "little")
    sample_rate = int.from_bytes(header[24:28], "little")
    return audio_format == 1 and channels == 1 and sample_rate == 16000


def to_wav(src: Path, dst: Path) -> None:
    """Convert any ffmpeg-readable audio (webm/opus, mp4/aac, ...) to 16 kHz mono PCM WAV."""
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(dst),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise TranscriptionUnavailable(
            f"ffmpeg failed converting {src.name}: {completed.stderr.strip()[:500]}"
        )


def _run_binary(wav_path: Path, binary: Path, timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(binary), str(wav_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionUnavailable(f"transcriber timed out after {timeout:.0f}s") from exc
    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except json.JSONDecodeError:
        payload = {}
    if completed.returncode != 0 or "error" in payload or "text" not in payload:
        detail = payload.get("error") or completed.stderr.strip()[:500] or stdout[:500]
        raise TranscriptionUnavailable(
            f"transcriber failed (exit {completed.returncode}): {detail}"
        )
    return payload


def transcribe_file(
    audio_path: Path,
    *,
    binary: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Transcribe ``audio_path`` on-device.

    Returns ``{"text", "engine", "locale", "ms"}`` as emitted by the Swift CLI. Raises
    ``TranscriptionUnavailable`` if the binary is not built (never auto-built), ffmpeg is
    missing, or the engine refuses/fails.
    """
    audio_path = Path(audio_path)
    binary = Path(binary) if binary is not None else BINARY_PATH
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        raise TranscriptionUnavailable(
            f"transcriber binary not built at {binary}; build it with: {BUILD_COMMAND}"
        )
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    if _is_16k_mono_pcm_wav(audio_path):
        return _run_binary(audio_path, binary, timeout)
    with tempfile.TemporaryDirectory(prefix="labeling-stt-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        to_wav(audio_path, wav_path)
        return _run_binary(wav_path, binary, timeout)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        print(json.dumps(transcribe_file(Path(arg))))
