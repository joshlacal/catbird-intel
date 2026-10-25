"""Tests for the on-device transcription helper (labeling.transcribe).

Run from catbird-intel/: ``python3 -m unittest labeling.tests.test_transcribe -v``
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from labeling.transcribe import (
    BINARY_PATH,
    TranscriptionUnavailable,
    _ffmpeg_path,
    _is_16k_mono_pcm_wav,
    to_wav,
    transcribe_file,
)


def _ffmpeg_available() -> bool:
    try:
        _ffmpeg_path()
    except TranscriptionUnavailable:
        return False
    return True


def _make_tone(path: Path, *, codec_args: list[str]) -> None:
    subprocess.run(
        [
            _ffmpeg_path(),
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-ac",
            "2",
            *codec_args,
            str(path),
        ],
        check=True,
    )


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not installed")
class ToWavTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_16k_mono(self, wav_path: Path) -> None:
        self.assertTrue(_is_16k_mono_pcm_wav(wav_path))
        with wave.open(str(wav_path), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getframerate(), 16000)
            self.assertEqual(reader.getsampwidth(), 2)
            self.assertGreater(reader.getnframes(), 15000)

    def test_webm_opus_to_wav(self) -> None:
        src = self.tmp / "in.webm"
        _make_tone(src, codec_args=["-c:a", "libopus"])
        dst = self.tmp / "out.wav"
        to_wav(src, dst)
        self._assert_16k_mono(dst)

    def test_mp4_aac_to_wav(self) -> None:
        src = self.tmp / "in.m4a"
        _make_tone(src, codec_args=["-c:a", "aac"])
        dst = self.tmp / "out.wav"
        to_wav(src, dst)
        self._assert_16k_mono(dst)

    def test_stereo_48k_wav_is_not_treated_as_ready(self) -> None:
        src = self.tmp / "in.wav"
        _make_tone(src, codec_args=["-c:a", "pcm_s16le"])
        self.assertFalse(_is_16k_mono_pcm_wav(src))

    def test_garbage_input_raises(self) -> None:
        src = self.tmp / "junk.webm"
        src.write_bytes(b"not audio at all")
        with self.assertRaises(TranscriptionUnavailable):
            to_wav(src, self.tmp / "out.wav")

    def test_missing_input_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            to_wav(self.tmp / "missing.webm", self.tmp / "out.wav")


class UnavailableBinaryTests(unittest.TestCase):
    def test_missing_binary_raises_with_build_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"")
            with self.assertRaises(TranscriptionUnavailable) as ctx:
                transcribe_file(audio, binary=Path(tmp) / "nope" / "transcribe")
        self.assertIn("build.sh", str(ctx.exception))

    def test_binary_error_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "transcribe"
            fake.write_text('#!/bin/sh\necho \'{"error": "refusing: test"}\'\nexit 4\n')
            fake.chmod(0o755)
            audio = Path(tmp) / "a.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\x00\x00" * 1600)
            with self.assertRaises(TranscriptionUnavailable) as ctx:
                transcribe_file(audio, binary=fake)
        self.assertIn("refusing: test", str(ctx.exception))


@unittest.skipUnless(BINARY_PATH.is_file(), "transcriber binary not built")
@unittest.skipUnless(shutil.which("say") and _ffmpeg_available(), "needs say + ffmpeg")
class RealTranscriptionTests(unittest.TestCase):
    def test_spoken_phrase_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "fox.wav"
            subprocess.run(
                ["say", "-o", str(wav), "--data-format=LEI16@16000", "the quick brown fox"],
                check=True,
            )
            webm = Path(tmp) / "fox.webm"
            subprocess.run(
                [_ffmpeg_path(), "-nostdin", "-loglevel", "error", "-y", "-i", str(wav),
                 "-c:a", "libopus", str(webm)],
                check=True,
            )
            for audio in (wav, webm):
                result = transcribe_file(audio)
                self.assertIn(
                    result["engine"], {"SpeechTranscriber", "SFSpeechRecognizer-ondevice"}
                )
                self.assertEqual(result["locale"], "en-US")
                self.assertIsInstance(result["ms"], int)
                self.assertIn("quick brown fox", result["text"].lower())


if __name__ == "__main__":
    unittest.main()
