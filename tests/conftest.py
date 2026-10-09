"""Shared pytest helpers.

Sync-engine helpers are prefixed ``synth_`` (synthetic subtitles, fake transcriber,
speech-like audio) and exposed through the ``synth`` fixture.
"""

from __future__ import annotations

import shutil
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------------------
# synth_*: synthetic data for the sync engine
# --------------------------------------------------------------------------------------

SYNTH_SR = 16000

_SYNTH_COMMON = (
    "the and you to of is it that what this was have are with he she we they don't i'm "
    "your just know be for not my me do can there here you're it's that's would will about "
    "all get got if him her them were been how why who when a an in on at so but no yes"
).split()

_SYNTH_RARE = (
    "lighthouse emerald harbor violin cathedral thunder whisper marble lantern blizzard "
    "orchard compass falcon velvet glacier saddle meadow carnival puzzle ribbon anchor "
    "basement canyon dagger engine fortress garlic hammer island jacket kettle ladder "
    "mirror needle ocean pepper quarry rocket silver tunnel umbrella valley wagon "
    "yellow zebra brother sister doctor captain sergeant teacher lawyer pilot farmer "
    "london paris madrid boston dallas chicago denver monday tuesday friday sunday "
    "money murder secret danger problem promise husband daughter morning evening "
    "kitchen bedroom garden window station hospital prison church market office "
    "remember believe understand forget imagine promise destroy protect explain "
    "beautiful dangerous terrible wonderful strange careful quiet broken golden "
    "airplane bicycle chocolate diamond elephant festival guitar helmet iceberg jungle "
    "kangaroo leopard mountain notebook octopus penguin quilt raincoat sandwich tomato"
).split()


@dataclass
class SynthCue:
    start: float
    end: float
    text: str


def synth_script(
    duration: float, seed: int = 0, start: float = 5.0, long_gaps: bool = True
) -> list[SynthCue]:
    """English-like dialogue cues filling ``duration`` seconds."""
    rng = np.random.default_rng(seed)
    cues: list[SynthCue] = []
    t = start
    while True:
        n = int(rng.integers(3, 11))
        words = []
        for _ in range(n):
            pool = _SYNTH_RARE if rng.random() < 0.35 else _SYNTH_COMMON
            words.append(pool[int(rng.integers(len(pool)))])
        text = " ".join(words).capitalize() + rng.choice([".", "?", "!", "..."])
        if n > 6 and rng.random() < 0.5:
            cut = len(" ".join(words[: n // 2]))
            text = text[:cut] + "\n" + text[cut + 1 :]
        speech = len(text) / rng.uniform(13.0, 17.0)
        dur = min(7.0, speech + rng.uniform(0.3, 1.2))
        if t + dur > duration - 2:
            break
        cues.append(SynthCue(round(t, 3), round(t + dur, 3), text))
        gap = rng.uniform(0.1, 2.5)
        if long_gaps and rng.random() < 0.03:
            gap += rng.uniform(15, 45)
        t += dur + gap
    return cues


def synth_srt(cues: list[SynthCue]) -> str:
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{ts(c.start)} --> {ts(c.end)}\n{c.text}\n")
    return "\n".join(out) + "\n"


def synth_speech_spans(
    cues: list[SynthCue], transform: Callable[[float], float], seed: int = 1
) -> list[tuple[float, float, SynthCue]]:
    """Where each cue is actually spoken in the audio: starts at transform(cue.start)
    (+ small onset jitter), lasts ~len(text)/15 s (compressed to the cue)."""
    rng = np.random.default_rng(seed)
    spans = []
    for c in cues:
        a = transform(c.start)
        b = transform(c.end)
        onset = a + rng.uniform(0.0, 0.12)
        speech = min(len(c.text) / rng.uniform(13.0, 17.0), max(0.3, b - onset - 0.05))
        spans.append((onset, onset + speech, c))
    return spans


class SynthFakeTranscriber:
    """Emits the ground-truth words of the cues spoken inside the requested window.

    ``drop``: probability a word is missing; ``noise``: probability a word is replaced
    by a random one; ``jitter``: std-dev (s) of word timestamp noise; ``garbage``: emit
    unrelated words only (simulates wrong language / music).
    """

    def __init__(
        self,
        spans: list[tuple[float, float, SynthCue]],
        language: str = "en",
        drop: float = 0.0,
        noise: float = 0.0,
        jitter: float = 0.03,
        garbage: bool = False,
        seed: int = 2,
    ):
        self.spans = spans
        self.language = language
        self.drop = drop
        self.noise = noise
        self.jitter = jitter
        self.garbage = garbage
        self.rng = np.random.default_rng(seed)
        self.calls: list[tuple[float, float]] = []

    def detect_language(self, audio, sr):
        return self.language, 0.95

    def transcribe(self, audio, sr, language, *, start: float = 0.0):
        from vlcsubsync.transcribe import Word

        end = start + len(audio) / sr
        self.calls.append((start, end))
        words = []
        vocab = ["zorp", "blick", "quanta", "mirth", "plover", "snark", "frabjous"]
        for a, b, cue in self.spans:
            if b < start or a > end:
                continue
            toks = cue.text.replace("\n", " ").split()
            total = sum(len(t) + 1 for t in toks)
            pos = 0
            for tok in toks:
                ws = a + (b - a) * pos / total
                we = a + (b - a) * (pos + len(tok)) / total
                pos += len(tok) + 1
                if ws < start or we > end:
                    continue
                if self.garbage:
                    tok = vocab[int(self.rng.integers(len(vocab)))] + str(int(ws) % 97)
                elif self.rng.random() < self.drop:
                    continue
                elif self.rng.random() < self.noise:
                    tok = vocab[int(self.rng.integers(len(vocab)))]
                j = float(self.rng.normal(0, self.jitter))
                words.append(Word(ws + j - start, we + j - start, tok, 0.9))
        words.sort(key=lambda w: w.start)
        return words


_synth_vowel_cache: dict[int, np.ndarray] = {}


def synth_vowel(seconds: float = 3.0) -> np.ndarray:
    """A buzzy, formant-shaped, syllable-modulated tone that Silero VAD calls speech."""
    key = int(seconds * 1000)
    if key in _synth_vowel_cache:
        return _synth_vowel_cache[key]
    n = int(seconds * SYNTH_SR)
    t = np.arange(n) / SYNTH_SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = np.cumsum(2 * np.pi * f0 / SYNTH_SR)
    x = np.zeros(n)
    for k in range(1, 30):
        f = k * f0
        amp = sum(
            np.exp(-(((f - F) / bw) ** 2)) for F, bw in [(700, 130), (1200, 150), (2600, 250)]
        )
        x += amp * np.sin(k * ph) / k**0.3
    env = np.sqrt(0.5 * (1 - np.cos(2 * np.pi * 4 * t)))
    x = (x * env / np.abs(x).max() * 0.3).astype(np.float32)
    _synth_vowel_cache[key] = x
    return x


def synth_audio(spans, duration: float, seed: int = 3) -> np.ndarray:
    """Speech-like audio at ``spans`` over faint noise (16 kHz mono float32)."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(duration * SYNTH_SR)) * 0.003).astype(np.float32)
    v = synth_vowel()
    for a, b, *_ in spans:
        i, j = int(a * SYNTH_SR), int(b * SYNTH_SR)
        i, j = max(0, i), min(len(audio), j)
        k = 0
        while i + k < j:
            m = min(len(v), j - i - k)
            audio[i + k : i + k + m] += v[:m]
            k += m
    return audio


def synth_write_wav(path: Path, audio: np.ndarray) -> Path:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SYNTH_SR)
        w.writeframes(pcm.tobytes())
    return path


def synth_errors(pred_starts, cues: list[SynthCue], transform) -> np.ndarray:
    truth = np.array([transform(c.start) for c in cues])
    return np.abs(np.asarray(pred_starts) - truth)


class _SynthKit:
    SR = SYNTH_SR
    Cue = SynthCue
    script = staticmethod(synth_script)
    srt = staticmethod(synth_srt)
    speech_spans = staticmethod(synth_speech_spans)
    FakeTranscriber = SynthFakeTranscriber
    audio = staticmethod(synth_audio)
    write_wav = staticmethod(synth_write_wav)
    errors = staticmethod(synth_errors)
    ffmpeg = shutil.which("ffmpeg") or (
        str(Path.home() / "bin" / "ffmpeg") if (Path.home() / "bin" / "ffmpeg").exists() else None
    )


@pytest.fixture
def synth() -> type[_SynthKit]:
    return _SynthKit


# --------------------------------------------------------------------------------------
# daemon lifecycle: never renice / re-schedule the test process itself
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _keep_test_process_priority(monkeypatch):
    """``serve()`` lowers its own priority; in tests it would hit pytest's process."""
    monkeypatch.setenv("VLC_SUBSYNC_PRIORITY", "normal")
