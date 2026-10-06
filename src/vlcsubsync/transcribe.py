"""faster-whisper wrapper behind a small :class:`Transcriber` protocol.

Models are loaded lazily and cached per ``(model, device, compute_type, threads)`` so
the daemon keeps them warm between jobs (:func:`get_transcriber`).

``device="auto"`` tries CUDA first and silently falls back to CPU on *any* CUDA error,
including errors raised during the first transcription (missing cuDNN/cuBLAS often
only surfaces then).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    start: float  # seconds, relative to the audio passed to transcribe()
    end: float
    text: str
    prob: float = 1.0


@runtime_checkable
class Transcriber(Protocol):
    """Anything that turns 16 kHz mono float32 audio into timed words."""

    def transcribe(
        self, audio: np.ndarray, sr: int, language: str | None, *, start: float = 0.0
    ) -> list[Word]:
        """Words with times relative to the start of ``audio``.

        ``start`` is the absolute position of ``audio`` in the media (informational;
        real implementations ignore it, test fakes use it).
        """
        ...

    def detect_language(self, audio: np.ndarray, sr: int) -> tuple[str | None, float]:
        """``(ISO 639-1 code or None, probability)`` of the spoken language."""
        ...


_CPU_COMPUTE_TYPES = {"int8", "int8_float32", "int16", "float32", "default", "auto"}


def _cpu_compute_type(compute_type: str) -> str:
    """CUDA-only compute types (float16, int8_float16, bfloat16, ...) → int8 on CPU."""
    return compute_type if compute_type in _CPU_COMPUTE_TYPES else "int8"


def is_english_only(model_name: str) -> bool:
    return model_name.endswith(".en")


def model_for_language(config: Config, language: str | None) -> str:
    """English (or unknown→None handled by caller) → ``model_en``; else ``model_multi``."""
    return config.model_en if language == "en" else config.model_multi


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def default_threads() -> int:
    return max(1, min(8, os.cpu_count() or 4))


class WhisperTranscriber:
    """faster-whisper model with lazy loading and CUDA→CPU fallback."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "int8",
        threads: int = 0,
        download_root: str | None = None,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.compute_type = compute_type
        self.threads = threads or default_threads()
        self.download_root = download_root
        self.device: str | None = None  # actual device once loaded
        self._model = None
        self._lock = threading.Lock()
        self._verified = False  # a transcription succeeded on the current device

    @property
    def multilingual(self) -> bool:
        return not is_english_only(self.model_name)

    # -- loading -----------------------------------------------------------------------
    def _candidates(self) -> list[tuple[str, str]]:
        if self.requested_device == "cpu":
            return [("cpu", self.compute_type)]
        cuda_ct = "int8_float16" if self.compute_type == "int8" else self.compute_type
        if self.requested_device == "cuda":
            return [("cuda", cuda_ct)]
        out = []
        if _cuda_available():
            out.append(("cuda", cuda_ct))
        out.append(("cpu", _cpu_compute_type(self.compute_type)))
        return out

    def _load(self, skip_cuda: bool = False):
        from faster_whisper import WhisperModel

        last: Exception | None = None
        for device, ct in self._candidates():
            if skip_cuda and device == "cuda":
                continue
            try:
                model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=ct,
                    cpu_threads=self.threads,
                    download_root=self.download_root,
                )
            except Exception as e:
                log.warning("loading %s on %s failed: %s", self.model_name, device, e)
                last = e
                continue
            self._model = model
            self.device = device
            self._verified = False
            log.info("loaded whisper model %s on %s (%s)", self.model_name, device, ct)
            return model
        raise RuntimeError(f"cannot load whisper model {self.model_name}: {last}")

    def load(self):
        with self._lock:
            return self._model or self._load()

    def _run(self, fn):
        """Run ``fn(model)``; on a CUDA failure in auto mode, reload on CPU and retry."""
        with self._lock:
            model = self._model or self._load()
            try:
                result = fn(model)
                self._verified = True
                return result
            except Exception as e:
                # Only a model that has never worked on CUDA falls back: missing
                # cuDNN/cuBLAS surfaces on first use. Once CUDA has produced a result,
                # later errors (bad input, transient OOM) are real errors.
                if self.device != "cuda" or self.requested_device != "auto" or self._verified:
                    raise
                log.warning("whisper on CUDA failed (%s); falling back to CPU", e)
                self._model = None
                model = self._load(skip_cuda=True)
                result = fn(model)
                self._verified = True
                return result

    # -- Transcriber API ---------------------------------------------------------------
    def transcribe(
        self, audio: np.ndarray, sr: int, language: str | None, *, start: float = 0.0
    ) -> list[Word]:
        if sr != 16000:
            raise ValueError("WhisperTranscriber expects 16 kHz audio")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        lang = "en" if not self.multilingual else language

        def run(model) -> list[Word]:
            segments, _info = model.transcribe(
                audio,
                language=lang,
                task="transcribe",
                beam_size=5 if self.device == "cuda" else 1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            words: list[Word] = []
            for seg in segments:  # generator: decoding happens here
                for w in seg.words or ():
                    text = w.word.strip()
                    if text:
                        words.append(Word(float(w.start), float(w.end), text, float(w.probability)))
            return words

        return self._run(run)

    def detect_language(self, audio: np.ndarray, sr: int) -> tuple[str | None, float]:
        if not self.multilingual:
            return "en", 1.0
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        def run(model):
            lang, prob, _all = model.detect_language(audio)
            return lang, float(prob)

        return self._run(run)


_cache: dict[tuple, WhisperTranscriber] = {}
_cache_lock = threading.Lock()


def get_transcriber(config: Config, language: str | None) -> WhisperTranscriber:
    """Cached transcriber for ``language`` (``"en"`` → ``model_en``, else ``model_multi``).

    The returned object loads its model lazily on first use and stays loaded.
    """
    name = model_for_language(config, language)
    return get_model(config, name)


def get_model(config: Config, model_name: str) -> WhisperTranscriber:
    key = (model_name, config.device, config.compute_type, config.threads)
    with _cache_lock:
        t = _cache.get(key)
        if t is None:
            download_root = config.extra.get("model_dir") or None
            t = WhisperTranscriber(
                model_name, config.device, config.compute_type, config.threads, download_root
            )
            _cache[key] = t
        return t


def clear_cache() -> None:
    """Drop all cached models (frees memory)."""
    with _cache_lock:
        _cache.clear()


def download_model(model_name: str, download_root: str | None = None) -> str:
    """Download (or find cached) ``model_name``; returns the local model directory."""
    from faster_whisper import download_model as _dl

    return _dl(model_name, cache_dir=download_root)
