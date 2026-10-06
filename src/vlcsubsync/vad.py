"""Speech activity detection over a whole file.

Uses faster-whisper's bundled Silero VAD (ONNX, CPU, ~1 s per 10 min of audio) and
falls back to a simple adaptive energy detector if it cannot be loaded.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
RESOLUTION = 0.01  # seconds per mask frame
_SILERO_HOP = 512  # samples per Silero frame (32 ms at 16 kHz)
_CHUNK_SECONDS = 600  # run Silero on 10-min chunks to bound memory

_model = None
_model_failed = False
_lock = threading.Lock()


def _silero():
    global _model, _model_failed
    with _lock:
        if _model is None and not _model_failed:
            try:
                from faster_whisper.vad import get_vad_model

                _model = get_vad_model()
            except Exception as e:  # onnxruntime missing / broken
                log.warning("Silero VAD unavailable (%s); using energy VAD", e)
                _model_failed = True
        return _model


def silero_probabilities(
    audio: np.ndarray, progress: Callable[[float], None] | None = None
) -> np.ndarray | None:
    """Per-32 ms speech probabilities, or None if Silero is unavailable."""
    model = _silero()
    if model is None:
        return None
    n = audio.shape[0]
    chunk = _CHUNK_SECONDS * SAMPLE_RATE
    out = []
    try:
        for i in range(0, n, chunk):
            piece = audio[i : i + chunk]
            pad = (-piece.shape[0]) % _SILERO_HOP
            if pad:
                piece = np.concatenate([piece, np.zeros(pad, dtype=np.float32)])
            out.append(np.asarray(model(piece.astype(np.float32, copy=False))).reshape(-1))
            if progress:
                progress(min(1.0, (i + chunk) / max(n, 1)))
    except Exception as e:
        log.warning("Silero VAD failed (%s); using energy VAD", e)
        return None
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out).astype(np.float32)


def energy_probabilities(audio: np.ndarray) -> np.ndarray:
    """Crude per-32 ms 'speech probability' from frame energy relative to the noise
    floor (used only if Silero is unavailable)."""
    n = audio.shape[0] // _SILERO_HOP
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * _SILERO_HOP].reshape(n, _SILERO_HOP)
    db = 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    floor = np.percentile(db, 15)
    peak = np.percentile(db, 95)
    span = max(peak - floor, 6.0)
    return np.clip((db - floor - 0.25 * span) / (0.5 * span), 0.0, 1.0).astype(np.float32)


def _hysteresis(p: np.ndarray, on: float, off: float) -> np.ndarray:
    above_on = p >= on
    above_off = p >= off
    mask = np.zeros(p.shape[0], dtype=bool)
    # Runs of above_off that contain at least one above_on frame are speech.
    if not above_off.any():
        return mask
    d = np.diff(np.concatenate([[0], above_off.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    cs = np.concatenate([[0], np.cumsum(above_on)])
    for s, e in zip(starts, ends, strict=True):
        if cs[e] - cs[s] > 0:
            mask[s:e] = True
    return mask


def speech_mask(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    resolution: float = RESOLUTION,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Boolean speech mask sampled every ``resolution`` seconds over ``audio``."""
    if sr != SAMPLE_RATE:
        raise ValueError("speech_mask expects 16 kHz audio")
    probs = silero_probabilities(audio, progress)
    if probs is None:
        probs = energy_probabilities(audio)
    frame_mask = _hysteresis(probs, 0.5, 0.35)
    n_out = int(np.ceil(audio.shape[0] / sr / resolution))
    if frame_mask.shape[0] == 0 or n_out == 0:
        return np.zeros(n_out, dtype=bool)
    idx = np.minimum(
        (np.arange(n_out) * resolution * sr / _SILERO_HOP).astype(np.int64),
        frame_mask.shape[0] - 1,
    )
    return frame_mask[idx]


def mask_segments(mask: np.ndarray, resolution: float = RESOLUTION) -> list[tuple[float, float]]:
    """Contiguous True runs of ``mask`` as ``(start, end)`` seconds."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(s * resolution, e * resolution) for s, e in zip(starts, ends, strict=True)]


def speech_segments(
    audio: np.ndarray, sr: int = SAMPLE_RATE, resolution: float = RESOLUTION
) -> list[tuple[float, float]]:
    return mask_segments(speech_mask(audio, sr, resolution), resolution)
