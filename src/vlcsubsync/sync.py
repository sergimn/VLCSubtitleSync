"""Orchestration: media → audio/VAD → Whisper windows → anchors → fit → output."""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import vad
from .align import (
    AlignResult,
    Anchor,
    Cue,
    Mapping,
    apply_mapping,
    find_anchors,
    fit_mapping,
    refine_local,
    subdivide,
    subtitle_tokens,
    vad_align,
)
from .config import Config
from .media import SAMPLE_RATE, probe, read_media
from .subtitles import (
    dialogue_events,
    find_sidecars,
    guess_language,
    load_subtitles,
    save_subtitles,
)
from .transcribe import Transcriber, Word, get_model, get_transcriber

log = logging.getLogger(__name__)

WINDOW_SECONDS = 30.0
MIN_WHISPER_ANCHORS = 6
LANG_PROB_MIN = 0.5

ProgressFn = Callable[[float, str], None]


class SyncError(Exception):
    """Synchronisation could not be attempted (no cues, no audio, ...)."""


@dataclass
class SubtitleSource:
    kind: Literal["embedded", "external"]
    index: int | None = None  # ordinal among the container's subtitle streams (embedded)
    path: str | None = None  # external file


@dataclass
class SyncResult:
    output_path: str
    method: str  # "whisper" | "vad" | "none"
    offset: float  # new = old*scale + offset for the dominant segment
    scale: float
    segments: int
    confidence: float
    anchors: int
    applied: bool
    message: str


def resolve_subtitle_source(
    media_path: str, sub_ordinal: int, sub_path: str | None = None
) -> SubtitleSource:
    """Map VLC's subtitle-track ordinal to a :class:`SubtitleSource`.

    VLC lists embedded subtitle streams first (all of them, image ones included), then
    auto-loaded sidecar files in :func:`find_sidecars` order.
    """
    if sub_path:
        return SubtitleSource("external", path=sub_path)
    n_embedded = len(probe(media_path).subtitles)
    if sub_ordinal < n_embedded:
        return SubtitleSource("embedded", index=sub_ordinal)
    sidecars = find_sidecars(media_path)
    k = sub_ordinal - n_embedded
    if 0 <= k < len(sidecars):
        return SubtitleSource("external", path=sidecars[k])
    raise SyncError(
        f"subtitle track {sub_ordinal} not found ({n_embedded} embedded, {len(sidecars)} external)"
    )


class _Progress:
    def __init__(self, fn: ProgressFn):
        self.fn = fn
        self.last = 0.0

    def __call__(self, p: float, msg: str) -> None:
        p = max(self.last, min(1.0, p))
        self.last = p
        try:
            self.fn(p, msg)
        except Exception:  # never let a UI callback break the job
            log.debug("progress callback failed", exc_info=True)

    def sub(self, lo: float, hi: float, msg: str) -> Callable[[float], None]:
        return lambda f: self(lo + (hi - lo) * f, msg)


# --------------------------------------------------------------------------------------
# Window selection
# --------------------------------------------------------------------------------------


def _speech_per_second(speech: np.ndarray, resolution: float) -> np.ndarray:
    per = int(round(1.0 / resolution))
    n = speech.size // per
    if n == 0:
        return np.zeros(0)
    return speech[: n * per].reshape(n, per).mean(axis=1)


def pick_windows(
    speech_sec: np.ndarray,
    duration: float,
    k: int,
    win: float = WINDOW_SECONDS,
    ranges: list[tuple[float, float]] | None = None,
    taken: list[float] | None = None,
) -> list[float]:
    """Choose window starts: one per range (default: k equal slices of the file), at the
    most speech-dense ``win`` seconds inside the range, avoiding existing windows."""
    taken = list(taken or [])
    w = int(win)
    total = speech_sec.size
    if duration <= 0:
        return []
    if ranges is None:
        k = max(1, min(k, int(math.ceil(duration / win))))
        ranges = [(duration * i / k, duration * (i + 1) / k) for i in range(k)]
    cs = np.concatenate([[0.0], np.cumsum(speech_sec)]) if total else np.zeros(1)
    starts: list[float] = []
    for lo, hi in ranges:
        lo_i = max(0, int(lo))
        hi_i = max(lo_i, int(min(hi, duration)) - w)
        cands = np.arange(lo_i, hi_i + 1)
        if total and cands.size:
            ends = np.minimum(cands + w, total)
            dens = cs[ends] - cs[np.minimum(cands, total)]
        else:
            dens = np.zeros(cands.size)
        if cands.size == 0:
            cands = np.array([max(0, int(min(lo, duration - win)))])
            dens = np.zeros(1)
        # penalise overlap with existing windows
        for t in taken + starts:
            ov = np.clip(win - np.abs(cands - t), 0, None) / win
            dens = dens - 2.0 * ov * win
        best = float(cands[int(np.argmax(dens))])
        if any(abs(best - t) < win * 0.5 for t in taken + starts):
            continue
        starts.append(max(0.0, min(best, max(0.0, duration - win))))
    return sorted(starts)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def _load_source(media_path: str, audio_index: int, subtitle: SubtitleSource, prog: _Progress):
    def decode_cb(f: float) -> None:
        prog(0.02 + 0.28 * f, "Decoding audio")

    if subtitle.kind == "embedded":
        if subtitle.index is None:
            raise SyncError("embedded subtitle source needs an index")
        audio, subs, info = read_media(
            media_path, audio_index, subtitle.index, lambda p, m: decode_cb(p)
        )
        fmt = subs.format or "srt"
    elif subtitle.kind == "external":
        if not subtitle.path:
            raise SyncError("external subtitle source needs a path")
        subs, fmt = load_subtitles(subtitle.path)
        audio, _none, info = read_media(media_path, audio_index, None, lambda p, m: decode_cb(p))
    else:
        raise SyncError(f"unknown subtitle source kind {subtitle.kind!r}")
    return audio, subs, fmt, info


def _fmt_offset(x: float) -> str:
    return f"{x:+.2f}s"


def _message(fit: AlignResult, cue_starts: list[float]) -> str:
    m = fit.mapping
    dom = m.dominant(cue_starts)
    parts = []
    if len(m.segments) > 1:
        offs = [s.map(max(s.start, 0.0)) - max(s.start, 0.0) for s in m.segments]
        parts.append(
            f"{len(m.segments)} segments, offset " + " → ".join(_fmt_offset(o) for o in offs)
        )
    else:
        parts.append(f"offset {_fmt_offset(dom.offset)}")
    if abs(dom.scale - 1.0) >= 0.0005:
        parts.append(f"drift {(dom.scale - 1.0) * 100:+.1f}%")
    return ", ".join(parts)


def sync_subtitles(
    media_path: str,
    audio_index: int,
    subtitle: SubtitleSource,
    output_path: str,
    config: Config,
    progress: ProgressFn = lambda p, m: None,
    transcriber: Transcriber | None = None,
) -> SyncResult:
    """Re-time ``subtitle`` to audio track ``audio_index`` of ``media_path``.

    Always writes ``output_path`` (with its extension adjusted to the output format:
    ``.ass``/``.ssa``/``.vtt`` sources keep their format, everything else is SRT) and
    returns where it was written. If the result is not trusted (``applied=False``)
    the output keeps the original timings.
    """
    t0 = time.monotonic()
    prog = _Progress(progress)
    prog(0.0, "Reading media")
    audio, subs, fmt, info = _load_source(media_path, audio_index, subtitle, prog)
    assert audio is not None
    duration = audio.shape[0] / SAMPLE_RATE
    events = dialogue_events(subs)
    if not events:
        raise SyncError("subtitle track has no text cues")
    cues = [Cue(ev.start / 1000.0, ev.end / 1000.0, txt) for _i, ev, txt in events]
    cue_starts = [c.start for c in cues]
    t_decode = time.monotonic()

    prog(0.3, "Detecting speech")
    speech = vad.speech_mask(audio, progress=prog.sub(0.3, 0.36, "Detecting speech"))
    speech_sec = _speech_per_second(speech, vad.RESOLUTION)
    t_vad = time.monotonic()

    sub_lang = guess_language(c.text for c in cues)
    log.info("subtitle language guess: %s", sub_lang)

    fit: AlignResult | None = None
    anchors: list[Anchor] = []
    reason = ""
    k = config.window_count(duration)
    windows = pick_windows(speech_sec, duration, k)

    # Language check / model selection.
    prog(0.37, "Detecting language")
    t = transcriber or get_transcriber(config, sub_lang)
    lang = sub_lang
    audio_lang, lang_prob = None, 0.0
    if windows:
        best_w = max(windows, key=lambda s: speech_sec[int(s) : int(s + WINDOW_SECONDS)].sum())
        a, b = int(best_w * SAMPLE_RATE), int((best_w + WINDOW_SECONDS) * SAMPLE_RATE)
        # English-only models can't identify the audio language (they always say
        # "en"), so foreign audio with English subtitles would go undetected; ask the
        # multilingual model instead.
        detector = t
        if transcriber is None and not getattr(t, "multilingual", True):
            detector = get_model(config, config.model_multi)
        try:
            audio_lang, lang_prob = detector.detect_language(audio[a:b], SAMPLE_RATE)
        except Exception as e:
            log.warning("language detection failed: %s", e)
    mismatch = (
        sub_lang is not None
        and audio_lang is not None
        and lang_prob >= LANG_PROB_MIN
        and audio_lang != sub_lang
    )
    if sub_lang is None and audio_lang is not None and lang_prob >= LANG_PROB_MIN:
        lang = audio_lang
    log.info("audio language: %s (%.2f); mismatch=%s", audio_lang, lang_prob, mismatch)

    if mismatch:
        reason = f"audio language '{audio_lang}' differs from subtitles '{sub_lang}'"
    elif windows:
        sub_tok = subtitle_tokens(cues)
        transcribed: list[tuple[float, list[Word]]] = []
        budget_extra = max(4, k // 2)
        planned = len(windows)

        def run_windows(starts: list[float]) -> None:
            for st in starts:
                n_done = len(transcribed)
                frac = n_done / max(planned, 1)
                prog(0.38 + 0.54 * frac, f"Transcribing {n_done + 1}/{planned}")
                a = int(st * SAMPLE_RATE)
                b = min(audio.shape[0], int((st + WINDOW_SECONDS) * SAMPLE_RATE))
                try:
                    words = t.transcribe(audio[a:b], SAMPLE_RATE, lang, start=st)
                except Exception as e:
                    log.warning("transcription of window @%.0fs failed: %s", st, e)
                    words = []
                transcribed.append((st, list(words)))

        run_windows(windows)
        anchors = find_anchors(sub_tok, transcribed)
        fit = fit_mapping(anchors, cues, speech, vad.RESOLUTION, len(transcribed))
        log.info(
            "fit: %d/%d anchors, conf %.2f, %d segments",
            fit.anchors, fit.total_anchors, fit.confidence, len(fit.mapping.segments),
        )  # fmt: skip

        # Adaptive refinement (bounded): ambiguous segment switches, sparse evidence.
        for _round in range(3):
            if budget_extra <= 0:
                break
            taken = [st for st, _w in transcribed]
            ranges: list[tuple[float, float]] = []
            for lo, hi in fit.ambiguous:
                lo, hi = max(0.0, min(lo, hi)), min(duration, max(lo, hi))
                if hi - lo > WINDOW_SECONDS:
                    mid = 0.5 * (lo + hi)
                    ranges.append((mid - WINDOW_SECONDS, mid + WINDOW_SECONDS))
            if fit.confidence < 0.8 or fit.anchors < 20:
                edges = sorted(taken) + [duration]
                prev = 0.0
                gaps = []
                for e in edges:
                    if e - prev > 2 * WINDOW_SECONDS:
                        gaps.append((e - prev, prev, e))
                    prev = e + WINDOW_SECONDS
                gaps.sort(reverse=True)
                for _g, lo, hi in gaps[: max(2, k // 3)]:
                    ranges.append((lo, hi))
            ranges = ranges[:budget_extra]
            if not ranges:
                break
            new = pick_windows(speech_sec, duration, len(ranges), ranges=ranges, taken=taken)
            if not new:
                break
            budget_extra -= len(new)
            planned += len(new)
            run_windows(new)
            anchors = find_anchors(sub_tok, transcribed)
            fit = fit_mapping(anchors, cues, speech, vad.RESOLUTION, len(transcribed))
            log.info(
                "refit (+%d windows): %d/%d anchors, conf %.2f, %d segments",
                len(new), fit.anchors, fit.total_anchors, fit.confidence,
                len(fit.mapping.segments),
            )  # fmt: skip

        # Bisection verification: check each segment at its midpoint (transcribing a
        # window there if none is near, within a budget), split where it disagrees.
        verify_left = config.verify_budget(duration)

        def verify_probe(centre: float, near: float) -> list[Anchor]:
            nonlocal anchors, verify_left, planned
            if verify_left <= 0 or not 0.0 <= centre <= duration:
                return anchors
            if any(abs(st + 0.5 * WINDOW_SECONDS - centre) <= near for st, _w in transcribed):
                return anchors
            taken = [st for st, _w in transcribed]
            lo = max(0.0, centre - near)
            hi = min(duration, centre + near)
            new = pick_windows(speech_sec, duration, 1, ranges=[(lo, hi)], taken=taken)
            if not new:
                return anchors
            verify_left -= len(new)
            planned += len(new)
            run_windows(new)
            anchors = find_anchors(sub_tok, transcribed)
            return anchors

        if fit.anchors >= MIN_WHISPER_ANCHORS:
            n_before = len(transcribed)
            fit, anchors = subdivide(
                fit, anchors, cues, verify_probe, speech, vad.RESOLUTION, len(transcribed)
            )
            log.info(
                "verification: %d checks, %d splits, %d folded, +%d windows: conf %.2f, "
                "%d segments",
                fit.details.get("verify_checks", 0), fit.details.get("verify_splits", 0),
                fit.details.get("verify_folded", 0), len(transcribed) - n_before,
                fit.confidence, len(fit.mapping.segments),
            )  # fmt: skip
        if fit.anchors < MIN_WHISPER_ANCHORS:
            reason = f"too few transcript matches ({fit.anchors})"
    else:
        reason = "no audio to transcribe"
    t_whisper = time.monotonic()

    prog(0.93, "Aligning")
    if fit is not None and fit.anchors >= MIN_WHISPER_ANCHORS:
        fit.mapping, info = refine_local(fit.mapping, anchors, cues, speech, vad.RESOLUTION)
        fit.details.update(info)
        log.info(
            "local refinement: segment shifts %s, speech-onset %+.3fs, %d wobble knots",
            ", ".join(f"{d:+.3f}" for d in info["segment_shifts"]) or "-",
            info["onset_shift"], info["knots"],
        )  # fmt: skip
    final = fit
    if fit is None or fit.anchors < MIN_WHISPER_ANCHORS or fit.confidence < config.min_confidence:
        v = vad_align(speech, cues, vad.RESOLUTION)
        log.info("VAD fallback: conf %.2f (%s)", v.confidence, v.details)
        if final is None or final.anchors < MIN_WHISPER_ANCHORS or v.confidence > final.confidence:
            final = v

    applied = (
        final is not None and final.method != "none" and (final.confidence >= config.min_confidence)
    )
    out_subs = copy.deepcopy(subs)
    if applied:
        assert final is not None
        apply_mapping(out_subs, final.mapping)
        dom = final.mapping.dominant(cue_starts)
        message = _message(final, cue_starts)
        if final.method == "vad":
            message += " (speech timing)"
        result = SyncResult(
            output_path="",
            method=final.method,
            offset=round(dom.offset, 3),
            scale=round(dom.scale, 6),
            segments=len(final.mapping.segments),
            confidence=round(final.confidence, 3),
            anchors=final.anchors,
            applied=True,
            message=message,
        )
    else:
        conf = final.confidence if final is not None else 0.0
        why = reason or "low confidence"
        message = f"not synced: {why} (confidence {conf:.2f})"
        result = SyncResult(
            output_path="",
            method="none",
            offset=0.0,
            scale=1.0,
            segments=1,
            confidence=round(conf, 3),
            anchors=final.anchors if final is not None else 0,
            applied=False,
            message=message,
        )
    prog(0.97, "Writing subtitles")
    result.output_path = save_subtitles(out_subs, output_path, fmt)
    t_end = time.monotonic()
    log.info(
        "sync done in %.1fs (decode %.1f, vad %.1f, whisper %.1f): %s",
        t_end - t0, t_decode - t0, t_vad - t_decode, t_whisper - t_vad, result.message,
    )  # fmt: skip
    prog(1.0, "Done")
    return result


__all__ = [
    "Mapping",
    "SubtitleSource",
    "SyncError",
    "SyncResult",
    "pick_windows",
    "resolve_subtitle_source",
    "sync_subtitles",
]
