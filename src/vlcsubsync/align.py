"""Subtitle ↔ audio alignment.

Pipeline (see DESIGN.md):

1. :func:`subtitle_tokens` – normalised subtitle tokens with estimated times.
2. :func:`find_anchors` – rare n-grams shared by transcript windows and the *whole*
   subtitle token stream → ``(sub_time, audio_time, weight)`` anchors.
3. :func:`fit_mapping` – robust piecewise-linear ``audio = scale*sub + offset``:
   RANSAC candidate lines → Viterbi labelling of time-sorted anchors with a segment
   switch penalty → per-segment IRLS refit (short segments share the dominant scale)
   → boundaries placed at cue gaps (using the speech mask when available).
4. :func:`subdivide` – bisection verification: check each segment at its midpoint
   (and every window inside it), split where it disagrees by more than 0.25 s and
   the split explains it, refit halves with known ratios, merge what agrees.
5. :func:`refine_local` – per-segment median residual, global speech-onset bias
   (:func:`refine_with_speech`), smooth local wobble knots; each bounded ±0.5 s.
6. :func:`vad_align` – fallback: FFT cross-correlation of the speech mask with the
   subtitle-on mask over framerate scale candidates (prior towards 1.0).
7. :func:`apply_mapping` – retime every event (cue durations scale with the
   segment, introduced overlaps are clamped, nothing is dropped).

Everything is deterministic (seeded RNG).
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .subtitles import tokenize
from .transcribe import Word

# Framerate conversion factors seen in the wild (new = old * scale).
SCALE_CANDIDATES = (
    1.0,
    25 / 23.976,
    23.976 / 25,
    24 / 23.976,
    23.976 / 24,
    25 / 24,
    24 / 25,
    # NTSC video vs film/PAL timings (e.g. subtitles made for a 29.97 fps TV
    # release played against a 23.976 fps web/Blu-ray video).
    29.97 / 23.976,
    23.976 / 29.97,
    29.97 / 25,
    25 / 29.97,
)
SCALE_MIN, SCALE_MAX = 0.78, 1.28

INLIER_THRESHOLD = 1.0  # s, residual for an anchor to count as consistent
CHARS_PER_SECOND = 15.0  # typical speaking rate used to place tokens inside a cue


# --------------------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------------------


@dataclass
class Cue:
    start: float
    end: float
    text: str  # plain text (tags stripped)


@dataclass
class SubTokens:
    tokens: list[str]
    times: np.ndarray  # estimated time of each token (s, subtitle clock, at scale 1)
    cue: np.ndarray  # cue index of each token
    # In-cue timing: time = cue start + min(lead / scale, cap). ``lead`` is the speech
    # before the token at the nominal speaking rate (audio clock), ``cap`` the same
    # fraction of the cue's duration (subtitle clock). See :func:`anchor_sub_times`.
    lead: np.ndarray | None = None
    cap: np.ndarray | None = None


@dataclass
class Anchor:
    sub_time: float
    audio_time: float
    weight: float
    window: int = -1
    cue: int = -1
    token: int = -1  # transcript token index within its window (groups alternatives)
    lead: float = 0.0  # in-cue timing of the subtitle token (see SubTokens)
    cap: float = 0.0


@dataclass
class Segment:
    start: float  # subtitle-clock time where this segment begins (-inf for the first)
    scale: float
    offset: float
    anchors: int = 0
    # Smooth local correction: (subtitle time, seconds) knots, linearly interpolated
    # and held flat outside (see :func:`refine_local`). Empty = none.
    knots: tuple[tuple[float, float], ...] = ()

    def line(self, t: float) -> float:
        return self.scale * t + self.offset

    def correction(self, t):
        if not self.knots:
            return 0.0 if np.ndim(t) == 0 else np.zeros(np.shape(t))
        kx = [k[0] for k in self.knots]
        ky = [k[1] for k in self.knots]
        c = np.interp(t, kx, ky)
        return float(c) if np.ndim(t) == 0 else c

    def map(self, t: float) -> float:
        return self.scale * t + self.offset + self.correction(t)


@dataclass
class Mapping:
    segments: list[Segment]

    @classmethod
    def identity(cls) -> Mapping:
        return cls([Segment(-math.inf, 1.0, 0.0)])

    @classmethod
    def linear(cls, scale: float, offset: float, anchors: int = 0) -> Mapping:
        return cls([Segment(-math.inf, scale, offset, anchors)])

    def segment_for(self, t: float) -> Segment:
        starts = [s.start for s in self.segments]
        i = max(0, bisect.bisect_right(starts, t) - 1)
        return self.segments[i]

    def __call__(self, t: float) -> float:
        return self.segment_for(t).map(t)

    def map_array(self, t: np.ndarray) -> np.ndarray:
        starts = np.array([s.start for s in self.segments])
        idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(self.segments) - 1)
        sc = np.array([s.scale for s in self.segments])[idx]
        of = np.array([s.offset for s in self.segments])[idx]
        out = sc * t + of
        for i, s in enumerate(self.segments):
            if s.knots:
                m = idx == i
                out[m] += s.correction(np.asarray(t)[m])
        return out

    def dominant(self, cue_starts: Sequence[float] | None = None) -> Segment:
        """Segment covering most cues (or most anchors if no cues are given)."""
        if len(self.segments) == 1:
            return self.segments[0]
        if cue_starts:
            counts = Counter(id(self.segment_for(t)) for t in cue_starts)
            return max(self.segments, key=lambda s: (counts.get(id(s), 0), s.anchors))
        return max(self.segments, key=lambda s: s.anchors)


@dataclass
class AlignResult:
    mapping: Mapping
    method: str  # "whisper" | "vad" | "none"
    confidence: float
    anchors: int  # inlier anchors (whisper) / 0 (vad)
    total_anchors: int = 0
    residual: float = 0.0  # median |residual| of inliers (s)
    ambiguous: list[tuple[float, float]] = field(default_factory=list)  # audio-time ranges
    sparse: list[tuple[float, float]] = field(default_factory=list)  # audio-time ranges
    details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Tokens and anchors
# --------------------------------------------------------------------------------------


def subtitle_tokens(cues: Sequence[Cue], cps: float = CHARS_PER_SECOND) -> SubTokens:
    """Tokenise all cues; each token's time is estimated inside its cue assuming speech
    starts at the cue start and proceeds at ``cps`` characters/s (compressed if the cue
    is too short for that)."""
    tokens: list[str] = []
    times: list[float] = []
    cue_idx: list[int] = []
    leads: list[float] = []
    caps: list[float] = []
    for ci, cue in enumerate(cues):
        toks = tokenize(cue.text)
        if not toks:
            continue
        lens = np.array([len(t) + 1 for t in toks], dtype=float)
        before = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
        total = float(lens.sum())
        dur = max(cue.end - cue.start, 0.0)
        frac = before / total
        lead = before / cps
        cap = frac * dur
        t = cue.start + np.minimum(lead, cap)
        tokens.extend(toks)
        times.extend(t.tolist())
        cue_idx.extend([ci] * len(toks))
        leads.extend(lead.tolist())
        caps.extend(cap.tolist())
    return SubTokens(
        tokens,
        np.asarray(times, dtype=float),
        np.asarray(cue_idx, dtype=int),
        np.asarray(leads, dtype=float),
        np.asarray(caps, dtype=float),
    )


def _word_tokens(words: Sequence[Word], offset: float) -> tuple[list[str], list[float]]:
    toks: list[str] = []
    times: list[float] = []
    for w in words:
        parts = tokenize(w.text, drop_annotations=False)
        if not parts:
            continue
        dur = max(w.end - w.start, 0.0)
        for k, p in enumerate(parts):
            toks.append(p)
            times.append(offset + w.start + dur * k / len(parts))
    return toks, times


def find_anchors(
    sub: SubTokens,
    windows: Sequence[tuple[float, Sequence[Word]]],
    max_n: int = 4,
) -> list[Anchor]:
    """Match transcript n-grams against the whole subtitle token stream.

    ``windows`` = ``[(window_start_seconds, words_relative_to_window_start), ...]``.
    Longer n-grams win. Every occurrence of a matched n-gram becomes a candidate anchor
    (≤48 occurrences for n≥4, ≤24 for n=3, ≤4 for n=2; unique unigrams of ≥6 letters
    get a small weight), weighted ∝ 1/occurrences × informativeness; the robust fit
    then keeps the globally consistent candidates.
    """
    n_sub = len(sub.tokens)
    if n_sub == 0:
        return []
    freq = Counter(sub.tokens)
    # informativeness of a token: rare tokens are worth more (in 0.2..1)
    max_f = max(freq.values())

    def info(tok: str) -> float:
        return 0.2 + 0.8 * (1.0 - math.log1p(freq.get(tok, 1) - 1) / math.log1p(max_f))

    index: dict[int, dict[tuple[str, ...], list[int]]] = {}
    for n in range(1, max_n + 1):
        d: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(n_sub - n + 1):
            d[tuple(sub.tokens[i : i + n])].append(i)
        index[n] = d
    # Non-unique n-grams are kept as weighted candidates (recurring lines, repeated
    # content): the robust fit picks the consistent occurrence.
    max_occ = {1: 1, 2: 4, 3: 24}
    best: dict[tuple[int, int, int], Anchor] = {}
    for wi, (wstart, words) in enumerate(windows):
        toks, times = _word_tokens(words, wstart)
        if not toks:
            continue
        win_counts = {n: Counter(tuple(toks[j : j + n]) for j in range(len(toks) - n + 1))
                      for n in range(1, max_n + 1)}  # fmt: skip
        covered_until = -1
        for j in range(len(toks)):
            for n in range(max_n, 0, -1):
                if j + n > len(toks):
                    continue
                gram = tuple(toks[j : j + n])
                pos = index[n].get(gram)
                if not pos:
                    continue
                if len(pos) > max_occ.get(n, 48):
                    continue
                gram_info = sum(info(t) for t in gram) / n
                if n == 1:
                    if j <= covered_until or len(gram[0]) < 6 or gram[0].isdigit():
                        continue
                    weight = 0.25 * gram_info
                elif n == 2:
                    if gram_info < 0.45:
                        continue
                    weight = 0.6 * gram_info
                else:
                    weight = min(1.0, 0.5 + gram_info) * (1.0 if n >= 4 else 0.85)
                weight /= len(pos) * win_counts[n][gram]
                for p in pos:
                    # anchor every token of the matched n-gram (keeps max weight)
                    for k in range(n):
                        key = (wi, j + k, p + k)
                        a = best.get(key)
                        if a is None or a.weight < weight:
                            best[key] = Anchor(
                                float(sub.times[p + k]),
                                float(times[j + k]),
                                weight,
                                wi,
                                int(sub.cue[p + k]),
                                j + k,
                                float(sub.lead[p + k]) if sub.lead is not None else 0.0,
                                float(sub.cap[p + k]) if sub.cap is not None else 0.0,
                            )
                covered_until = max(covered_until, j + n - 1)
                break
    anchors = list(best.values())
    anchors.sort(key=lambda a: (a.sub_time, a.audio_time))
    return anchors


# --------------------------------------------------------------------------------------
# Robust fitting
# --------------------------------------------------------------------------------------


def _score(res: np.ndarray, w: np.ndarray, thr: float) -> np.ndarray:
    """Truncated-quadratic (MSAC-like) support; res shape (H, N)."""
    return (w * np.clip(1.0 - (res / thr) ** 2, 0.0, None)).sum(axis=-1)


def _ransac_line(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator, thr: float = 0.6
) -> tuple[float, float, float] | None:
    """Best (scale, offset, support) line; hypotheses from known framerate scales through
    single anchors plus free-scale lines through random anchor pairs."""
    n = x.shape[0]
    if n == 0:
        return None
    pick = np.arange(n) if n <= 250 else rng.choice(n, 250, replace=False, p=w / w.sum())
    hs, ho = [], []
    for s in SCALE_CANDIDATES:
        hs.append(np.full(pick.shape[0], s))
        ho.append(y[pick] - s * x[pick])
    if n >= 2:
        i = rng.integers(0, n, 600)
        j = rng.integers(0, n, 600)
        dx = x[j] - x[i]
        ok = np.abs(dx) > 30.0
        if ok.any():
            s = (y[j][ok] - y[i][ok]) / dx[ok]
            o = y[i][ok] - s * x[i][ok]
            good = (s > SCALE_MIN) & (s < SCALE_MAX)
            hs.append(s[good])
            ho.append(o[good])
    S = np.concatenate(hs)
    offs = np.concatenate(ho)
    best_s, best_o, best_score = 1.0, 0.0, -1.0
    chunk = max(1, 400_000 // max(n, 1))
    for k in range(0, S.shape[0], chunk):
        s = S[k : k + chunk, None]
        o = offs[k : k + chunk, None]
        sc = _score(y[None, :] - (s * x[None, :] + o), w[None, :], thr)
        m = int(np.argmax(sc))
        if sc[m] > best_score:
            best_score = float(sc[m])
            best_s, best_o = float(S[k + m]), float(offs[k + m])
    return best_s, best_o, best_score


def _irls(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scale: float,
    offset: float,
    fixed_scale: bool,
    iters: int = 8,
    c: float = 0.5,
) -> tuple[float, float]:
    """Iteratively reweighted least squares with Tukey biweight (cutoff ``c`` s)."""
    s, o = scale, offset
    for _ in range(iters):
        r = y - (s * x + o)
        u = r / c
        tw = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0) * w
        if tw.sum() <= 1e-9:
            break
        if fixed_scale or np.ptp(x[tw > 0]) < 60.0:
            o = float(np.sum(tw * (y - s * x)) / tw.sum())
        else:
            xm = np.sum(tw * x) / tw.sum()
            ym = np.sum(tw * y) / tw.sum()
            var = np.sum(tw * (x - xm) ** 2)
            if var <= 1e-9:
                o = float(ym - s * xm)
            else:
                s_new = float(np.sum(tw * (x - xm) * (y - ym)) / var)
                s = min(max(s_new, SCALE_MIN), SCALE_MAX)
                o = float(ym - s * xm)
    return s, o


def _window_medians(r: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Median residual of each window (inliers only)."""
    m = np.abs(r) < INLIER_THRESHOLD
    return np.array([np.median(r[m & (win == k)]) for k in np.unique(win[m])])


def _slope_se(x: np.ndarray, r: np.ndarray, win: np.ndarray) -> float:
    """Standard error of a line's slope when each window is one noisy observation
    (its words share a timestamp bias): robust spread of the per-window median
    residuals over sqrt(windows) x spread of the window positions."""
    m = np.abs(r) < INLIER_THRESHOLD
    ks = np.unique(win[m])
    if ks.size < 4:
        return 0.0
    d = np.array([np.median(r[m & (win == k)]) for k in ks])
    xw = np.array([np.median(x[m & (win == k)]) for k in ks])
    spread = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), 0.03)
    sx = float(np.std(xw))
    return spread / (math.sqrt(ks.size) * sx) if sx > 0 else 0.0


def _snap_scale(x, y, w, s, o, win=None) -> tuple[float, float]:
    """Snap a free-fit scale to a known framerate ratio if it is very close and the fit
    does not get worse: per anchor, or per transcription window when ``win`` is given
    (a window's word timestamps share a bias, so one window with many anchors must not
    tilt the line away from an exact ratio)."""
    for cand in sorted(SCALE_CANDIDATES, key=lambda c: abs(s - c)):
        if abs(s - cand) < 0.0015 and cand != s:
            s2, o2 = _irls(x, y, w, cand, o + (s - cand) * float(np.median(x)), True)
            r1 = y - (s * x + o)
            r2 = y - (s2 * x + o2)
            m1 = np.abs(r1) < INLIER_THRESHOLD
            m2 = np.abs(r2) < INLIER_THRESHOLD
            a1 = np.abs(r1[m1] if m1.any() else r1)
            a2 = np.abs(r2[m2] if m2.any() else r2)
            if m2.sum() >= m1.sum() and np.median(a2) <= np.median(a1) + 0.05:
                return s2, o2
            if win is not None and m2.sum() >= 0.97 * m1.sum():
                d1 = _window_medians(r1, win)
                d2 = _window_medians(r2, win)
                if d1.size >= 4 and np.median(np.abs(d2)) <= np.median(np.abs(d1)) + 0.02:
                    return s2, o2
                # ... or the free slope is within 2 standard errors of the ratio, the
                # error estimated from the spread of the per-window medians
                if d1.size >= 4 and abs(s - cand) <= 2.0 * _slope_se(x, r1, win):
                    return s2, o2
    return s, o


def _best_candidate_scale(x, y, w, s0: float, o0: float) -> tuple[float, float]:
    """For short spans a free scale is unreliable: pick the known framerate ratio that
    fits best (1.0 wins near-ties)."""
    xm = float(np.median(x))
    span = float(np.ptp(x)) if x.size else 0.0
    best = None
    for cand in SCALE_CANDIDATES:
        if cand != 1.0 and abs(cand - 1.0) * span < 0.4:
            continue  # indistinguishable from 1.0 over this span
        if abs(cand - 1.0) > 0.1 and x.size < 10:
            continue  # NTSC-sized drift (±17-25%) needs more evidence than a few anchors
        s, o = _irls(x, y, w, cand, o0 + (s0 - cand) * xm, True)
        r = y - (s * x + o)
        cost = float((w * np.minimum((r / 0.5) ** 2, 1.0)).sum())
        # Prior towards no drift, stronger for bigger ratios, so a coincidental fit of
        # a few anchors can't pick an extreme scale on a near-tie.
        cost *= 0.98 if cand == 1.0 else 1.0 + 2.0 * abs(cand - 1.0)
        if best is None or cost < best[0]:
            best = (cost, s, o)
    assert best is not None
    return best[1], best[2]


@dataclass
class _Line:
    scale: float
    offset: float


def _candidate_lines(x, y, w, rng, max_lines: int = 6) -> list[_Line]:
    lines: list[_Line] = []
    remaining = np.ones(x.shape[0], dtype=bool)
    total_w = float(w.sum())
    for _ in range(max_lines):
        idx = np.flatnonzero(remaining)
        if idx.shape[0] < 3:
            break
        found = _ransac_line(x[idx], y[idx], w[idx], rng)
        if found is None:
            break
        s, o, support = found
        inl = np.abs(y[idx] - (s * x[idx] + o)) < INLIER_THRESHOLD
        n_in = int(inl.sum())
        if lines and (n_in < 4 or support < max(2.0, 0.03 * total_w)):
            break
        if not lines and n_in < 3:
            break
        fixed = np.ptp(x[idx][inl]) < 300.0 if n_in else True
        s, o = _irls(x[idx][inl], y[idx][inl], w[idx][inl], s, o, fixed)
        lines.append(_Line(s, o))
        remaining[idx[np.abs(y[idx] - (s * x[idx] + o)) < INLIER_THRESHOLD]] = False
    return lines


def _viterbi_costs(cost: np.ndarray, penalty: float) -> np.ndarray:
    """Min-cost labelling of a sequence (rows of ``cost``) with a per-switch penalty."""
    n, L = cost.shape
    acc = cost[0].copy()
    back = np.zeros((n, L), dtype=np.int32)
    for i in range(1, n):
        best_prev = int(np.argmin(acc))
        switch = acc[best_prev] + penalty
        choose_switch = switch < acc
        back[i] = np.where(choose_switch, best_prev, np.arange(L))
        acc = np.where(choose_switch, switch, acc) + cost[i]
    labels = np.zeros(n, dtype=np.int32)
    labels[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        labels[i - 1] = back[i, labels[i]]
    return labels


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(start_idx, end_idx_exclusive, label) runs."""
    out = []
    i = 0
    n = labels.shape[0]
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        out.append((i, j, int(labels[i])))
        i = j
    return out


def _choose_boundary(
    xa: float,
    xb: float,
    left: Segment,
    right: Segment,
    cues: Sequence[Cue] | None,
    speech: np.ndarray | None,
    resolution: float,
) -> float:
    """Pick where (in subtitle time) the segment switch happens between the last anchor
    of the left segment ``xa`` and the first anchor of the right one ``xb``."""
    if not cues:
        return 0.5 * (xa + xb)
    inside = [i for i, c in enumerate(cues) if xa < c.start < xb]
    if not inside:
        return 0.5 * (xa + xb)
    # candidate k: cues inside[:k] go left, inside[k:] go right
    cand_times = [cues[i].start for i in inside] + [xb]
    if speech is not None and speech.size:
        cs = np.concatenate([[0], np.cumsum(speech.astype(np.int32))])

        def overlap(seg: Segment, c: Cue) -> float:
            a = int(round(seg.map(c.start) / resolution))
            b = int(round(seg.map(c.end) / resolution))
            a = min(max(a, 0), speech.size)
            b = min(max(b, 0), speech.size)
            if b <= a:
                return 0.0
            on = cs[b] - cs[a]
            return float(on - 0.5 * ((b - a) - on))  # reward speech, punish silence

        lv = [overlap(left, cues[i]) for i in inside]
        rv = [overlap(right, cues[i]) for i in inside]
        scores = [sum(lv[:k]) + sum(rv[k:]) for k in range(len(inside) + 1)]
        best = max(scores)
        if best - min(scores) > 0.3 / resolution:  # ≥0.3 s of speech difference
            ks = [k for k, s in enumerate(scores) if s >= best - 1e-9]
            k = ks[len(ks) // 2]
            return cand_times[k] if k < len(inside) else xb
    # no audio evidence: cut at the largest gap between consecutive cues
    best_gap, best_t = -1.0, 0.5 * (xa + xb)
    prev_end = xa
    for i in inside:
        gap = cues[i].start - prev_end
        if gap > best_gap:
            best_gap, best_t = gap, cues[i].start
        prev_end = max(prev_end, cues[i].end)
    return best_t


def _fit_segment(xs, ys, ws, line: _Line, dom_s: float | None, wins=None) -> tuple[float, float]:
    """Refit one segment. Long, well-supported segments get a free scale (snapped to a
    framerate ratio when close); short ones reuse the dominant scale (or the best known
    ratio if there is no dominant scale yet)."""
    free = xs.size >= 15 and float(np.ptp(xs)) >= 600.0
    if free and (dom_s is None or abs(line.scale - dom_s) > 0.002):
        s, o = _irls(xs, ys, ws, line.scale, line.offset, False)
        return _snap_scale(xs, ys, ws, s, o, wins)
    if dom_s is None:
        return _best_candidate_scale(xs, ys, ws, line.scale, line.offset)
    o0 = float(np.median(ys - dom_s * xs))
    return _irls(xs, ys, ws, dom_s, o0, True)


@dataclass
class _AnchorArrays:
    """Anchors as arrays, sorted by subtitle time. ``grp`` numbers transcript tokens
    (a token's candidate anchors are alternatives, one of them at most is right)."""

    anchors: list[Anchor]
    x0: np.ndarray  # subtitle time at scale 1
    y: np.ndarray
    w: np.ndarray
    win: np.ndarray
    grp: np.ndarray
    base: np.ndarray  # cue start
    lead: np.ndarray
    cap: np.ndarray

    @classmethod
    def build(cls, anchors: Sequence[Anchor]) -> _AnchorArrays:
        A = sorted(anchors, key=lambda a: (a.sub_time, a.audio_time))
        x0 = np.array([a.sub_time for a in A], dtype=float)
        lead = np.array([a.lead for a in A], dtype=float)
        cap = np.array([a.cap for a in A], dtype=float)
        gkeys: dict[tuple, int] = {}
        grp = np.array(
            [
                gkeys.setdefault((a.window, a.token) if a.token >= 0 else ("i", i), len(gkeys))
                for i, a in enumerate(A)
            ],
            dtype=int,
        )
        return cls(
            A,
            x0,
            np.array([a.audio_time for a in A], dtype=float),
            np.array([a.weight for a in A], dtype=float),
            np.array([a.window for a in A], dtype=int),
            grp,
            x0 - np.minimum(lead, cap),
            lead,
            cap,
        )

    def x(self, scale) -> np.ndarray:
        """Subtitle times with in-cue offsets for drift ``scale`` (scalar or per anchor)."""
        return self.base + np.minimum(self.lead / scale, self.cap)

    def x_for(self, mapping: Mapping) -> np.ndarray:
        starts = np.array([s.start for s in mapping.segments])
        idx = np.clip(np.searchsorted(starts, self.x0, side="right") - 1, 0, None)
        return self.x(np.array([s.scale for s in mapping.segments])[idx])


def fit_mapping(
    anchors: Sequence[Anchor],
    cues: Sequence[Cue] | None = None,
    speech: np.ndarray | None = None,
    resolution: float = 0.01,
    n_windows: int = 0,
    seed: int = 0,
    segment_penalty: float = 0.9,
) -> AlignResult:
    """Robust piecewise-linear fit of ``audio_time = scale*sub_time + offset``.

    Candidate lines come from RANSAC; each transcription window is then labelled with
    a line by a Viterbi pass over windows in audio order (emission = how much of the
    window's support the line explains, switch cost ``segment_penalty`` ≈ windows of
    evidence needed for a cut). Working per window rather than per anchor makes
    repeated content harmless: every occurrence of a recurring phrase is a candidate
    anchor, but only the occurrence consistent with its neighbours is used.
    """
    if len(anchors) < 3:
        return AlignResult(Mapping.identity(), "none", 0.0, 0, len(anchors))
    arr = _AnchorArrays.build(anchors)
    A = arr.anchors
    y, w, win, grp = arr.y, arr.w, arr.win, arr.grp

    lines = _candidate_lines(arr.x0, y, w, np.random.default_rng(seed))
    if not lines:
        return AlignResult(Mapping.identity(), "none", 0.0, 0, len(A))
    # In-cue token times assume the nominal speaking rate in the *audio* clock: under
    # drift they move with the dominant scale (else later words in long cues are
    # biased by up to (1 - 1/scale) x their in-cue offset). Re-run with corrected times.
    x_scale = lines[0].scale
    x = arr.x(x_scale)
    if abs(x_scale - 1.0) > 1e-3:
        lines = _candidate_lines(x, y, w, np.random.default_rng(seed)) or lines
    L = len(lines)
    S = np.array([ln.scale for ln in lines])
    O = np.array([ln.offset for ln in lines])  # noqa: E741
    res_all = np.abs(y[:, None] - (S[None, :] * x[:, None] + O[None, :]))  # (N, L)
    support = w[:, None] * np.clip(1.0 - (res_all / 0.6) ** 2, 0.0, None)

    # --- window-level labelling -------------------------------------------------------
    win_ids = sorted(set(win.tolist()), key=lambda k: float(y[win == k].min()))
    sup_w = np.array([support[win == k].sum(axis=0) for k in win_ids])  # (W, L)
    smax = sup_w.max(axis=1, keepdims=True)
    cost = np.where(smax >= 0.5, 1.0 - sup_w / np.maximum(smax, 1e-9), 0.0)
    # Tie-break for genuinely ambiguous (periodic) evidence: prefer smaller shifts.
    cost = cost + 1e-3 * np.abs(O)[None, :] / 60.0
    labels = _viterbi_costs(cost, segment_penalty) if L > 1 else np.zeros(len(win_ids), int)

    def members(run_windows: list[int], lab: int) -> np.ndarray:
        m = np.isin(win, run_windows) & (res_all[:, lab] < INLIER_THRESHOLD)
        return m

    # Merge weak runs (too little consistent evidence) into a neighbour.
    for _ in range(50):
        runs = _runs(labels)
        if len(runs) <= 1:
            break
        weak = None
        for ri, (a, b, lab) in enumerate(runs):
            m = members(win_ids[a:b], lab)
            n_groups = len(set(grp[m].tolist()))
            strength = float(w[m].sum())
            if n_groups < 4 or strength < 1.5:
                if weak is None or strength < weak[1]:
                    weak = (ri, strength)
        if weak is None:
            break
        ri = weak[0]
        a, b, _lab = runs[ri]
        neigh = [runs[j][2] for j in (ri - 1, ri + 1) if 0 <= j < len(runs)]
        labels[a:b] = max(neigh, key=lambda lab, a=a, b=b: float(sup_w[a:b, lab].sum()))

    runs = _runs(labels)
    groups: list[tuple[np.ndarray, _Line, list[int]]] = []
    for a, b, lab in runs:
        m = members(win_ids[a:b], lab)
        if not m.any():
            continue
        groups.append((m, lines[lab], win_ids[a:b]))
    if not groups:
        return AlignResult(Mapping.identity(), "none", 0.0, 0, len(A))

    # Dominant segment = widest sub-time span; its scale is shared by short segments.
    dom_i = max(range(len(groups)), key=lambda i: float(np.ptp(x[groups[i][0]])))
    m, ln, _ = groups[dom_i]
    dom_s, dom_o = _fit_segment(x[m], y[m], w[m], ln, None, win[m])
    fitted: list[tuple[np.ndarray, Segment]] = []
    for i, (m, ln, _wins) in enumerate(groups):
        s, o = (dom_s, dom_o) if i == dom_i else _fit_segment(x[m], y[m], w[m], ln, dom_s, win[m])
        # members w.r.t. the refined line
        m2 = np.isin(win, _wins) & (np.abs(y - (s * x + o)) < INLIER_THRESHOLD)
        if not m2.any():
            m2 = m
        fitted.append((m2, Segment(-math.inf, s, o, len(set(grp[m2].tolist())))))

    # Order by subtitle time and merge consecutive, effectively identical segments.
    fitted.sort(key=lambda t: float(np.median(x[t[0]])))
    merged: list[tuple[np.ndarray, Segment]] = []
    for m, sg in fitted:
        if merged:
            pm, ps = merged[-1]
            xb = float(x[m].min())
            if abs(ps.map(xb) - sg.map(xb)) < 0.15 and abs(ps.scale - sg.scale) < 0.002:
                merged[-1] = (
                    pm | m,
                    Segment(-math.inf, ps.scale, ps.offset, ps.anchors + sg.anchors),
                )
                continue
        merged.append((m, sg))

    # Boundaries between consecutive segments (subtitle clock).
    ambiguous: list[tuple[float, float]] = []
    final: list[Segment] = []
    for k, (m, sg) in enumerate(merged):
        if k == 0:
            sg.start = -math.inf
        else:
            pm, ps = merged[k - 1]
            xa = float(x[pm].max())
            xb = float(x[m].min())
            if xb <= xa:
                sg.start = 0.5 * (xa + xb)
            else:
                sg.start = _choose_boundary(xa, xb, ps, sg, cues, speech, resolution)
                if xb - xa > 60.0:
                    ambiguous.append((ps.map(xa), sg.map(xb)))
            sg.start = (
                max(sg.start, final[-1].start + 1e-3) if final[-1].start > -math.inf else sg.start
            )
        final.append(sg)
    mapping = Mapping(final)

    # Windows that disagree with their segment deserve a closer look.
    win_line = {k: lab for k, lab in zip(win_ids, labels.tolist(), strict=True)}
    for i, k in enumerate(win_ids):
        if smax[i, 0] >= 1.0 and cost[i, win_line[k]] > 0.5:
            yk = y[win == k]
            ambiguous.append((float(yk.min()) - 60.0, float(yk.max()) + 60.0))

    conf, n_in, n_groups, med, stats = _fit_stats(mapping, arr, n_windows)
    return AlignResult(
        mapping,
        "whisper",
        conf,
        n_in,
        n_groups,
        med,
        ambiguous,
        details={**stats, "lines": L, "x_scale": x_scale},
    )


def _fit_stats(mapping: Mapping, arr: _AnchorArrays, n_windows: int):
    """Confidence and statistics of ``mapping`` against the anchors: one vote per
    transcript token (its candidates are alternatives), so repeated phrases do not
    dilute the inlier ratio. ``n_windows`` = windows transcribed before verification.
    Returns ``(confidence, inliers, groups, median, details)``.
    """
    x = arr.x_for(mapping)
    y, w, win, grp = arr.y, arr.w, arr.win, arr.grp
    signed = y - mapping.map_array(x)
    res = np.abs(signed)
    inl = res < INLIER_THRESHOLD
    n_groups = int(grp.max()) + 1
    g_weight = np.bincount(grp, weights=w, minlength=n_groups)
    g_in = np.bincount(grp, weights=inl.astype(float), minlength=n_groups) > 0
    n_in = int(g_in.sum())
    ratio = float(g_weight[g_in].sum() / g_weight.sum()) if g_weight.sum() > 0 else 0.0
    # Residual term: a window's words share a timestamp bias of ~WINDOW_BIAS on real
    # audio (that is not misfit of the line), so up to that much of each window's
    # median residual is removed first. A line off by more still shows the excess.
    adj = signed.copy()
    for k in set(win[inl].tolist()):
        mk = inl & (win == k)
        adj[mk] -= np.clip(np.median(signed[mk]), -WINDOW_BIAS, WINDOW_BIAS)
    med = float(np.median(np.abs(adj[inl]))) if inl.any() else 99.0
    with_inliers = set(win[inl].tolist()) if inl.any() else set()
    # Windows numbered >= n_windows were added by verification probes: they count
    # only when they contribute inliers (a probe on music must not lower coverage).
    probes_used = sum(1 for k in with_inliers if n_windows and k >= n_windows)
    base_windows = {k for k in set(win.tolist()) if not n_windows or k < n_windows}
    n_win = max(n_windows, len(base_windows), 1) + probes_used
    coverage = min(1.0, len(with_inliers) / n_win)
    conf = (
        (1.0 - math.exp(-n_in / 12.0))
        * (0.35 + 0.65 * ratio)
        * (0.4 + 0.6 * coverage)
        * math.exp(-max(0.0, med - 0.25) / 0.4)
        * (0.95 ** (len(mapping.segments) - 1))
    )
    if n_in < 6:
        conf = min(conf, 0.2)
    conf = float(max(0.0, min(1.0, conf)))
    return conf, n_in, n_groups, med, {"ratio": ratio, "coverage": coverage}


def speech_onsets(speech: np.ndarray, resolution: float = 0.01, min_silence: float = 0.2):
    """Times (s) where speech starts after at least ``min_silence`` of non-speech."""
    if speech.size == 0:
        return np.zeros(0)
    d = np.diff(np.concatenate([[0], speech.astype(np.int8)]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([speech.astype(np.int8), [0]])) == -1)
    keep = []
    prev_end = -(10**9)
    for st, en in zip(starts, ends, strict=True):
        if (st - prev_end) * resolution >= min_silence:
            keep.append(st)
        prev_end = en + 1
    return np.asarray(keep, dtype=float) * resolution


def _onset_deltas(
    mapping: Mapping, cue_starts: np.ndarray, onsets: np.ndarray, search: float
) -> tuple[np.ndarray, np.ndarray]:
    """For every cue: (nearest speech onset - mapped start, |delta| <= search)."""
    starts = mapping.map_array(cue_starts)
    if onsets.size == 0:
        return np.zeros_like(starts), np.zeros(starts.shape, dtype=bool)
    idx = np.searchsorted(onsets, starts)
    lo = onsets[np.clip(idx - 1, 0, onsets.size - 1)]
    hi = onsets[np.clip(idx, 0, onsets.size - 1)]
    nearest = np.where(np.abs(lo - starts) <= np.abs(hi - starts), lo, hi)
    delta = nearest - starts
    return delta, np.abs(delta) <= search


def _median_se(d: np.ndarray) -> tuple[float, float, float]:
    """(median, MAD, standard error of the median for a ~normal core)."""
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, 1.858 * max(mad, 0.01) / math.sqrt(d.size)


def _shift_mapping(mapping: Mapping, shifts: Sequence[float] | float) -> Mapping:
    if isinstance(shifts, (int, float)):
        shifts = [float(shifts)] * len(mapping.segments)
    return Mapping(
        [
            Segment(s.start, s.scale, s.offset + d, s.anchors, s.knots)
            for s, d in zip(mapping.segments, shifts, strict=True)
        ]
    )


def refine_with_speech(
    mapping: Mapping,
    cues: Sequence[Cue],
    speech: np.ndarray,
    resolution: float = 0.01,
    search: float = 0.5,
    max_shift: float = 0.4,
    max_se: float = 0.04,
) -> tuple[Mapping, float]:
    """Remove a constant residual bias (Whisper word timestamps are typically a bit
    late/early depending on the model) by snapping mapped cue starts to nearby speech
    onsets. Applied when the median shift is *precise*: enough cues, standard error of
    the median ≤ ``max_se`` and a coherent core (MAD ≤ 0.25 s; real dialogue has a
    0.1-0.2 s MAD, so a spread gate alone rejects useful shifts).

    Returns ``(mapping, shift_applied)``.
    """
    onsets = speech_onsets(speech, resolution)
    if onsets.size < 5 or not cues:
        return mapping, 0.0
    delta, ok = _onset_deltas(mapping, np.array([c.start for c in cues]), onsets, search)
    if int(ok.sum()) < max(8, int(0.25 * len(cues))):
        return mapping, 0.0
    shift, mad, se = _median_se(delta[ok])
    if mad > 0.25 or se > max_se or abs(shift) > max_shift or abs(shift) < 0.02:
        return mapping, 0.0
    return _shift_mapping(mapping, shift), shift


# --------------------------------------------------------------------------------------
# Bisection verification and local refinement
# --------------------------------------------------------------------------------------

VERIFY_TOLERANCE = 0.25  # s, disagreement of a verification window with its segment
MIN_SEGMENT = 120.0  # s (subtitle clock), no split below 2x this
MAX_DEPTH = 5
LOCAL_MAX_SHIFT = 0.5  # s, bound of each local correction
WINDOW_BIAS = 0.2  # s, shared timestamp bias of one Whisper window (real-world ~0.2-0.4)
WOBBLE_PRIOR = 0.2  # s, prior std-dev of slow local wobble (shrinkage)
WOBBLE_STEP = 120.0  # s (subtitle clock) between correction knots
WOBBLE_HALF_WIDTH = 120.0  # s, neighbourhood of a knot


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(int(np.searchsorted(c, 0.5 * c[-1])), v.size - 1)])


def _best_per_group(r: np.ndarray, grp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Within ``mask``, keep only the candidate with the smallest |r| of each group."""
    idx = np.flatnonzero(mask)
    out = np.zeros(mask.shape, dtype=bool)
    if idx.size == 0:
        return out
    order = idx[np.lexsort((np.abs(r[idx]), grp[idx]))]
    first = np.ones(order.size, dtype=bool)
    first[1:] = grp[order][1:] != grp[order][:-1]
    out[order[first]] = True
    return out


def _window_centres(arr: _AnchorArrays) -> dict[int, float]:
    # every candidate of a window has its audio time inside that window
    return {int(k): float(0.5 * (arr.y[arr.win == k].min() + arr.y[arr.win == k].max()))
            for k in np.unique(arr.win)}  # fmt: skip


def _window_verdict(
    arr: _AnchorArrays, k: int, seg: Segment, tol: float, min_groups: int = 6
) -> tuple[str, float]:
    """Does window ``k`` agree with ``seg``? → ("agree"|"disagree"|"unknown", deviation).

    The window's own consensus is the mode of the residuals of *all* its candidate
    anchors (whatever their distance from the line), kept if ≥ ``min_groups``
    transcript tokens support it. Disagreement: that consensus is more than ``tol``
    off the line (10 s off is a disagreement, not missing evidence), or the line
    explains less than half of what the consensus explains. "unknown" means the
    window has no coherent evidence at all (music, silence, garbled transcript).
    """
    m = arr.win == k
    if not m.any():
        return "unknown", 0.0
    r_all = arr.y - seg.line(arr.x(seg.scale))
    rr, ww, gg = r_all[m], arr.w[m], arr.grp[m]
    if len(np.unique(gg)) < min_groups:
        return "unknown", 0.0
    kern = np.clip(1.0 - ((rr[None, :] - rr[:, None]) / 0.3) ** 2, 0, None)
    sup = (ww[None, :] * kern).sum(1)
    mode = float(rr[int(np.argmax(sup))])
    near = np.flatnonzero(np.abs(rr - mode) < 0.5)
    # one candidate per transcript token: the closest to the mode
    near = near[np.lexsort((np.abs(rr[near] - mode), gg[near]))]
    keep = np.ones(near.size, dtype=bool)
    keep[1:] = gg[near][1:] != gg[near][:-1]
    core = near[keep]
    if core.size < min_groups:
        return "unknown", 0.0
    dev = _wmedian(rr[core], ww[core])
    b0 = _best_per_group(rr, gg, np.abs(rr) < 0.5)
    sup0 = float((ww[b0] * np.clip(1.0 - (rr[b0] / 0.3) ** 2, 0, None)).sum())
    supm = float((ww[core] * np.clip(1.0 - ((rr[core] - mode) / 0.3) ** 2, 0, None)).sum())
    inconsistent = sup0 < 0.5 * supm
    return ("disagree" if abs(dev) > tol or inconsistent else "agree"), dev


def _split_point(cues: Sequence[Cue] | None, lo: float, hi: float) -> float:
    """Best cue gap near the middle of [lo, hi] (subtitle clock): the largest gap
    between consecutive cues, discounted with the distance from the midpoint."""
    mid = 0.5 * (lo + hi)
    q = 0.25 * (hi - lo)
    if not cues:
        return mid
    best, best_t = -math.inf, mid
    prev_end = -math.inf
    for c in cues:
        if mid - q < c.start < mid + q and prev_end > -math.inf:
            gap = c.start - prev_end
            score = gap * (1.0 - 0.5 * abs(c.start - mid) / q)
            if score > best:
                best, best_t = score, c.start
        prev_end = max(prev_end, c.end)
    return best_t


def _refit_range(
    arr: _AnchorArrays,
    lo: float,
    hi: float,
    parent: Segment,
    dom_s: float,
    rng: np.random.Generator,
    gate: float = 5.0,
) -> tuple[Segment, np.ndarray] | None:
    """Fit ``[lo, hi)`` (subtitle clock) on its own: RANSAC line among candidates
    within ``gate`` s of the parent line selects the inliers, then a fixed known
    ratio is fitted (see below).
    Returns the segment and its inlier mask (one candidate per transcript token)."""
    xp = arr.x(parent.scale)
    rp = arr.y - parent.line(xp)
    m = (arr.x0 >= lo) & (arr.x0 < hi) & (np.abs(rp) < gate)
    if len(np.unique(arr.grp[m])) < 6:
        return None
    found = _ransac_line(xp[m], arr.y[m], arr.w[m], rng)
    if found is None:
        return None
    s, o, _sup = found
    x = arr.x(s)
    r = arr.y - (s * x + o)
    inl = _best_per_group(r, arr.grp, m & (np.abs(r) < INLIER_THRESHOLD))
    if inl.sum() < 6:
        return None
    # Known ratios only (a free scale over a few windows follows local wobble and
    # extrapolates badly): the dominant scale, the parent's, and the best-fitting known
    # ratio (prior towards 1.0). Judged per window (mean |median residual|); 1.0, then
    # the dominant scale, win near-ties.
    best_s, best_o = _best_candidate_scale(x[inl], arr.y[inl], arr.w[inl], s, o)
    options = {best_s: best_o}
    for sc in (dom_s, parent.scale, 1.0):
        if sc not in options:
            xs = arr.x(sc)
            options[sc] = _irls(xs[inl], arr.y[inl], arr.w[inl], sc,
                                float(np.median((arr.y - sc * xs)[inl])), True)[1]  # fmt: skip
    per_option = {}
    for sc, of in options.items():
        rr = arr.y - (sc * arr.x(sc) + of)
        mm = _best_per_group(rr, arr.grp, m & (np.abs(rr) < INLIER_THRESHOLD))
        per_option[sc] = {
            int(k): abs(float(np.median(rr[mm & (arr.win == k)])))
            for k in np.unique(arr.win[mm])
            if (mm & (arr.win == k)).sum() >= 3
        }
    # windows with evidence under any option; one without inliers costs a full second
    wins = set().union(*per_option.values())
    if not wins:
        return None
    scored = []
    for sc, of in options.items():
        e = float(np.mean([per_option[sc].get(k, INLIER_THRESHOLD) for k in wins]))
        e -= 0.02 if sc == 1.0 else (0.01 if sc == dom_s else 0.0)
        scored.append((e, sc, of))
    _e, s2, o2 = min(scored)
    x2 = arr.x(s2)
    r2 = arr.y - (s2 * x2 + o2)
    inl = _best_per_group(r2, arr.grp, m & (np.abs(r2) < INLIER_THRESHOLD))
    if inl.sum() < 6:
        return None
    return Segment(lo, s2, o2, int(inl.sum())), inl


def _window_spread(arr: _AnchorArrays, seg: Segment, mask: np.ndarray) -> tuple[float, int]:
    """Robust std-dev of per-window median residuals under ``seg`` (≥ the within-window
    standard error) and the number of windows with evidence."""
    x = arr.x(seg.scale)
    r = arr.y - seg.line(x)
    meds, ses = [], []
    for k in np.unique(arr.win[mask]):
        mk = mask & (arr.win == k)
        if mk.sum() >= 3:
            med, _mad, se = _median_se(r[mk])
            meds.append(med)
            ses.append(se)
    if not meds:
        return WINDOW_BIAS, 0
    d = np.array(meds)
    spread = 1.4826 * float(np.median(np.abs(d - np.median(d)))) if d.size >= 3 else WINDOW_BIAS
    return max(spread, float(np.median(ses)), 0.03), len(meds)


def _range_error(
    arr: _AnchorArrays, lo: float, hi: float, parts: list[tuple[float, float, Segment]]
) -> float:
    """Mean |per-window median residual| over [lo, hi) under piecewise ``parts``."""
    meds = []
    for a, b, seg in parts:
        r = arr.y - seg.line(arr.x(seg.scale))
        m = _best_per_group(
            r, arr.grp, (arr.x0 >= a) & (arr.x0 < b) & (np.abs(r) < INLIER_THRESHOLD)
        )
        for k in np.unique(arr.win[m]):
            mk = m & (arr.win == k)
            if mk.sum() >= 3:
                meds.append(abs(float(np.median(r[mk]))))
    return float(np.mean(meds)) if meds else 0.0


def _verifies(arr: _AnchorArrays, lo: float, hi: float, seg: Segment, tol: float) -> bool:
    """Positive verification of ``seg`` over [lo, hi]: at least one transcription
    window inside agrees with it, and every window inside that has coherent evidence
    agrees (no evidence is not agreement, and a window far off the line disagrees)."""
    a_lo, a_hi = sorted((seg.map(lo), seg.map(hi)))
    verdicts = [
        _window_verdict(arr, k, seg, tol)[0]
        for k, c in _window_centres(arr).items()
        if a_lo <= c <= a_hi
    ]
    return "agree" in verdicts and "disagree" not in verdicts


def _change_point(arr: _AnchorArrays, p: _Piece) -> tuple[float, float] | None:
    """Best single change point of the per-window median residuals under ``p.seg``
    (L1 cost, ≥ 2 windows per side). Returns the subtitle-time gap between the last
    window before and the first window after it, or None."""
    x = arr.x(p.seg.scale)
    r = arr.y - p.seg.line(x)
    m = _best_per_group(
        r, arr.grp, (arr.x0 >= p.lo) & (arr.x0 < p.hi) & (np.abs(r) < INLIER_THRESHOLD)
    )
    rows = []
    for k in np.unique(arr.win[m]):
        mk = m & (arr.win == k)
        if mk.sum() >= 3:
            rows.append((float(arr.x0[mk].min()), float(arr.x0[mk].max()), float(np.median(r[mk]))))
    rows.sort()
    if len(rows) < 4:
        return None
    d = np.array([t[2] for t in rows])
    best = None
    for i in range(2, len(rows) - 1):
        a, b = d[:i], d[i:]
        cost = float(np.abs(a - np.median(a)).sum() + np.abs(b - np.median(b)).sum())
        if best is None or cost < best[0]:
            best = (cost, i)
    if best is None:
        return None
    i = best[1]
    lo, hi = rows[i - 1][1], rows[i][0]
    return (lo, hi) if hi > lo else None


def _fold_votes(
    own_v: dict[int, tuple[str, float]],
    nb_v: dict[int, tuple[str, float]],
    centres: dict[int, float],
    tol: float,
) -> tuple[int, int]:
    """Votes (own, neighbour) of the windows inside a short segment. Each window
    votes for the line its consensus is closer to, if that line is within 2x ``tol``
    (a fold compares two lines; it is not a verification against one). Overlapping
    windows (centres < 30 s apart) cover the same audio and share its timestamp bias,
    so they cast one vote together (their majority)."""
    prefs = []
    for k in sorted(own_v, key=lambda k: centres[k]):
        (vo, do), (vn, dn) = own_v[k], nb_v[k]
        if vo == "unknown" or vn == "unknown":
            continue
        best = min(abs(do), abs(dn))
        if best <= 2 * tol:
            prefs.append((centres[k], 1 if abs(do) <= abs(dn) else -1))
    own = nb = 0
    i = 0
    while i < len(prefs):
        j = i
        while j + 1 < len(prefs) and prefs[j + 1][0] - prefs[j][0] < 30.0:
            j += 1
        tally = sum(v for _c, v in prefs[i : j + 1])
        own += tally > 0
        nb += tally < 0
        i = j + 1
    return own, nb


@dataclass
class _Piece:
    lo: float  # subtitle clock
    hi: float
    seg: Segment
    depth: int = 0
    split: bool = False  # created by a split (its start is ours to place)


def subdivide(
    fit: AlignResult,
    anchors: Sequence[Anchor],
    cues: Sequence[Cue],
    probe=None,
    speech: np.ndarray | None = None,
    resolution: float = 0.01,
    n_windows: int = 0,
    tol: float = VERIFY_TOLERANCE,
    min_len: float = MIN_SEGMENT,
    max_depth: int = MAX_DEPTH,
    seed: int = 0,
) -> tuple[AlignResult, list[Anchor]]:
    """Bisection verification of a fit.

    Each segment's mapping is checked at its midpoint against a transcription window
    there: ``probe(audio_time, near)`` returns ``(anchors, transcribed)``, transcribing
    a window centred at ``audio_time`` if none lies within ``near`` s and the window
    budget allows (``probe=None``: only existing windows are used). A probe window
    without coherent evidence (music, silence) is not agreement: other positions are
    tried while the budget lasts, and a segment with no evidence at all is left as it
    is and counted in ``details["verify_unverified"]``. If the
    window disagrees (see :func:`_window_verdict`), the segment is split at the best
    cue gap near its midpoint and both halves are refitted on their own (known-ratio
    snapping, scale prior, short halves share the dominant scale). A split is kept
    only if the halves differ by more than ``tol`` plus twice the standard error
    implied by the spread of per-window residuals (a single window's timestamp bias
    is not a section of different timing). Recursion stops at ``min_len``,
    ``max_depth`` or when the probe has nothing new; finally, adjacent segments that
    agree within ``tol`` are merged.

    Returns ``(fit, anchors)`` (anchors include probe windows).
    """
    if fit.method != "whisper" or not cues or fit.anchors < 6:
        return fit, list(anchors)
    rng = np.random.default_rng(seed)
    cue_starts = [c.start for c in cues]
    first, last = min(cue_starts), max(c.end for c in cues)
    segs = fit.mapping.segments
    dom_s = fit.mapping.dominant(cue_starts).scale
    queue: list[_Piece] = []
    for i, s in enumerate(segs):
        lo = first if i == 0 else max(s.start, first)
        hi = last if i + 1 == len(segs) else min(segs[i + 1].start, last)
        if hi > lo:
            queue.append(_Piece(lo, hi, s))
    if not queue:
        return fit, list(anchors)
    cur = list(anchors)
    arr = _AnchorArrays.build(cur)
    done: list[_Piece] = []
    checks = splits = unverified = empty_probes = 0
    changed = False

    def run_probe(centre: float, near: float) -> set[int] | None:
        """Probe; None if nothing was transcribed, else the new windows' ids that
        have coherent evidence (empty: the window gave nothing usable)."""
        nonlocal cur, arr
        if probe is None:
            return None
        before = set(np.unique(arr.win).tolist())
        new, transcribed = probe(centre, near)
        if not transcribed:
            return None
        cur = list(new)
        arr = _AnchorArrays.build(cur)
        ids = set(np.unique(arr.win).tolist()) - before
        return ids

    def probe_with_evidence(centres: list[float], near: float, seg: Segment) -> None:
        """Probe the first position; if the window yields no coherent evidence (music,
        silence), retry the next positions while the budget lasts."""
        nonlocal empty_probes
        for n, c in enumerate(centres):
            ids = run_probe(c, near)
            if ids is None:
                if n == 0:
                    return  # a window is already near the first position (or no budget)
                continue
            if any(_window_verdict(arr, k, seg, tol)[0] != "unknown" for k in ids):
                return
            empty_probes += 1

    while queue:
        p = queue.pop(0)
        if p.hi - p.lo < 2 * min_len or p.depth >= max_depth:
            done.append(p)
            continue
        mid = 0.5 * (p.lo + p.hi)
        centre = p.seg.map(mid)
        near = min(90.0, max(30.0, (p.hi - p.lo) * p.seg.scale / 8.0))
        span = 0.25 * (p.hi - p.lo) * p.seg.scale
        probe_with_evidence([centre, centre + span, centre - span], near, p.seg)
        if not len(arr.anchors):
            done.append(p)
            unverified += 1
            continue
        # The midpoint window, plus every other window already inside the segment (a
        # compromise line through two different sections is right at its middle).
        centres = _window_centres(arr)
        a_lo, a_hi = sorted((p.seg.map(p.lo), p.seg.map(p.hi)))
        inside = [j for j, c in centres.items() if a_lo <= c <= a_hi]
        k = min(centres, key=lambda j: abs(centres[j] - centre))
        if abs(centres[k] - centre) <= near + 15.0 and k not in inside:
            inside.append(k)
        checks += 1
        verdicts = [_window_verdict(arr, j, p.seg, tol)[0] for j in inside]
        if "disagree" not in verdicts and "agree" not in verdicts:
            unverified += 1  # no coherent evidence anywhere in the segment: leave it
            done.append(p)
            continue
        if "disagree" not in verdicts:
            # Verified; still prefer a known-ratio refit that halves the per-window
            # error (an initial compromise line can stay within the tolerance).
            whole = _refit_range(arr, p.lo, p.hi, p.seg, dom_s, rng)
            if whole is not None:
                e_old = _range_error(arr, p.lo, p.hi, [(p.lo, p.hi, p.seg)])
                e_new = _range_error(arr, p.lo, p.hi, [(p.lo, p.hi, whole[0])])
                if e_new <= 0.5 * e_old and e_old - e_new > 0.05:
                    ws = whole[0]
                    p = _Piece(p.lo, p.hi, Segment(p.seg.start, ws.scale, ws.offset,
                                                   ws.anchors), p.depth, p.split)  # fmt: skip
                    changed = True
            done.append(p)
            continue
        # Candidate cuts: the best cue gap near the midpoint (bisection), and the cue
        # gap at the change point of the per-window residuals (sections at 1/3 and
        # 2/3 leave both midpoint halves mixed). The one explaining more wins.
        cuts = {_split_point(cues, p.lo, p.hi)}
        cp = _change_point(arr, p)
        if cp is not None:
            cuts.add(_split_point(cues, *cp))
        significant = False
        best = None
        e_parent = _range_error(arr, p.lo, p.hi, [(p.lo, p.hi, p.seg)])
        for cut in sorted(cuts):
            if cut - p.lo < min_len or p.hi - cut < min_len:
                continue
            left = _refit_range(arr, p.lo, cut, p.seg, dom_s, rng)
            right = _refit_range(arr, cut, p.hi, p.seg, dom_s, rng)
            if left is None or right is None:
                continue
            (ls_, lm), (rs_, rm) = left, right
            e_split = _range_error(arr, p.lo, p.hi, [(p.lo, cut, ls_), (cut, p.hi, rs_)])
            if best is None or e_split < best[0]:
                best = (e_split, cut, ls_, lm, rs_, rm)
        if best is not None:
            e_split, cut, ls, lm, rs, rm = best
            sl, nl = _window_spread(arr, ls, lm)
            sr, nr = _window_spread(arr, rs, rm)
            diff = max(abs(ls.line(t) - rs.line(t)) for t in (p.lo, cut, p.hi))
            se = max(sl, sr) * math.sqrt(1.0 / max(nl, 1) + 1.0 / max(nr, 1))
            # Significant: the halves differ beyond the tolerance plus noise, and the
            # split explains the disagreement: the error halves, or one half is
            # consistent on its own and the other is left to the recursion (a step
            # vanishes; a smooth wobble does not, that is refine_local's job).
            explains = e_split <= 0.5 * e_parent or (
                e_split <= 0.8 * e_parent
                and (_verifies(arr, p.lo, cut, ls, tol) or _verifies(arr, cut, p.hi, rs, tol))
            )
            significant = nl > 0 and nr > 0 and diff > tol + 2.0 * se and explains
        if not significant:
            # Not two sections: maybe the segment's own line is off (e.g. a compromise
            # scale); keep a refit of the whole range if it verifies where p did not.
            whole = _refit_range(arr, p.lo, p.hi, p.seg, dom_s, rng)
            if whole is not None and _verifies(arr, p.lo, p.hi, whole[0], tol):
                ws = whole[0]
                p = _Piece(p.lo, p.hi, Segment(p.seg.start, ws.scale, ws.offset, ws.anchors),
                           p.depth, p.split)  # fmt: skip
                changed = True
            done.append(p)
            continue
        splits += 1
        changed = True
        queue[0:0] = [
            _Piece(p.lo, cut, Segment(p.seg.start, ls.scale, ls.offset, ls.anchors),
                   p.depth + 1, p.split),
            _Piece(cut, p.hi, Segment(cut, rs.scale, rs.offset, rs.anchors), p.depth + 1, True),
        ]  # fmt: skip

    # Segments too short to split (typically one or two windows that fit_mapping set
    # apart): probe once more inside, and fold one into a neighbour when at least as
    # many of its windows agree with the neighbour as with the segment itself (ties
    # favour fewer segments).
    folded = 0
    folded_windows: set[int] = set()  # windows outvoted by a fold (biased timestamps)
    i = 0
    while len(done) > 1 and i < len(done):
        p = done[i]
        if p.hi - p.lo >= 2 * min_len:
            i += 1
            continue
        a_lo, a_hi = sorted((p.seg.map(p.lo), p.seg.map(p.hi)))
        if probe is not None and a_hi - a_lo >= 60.0 and len(arr.anchors):
            # independent evidence: the points of the segment farthest from any window
            # (one window's timestamps can be off by more than a second on its own)
            taken = np.array(list(_window_centres(arr).values()))
            grid = np.linspace(a_lo + 15.0, a_hi - 15.0, 32)
            dist = np.min(np.abs(grid[:, None] - taken[None, :]), axis=1)
            order = [float(grid[g]) for g in np.argsort(-dist)[:8]]
            # first the farthest point, then the farthest one away from it
            far = [order[0]] + [c for c in order[1:] if abs(c - order[0]) >= 30.0][:1]
            probe_with_evidence(far, 15.0, p.seg)
        centres = _window_centres(arr)
        inside = [k for k, c in centres.items() if a_lo <= c <= a_hi]
        checks += 1
        own_v = {k: _window_verdict(arr, k, p.seg, tol) for k in inside}
        if all(v[0] == "unknown" for v in own_v.values()):
            unverified += 1
        target = None
        for j in (i - 1, i + 1):
            if 0 <= j < len(done):
                nb_v = {k: _window_verdict(arr, k, done[j].seg, tol) for k in inside}
                own, votes = _fold_votes(own_v, nb_v, centres, tol)
                if votes >= max(own, 1) and (target is None or votes > target[1]):
                    target = (j, votes)
        if target is None:
            i += 1
            continue
        j = target[0]
        folded_windows |= {k for k, v in own_v.items() if v[0] == "agree"}
        a, b = done[min(i, j)], done[max(i, j)]
        nb = done[j].seg
        done[min(i, j) : max(i, j) + 1] = [
            _Piece(a.lo, b.hi, Segment(a.seg.start, nb.scale, nb.offset, nb.anchors),
                   max(a.depth, b.depth), a.split)
        ]  # fmt: skip
        folded += 1
        changed = True
        i = min(i, j)

    if not changed:
        fit.details.update(
            verify_checks=checks, verify_splits=0, verify_folded=0,
            verify_unverified=unverified, verify_empty_probes=empty_probes,
        )  # fmt: skip
        if len(cur) != len(anchors):  # new windows: refresh statistics
            conf, n_in, n_groups, med, stats = _fit_stats(fit.mapping, arr, n_windows)
            fit = AlignResult(fit.mapping, fit.method, conf, n_in, n_groups, med,
                              fit.ambiguous, fit.sparse, {**fit.details, **stats})  # fmt: skip
        return fit, cur

    # Merge adjacent pieces when one refit of both verifies against all their windows.
    merged = True
    while merged and len(done) > 1:
        merged = False
        for i in range(len(done) - 1):
            a, b = done[i], done[i + 1]
            ref = _refit_range(arr, a.lo, b.hi, a.seg, dom_s, rng)
            if ref is None or not _verifies(arr, a.lo, b.hi, ref[0], tol):
                continue
            seg = ref[0]
            done[i : i + 2] = [
                _Piece(a.lo, b.hi, Segment(a.seg.start, seg.scale, seg.offset, seg.anchors),
                       max(a.depth, b.depth), a.split)
            ]  # fmt: skip
            merged = True
            break

    # Boundaries of new splits: between the anchored regions of both sides, at a cue
    # gap (or by speech overlap when the difference is audible).
    x = arr.x_for(fit.mapping)
    final: list[Segment] = []
    for i, p in enumerate(done):
        seg = Segment(p.seg.start, p.seg.scale, p.seg.offset, p.seg.anchors)
        if i == 0:
            seg.start = -math.inf
        elif p.split:
            prev = final[-1]
            ra = np.abs(arr.y - prev.line(x)) < INLIER_THRESHOLD
            rb = np.abs(arr.y - seg.line(x)) < INLIER_THRESHOLD
            la = ra & (arr.x0 >= done[i - 1].lo) & (arr.x0 < p.lo)
            lb = rb & (arr.x0 >= p.lo) & (arr.x0 < p.hi)
            xa = float(arr.x0[la].max()) if la.any() else p.lo
            xb = float(arr.x0[lb].min()) if lb.any() else p.lo
            seg.start = (
                _choose_boundary(xa, xb, prev, seg, cues, speech, resolution) if xb > xa else p.lo
            )
        else:
            seg.start = p.lo if p.seg.start == -math.inf else p.seg.start
        if final and final[-1].start > -math.inf:
            seg.start = max(seg.start, final[-1].start + 1e-3)
        final.append(seg)
    mapping = Mapping(final)
    conf, n_in, n_groups, med, stats = _fit_stats(mapping, arr, n_windows)
    details = {**fit.details, **stats, "verify_checks": checks, "verify_splits": splits,
               "verify_folded": folded, "verify_unverified": unverified,
               "verify_empty_probes": empty_probes}  # fmt: skip
    # Safety net against a bad refit: never trade a fit for one that explains clearly
    # fewer anchors (folding a biased window legitimately drops a few) or explains them
    # worse (the confidence itself also pays a small per-segment penalty).
    # Windows outvoted by a fold are left out of the comparison: losing their
    # (biased) inliers is the point of the fold.
    kept = [a for a in cur if a.window not in folded_windows]
    ref = _AnchorArrays.build(kept) if folded_windows and len(kept) >= 3 else arr
    _c1, n_in1, _g1, med1, _s1 = _fit_stats(mapping, ref, n_windows)
    _c0, n_in0, _g0, med0, _s0 = _fit_stats(fit.mapping, ref, n_windows)
    if n_in1 < 0.9 * n_in0 or med1 > med0 + 0.05:
        details.update(verify_splits=0, verify_rejected=True)
        conf0, n_in0, n_groups0, med0, stats0 = _fit_stats(fit.mapping, arr, n_windows)
        return AlignResult(fit.mapping, fit.method, conf0, n_in0, n_groups0, med0,
                           fit.ambiguous, fit.sparse, {**details, **stats0}), cur  # fmt: skip
    return AlignResult(mapping, "whisper", conf, n_in, n_groups, med, [], [], details), cur


def refine_local(
    mapping: Mapping,
    anchors: Sequence[Anchor],
    cues: Sequence[Cue],
    speech: np.ndarray | None = None,
    resolution: float = 0.01,
    max_shift: float = LOCAL_MAX_SHIFT,
    wobble: bool = True,
) -> tuple[Mapping, dict]:
    """Local refinement of a fitted mapping, in three consistent steps:

    1. per segment: shift by the robust (weighted) median residual of its inlier
       anchors (Whisper word time - mapped subtitle token time), bounded ±``max_shift``;
    2. global: :func:`refine_with_speech` measures what is *left* against speech
       onsets (independent of Whisper's timestamp bias) and removes it if precise;
    3. per segment, if ``wobble``: smooth correction knots every ``WOBBLE_STEP`` s
       from neighbourhood medians of the remaining anchor residuals (per window, with
       ``WINDOW_BIAS`` allowed for a window's shared timestamp bias) and of speech-onset
       deltas, combined by precision and shrunk towards 0; linearly interpolated, so
       slow wobble is followed without per-cue jitter.

    Each step works on the residual of the previous one, so they cannot fight: the
    anchors decide differences *between* parts of the file, speech onsets the common
    absolute bias. The result is one continuous correction relative to the fitted
    lines, bounded by ±``max_shift`` in *total* (see :func:`_assemble_correction`).
    Returns ``(mapping, info)``.
    """
    info: dict = {"segment_shifts": [], "onset_shift": 0.0, "knots": 0}
    if not anchors or len(anchors) < 3:
        return mapping, info
    fitted = mapping
    arr = _AnchorArrays.build(anchors)
    cue_starts = np.array([c.start for c in cues]) if cues else np.zeros(0)
    first = float(cue_starts.min()) if cue_starts.size else -math.inf
    last = float(max(c.end for c in cues)) if cues else math.inf
    segs = mapping.segments

    def seg_range(i: int) -> tuple[float, float]:
        lo = -math.inf if i == 0 else segs[i].start
        hi = math.inf if i + 1 == len(segs) else segs[i + 1].start
        return lo, hi

    def residuals(seg: Segment, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        x = arr.x(seg.scale)
        r = arr.y - seg.map(x)
        m = (arr.x0 >= lo) & (arr.x0 < hi) & (np.abs(r) < INLIER_THRESHOLD)
        return r, _best_per_group(r, arr.grp, m)

    # 1. per-segment median residual
    shifts = []
    for i, seg in enumerate(segs):
        r, b = residuals(seg, *seg_range(i))
        d = 0.0
        if b.sum() >= 8:
            med0 = _wmedian(r[b], arr.w[b])
            core = b & (np.abs(r - med0) < 0.5)
            if core.sum() >= 8:
                d = max(-max_shift, min(max_shift, _wmedian(r[core], arr.w[core])))
        shifts.append(d)
    mapping = _shift_mapping(mapping, shifts)
    info["segment_shifts"] = shifts

    # 2. global bias against speech onsets
    onsets = np.zeros(0)
    if speech is not None and speech.size and cues:
        mapping, info["onset_shift"] = refine_with_speech(mapping, cues, speech, resolution)
        onsets = speech_onsets(speech, resolution)
    # 3. smooth neighbourhood corrections (relative knots per segment)
    segs = mapping.segments
    rel_knots: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(segs)
    if not wobble:
        return _assemble_correction(fitted, mapping, rel_knots, first, last, max_shift, info)
    delta, ok = (
        _onset_deltas(mapping, cue_starts, onsets, 0.5) if onsets.size >= 5 else (None, None)
    )
    for i, seg in enumerate(segs):
        lo, hi = seg_range(i)
        lo_c, hi_c = max(lo, first), min(hi, last)
        r, b = residuals(seg, lo, hi)
        # Both sources only contribute variation around their own baseline in this
        # segment (the absolute level was settled by steps 1-2).
        base_a = _wmedian(r[b], arr.w[b]) if b.sum() >= 8 else 0.0
        wins = []  # per window: (position, median, variance)
        for k in np.unique(arr.win[b]):
            mk = b & (arr.win == k)
            if mk.sum() >= 5:
                med, _mad, se = _median_se(r[mk])
                wins.append((float(np.median(arr.x0[mk])), med - base_a, se**2 + WINDOW_BIAS**2))
        in_seg = None
        base_o = 0.0
        if delta is not None:
            in_seg = ok & (cue_starts >= lo) & (cue_starts < hi)
            base_o = float(np.median(delta[in_seg])) if in_seg.sum() >= 8 else 0.0
        n_knots = int((hi_c - lo_c) // WOBBLE_STEP)
        if n_knots < 2:
            continue
        kx = lo_c + (np.arange(n_knots) + 0.5) * (hi_c - lo_c) / n_knots
        ky = np.zeros(n_knots)
        has = np.zeros(n_knots, dtype=bool)
        for j, t in enumerate(kx):
            est, prec = [], []
            for pos, med, var in wins:
                if abs(pos - t) <= WOBBLE_HALF_WIDTH:
                    est.append(med)
                    prec.append(1.0 / var)
            if in_seg is not None:
                m = in_seg & (np.abs(cue_starts - t) <= WOBBLE_HALF_WIDTH)
                if m.sum() >= 8:
                    med, mad, se = _median_se(delta[m])
                    if mad <= 0.3:
                        est.append(med - base_o)
                        prec.append(1.0 / se**2)
            if not est:
                continue
            p = np.array(prec)
            v = 1.0 / p.sum()
            e = float((np.array(est) * p).sum() * v)
            ky[j] = e * WOBBLE_PRIOR**2 / (WOBBLE_PRIOR**2 + v)  # shrink towards 0
            has[j] = True
        if has.sum() < 2:
            continue
        kx, ky = kx[has], ky[has]
        if ky.size >= 3:  # smooth: [1/4, 1/2, 1/4]
            ky = np.concatenate(
                [
                    [(2 * ky[0] + ky[1]) / 3],
                    0.25 * ky[:-2] + 0.5 * ky[1:-1] + 0.25 * ky[2:],
                    [(ky[-2] + 2 * ky[-1]) / 3],
                ]
            )
        if np.abs(ky).max() >= 0.02:
            rel_knots[i] = (kx, ky)
    return _assemble_correction(fitted, mapping, rel_knots, first, last, max_shift, info)


def _assemble_correction(
    fitted: Mapping,
    shifted: Mapping,
    rel_knots: list[tuple[np.ndarray, np.ndarray] | None],
    first: float,
    last: float,
    max_shift: float,
    info: dict,
) -> tuple[Mapping, dict]:
    """Turn the refinement steps into one correction curve c(t) relative to the fitted
    lines: per segment its shift (steps 1-2) plus its wobble knots (step 3), a shared
    knot at every segment boundary (the mean of the two sides) so c is continuous
    there, and every value clipped to ±``max_shift``. Since c is linear between knots
    and flat outside them, the *total* correction is bounded by ``max_shift`` and the
    mapping jumps at a boundary exactly as much as the fitted lines do: refinement
    alone never creates a backward jump (which ``retime`` would resolve by squeezing
    cues), only a genuine cut does."""
    segs0, segs1 = fitted.segments, shifted.segments
    n = len(segs0)
    base = [b.offset - a.offset for a, b in zip(segs0, segs1, strict=True)]
    pts: list[list[tuple[float, float]]] = []
    for i in range(n):
        lo = max(segs0[i].start, first) if i else first
        hi = min(segs0[i + 1].start, last) if i + 1 < n else last
        if rel_knots[i] is not None:
            kx, ky = rel_knots[i]
            pts.append([(float(x), base[i] + float(y)) for x, y in zip(kx, ky, strict=True)])
        elif math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            pts.append([(0.5 * (lo + hi), base[i])])
        else:
            pts.append([])
    for i in range(1, n):  # shared boundary knots
        t = segs0[i].start
        left = pts[i - 1][-1][1] if pts[i - 1] else base[i - 1]
        right = pts[i][0][1] if pts[i] else base[i]
        v = 0.5 * (left + right)
        pts[i - 1] = [q for q in pts[i - 1] if q[0] < t - 1e-6] + [(t, v)]
        pts[i] = [(t, v)] + [q for q in pts[i] if q[0] > t + 1e-6]
    out: list[Segment] = []
    total = 0.0
    for i, (s0, s1) in enumerate(zip(segs0, segs1, strict=True)):
        b = max(-max_shift, min(max_shift, base[i]))
        rel = tuple((float(t), round(max(-max_shift, min(max_shift, v)) - b, 4)) for t, v in pts[i])
        if not rel or max(abs(c) for _t, c in rel) < 1e-3:
            rel = ()
        total = max([total, abs(b)] + [abs(b + c) for _t, c in rel])
        out.append(Segment(s1.start, s0.scale, s0.offset + b, s1.anchors, rel))
        info["knots"] += len(rel)
    info["max_correction"] = total
    return Mapping(out), info


# --------------------------------------------------------------------------------------
# VAD fallback
# --------------------------------------------------------------------------------------


def _cue_mask(cues: Sequence[Cue], scale: float, resolution: float, length: int) -> np.ndarray:
    m = np.zeros(length, dtype=np.float32)
    for c in cues:
        a = int(round(c.start * scale / resolution))
        b = int(round(c.end * scale / resolution))
        a, b = max(a, 0), min(b, length)
        if b > a:
            m[a:b] = 1.0
    return m


def vad_align(
    speech: np.ndarray,
    cues: Sequence[Cue],
    resolution: float = 0.01,
    max_offset: float | None = None,
    scales: Sequence[float] = SCALE_CANDIDATES,
) -> AlignResult:
    """Global ``audio = scale*sub + offset`` by cross-correlating the speech mask with
    the subtitle-on mask (FFT), over framerate scale candidates."""
    cues = [c for c in cues if c.end > c.start]
    if not cues or speech.size == 0 or not speech.any():
        return AlignResult(Mapping.identity(), "none", 0.0, 0)
    n = speech.size
    audio_dur = n * resolution
    if max_offset is None:
        max_offset = min(1200.0, max(60.0, 0.5 * audio_dur))
    sp = speech.astype(np.float32) * 2.0 - 1.0  # +1 speech, -1 non-speech
    sub_end = max(c.end for c in cues)
    per_scale = []  # (score, peak, scale, lag, vals, lags)
    for s in scales:
        m_len = int(math.ceil(sub_end * s / resolution)) + 1
        sub = _cue_mask(cues, s, resolution, m_len)
        on = float(sub.sum())
        if on <= 0:
            continue
        size = 1 << int(math.ceil(math.log2(n + m_len)))
        F = np.fft.rfft(sp, size) * np.conj(np.fft.rfft(sub, size))
        corr = np.fft.irfft(F, size)  # corr[lag] = sum_t sub[t]*sp[t+lag]
        max_lag = int(max_offset / resolution)
        lags = np.concatenate(
            [np.arange(0, min(max_lag, n) + 1), np.arange(-min(max_lag, m_len), 0)]
        )
        vals = corr[lags % size] / on
        k = int(np.argmax(vals))
        # Height above this scale's own baseline, with the same prior towards no drift
        # as _best_candidate_scale: more candidates must not mean more chances for a
        # noise maximum to win.
        height = float(vals[k] - np.median(vals))
        score = height / (0.98 if s == 1.0 else 1.0 + 2.0 * abs(s - 1.0))
        per_scale.append((score, float(vals[k]), s, int(lags[k]), vals, lags))
    if not per_scale:
        return AlignResult(Mapping.identity(), "none", 0.0, 0)
    per_scale.sort(key=lambda t: t[0], reverse=True)
    score, peak, s, lag, vals, lags = per_scale[0]
    away = np.abs(lags - lag) * resolution > 3.0
    second = float(vals[away].max()) if away.any() else -1.0
    base = float(np.median(vals))
    spread = float(np.median(np.abs(vals - base))) * 1.4826 + 1e-6
    z = (peak - base) / spread
    prominence = (peak - second) / max(peak - base, 1e-6)
    # Across scales: the runner-up among scales that map the file distinguishably
    # differently (> 3 s apart at its end) must be clearly lower.
    rivals = [t[0] for t in per_scale[1:] if abs(t[2] - s) * sub_end > 3.0]
    cross = (score - rivals[0]) / max(score, 1e-6) if rivals else 1.0
    conf = (
        min(1.0, max(0.0, (peak - 0.05) / 0.3))
        * min(1.0, max(0.0, min(prominence, 2.0 * cross) / 0.3))
        * min(1.0, max(0.0, (z - 3.0) / 4.0))
    )
    offset = lag * resolution
    return AlignResult(
        Mapping.linear(s, offset),
        "vad",
        float(conf),
        0,
        details={
            "peak": peak,
            "second": second,
            "z": z,
            "prominence": prominence,
            "cross_scale": cross,
        },
    )


# --------------------------------------------------------------------------------------
# Applying a mapping
# --------------------------------------------------------------------------------------


def retime(
    times: Sequence[tuple[float, float]], mapping: Mapping, min_duration: float = 0.2
) -> list[tuple[float, float]]:
    """Map ``(start, end)`` pairs (seconds); the result is in input order and nothing
    is dropped. Each cue uses the segment of its start and durations scale with it.

    Afterwards, in original start order:

    * start order is preserved: a cue mapped after a later cue (backwards cut) is
      pulled to ``min_duration`` before it. This cascades, so the cues before the
      cut that land in the region claimed by the next segment become ~0.2 s flashes;
      that only happens where the mapping itself has a backwards cut, i.e. the
      subtitle has content the audio lacks (or the cut was misplaced);
    * nothing starts before 0, and cues mapped to >= 0 are never moved: cues mapped
      before 0 get ordered slots in [0, first non-negative start), at most
      ``min_duration`` apart. A cue straddling 0 keeps its true end; one entirely
      before 0 gets a minimal duration (both are extended to that minimum only);
    * a cue ends at or before the next cue's start unless the two overlapped in the
      original (those overlaps are intentional), and every duration is positive.
    """
    n = len(times)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (times[i][0], times[i][1], i))
    starts: list[float] = []
    durs: list[float] = []
    for i in order:
        st, en = times[i]
        seg = mapping.segment_for(st)
        starts.append(seg.map(st))
        durs.append(max(en - st, 0.0) * seg.scale)

    # Backwards cut: keep start order by pulling earlier cues back (cascades).
    for k in range(n - 2, -1, -1):
        if starts[k] > starts[k + 1]:
            starts[k] = starts[k + 1] - min_duration

    # Negative starts (a prefix, starts are non-decreasing now): squeeze them into
    # ordered slots in [0, first non-negative start), at most ``min_duration`` apart.
    # Cues mapped to >= 0 are never moved; with no room the prefix collapses (tiny
    # durations / overlapping each other) instead.
    m = 0
    while m < n and starts[m] < 0.0:
        m += 1
    if m:
        room = starts[m] if m < n else math.inf
        step = min(min_duration, room / m)
        minimal = step if step > 0.0 else min_duration
        for k in range(m):
            true_end = starts[k] + durs[k]
            starts[k] = k * step
            # keep the true end (straddling 0); only extend up to the minimal duration
            durs[k] = max(true_end - starts[k], minimal)

    out: list[tuple[float, float]] = [(0.0, 0.0)] * n
    for k, i in enumerate(order):
        s, e = starts[k], starts[k] + durs[k]
        nxt = math.inf
        if k + 1 < n:
            j = order[k + 1]
            orig_overlap = times[i][1] > times[j][0] + 1e-3
            if not orig_overlap:
                nxt = starts[k + 1]
        if e > nxt:
            e = nxt
        if e <= s:
            e = min(s + min_duration, nxt) if nxt > s else s + min_duration
        out[i] = (s, e)
    return out


def apply_mapping(subs, mapping: Mapping) -> None:
    """Retime a ``pysubs2.SSAFile`` in place (all events, including comments)."""
    times = [(ev.start / 1000.0, ev.end / 1000.0) for ev in subs]
    for ev, (s, e) in zip(subs, retime(times, mapping), strict=True):
        ev.start = int(round(s * 1000))
        ev.end = int(round(e * 1000))
