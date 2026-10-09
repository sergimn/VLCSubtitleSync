"""Alignment core on synthetic subtitles + a fake transcriber (no audio, no models)."""

import math

import numpy as np
import pysubs2
import pytest

from vlcsubsync.align import (
    Anchor,
    Cue,
    Mapping,
    Segment,
    apply_mapping,
    find_anchors,
    fit_mapping,
    refine_local,
    refine_with_speech,
    retime,
    subdivide,
    subtitle_tokens,
    vad_align,
)
from vlcsubsync.config import Config
from vlcsubsync.sync import pick_windows

MEDIAN_MAX = 0.15
P95_MAX = 0.4


def _align(synth, transform, duration=45 * 60, seed=1, speech=None, **fake_kw):
    script = synth.script(duration - 30, seed=seed)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    spans = synth.speech_spans(script, transform)
    fake = synth.FakeTranscriber(spans, **fake_kw)
    k = Config().window_count(duration)
    starts = pick_windows(np.ones(int(duration)), duration, k)
    windows = [(s, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=s)) for s in starts]
    anchors = find_anchors(subtitle_tokens(cues), windows)
    fit = fit_mapping(anchors, cues, speech, n_windows=len(windows))
    pred = [s for s, _e in retime([(c.start, c.end) for c in cues], fit.mapping)]
    truth = np.array([transform(c.start) for c in cues])
    keep = truth >= 0  # cues shifted before t=0 are clamped, not comparable
    err = np.abs(np.array(pred) - truth)[keep]
    return fit, err


def _assert_accurate(err, median_max=None):
    median_max = MEDIAN_MAX if median_max is None else median_max
    assert np.median(err) < median_max, np.median(err)
    assert np.percentile(err, 95) < P95_MAX, np.percentile(err, 95)


@pytest.mark.parametrize("offset", [0.5, -0.5, 2.0, -7.0, 30.0, 90.0, -90.0])
def test_pure_offset(synth, offset):
    fit, err = _align(synth, lambda t: t + offset, duration=20 * 60)
    _assert_accurate(err)
    assert len(fit.mapping.segments) == 1
    seg = fit.mapping.segments[0]
    assert seg.scale == pytest.approx(1.0)
    assert seg.offset == pytest.approx(offset, abs=0.15)
    assert fit.confidence > 0.8


@pytest.mark.parametrize(
    "scale", [25 / 23.976, 23.976 / 25, 29.97 / 23.976, 23.976 / 29.97, 29.97 / 25, 25 / 29.97]
)
def test_framerate_drift(synth, scale):
    fit, err = _align(synth, lambda t: t * scale)
    # In-cue word times follow the fitted scale (speech rate is nominal in the audio
    # clock), so even 0.8x compression meets the normal tolerance.
    _assert_accurate(err)
    assert fit.mapping.segments[0].scale == pytest.approx(scale, abs=2e-4)


def test_offset_plus_drift(synth):
    fit, err = _align(synth, lambda t: t * 25 / 23.976 - 12.5, drop=0.1)
    _assert_accurate(err)
    seg = fit.mapping.segments[0]
    assert seg.scale == pytest.approx(25 / 23.976, abs=2e-4)
    assert seg.offset == pytest.approx(-12.5, abs=0.2)


def test_unusual_drift_is_fitted_freely(synth):
    fit, err = _align(synth, lambda t: t * 1.013 + 1.0)
    _assert_accurate(err)
    assert fit.mapping.segments[0].scale == pytest.approx(1.013, abs=5e-4)


def test_noisy_and_missing_words(synth):
    fit, err = _align(
        synth, lambda t: t + 4.2, drop=0.35, noise=0.25, jitter=0.12, duration=30 * 60
    )
    _assert_accurate(err)
    assert fit.confidence > 0.6


def test_mid_file_cut_piecewise(synth):
    def f(t):
        return t + 3.0 if t < 1200 else t + 33.0

    fit, err = _align(synth, f, duration=40 * 60)
    segs = fit.mapping.segments
    assert len(segs) == 2
    assert segs[0].offset == pytest.approx(3.0, abs=0.15)
    assert segs[1].offset == pytest.approx(33.0, abs=0.15)
    assert 1000 < segs[1].start < 1450  # somewhere between the anchored regions
    # without audio, cues between the two anchored regions may land on either side
    assert np.median(err) < MEDIAN_MAX
    assert fit.ambiguous, "boundary uncertainty must be reported for adaptive windows"


def test_cut_boundary_uses_speech_mask(synth):
    def f(t):
        return t + 3.0 if t < 1200 else t + 33.0

    duration = 40 * 60
    script = synth.script(duration - 30, seed=1)
    spans = synth.speech_spans(script, f)
    res = 0.01
    speech = np.zeros(int(duration / res) + 100, dtype=bool)
    for a, b, _c in spans:
        speech[int(a / res) : int(b / res)] = True
    fit, err = _align(synth, f, duration=duration, speech=speech)
    assert len(fit.mapping.segments) == 2
    _assert_accurate(err)
    assert err.max() < 1.0


def test_two_cuts(synth):
    def f(t):
        return t + 2.0 if t < 900 else (t + 32.0 if t < 1800 else t + 14.0)

    fit, err = _align(synth, f, duration=45 * 60)
    offs = [s.offset for s in fit.mapping.segments]
    assert offs == pytest.approx([2.0, 32.0, 14.0], abs=0.2)
    assert np.median(err) < MEDIAN_MAX


def test_garbage_transcript_has_low_confidence(synth):
    fit, _err = _align(synth, lambda t: t + 3, garbage=True, duration=15 * 60)
    assert fit.confidence < 0.3


def test_deterministic(synth):
    a, _ = _align(synth, lambda t: t * 1.0427 + 3, drop=0.3, noise=0.2, duration=15 * 60)
    b, _ = _align(synth, lambda t: t * 1.0427 + 3, drop=0.3, noise=0.2, duration=15 * 60)
    assert a.mapping == b.mapping
    assert a.confidence == b.confidence


def test_too_few_anchors():
    fit = fit_mapping([Anchor(1.0, 2.0, 1.0), Anchor(5.0, 6.0, 1.0)])
    assert fit.method == "none" and fit.confidence == 0.0


def test_repeated_ngrams_are_candidates_with_shared_weight():
    from vlcsubsync.transcribe import Word

    cues = [Cue(i * 5.0, i * 5.0 + 2, "we have to go now") for i in range(10)]
    words = [Word(0.1 * k, 0.1 * k + 0.1, w) for k, w in enumerate("we have to go now".split())]
    anchors = find_anchors(subtitle_tokens(cues), [(100.0, words)])
    assert len({a.cue for a in anchors}) == 10  # every occurrence is a candidate
    first = [a for a in anchors if a.token == 0]
    assert len(first) == 10
    assert sum(a.weight for a in first) <= 1.0 + 1e-9


@pytest.mark.parametrize("transform", [lambda t: t + 2.5, lambda t: t * 25 / 23.976 - 4.0])
def test_repeated_content(synth, transform):
    """A short episode looped several times (every line recurs) must still align: the
    locally consistent occurrence wins."""
    base = synth.script(300, seed=11)
    script = []
    for k in range(9):
        script += [synth.Cue(c.start + k * 300, c.end + k * 300, c.text) for c in base]
    cues = [Cue(c.start, c.end, c.text) for c in script]
    spans = synth.speech_spans(script, transform)
    fake = synth.FakeTranscriber(spans, drop=0.2, noise=0.1)
    duration = 2700
    starts = pick_windows(np.ones(duration), duration, Config().window_count(duration))
    windows = [(s, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=s)) for s in starts]
    fit = fit_mapping(find_anchors(subtitle_tokens(cues), windows), cues, None, n_windows=11)
    assert len(fit.mapping.segments) == 1
    pred = np.array([s for s, _e in retime([(c.start, c.end) for c in cues], fit.mapping)])
    truth = np.array([transform(c.start) for c in cues])
    keep = truth >= 0
    _assert_accurate(np.abs(pred - truth)[keep])
    assert fit.confidence > 0.6


def test_recurring_phrases_in_normal_dialogue(synth):
    """Catch-phrases repeated throughout an episode don't create false segments."""
    script = synth.script(30 * 60, seed=3)
    for i in range(0, len(script), 9):
        script[i] = synth.Cue(script[i].start, script[i].end, "Previously on the show.")
    cues = [Cue(c.start, c.end, c.text) for c in script]

    def f(t):
        return t + 11.0

    fake = synth.FakeTranscriber(synth.speech_spans(script, f), drop=0.2)
    starts = pick_windows(np.ones(1800), 1800, 8)
    windows = [(s, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=s)) for s in starts]
    fit = fit_mapping(find_anchors(subtitle_tokens(cues), windows), cues, None, n_windows=8)
    assert len(fit.mapping.segments) == 1
    assert fit.mapping.segments[0].offset == pytest.approx(11.0, abs=0.15)


# --- VAD fallback -------------------------------------------------------------------


def _masks(synth, transform, duration, seed=4):
    script = synth.script(duration - 30, seed=seed)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    spans = synth.speech_spans(script, transform)
    res = 0.01
    speech = np.zeros(int(duration / res), dtype=bool)
    for a, b, _c in spans:
        speech[max(0, int(a / res)) : max(0, int(b / res))] = True
    return cues, speech


@pytest.mark.parametrize(
    "transform,scale",
    [
        (lambda t: t + 2.5, 1.0),
        (lambda t: t - 40.0, 1.0),
        (lambda t: t * 25 / 23.976 + 1.0, 25 / 23.976),
        (lambda t: t * 24 / 25 - 3.0, 24 / 25),
        (lambda t: t * 29.97 / 23.976 + 0.5, 29.97 / 23.976),
        (lambda t: t * 23.976 / 29.97 + 2.0, 23.976 / 29.97),
    ],
)
def test_vad_align(synth, transform, scale):
    cues, speech = _masks(synth, transform, 20 * 60)
    r = vad_align(speech, cues)
    assert r.method == "vad"
    seg = r.mapping.segments[0]
    assert seg.scale == pytest.approx(scale)
    pred = np.array([seg.map(c.start) for c in cues])
    truth = np.array([transform(c.start) for c in cues])
    keep = truth >= 0
    err = np.abs(pred - truth)[keep]
    _assert_accurate(err)
    # The mask correlation is weaker under NTSC-sized stretching: the right mapping is
    # found but stays below the apply gate, so the fallback errs on the safe side.
    assert r.confidence > (0.3 if abs(scale - 1.0) > 0.1 else 0.6)
    # the cross-scale term must not cut genuine matches
    assert r.details["cross_scale"] > 0.15


def test_vad_align_long_file_keeps_cross_scale(synth):
    """On a long file 1.0 and 24/23.976 map > 3 s apart and count as rivals; the
    right scale must still stand out."""
    cues, speech = _masks(synth, lambda t: t + 2.5, 52 * 60)
    r = vad_align(speech, cues)
    assert r.mapping.segments[0].scale == pytest.approx(1.0)
    assert r.confidence > 0.6 and r.details["cross_scale"] > 0.15


def test_vad_align_unrelated_is_low_confidence(synth):
    cues, _ = _masks(synth, lambda t: t, 20 * 60, seed=4)
    _, speech = _masks(synth, lambda t: t, 20 * 60, seed=99)
    r = vad_align(speech, cues)
    assert r.confidence < 0.5


@pytest.mark.parametrize("seed", range(20))
def test_sparse_anchors_do_not_pick_extreme_scale(seed):
    """A handful of noisy anchors at true scale 1.0 must not be fitted as NTSC drift."""
    from vlcsubsync.align import _best_candidate_scale

    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    x = np.sort(rng.uniform(0, 500, n))
    y = x + 3.0 + rng.normal(0, 0.4, n)
    w = np.ones(n)
    s, o = _best_candidate_scale(x, y, w, 1.0, 3.0)
    assert abs(s - 1.0) < 0.1, (s, o)


def test_vad_align_empty():
    assert vad_align(np.zeros(100, bool), [Cue(0, 1, "x")]).method == "none"


@pytest.mark.parametrize("seed", [99, 7, 13, 21, 42, 5, 6])
def test_vad_align_unrelated_across_scales(synth, seed):
    """More scale candidates must not raise the odds of a noise maximum: the winner
    is compared with the best distinguishable rival scale."""
    cues, _ = _masks(synth, lambda t: t, 20 * 60, seed=4)
    _, speech = _masks(synth, lambda t: t, 20 * 60, seed=seed)
    r = vad_align(speech, cues)
    assert r.confidence < 0.3
    assert r.details["cross_scale"] < 0.3


# --- applying a mapping ---------------------------------------------------------------


def _check_retimed(times, out, min_gap=0.0):
    """Invariants of :func:`retime`: nothing dropped, starts ≥ 0 and sorted in original
    order, positive durations, and no overlap between cues that did not overlap."""
    assert len(out) == len(times)
    order = sorted(range(len(times)), key=lambda i: (times[i][0], times[i][1], i))
    for i in order:
        s, e = out[i]
        assert s >= 0.0 and e > s, (i, s, e)
    for a, b in zip(order, order[1:], strict=False):
        assert out[a][0] <= out[b][0], (a, b, out[a], out[b])
        if times[a][1] <= times[b][0]:
            assert out[a][1] <= out[b][0] + 1e-9, (a, b, out[a], out[b])


def test_retime_keeps_everything_and_fixes_overlaps():
    m = Mapping(
        [Segment(-math.inf, 1.0, 10.0), Segment(100.0, 1.0, 4.0)]  # backwards jump at 100 s
    )
    times = [(90.0, 99.0), (99.5, 102.0), (100.5, 103.0), (200.0, 201.0)]
    out = retime(times, m)
    _check_retimed(times, out)
    # cue 1 still uses segment 0 (its start < 100) and would land after cue 2, which
    # jumps back: it is pulled in front of cue 2 instead, and cue 0 ends before it.
    assert out[0][0] == 100.0
    assert out[2] == (104.5, 107.0)
    assert out[1] == pytest.approx((104.3, 106.8))  # overlapped cue 2 originally: kept
    assert out[0][1] == pytest.approx(104.3)
    assert out[3] == (204.0, 205.0)


def test_retime_overlap_clamp_when_next_cue_maps_before_this_one():
    # Old clamp kept the overlap when the next cue's mapped start was <= this one's.
    m = Mapping([Segment(-math.inf, 1.0, 0.0), Segment(10.0, 1.0, -3.0)])
    times = [(8.0, 9.5), (10.0, 12.0), (12.5, 14.0)]
    out = retime(times, m)
    _check_retimed(times, out)
    assert out[1] == (7.0, 9.0) and out[2] == (9.5, 11.0)
    assert out[0][1] <= out[1][0]


def test_retime_backward_cut_keeps_order_and_every_cue():
    m = Mapping([Segment(-math.inf, 1.0, 10.0), Segment(100.0, 1.0, 4.0)])
    times = [(90.0 + i, 90.8 + i) for i in range(21)]  # 90 … 110 s
    out = retime(times, m)
    _check_retimed(times, out)
    starts = [s for s, _e in out]
    assert starts == sorted(starts)
    # cues away from the cut are untouched
    assert out[0] == (100.0, 100.8)
    assert out[-1] == pytest.approx((114.0, 114.8))


def test_retime_negative_times_keep_order():
    m = Mapping.linear(1.0, -10.0)
    times = [(1.0, 2.0), (3.0, 4.0), (9.0, 11.0), (12.0, 13.0)]
    out = retime(times, m)
    _check_retimed(times, out)
    assert [s for s, _e in out[:3]] == pytest.approx([0.0, 0.2, 0.4])
    assert out[0][1] == pytest.approx(0.2)  # entirely before 0: minimal duration
    assert out[2] == pytest.approx((0.4, 1.0))  # straddles 0: keeps its true end
    assert out[3] == (2.0, 3.0)  # untouched


def test_retime_negative_times_squeeze_before_first_good_cue():
    m = Mapping.linear(1.0, -10.0)
    times = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (10.3, 11.0)]
    out = retime(times, m)
    _check_retimed(times, out)
    assert out[4] == pytest.approx((0.3, 1.0))  # not pushed
    assert all(e <= 0.3 + 1e-9 for _s, e in out[:4])


def test_retime_straddling_cue_is_only_extended_to_the_minimum():
    out = retime([(9.0, 10.1), (20.0, 21.0)], Mapping.linear(1.0, -10.0))
    assert out[0] == pytest.approx((0.0, 0.2))  # true end 0.1, extended to 0.2
    assert out[1] == (10.0, 11.0)


def test_retime_dense_negative_prefix_never_moves_real_cues():
    m = Mapping.linear(1.0, -10.0)
    neg = [(i * 0.09, i * 0.09 + 0.08) for i in range(100)]  # all map below 0
    good = [(10.5 + i, 11.2 + i) for i in range(20)]  # first good cue at 0.5 s
    times = neg + good
    out = retime(times, m)
    _check_retimed(times, out)
    assert out[100:] == pytest.approx([(s - 10.0, e - 10.0) for s, e in good])
    assert all(0.0 <= s and e <= 0.5 + 1e-9 for s, e in out[:100])


def test_retime_negative_prefix_without_room_collapses_at_zero():
    m = Mapping.linear(1.0, -10.0)
    times = [(1.0, 2.0), (3.0, 4.0), (10.0, 11.0)]  # good cue exactly at 0
    out = retime(times, m)
    assert out[2] == (0.0, 1.0)
    assert all(s == 0.0 and e > s for s, e in out)


def test_retime_preserves_original_overlaps_and_clamps_negative():
    m = Mapping.linear(1.0, -5.0)
    times = [(1.0, 4.0), (2.0, 3.0), (10.0, 12.0)]  # first two overlap on purpose
    out = retime(times, m)
    _check_retimed(times, out)
    assert out[0][0] == 0.0 and out[0][1] > 0.0
    assert out[1][0] > out[0][0] and out[1][1] > out[1][0]
    assert out[2] == (5.0, 7.0)


@pytest.mark.parametrize("seed", range(20))
def test_retime_invariants_random(seed):
    rng = np.random.default_rng(seed)
    segs = [Segment(-math.inf, float(rng.uniform(0.95, 1.05)), float(rng.uniform(-20, 20)))]
    for b in sorted(rng.uniform(10, 290, size=int(rng.integers(0, 4)))):
        segs.append(Segment(float(b), segs[-1].scale, segs[-1].offset + float(rng.uniform(-8, 8))))
    starts = np.sort(rng.uniform(0, 300, size=80))
    times = [(float(t), float(t + rng.uniform(0.0, 4.0))) for t in starts]
    _check_retimed(times, retime(times, Mapping(segs)))


def test_retime_scales_durations():
    out = retime([(100.0, 104.0)], Mapping.linear(25 / 23.976, 0.0))
    assert out[0][1] - out[0][0] == pytest.approx(4.0 * 25 / 23.976)


def test_apply_mapping_on_ssafile():
    subs = pysubs2.SSAFile()
    subs.append(pysubs2.SSAEvent(start=1000, end=2000, text="a"))
    subs.append(pysubs2.SSAEvent(start=3000, end=4000, text="b", type="Comment"))
    apply_mapping(subs, Mapping.linear(1.0, 2.5))
    assert [(e.start, e.end) for e in subs] == [(3500, 4500), (5500, 6500)]


def test_mapping_dominant():
    m = Mapping([Segment(-math.inf, 1.0, 1.0, 5), Segment(100.0, 1.0, 9.0, 50)])
    assert m.dominant([1, 2, 3, 150]).offset == 1.0
    assert m.dominant().offset == 9.0
    assert m(50) == 51 and m(150) == 159
    assert list(m.map_array(np.array([50.0, 150.0]))) == [51.0, 159.0]


# --- in-cue timing and scale snapping -----------------------------------------------


def test_in_cue_token_times_follow_scale():
    from vlcsubsync.align import _AnchorArrays

    tok = subtitle_tokens([Cue(100.0, 110.0, "one two three four five six seven eight")])
    rows = zip(tok.times, tok.lead, tok.cap, strict=True)
    a = [
        Anchor(float(t), 0.0, 1.0, 0, 0, i, float(ld), float(cp))
        for i, (t, ld, cp) in enumerate(rows)
    ]
    arr = _AnchorArrays.build(a)
    assert list(arr.x(1.0)) == pytest.approx(list(tok.times))
    # under 1.25x drift the same speech takes 1/1.25 of the subtitle-clock time
    assert list(arr.x(1.25) - 100.0) == pytest.approx(list((tok.times - 100.0) / 1.25))


@pytest.mark.parametrize("seed", range(5))
def test_snap_tolerates_per_window_timestamp_bias(seed):
    """Real Whisper windows each carry a shared timestamp bias (~0.2-0.4 s): the free
    slope through 12 such windows is off the exact NTSC ratio by a few 1e-4, which is
    within its standard error, so the exact ratio is kept."""
    from vlcsubsync.align import _snap_scale

    rng = np.random.default_rng(seed)
    xs, ys, wins = [], [], []
    for k in range(12):
        x = 60.0 + 130.0 * k + np.sort(rng.uniform(0, 24, 40))
        bias = rng.normal(0, 0.3)
        xs.append(x)
        ys.append(1.25 * x + 0.3 + bias + rng.normal(0, 0.25, x.size))
        wins.append(np.full(x.size, k))
    x, y, win = np.concatenate(xs), np.concatenate(ys), np.concatenate(wins)
    w = np.ones_like(x)
    s_free = float(np.polyfit(x, y, 1)[0])
    s, _o = _snap_scale(x, y, w, s_free, float(np.median(y - s_free * x)), win)
    assert s == 1.25


def test_24_vs_23976_is_not_snapped_away(synth):
    """24/23.976 (+0.1%) stays distinguishable from 1.0 on clean data."""
    fit, err = _align(synth, lambda t: t * 24 / 23.976 + 1.0)
    assert fit.mapping.segments[0].scale == pytest.approx(24 / 23.976, abs=1e-5)
    _assert_accurate(err)


# --- local refinement ---------------------------------------------------------------


def _speech_mask(spans, duration, res=0.01):
    speech = np.zeros(int(duration / res) + 100, dtype=bool)
    for a, b, _c in spans:
        speech[max(0, int(a / res)) : max(0, int(b / res))] = True
    return speech


def _align_full(synth, transform, duration=45 * 60, seed=1, budget=4, use_speech=True, **kw):
    """fit_mapping → subdivide (probing new windows within ``budget``) → refine_local.
    Returns (initial fit errors, final fit, final errors, windows transcribed by probes)."""
    script = synth.script(duration - 30, seed=seed)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    spans = synth.speech_spans(script, transform)
    speech = _speech_mask(spans, duration) if use_speech else None
    fake = synth.FakeTranscriber(spans, **kw)
    starts = pick_windows(np.ones(int(duration)), duration, Config().window_count(duration))
    windows = [(s, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=s)) for s in starts]
    tok = subtitle_tokens(cues)
    anchors = find_anchors(tok, windows)
    fit = fit_mapping(anchors, cues, speech, n_windows=len(windows))
    probes = []

    def probe(centre, near):
        nonlocal anchors
        if len(probes) >= budget or any(abs(s + 15 - centre) <= near for s, _w in windows):
            return anchors, False
        st = max(0.0, min(duration - 30.0, centre - 15.0))
        probes.append(st)
        windows.append((st, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=st)))
        anchors = find_anchors(tok, windows)
        return anchors, True

    truth = np.array([transform(c.start) for c in cues])
    keep = truth >= 0
    times = [(c.start, c.end) for c in cues]

    def errors(mapping):
        return np.abs(np.array([s for s, _e in retime(times, mapping)]) - truth)[keep]

    err0 = errors(fit.mapping)
    fit, anchors = subdivide(fit, anchors, cues, probe, speech, 0.01, len(windows))
    fit.mapping, _info = refine_local(fit.mapping, anchors, cues, speech, 0.01)
    return err0, fit, errors(fit.mapping), probes


def test_piecewise_drift_scale_change_after_cut(synth):
    """Film timing (1.0) up to a cut, PAL-sped (25/23.976) after it."""
    r = 25 / 23.976

    def f(t):
        return t + 1.0 if t < 1500 else 1501.0 + (t - 1500) * r

    _err0, fit, err, _probes = _align_full(synth, f)
    segs = fit.mapping.segments
    assert [s.scale for s in segs] == pytest.approx([1.0, r], abs=2e-4)
    assert 1400 < segs[1].start < 1600
    _assert_accurate(err)


@pytest.mark.parametrize("use_speech", [True, False])
def test_slow_wobble_is_followed_smoothly(synth, use_speech):
    """Offset wobbling ±0.3 s sinusoidally (20 min period): local refinement follows it
    with a few smooth knots and no extra segments."""

    def f(t):
        return t + 3.0 + 0.3 * math.sin(2 * math.pi * t / 1200)

    err0, fit, err, _probes = _align_full(synth, f, use_speech=use_speech)
    assert len(fit.mapping.segments) == 1
    seg = fit.mapping.segments[0]
    assert 2 <= len(seg.knots) <= 45 * 60 / 120
    assert max(abs(c) for _x, c in seg.knots) <= 0.5
    assert np.median(err0) > 0.15
    assert np.median(err) < (0.08 if use_speech else 0.12)
    assert np.percentile(err, 95) < np.percentile(err0, 95) - 0.1


def test_refine_local_median_shift_is_bounded():
    cues = [Cue(10.0 * i, 10.0 * i + 2.0, f"w{i}") for i in range(60)]
    anchors = [Anchor(c.start, c.start + 5.0 + 0.8, 1.0, window=i // 10, cue=i, token=i)
               for i, c in enumerate(cues)]  # fmt: skip
    m, info = refine_local(Mapping.linear(1.0, 5.0), anchors, cues, wobble=False)
    assert info["segment_shifts"] == [pytest.approx(0.5)]
    assert m.segments[0].offset == pytest.approx(5.5)


def test_refine_with_speech_uses_precision_not_spread():
    """Real dialogue: cue starts vs speech onsets scatter by ~0.2 s (MAD), but over
    hundreds of cues the median is precise; the old MAD <= 0.12 gate rejected it."""
    rng = np.random.default_rng(0)
    res = 0.01
    starts = np.cumsum(rng.uniform(2.5, 4.0, 300))
    cues = [Cue(float(t), float(t) + 1.5, "x") for t in starts]
    speech = np.zeros(int((starts[-1] + 10) / res), dtype=bool)
    for t in starts:
        on = t + 0.15 + rng.normal(0, 0.25)  # true bias +0.15 s, noisy onsets
        speech[int(on / res) : int((on + 1.2) / res)] = True
    m, shift = refine_with_speech(Mapping.identity(), cues, speech, res)
    assert shift == pytest.approx(0.15, abs=0.05)
    # too few cues for a precise median: untouched
    m2, shift2 = refine_with_speech(Mapping.identity(), cues[:15], speech, res)
    assert shift2 == 0.0


def test_mapping_knots_interpolate_and_hold():
    seg = Segment(-math.inf, 1.0, 2.0, knots=((100.0, 0.2), (200.0, -0.2)))
    m = Mapping([seg])
    assert m(50.0) == pytest.approx(52.2)  # held flat before the first knot
    assert m(150.0) == pytest.approx(152.0)
    assert m(400.0) == pytest.approx(401.8)
    assert list(m.map_array(np.array([50.0, 150.0, 400.0]))) == pytest.approx([52.2, 152.0, 401.8])
    # retime uses the corrected start; durations still scale with the segment
    assert retime([(150.0, 152.0)], m)[0] == pytest.approx((152.0, 154.0))


def _dense_cues_and_anchors(bias, spacing=0.5, duration=1200.0, offset=2.0):
    """A cue every ``spacing`` s (``0.8 * spacing`` long), one anchor per cue at
    ``offset + bias(t)`` s, one window per 30 s."""
    cues, anchors = [], []
    for i, t in enumerate(np.arange(5.0, duration, spacing)):
        t = float(t)
        cues.append(Cue(t, t + 0.8 * spacing, f"w{i}"))
        anchors.append(Anchor(t, t + offset + bias(t), 1.0, int(t // 30), i, i))
    return cues, anchors


def test_refinement_total_correction_is_bounded(synth):
    """Steps 1 (per-segment median), 2 (speech onsets) and 3 (wobble) each have their own
    bound; together they used to reach ~1.4 s. The total relative to the fitted line is
    bounded by max_shift."""
    cues, anchors = _dense_cues_and_anchors(lambda t: 0.45 + 0.3 * math.sin(t / 150.0))
    res = 0.01
    speech = np.zeros(int(1300 / res), dtype=bool)
    for c in cues[::4]:  # speech starts another 0.35 s later than the anchors say
        on = c.start + 2.0 + 0.45 + 0.35
        speech[int(on / res) : int((on + 1.0) / res)] = True
    fitted = Mapping([Segment(-math.inf, 1.0, 2.0)])
    m, info = refine_local(fitted, anchors, cues, speech, res)
    t = np.linspace(0, 1250, 5001)
    corr = m.map_array(t) - fitted.map_array(t)
    assert np.abs(corr).max() <= 0.5 + 1e-6
    assert info["max_correction"] <= 0.5 + 1e-6
    assert corr.max() > 0.4  # it did correct, up to the bound


@pytest.mark.parametrize("wobble", [True, False])
def test_refinement_is_continuous_at_segment_boundaries(wobble):
    """Two segments with the same fitted line at the boundary; the anchors pull the
    first one +0.4 s and the second -0.4 s. Independent per-segment shifts made the
    mapping jump back 0.8 s at the boundary, and retime squeezed the cues before it
    into 0.2 s flashes. The correction is continuous: no reordering, no flashes."""
    cues, anchors = _dense_cues_and_anchors(lambda t: 0.4 if t < 600 else -0.4)
    fitted = Mapping([Segment(-math.inf, 1.0, 2.0), Segment(600.0, 1.0, 2.0)])
    m, _info = refine_local(fitted, anchors, cues, None, 0.01, wobble=wobble)
    eps = 1e-6
    assert m(600.0 - eps) == pytest.approx(m(600.0), abs=1e-3)  # continuous
    times = [(c.start, c.end) for c in cues]
    out = retime(times, m)
    starts = [s for s, _e in out]
    assert starts == sorted(starts)
    durs = np.array([e - s for s, e in out])
    assert durs.min() > 0.3  # cues are 0.4 s long; no 0.2 s flashes
    # away from the boundary each segment still gets its own correction
    assert m(100.0) - fitted(100.0) == pytest.approx(0.4, abs=0.05)
    assert m(1100.0) - fitted(1100.0) == pytest.approx(-0.4, abs=0.05)


def test_genuine_backward_cut_still_jumps():
    """Continuity is only for the correction: a fitted backward cut keeps its jump."""
    cues, anchors = _dense_cues_and_anchors(lambda t: 0.0 if t < 600 else -5.0)
    fitted = Mapping([Segment(-math.inf, 1.0, 2.0), Segment(600.0, 1.0, -3.0)])
    m, _info = refine_local(fitted, anchors, cues, None, 0.01)
    assert m(600.0) - m(600.0 - 1e-6) == pytest.approx(-5.0, abs=0.05)


@pytest.mark.parametrize("period,phase", [(1200, 1.3), (1800, 0.0), (1800, 2.2), (900, 4.0)])
def test_wobble_periods_and_phases_not_chopped(synth, period, phase):
    """Verification must leave smooth wobble to local refinement, whatever its period
    and phase (no extra segments), and refinement must improve it."""

    def f(t):
        return t + 3.0 + 0.3 * math.sin(2 * math.pi * t / period + phase)

    err0, fit, err, _probes = _align_full(synth, f)
    assert len(fit.mapping.segments) == 1
    assert np.median(err) < 0.1
    assert np.percentile(err, 95) < np.percentile(err0, 95)


# --- bisection verification ---------------------------------------------------------


def test_three_sections_with_small_offset_steps(synth):
    """Offsets 2.0 / 2.6 / 1.8 s: steps below the 1 s inlier threshold, which the
    initial fit absorbs into a tilted compromise line; verification separates them."""

    def f(t):
        return t + 2.0 if t < 900 else (t + 2.6 if t < 1800 else t + 1.8)

    err0, fit, err, probes = _align_full(synth, f)
    segs = fit.mapping.segments
    assert len(segs) == 3
    assert [s.scale for s in segs] == pytest.approx([1.0, 1.0, 1.0])
    # speech onsets lie 0-0.12 s after the cue starts (synthetic), hence the tolerance
    assert [s.offset for s in segs] == pytest.approx([2.0, 2.6, 1.8], abs=0.12)
    assert 800 < segs[1].start < 1000 and 1700 < segs[2].start < 1900
    assert np.percentile(err, 95) < 0.15 < np.percentile(err0, 95)
    assert len(probes) <= 4


def test_subdivide_leaves_consistent_fit_alone(synth):
    err0, fit, err, probes = _align_full(synth, lambda t: t * 25 / 23.976 - 4.0, drop=0.2)
    assert len(fit.mapping.segments) == 1
    assert fit.mapping.segments[0].scale == pytest.approx(25 / 23.976)
    assert fit.details["verify_splits"] == 0
    assert len(probes) <= 1
    _assert_accurate(err)


def test_subdivide_respects_budget_and_min_length(synth):
    def f(t):
        return t + 2.0 if t < 900 else (t + 2.6 if t < 1800 else t + 1.8)

    _e0, fit0, _err, probes = _align_full(synth, f, budget=0)
    assert probes == []
    # segments never shorter than the minimum (except the open-ended first one)
    starts = [s.start for s in fit0.mapping.segments[1:]]
    assert all(b - a >= 120.0 for a, b in zip(starts, starts[1:], strict=False))


def test_short_segment_from_one_biased_window_is_folded(synth):
    """A window whose timestamps are all 1.6 s late (seen with beam-1 CPU decoding at
    an episode's cold open) must not keep a section of its own: an independent probe
    inside the short segment sides with the neighbour, and the tie folds it."""
    from vlcsubsync.align import AlignResult
    from vlcsubsync.transcribe import Word

    duration = 30 * 60
    script = synth.script(duration - 30, seed=1)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    fake = synth.FakeTranscriber(synth.speech_spans(script, lambda t: t + 2.0))

    def transcribe(st):
        words = fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=st)
        bias = 1.6 if st == 60.0 else 0.0
        return [Word(w.start + bias, w.end + bias, w.text) for w in words]

    windows = [(st, transcribe(st)) for st in (60.0, 300.0, 600.0, 900.0, 1200.0, 1500.0)]
    tok = subtitle_tokens(cues)
    anchors = find_anchors(tok, windows)
    mapping = Mapping([Segment(-math.inf, 1.0, 3.6, 20), Segment(150.0, 1.0, 2.0, 200)])
    fit = AlignResult(mapping, "whisper", 0.9, 200, len(anchors))
    probes = []

    def probe(centre, near):
        nonlocal anchors
        st = max(0.0, centre - 15.0)
        if len(probes) >= 2 or any(abs(s + 15 - centre) <= near for s, _w in windows):
            return anchors, False
        probes.append(st)
        windows.append((st, transcribe(st)))
        anchors = find_anchors(tok, windows)
        return anchors, True

    out, _anchors = subdivide(fit, anchors, cues, probe, None, 0.01, len(windows))
    assert len(probes) == 1 and not 60.0 <= probes[0] <= 90.0  # away from the bad window
    assert out.details["verify_folded"] == 1
    assert [(s.scale, s.offset) for s in out.mapping.segments] == pytest.approx(
        [(1.0, 2.0)], abs=0.05
    )


@pytest.mark.parametrize("length", [200.0, 330.0])  # fold path (< 2x120 s) / main loop
@pytest.mark.parametrize("budget", [0, 3])
def test_genuine_short_cut_section_survives_verification(synth, length, budget):
    """An inserted scene: the subtitles are 12 s off for one short section that holds
    under 10% of the anchors. Windows 12 s off the neighbours' line are disagreement,
    not missing evidence, so neither the merge, the whole-range repair nor the fold
    may swallow the section. A small 0.6 s step earlier in the file makes the
    verification split there, so the merge loop runs (it used to merge the section
    away: its windows had no candidates within 2 s and counted as "unknown")."""
    from vlcsubsync.align import AlignResult

    a, b = 1300.0, 1300.0 + length

    def f(t):
        return t + 14.6 if a <= t < b else (t + 2.0 if t < 500 else t + 2.6)

    duration = 40 * 60
    script = synth.script(duration - 30, seed=1)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    fake = synth.FakeTranscriber(synth.speech_spans(script, f))
    # dense windows elsewhere, two inside the section (one alone is an outlier to
    # fit_mapping, which is right: one window's timestamps can be off on their own)
    starts = [st for st in (30.0 + 100.0 * i for i in range(24)) if not a - 40 < st < b + 20]
    starts += [a + 14.0 + 10.0, b + 14.0 - 45.0]
    windows = [(st, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=st)) for st in starts]
    tok = subtitle_tokens(cues)
    anchors = find_anchors(tok, windows)
    fit = fit_mapping(anchors, cues, None, n_windows=len(windows))
    assert [s.offset for s in fit.mapping.segments] == pytest.approx([2.6, 14.6, 2.6], abs=0.2)
    tokens = {(x.window, x.token) for x in anchors}
    inside = {(x.window, x.token) for x in anchors if a <= x.sub_time < b}
    assert len(inside) < 0.1 * len(tokens)
    probes = []

    def probe(centre, near):
        nonlocal anchors
        if len(probes) >= budget or any(abs(s + 15 - centre) <= near for s, _w in windows):
            return anchors, False
        st = max(0.0, min(duration - 30.0, centre - 15.0))
        probes.append(st)
        windows.append((st, fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=st)))
        anchors = find_anchors(tok, windows)
        return anchors, True

    out, _ = subdivide(fit, anchors, cues, probe, None, 0.01, len(windows))
    assert isinstance(out, AlignResult)
    segs = out.mapping.segments
    assert [s.offset for s in segs] == pytest.approx([2.0, 2.6, 14.6, 2.6], abs=0.2)
    assert a - 60 < segs[2].start < a + 60 and b - 60 < segs[3].start < b + 60
    assert out.details["verify_splits"] >= 1  # the merge loop did run
    assert out.details.get("verify_folded", 0) == 0


def test_probe_without_evidence_is_not_verification(synth):
    """Probe windows on music or silence add no anchors: the probe is retried at other
    positions while the budget lasts, and a segment whose windows have no coherent
    evidence is left unverified (and reported), not counted as verified."""
    from vlcsubsync.align import AlignResult

    duration = 30 * 60
    script = synth.script(duration - 30, seed=1)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    fake = synth.FakeTranscriber(synth.speech_spans(script, lambda t: t + 2.0))
    garbage = synth.FakeTranscriber(synth.speech_spans(script, lambda t: t + 2.0), garbage=True)

    def transcribe(st):  # real speech before 900 s; music (garbage words) after
        t = fake if st < 900 else garbage
        return t.transcribe(np.zeros(30 * 16000), 16000, "en", start=st)

    windows = [(st, transcribe(st)) for st in (60.0, 300.0, 600.0, 1000.0, 1300.0, 1600.0)]
    tok = subtitle_tokens(cues)
    anchors = find_anchors(tok, windows)
    mapping = Mapping([Segment(-math.inf, 1.0, 2.0, 100), Segment(950.0, 1.0, 2.0, 0)])
    fit = AlignResult(mapping, "whisper", 0.9, 100, len(anchors))
    probes = []

    def probe(centre, near):  # every probe lands on silence
        if len(probes) >= 3:
            return anchors, False
        probes.append(centre)
        return anchors, True

    out, _ = subdivide(fit, anchors, cues, probe, None, 0.01, len(windows))
    assert len(probes) == 3  # retried at other positions until the budget ran out
    assert out.details["verify_empty_probes"] == 3
    assert out.details["verify_unverified"] >= 1  # the music-only segment
    assert out.mapping == mapping  # left as it was


@pytest.mark.parametrize("bias", [1.0, 1.6])
@pytest.mark.parametrize("biased_starts", [(60.0,), (60.0, 72.0)])
def test_biased_cold_open_windows_end_as_one_segment(synth, bias, biased_starts):
    from vlcsubsync.align import AlignResult

    """One exact line, but the window(s) over the cold open come back with timestamps
    ``bias`` s late (seen with beam-1 CPU decoding; two overlapping windows when an
    adaptive window lands on the same audio). Whether or not fit_mapping sets them
    apart, verification must end with one segment: overlapping windows share their
    bias and vote once, and an independent probe votes for the line it is closer to."""
    from vlcsubsync.transcribe import Word

    duration = 30 * 60
    script = synth.script(duration - 30, seed=1)
    cues = [Cue(c.start, c.end, c.text) for c in script]
    fake = synth.FakeTranscriber(synth.speech_spans(script, lambda t: t + 2.0))

    def transcribe(st):
        words = fake.transcribe(np.zeros(30 * 16000), 16000, "en", start=st)
        b = bias if st in biased_starts else 0.0
        return [Word(w.start + b, w.end + b, w.text) for w in words]

    starts = [*biased_starts, 300.0, 540.0, 780.0, 1020.0, 1260.0, 1500.0, 1700.0]
    windows = [(st, transcribe(st)) for st in starts]
    tok = subtitle_tokens(cues)
    anchors = find_anchors(tok, windows)
    fit = fit_mapping(anchors, cues, None, n_windows=len(windows))
    if len(fit.mapping.segments) == 1:  # make sure the fold path is exercised
        mapping = Mapping([Segment(-math.inf, 1.0, 2.0 + bias, 20), Segment(150.0, 1.0, 2.0, 200)])
        fit = AlignResult(mapping, "whisper", 0.9, 200, len(anchors))
    probes = []

    def probe(centre, near):
        nonlocal anchors
        st = max(0.0, centre - 15.0)
        if len(probes) >= 3 or any(abs(s + 15 - centre) <= near for s, _w in windows):
            return anchors, False
        probes.append(st)
        windows.append((st, transcribe(st)))
        anchors = find_anchors(tok, windows)
        return anchors, True

    out, _ = subdivide(fit, anchors, cues, probe, None, 0.01, len(starts))
    segs = out.mapping.segments
    assert len(segs) == 1, [(s.start, s.offset) for s in segs]
    assert segs[0].scale == 1.0 and segs[0].offset == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("bias,expect", [(0.2, 0.05), (0.6, 0.4)])
def test_fit_stats_allows_a_shared_bias_per_window(bias, expect):
    """Each window's words share a timestamp bias: up to 0.2 s of it is not misfit,
    anything beyond still shows in the residual."""
    from vlcsubsync.align import _AnchorArrays, _fit_stats

    rng = np.random.default_rng(0)
    anchors = []
    for k in range(6):
        b = bias if k % 2 else -bias
        for j in range(40):
            t = 100.0 + 200.0 * k + j
            anchors.append(Anchor(t, t + 2.0 + b + rng.normal(0, 0.05), 1.0, k, -1, j))
    _conf, _n, _g, med, _d = _fit_stats(Mapping.linear(1.0, 2.0), _AnchorArrays.build(anchors), 6)
    assert med == pytest.approx(expect, abs=0.03)


def test_probe_window_without_inliers_does_not_lower_coverage():
    """A verification probe (window id >= n_windows) on music adds only stray matches:
    it must not count against coverage; one with inliers counts as covered."""
    from vlcsubsync.align import _AnchorArrays, _fit_stats

    rng = np.random.default_rng(0)
    anchors = []
    for k in range(6):
        for j in range(20):
            t = 100.0 + 200.0 * k + j
            anchors.append(Anchor(t, t + 2.0 + rng.normal(0, 0.05), 1.0, k, -1, j))
    stray = [Anchor(50.0 + j, 900.0 + 37.0 * j, 1.0, 6, -1, j) for j in range(5)]
    mapping = Mapping.linear(1.0, 2.0)
    *_, d = _fit_stats(mapping, _AnchorArrays.build(anchors + stray), 6)
    assert d["coverage"] == pytest.approx(1.0)
    good = [Anchor(1300.0 + j, 1302.0, 1.0, 6, -1, j) for j in range(5)]
    *_, d = _fit_stats(mapping, _AnchorArrays.build(anchors + good), 6)
    assert d["coverage"] == pytest.approx(1.0)
