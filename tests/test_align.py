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
    retime,
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
