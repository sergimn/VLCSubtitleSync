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


def _assert_accurate(err):
    assert np.median(err) < MEDIAN_MAX, np.median(err)
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


@pytest.mark.parametrize("scale", [25 / 23.976, 23.976 / 25])
def test_framerate_drift(synth, scale):
    fit, err = _align(synth, lambda t: t * scale)
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
    assert r.confidence > 0.6


def test_vad_align_unrelated_is_low_confidence(synth):
    cues, _ = _masks(synth, lambda t: t, 20 * 60, seed=4)
    _, speech = _masks(synth, lambda t: t, 20 * 60, seed=99)
    r = vad_align(speech, cues)
    assert r.confidence < 0.5


def test_vad_align_empty():
    assert vad_align(np.zeros(100, bool), [Cue(0, 1, "x")]).method == "none"


# --- applying a mapping ---------------------------------------------------------------


def test_retime_keeps_everything_and_fixes_overlaps():
    m = Mapping(
        [Segment(-math.inf, 1.0, 10.0), Segment(100.0, 1.0, 4.0)]  # backwards jump at 100 s
    )
    times = [(90.0, 99.0), (99.5, 102.0), (100.5, 103.0), (200.0, 201.0)]
    out = retime(times, m)
    assert len(out) == 4
    assert out[0] == (100.0, 109.0)
    # cue 1 still uses segment 0 (its start < 100); cue 2 jumps back; overlaps clamped
    for s1, e1 in out:
        assert e1 > s1
    assert out[1][1] <= out[2][0] or out[2][0] < out[1][0]
    assert out[3] == (204.0, 205.0)


def test_retime_preserves_original_overlaps_and_clamps_negative():
    m = Mapping.linear(1.0, -5.0)
    times = [(1.0, 4.0), (2.0, 3.0), (10.0, 12.0)]  # first two overlap on purpose
    out = retime(times, m)
    assert out[0][0] == 0.0 and out[0][1] > 0.0
    assert out[1][0] == 0.0 and out[1][1] > out[1][0]
    assert out[2] == (5.0, 7.0)


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
