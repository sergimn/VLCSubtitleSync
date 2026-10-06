"""End-to-end orchestration tests: synthetic speech-like WAV + fake transcriber.

The default run uses the energy VAD (Silero is exercised by the slow tests on real
speech fixtures) and never loads a Whisper model.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pysubs2
import pytest

from vlcsubsync import transcribe as tr
from vlcsubsync import vad
from vlcsubsync.config import Config
from vlcsubsync.sync import (
    SubtitleSource,
    SyncError,
    SyncResult,
    resolve_subtitle_source,
    sync_subtitles,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def energy_vad(monkeypatch):
    monkeypatch.setattr(vad, "silero_probabilities", lambda audio, progress=None: None)


def _case(tmp_path, synth, transform, duration, seed=5, audio_seed=None, **fake_kw):
    script = synth.script(duration - 40, seed=seed)
    spans = synth.speech_spans(script, transform)
    if audio_seed is None:
        audio_spans = spans
    else:  # speech unrelated to the subtitles
        audio_spans = synth.speech_spans(synth.script(duration - 40, seed=audio_seed), transform)
    wav = synth.write_wav(tmp_path / "movie.wav", synth.audio(audio_spans, duration))
    srt = tmp_path / "movie.srt"
    srt.write_text(synth.srt(script), encoding="utf-8")
    fake = synth.FakeTranscriber(spans, **fake_kw)
    return script, wav, srt, fake


def _errors(synth, out_path, script, transform):
    out = pysubs2.load(out_path)
    assert len(out) == len(script)
    pred = np.array([e.start / 1000 for e in out])
    truth = np.array([transform(c.start) for c in script])
    keep = truth >= 0
    return np.abs(pred - truth)[keep]


def _check(err):
    assert np.median(err) < 0.15, np.median(err)
    assert np.percentile(err, 95) < 0.4, np.percentile(err, 95)


@pytest.mark.parametrize(
    "name,transform",
    [
        ("offset", lambda t: t + 7.3),
        ("negative", lambda t: t - 2.4),
        ("drift", lambda t: t * 25 / 23.976 - 2.0),
        ("slowdown", lambda t: t * 23.976 / 25 + 1.0),
    ],
)
def test_sync_whisper_path(tmp_path, synth, energy_vad, name, transform):
    script, wav, srt, fake = _case(tmp_path, synth, transform, 10 * 60, drop=0.2, noise=0.1)
    progress = []
    r = sync_subtitles(
        str(wav),
        0,
        SubtitleSource("external", path=str(srt)),
        str(tmp_path / "out.srt"),
        Config(),
        lambda p, m: progress.append((p, m)),
        fake,
    )
    assert isinstance(r, SyncResult)
    assert r.applied and r.method == "whisper"
    assert r.output_path == str(tmp_path / "out.srt")
    assert r.confidence > 0.7 and r.anchors > 20 and r.segments == 1
    _check(_errors(synth, r.output_path, script, transform))
    # progress: monotonic fractions in [0, 1], ends at 1.0, mentions the stages
    ps = [p for p, _m in progress]
    assert ps == sorted(ps) and ps[0] >= 0 and ps[-1] == 1.0
    msgs = {m.split(" ")[0] for _p, m in progress}
    assert {"Decoding", "Transcribing", "Done"} <= msgs
    assert any(m.startswith("Transcribing 1/") for _p, m in progress)
    assert "offset" in r.message


def test_sync_cut_piecewise(tmp_path, synth, energy_vad):
    def f(t):
        return t + 3.0 if t < 1200 else t + 33.0

    script, wav, srt, fake = _case(tmp_path, synth, f, 25 * 60, drop=0.15, noise=0.1)
    r = sync_subtitles(
        str(wav), 0, SubtitleSource("external", path=str(srt)), str(tmp_path / "o.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert r.applied and r.segments == 2
    assert r.offset == pytest.approx(3.0, abs=0.15)
    assert "→" in r.message
    err = _errors(synth, r.output_path, script, f)
    _check(err)
    # adaptive windows were added around the cut
    assert len(fake.calls) > Config().window_count(25 * 60)


def test_language_mismatch_uses_vad(tmp_path, synth, energy_vad):
    def f(t):
        return t + 4.0

    script, wav, srt, fake = _case(tmp_path, synth, f, 10 * 60, language="es")
    r = sync_subtitles(
        str(wav), 0, SubtitleSource("external", path=str(srt)), str(tmp_path / "o.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert fake.calls == []  # no transcription for a mismatched language
    assert r.method == "vad" and r.applied
    assert r.offset == pytest.approx(4.0, abs=0.15)
    _check(_errors(synth, r.output_path, script, f))


def test_english_only_model_uses_multilingual_detector(tmp_path, synth, energy_vad, monkeypatch):
    """English subs + .en model: the audio language must come from the multilingual model."""
    import vlcsubsync.sync as sync_mod

    def f(t):
        return t + 4.0

    script, wav, srt, fake = _case(tmp_path, synth, f, 10 * 60)
    fake.multilingual = False  # behaves like base.en: would always answer "en"

    class Detector:
        multilingual = True

        def detect_language(self, audio, sr):
            return "fr", 0.95

    used = []
    monkeypatch.setattr(sync_mod, "get_transcriber", lambda config, lang: fake)
    monkeypatch.setattr(sync_mod, "get_model", lambda config, name: used.append(name) or Detector())
    r = sync_subtitles(
        str(wav), 0, SubtitleSource("external", path=str(srt)), str(tmp_path / "o.srt"),
        Config(),
    )  # fmt: skip
    assert used == [Config().model_multi]
    assert fake.calls == []  # mismatch detected: no English transcription of French audio
    assert r.method == "vad"


def test_too_few_anchors_not_applied(tmp_path, synth, energy_vad):
    def f(t):
        return t + 6.0

    script, wav, srt, fake = _case(tmp_path, synth, f, 8 * 60, audio_seed=77, garbage=True)
    r = sync_subtitles(
        str(wav), 0, SubtitleSource("external", path=str(srt)), str(tmp_path / "o.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert not r.applied
    assert r.method == "none" and r.offset == 0.0 and r.scale == 1.0
    assert r.message.startswith("not synced")
    out = pysubs2.load(r.output_path)
    orig = pysubs2.load(str(srt))
    assert [(e.start, e.end, e.text) for e in out] == [(e.start, e.end, e.text) for e in orig]


def test_garbage_transcript_falls_back_to_vad(tmp_path, synth, energy_vad):
    def f(t):
        return t - 3.5

    script, wav, srt, fake = _case(tmp_path, synth, f, 10 * 60, garbage=True)
    r = sync_subtitles(
        str(wav), 0, SubtitleSource("external", path=str(srt)), str(tmp_path / "o.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert r.applied and r.method == "vad"
    _check(_errors(synth, r.output_path, script, f))


def test_external_ass_keeps_format_and_styles(tmp_path, synth, energy_vad):
    def f(t):
        return t + 2.0

    script, wav, srt, fake = _case(tmp_path, synth, f, 6 * 60)
    subs = pysubs2.load(str(srt))
    subs.styles["Sign"] = pysubs2.SSAStyle(fontname="Georgia", fontsize=33)
    subs[0].style = "Sign"
    subs[1].text = "{\\an8\\i1}" + subs[1].text
    ass = tmp_path / "movie.ass"
    subs.save(str(ass))
    r = sync_subtitles(
        str(wav), 0, SubtitleSource("external", path=str(ass)), str(tmp_path / "o.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert r.applied
    assert r.output_path == str(tmp_path / "o.ass")
    out = pysubs2.load(r.output_path)
    assert out.styles["Sign"].fontname == "Georgia"
    assert out[0].style == "Sign"
    assert out[1].text.startswith("{\\an8\\i1}")
    assert out[0].start == pytest.approx(subs[0].start + 2000, abs=150)


def test_embedded_ass_in_mkv(tmp_path, synth, energy_vad):
    if not synth.ffmpeg:
        pytest.skip("ffmpeg not available")

    def f(t):
        return t - 1.5  # subtitles late by 1.5 s

    script, wav, srt, fake = _case(tmp_path, synth, f, 5 * 60)
    subs = pysubs2.load(str(srt))
    subs.styles["Default"].fontname = "Verdana"
    ass = tmp_path / "in.ass"
    subs.save(str(ass))
    mkv = tmp_path / "movie.mkv"
    subprocess.run(
        [synth.ffmpeg, "-loglevel", "error", "-y", "-i", str(wav), "-i", str(srt), "-i", str(ass),
         "-map", "0", "-map", "1", "-map", "2", "-c:a", "aac", "-c:s:0", "srt", "-c:s:1", "ass",
         str(mkv)],
        check=True,
    )  # fmt: skip
    r = sync_subtitles(
        str(mkv), 0, SubtitleSource("embedded", index=1), str(tmp_path / "o.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert r.applied and r.output_path.endswith(".ass")
    assert pysubs2.load(r.output_path).styles["Default"].fontname == "Verdana"
    _check(_errors(synth, r.output_path, script, f))
    r2 = sync_subtitles(
        str(mkv), 0, SubtitleSource("embedded", index=0), str(tmp_path / "o2.srt"),
        Config(), transcriber=fake,
    )  # fmt: skip
    assert r2.applied and r2.output_path.endswith(".srt")
    _check(_errors(synth, r2.output_path, script, f))


def test_empty_subtitles_raise(tmp_path, synth, energy_vad):
    wav = synth.write_wav(tmp_path / "a.wav", np.zeros(16000 * 5, np.float32))
    s = tmp_path / "a.ass"
    pysubs2.SSAFile().save(str(s))
    with pytest.raises(Exception):  # noqa: B017 - SubtitleError or SyncError
        sync_subtitles(str(wav), 0, SubtitleSource("external", path=str(s)), str(tmp_path / "o"),
                       Config(), transcriber=synth.FakeTranscriber([]))  # fmt: skip


def test_resolve_subtitle_source(tmp_path, synth):
    wav = synth.write_wav(tmp_path / "movie.wav", np.zeros(16000, np.float32))
    for name in ("movie.en.srt", "movie.srt"):
        (tmp_path / name).write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n")
    assert resolve_subtitle_source(str(wav), 0) == SubtitleSource(
        "external", path=str(tmp_path / "movie.srt")
    )
    assert resolve_subtitle_source(str(wav), 1).path == str(tmp_path / "movie.en.srt")
    assert resolve_subtitle_source(str(wav), 5, "/x/y.srt").path == "/x/y.srt"
    with pytest.raises(SyncError):
        resolve_subtitle_source(str(wav), 2)


# --- transcriber management ---------------------------------------------------------


class _FakeSeg:
    def __init__(self, words):
        self.words = words


class _FakeW:
    def __init__(self, s, e, w):
        self.start, self.end, self.word, self.probability = s, e, w, 0.9


def _fake_whisper_model(fail_on_cuda_transcribe=True, fail_load_cuda=False):
    created = []

    class FakeModel:
        def __init__(self, name, device, compute_type, cpu_threads, download_root):
            if device == "cuda" and fail_load_cuda:
                raise RuntimeError("CUDA driver missing")
            self.device = device
            created.append((name, device, compute_type))

        def transcribe(self, audio, **kw):
            def gen():
                if self.device == "cuda" and fail_on_cuda_transcribe:
                    raise RuntimeError("Library libcudnn_ops.so.9 cannot be loaded")
                yield _FakeSeg([_FakeW(0.5, 0.9, " Hello"), _FakeW(1.0, 1.4, " world.")])

            return gen(), None

        def detect_language(self, audio):
            return "fr", 0.9, []

    return FakeModel, created


@pytest.mark.parametrize("fail_load", [False, True])
def test_cuda_failure_falls_back_to_cpu(monkeypatch, fail_load):
    import faster_whisper

    Fake, created = _fake_whisper_model(fail_load_cuda=fail_load)
    monkeypatch.setattr(faster_whisper, "WhisperModel", Fake)
    monkeypatch.setattr(tr, "_cuda_available", lambda: True)
    t = tr.WhisperTranscriber("base.en", device="auto")
    words = t.transcribe(np.zeros(16000, np.float32), 16000, "en")
    assert [w.text for w in words] == ["Hello", "world."]
    assert t.device == "cpu"
    assert created[-1][1] == "cpu"
    # stays on CPU afterwards
    t.transcribe(np.zeros(16000, np.float32), 16000, "en")
    assert created[-1][1] == "cpu" and sum(1 for c in created if c[1] == "cpu") == 1


def test_cuda_only_device_does_not_fall_back(monkeypatch):
    import faster_whisper

    Fake, _created = _fake_whisper_model()
    monkeypatch.setattr(faster_whisper, "WhisperModel", Fake)
    t = tr.WhisperTranscriber("base", device="cuda")
    with pytest.raises(RuntimeError, match="cudnn"):
        t.transcribe(np.zeros(16000, np.float32), 16000, "fr")


def test_cuda_error_after_success_is_not_swallowed(monkeypatch):
    """Once CUDA has worked, a later failure is a real error, not a reason to go CPU."""
    import faster_whisper

    Fake, created = _fake_whisper_model(fail_on_cuda_transcribe=False)
    monkeypatch.setattr(faster_whisper, "WhisperModel", Fake)
    monkeypatch.setattr(tr, "_cuda_available", lambda: True)
    t = tr.WhisperTranscriber("base.en", device="auto")
    t.transcribe(np.zeros(16000, np.float32), 16000, "en")
    assert t.device == "cuda"

    def boom(model):
        raise ValueError("bad audio")

    with pytest.raises(ValueError):
        t._run(boom)
    assert t.device == "cuda" and [c[1] for c in created] == ["cuda"]


@pytest.mark.parametrize(
    ("requested", "cpu"),
    [("int8", "int8"), ("float32", "float32"), ("float16", "int8"), ("int8_float16", "int8")],
)
def test_cpu_fallback_uses_cpu_compute_type(monkeypatch, requested, cpu):
    import faster_whisper

    Fake, created = _fake_whisper_model(fail_load_cuda=True)
    monkeypatch.setattr(faster_whisper, "WhisperModel", Fake)
    monkeypatch.setattr(tr, "_cuda_available", lambda: True)
    t = tr.WhisperTranscriber("base.en", device="auto", compute_type=requested)
    t.transcribe(np.zeros(16000, np.float32), 16000, "en")
    assert created == [("base.en", "cpu", cpu)]


def test_detect_language(monkeypatch):
    import faster_whisper

    Fake, _ = _fake_whisper_model()
    monkeypatch.setattr(faster_whisper, "WhisperModel", Fake)
    monkeypatch.setattr(tr, "_cuda_available", lambda: False)
    assert tr.WhisperTranscriber("base.en").detect_language(np.zeros(10), 16000) == ("en", 1.0)
    assert tr.WhisperTranscriber("base").detect_language(np.zeros(10), 16000) == ("fr", 0.9)


def test_get_transcriber_cache_and_model_choice():
    tr.clear_cache()
    cfg = Config(model_en="tiny.en", model_multi="tiny")
    en = tr.get_transcriber(cfg, "en")
    assert en.model_name == "tiny.en"
    assert tr.get_transcriber(cfg, "en") is en
    assert tr.get_transcriber(cfg, "es").model_name == "tiny"
    assert tr.get_transcriber(cfg, None).model_name == "tiny"
    assert isinstance(en, tr.Transcriber)
    tr.clear_cache()


# --- slow: real Whisper on the TTS fixtures -------------------------------------------


def _fixture_cases():
    manifest = FIXTURES / "manifest.json"
    if not manifest.exists():
        return []
    files = json.loads(manifest.read_text())["files"]
    cases = []
    for media, spec in files.items():
        truth = spec.get("truth")
        if not truth:
            continue
        for variant in spec.get("variants", {}):
            cases.append((media, variant, truth, "whisper"))
        for variant in spec.get("mismatch_variants", {}):
            cases.append((media, variant, truth, "vad"))
    return cases


@pytest.mark.slow
@pytest.mark.parametrize("media,variant,truth,method", _fixture_cases() or [None])
def test_real_whisper_on_fixtures(tmp_path, media, variant, truth, method):
    if media is None or not (FIXTURES / media).exists():
        pytest.skip("speech fixtures not available")
    cfg = Config(model_en="tiny.en", model_multi="tiny", device="cpu")
    r = sync_subtitles(
        str(FIXTURES / media), 0, SubtitleSource("external", path=str(FIXTURES / variant)),
        str(tmp_path / "out.srt"), cfg,
    )  # fmt: skip
    assert r.applied, r.message
    assert r.method == method
    out = pysubs2.load(r.output_path)
    ref = pysubs2.load(str(FIXTURES / truth))
    err = np.abs(np.array([e.start for e in out]) - np.array([e.start for e in ref])) / 1000
    assert np.median(err) < 0.15, np.median(err)
    assert np.percentile(err, 95) < 0.4, np.percentile(err, 95)
