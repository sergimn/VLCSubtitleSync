import subprocess

import numpy as np
import pytest

from vlcsubsync import media
from vlcsubsync.media import MediaError, UnsupportedSubtitle

SRT = (
    "1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i> there\nsecond line\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nBye\n"
)
ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Sign,Georgia,30,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,2,8,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,Ann,0,0,0,,{\\i1}Hello{\\i0}, there\\Nsecond
Dialogue: 1,0:00:03.00,0:00:04.25,Sign,,0,0,0,,{\\pos(320,40)}Bye
"""  # noqa: E501


def _ffmpeg(synth, *args):
    if not synth.ffmpeg:
        pytest.skip("ffmpeg not available")
    subprocess.run([synth.ffmpeg, "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def mkv(tmp_path, synth):
    (tmp_path / "a.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "a.ass").write_text(ASS, encoding="utf-8")
    out = tmp_path / "t.mkv"
    _ffmpeg(
        synth,
        "-f", "lavfi", "-i", "sine=f=440:d=6:sample_rate=44100",
        "-f", "lavfi", "-i", "anoisesrc=d=6:a=0.1:r=48000",
        "-i", str(tmp_path / "a.srt"), "-i", str(tmp_path / "a.ass"),
        "-map", "0", "-map", "1", "-map", "2", "-map", "3",
        "-c:a", "aac", "-ac:a:1", "2", "-c:s:0", "srt", "-c:s:1", "ass",
        "-metadata:s:a:1", "language=spa", "-metadata:s:s:0", "language=eng",
        str(out),
    )  # fmt: skip
    return out


def test_probe(mkv):
    info = media.probe(str(mkv))
    assert info.duration == pytest.approx(6.0, abs=0.1)
    assert [a.ordinal for a in info.audio] == [0, 1]
    assert info.audio[1].language == "es"
    assert info.audio[1].channels == 2
    assert [s.codec for s in info.subtitles] == ["srt", "ssa"] or [
        s.codec for s in info.subtitles
    ] == ["subrip", "ass"]
    assert info.subtitles[0].language == "en"
    assert info.n_text_subtitles == 2


def test_decode_audio_resamples_to_16k_mono(mkv):
    a0 = media.decode_audio(str(mkv), 0)
    a1 = media.decode_audio(str(mkv), 1)
    for a in (a0, a1):
        assert a.dtype == np.float32 and a.ndim == 1
        assert a.shape[0] / 16000 == pytest.approx(6.0, abs=0.1)
    # track 0 is a 440 Hz sine: check the dominant frequency survived resampling
    seg = a0[16000:32000]
    freq = np.argmax(np.abs(np.fft.rfft(seg))) * 16000 / seg.shape[0]
    assert freq == pytest.approx(440, abs=3)
    assert a1.std() > 0.01  # noise track decoded, not the sine
    assert abs(np.corrcoef(a0[:16000], a1[:16000])[0, 1]) < 0.2


def test_extract_srt_track(mkv):
    subs = media.extract_subtitles(str(mkv), 0)
    assert subs.format == "srt"
    assert [(e.start, e.end) for e in subs] == [(1000, 2500), (3000, 4000)]
    assert subs[0].text == "{\\i1}Hello{\\i0} there\\Nsecond line"
    assert subs[1].text == "Bye"


def test_extract_ass_track_keeps_styles(mkv):
    subs = media.extract_subtitles(str(mkv), 1)
    assert subs.format == "ass"
    assert set(subs.styles) >= {"Default", "Sign"}
    assert subs.styles["Sign"].fontname == "Georgia"
    assert [(e.start, e.end) for e in subs] == [(1000, 2500), (3000, 4250)]
    assert subs[0].text == "{\\i1}Hello{\\i0}, there\\Nsecond"
    assert subs[0].name == "Ann"
    assert subs[1].style == "Sign" and subs[1].layer == 1
    assert subs[1].text == "{\\pos(320,40)}Bye"


def test_single_pass_audio_and_subs(mkv):
    audio, subs, info = media.read_media(str(mkv), 1, 1)
    assert audio is not None and subs is not None
    assert len(subs) == 2 and len(info.audio) == 2


def test_mov_text_in_mp4(tmp_path, synth):
    (tmp_path / "a.srt").write_text(SRT, encoding="utf-8")
    out = tmp_path / "t.mp4"
    _ffmpeg(
        synth,
        "-f", "lavfi", "-i", "sine=f=440:d=6",
        "-i", str(tmp_path / "a.srt"),
        "-map", "0", "-map", "1", "-c:a", "aac", "-c:s", "mov_text", str(out),
    )  # fmt: skip
    subs = media.extract_subtitles(str(out), 0)
    assert [(e.start, e.end) for e in subs] == [(1000, 2500), (3000, 4000)]
    assert subs[0].plaintext == "Hello there\nsecond line"


def test_wav_decoding(tmp_path, synth):
    tone = (0.5 * np.sin(2 * np.pi * 300 * np.arange(16000 * 3) / 16000)).astype(np.float32)
    p = synth.write_wav(tmp_path / "x.wav", tone)
    info = media.probe(str(p))
    assert len(info.audio) == 1 and info.subtitles == []
    a = media.decode_audio(str(p))
    assert a.shape[0] == tone.shape[0]
    assert np.max(np.abs(a - tone)) < 1e-3


def test_errors(tmp_path, mkv):
    with pytest.raises(MediaError):
        media.probe(str(tmp_path / "missing.mkv"))
    with pytest.raises(MediaError):
        media.decode_audio(str(mkv), 5)
    with pytest.raises(Exception, match="subtitle track 7 not found"):
        media.extract_subtitles(str(mkv), 7)


def test_image_subtitles_rejected(mkv, monkeypatch):
    real = media._info_from_container

    def fake_info(container, path):
        info = real(container, path)
        info.subtitles[0].codec = "hdmv_pgs_subtitle"
        info.subtitles[0].is_text = False
        return info

    monkeypatch.setattr(media, "_info_from_container", fake_info)
    with pytest.raises(UnsupportedSubtitle, match="image-based"):
        media.extract_subtitles(str(mkv), 0)
    # the ordinal still counts image tracks: track 1 is the ASS one
    assert media.extract_subtitles(str(mkv), 1).format == "ass"


@pytest.mark.parametrize(
    "codec,text",
    [
        ("hdmv_pgs_subtitle", False),
        ("dvd_subtitle", False),
        ("dvb_subtitle", False),
        ("subrip", True),
        ("ass", True),
        ("mov_text", True),
        ("webvtt", True),
    ],
)
def test_text_codec_classification(codec, text):
    assert media._is_text_codec(codec) is text
