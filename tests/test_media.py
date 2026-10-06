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


# --- audio timeline follows the media timestamps -------------------------------------


def _rms(a, t0, t1):
    seg = a[int(t0 * 16000) : int(t1 * 16000)]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))


def _shifted_tone(tmp_path, synth, shift):
    """2 s, 440 Hz tone whose second half (from the first frame at t >= 1 s) is muxed
    ``shift`` seconds later (gap) or earlier (overlap). FLAC frames are 72 ms here."""
    out = tmp_path / f"shift{shift}.mkv"
    _ffmpeg(
        synth,
        "-f", "lavfi", "-i", "sine=f=440:d=2:sample_rate=16000",
        "-af", f"asetpts='if(gte(T,1),PTS+({shift})/TB,PTS)'",
        "-c:a", "flac", str(out),
    )  # fmt: skip
    return out


def test_decode_audio_fills_mid_stream_pts_gap_with_silence(tmp_path, synth):
    a = media.decode_audio(str(_shifted_tone(tmp_path, synth, 2.0)))
    # 1.008 s tone, 2 s gap, 0.992 s tone
    assert a.shape[0] / 16000 == pytest.approx(4.0, abs=0.02)
    assert _rms(a, 0.1, 0.9) > 0.05
    assert _rms(a, 1.1, 2.9) < 1e-3
    assert _rms(a, 3.1, 3.9) > 0.05


def test_decode_audio_drops_overlapping_samples(tmp_path, synth):
    a = media.decode_audio(str(_shifted_tone(tmp_path, synth, -0.5)))
    # the second half starts 0.5 s before the first one ends: 0.5 s is dropped
    assert a.shape[0] / 16000 == pytest.approx(1.5, abs=0.02)
    assert _rms(a, 0.0, 1.5) > 0.05


def _frame(x, sr=16000):
    import av

    f = av.AudioFrame.from_ndarray(x.reshape(1, -1).astype(np.float32), format="flt", layout="mono")
    f.sample_rate = sr
    return f


def test_audio_assembler_trims_before_zero_and_pads_after():
    ramp = np.arange(16000, dtype=np.float32) / 16000  # 1 s, value == time
    asm = media._AudioAssembler(16000)
    asm.feed(_frame(ramp[:8000]), -0.25)  # first frame starts before t=0
    asm.feed(_frame(ramp[8000:]), 0.25)
    a = asm.result()
    assert a.shape[0] == 12000  # 0.25 s trimmed
    assert a[0] == pytest.approx(0.25, abs=1e-3)
    assert a[8000] == pytest.approx(0.75, abs=1e-3)

    asm = media._AudioAssembler(16000)
    asm.feed(_frame(ramp[:8000]), 0.5)  # leading silence
    asm.feed(_frame(ramp[8000:]), 1.0 + 0.03)  # jitter below the tolerance is ignored
    a = asm.result()
    assert a.shape[0] == 24000
    assert not a[:8000].any()
    assert a[8000] == pytest.approx(0.0, abs=1e-3) and a[16000] == pytest.approx(0.5, abs=1e-3)


# --- subtitle text cleanup -------------------------------------------------------------


def test_extract_subtitles_strips_control_characters(tmp_path, synth):
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nRaymond. Sup?@@NUL@@\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nA@@CTL@@b\nnext line\n"
    )
    (tmp_path / "p.srt").write_text(srt, encoding="utf-8")
    src = tmp_path / "p.mkv"
    _ffmpeg(
        synth,
        "-f", "lavfi", "-i", "sine=d=5", "-i", str(tmp_path / "p.srt"),
        "-map", "0", "-map", "1", "-c:a", "flac", "-c:s", "srt",
        "-write_crc32", "0", str(src),
    )  # fmt: skip
    # Patch control bytes into the stored text (same length; no CRCs to fix up).
    data = src.read_bytes()
    assert data.count(b"@@NUL@@") == 1 and data.count(b"@@CTL@@") == 1
    data = data.replace(b"@@NUL@@", b"\x00" * 7).replace(b"@@CTL@@", b"\x07\x01\x7f\t\x1b\x00\x00")
    dst = tmp_path / "nul.mkv"
    dst.write_bytes(data)

    subs = media.extract_subtitles(str(dst), 0)
    assert [(e.start, e.end) for e in subs] == [(1000, 2000), (3000, 4000)]
    assert subs[0].plaintext == "Raymond. Sup?"
    assert subs[1].plaintext == "A b\nnext line"


def test_clean_text():
    assert media._clean_text("a\x00b\r\nc\rd\te\x1b\x85") == "ab\nc\nd e"
