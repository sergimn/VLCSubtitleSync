"""Media access through PyAV (bundled FFmpeg; no system ffmpeg needed).

* :func:`probe` – audio / subtitle stream listing (ordinals match VLC's track order).
* :func:`decode_audio` – one audio stream → 16 kHz mono float32 numpy array.
* :func:`extract_subtitles` – one embedded *text* subtitle stream → ``pysubs2.SSAFile``.
* :func:`read_media` – both in a single demux pass (subtitles are interleaved with
  audio in MKV/MP4, so a second pass would read the whole file again).

All timestamps are relative to the container start time, which is what VLC (and
external subtitle files) use as t=0.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pysubs2

from .subtitles import SubtitleError, UnsupportedSubtitle

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

__all__ = [
    "SAMPLE_RATE",
    "MediaError",
    "MediaInfo",
    "StreamInfo",
    "UnsupportedSubtitle",
    "decode_audio",
    "extract_subtitles",
    "probe",
    "read_media",
]

TEXT_SUB_CODECS = {
    "ass", "ssa", "subrip", "srt", "mov_text", "webvtt", "text", "microdvd", "subviewer",
    "subviewer1", "sami", "realtext", "jacosub", "mpl2", "pjs", "vplayer", "stl", "eia_608",
    "ttml",
}  # fmt: skip
IMAGE_SUB_CODECS = {
    "hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub", "dvb_subtitle", "dvbsub",
    "xsub", "dvb_teletext", "arib_caption",
}  # fmt: skip

_ISO639_2_TO_1 = {
    "eng": "en", "spa": "es", "fre": "fr", "fra": "fr", "ger": "de", "deu": "de",
    "ita": "it", "por": "pt", "dut": "nl", "nld": "nl", "rus": "ru", "jpn": "ja",
    "chi": "zh", "zho": "zh", "kor": "ko", "pol": "pl", "swe": "sv", "nor": "no",
    "nob": "no", "dan": "da", "fin": "fi", "tur": "tr", "gre": "el", "ell": "el",
    "cze": "cs", "ces": "cs", "hun": "hu", "rum": "ro", "ron": "ro", "ara": "ar",
    "heb": "he", "hin": "hi", "cat": "ca", "ukr": "uk", "tha": "th", "vie": "vi",
    "ind": "id", "baq": "eu", "eus": "eu", "glg": "gl", "bul": "bg", "hrv": "hr",
    "srp": "sr", "slv": "sl", "slo": "sk", "slk": "sk",
}  # fmt: skip


class MediaError(Exception):
    """Media file cannot be opened / stream not found / decode failure."""


@dataclass
class StreamInfo:
    ordinal: int  # ordinal among streams of the same kind (VLC track order)
    index: int  # container stream index
    kind: str  # "audio" | "subtitle"
    codec: str
    language: str | None  # ISO 639-1 when known, else raw tag
    title: str | None = None
    is_text: bool = True  # subtitles only
    channels: int = 0  # audio only
    sample_rate: int = 0  # audio only


@dataclass
class MediaInfo:
    path: str
    duration: float  # seconds (0.0 if unknown)
    audio: list[StreamInfo] = field(default_factory=list)
    subtitles: list[StreamInfo] = field(default_factory=list)

    @property
    def n_text_subtitles(self) -> int:
        return sum(1 for s in self.subtitles if s.is_text)


def _lang(tag: str | None) -> str | None:
    if not tag:
        return None
    t = tag.strip().lower()
    if not t or t in ("und", "unk", "mis", "zxx"):
        return None
    if len(t) == 2:
        return t
    return _ISO639_2_TO_1.get(t, t)


def _codec_name(stream) -> str:
    try:
        return (stream.codec_context.name or "").lower()
    except Exception:  # some subtitle codecs have no decoder in the bundled ffmpeg
        try:
            return (stream.codec.name or "").lower()
        except Exception:
            return "unknown"


def _is_text_codec(codec: str) -> bool:
    if codec in IMAGE_SUB_CODECS:
        return False
    return True if codec in TEXT_SUB_CODECS else codec not in ("unknown", "")


def _open(path: str):
    import av

    try:
        return av.open(str(path))
    except Exception as e:
        raise MediaError(f"cannot open media {path}: {e}") from e


def _info_from_container(container, path: str) -> MediaInfo:
    duration = 0.0
    if container.duration:
        duration = container.duration / 1_000_000.0
    info = MediaInfo(path=str(path), duration=duration)
    for s in container.streams:
        md = dict(s.metadata or {})
        title = md.get("title") or md.get("handler_name")
        lang = _lang(getattr(s, "language", None) or md.get("language"))
        if s.type == "audio":
            cc = s.codec_context
            info.audio.append(
                StreamInfo(
                    ordinal=len(info.audio),
                    index=s.index,
                    kind="audio",
                    codec=_codec_name(s),
                    language=lang,
                    title=title,
                    channels=getattr(cc, "channels", 0) or 0,
                    sample_rate=getattr(cc, "sample_rate", 0) or 0,
                )
            )
            if not duration and s.duration and s.time_base:
                duration = float(s.duration * s.time_base)
        elif s.type == "subtitle":
            codec = _codec_name(s)
            info.subtitles.append(
                StreamInfo(
                    ordinal=len(info.subtitles),
                    index=s.index,
                    kind="subtitle",
                    codec=codec,
                    language=lang,
                    title=title,
                    is_text=_is_text_codec(codec),
                )
            )
    info.duration = duration
    return info


def probe(path: str) -> MediaInfo:
    """List audio and subtitle streams of ``path``."""
    container = _open(path)
    try:
        return _info_from_container(container, path)
    finally:
        container.close()


# --------------------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------------------


class _AudioBuffer:
    """Growable float32 buffer (amortised O(1) appends, one contiguous array)."""

    def __init__(self, capacity: int):
        self.buf = np.zeros(max(capacity, SAMPLE_RATE), dtype=np.float32)
        self.n = 0

    def pad_to(self, n: int) -> None:
        if n > self.n:
            self._reserve(n)
            self.buf[self.n : n] = 0.0
            self.n = n

    def append(self, x: np.ndarray) -> None:
        m = x.shape[0]
        self._reserve(self.n + m)
        self.buf[self.n : self.n + m] = x
        self.n += m

    def _reserve(self, need: int) -> None:
        if need > self.buf.shape[0]:
            new = np.zeros(max(need, int(self.buf.shape[0] * 1.5)), dtype=np.float32)
            new[: self.n] = self.buf[: self.n]
            self.buf = new

    def result(self) -> np.ndarray:
        return self.buf[: self.n].copy() if self.n < self.buf.shape[0] else self.buf


# Mid-stream timestamp jumps smaller than this are treated as jitter (MKV stores ms).
PTS_TOLERANCE = 0.05
# Mid-stream jumps larger than this are timestamp discontinuities (MPEG-TS rollover,
# concatenated files, bad muxes), not real gaps: the frame is kept contiguous.
MAX_PTS_JUMP = 30.0


class _AudioAssembler:
    """Resamples decoded frames to 16 kHz mono and places them on the media timeline.

    Sample position ``i`` of the result is media time ``i / SAMPLE_RATE`` (relative to
    the container start). Frames are normally laid end to end; when a frame's pts says
    it starts more than :data:`PTS_TOLERANCE` after the end of the previous one the gap
    is filled with silence, and when it starts earlier (overlap) the overlapping
    samples are dropped; a jump above :data:`MAX_PTS_JUMP` is a clock discontinuity:
    the frame stays contiguous and the new clock becomes the reference for what
    follows (later gaps and overlaps are measured against it). The first frame is
    placed exactly: leading silence is inserted if it starts after t=0, and audio
    before t=0 is trimmed.
    """

    def __init__(self, capacity: int):
        self.buf = _AudioBuffer(capacity)
        self.resampler = self._new_resampler()
        self.cursor: float | None = None  # media time where the next frame would start
        self.drop = 0  # output samples still to discard (overlap / before t=0)
        self.offset = 0.0  # pts clock minus media timeline (changes at discontinuities)

    @staticmethod
    def _new_resampler():
        import av

        return av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

    def feed(self, frame, t: float | None) -> None:
        """Add one decoded frame starting at media time ``t`` (None if unknown)."""
        rate = frame.sample_rate or SAMPLE_RATE
        dur = frame.samples / rate
        if t is None:
            t = self.cursor if self.cursor is not None else 0.0
        else:
            t -= self.offset
        expected = 0.0 if self.cursor is None else self.cursor
        tol = 0.5 / SAMPLE_RATE if self.cursor is None else PTS_TOLERANCE
        delta = t - expected
        if self.cursor is not None and abs(delta) > MAX_PTS_JUMP:
            log.warning(
                "audio timestamp jump of %+.1f s at %.1f s; treating it as a discontinuity",
                delta,
                expected,
            )
            self.offset += delta
            t, delta = expected, 0.0
        if delta > tol:  # gap: flush what is buffered, then pad with silence
            self._flush()
            self.drop = 0
            self.buf.pad_to(int(round(t * SAMPLE_RATE)))
            self.cursor = t + dur
        elif delta < -tol:  # overlap (or before t=0): drop what was already covered
            self.drop += int(round(-delta * SAMPLE_RATE))
            self.cursor = t + dur
        else:
            self.cursor = expected + dur
        self._append(self.resampler.resample(frame))

    def _append(self, outs) -> None:
        for out in outs:
            x = out.to_ndarray().reshape(-1)
            if self.drop:
                k = min(self.drop, x.shape[0])
                x = x[k:]
                self.drop -= k
            if x.shape[0]:
                self.buf.append(x)

    def _flush(self) -> None:
        self._append(self.resampler.resample(None))
        self.resampler = self._new_resampler()

    def result(self) -> np.ndarray:
        self._flush()
        return self.buf.result()


# --------------------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------------------


class _SubCollector:
    def __init__(self, stream, start_offset: float):
        self.codec = _codec_name(stream)
        self.start_offset = start_offset
        self.items: list[tuple[float, float | None, str]] = []  # (start, end|None, payload)
        extradata = None
        try:
            extradata = stream.codec_context.extradata
        except Exception:
            pass
        self.header = (
            _clean_text(extradata.decode("utf-8", "replace"))
            if extradata and self.codec in ("ass", "ssa")
            else ""
        )

    def add(self, packet) -> None:
        if packet.pts is None:
            return
        data = bytes(packet)
        if not data:
            return
        tb = packet.time_base
        start = float(packet.pts * tb) - self.start_offset
        end = start + float(packet.duration * tb) if packet.duration else None
        text = self._payload(data)
        if text is not None:
            text = _clean_text(text)
        if text is None or not text.strip():
            return
        self.items.append((start, end, text))

    def _payload(self, data: bytes) -> str | None:
        if self.codec == "mov_text":
            if len(data) < 2:
                return None
            n = int.from_bytes(data[:2], "big")
            return data[2 : 2 + n].decode("utf-8", "replace")
        return data.decode("utf-8", "replace")

    def build(self) -> pysubs2.SSAFile:
        items = sorted(self.items, key=lambda x: x[0])
        # Fill missing durations: until the next event, at most 5 s.
        timed: list[tuple[float, float, str]] = []
        for i, (s, e, t) in enumerate(items):
            if e is None or e <= s:
                nxt = items[i + 1][0] if i + 1 < len(items) else s + 5.0
                e = min(s + 5.0, max(nxt, s + 0.5))
            timed.append((max(0.0, s), max(0.0, e), t))
        if self.codec in ("ass", "ssa"):
            return self._build_ass(timed)
        subs = pysubs2.SSAFile()
        for s, e, t in timed:
            subs.append(_srt_event(s, e, t))
        subs.format = "srt"
        return subs

    def _build_ass(self, timed: list[tuple[float, float, str]]) -> pysubs2.SSAFile:
        header = self.header.strip()
        if "[Events]" in header:
            header = header[: header.index("[Events]")].rstrip()
        try:
            subs = pysubs2.SSAFile.from_string(header + "\n\n" + _ASS_EVENTS_HEADER, "ass")
        except Exception:
            subs = pysubs2.SSAFile()
        subs.events.clear()
        for s, e, payload in timed:
            subs.append(_ass_event(s, e, payload))
        subs.format = "ass"
        return subs


# C0/C1 control characters and DEL, except newline (muxers sometimes leave a trailing NUL).
_CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def _clean_text(text: str) -> str:
    """Normalise line breaks to ``\\n``, tabs to spaces, and drop other control chars."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    return _CONTROL_CHARS.sub("", text)


_ASS_EVENTS_HEADER = (
    "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def _ms(t: float) -> int:
    return int(round(t * 1000))


def _ass_event(start: float, end: float, payload: str) -> pysubs2.SSAEvent:
    line = payload.strip("\n")
    if line.startswith("Dialogue:"):
        # Some muxers store full event lines: Layer,Start,End,Style,Name,ML,MR,MV,Effect,Text
        parts = line[len("Dialogue:") :].strip().split(",", 9)
        if len(parts) == 10:
            layer, _s, _e, style, name, ml, mr, mv, effect, text = parts
        else:
            layer, style, name, ml, mr, mv, effect, text = (
                "0",
                "Default",
                "",
                "0",
                "0",
                "0",
                "",
                line,
            )
    else:
        # Matroska: ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        parts = line.split(",", 8)
        if len(parts) == 9:
            _ro, layer, style, name, ml, mr, mv, effect, text = parts
        else:
            layer, style, name, ml, mr, mv, effect, text = (
                "0",
                "Default",
                "",
                "0",
                "0",
                "0",
                "",
                line,
            )
    ev = pysubs2.SSAEvent(start=_ms(start), end=_ms(end), text=text.replace("\n", "\\N"))
    ev.style = style.strip() or "Default"
    ev.name = name
    ev.effect = effect
    for attr, val in (("layer", layer), ("marginl", ml), ("marginr", mr), ("marginv", mv)):
        try:
            setattr(ev, attr, int(val.strip() or 0))
        except ValueError:
            pass
    return ev


def _srt_event(start: float, end: float, text: str) -> pysubs2.SSAEvent:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    # Reuse pysubs2's SRT tag conversion (<i>, <b>, <font>…) by parsing a 1-cue SRT.
    try:
        one = pysubs2.SSAFile.from_string(f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n\n", "srt")
        converted = one[0].text if len(one) else text.replace("\n", "\\N")
    except Exception:
        converted = text.replace("\n", "\\N")
    return pysubs2.SSAEvent(start=_ms(start), end=_ms(end), text=converted)


# --------------------------------------------------------------------------------------
# Single-pass reader
# --------------------------------------------------------------------------------------

ProgressFn = Callable[[float, str], None]


def read_media(
    path: str,
    audio_index: int | None = 0,
    subtitle_index: int | None = None,
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray | None, pysubs2.SSAFile | None, MediaInfo]:
    """Decode audio stream ``audio_index`` (ordinal) and/or extract embedded subtitle
    stream ``subtitle_index`` (ordinal) in one demux pass.

    Returns ``(audio, subs, info)``; ``audio`` is 16 kHz mono float32 or None if
    ``audio_index`` is None, ``subs`` is None if ``subtitle_index`` is None.
    Raises :class:`MediaError` / :class:`UnsupportedSubtitle`.
    """
    import av

    container = _open(path)
    try:
        info = _info_from_container(container, path)
        start_offset = (container.start_time or 0) / 1_000_000.0
        streams = []
        astream = None
        if audio_index is not None:
            if not 0 <= audio_index < len(info.audio):
                raise MediaError(
                    f"audio track {audio_index} not found ({len(info.audio)} audio streams)"
                )
            astream = container.streams[info.audio[audio_index].index]
            streams.append(astream)
        collector = None
        sstream = None
        if subtitle_index is not None:
            if not 0 <= subtitle_index < len(info.subtitles):
                raise SubtitleError(
                    f"subtitle track {subtitle_index} not found "
                    f"({len(info.subtitles)} embedded subtitle streams)"
                )
            sinfo = info.subtitles[subtitle_index]
            if not sinfo.is_text:
                raise UnsupportedSubtitle(
                    f"subtitle track {subtitle_index} is image-based ({sinfo.codec}); "
                    "only text subtitles can be synced"
                )
            sstream = container.streams[sinfo.index]
            collector = _SubCollector(sstream, start_offset)
            streams.append(sstream)
        if not streams:
            return None, None, info

        asm = None
        if astream is not None:
            astream.thread_type = "AUTO"
            est = int((info.duration or 60.0) * SAMPLE_RATE * 1.02) + SAMPLE_RATE
            asm = _AudioAssembler(est)

        duration = info.duration or 0.0
        last_report = -1.0
        decode_errors = 0
        for packet in container.demux(streams):
            if packet.stream is sstream:
                collector.add(packet)
                continue
            if packet.stream is not astream:
                continue
            try:
                frames = packet.decode()
            except av.error.FFmpegError as e:  # corrupt packet: skip it
                decode_errors += 1
                if decode_errors <= 3:
                    log.debug("audio decode error: %s", e)
                continue
            for frame in frames:
                t = None
                if frame.pts is not None and frame.time_base is not None:
                    t = float(frame.pts * frame.time_base) - start_offset
                asm.feed(frame, t)
            if progress and duration > 0 and packet.pts is not None:
                p = min(1.0, max(0.0, float(packet.pts * packet.time_base) / duration))
                if p - last_report >= 0.01:
                    last_report = p
                    progress(p, "Decoding audio")
        audio = None
        if asm is not None:
            audio = asm.result()
            if audio.shape[0] == 0:
                raise MediaError("audio track decoded to no samples")
        subs = collector.build() if collector is not None else None
        if audio is not None and info.duration <= 0:
            info.duration = audio.shape[0] / SAMPLE_RATE
        return audio, subs, info
    except (MediaError, SubtitleError):
        raise
    except Exception as e:
        raise MediaError(f"failed to read {path}: {e}") from e
    finally:
        container.close()


def decode_audio(path: str, audio_index: int = 0, progress: ProgressFn | None = None) -> np.ndarray:
    """Audio stream ``audio_index`` (ordinal among audio streams) → 16 kHz mono float32."""
    audio, _subs, _info = read_media(path, audio_index, None, progress)
    assert audio is not None
    return audio


def extract_subtitles(path: str, subtitle_index: int) -> pysubs2.SSAFile:
    """Embedded text subtitle stream ``subtitle_index`` (ordinal among *all* subtitle
    streams, like VLC) → SSAFile (``.format`` is "ass" for ASS/SSA tracks, else "srt").
    Raises :class:`UnsupportedSubtitle` for image subtitles (PGS, VobSub, DVB)."""
    _audio, subs, _info = read_media(path, None, subtitle_index)
    assert subs is not None
    return subs
