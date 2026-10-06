"""File protocol shared by the VLC Lua scripts and the daemon.

All files are UTF-8 ``key=value`` lines (see DESIGN.md, "File protocol").  Writers
always write a temporary file and atomically rename it over the target; readers are
tolerant (ignore blank lines, comments, unknown keys, a UTF-8 BOM and CRLF endings).
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import normalize_mode, normalize_sync_mode

PROTOCOL_VERSION = 1

REQUEST_SUFFIX = ".req"
STATUS_SUFFIX = ".status"
TMP_SUFFIX = ".tmp"

REQUESTS_DIR = "requests"
JOBS_DIR = "jobs"
OUT_DIR = "out"
HEARTBEAT_FILE = "heartbeat"
CONTROL_FILE = "control"
INTF_STATE_FILE = "intf_state"

STATES = ("queued", "running", "done", "error")

# Mapping segments in a done status (see DESIGN.md "Mapping segments").
MAX_SEGMENTS = 256
MAX_KNOTS = 4096  # per segment

_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,128}$")
_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# C0 control characters (incl. CR/LF/TAB), DEL and the Unicode line/paragraph separators.
_CTRL_RE = re.compile("[\x00-\x1f\x7f  \x85]")


class ProtocolError(ValueError):
    """Raised for malformed protocol data (bad id, bad key...)."""


# --------------------------------------------------------------------------- helpers


def is_valid_id(value: str) -> bool:
    """True if *value* is a safe job id (``[0-9A-Za-z_-]+``; no path components)."""
    return isinstance(value, str) and bool(_ID_RE.match(value))


def validate_id(value: str) -> str:
    if not is_valid_id(value):
        raise ProtocolError(f"invalid job id: {value!r}")
    return value


def sanitize_value(value: object) -> str:
    """Render *value* as a single protocol line value.

    Newlines and other control characters become spaces; booleans become ``1``/``0``;
    ``None`` becomes the empty string.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value)
    return _CTRL_RE.sub(" ", text)


def format_kv(data: Mapping[str, object]) -> str:
    lines = []
    for key, value in data.items():
        if not _KEY_RE.match(key):
            raise ProtocolError(f"invalid key: {key!r}")
        lines.append(f"{key}={sanitize_value(value)}")
    return "\n".join(lines) + "\n"


def parse_kv(text: str) -> dict[str, str]:
    """Parse ``key=value`` lines.  Later duplicates win.  Never raises."""
    result: dict[str, str] = {}
    if text.startswith("﻿"):
        text = text[1:]
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        result[key] = value.strip(" \t")
    return result


def read_kv(path: str | os.PathLike[str]) -> dict[str, str] | None:
    """Read a key=value file.  Returns ``None`` if it does not exist / is unreadable."""
    try:
        data = Path(path).read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return None
    return parse_kv(data.decode("utf-8", errors="replace"))


def write_text_atomic(path: str | os.PathLike[str], text: str, retries: int = 10) -> None:
    """Write *text* (UTF-8) to *path* atomically: temp file in same dir + ``os.replace``.

    On Windows the replace can transiently fail while a reader holds the target open;
    retry a few times.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=TMP_SUFFIX, dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
            fh.flush()
        delay = 0.02
        for attempt in range(retries):
            try:
                os.replace(tmp_name, target)
                break
            except PermissionError:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.5)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_kv(path: str | os.PathLike[str], data: Mapping[str, object]) -> None:
    write_text_atomic(path, format_kv(data))


def _to_int(value: str | None, default: int | None) -> int | None:
    if value is None or value.strip() == "":
        return default
    try:
        return int(float(value.strip()))
    except ValueError:
        return default


def _to_float(value: str | None, default: float | None) -> float | None:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def media_path_from_value(value: str) -> str:
    """Accept either a plain path or a ``file://`` URI (decoded to a local path)."""
    value = value.strip()
    if value.lower().startswith("file://"):
        parts = urlsplit(value)
        path = unquote(parts.path)
        if parts.netloc and parts.netloc.lower() != "localhost":
            path = f"//{parts.netloc}{path}"  # UNC share
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        if os.name == "nt":
            path = path.replace("/", "\\")
        return path
    return value


# --------------------------------------------------------------------------- mapping


@dataclass(frozen=True)
class MapSegment:
    """One piece of the subtitle → audio mapping, in the subtitle clock (seconds):
    ``audio = scale*sub + offset + correction(sub)`` for ``sub_start <= sub < sub_end``.

    ``None`` bounds are open (the first segment starts at -inf, the last ends at
    +inf). ``knots`` are the local refinement's ``(sub_time, seconds)`` points,
    linearly interpolated and held flat outside (``align.Segment.knots``)."""

    sub_start: float | None
    sub_end: float | None
    scale: float
    offset: float
    knots: tuple[tuple[float, float], ...] = ()

    def correction(self, t: float) -> float:
        k = self.knots
        if not k:
            return 0.0
        if t <= k[0][0]:
            return k[0][1]
        for (x0, y0), (x1, y1) in zip(k, k[1:], strict=False):
            if t <= x1:
                return y0 if x1 <= x0 else y0 + (y1 - y0) * (t - x0) / (x1 - x0)
        return k[-1][1]

    def audio(self, t: float) -> float:
        return self.scale * t + self.offset + self.correction(t)


def _fmt_bound(x: float | None) -> str:
    return "" if x is None or not math.isfinite(x) else f"{x:.3f}"


def format_segment(seg: MapSegment) -> str:
    """``<sub_start>,<sub_end>,<scale>,<offset>``; an open bound is empty."""
    return f"{_fmt_bound(seg.sub_start)},{_fmt_bound(seg.sub_end)},{seg.scale:.7f},{seg.offset:.4f}"


def format_knots(knots: tuple[tuple[float, float], ...]) -> str:
    """``<sub_time>:<seconds>;...``"""
    return ";".join(f"{t:.3f}:{c:.4f}" for t, c in knots[:MAX_KNOTS])


def _finite(text: str) -> float | None:
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_segment(value: str, knots: str = "") -> MapSegment | None:
    parts = [p.strip() for p in (value or "").split(",")]
    if len(parts) != 4:
        return None
    lo = None if parts[0] == "" else _finite(parts[0])
    hi = None if parts[1] == "" else _finite(parts[1])
    scale, offset = _finite(parts[2]), _finite(parts[3])
    if (parts[0] and lo is None) or (parts[1] and hi is None):
        return None
    if scale is None or offset is None or scale <= 0:
        return None
    if lo is not None and hi is not None and hi < lo:
        return None
    pts: list[tuple[float, float]] = []
    for item in (knots or "").split(";"):
        if not item.strip():
            continue
        t, sep, c = item.partition(":")
        tv, cv = _finite(t), _finite(c)
        if not sep or tv is None or cv is None:
            return None
        pts.append((tv, cv))
    pts.sort()
    return MapSegment(lo, hi, scale, offset, tuple(pts[:MAX_KNOTS]))


# --------------------------------------------------------------------------- dataclasses


@dataclass
class Request:
    """``<q>/requests/<id>.req`` (Lua -> daemon)."""

    id: str
    media: str
    audio_index: int = 0
    sub_index: int | None = None
    sub_path: str = ""
    audio_label: str = ""
    sub_label: str = ""
    force: bool = False
    # Optional sync mode override ("fast" | "thorough" | "exhaustive"); "" = config's.
    mode: str = ""
    version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "version": self.version,
            "id": self.id,
            "media": self.media,
            "audio_index": self.audio_index,
            "audio_label": self.audio_label,
            "sub_index": "" if self.sub_index is None else self.sub_index,
            "sub_label": self.sub_label,
            "sub_path": self.sub_path,
            "force": self.force,
        }
        mode = normalize_mode(self.mode)
        if mode:
            out["mode"] = mode
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, str], fallback_id: str | None = None) -> Request:
        """Build a request from parsed key/values.

        Raises ``ProtocolError`` if the id is invalid or the media path missing.
        """
        req_id = (data.get("id") or "").strip()
        if not is_valid_id(req_id):
            if fallback_id is not None and is_valid_id(fallback_id):
                req_id = fallback_id
            else:
                raise ProtocolError(f"invalid job id: {req_id!r}")
        media = media_path_from_value(data.get("media") or "")
        if not media:
            raise ProtocolError("request has no media path")
        audio_index = _to_int(data.get("audio_index"), 0)
        sub_index = _to_int(data.get("sub_index"), None)
        if sub_index is not None and sub_index < 0:
            sub_index = None
        sub_path = data.get("sub_path") or ""
        if sub_path:
            sub_path = media_path_from_value(sub_path)
        return cls(
            id=req_id,
            media=media,
            audio_index=max(0, audio_index or 0),
            sub_index=sub_index,
            sub_path=sub_path,
            audio_label=data.get("audio_label") or "",
            sub_label=data.get("sub_label") or "",
            force=_to_bool(data.get("force")),
            mode=normalize_mode(data.get("mode")) or "",  # unknown modes are ignored
            version=_to_int(data.get("version"), PROTOCOL_VERSION) or PROTOCOL_VERSION,
        )


@dataclass
class Status:
    """``<q>/jobs/<id>.status`` (daemon -> Lua)."""

    id: str
    state: str = "queued"
    progress: float = 0.0
    message: str = ""
    output: str | None = None
    applied: bool | None = None
    method: str | None = None
    offset: float | None = None
    scale: float | None = None
    confidence: float | None = None
    # Done + applied: the full mapping (track and delay mode), see MapSegment.
    segments: list[MapSegment] | None = None
    # The helper's configured sync_mode ("track" | "delay"); the VLC side uses it
    # unless the extension's toggle wrote its own choice into <q>/control.
    sync_mode: str | None = None
    time: float | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "id": self.id,
            "state": self.state,
            "progress": f"{max(0.0, min(1.0, float(self.progress))):.3f}",
            "message": self.message,
        }
        if self.output is not None:
            out["output"] = self.output
        if self.applied is not None:
            out["applied"] = self.applied
        if self.method is not None:
            out["method"] = self.method
        if self.offset is not None:
            out["offset"] = f"{self.offset:.3f}"
        if self.scale is not None:
            out["scale"] = f"{self.scale:.6f}"
        if self.confidence is not None:
            out["confidence"] = f"{self.confidence:.3f}"
        if self.segments:
            segs = list(self.segments)[:MAX_SEGMENTS]
            out["segments"] = len(segs)
            for i, seg in enumerate(segs):
                out[f"seg{i}"] = format_segment(seg)
                if seg.knots:
                    out[f"seg{i}_knots"] = format_knots(seg.knots)
        sync_mode = normalize_sync_mode(self.sync_mode)
        if sync_mode:
            out["sync_mode"] = sync_mode
        out["time"] = int(self.time if self.time is not None else time.time())
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> Status:
        applied = data.get("applied")
        segments: list[MapSegment] | None = None
        n = _to_int(data.get("segments"), None)
        if n is not None and 0 < n <= MAX_SEGMENTS:
            parsed = [
                parse_segment(data.get(f"seg{i}", ""), data.get(f"seg{i}_knots", ""))
                for i in range(n)
            ]
            # all or nothing: a partial mapping would mis-time the gaps
            if all(p is not None for p in parsed):
                segments = [p for p in parsed if p is not None]
        return cls(
            id=data.get("id", ""),
            state=data.get("state", "queued"),
            progress=_to_float(data.get("progress"), 0.0) or 0.0,
            message=data.get("message", ""),
            output=data.get("output") or None,
            applied=None if applied in (None, "") else _to_bool(applied),
            method=data.get("method") or None,
            offset=_to_float(data.get("offset"), None),
            scale=_to_float(data.get("scale"), None),
            confidence=_to_float(data.get("confidence"), None),
            segments=segments,
            sync_mode=normalize_sync_mode(data.get("sync_mode")),
            time=_to_float(data.get("time"), None),
        )


@dataclass
class Heartbeat:
    """``<q>/heartbeat`` (daemon)."""

    time: float
    pid: int
    version: str

    def to_dict(self) -> dict[str, object]:
        return {"time": int(self.time), "pid": self.pid, "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> Heartbeat:
        return cls(
            time=_to_float(data.get("time"), 0.0) or 0.0,
            pid=_to_int(data.get("pid"), 0) or 0,
            version=data.get("version", ""),
        )

    def age(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.time


# --------------------------------------------------------------------------- file helpers


def request_path(queue_dir: str | os.PathLike[str], req_id: str) -> Path:
    return Path(queue_dir) / REQUESTS_DIR / f"{validate_id(req_id)}{REQUEST_SUFFIX}"


def status_path(queue_dir: str | os.PathLike[str], req_id: str) -> Path:
    return Path(queue_dir) / JOBS_DIR / f"{validate_id(req_id)}{STATUS_SUFFIX}"


def write_request(queue_dir: str | os.PathLike[str], request: Request) -> Path:
    path = request_path(queue_dir, request.id)
    write_kv(path, request.to_dict())
    return path


def read_request(path: str | os.PathLike[str]) -> Request:
    data = read_kv(path)
    if data is None:
        raise ProtocolError(f"cannot read request {path}")
    stem = Path(path).name
    if stem.endswith(REQUEST_SUFFIX):
        stem = stem[: -len(REQUEST_SUFFIX)]
    return Request.from_dict(data, fallback_id=stem)


def write_status(queue_dir: str | os.PathLike[str], status: Status) -> Path:
    path = status_path(queue_dir, status.id)
    write_kv(path, status.to_dict())
    return path


def read_status(queue_dir: str | os.PathLike[str], req_id: str) -> Status | None:
    data = read_kv(status_path(queue_dir, req_id))
    return None if data is None else Status.from_dict(data)


def write_heartbeat(queue_dir: str | os.PathLike[str], heartbeat: Heartbeat) -> None:
    write_kv(Path(queue_dir) / HEARTBEAT_FILE, heartbeat.to_dict())


def read_heartbeat(queue_dir: str | os.PathLike[str]) -> Heartbeat | None:
    data = read_kv(Path(queue_dir) / HEARTBEAT_FILE)
    return None if data is None else Heartbeat.from_dict(data)
