"""Which subtitle files a running VLC has open (best effort, per OS).

VLC 3's Lua API cannot tell which file an external subtitle track came from, but VLC
keeps every subtitle file it loaded open while the video plays. So the helper asks the
OS which files the VLC process that is playing the video has open (see DESIGN.md
"External subs"):

* Linux: ``/proc/<pid>/fd`` of the user's own ``vlc`` processes (snap and Flatpak
  VLC too);
* macOS: ``lsof`` for processes named ``VLC``;
* Windows: the Restart Manager, which lists the processes that have a given file
  open. It cannot list a process's files, so only ``candidates`` are checked there.

Every function returns ``None`` when it cannot tell (no VLC playing that file, the OS
refused, a tool is missing, a timeout), never raises.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

log = logging.getLogger("vlcsubsync.openfiles")

VLC_PROCESS_NAMES = ("vlc",)

# VLC 3.0's SLAVE_SPU_EXTENSIONS (include/vlc_input_item.h): every file VLC may load
# as an external subtitle track.
VLC_SUBTITLE_EXTS = frozenset(
    "." + e
    for e in (
        "aqt ass cdg dks idx jss mpl2 mpsub pjs psb rt sami sbv scc smi srt ssa stl sub "
        "ttml tt usf vtt webvtt"
    ).split()
)


def is_subtitle_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VLC_SUBTITLE_EXTS


def same_file_key(path: str) -> str:
    """Comparable form of a path (symlinks resolved, case folded where the OS does)."""
    try:
        path = os.path.realpath(path)
    except (OSError, ValueError):
        pass
    return os.path.normcase(os.path.normpath(path))


def _name_matches(name: str, names: Sequence[str]) -> bool:
    base = os.path.basename(name.replace("\\", "/")).casefold()
    if base.endswith(".exe"):
        base = base[:-4]
    return base in {n.casefold() for n in names}


def _subtitles_of(
    processes: Iterable[tuple[int, str, list[str]]], media: str, names: Sequence[str]
) -> list[str] | None:
    """Subtitle files of the processes in ``names`` that have ``media`` open, in the
    order listed; None when no such process has it open."""
    media_key = same_file_key(media)
    own = os.getpid()
    found = False
    out: list[str] = []
    seen: set[str] = set()
    for pid, name, paths in processes:
        if pid == own or not _name_matches(name, names):
            continue
        keys = [same_file_key(p) for p in paths]
        if media_key not in keys:
            continue
        found = True
        for p, k in zip(paths, keys, strict=True):
            if k not in seen and is_subtitle_file(p):
                seen.add(k)
                out.append(p)
    return out if found else None


# --------------------------------------------------------------------------- Linux


def _linux_processes(proc: Path, names: Sequence[str]) -> list[tuple[int, str, list[str]]]:
    out = []
    try:
        entries = os.listdir(proc)
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        pdir = proc / entry
        try:
            name = (pdir / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not _name_matches(name, names):
            continue
        paths = []
        try:
            fds = os.listdir(pdir / "fd")
        except OSError:  # another user's process, gone, or the kernel refused
            continue
        for fd in sorted(fds, key=lambda s: int(s) if s.isdigit() else 0):
            try:
                target = os.readlink(pdir / "fd" / fd)
            except OSError:
                continue
            if target.endswith(" (deleted)"):
                # replaced since VLC opened it (saved again through a rename): the
                # same subtitle file to VLC, if the path is there again
                target = target[: -len(" (deleted)")]
                if not os.path.isfile(target):
                    continue
            if target.startswith("/"):
                paths.append(target)
        out.append((int(entry), name, paths))
    return out


# --------------------------------------------------------------------------- macOS

_LSOF_ESCAPE = re.compile(r"\\x([0-9a-fA-F]{2})|\\([\\tnrbf])")
_LSOF_SIMPLE = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r", "b": "\b", "f": "\f"}


def _lsof_unescape(name: str) -> str:
    """lsof prints bytes it deems unprintable (all non-ASCII ones) as ``\\xNN``."""
    if "\\" not in name:
        return name
    buf = bytearray()
    pos = 0
    for m in _LSOF_ESCAPE.finditer(name):
        buf += name[pos : m.start()].encode("utf-8", "surrogateescape")
        if m.group(1):
            buf.append(int(m.group(1), 16))
        else:
            buf += _LSOF_SIMPLE[m.group(2)].encode()
        pos = m.end()
    buf += name[pos:].encode("utf-8", "surrogateescape")
    return buf.decode("utf-8", "replace")


def parse_lsof(output: str) -> list[tuple[int, str, list[str]]]:
    """Parse ``lsof -F pcn`` output into (pid, command, paths)."""
    procs: list[tuple[int, str, list[str]]] = []
    for line in output.splitlines():
        if not line:
            continue
        tag, val = line[0], line[1:]
        if tag == "p":
            try:
                procs.append((int(val), "", []))
            except ValueError:
                continue
        elif not procs:
            continue
        elif tag == "c":
            pid, _, paths = procs[-1]
            procs[-1] = (pid, val, paths)
        elif tag == "n" and val.startswith("/"):
            procs[-1][2].append(_lsof_unescape(val))
    return procs


def _darwin_processes(
    names: Sequence[str], timeout: float, run: Callable[..., subprocess.CompletedProcess]
) -> list[tuple[int, str, list[str]]] | None:
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    cmd = [lsof, "-nP", "-w", "+c", "0", "-F", "pcn"]
    for n in names:
        cmd += ["-c", f"/^{re.escape(n)}$/i"]
    try:
        cp = run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("lsof failed: %s", exc)
        return None
    # lsof exits 1 when it found nothing to list
    if cp.returncode not in (0, 1):
        log.debug("lsof exited with %s: %s", cp.returncode, (cp.stderr or "").strip()[:200])
        return None
    return parse_lsof(cp.stdout or "")


# --------------------------------------------------------------------------- Windows


def _windows_holders(path: str):  # pragma: no cover - Windows only
    """(pid, app name) of the processes that have ``path`` open, via the Restart
    Manager (rstrtmgr.dll), or None if it cannot tell."""
    import ctypes
    from ctypes import wintypes

    class RM_UNIQUE_PROCESS(ctypes.Structure):  # noqa: N801 - Windows API name
        _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", wintypes.FILETIME)]

    class RM_PROCESS_INFO(ctypes.Structure):  # noqa: N801 - Windows API name
        _fields_ = [
            ("Process", RM_UNIQUE_PROCESS),
            ("strAppName", wintypes.WCHAR * 256),  # CCH_RM_MAX_APP_NAME + 1
            ("strServiceShortName", wintypes.WCHAR * 64),  # CCH_RM_MAX_SVC_NAME + 1
            ("ApplicationType", ctypes.c_int),
            ("AppStatus", wintypes.ULONG),
            ("TSSessionId", wintypes.DWORD),
            ("bRestartable", wintypes.BOOL),
        ]

    error_more_data = 234
    try:
        rm = ctypes.WinDLL("rstrtmgr")
    except OSError:
        return None
    rm.RmStartSession.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, wintypes.LPWSTR]
    rm.RmStartSession.restype = wintypes.DWORD
    rm.RmRegisterResources.argtypes = [
        wintypes.DWORD,
        wintypes.UINT,
        ctypes.POINTER(wintypes.LPCWSTR),
        wintypes.UINT,
        ctypes.c_void_p,
        wintypes.UINT,
        ctypes.c_void_p,
    ]
    rm.RmRegisterResources.restype = wintypes.DWORD
    rm.RmGetList.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(wintypes.UINT),
        ctypes.POINTER(wintypes.UINT),
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    rm.RmGetList.restype = wintypes.DWORD
    rm.RmEndSession.argtypes = [wintypes.DWORD]
    rm.RmEndSession.restype = wintypes.DWORD

    session = wintypes.DWORD(0)
    key = ctypes.create_unicode_buffer(33)  # CCH_RM_SESSION_KEY + 1
    if rm.RmStartSession(ctypes.byref(session), 0, key) != 0:
        return None
    try:
        files = (wintypes.LPCWSTR * 1)(path)
        if rm.RmRegisterResources(session, 1, files, 0, None, 0, None) != 0:
            return None
        reasons = wintypes.DWORD(0)
        n = 0
        for _ in range(5):  # the list can grow between the two calls
            infos = (RM_PROCESS_INFO * max(n, 1))()
            needed = wintypes.UINT(0)
            count = wintypes.UINT(n)
            ret = rm.RmGetList(
                session, ctypes.byref(needed), ctypes.byref(count), infos, ctypes.byref(reasons)
            )
            if ret == 0:
                return [
                    (int(infos[i].Process.dwProcessId), infos[i].strAppName)
                    for i in range(count.value)
                ]
            if ret != error_more_data:
                return None
            n = needed.value
        return None
    finally:
        rm.RmEndSession(session)


def _windows_image_name(pid: int) -> str:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    # a private handle: setting argtypes on the shared ctypes.windll.kernel32 would
    # change it for the daemon's other callers
    kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        kernel32.CloseHandle(handle)


def _windows_subtitles(
    media: str,
    candidates: Sequence[str],
    names: Sequence[str],
    timeout: float,
    holders: Callable[[str], list[tuple[int, str]] | None],
    image_name: Callable[[int], str],
) -> list[str] | None:
    deadline = time.monotonic() + timeout
    media_holders = holders(media)
    if not media_holders:
        return None
    pids = set()
    for pid, app in media_holders:
        if pid == os.getpid():
            continue
        image = image_name(pid)
        # the image path if we may read it, else the app name ("VLC media player")
        words = [image] if image else app.split()
        if any(_name_matches(w, names) for w in words):
            pids.add(pid)
    if not pids:
        return None
    out = []
    for c in candidates:
        if time.monotonic() > deadline:
            log.debug("open-file check timed out")
            return None
        h = holders(c)
        if h is None:
            return None
        if any(pid in pids for pid, _ in h):
            out.append(c)
    return out


# --------------------------------------------------------------------------- entry point


def subtitles_open_in_vlc(
    media: str,
    candidates: Iterable[str] = (),
    *,
    names: Sequence[str] = VLC_PROCESS_NAMES,
    timeout: float = 5.0,
    platform: str | None = None,
    proc: Path = Path("/proc"),
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[str] | None:
    """Subtitle files that a VLC process playing ``media`` has open, or None when
    that cannot be told (no VLC process has ``media`` open, unsupported OS, error).

    On Linux and macOS this lists every subtitle file of that process; on Windows only
    the ``candidates`` are checked. ``names`` are the process names to look at.
    """
    platform = platform or sys.platform
    cands = [c for c in candidates if is_subtitle_file(c)]
    try:
        if platform.startswith("linux"):
            if not proc.is_dir():
                return None
            return _subtitles_of(_linux_processes(proc, names), media, names)
        if platform == "darwin":
            procs = _darwin_processes(names, timeout, run)
            return None if procs is None else _subtitles_of(procs, media, names)
        if platform == "win32":  # pragma: no cover - Windows only
            return _windows_subtitles(
                media, cands, names, timeout, _windows_holders, _windows_image_name
            )
    except Exception as exc:  # noqa: BLE001 - best effort, never fail a job over it
        log.debug("cannot list VLC's open files: %s", exc)
    return None
