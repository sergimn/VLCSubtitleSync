"""Daemon process helpers: free Whisper when idle, run at low priority.

:func:`unload_models` is called by the daemon's worker thread a minute after the last
job, so it never races with a running job. :func:`lower_priority` is called once by
``serve`` before any thread exists (Linux nice and I/O priority are per thread and
inherited). Every step is best effort.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
from typing import Any

log = logging.getLogger("vlcsubsync.lifecycle")

MODEL_IDLE_SECONDS = 60.0  # unload Whisper after this long without a job


# --------------------------------------------------------------------------- models


def unload_models() -> int:
    """Free the Whisper (and Silero VAD) models if they were loaded.

    Only touches modules that are already imported, so it never imports
    faster-whisper. Returns the number of Whisper models released. The daemon calls
    it on its worker thread, so it never races with a running job.
    """
    released = 0
    tr = sys.modules.get("vlcsubsync.transcribe")
    if tr is not None:
        released = int(tr.clear_cache() or 0)
    vad = sys.modules.get("vlcsubsync.vad")
    if vad is not None and getattr(vad, "_model", None) is not None:
        lock = getattr(vad, "_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            vad._model = None
    # faster-whisper caches the Silero session itself (functools.lru_cache), and its
    # own vad_filter path uses the same cache: clear it, or nothing is freed.
    fw_vad = sys.modules.get("faster_whisper.vad")
    cached = getattr(fw_vad, "get_vad_model", None)
    if cached is not None and hasattr(cached, "cache_clear"):
        cached.cache_clear()
    gc.collect()
    _malloc_trim()
    return released


def _malloc_trim() -> None:
    """Give freed heap back to the OS (glibc keeps it otherwise)."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def rss_mb() -> float | None:
    """Resident memory of this process in MiB (best effort, Linux only, for logs)."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", encoding="ascii") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return None
    return None


# --------------------------------------------------------------------------- priority

NICE_LEVEL = 10

# Linux ioprio_set(2): syscall numbers per architecture
IOPRIO_WHO_PROCESS = 1
IOPRIO_CLASS_IDLE = 3
IOPRIO_CLASS_SHIFT = 13
IOPRIO_SET_SYSCALL = {
    "x86_64": 251,
    "amd64": 251,
    "aarch64": 30,
    "arm64": 30,
    "riscv64": 30,
    "i386": 289,
    "i686": 289,
    "armv7l": 314,
    "ppc64le": 273,
    "s390x": 282,
}

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def lower_priority(
    *,
    platform: str | None = None,
    os_mod: Any = os,
    ctypes_mod: Any = None,
    machine: str | None = None,
) -> list[str]:
    """Make this process polite to VLC. Returns what was applied (for the log).

    * POSIX: nice 10 (never lowers a higher nice value, e.g. the systemd unit's).
    * Linux: also the idle I/O class (``ioprio_set``) and ``SCHED_BATCH``. The systemd
      unit sets the same; this covers the daemon spawned by VLC itself.
    * Windows: ``BELOW_NORMAL_PRIORITY_CLASS``.

    Call it before starting threads: on Linux nice and I/O priority are per thread
    and inherited by threads created afterwards. Never raises.
    ``VLC_SUBSYNC_PRIORITY=normal`` skips it.
    """
    if os.environ.get("VLC_SUBSYNC_PRIORITY", "").strip().lower() == "normal":
        return []
    plat = (platform or sys.platform).lower()
    try:
        if plat.startswith(("win", "cygwin")):
            return _lower_priority_windows(ctypes_mod)
        applied = _lower_nice(os_mod)
        if plat.startswith("linux"):
            applied += _linux_batch(os_mod)
            applied += _linux_idle_io(ctypes_mod, machine)
        return applied
    except Exception as exc:  # noqa: BLE001 - priority is best effort, never fatal
        log.debug("lowering priority failed: %s", exc)
        return []


def _lower_nice(os_mod: Any) -> list[str]:
    try:
        current = os_mod.getpriority(os_mod.PRIO_PROCESS, 0)
    except (OSError, AttributeError):
        current = None
    if current is not None and current >= NICE_LEVEL:
        return [f"nice {current} (unchanged)"]
    try:
        new = os_mod.nice(NICE_LEVEL - (current or 0))
        return [f"nice {new}"]
    except (OSError, AttributeError) as exc:
        log.debug("nice failed: %s", exc)
        return []


def _linux_batch(os_mod: Any) -> list[str]:
    try:
        os_mod.sched_setscheduler(0, os_mod.SCHED_BATCH, os_mod.sched_param(0))
        return ["SCHED_BATCH"]
    except (OSError, AttributeError) as exc:
        log.debug("SCHED_BATCH failed: %s", exc)
        return []


def _linux_idle_io(ctypes_mod: Any, machine: str | None) -> list[str]:
    if machine is None:
        import platform as _platform

        machine = _platform.machine()
    nr = IOPRIO_SET_SYSCALL.get((machine or "").lower())
    if nr is None:
        return []
    try:
        if ctypes_mod is None:
            import ctypes as ctypes_mod  # noqa: N813
        libc = ctypes_mod.CDLL(None, use_errno=True)
        prio = IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT
        if libc.syscall(nr, IOPRIO_WHO_PROCESS, 0, prio) == 0:
            return ["I/O class idle"]
        log.debug("ioprio_set failed: errno %s", ctypes_mod.get_errno())
    except (OSError, AttributeError) as exc:
        log.debug("ioprio_set unavailable: %s", exc)
    return []


def _lower_priority_windows(ctypes_mod: Any) -> list[str]:
    try:
        if ctypes_mod is None:
            import ctypes as ctypes_mod  # noqa: N813
        kernel32 = ctypes_mod.windll.kernel32
        if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
            return ["BELOW_NORMAL_PRIORITY_CLASS"]
    except (OSError, AttributeError) as exc:
        log.debug("SetPriorityClass unavailable: %s", exc)
    return []
