"""Daemon process helpers: run at low priority so VLC playback stays smooth.

:func:`lower_priority` is called once by ``serve`` before any thread exists (Linux
nice and I/O priority are per thread and inherited). Every step is best effort.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

log = logging.getLogger("vlcsubsync.lifecycle")


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
