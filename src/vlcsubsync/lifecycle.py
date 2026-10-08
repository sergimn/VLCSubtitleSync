"""Daemon lifecycle: run only while VLC runs, unload Whisper when idle, low priority.

The helper is started on demand when VLC starts (systemd path unit, launchd
WatchPaths/QueueDirectories, or the Lua interface spawning it; see DESIGN.md
"Lifecycle") and exits by itself once VLC is gone:

* VLC's interface script rewrites ``<q>/intf_state`` (``time=<unix seconds>``) at
  least every 5 s and writes ``state=stopped`` when VLC closes.
* :class:`IdleExitPolicy` decides when to exit: no live VLC in any watched queue dir
  and no job queued or running for ``idle_grace`` seconds. A freshly started daemon
  waits ``startup_grace`` seconds for VLC (or a first request) to show up, unless
  VLC has just said goodbye with ``state=stopped``.

Everything takes explicit clocks so it can be tested without sleeping.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import protocol as P

log = logging.getLogger("vlcsubsync.lifecycle")

INTF_FRESH_SECONDS = 20.0  # intf_state older than this = VLC is gone (crashed / killed)
IDLE_GRACE_SECONDS = 15.0  # stay this long after the last VLC / job activity
STARTUP_GRACE_SECONDS = 60.0  # a fresh daemon waits this long for VLC or a request
MODEL_IDLE_SECONDS = 60.0  # unload Whisper after this long without a job
STALE_REQUEST_JUNK_SECONDS = 60.0  # non-.req leftovers in requests/ older than this
STUCK_REQUEST_SECONDS = 10.0  # at exit, a .req still there after this long is stuck
REJECTED_DIR = "rejected"  # <q>/rejected/: what could not be removed from requests/


# --------------------------------------------------------------------------- VLC liveness


@dataclass(frozen=True)
class IntfState:
    """What one queue dir's ``intf_state`` says about its VLC instance."""

    alive: bool  # VLC is running (fresh state, not stopped)
    stopped_at: float | None = None  # wall time of an explicit state=stopped
    age: float | None = None  # seconds since the state was written (None: no file)


def read_intf_state(queue_dir: Path, now: float, max_age: float = INTF_FRESH_SECONDS) -> IntfState:
    """Read ``<queue_dir>/intf_state`` at wall time ``now``."""
    path = Path(queue_dir) / P.INTF_STATE_FILE
    data = P.read_kv(path)
    if not data:
        return IntfState(alive=False)
    try:
        stamp: float | None = float(data.get("time", ""))
    except ValueError:
        stamp = None
    if stamp is None:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return IntfState(alive=False)
    age = now - stamp
    if data.get("state", "").strip().lower() == "stopped":
        return IntfState(alive=False, stopped_at=stamp, age=age)
    # a timestamp slightly in the future (clock adjustments) still counts as fresh
    return IntfState(alive=age <= max_age, age=age)


@dataclass(frozen=True)
class VlcActivity:
    alive: bool  # at least one VLC instance is running
    just_stopped: bool  # none alive, and one said state=stopped recently


def vlc_activity(
    queue_dirs: Iterable[Path],
    now: float,
    *,
    max_age: float = INTF_FRESH_SECONDS,
    recent: float = STARTUP_GRACE_SECONDS,
) -> VlcActivity:
    states = [read_intf_state(q, now, max_age) for q in queue_dirs]
    alive = any(s.alive for s in states)
    just_stopped = not alive and any(
        s.stopped_at is not None and s.age is not None and s.age <= recent for s in states
    )
    return VlcActivity(alive=alive, just_stopped=just_stopped)


class IdleExitPolicy:
    """Decides when an on-demand daemon should exit. Feed it with :meth:`update`."""

    def __init__(
        self,
        *,
        idle_grace: float = IDLE_GRACE_SECONDS,
        startup_grace: float = STARTUP_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.idle_grace = idle_grace
        self.startup_grace = startup_grace
        self.clock = clock
        self.started = clock()
        self.idle_since: float | None = None
        self.seen_activity = False  # saw a live VLC or a job

    def update(self, *, vlc_alive: bool, busy: bool, just_stopped: bool = False) -> bool:
        """Record the current situation; returns True when the daemon should exit."""
        now = self.clock()
        if vlc_alive or busy:
            self.seen_activity = True
            self.idle_since = None
            return False
        if self.idle_since is None:
            self.idle_since = now
        if now - self.idle_since < self.idle_grace:
            return False
        in_startup = now - self.started < self.startup_grace
        # Just started and neither VLC nor a request showed up yet: VLC may still be
        # starting. An explicit, recent state=stopped ends that wait.
        return not (in_startup and not self.seen_activity and not just_stopped)

    def reset(self) -> None:
        self.idle_since = None


def clean_request_junk(
    queue_dirs: Iterable[Path],
    now: float,
    max_age: float = STALE_REQUEST_JUNK_SECONDS,
    *,
    stuck_requests: bool = False,
    stuck_age: float = STUCK_REQUEST_SECONDS,
) -> int:
    """Empty ``requests/`` of anything that would start the daemon over and over.

    The service managers start the daemon while ``requests/`` is non-empty (systemd
    ``DirectoryNotEmpty=``, launchd ``QueueDirectories``), so nothing may linger there:

    * non-``.req`` entries (abandoned ``.tmp`` files, directories) older than
      ``max_age`` are deleted;
    * with ``stuck_requests`` (used when the daemon exits, right after a last poll),
      ``.req`` files older than ``stuck_age`` are ones the daemon could not read or
      delete; they are moved to ``<q>/rejected/``.

    Whatever cannot be deleted is moved to ``<q>/rejected/`` as well; if even that
    fails it is logged once. Returns the number of entries removed or moved.
    """
    done = 0
    for q in queue_dirs:
        q = Path(q)
        try:
            entries = list(os.scandir(q / P.REQUESTS_DIR))
        except OSError:
            continue
        for e in entries:
            is_req = e.name.endswith(P.REQUEST_SUFFIX)
            if is_req and not stuck_requests:
                continue
            try:
                age = now - e.stat(follow_symlinks=False).st_mtime
            except OSError:
                continue
            if age <= (stuck_age if is_req else max_age):
                continue
            if is_req:
                log.warning("request %s could not be processed; moving it aside", e.name)
            elif _delete_entry(e):
                done += 1
                continue
            if _move_aside(q, e):
                done += 1
    return done


def _delete_entry(e: os.DirEntry[str]) -> bool:
    try:
        if e.is_dir(follow_symlinks=False):
            shutil.rmtree(e.path)
        else:
            os.unlink(e.path)
        return True
    except OSError as exc:
        log.debug("cannot delete %s: %s", e.path, exc)
        return False


_unmovable_reported: set[str] = set()


def _move_aside(q: Path, e: os.DirEntry[str]) -> bool:
    """Move a ``requests/`` entry to ``<q>/rejected/`` so the dir becomes empty."""
    rejected = q / REJECTED_DIR
    try:
        rejected.mkdir(exist_ok=True)
        dest = rejected / e.name
        n = 1
        while dest.exists() or dest.is_symlink():
            dest = rejected / f"{e.name}.{n}"
            n += 1
        os.replace(e.path, dest)
        log.warning("moved %s to %s (could not be removed)", e.path, dest)
        return True
    except OSError as exc:
        if e.path not in _unmovable_reported:
            _unmovable_reported.add(e.path)
            log.warning(
                "cannot remove or move %s (%s): it keeps the helper being restarted "
                "until it is removed by hand",
                e.path,
                exc,
            )
        return False


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
