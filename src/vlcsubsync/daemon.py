"""``vlc-subsync serve``: watches the VLC queue dirs and runs sync jobs.

See DESIGN.md ("File protocol").  One worker thread runs jobs one at a time; the main
thread polls the ``requests/`` dirs, writes heartbeats and does housekeeping.  The job
runner and the subtitle-source resolver are injectable so the daemon can be tested
without the sync engine.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from . import __version__
from . import protocol as P
from .config import (
    DEFAULT_MODE,
    DEFAULT_SYNC_MODE,
    MODES,
    mode_rank,
    normalize_mode,
    normalize_sync_mode,
)

log = logging.getLogger("vlcsubsync.daemon")

APP_NAME = "vlc-subsync"
SUBTITLE_EXTS = (".srt", ".ass", ".ssa", ".vtt", ".sub")
OUTPUT_EXTS = (".srt", ".ass", ".ssa", ".vtt")
MAX_AGE_SECONDS = 7 * 24 * 3600
REQUEST_EXPIRY_SECONDS = 3600
ALREADY_RUNNING = 3

# --------------------------------------------------------------------------- user paths


def _platformdirs():
    import platformdirs

    return platformdirs


def user_cache_dir() -> Path:
    env = os.environ.get("VLC_SUBSYNC_CACHE_DIR")
    if env:
        return Path(env)
    return Path(_platformdirs().user_cache_dir(APP_NAME, appauthor=False))


def user_state_dir() -> Path:
    env = os.environ.get("VLC_SUBSYNC_STATE_DIR")
    if env:
        return Path(env)
    return Path(_platformdirs().user_state_dir(APP_NAME, appauthor=False))


def user_log_dir() -> Path:
    env = os.environ.get("VLC_SUBSYNC_LOG_DIR")
    if env:
        return Path(env)
    return Path(_platformdirs().user_log_dir(APP_NAME, appauthor=False))


def user_config_dir() -> Path:
    env = os.environ.get("VLC_SUBSYNC_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(_platformdirs().user_config_dir(APP_NAME, appauthor=False))


def lock_file_path() -> Path:
    return user_state_dir() / "daemon.lock"


def results_cache_dir() -> Path:
    return user_cache_dir() / "results"


def platform_key(platform: str | None = None) -> str:
    plat = (platform or sys.platform).lower()
    if plat.startswith(("win", "cygwin")):
        return "windows"
    if plat in ("darwin", "macos", "mac"):
        return "macos"
    return "linux"


def vlc_data_dirs(
    platform: str | None = None,
    home: Path | None = None,
    env: dict[str, str] | None = None,
) -> list[tuple[str, Path]]:
    """Candidate VLC *user data* dirs (``vlc.config.userdatadir()``) as (kind, path)."""
    env = dict(os.environ) if env is None else env
    home = Path(home) if home is not None else Path.home()
    plat = platform_key(platform)
    if plat == "windows":
        appdata = env.get("APPDATA") or str(home / "AppData" / "Roaming")
        return [("windows", Path(appdata) / "vlc")]
    if plat == "macos":
        return [("macos", home / "Library" / "Application Support" / "org.videolan.vlc")]
    xdg_data = env.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return [
        ("native", Path(xdg_data) / "vlc"),
        ("snap", home / "snap" / "vlc" / "current" / ".local" / "share" / "vlc"),
        ("flatpak", home / ".var" / "app" / "org.videolan.VLC" / "data" / "vlc"),
    ]


def default_queue_dirs(
    platform: str | None = None, home: Path | None = None, env: dict[str, str] | None = None
) -> list[Path]:
    """All candidate queue dirs for this platform (existing or not).

    ``VLC_SUBSYNC_QUEUE_DIRS`` (``os.pathsep``-separated) replaces the defaults.
    """
    env_map = dict(os.environ) if env is None else env
    override = env_map.get("VLC_SUBSYNC_QUEUE_DIRS")
    if override:
        return [Path(p) for p in override.split(os.pathsep) if p]
    return [d / "subsync" for _, d in vlc_data_dirs(platform, home, env_map)]


def ensure_queue_layout(queue_dir: Path) -> None:
    for sub in (P.REQUESTS_DIR, P.JOBS_DIR, P.OUT_DIR):
        (queue_dir / sub).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- logging


def setup_logging(
    *, to_stderr: bool = True, to_file: bool = True, level: int = logging.INFO
) -> Path | None:
    root = logging.getLogger("vlcsubsync")
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_vlcsubsync", False):
            root.removeHandler(h)
            h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_path: Path | None = None
    if to_file:
        try:
            log_dir = user_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "daemon.log"
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            fh._vlcsubsync = True  # type: ignore[attr-defined]
            root.addHandler(fh)
        except OSError:
            log_path = None
    if to_stderr and sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh._vlcsubsync = True  # type: ignore[attr-defined]
        root.addHandler(sh)
    return log_path


# --------------------------------------------------------------------------- process helpers


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class InstanceLock:
    """Single-instance lock: an OS advisory lock on a file that also stores our pid.

    The OS releases the lock when the process dies, so a leftover file from a crashed
    daemon is never a problem.  If the filesystem does not support locking we fall
    back to "pid in the file is alive".
    """

    _WIN_LOCK_OFFSET = 1 << 20

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh: Any = None

    @staticmethod
    def read_pid(path: Path) -> int | None:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        try:
            return int(text.splitlines()[0]) if text else None
        except ValueError:
            return None

    def holder_pid(self) -> int | None:
        return self.read_pid(self.path)

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115
        try:
            locked = self._os_lock(fh)
        except OSError as exc:  # locking unsupported: fall back to pid check
            log.debug("advisory locking unavailable (%s); using pid check", exc)
            pid = self.read_pid(self.path)
            locked = not (pid and pid != os.getpid() and pid_alive(pid))
        if not locked:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return True

    def _os_lock(self, fh: Any) -> bool:
        if os.name == "nt":
            import msvcrt

            fh.seek(self._WIN_LOCK_OFFSET)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except PermissionError:
                return False
            except OSError as exc:
                if getattr(exc, "errno", None) in (13, 36):  # EACCES / EDEADLOCK
                    return False
                raise
            return True
        import fcntl

        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as exc:
            import errno

            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return False
            raise
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.seek(0)
            fh.truncate()
        except OSError:
            pass
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(self._WIN_LOCK_OFFSET)
                with contextlib.suppress(OSError):
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                with contextlib.suppress(OSError):
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
        # The file is deliberately not unlinked: unlinking a lock file races with
        # another process that already opened it.

    def __enter__(self) -> InstanceLock:
        if not self.acquire():
            raise RuntimeError(f"another instance holds {self.path}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def daemon_running_pid(lock_path: Path | None = None) -> int | None:
    """PID of a running daemon (per the lock file), or ``None`` if none is running."""
    path = lock_path or lock_file_path()
    if not path.exists():
        return None
    pid = InstanceLock.read_pid(path)
    probe = InstanceLock(path)
    if probe.acquire():  # nobody holds it -> not running (stale file)
        probe.release()
        return None
    return pid


# --------------------------------------------------------------------------- jobs


class JobError(Exception):
    """A user-facing job failure (message goes to the status file)."""


class JobCancelled(Exception):
    """Raised from the progress callback when a job was superseded or the daemon stops."""


@dataclass
class ResolvedSource:
    kind: Literal["embedded", "external"]
    index: int | None = None
    path: str | None = None

    def identity(self) -> str:
        if self.kind == "embedded":
            return f"embedded:{self.index}"
        p = Path(self.path or "")
        try:
            st = p.stat()
            return f"external:{os.path.abspath(p)}:{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return f"external:{os.path.abspath(p)}"


@dataclass
class JobSpec:
    """What a job runner gets."""

    id: str
    media: str
    audio_index: int
    source: ResolvedSource
    output_path: str
    config: Any = None


class SyncResultLike(Protocol):
    output_path: str
    method: str
    offset: float
    scale: float
    confidence: float
    applied: bool
    message: str


ProgressFn = Callable[[float, str], None]
JobRunner = Callable[[JobSpec, ProgressFn], Any]
SourceResolver = Callable[[P.Request], ResolvedSource]


def count_embedded_subtitles(media_path: str) -> int:
    """Number of subtitle streams in the container (VLC lists these first).

    All subtitle streams count (text and bitmap), because VLC lists them all.
    """
    from .media import probe

    return len(probe(media_path).subtitles)


def default_find_sidecars(media_path: str) -> list[str]:
    from .subtitles import find_sidecars

    return [str(p) for p in find_sidecars(media_path)]


def resolve_source(
    media: str,
    sub_index: int | None,
    sub_path: str = "",
    *,
    count_embedded: Callable[[str], int] = count_embedded_subtitles,
    find_sidecars: Callable[[str], Sequence[Any]] = default_find_sidecars,
) -> ResolvedSource:
    """Map VLC's subtitle ordinal to a concrete source.

    VLC lists embedded subtitle streams first, then external (sidecar) files.  An
    explicit ``sub_path`` always wins.
    """
    if sub_path:
        if not os.path.isfile(sub_path):
            raise JobError(f"Subtitle file not found: {sub_path}")
        return ResolvedSource("external", path=sub_path)
    if sub_index is None:
        raise JobError("No subtitle track selected")
    try:
        n_embedded = int(count_embedded(media))
    except Exception as exc:  # noqa: BLE001
        raise JobError(f"Cannot read media: {exc}") from exc
    if sub_index < n_embedded:
        return ResolvedSource("embedded", index=sub_index)
    k = sub_index - n_embedded
    sidecars = [str(p) for p in find_sidecars(media)]
    if k < len(sidecars):
        return ResolvedSource("external", path=sidecars[k])
    raise JobError(
        f"Subtitle track {sub_index + 1} not found "
        f"({n_embedded} embedded, {len(sidecars)} external files)"
    )


def effective_mode(request: P.Request, config: Any) -> str:
    """Sync mode of a job: the request's ``mode=`` if valid, else the config's."""
    return (
        normalize_mode(request.mode)
        or normalize_mode(getattr(config, "mode", None))
        or DEFAULT_MODE
    )


def load_config() -> Any:
    """Load the user's Config (lazy import of the engine's config module)."""
    try:
        from . import config as config_mod
    except ImportError:
        return None
    for name in ("load_config", "load"):
        fn = getattr(config_mod, name, None)
        if callable(fn):
            return fn()
    cfg_cls = getattr(config_mod, "Config", None)
    if cfg_cls is not None:
        loader = getattr(cfg_cls, "load", None)
        return loader() if callable(loader) else cfg_cls()
    return None


class EngineRunner:
    """Default job runner: calls ``vlcsubsync.sync.sync_subtitles``.

    Whisper models stay warm between jobs through ``vlcsubsync.transcribe``'s
    module-level model cache (the daemon process is long-lived).
    """

    def __call__(self, spec: JobSpec, progress: ProgressFn) -> Any:
        from .sync import SubtitleSource, sync_subtitles

        config = spec.config
        if config is None:
            from .config import Config

            config = Config.load()
        source = SubtitleSource(
            kind=spec.source.kind, index=spec.source.index, path=spec.source.path
        )
        return sync_subtitles(
            spec.media, spec.audio_index, source, spec.output_path, config, progress=progress
        )


@dataclass
class _Job:
    request: P.Request
    queue_dir: Path
    received: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    cancel_reason: str = ""

    @property
    def media_key(self) -> tuple[str, str]:
        return (str(self.queue_dir), os.path.normcase(os.path.abspath(self.request.media)))

    @property
    def params(self) -> tuple[Any, ...]:
        r = self.request
        return (r.audio_index, r.sub_index, r.sub_path)


class _StatusWriter:
    """Throttled status-file writer for one job."""

    def __init__(self, queue_dir: Path, job_id: str, min_interval: float):
        self.queue_dir = queue_dir
        self.job_id = job_id
        self.min_interval = min_interval
        self._last_write = 0.0
        self._last_state = ""

    def write(self, status: P.Status, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and status.state == self._last_state
            and now - self._last_write < self.min_interval
        ):
            return
        try:
            P.write_status(self.queue_dir, status)
        except OSError as exc:
            log.warning("cannot write status for %s: %s", self.job_id, exc)
            return
        self._last_write = now
        self._last_state = status.state


def _short(msg: object, limit: int = 240) -> str:
    text = " ".join(str(msg).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- daemon


class Daemon:
    def __init__(
        self,
        queue_dirs: Iterable[str | os.PathLike[str]] = (),
        *,
        use_default_queues: bool = True,
        runner: JobRunner | None = None,
        resolver: SourceResolver | None = None,
        config_loader: Callable[[], Any] = load_config,
        cache_dir: Path | None = None,
        lock_path: Path | None = None,
        poll_interval: float = 0.4,
        heartbeat_interval: float = 2.0,
        status_interval: float = 0.5,
        housekeeping_interval: float = 3600.0,
        rescan_interval: float = 30.0,
        max_age: float = MAX_AGE_SECONDS,
        version: str = __version__,
    ):
        self.extra_queue_dirs = [Path(q) for q in queue_dirs]
        self.use_default_queues = use_default_queues
        self.runner: JobRunner = runner or EngineRunner()
        self.resolver: SourceResolver = resolver or (
            lambda r: resolve_source(r.media, r.sub_index, r.sub_path)
        )
        self.config_loader = config_loader
        self.cache_dir = Path(cache_dir) if cache_dir else results_cache_dir()
        self.lock = InstanceLock(lock_path or lock_file_path())
        self.poll_interval = poll_interval
        self.heartbeat_interval = min(heartbeat_interval, 2.0)
        self.status_interval = status_interval
        self.housekeeping_interval = housekeeping_interval
        self.rescan_interval = rescan_interval
        self.max_age = max_age
        self.version = version

        self.queue_dirs: list[Path] = []
        self.stop_event = threading.Event()
        self._cond = threading.Condition()
        self._pending: list[_Job] = []
        self._running: _Job | None = None
        self._seen_requests: dict[Path, float] = {}
        self._worker: threading.Thread | None = None
        self.jobs_completed = 0

    # ---------------------------------------------------------------- queue dirs
    def scan_queue_dirs(self) -> list[Path]:
        found: list[Path] = []
        candidates = list(self.extra_queue_dirs)
        if self.use_default_queues:
            candidates += default_queue_dirs()
        for q in candidates:
            # Explicit dirs are always used; default dirs when they (or VLC's data
            # dir containing them) exist.
            explicit = q in self.extra_queue_dirs
            if not (explicit or q.is_dir() or q.parent.is_dir()):
                continue
            try:
                ensure_queue_layout(q)
            except OSError as exc:
                log.warning("cannot use queue dir %s: %s", q, exc)
                continue
            if q not in found:
                found.append(q)
        new = [q for q in found if q not in self.queue_dirs]
        for q in new:
            log.info("watching queue dir %s", q)
        self.queue_dirs = found
        return found

    # ---------------------------------------------------------------- heartbeat
    def write_heartbeats(self) -> None:
        hb = P.Heartbeat(time=time.time(), pid=os.getpid(), version=self.version)
        for q in self.queue_dirs:
            try:
                P.write_heartbeat(q, hb)
            except OSError as exc:
                log.warning("heartbeat write failed for %s: %s", q, exc)

    def remove_heartbeats(self) -> None:
        for q in self.queue_dirs:
            hb = P.read_heartbeat(q)
            if hb is not None and hb.pid == os.getpid():
                with contextlib.suppress(OSError):
                    (q / P.HEARTBEAT_FILE).unlink()

    # ---------------------------------------------------------------- requests
    def poll_requests(self) -> int:
        """Pick up new ``.req`` files.  Returns the number of new jobs."""
        picked = 0
        for q in self.queue_dirs:
            req_dir = q / P.REQUESTS_DIR
            try:
                entries = sorted(
                    (e for e in os.scandir(req_dir) if e.name.endswith(P.REQUEST_SUFFIX)),
                    key=lambda e: (e.stat().st_mtime, e.name),
                )
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError as exc:
                log.debug("scan %s failed: %s", req_dir, exc)
                continue
            for entry in entries:
                path = Path(entry.path)
                if path in self._seen_requests:
                    self._try_unlink(path)  # deletion failed earlier (Windows lock)
                    continue
                if self._pick_up(q, path):
                    picked += 1
        # forget seen entries whose file is gone
        for path in [p for p in self._seen_requests if not p.exists()]:
            del self._seen_requests[path]
        return picked

    def _try_unlink(self, path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()

    def _pick_up(self, queue_dir: Path, path: Path) -> bool:
        stem = path.name[: -len(P.REQUEST_SUFFIX)]
        try:
            mtime = path.stat().st_mtime
            request = P.read_request(path)
        except FileNotFoundError:
            return False
        except P.ProtocolError as exc:
            log.warning("bad request %s: %s", path.name, exc)
            self._seen_requests[path] = time.time()
            self._try_unlink(path)
            if P.is_valid_id(stem):
                self._safe_status(
                    queue_dir, P.Status(id=stem, state="error", message=f"Bad request: {exc}")
                )
            return False
        except OSError as exc:  # e.g. sharing violation on Windows: retry next poll
            log.debug("cannot read %s yet: %s", path, exc)
            return False
        self._seen_requests[path] = time.time()
        self._try_unlink(path)
        if time.time() - mtime > REQUEST_EXPIRY_SECONDS:
            log.info("discarding expired request %s", request.id)
            self._safe_status(
                queue_dir, P.Status(id=request.id, state="error", message="Request expired")
            )
            return False
        log.info(
            "request %s: media=%s audio=%s sub=%s%s",
            request.id,
            request.media,
            request.audio_index,
            request.sub_index,
            f" sub_path={request.sub_path}" if request.sub_path else "",
        )
        self.enqueue(_Job(request=request, queue_dir=queue_dir))
        return True

    def enqueue(self, job: _Job) -> None:
        superseded: list[_Job] = []
        with self._cond:
            keep = []
            for other in self._pending:
                if other.media_key == job.media_key:
                    superseded.append(other)
                else:
                    keep.append(other)
            self._pending = keep
            running = self._running
            if (
                running is not None
                and running.media_key == job.media_key
                and running.params != job.params
            ):
                running.cancel_reason = "Superseded by a newer request"
                running.cancel.set()
            position = len(self._pending) + (1 if self._running else 0)
            msg = "Queued" if position == 0 else f"Queued ({position} ahead)"
            # Written before the worker can see the job so "queued" never
            # overwrites "running".
            self._safe_status(
                job.queue_dir, P.Status(id=job.request.id, state="queued", message=msg)
            )
            self._pending.append(job)
            self._cond.notify_all()
        for old in superseded:
            log.info("request %s superseded by %s", old.request.id, job.request.id)
            self._safe_status(
                old.queue_dir,
                P.Status(id=old.request.id, state="error", message="Superseded by a newer request"),
            )

    @staticmethod
    def _safe_status(queue_dir: Path, status: P.Status) -> None:
        try:
            P.write_status(queue_dir, status)
        except OSError as exc:
            log.warning("cannot write status %s: %s", status.id, exc)

    # ---------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self.stop_event.is_set():
                    self._cond.wait(timeout=1.0)
                if self.stop_event.is_set():
                    return
                job = self._pending.pop(0)
                self._running = job
            try:
                self.run_job(job)
            except Exception:  # noqa: BLE001 - never let a job kill the worker
                log.exception("unexpected failure in job %s", job.request.id)
            finally:
                with self._cond:
                    self._running = None
                self.jobs_completed += 1

    def cache_key(
        self, job: _Job, source: ResolvedSource, config: Any, mode: str | None = None
    ) -> str:
        """Result-cache key; ``mode`` defaults to the job's effective sync mode."""
        r = job.request
        media = os.path.abspath(r.media)
        st = os.stat(media)
        parts = [
            media,
            str(st.st_size),
            str(st.st_mtime_ns),
            str(r.audio_index),
            source.identity(),
            str(getattr(config, "model_en", "")),
            str(getattr(config, "model_multi", "")),
            self.version,
            mode or effective_mode(r, config),
        ]
        return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()

    def cache_lookup_keys(
        self, job: _Job, source: ResolvedSource, config: Any
    ) -> list[tuple[str, str]]:
        """``(mode, key)`` pairs whose results may satisfy this job: the most thorough
        mode first, down to the job's own mode. A result of a more thorough mode is at
        least as good, so an exhaustive result also answers a later fast/thorough
        request (never the other way round), but only if it was applied (see
        :meth:`_cached_result`)."""
        want = mode_rank(effective_mode(job.request, config))
        return [
            (m, self.cache_key(job, source, config, m))
            for m in reversed(MODES)
            if mode_rank(m) >= want
        ]

    def _cached_result(
        self, job: _Job, source: ResolvedSource, config: Any
    ) -> tuple[Path, dict[str, str]] | None:
        """Cached result for ``job``, or None.

        The job's own mode may answer with any cached result, including an unapplied
        one (no point redoing hopeless work in the same mode). Another, more thorough
        mode only answers with an *applied* result: an unapplied exhaustive run (e.g.
        windows lost to CUDA OOM, then the VAD fallback) must not block a fast sync
        that might succeed.
        """
        own = effective_mode(job.request, config)
        for m, key in self.cache_lookup_keys(job, source, config):
            hit = self._cache_lookup(key)
            if hit is None:
                continue
            if m != own and not P._to_bool(hit[1].get("applied")):
                log.debug("ignoring unapplied cached %s result for a %s job", m, own)
                continue
            if P._to_bool(hit[1].get("applied")) and not hit[1].get("segments"):
                # cached before results carried their mapping: delay mode cannot
                # use it, so re-sync once (the new result replaces it)
                log.info("cached %s result has no mapping; re-syncing", m)
                continue
            return hit
        return None

    def _cache_drop_other_modes(
        self, job: _Job, source: ResolvedSource, config: Any, keep: str
    ) -> None:
        """Remove the cached results of every mode but ``keep`` for this file and
        tracks, so the newest (forced) result is what later lookups find."""
        for m in MODES:
            if m == keep:
                continue
            key = self.cache_key(job, source, config, m)
            meta_path = self.cache_dir / f"{key}.meta"
            meta = P.read_kv(meta_path)
            paths = [meta_path]
            if meta and meta.get("file"):
                paths.append(self.cache_dir / meta["file"])
            for p in paths:
                try:
                    p.unlink()
                    log.debug("dropped cached %s result %s", m, p.name)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    log.warning("cannot drop cached result %s: %s", p, exc)

    def _cache_lookup(self, key: str) -> tuple[Path, dict[str, str]] | None:
        meta = P.read_kv(self.cache_dir / f"{key}.meta")
        if not meta:
            return None
        out = self.cache_dir / meta.get("file", "")
        if not meta.get("file") or not out.is_file():
            return None
        now = time.time()
        for p in (out, self.cache_dir / f"{key}.meta"):
            with contextlib.suppress(OSError):
                os.utime(p, (now, now))
        return out, meta

    def _cache_store(self, key: str, output: Path, status: P.Status) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            name = f"{key}{output.suffix.lower() or '.srt'}"
            tmp = self.cache_dir / f".{name}.tmp"
            shutil.copyfile(output, tmp)
            os.replace(tmp, self.cache_dir / name)
            meta = {k: v for k, v in status.to_dict().items() if k not in ("id", "output", "time")}
            meta["file"] = name
            P.write_kv(self.cache_dir / f"{key}.meta", meta)
        except OSError as exc:
            log.warning("cannot store result in cache: %s", exc)

    @staticmethod
    def _output_ext(source: ResolvedSource) -> str:
        if source.kind == "external" and source.path:
            ext = Path(source.path).suffix.lower()
            if ext in OUTPUT_EXTS:
                return ext
        return ".srt"

    def run_job(self, job: _Job) -> P.Status:
        r = job.request
        writer = _StatusWriter(job.queue_dir, r.id, self.status_interval)
        status = P.Status(id=r.id, state="running", progress=0.0, message="Preparing")
        writer.write(status, force=True)
        started = time.monotonic()
        try:
            if job.cancel.is_set():
                raise JobCancelled(job.cancel_reason or "Cancelled")
            if not os.path.isfile(r.media):
                raise JobError(f"Media file not found: {r.media}")
            source = self.resolver(r)
            config = self.config_loader()
            mode = effective_mode(r, config)
            if r.mode and hasattr(config, "with_mode"):
                config = config.with_mode(r.mode)
            key = self.cache_key(job, source, config, mode)
            out_dir = job.queue_dir / P.OUT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)

            cached = None if r.force else self._cached_result(job, source, config)
            if cached is not None:
                cached_file, meta = cached
                dest = out_dir / f"{r.id}{cached_file.suffix}"
                shutil.copyfile(cached_file, dest)
                done = P.Status.from_dict(meta)
                done.id = r.id
                done.state = "done"
                done.progress = 1.0
                done.output = str(dest)
                done.time = None
                done.sync_mode = _sync_mode(config)  # the current setting, not the cached one
                log.info("job %s: cache hit (mode %s)", r.id, mode)
                writer.write(done, force=True)
                return done

            spec = JobSpec(
                id=r.id,
                media=r.media,
                audio_index=r.audio_index,
                source=source,
                output_path=str(out_dir / f"{r.id}{self._output_ext(source)}"),
                config=config,
            )

            def progress(frac: float, message: str = "") -> None:
                if job.cancel.is_set() or self.stop_event.is_set():
                    raise JobCancelled(job.cancel_reason or "Daemon stopping")
                try:
                    frac = float(frac)
                except (TypeError, ValueError):
                    frac = 0.0
                writer.write(
                    P.Status(
                        id=r.id,
                        state="running",
                        progress=max(0.0, min(0.99, frac)),
                        message=_short(message) or "Working",
                    )
                )

            result = self.runner(spec, progress)
            if job.cancel.is_set():
                raise JobCancelled(job.cancel_reason or "Cancelled")
            output = Path(getattr(result, "output_path", "") or spec.output_path)
            if not output.is_file():
                raise JobError("Sync produced no output file")
            done = P.Status(
                id=r.id,
                state="done",
                progress=1.0,
                message=_short(getattr(result, "message", "") or "Done"),
                output=str(output),
                applied=bool(getattr(result, "applied", True)),
                method=str(getattr(result, "method", "") or "") or None,
                offset=_float_or_none(getattr(result, "offset", None)),
                scale=_float_or_none(getattr(result, "scale", None)),
                confidence=_float_or_none(getattr(result, "confidence", None)),
                segments=_segments_or_none(result),
                sync_mode=_sync_mode(config),
            )
            writer.write(done, force=True)
            self._cache_store(key, output, done)
            if r.force:
                # the user asked for a fresh result: it must not be shadowed by an
                # older result of another (more thorough) mode on the next open
                self._cache_drop_other_modes(job, source, config, mode)
            log.info(
                "job %s done in %.1fs (mode %s): %s (applied=%s)",
                r.id,
                time.monotonic() - started,
                mode,
                done.message,
                done.applied,
            )
            return done
        except JobCancelled as exc:
            log.info("job %s cancelled: %s", r.id, exc)
            status = P.Status(id=r.id, state="error", message=_short(exc) or "Cancelled")
        except JobError as exc:
            log.warning("job %s failed: %s", r.id, exc)
            status = P.Status(id=r.id, state="error", message=_short(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s crashed", r.id)
            status = P.Status(
                id=r.id, state="error", message=_short(f"{type(exc).__name__}: {exc}")
            )
        writer.write(status, force=True)
        return status

    # ---------------------------------------------------------------- housekeeping
    def housekeeping(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        removed = 0
        dirs: list[tuple[Path, float]] = []
        for q in self.queue_dirs:
            dirs += [(q / P.JOBS_DIR, self.max_age), (q / P.OUT_DIR, self.max_age)]
            dirs += [(q / P.REQUESTS_DIR, 3600.0)]
        dirs.append((self.cache_dir, self.max_age))
        active_ids = set()
        with self._cond:
            for j in [*self._pending, *([self._running] if self._running else [])]:
                active_ids.add(j.request.id)
        for d, max_age in dirs:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                try:
                    if not e.is_file():
                        continue
                    age_limit = 3600.0 if e.name.endswith(P.TMP_SUFFIX) else max_age
                    if d.name == P.REQUESTS_DIR and e.name.endswith(P.REQUEST_SUFFIX):
                        continue  # handled by poll_requests
                    if e.name.split(".")[0] in active_ids:
                        continue
                    if now - e.stat().st_mtime > age_limit:
                        os.unlink(e.path)
                        removed += 1
                except OSError:
                    continue
        if removed:
            log.info("housekeeping removed %d old files", removed)
        return removed

    # ---------------------------------------------------------------- main loop
    def start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, name="subsync-worker")
            self._worker.daemon = True
            self._worker.start()

    def request_stop(self, *_: object) -> None:
        self.stop_event.set()
        with self._cond:
            if self._running is not None:
                self._running.cancel_reason = "Daemon stopping"
                self._running.cancel.set()
            self._cond.notify_all()

    def run(self, *, acquire_lock: bool = True) -> int:
        if acquire_lock and not self.lock.acquire():
            pid = self.lock.holder_pid()
            log.error("another vlc-subsync daemon is already running (pid %s)", pid)
            return ALREADY_RUNNING
        try:
            self.scan_queue_dirs()
            if not self.queue_dirs:
                log.warning("no VLC queue dirs found yet; will keep looking")
            log.info("vlc-subsync daemon %s started (pid %d)", self.version, os.getpid())
            self.start_worker()
            last_hb = last_scan = 0.0
            last_hk = time.monotonic() - self.housekeeping_interval + 5.0
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.use_default_queues and now - last_scan >= self.rescan_interval:
                    self.scan_queue_dirs()
                    last_scan = now
                if now - last_hb >= self.heartbeat_interval:
                    self.write_heartbeats()
                    last_hb = now
                try:
                    self.poll_requests()
                except Exception:  # noqa: BLE001
                    log.exception("error while polling requests")
                if now - last_hk >= self.housekeeping_interval:
                    try:
                        self.housekeeping()
                    except Exception:  # noqa: BLE001
                        log.exception("housekeeping failed")
                    last_hk = now
                self.stop_event.wait(self.poll_interval)
        finally:
            self._shutdown()
            if acquire_lock:
                self.lock.release()
        log.info("vlc-subsync daemon stopped")
        return 0

    def _shutdown(self) -> None:
        self.request_stop()
        if self._worker is not None:
            self._worker.join(timeout=10.0)
        with self._cond:
            pending, self._pending = self._pending, []
        for job in pending:
            with contextlib.suppress(OSError):
                P.write_status(
                    job.queue_dir,
                    P.Status(id=job.request.id, state="error", message="Daemon stopped"),
                )
        self.remove_heartbeats()


def _sync_mode(config: Any) -> str:
    """The configured sync_mode (``track`` unless the config says ``delay``)."""
    return normalize_sync_mode(getattr(config, "sync_mode", None)) or DEFAULT_SYNC_MODE


def _segments_or_none(result: Any) -> list[P.MapSegment] | None:
    """The result's mapping (``SyncResult.mapping_segments``) when it was applied."""
    if not getattr(result, "applied", True):
        return None
    segs = getattr(result, "mapping_segments", None)
    if not segs:
        return None
    return [s for s in segs if isinstance(s, P.MapSegment)] or None


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def install_signal_handlers(daemon: Daemon) -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGTERM", "SIGINT", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sig, daemon.request_stop)


def serve(
    queue_dirs: Sequence[str] = (),
    *,
    use_default_queues: bool = True,
    log_to_stderr: bool = True,
    verbose: bool = False,
) -> int:
    """Entry point for ``vlc-subsync serve``."""
    setup_logging(to_stderr=log_to_stderr, level=logging.DEBUG if verbose else logging.INFO)
    daemon = Daemon(queue_dirs, use_default_queues=use_default_queues)
    install_signal_handlers(daemon)
    try:
        rc = daemon.run()
    except KeyboardInterrupt:
        return 0
    # "Already running" exits 0 so service managers don't restart-loop.
    return 0 if rc == ALREADY_RUNNING else rc
