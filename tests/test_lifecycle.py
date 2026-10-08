"""Daemon process helpers: model unloading (fake clock), thread count, priority."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlcsubsync import daemon as D
from vlcsubsync import lifecycle as L
from vlcsubsync import protocol as P
from vlcsubsync import transcribe as T

# ------------------------------------------------------------------ model unloading


class FakeClock:
    """Monotonic and wall clocks that only move when told to."""

    def __init__(self, wall: float = 1_800_000_000.0):
        self.mono = 1000.0
        self.wall = wall

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def queue(tmp_path):
    q = tmp_path / "vlc" / "subsync"
    D.ensure_queue_layout(q)
    return q


@pytest.fixture
def denv(tmp_path, monkeypatch):
    for name in ("CACHE", "STATE", "LOG"):
        monkeypatch.setenv(f"VLC_SUBSYNC_{name}_DIR", str(tmp_path / name.lower()))
    media = tmp_path / "movie.mkv"
    media.write_bytes(b"media")
    return SimpleNamespace(tmp=tmp_path, media=str(media))


class GatedRunner:
    """Job runner that blocks until released (a job "in flight")."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, spec, progress):
        self.started.set()
        assert self.release.wait(10)
        Path(spec.output_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
        return SimpleNamespace(output_path=spec.output_path, applied=True, message="ok")


def make_daemon(denv, queue, clock, runner=None, **kw):
    return D.Daemon(
        [queue],
        use_default_queues=False,
        runner=runner or GatedRunner(),
        resolver=lambda r: D.ResolvedSource("embedded", index=0),
        config_loader=lambda: SimpleNamespace(model_en="base.en", model_multi="base"),
        cache_dir=denv.tmp / "cache",
        lock_path=denv.tmp / "state" / "daemon.lock",
        clock=clock.monotonic,
        **kw,
    )


def test_model_unload_timer(denv, queue, clock):
    unloads = []

    class Runner:
        def __call__(self, spec, progress):
            Path(spec.output_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
            return SimpleNamespace(output_path=spec.output_path, applied=True, message="ok")

    d = make_daemon(
        denv,
        queue,
        clock,
        runner=Runner(),
        model_unloader=lambda: unloads.append(1) or 2,
        worker_wait=0.01,
    )
    assert not d.model_unload_due()  # nothing loaded yet
    d.scan_queue_dirs()
    d.start_worker()
    try:
        P.write_request(queue, P.Request(id="r1", media=denv.media))
        d.poll_requests()
        deadline = time.monotonic() + 5
        while d.jobs_completed < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert d.jobs_completed == 1
        clock.advance(59)
        time.sleep(0.1)
        assert unloads == []
        clock.advance(2)
        deadline = time.monotonic() + 5
        while not unloads and time.monotonic() < deadline:
            time.sleep(0.01)
        assert unloads == [1]
        time.sleep(0.1)
        assert unloads == [1]  # once, not on every idle tick
        assert not d.model_unload_due()
    finally:
        d.request_stop()
        d._worker.join(5)


def test_model_unload_never_runs_during_a_job(denv, queue, clock):
    runner = GatedRunner()
    unloads = []
    d = make_daemon(
        denv,
        queue,
        clock,
        runner=runner,
        model_unloader=lambda: unloads.append(1) or 0,
        worker_wait=0.01,
    )
    d.scan_queue_dirs()
    d.start_worker()
    try:
        P.write_request(queue, P.Request(id="r1", media=denv.media))
        d.poll_requests()
        assert runner.started.wait(5)
        clock.advance(3600)
        time.sleep(0.1)
        assert unloads == []  # the worker is busy with the job
        runner.release.set()
        deadline = time.monotonic() + 5
        while d.jobs_completed < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        assert unloads == []  # the job just ended: the timer restarts
        clock.advance(61)
        deadline = time.monotonic() + 5
        while not unloads and time.monotonic() < deadline:
            time.sleep(0.01)
        assert unloads == [1]
    finally:
        runner.release.set()
        d.request_stop()
        d._worker.join(5)


def test_unload_models_logs(denv, queue, clock, caplog):
    d = make_daemon(denv, queue, clock, model_unloader=lambda: 2)
    d._models_loaded = True
    with caplog.at_level("INFO", logger="vlcsubsync.daemon"):
        assert d.unload_models() == 2
    assert any("unloaded 2 Whisper model(s)" in r.getMessage() for r in caplog.records)


def test_failed_unload_is_retried(denv, queue, clock):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("busy")
        return 1

    d = make_daemon(denv, queue, clock, model_unloader=flaky)
    d._models_loaded = True
    clock.advance(61)
    assert d.model_unload_due()
    assert d.unload_models() == 0  # failed
    assert not d.model_unload_due()  # retried after another model_idle, not at once
    clock.advance(61)
    assert d.model_unload_due()
    assert d.unload_models() == 1
    assert not d.model_unload_due() and d.models_unloaded == 1


def test_clear_cache_releases_loaded_models():
    T.clear_cache()
    cfg = SimpleNamespace(device="cpu", compute_type="int8", threads=0, extra={})
    t = T.get_model(cfg, "tiny.en")
    t._model = object()
    t.device = "cpu"
    assert L.unload_models() == 1
    assert t._model is None and t.device is None
    assert T.get_model(cfg, "tiny.en") is not t  # reloads lazily next time
    T.clear_cache()


# ------------------------------------------------------------------ thread count


def test_default_threads(monkeypatch):
    for cpus, want in ((None, 1), (1, 1), (2, 1), (4, 2), (12, 6), (20, 8), (64, 8)):
        monkeypatch.setattr(T.os, "cpu_count", lambda c=cpus: c)
        assert T.default_threads() == want
    monkeypatch.setattr(T.os, "cpu_count", lambda: 4)
    assert T.WhisperTranscriber("tiny", threads=3).threads == 3  # config overrides


# ------------------------------------------------------------------ priority


class FakeOs:
    PRIO_PROCESS = 0
    SCHED_BATCH = 3

    def __init__(self, nice=0, fail=()):
        self.value = nice
        self.fail = set(fail)
        self.calls = []

    def getpriority(self, which, who):
        if "getpriority" in self.fail:
            raise OSError("nope")
        return self.value

    def nice(self, inc):
        if "nice" in self.fail:
            raise PermissionError("nope")
        self.calls.append(("nice", inc))
        self.value += inc
        return self.value

    def sched_param(self, prio):
        return prio

    def sched_setscheduler(self, pid, policy, param):
        if "sched" in self.fail:
            raise OSError("nope")
        self.calls.append(("sched", policy))


class FakeLibc:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls = []

    def syscall(self, *args):
        self.calls.append(args)
        return self.rc


class FakeCtypes:
    def __init__(self, libc=None, kernel32=None):
        self.libc = libc
        self.windll = SimpleNamespace(kernel32=kernel32)

    def CDLL(self, name, use_errno=False):  # noqa: N802
        if self.libc is None:
            raise OSError("no libc")
        return self.libc

    def get_errno(self):
        return 1


@pytest.fixture
def allow_priority(monkeypatch):
    monkeypatch.delenv("VLC_SUBSYNC_PRIORITY", raising=False)


def test_priority_linux(allow_priority):
    fos, libc = FakeOs(), FakeLibc()
    applied = L.lower_priority(
        platform="linux", os_mod=fos, ctypes_mod=FakeCtypes(libc), machine="x86_64"
    )
    assert applied == ["nice 10", "SCHED_BATCH", "I/O class idle"]
    assert ("nice", 10) in fos.calls and ("sched", 3) in fos.calls
    assert libc.calls == [(251, 1, 0, 3 << 13)]


def test_priority_keeps_higher_nice_and_survives_failures(allow_priority):
    fos = FakeOs(nice=15, fail={"sched"})
    applied = L.lower_priority(
        platform="linux", os_mod=fos, ctypes_mod=FakeCtypes(None), machine="aarch64"
    )
    assert applied == ["nice 15 (unchanged)"]
    assert fos.calls == []
    # unknown arch: no ioprio syscall; nice failing is not fatal
    fos = FakeOs(fail={"nice", "sched"})
    libc = FakeLibc()
    assert (
        L.lower_priority(platform="linux", os_mod=fos, ctypes_mod=FakeCtypes(libc), machine="mips")
        == []
    )
    assert libc.calls == []


def test_priority_macos_only_nice(allow_priority):
    fos = FakeOs(nice=0)
    assert L.lower_priority(platform="darwin", os_mod=fos) == ["nice 10"]
    assert fos.calls == [("nice", 10)]


def test_priority_windows(allow_priority):
    calls = []
    kernel32 = SimpleNamespace(
        GetCurrentProcess=lambda: "h",
        SetPriorityClass=lambda h, cls: calls.append((h, cls)) or 1,
    )
    applied = L.lower_priority(platform="win32", ctypes_mod=FakeCtypes(kernel32=kernel32))
    assert applied == ["BELOW_NORMAL_PRIORITY_CLASS"]
    assert calls == [("h", 0x4000)]
    # no windll (e.g. not really Windows): never raises
    assert L.lower_priority(platform="win32", ctypes_mod=SimpleNamespace()) == []


def test_priority_never_raises(allow_priority):
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("unexpected")

    assert L.lower_priority(platform="linux", os_mod=Boom(), ctypes_mod=Boom()) == []


def test_priority_opt_out(monkeypatch):
    monkeypatch.setenv("VLC_SUBSYNC_PRIORITY", "normal")
    fos = FakeOs()
    assert L.lower_priority(platform="linux", os_mod=fos) == []
    assert fos.calls == []


def test_serve_lowers_priority_before_running(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(L, "lower_priority", lambda: order.append("priority") or ["nice 10"])

    class FakeDaemon:
        def __init__(self, queue_dirs, use_default_queues):
            order.append("daemon")

        def run(self):
            order.append("run")
            return 0

        def request_stop(self, *a):
            pass

    monkeypatch.setattr(D, "Daemon", FakeDaemon)
    monkeypatch.setattr(D, "setup_logging", lambda **kw: None)
    assert D.serve([str(tmp_path)], log_to_stderr=False) == 0
    assert order == ["priority", "daemon", "run"]


def test_unload_models_releases_the_cached_silero_model(monkeypatch):
    """vad._model and faster-whisper's own lru_cache both hold the Silero session."""
    import functools
    import sys
    import types
    import weakref

    from vlcsubsync import vad

    class Session:
        pass

    @functools.lru_cache
    def get_vad_model():
        return Session()

    fake = types.ModuleType("faster_whisper.vad")
    fake.get_vad_model = get_vad_model
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", fake)
    monkeypatch.setattr(vad, "_model", get_vad_model())
    ref = weakref.ref(vad._model)
    L.unload_models()
    assert vad._model is None
    assert get_vad_model.cache_info().currsize == 0
    assert ref() is None
