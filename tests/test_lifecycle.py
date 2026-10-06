"""Helper lifecycle: idle exit, model unloading and process priority (fake clocks)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlcsubsync import daemon as D
from vlcsubsync import lifecycle as L
from vlcsubsync import protocol as P
from vlcsubsync import transcribe as T


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


def intf(q: Path, clock: FakeClock, state: str = "idle", age: float = 0.0) -> None:
    P.write_kv(q / P.INTF_STATE_FILE, {"time": int(clock.wall - age), "state": state})


# ------------------------------------------------------------------ intf_state


def test_read_intf_state_fresh_stale_stopped(queue, clock):
    assert L.read_intf_state(queue, clock.wall) == L.IntfState(alive=False)
    intf(queue, clock, age=5)
    assert L.read_intf_state(queue, clock.wall).alive
    intf(queue, clock, age=21)
    s = L.read_intf_state(queue, clock.wall)
    assert not s.alive and s.stopped_at is None and s.age == pytest.approx(21)
    intf(queue, clock, state="stopped", age=1)
    s = L.read_intf_state(queue, clock.wall)
    assert not s.alive and s.stopped_at == int(clock.wall - 1)
    # missing time: the file's mtime is used
    P.write_kv(queue / P.INTF_STATE_FILE, {"state": "idle"})
    assert L.read_intf_state(queue, time.time()).alive


def test_vlc_activity_any_queue(tmp_path, clock):
    q1, q2 = tmp_path / "a", tmp_path / "b"
    for q in (q1, q2):
        D.ensure_queue_layout(q)
    intf(q1, clock, state="stopped")
    intf(q2, clock, age=3)
    a = L.vlc_activity([q1, q2], clock.wall)
    assert a.alive and not a.just_stopped
    intf(q2, clock, age=300)
    a = L.vlc_activity([q1, q2], clock.wall)
    assert not a.alive and a.just_stopped


# ------------------------------------------------------------------ policy


def test_policy_exits_after_grace_when_vlc_gone(clock):
    pol = L.IdleExitPolicy(clock=clock.monotonic)
    assert not pol.update(vlc_alive=True, busy=False)
    clock.advance(100)
    assert not pol.update(vlc_alive=False, busy=False)  # idle timer starts
    clock.advance(14)
    assert not pol.update(vlc_alive=False, busy=False)
    clock.advance(1.5)
    assert pol.update(vlc_alive=False, busy=False)


def test_policy_busy_keeps_running_and_restarts_grace(clock):
    pol = L.IdleExitPolicy(clock=clock.monotonic)
    pol.update(vlc_alive=True, busy=False)
    clock.advance(100)
    pol.update(vlc_alive=False, busy=False)
    clock.advance(10)
    assert not pol.update(vlc_alive=False, busy=True)  # job in flight
    clock.advance(600)
    assert not pol.update(vlc_alive=False, busy=True)
    assert not pol.update(vlc_alive=False, busy=False)  # finished: grace restarts
    clock.advance(14)
    assert not pol.update(vlc_alive=False, busy=False)
    clock.advance(2)
    assert pol.update(vlc_alive=False, busy=False)


def test_policy_startup_grace(clock):
    pol = L.IdleExitPolicy(clock=clock.monotonic)
    for _ in range(59):
        assert not pol.update(vlc_alive=False, busy=False)
        clock.advance(1)
    clock.advance(1.5)
    assert pol.update(vlc_alive=False, busy=False)


def test_policy_recent_stop_ends_startup_grace(clock):
    pol = L.IdleExitPolicy(clock=clock.monotonic)
    assert not pol.update(vlc_alive=False, busy=False, just_stopped=True)
    clock.advance(15.5)
    assert pol.update(vlc_alive=False, busy=False, just_stopped=True)


def test_clean_request_junk(queue, clock):
    req = queue / "requests"
    (req / "a.req").write_text("x")
    old_tmp = req / "b.req.tmp"
    old_tmp.write_text("x")
    os.utime(old_tmp, (clock.wall - 120, clock.wall - 120))
    (req / "c.req.tmp").write_text("x")  # fresh: may be in the middle of a rename
    os.utime(req / "c.req.tmp", (clock.wall, clock.wall))
    assert L.clean_request_junk([queue], clock.wall) == 1
    assert sorted(p.name for p in req.iterdir()) == ["a.req", "c.req.tmp"]


# ------------------------------------------------------------------ daemon


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
        wall_clock=clock.time,
        **kw,
    )


def test_daemon_idle_exit_fresh_stale_stopped(denv, queue, clock):
    d = make_daemon(denv, queue, clock, idle_exit=True)
    d.scan_queue_dirs()
    intf(queue, clock)
    assert not d.check_idle_exit()
    clock.advance(30)  # VLC wrote nothing for 30 s: crashed or killed
    assert not d.check_idle_exit()  # grace starts
    clock.advance(16)
    assert d.check_idle_exit()

    d = make_daemon(denv, queue, clock, idle_exit=True)
    d.scan_queue_dirs()
    intf(queue, clock)
    assert not d.check_idle_exit()
    clock.advance(5)
    intf(queue, clock, state="stopped")  # VLC closed
    assert not d.check_idle_exit()
    clock.advance(15.5)
    assert d.check_idle_exit()


def test_daemon_startup_grace_then_exit(denv, queue, clock):
    d = make_daemon(denv, queue, clock, idle_exit=True)
    d.scan_queue_dirs()
    for _ in range(12):
        assert not d.check_idle_exit()
        clock.advance(5)
    clock.advance(1)
    assert d.check_idle_exit()


def test_daemon_job_in_flight_blocks_exit(denv, queue, clock):
    runner = GatedRunner()
    d = make_daemon(denv, queue, clock, runner=runner, idle_exit=True)
    d.scan_queue_dirs()
    d.start_worker()
    try:
        intf(queue, clock, state="stopped")
        P.write_request(queue, P.Request(id="r1", media=denv.media))
        assert d.poll_requests() == 1
        assert runner.started.wait(5)
        for _ in range(10):
            clock.advance(30)
            assert not d.check_idle_exit()
        runner.release.set()
        deadline = time.monotonic() + 5
        while d.busy() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not d.busy()
        assert not d.check_idle_exit()
        clock.advance(16)
        assert d.check_idle_exit()
    finally:
        runner.release.set()
        d.request_stop()
        d._worker.join(5)


def test_daemon_run_exits_by_itself(denv, queue, clock):
    d = make_daemon(denv, queue, clock, idle_exit=True, poll_interval=0.01, idle_check_interval=0.0)
    intf(queue, clock, state="stopped")
    t = threading.Thread(target=d.run)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while not (queue / P.HEARTBEAT_FILE).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        assert t.is_alive()
        clock.advance(16)
        t.join(5)
        assert not t.is_alive()
        assert "VLC is not running" in d.exit_reason
        assert not (queue / P.HEARTBEAT_FILE).exists()
    finally:
        d.request_stop()
        t.join(5)


def test_persistent_daemon_never_idle_exits(denv, queue, clock):
    d = make_daemon(denv, queue, clock, poll_interval=0.01)
    assert d.idle_exit is False
    t = threading.Thread(target=d.run)
    t.start()
    time.sleep(0.1)
    clock.advance(3600)
    time.sleep(0.1)
    assert t.is_alive()
    d.request_stop()
    t.join(5)


# ------------------------------------------------------------------ model unloading


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
        def __init__(self, queue_dirs, use_default_queues, idle_exit):
            order.append(("daemon", idle_exit))

        def run(self):
            order.append("run")
            return 0

        def request_stop(self, *a):
            pass

    monkeypatch.setattr(D, "Daemon", FakeDaemon)
    monkeypatch.setattr(D, "setup_logging", lambda **kw: None)
    assert D.serve([str(tmp_path)], log_to_stderr=False) == 0
    assert order == ["priority", ("daemon", True), "run"]
    order.clear()
    D.serve([str(tmp_path)], log_to_stderr=False, persistent=True)
    assert ("daemon", False) in order
