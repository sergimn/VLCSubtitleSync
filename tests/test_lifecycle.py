"""Daemon process helpers: Whisper thread count and process priority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlcsubsync import daemon as D
from vlcsubsync import lifecycle as L
from vlcsubsync import transcribe as T


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


@pytest.mark.parametrize(
    "kernel,pointer,nr", [("x86_64", 8, 251), ("x86_64", 4, 289), ("aarch64", 4, 314)]
)
def test_priority_ioprio_uses_the_process_abi(allow_priority, monkeypatch, kernel, pointer, nr):
    """32-bit Python on a 64-bit kernel must use the 32-bit syscall number."""
    import platform
    import struct

    libc = FakeLibc()
    monkeypatch.setattr(platform, "machine", lambda: kernel)
    monkeypatch.setattr(struct, "calcsize", lambda fmt: pointer)
    L._linux_idle_io(FakeCtypes(libc), None)
    assert libc.calls and libc.calls[0][0] == nr
