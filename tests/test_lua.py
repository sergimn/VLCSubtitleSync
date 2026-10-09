"""Run the VLC Lua scripts (intf + extension) under lupa with a mocked ``vlc`` API.

The mock lives in ``tests/lua/vlc_mock.lua``; the daemon side of the file protocol is
simulated from Python by writing status/heartbeat files into a real temp queue dir.
Each test runs on every Lua flavour VLC may embed (5.1, 5.2, LuaJIT) that lupa offers.
"""

from __future__ import annotations

import importlib
import os
import re
import time
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

ROOT = Path(__file__).resolve().parent.parent
INTF = ROOT / "src" / "vlcsubsync" / "lua" / "intf" / "subsync.lua"
EXT = ROOT / "src" / "vlcsubsync" / "lua" / "extensions" / "subsync_ext.lua"
MOCK = Path(__file__).resolve().parent / "lua" / "vlc_mock.lua"

RUNTIMES = []
for _name in ("lua51", "lua52", "luajit21"):
    try:
        importlib.import_module(f"lupa.{_name}")
        RUNTIMES.append(_name)
    except ImportError:  # pragma: no cover - depends on the lupa build
        pass

MOVIE_URI = "file:///media/My%20Movies/Am%C3%A9lie%20(2001)%20%5B50%25%5D.mkv"
MOVIE_PATH = "/media/My Movies/Amélie (2001) [50%].mkv"
AUDIO = [(10, "Track 1 - [French]"), (11, "Track 2 - [English]")]
SUBS = [(20, "Track 1 - [English]"), (21, "Track 2 - [Spanish]")]

TICK = 500_000


def parse_kv(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def write_kv(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pytmp")
    tmp.write_text("".join(f"{k}={v}\n" for k, v in data.items()), encoding="utf-8")
    os.replace(tmp, path)


class Harness:
    """A Lua runtime with the mocked vlc global and one script loaded."""

    def __init__(
        self,
        runtime: str,
        tmp_path: Path,
        script: Path,
        make_path: bool = False,
        windows: bool = False,
    ):
        mod = importlib.import_module(f"lupa.{runtime}")
        self.lua = mod.LuaRuntime(unpack_returned_tuples=True)
        # The scripts detect Windows from package.config's directory separator.
        # Pin it so results don't depend on the OS running the tests.
        sep = "\\\\" if windows else "/"
        self.lua.execute(f'package.config = "{sep}" .. package.config:sub(2)')
        self.userdata = tmp_path / "vlcdata"
        self.userdata.mkdir()
        self.q = self.userdata / "subsync"
        g = self.lua.globals()

        def mkdir(path, mode):
            try:
                os.mkdir(path)
                return 0
            except FileExistsError:
                return -1

        g.py_mkdir = mkdir
        g.py_userdata = str(self.userdata)
        g.py_make_path = make_path
        self.mock = self.lua.execute(
            f"""
            local Mock = dofile({str(MOCK)!r})
            local opts = {{ userdatadir = py_userdata,
                            mkdir = function(p, m) return py_mkdir(p, m) end }}
            if py_make_path then
                opts.make_path = function(uri)
                    local p = uri:gsub("^file://", "")
                    p = p:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
                    return "MP:" .. p
                end
            end
            local m = Mock.new(opts)
            vlc = m.vlc
            SUBSYNC_TEST = true
            return m
            """
        )
        self.lua.execute(f"dofile({str(script)!r})")
        self.S = g.subsync
        self.E = g.subsync_ext

    # ---- mock helpers -------------------------------------------------------------
    def table(self, items):
        return self.lua.table_from(
            [self.lua.table_from({"id": i, "label": label}) for i, label in items]
        )

    def set_input(self, uri=MOVIE_URI, audio=AUDIO, subs=SUBS, audio_sel=10, spu_sel=20):
        self.mock.set_input(uri, self.table(audio), self.table(subs), audio_sel, spu_sel)

    def select(self, var, es_id):
        self.mock.select(var, es_id)

    def selected(self, var):
        return self.mock.input.sel[var]

    def spu_ids(self):
        return [t.id for t in self.mock.input.spu.values()]

    def osd(self) -> list[str]:
        return list(self.mock.osd.values())

    def logs(self) -> list[str]:
        return list(self.mock.logs.values())

    def added(self) -> list[tuple[str, bool]]:
        return [(a.path, a.select) for a in self.mock.added.values()]

    # ---- intf driving ---------------------------------------------------------------
    def init(self):
        self.S.init()

    def tick(self, n: int = 1, dt: int = TICK):
        for _ in range(n):
            self.S.tick()  # errors propagate to the test
            self.mock.now = self.mock.now + dt

    def settle(self):
        """Tick past the debounce window."""
        self.tick(5)

    # ---- fake daemon -------------------------------------------------------------
    def requests(self) -> list[dict[str, str]]:
        d = self.q / "requests"
        if not d.exists():
            return []
        return [parse_kv(p) for p in sorted(d.glob("*.req"))]

    def heartbeat(self, age: int = 0):
        write_kv(
            self.q / "heartbeat", {"time": int(time.time()) - age, "pid": 1234, "version": "0.1.0"}
        )

    def status(self, req_id: str, **kv):
        data = {"id": req_id, **kv}
        write_kv(self.q / "jobs" / f"{req_id}.status", data)

    def finish(self, req_id: str, message="offset +2.35s, drift +4.1%", applied=1, **extra):
        out = self.q / "out" / f"{req_id}.srt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
        self.status(
            req_id,
            state="done",
            progress="1.0",
            message=message,
            output=str(out),
            applied=applied,
            method="whisper",
            offset="2.350",
            scale="1.0417",
            confidence="0.93",
            **extra,
        )
        # the daemon deletes the request once picked up
        req = self.q / "requests" / f"{req_id}.req"
        if req.exists():
            req.unlink()
        return str(out)

    def control(self, **kv):
        write_kv(self.q / "control", kv)

    def intf_state(self) -> dict[str, str]:
        return parse_kv(self.q / "intf_state")


@pytest.fixture(params=RUNTIMES)
def runtime(request):
    return request.param


@pytest.fixture
def h(runtime, tmp_path):
    hh = Harness(runtime, tmp_path, INTF)
    hh.heartbeat()
    hh.init()
    return hh


# =============================================================================== intf


def test_scripts_parse_on_all_runtimes(runtime):
    mod = importlib.import_module(f"lupa.{runtime}")
    lua = mod.LuaRuntime()
    for script in (INTF, EXT):
        err = lua.execute(f"local f, e = loadfile({str(script)!r}); return e")
        assert err is None, err


def test_windows_uri_decodes_to_drive_path(runtime, tmp_path):
    w = Harness(runtime, tmp_path, INTF, windows=True)
    assert w.S.uri_to_path("file:///C:/My%20Movies/a%20b.mkv") == "C:\\My Movies\\a b.mkv"
    assert w.S.uri_to_path("file://server/share/x.mkv") == "\\\\server\\share\\x.mkv"


def test_request_written_with_ordinals_labels_and_decoded_path(h):
    h.set_input(audio_sel=11, spu_sel=21)
    h.tick()
    assert h.requests() == []  # debounce
    h.settle()
    reqs = h.requests()
    assert len(reqs) == 1
    r = reqs[0]
    assert r["version"] == "1"
    assert re.fullmatch(r"\d+_\d+", r["id"])
    assert r["media"] == MOVIE_PATH
    assert r["audio_index"] == "1" and r["audio_label"] == "Track 2 - [English]"
    assert r["sub_index"] == "1" and r["sub_label"] == "Track 2 - [Spanish]"
    assert r["force"] == "0" and r["sub_path"] == ""
    assert (h.q / "requests" / f"{r['id']}.req").exists()
    assert not list((h.q / "requests").glob("*.tmp"))
    assert any(o.startswith("Syncing subtitles") for o in h.osd())
    assert h.mock.last_osd.position == "top-right"
    assert h.mock.last_osd.duration == 3000000
    assert h.intf_state()["state"] == "syncing"


def test_make_path_is_preferred_when_available(runtime, tmp_path):
    hh = Harness(runtime, tmp_path, INTF, make_path=True)
    hh.heartbeat()
    hh.init()
    hh.set_input()
    hh.settle()
    assert hh.requests()[0]["media"] == "MP:" + MOVIE_PATH


def test_debounce_coalesces_quick_changes(h):
    h.set_input(spu_sel=20)
    h.tick(2)  # 1 s
    h.select("spu-es", 21)
    h.tick(2)  # 1 s after the change: still debouncing
    assert h.requests() == []
    h.tick(2)
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["sub_index"] == "1"


def test_no_request_when_subtitles_disabled(h):
    h.set_input(spu_sel=-1)
    h.tick(10)
    assert h.requests() == []
    # disabling during the debounce window cancels the trigger
    h.select("spu-es", 20)
    h.tick()
    h.select("spu-es", -1)
    h.tick(10)
    assert h.requests() == []


def test_non_file_uri_ignored(h):
    h.set_input(uri="https://example.com/stream.m3u8")
    h.tick(10)
    assert h.requests() == []
    assert any("not a local file" in line for line in h.logs())


def test_done_adds_selects_and_memoizes(h):
    h.set_input()
    h.settle()
    req = h.requests()[0]
    h.status(req["id"], state="running", progress="0.42", message="Transcribing 3/10")
    h.tick(8)
    assert any("42%" in o and "Transcribing 3/10" in o for o in h.osd())
    out = h.finish(req["id"])
    h.tick(2)
    assert h.added() == [(out, True)]
    new_id = h.spu_ids()[-1]
    assert h.selected("spu-es") == new_id
    assert any(o == "Subtitles synced: offset +2.35s, drift +4.1%" for o in h.osd())
    st = h.intf_state()
    assert st["state"] == "done" and "offset +2.35s" in st["last_result"]
    # selecting our synced track must not trigger another sync
    h.tick(10)
    assert h.requests() == []
    # go back to the original track -> our synced track is re-selected, no new request
    h.select("spu-es", 20)
    h.settle()
    assert h.requests() == []
    assert h.selected("spu-es") == new_id
    # our track is excluded from ordinals: picking sub 21 is still ordinal 1
    h.select("spu-es", 21)
    h.settle()
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["sub_index"] == "1"


def test_delayed_track_appearance(h):
    h.mock.add_delay_us = 2_000_000
    h.set_input()
    h.settle()
    req = h.requests()[0]
    h.finish(req["id"])
    h.tick(8)
    new_id = h.spu_ids()[-1]
    assert new_id >= 100
    assert h.selected("spu-es") == new_id
    h.tick(10)
    assert h.requests() == []  # the auto-selected synced track was recognised as ours


def test_audio_change_triggers_resync(h):
    h.set_input(audio_sel=10, spu_sel=20)
    h.settle()
    r1 = h.requests()[0]
    h.finish(r1["id"])
    h.tick(3)
    synced_a0 = h.selected("spu-es")
    assert synced_a0 >= 100
    # switch audio while our synced track is selected -> resync source sub 0 with audio 1
    h.select("audio-es", 11)
    h.settle()
    reqs = h.requests()
    assert len(reqs) == 1
    assert reqs[0]["audio_index"] == "1" and reqs[0]["sub_index"] == "0"
    h.finish(reqs[0]["id"])
    h.tick(3)
    synced_a1 = h.selected("spu-es")
    assert synced_a1 not in (synced_a0, 20)
    # back to audio 0 -> re-select the first synced track from memo, no request
    h.select("audio-es", 10)
    h.settle()
    assert h.requests() == []
    assert h.selected("spu-es") == synced_a0


def test_newer_request_supersedes_pending_one(h):
    h.set_input(spu_sel=20)
    h.settle()
    first = h.requests()[0]["id"]
    h.select("spu-es", 21)
    h.settle()
    reqs = h.requests()
    assert [r["sub_index"] for r in reqs] == ["1"]
    assert reqs[0]["id"] != first
    # a late status for the superseded job is ignored
    h.finish(first)
    h.tick(3)
    assert h.added() == []


def test_error_status_shows_osd(h):
    h.set_input()
    h.settle()
    req = h.requests()[0]
    h.status(req["id"], state="error", message="Cannot decode audio")
    h.tick(2)
    assert "SubSync error: Cannot decode audio" in h.osd()
    assert h.added() == []
    st = h.intf_state()
    assert st["state"] == "error" and "Cannot decode audio" in st["last_result"]


def test_not_applied_keeps_original(h):
    h.set_input()
    h.settle()
    req = h.requests()[0]
    h.finish(req["id"], message="low confidence 0.21", applied=0)
    h.tick(2)
    assert h.added() == []
    assert any("not synced" in o and "low confidence" in o for o in h.osd())
    # re-selecting the same combination does not retry automatically
    h.select("spu-es", 21)
    h.settle()
    for r in h.requests():
        (h.q / "requests" / f"{r['id']}.req").unlink()
    h.select("spu-es", 20)
    h.settle()
    assert h.requests() == []


def test_control_auto_off_and_sync_now(h):
    h.control(auto=0, sync_now=5)  # pre-existing counter is only a baseline
    h.set_input()
    h.settle()
    assert h.requests() == []
    assert h.intf_state()["auto"] == "0"
    h.control(auto=0, sync_now=6)
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["force"] == "0"
    out = h.finish(reqs[0]["id"])
    h.tick(2)
    assert h.added() == [(out, True)]
    # sync_now while our synced track is selected -> re-sync the source with force=1
    h.control(auto=0, sync_now=7)
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["force"] == "1" and reqs[0]["sub_index"] == "0"
    # auto back on
    h.control(auto=1, sync_now=7)
    h.tick()
    assert h.intf_state()["auto"] == "1"


def test_default_requests_have_no_mode(h):
    h.set_input()
    h.settle()
    assert "mode" not in h.requests()[0]
    assert "Syncing subtitles…" in h.osd()


def test_sync_now_exhaustive_writes_mode_and_warns(h):
    h.control(auto=0, sync_now=1)  # baseline
    h.set_input()
    h.settle()
    assert h.requests() == []
    h.control(auto=0, sync_now=2, sync_now_mode="exhaustive")
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1
    r = reqs[0]
    assert r["mode"] == "exhaustive" and r["force"] == "0"
    assert "Syncing subtitles (exhaustive, may take a while)…" in h.osd()
    assert "may take a while" in h.intf_state()["message"]
    # progress OSD keeps the warning
    h.status(r["id"], state="running", progress="0.42", message="Transcribing 20/58")
    h.tick(8)
    assert any(
        o.startswith("Syncing subtitles (exhaustive, may take a while)… 42%") for o in h.osd()
    )
    out = h.finish(r["id"])
    h.tick(2)
    assert h.added() == [(out, True)]
    # a later plain "Sync now" (no mode key) goes back to the default
    h.control(auto=0, sync_now=3)
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1 and "mode" not in reqs[0]


def test_sync_now_unknown_mode_is_ignored(h):
    h.control(auto=0, sync_now=1)
    h.set_input()
    h.settle()
    h.control(auto=0, sync_now=2, sync_now_mode="bogus")
    h.tick()
    assert "mode" not in h.requests()[0]


def test_sync_now_without_subtitle_track(h):
    h.control(sync_now=1)
    h.set_input(spu_sel=-1)
    h.tick()
    h.control(sync_now=2)
    h.tick()
    assert h.requests() == []
    assert "SubSync: select a subtitle track first" in h.osd()


def test_missing_heartbeat_warns_once_and_tries_autostart(runtime, tmp_path):
    hh = Harness(runtime, tmp_path, INTF)
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "vlc-subsync").write_text("#!/bin/sh\n")
    env = {"HOME": str(home)}
    calls = []
    hh.S.getenv = lambda name: env.get(name)
    hh.S.execute = lambda cmd: calls.append(cmd) or 0
    hh.init()
    hh.set_input()
    hh.settle()
    assert len(hh.requests()) == 1  # still queued for when the daemon comes up
    hh.select("spu-es", 21)
    hh.settle()
    hh.tick(20)
    warnings = [o for o in hh.osd() if "helper not running" in o]
    assert len(warnings) == 1
    assert calls == [f"'{home}/.local/bin/vlc-subsync' serve >/dev/null 2>&1 &"]
    assert hh.intf_state()["daemon"] == "0"


def test_sandboxed_vlc_does_not_spawn(runtime, tmp_path):
    hh = Harness(runtime, tmp_path, INTF)
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "vlc-subsync").write_text("#!/bin/sh\n")
    env = {"HOME": str(home), "SNAP": "/snap/vlc/4472"}
    calls = []
    hh.S.getenv = lambda name: env.get(name)
    hh.S.execute = lambda cmd: calls.append(cmd) or 0
    hh.heartbeat(age=60)  # stale
    hh.init()
    hh.set_input()
    hh.settle()
    assert calls == []
    assert "SubSync helper not running" in hh.osd()


# ------------------------------------------------------------------ launcher-file spawning


def launcher_harness(runtime, tmp_path, *, windows, mode="spawn", exe=None, args=""):
    """Harness whose queue dir holds a launcher file (as `vlc-subsync setup` writes)."""
    hh = Harness(runtime, tmp_path, INTF, windows=windows)
    sep = "\\" if windows else "/"
    q = str(hh.userdata) + sep + "subsync"  # what the intf computes (M.queue_dir)
    os.makedirs(q, exist_ok=True)
    if exe is None:
        exe = tmp_path / "Python Scripts" / "vlc-subsync-daemon.exe"
        exe.parent.mkdir()
        exe.write_text("")
    write_kv(Path(q + sep + "launcher"), {"version": 1, "mode": mode, "exe": exe, "args": args})
    calls = []
    hh.S.execute = lambda cmd: calls.append(cmd) or 0
    hh.S.getenv = lambda name: None
    return hh, calls, q + sep, str(exe)


def test_windows_launcher_spawns_gui_exe_when_no_heartbeat(runtime, tmp_path):
    hh, calls, qp, exe = launcher_harness(runtime, tmp_path, windows=True)
    hh.init()
    # started at VLC startup, from the launcher's absolute path, without waiting
    assert calls == [f'start "" /B "{exe}"']
    assert any("starting helper" in line for line in hh.logs())
    hh.tick(20)  # still no heartbeat, but it was started a moment ago
    hh.set_input()
    hh.settle()
    assert len(calls) == 1
    assert "SubSync helper not running – starting it…" in hh.osd()
    # a minute without a heartbeat: start it again, at most SPAWN_MAX times per session
    for _ in range(5):
        hh.mock.now = hh.mock.now + 61_000_000
        hh.tick(5)
    assert calls == [f'start "" /B "{exe}"'] * 3


def test_windows_launcher_args_are_quoted(runtime, tmp_path):
    hh, calls, qp, exe = launcher_harness(
        runtime, tmp_path, windows=True, args="-m vlcsubsync.cli serve --no-console"
    )
    hh.init()
    assert calls == [f'start "" /B "{exe}" "-m" "vlcsubsync.cli" "serve" "--no-console"']


def test_launcher_spawn_is_heartbeat_triggered(runtime, tmp_path):
    hh, calls, qp, exe = launcher_harness(runtime, tmp_path, windows=False, args="serve")
    write_kv(Path(qp + "heartbeat"), {"time": int(time.time()), "pid": 1, "version": "0.1.0"})
    hh.init()
    hh.set_input()
    hh.settle()
    hh.tick(10)
    assert calls == []  # the helper is alive
    # the helper went away (crashed / exited): its heartbeat goes stale
    write_kv(Path(qp + "heartbeat"), {"time": int(time.time()) - 60, "pid": 1, "version": "0"})
    hh.tick(5)
    assert calls == [f"'{exe}' 'serve' >/dev/null 2>&1 &"]
    write_kv(Path(qp + "heartbeat"), {"time": int(time.time()), "pid": 2, "version": "0.1.0"})
    hh.mock.now = hh.mock.now + 120_000_000
    hh.tick(5)
    assert len(calls) == 1  # alive again: nothing more


def test_launcher_service_mode_never_spawns(runtime, tmp_path):
    hh, calls, qp, exe = launcher_harness(runtime, tmp_path, windows=False, mode="service")
    hh.init()
    hh.set_input()
    hh.settle()
    hh.mock.now = hh.mock.now + 120_000_000
    hh.tick(10)
    assert calls == []  # systemd / launchd start the helper
    assert len(hh.requests()) == 1  # the request waits for it
    assert "SubSync helper not running" in hh.osd()


def test_launcher_missing_exe_logs_and_does_not_spawn(runtime, tmp_path):
    hh, calls, qp, exe = launcher_harness(
        runtime, tmp_path, windows=True, exe=tmp_path / "gone" / "vlc-subsync-daemon.exe"
    )
    hh.init()
    hh.tick(10)
    assert calls == []
    assert sum("helper not found" in line for line in hh.logs()) == 1


def test_launcher_spawn_skipped_in_sandboxed_vlc(runtime, tmp_path):
    hh, calls, qp, exe = launcher_harness(runtime, tmp_path, windows=False, args="serve")
    hh.S.getenv = lambda name: "/snap/vlc/x1" if name == "SNAP" else None
    hh.init()
    hh.tick(10)
    assert calls == []


def test_input_change_and_stop_reset_state(h):
    h.set_input()
    h.settle()
    first = h.requests()[0]["id"]
    h.set_input(uri="file:///tmp/other.mkv")
    h.tick()
    # the unpicked request of the previous media is withdrawn
    assert not (h.q / "requests" / f"{first}.req").exists()
    h.settle()
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["media"] == "/tmp/other.mkv"
    h.mock.stop()
    h.tick(2)
    assert h.intf_state()["state"] == "idle"


def test_run_loop_exits_on_interrupted_mwait(runtime, tmp_path):
    hh = Harness(runtime, tmp_path, INTF)
    hh.heartbeat()
    hh.set_input()
    hh.mock.interrupt_after = 8
    hh.S.run()  # must return, not raise
    assert any("[subsync] request " in line for line in hh.logs())
    assert hh.requests() == []  # the unpicked request was withdrawn on shutdown
    assert hh.intf_state()["state"] == "stopped"


def test_tick_errors_are_caught_and_logged(h):
    h.set_input()
    h.lua.execute(
        "vlc.var.get_list = function() error('boom') end; "
        "vlc.var.get = function() error('boom2') end"
    )
    assert h.S.safe_tick() in (True, False)
    h.lua.execute("subsync.tick = function() error('kaboom') end")
    assert h.S.safe_tick() is False
    assert any("[subsync] tick failed" in line and "kaboom" in line for line in h.logs())


# ======================================================================= delay mode
# Experimental "no extra track" mode: the original track stays selected and is
# corrected live through the input's "spu-delay" (µs; positive = later).


def seg_status(*segs, knots=None):
    """Status keys for mapping segments (sub_start, sub_end, scale, offset)."""
    out = {"segments": len(segs)}
    for i, (lo, hi, scale, offset) in enumerate(segs):
        lo_s = "" if lo is None else f"{lo:.3f}"
        hi_s = "" if hi is None else f"{hi:.3f}"
        out[f"seg{i}"] = f"{lo_s},{hi_s},{scale:.7f},{offset:.4f}"
    for i, k in (knots or {}).items():
        out[f"seg{i}_knots"] = ";".join(f"{t:.3f}:{c:.4f}" for t, c in k)
    return out


def spu_delay(hh) -> int:
    return int(hh.selected("spu-delay"))


def delay_sets(hh) -> list[int]:
    return [int(v.value) for v in hh.mock.var_sets.values() if v.name == "spu-delay"]


def expected_us(T, scale, offset, lookahead=0.0):
    """spu-delay the intf aims at for playback time T (one linear segment): the
    mapping's delay at T + lead, lead = lookahead + the negative part of the delay
    (VLC decodes subtitles that much earlier)."""

    def d(t):
        return t - (t - offset) / scale

    lead = lookahead + max(0.0, -d(T))
    return round(d(T + lead) * 1e6)


def start_delay(
    hh, *segs, knots=None, t0=10.0, via="control", message="offset +2.50s", lookahead=0.0
):
    """Play, sync with the given mapping in delay mode; returns the request id.

    The lookahead (subtitles take their delay when decoded, ahead of display) is
    off by default so the expected values are the mapping at "time" itself."""
    hh.S.LOOKAHEAD_S = lookahead
    if via == "control":
        hh.control(auto=1, sync_now=0, sync_mode="delay")
    hh.set_input()
    hh.mock.set_time(t0)
    hh.settle()
    req = hh.requests()[0]
    extra = seg_status(*segs, knots=knots)
    if via == "status":
        extra["sync_mode"] = "delay"
    hh.finish(req["id"], message=message, **extra)
    hh.tick()
    return req["id"]


def test_delay_constant_offset(h):
    start_delay(h, (None, None, 1.0, 2.5))
    assert spu_delay(h) == 2_500_000
    assert h.selected("spu-es") == 20  # the original track stays selected
    assert h.added() == []
    assert "Subtitles synced (live delay, experimental): offset +2.50s" in h.osd()
    st = h.intf_state()
    assert st["sync_mode"] == "delay" and st["delay_active"] == "1"
    # a constant mapping never needs another set, however time moves
    n = len(delay_sets(h))
    for T in (20, 60, 61.5, 300):
        h.mock.set_time(T)
        h.tick()
    assert len(delay_sets(h)) == n
    assert any("[subsync] spu-delay=2500000 us" in line for line in h.logs())


def test_delay_negative_offset_sign(h):
    """Subtitles 7 s late: audio = sub - 7 -> spu-delay -7 s (earlier)."""
    start_delay(h, (None, None, 1.0, -7.0), t0=30)
    assert spu_delay(h) == -7_000_000
    # VLC pauses playback for every new low of a negative delay: say so
    assert any(o.endswith("may pause playback up to 7 s in total") for o in h.osd())
    assert any("needs spu-delay down to -7.0 s" in line for line in h.logs())


def test_delay_positive_needs_no_pause_warning(h):
    start_delay(h, (None, None, 1.0, 2.5))
    assert not any("may pause" in o for o in h.osd())


def test_min_delay_over_the_file(h):
    segs = h.S.parse_segments(h.lua.table_from(seg_status((None, None, 23.976 / 25, 1.44))))
    # d(T) = T - (T - 1.44) / 0.959: +1.44 at 0, about -6.3 s at 182 s
    assert h.S.min_delay(segs, 182.0) == pytest.approx(182 - (182 - 1.44) * 25 / 23.976)
    cut = h.S.parse_segments(
        h.lua.table_from(seg_status((None, 100.0, 1.0, 2.0), (100.0, None, 1.0, -10.0)))
    )
    assert h.S.min_delay(cut, 0) == pytest.approx(-10.0)


def test_delay_follows_drift(h):
    scale, offset = 23.976 / 25, 1.44  # sidecar timed for 25 fps on 23.976 audio
    start_delay(h, (None, None, scale, offset), t0=10)
    assert abs(spu_delay(h) - expected_us(10, scale, offset)) <= 1
    values = [spu_delay(h)]
    for k in range(21, 361):  # 10 s .. 180 s, one tick per 0.5 s of playback
        T = k / 2
        h.mock.set_time(T)
        h.tick()
        target = expected_us(T, scale, offset)
        # within the tolerance plus one tick's change (~21 ms)
        assert abs(spu_delay(h) - target) <= 40_000 + 21_000
        values.append(spu_delay(h))
    # the delay grows (more negative) over time: -0.4 s at 10 s, -6.3 s at 180 s
    assert values[0] > -500_000 and values[-1] < -6_000_000
    assert values == sorted(values, reverse=True)
    sets = delay_sets(h)
    # updated in steps of just over the 40 ms tolerance, not on every tick
    assert 100 < len(sets) < 180
    assert all(abs(b - a) > 40_000 for a, b in zip(sets, sets[1:], strict=False))
    assert h.added() == []


def test_delay_small_changes_are_not_applied(h):
    scale = 1.001  # +1 ms per second
    start_delay(h, (None, None, scale, 0.0), t0=100)
    n = len(delay_sets(h))
    T = 100.0
    for _ in range(60):  # 30 s of normal playback, one tick per 0.5 s
        T += 0.5
        h.mock.set_time(T)
        h.tick()
    assert len(delay_sets(h)) == n  # 30 ms of change: below the tolerance
    for _ in range(30):
        T += 0.5
        h.mock.set_time(T)
        h.tick()
    assert len(delay_sets(h)) == n + 1


def test_delay_piecewise_cut(h):
    # ad-break style: audio = sub + 2 before sub 100 s, audio = sub + 10 after
    # (audio 102..110 s is not in the subtitles: a forward gap)
    start_delay(h, (None, 100.0, 1.0, 2.0), (100.0, None, 1.0, 10.0), t0=50)
    assert spu_delay(h) == 2_000_000
    for T, want in ((80, 2_000_000), (101.5, 2_000_000), (105, 10_000_000), (200, 10_000_000)):
        h.mock.set_time(T)
        h.tick()
        assert spu_delay(h) == want, T
    # backwards seek into the first part
    h.mock.set_time(40)
    h.tick()
    assert spu_delay(h) == 2_000_000


def test_delay_overlapping_cut_prefers_later_segment(h):
    # backward cut: sub 0..100 -> audio 0..100, sub 100.. -> audio 90..
    start_delay(h, (None, 100.0, 1.0, 0.0), (100.0, None, 1.0, -10.0), t0=50)
    assert spu_delay(h) == 0
    h.mock.set_time(95)  # inside both audio spans
    h.tick()
    assert spu_delay(h) == -10_000_000


def test_delay_knots_are_followed(h):
    knots = {0: [(0.0, 0.0), (100.0, 0.3), (200.0, 0.3)]}
    start_delay(h, (None, None, 1.0, 1.0), knots=knots, t0=0.5)
    assert abs(spu_delay(h) - 1_000_000) < 5_000
    h.mock.set_time(150)
    h.tick()
    assert abs(spu_delay(h) - 1_300_000) < 2_000


def test_delay_seek_applies_immediately(h):
    scale = 1.0005  # 0.5 ms/s: a 60 s seek moves the target by only 30 ms
    start_delay(h, (None, None, scale, 0.0), t0=100)
    n = len(delay_sets(h))
    h.mock.set_time(160)
    h.tick()
    sets = delay_sets(h)
    assert len(sets) == n + 1  # below the tolerance, but a seek re-sets at once
    assert sets[-1] == expected_us(160, scale, 0.0)
    assert any("seek" in line and "spu-delay=" in line for line in h.logs())
    # normal playback afterwards does not count as a seek
    h.mock.set_time(160.5)
    h.tick()
    assert len(delay_sets(h)) == n + 1


def test_delay_keeps_manual_user_bias(h):
    start_delay(h, (None, None, 1.0, 2.5), t0=10)
    assert spu_delay(h) == 2_500_000
    # the user presses "h"/"g" (50 ms steps) or uses Track Synchronization
    h.select("spu-delay", 2_600_000)
    h.tick()
    assert spu_delay(h) == 2_600_000  # left alone: it is the user's choice
    assert any("bias now 100000 us" in line for line in h.logs())
    # the bias stays on top of later corrections
    h.mock.set_time(400)
    h.tick()
    assert spu_delay(h) == 2_600_000
    # switching the track away gives back only the user's own part
    h.select("spu-es", 21)
    h.tick()
    assert spu_delay(h) == 100_000
    assert h.intf_state()["delay_active"] == "0"


def test_delay_existing_user_delay_is_the_initial_bias(h):
    h.control(auto=1, sync_now=0, sync_mode="delay")
    h.set_input()
    h.select("spu-delay", -300_000)  # set by the user before the sync finished
    h.mock.set_time(10)
    h.settle()
    req = h.requests()[0]
    h.finish(req["id"], **seg_status((None, None, 1.0, 2.0)))
    h.tick()
    assert spu_delay(h) == 1_700_000
    h.select("spu-es", -1)
    h.tick()
    assert spu_delay(h) == -300_000


def test_delay_track_switch_resets_and_reselect_reapplies(h):
    start_delay(h, (None, None, 1.0, 3.0), t0=10)
    assert spu_delay(h) == 3_000_000
    h.select("spu-es", 21)  # another subtitle track: not corrected
    h.tick()
    assert spu_delay(h) == 0
    h.settle()
    other = [r for r in h.requests() if r["sub_index"] == "1"]
    assert len(other) == 1  # the other track gets its own sync
    # back to the synced original: the remembered mapping is applied again
    h.select("spu-es", 20)
    h.settle()
    assert spu_delay(h) == 3_000_000
    assert [r for r in h.requests() if r["sub_index"] == "0"] == []
    # audio change: reset, and a new sync for the new combination
    h.select("audio-es", 11)
    h.tick()
    assert spu_delay(h) == 0
    assert h.added() == []


def test_delay_input_change_and_stop_forget_mapping(h):
    start_delay(h, (None, None, 1.0, 3.0), t0=10)
    h.mock.stop()
    h.tick()
    assert h.S.state.delay is None
    h.set_input(uri="file:///other.mkv")
    h.tick()
    assert h.S.state.delay is None
    assert spu_delay(h) == 0  # a new input starts from its own spu-delay


def test_delay_mode_from_helper_config(h):
    """No toggle used: the helper's configured sync_mode (in the status) decides."""
    start_delay(h, (None, None, 1.0, 2.0), via="status")
    assert spu_delay(h) == 2_000_000
    assert h.added() == []
    assert h.intf_state()["sync_mode"] == "delay"


def test_delay_toggle_overrides_helper_config(h):
    h.control(auto=1, sync_now=0, sync_mode="track")
    h.set_input()
    h.settle()
    req = h.requests()[0]
    out = h.finish(req["id"], sync_mode="delay", **seg_status((None, None, 1.0, 2.0)))
    h.tick(2)
    assert h.added() == [(out, True)]
    assert delay_sets(h) == []


def test_delay_mode_without_segments_resyncs_once_then_gives_up(h):
    """A result without mapping (cached before mappings were sent, or an old
    helper): re-sync once with force; never add a track; never remember it as
    "not applied" (a later selection asks again)."""
    h.control(auto=1, sync_now=0, sync_mode="delay")
    h.set_input()
    h.mock.set_time(10)
    h.settle()
    req = h.requests()[0]
    assert req["force"] == "0"
    h.finish(req["id"])  # no segments
    h.tick()
    retry = h.requests()
    assert len(retry) == 1 and retry[0]["force"] == "1" and retry[0]["id"] != req["id"]
    assert retry[0]["sub_index"] == req["sub_index"]
    assert retry[0]["audio_label"] == req["audio_label"]
    assert not any("needs a newer helper" in o for o in h.osd())
    # the fresh sync has the mapping: applied live
    h.finish(retry[0]["id"], **seg_status((None, None, 1.0, 2.0)))
    h.tick()
    assert spu_delay(h) == 2_000_000 and h.added() == []


def test_delay_mode_old_helper_gives_up_without_memoizing(h):
    h.control(auto=1, sync_now=0, sync_mode="delay")
    h.set_input()
    h.settle()
    h.finish(h.requests()[0]["id"])
    h.tick()
    h.finish(h.requests()[0]["id"])  # the forced retry has no mapping either
    h.tick(3)
    assert h.added() == [] and delay_sets(h) == []
    assert any("live delay needs a newer helper" in o for o in h.osd())
    assert h.requests() == []  # no loop
    # nothing memoized as "not applied": re-selecting the track asks again
    h.select("spu-es", 21)
    h.settle()
    h.select("spu-es", 20)
    h.settle()
    assert [r["sub_index"] for r in h.requests()][-1] == "0"


def test_default_track_mode_ignores_segments(h):
    """Default: the synced file is added as a track even though segments are sent."""
    h.set_input()
    h.mock.set_time(10)
    h.settle()
    req = h.requests()[0]
    out = h.finish(req["id"], **seg_status((None, None, 1.0, 2.0)))
    h.tick(2)
    assert h.added() == [(out, True)]
    assert delay_sets(h) == []
    assert h.intf_state()["sync_mode"] == "track"
    assert h.intf_state()["delay_active"] == "0"


def test_delay_toggle_while_playing_moves_result(h):
    start_delay(h, (None, None, 1.0, 2.0), t0=10)
    assert h.added() == []
    # off: the user's delay comes back and the synced file is loaded as a track
    h.control(auto=1, sync_now=0, sync_mode="track")
    h.tick(2)
    assert spu_delay(h) == 0
    assert len(h.added()) == 1
    ours = h.spu_ids()[-1]
    assert h.selected("spu-es") == ours
    # on again: back to the original track, corrected live
    h.control(auto=1, sync_now=0, sync_mode="delay")
    h.tick(2)
    assert h.selected("spu-es") == 20
    assert spu_delay(h) == 2_000_000
    assert len(h.added()) == 1


def test_delay_toggle_off_reselects_loaded_synced_track(h):
    """Track-mode sync, then live delay on/off twice: turning delay off goes back
    to the synced track already loaded instead of adding the same file again."""
    h.control(auto=1, sync_now=0, sync_mode="track")
    h.set_input()
    h.mock.set_time(10)
    h.settle()
    req = h.requests()[0]
    out = h.finish(req["id"], **seg_status((None, None, 1.0, 2.0)))
    h.tick(2)
    assert h.added() == [(out, True)]
    ours = h.spu_ids()[-1]
    assert h.selected("spu-es") == ours
    for _ in range(2):
        h.control(auto=1, sync_now=0, sync_mode="delay")
        h.tick(2)
        assert h.selected("spu-es") == 20
        assert spu_delay(h) == 2_000_000
        h.control(auto=1, sync_now=0, sync_mode="track")
        h.tick(2)
        assert spu_delay(h) == 0
        assert h.selected("spu-es") == ours
    assert h.added() == [(out, True)]
    assert h.spu_ids().count(ours) == 1


def test_delay_sync_now_forces_resync(h):
    start_delay(h, (None, None, 1.0, 2.0), t0=10)
    h.control(auto=1, sync_now=1, sync_mode="delay")
    h.tick(2)
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["force"] == "1"


def test_delay_lookahead_aims_at_subtitles_decoded_now(h):
    """With the lookahead, the delay is the mapping's at time + lead, where lead is
    LOOKAHEAD_S plus the negative part of the delay (VLC decodes that much earlier)."""
    scale, offset = 23.976 / 25, 1.44
    start_delay(h, (None, None, scale, offset), t0=100, lookahead=1.0)
    assert spu_delay(h) == expected_us(100, scale, offset, lookahead=1.0)
    plain = 100 - (100 - offset) / scale  # about -2.8 s at 100 s
    assert spu_delay(h) < round(plain * 1e6) - 150_000  # aimed ~3.8 s ahead
    assert h.S.LOOKAHEAD_S == 1.0


def test_delay_default_lookahead():
    src = INTF.read_text(encoding="utf-8")
    assert "M.LOOKAHEAD_S = 1.0" in src


def test_parse_segments_rejects_bad_input(h):
    assert h.S.parse_segments(h.lua.table_from({"segments": "1", "seg0": ",,1.0,2.0"})) is not None
    for bad in (
        {"segments": "2", "seg0": ",,1.0,2.0"},  # missing seg1
        {"segments": "1", "seg0": ",,0,2.0"},  # scale <= 0
        {"segments": "1", "seg0": ",,abc,2.0"},
        {"segments": "1", "seg0": "x,,1.0,2.0"},
        {"segments": "1", "seg0": ",,1.0,2.0", "seg0_knots": "1:2;bad"},
        {"segments": "0"},
    ):
        assert h.S.parse_segments(h.lua.table_from(bad)) is None, bad


def test_delay_same_uri_replay_is_a_new_input(h):
    """Repeat-one / loop / replay: a new input object with the same URI and a fresh
    spu-delay. Its 0 must not become a +5.36 s "user bias"; the mapping is applied
    again from memory, without a new request."""
    start_delay(h, (None, None, 23.976 / 25, 1.44), t0=170)
    assert spu_delay(h) < -5_000_000
    assert h.selected("subsync-delay").split("|")[1:] == ["0", str(spu_delay(h))]
    # the same file starts again inside one tick: new input, spu-delay from sub-delay
    h.set_input()
    h.mock.set_time(0.5)
    h.tick()
    assert h.S.state.delay is None
    assert not any("bias now" in line for line in h.logs())
    assert any("new input object for the same media" in line for line in h.logs())
    h.settle()
    assert spu_delay(h) == expected_us(0.5, 23.976 / 25, 1.44)
    assert h.S.state.delay.bias == 0
    assert h.requests() == []  # remembered, no new sync


def test_delay_restarted_intf_does_not_take_its_old_correction_as_bias(h, runtime, tmp_path):
    start_delay(h, (None, None, 1.0, 3.0), t0=10)
    h.select("spu-delay", 3_200_000)  # the user adds 0.2 s
    h.tick()
    assert h.S.state.delay.bias == 200_000
    # the intf restarts (VLC keeps playing the same input and its variables)
    h.S.reset()
    h.tick()
    h.settle()
    h.finish(h.requests()[0]["id"], **seg_status((None, None, 1.0, 3.0)))
    h.tick()
    assert h.S.state.delay.bias == 200_000  # the user's part, not 3.2 s
    assert spu_delay(h) == 3_200_000
    assert any("found an earlier correction" in line for line in h.logs())


def test_delay_intf_exit_restores_user_delay(h):
    start_delay(h, (None, None, 1.0, 2.0), t0=10)
    h.select("spu-delay", 2_100_000)  # pressed H twice just before closing
    h.S.shutdown()  # what run() does when VLC interrupts mwait()
    assert spu_delay(h) == 100_000
    assert h.selected("subsync-delay") == ""
    assert any("stop (exit)" in line for line in h.logs())
    assert h.intf_state()["state"] == "stopped"


def test_run_calls_shutdown(runtime, tmp_path):
    src = INTF.read_text(encoding="utf-8")
    run = src[src.index("function M.run()") :]
    assert "M.shutdown()" in run[: run.index("\nend\n")]


def test_delay_stop_keeps_last_user_change_of_the_tick(h):
    start_delay(h, (None, None, 1.0, 2.0), t0=10)
    # G pressed (-50 ms) and the track switched within the same tick
    h.select("spu-delay", 1_950_000)
    h.select("spu-es", 21)
    h.tick()
    assert spu_delay(h) == -50_000


def test_delay_replace_keeps_last_user_change(h):
    start_delay(h, (None, None, 1.0, 2.0), t0=10)
    h.select("spu-delay", 2_050_000)
    # a forced re-sync of the same combination finishes before the next tick
    h.control(auto=1, sync_now=1, sync_mode="delay")
    h.tick(2)
    req = h.requests()[0]
    h.select("spu-delay", 2_100_000)
    h.finish(req["id"], **seg_status((None, None, 1.0, 4.0)))
    h.tick()
    assert h.S.state.delay.bias == 100_000
    assert spu_delay(h) == 4_100_000


def test_delay_ignores_sub_millisecond_differences(h):
    start_delay(h, (None, None, 1.0, 2.0), t0=10)
    h.select("spu-delay", 2_000_400)  # rounding somewhere, not the user
    h.tick()
    assert h.S.state.delay.bias == 0
    assert not any("bias now" in line for line in h.logs())


def test_delay_many_segments_and_knots_are_fast(h):
    import time as _time

    n, per = 256, 16  # 4096 knots: the cap
    segs, knots = [], {}
    for i in range(n):
        lo = None if i == 0 else i * 30.0
        hi = None if i == n - 1 else (i + 1) * 30.0
        segs.append((lo, hi, 1.0 + (i % 3) * 0.001, 0.5 * i))
        knots[i] = [(i * 30.0 + j * 2.0, 0.01 * ((j % 5) - 2)) for j in range(per)]
    st = seg_status(*segs, knots=knots)
    parsed = h.S.parse_segments(h.lua.table_from(st))
    assert parsed is not None and len(parsed) == n
    t0 = _time.perf_counter()
    for k in range(2000):
        h.S.delay_at(parsed, (k * 3.7) % (n * 30.0))
    h.S.min_delay(parsed, n * 30.0)
    assert _time.perf_counter() - t0 < 2.0
    # more than the knot cap: the extra knots are ignored, the segments kept
    st["seg5_knots"] = ";".join(f"{150 + j * 0.01:.3f}:0.0100" for j in range(50))
    parsed = h.S.parse_segments(h.lua.table_from(st))
    assert parsed is not None and len(parsed) == n
    total = sum(len(parsed[i].knots) for i in range(1, n + 1))
    assert total <= 4096 and len(parsed[6].knots) == 50
    assert len(parsed[n].knots) == 0
    assert any("more than 4096 knots" in line for line in h.logs())


# ========================================================================== extension


@pytest.fixture
def ext(runtime, tmp_path):
    return Harness(runtime, tmp_path, EXT)


def write_intf_state(
    hh: Harness, age=0, state="idle", last_result="", modes="fast,thorough,exhaustive"
):
    data = {
        "time": int(time.time()) - age,
        "state": state,
        "message": "",
        "last_result": last_result,
        "auto": 1,
    }
    if modes is not None:  # None: an intf from before sync modes
        data["modes"] = modes
    write_kv(hh.q / "intf_state", data)


def test_ext_descriptor_and_menu(ext):
    d = ext.lua.eval("descriptor()")
    assert d.title == "SubSync – automatic subtitle sync"
    assert list(d.capabilities.values()) == ["menu", "input-listener"]
    ext.lua.eval("activate()")
    assert (ext.q / "requests").is_dir()
    menu = dict(ext.lua.eval("menu()").items())
    assert menu == {
        1: "Sync subtitles now",
        2: "Sync now (exhaustive)",
        3: "Auto-sync: ON",
        4: "Status…",
        6: "Experimental: no extra track (live delay): OFF",
    }


def test_ext_toggle_auto_writes_control(ext):
    write_intf_state(ext)
    ext.control(auto=1, sync_now=3)
    ext.lua.eval("trigger_menu(3)")
    c = parse_kv(ext.q / "control")
    assert c == {"auto": "0", "sync_now": "3"}
    assert dict(ext.lua.eval("menu()").items())[3] == "Auto-sync: OFF"
    ext.lua.eval("trigger_menu(3)")
    assert parse_kv(ext.q / "control")["auto"] == "1"


def test_ext_sync_now_signals_running_intf(ext):
    write_intf_state(ext)
    ext.lua.eval("trigger_menu(1)")
    assert parse_kv(ext.q / "control") == {"auto": "1", "sync_now": "1"}
    ext.lua.eval("trigger_menu(1)")
    assert parse_kv(ext.q / "control")["sync_now"] == "2"
    assert ext.requests() == []


def test_ext_fallback_sync_and_load(ext):
    write_intf_state(ext, age=600, state="idle")  # stale -> intf not running
    ext.heartbeat()
    ext.set_input(audio_sel=11, spu_sel=21)
    ext.lua.eval("trigger_menu(1)")
    reqs = ext.requests()
    assert len(reqs) == 1
    r = reqs[0]
    assert r["media"] == MOVIE_PATH and r["audio_index"] == "1" and r["sub_index"] == "1"
    dlg = ext.mock.last_dialog
    assert dlg.shown and "Load synced result" in [w.text for w in dlg.widgets.values()]
    assert dict(ext.lua.eval("menu()").items())[5] == "Load synced result"
    # not done yet
    dlg.button(dlg, "Load synced result").cb()
    assert ext.added() == []
    out = ext.finish(r["id"])
    dlg.button(dlg, "Load synced result").cb()
    assert ext.added() == [(out, True)]
    html = dlg.widgets[1].text
    assert "Synced subtitles loaded" in html
    # the loaded track is excluded from ordinals next time
    ext.lua.eval("input_changed()")
    assert ext.spu_ids()[-1] in [k for k in ext.E.state.ours.keys()]


def test_ext_sync_now_exhaustive_signals_running_intf(ext):
    write_intf_state(ext)
    ext.lua.eval("trigger_menu(2)")
    c = parse_kv(ext.q / "control")
    assert c == {"auto": "1", "sync_now": "1", "sync_now_mode": "exhaustive"}
    assert any("may take a while" in o for o in ext.osd())
    # a plain "Sync subtitles now" afterwards drops the mode again
    ext.lua.eval("trigger_menu(1)")
    assert parse_kv(ext.q / "control") == {"auto": "1", "sync_now": "2"}
    assert ext.requests() == []


def test_ext_exhaustive_with_old_intf_asks_for_restart(ext):
    """A pre-modes intf (still running after an upgrade) would ignore sync_now_mode."""
    write_intf_state(ext, modes=None)
    ext.control(auto=1, sync_now=4)
    ext.lua.eval("trigger_menu(2)")
    assert parse_kv(ext.q / "control") == {"auto": "1", "sync_now": "4"}  # untouched
    assert ext.requests() == []
    assert "Restart VLC" in ext.mock.last_dialog.widgets[1].text
    assert not any("may take a while" in o for o in ext.osd())
    # the plain item still works through that intf
    ext.lua.eval("trigger_menu(1)")
    assert parse_kv(ext.q / "control")["sync_now"] == "5"


def test_intf_state_advertises_modes(h):
    h.tick()
    assert h.intf_state()["modes"].split(",") == ["fast", "thorough", "exhaustive"]


def test_ext_fallback_exhaustive_request(ext):
    ext.heartbeat()
    ext.set_input()
    ext.lua.eval("trigger_menu(2)")
    reqs = ext.requests()
    assert len(reqs) == 1 and reqs[0]["mode"] == "exhaustive"
    assert "Syncing subtitles (exhaustive, may take a while)…" in ext.osd()
    ext.status(reqs[0]["id"], state="running", progress="0.3", message="Transcribing 3/58")
    assert "exhaustive, may take a while" in ext.E.check_job(False)
    # the default item still writes no mode
    (ext.q / "requests" / f"{reqs[0]['id']}.req").unlink()
    ext.lua.eval("trigger_menu(1)")
    assert "mode" not in ext.requests()[0]


def test_ext_fallback_loads_on_input_changed(ext):
    ext.set_input()
    ext.lua.eval("trigger_menu(1)")
    r = ext.requests()[0]
    out = ext.finish(r["id"])
    ext.lua.eval("input_changed()")
    assert ext.added() == [(out, True)]


def test_ext_fallback_requires_subtitle(ext):
    ext.set_input(spu_sel=-1)
    ext.lua.eval("trigger_menu(1)")
    assert ext.requests() == []
    assert "Select a subtitle track first." in ext.mock.last_dialog.widgets[1].text


def test_ext_status_dialog_hints(ext):
    ext.lua.eval("trigger_menu(4)")
    html = ext.mock.last_dialog.widgets[1].text
    assert "never seen" in html
    assert "vlc-subsync setup" in html and "restart VLC" in html
    ext.heartbeat()
    write_intf_state(ext, state="done", last_result="synced: offset +1.00s")
    ext.lua.eval("trigger_menu(4)")
    html = ext.mock.last_dialog.widgets[1].text
    assert "Helper daemon:</b> running" in html
    assert "interface script running" in html
    assert "synced: offset +1.00s" in html
    assert "restart VLC" not in html
    # previous dialog was deleted when a new one opened
    assert ext.mock.dialogs[1].deleted


def test_ext_toggle_delay_mode_writes_control(ext):
    write_intf_state(ext)
    with open(ext.q / "intf_state", "a", encoding="utf-8") as fh:
        fh.write("sync_modes=track,delay\n")
    ext.control(auto=0, sync_now=5)
    assert dict(ext.lua.eval("menu()").items())[6].endswith("live delay): OFF")
    ext.lua.eval("trigger_menu(6)")
    assert parse_kv(ext.q / "control") == {"auto": "0", "sync_now": "5", "sync_mode": "delay"}
    assert dict(ext.lua.eval("menu()").items())[6] == (
        "Experimental: no extra track (live delay): ON"
    )
    assert "SubSync live delay (experimental) ON" in ext.osd()
    # other control writes keep the choice
    ext.lua.eval("trigger_menu(1)")
    assert parse_kv(ext.q / "control")["sync_mode"] == "delay"
    ext.lua.eval("trigger_menu(3)")
    assert parse_kv(ext.q / "control")["sync_mode"] == "delay"
    ext.lua.eval("trigger_menu(6)")
    assert parse_kv(ext.q / "control")["sync_mode"] == "track"
    assert dict(ext.lua.eval("menu()").items())[6].endswith("live delay): OFF")


def test_ext_delay_menu_reflects_helper_config(ext):
    write_intf_state(ext)
    with open(ext.q / "intf_state", "a", encoding="utf-8") as fh:
        fh.write("sync_mode=delay\nsync_modes=track,delay\n")
    assert dict(ext.lua.eval("menu()").items())[6].endswith("live delay): ON")
    ext.lua.eval("trigger_menu(6)")  # toggling turns it off explicitly
    assert parse_kv(ext.q / "control")["sync_mode"] == "track"


def test_ext_delay_toggle_with_old_intf_asks_for_restart(ext):
    write_intf_state(ext)  # no sync_modes key: an intf from before delay mode
    ext.control(auto=1, sync_now=2)
    ext.lua.eval("trigger_menu(6)")
    assert parse_kv(ext.q / "control") == {"auto": "1", "sync_now": "2"}
    assert "Restart VLC" in ext.mock.last_dialog.widgets[1].text


def test_intf_state_reports_sync_mode(h):
    h.tick()
    st = h.intf_state()
    assert st["sync_mode"] == "track" and st["sync_modes"] == "track,delay"
    h.control(auto=1, sync_now=0, sync_mode="delay")
    h.tick()
    assert h.intf_state()["sync_mode"] == "delay"


def test_luacheck():
    import shutil
    import subprocess

    exe = ROOT / ".cache" / "lua" / "bin" / "luacheck"
    cmd = str(exe) if exe.exists() else shutil.which("luacheck")
    if not cmd:
        pytest.skip("luacheck not installed")
    res = subprocess.run(
        [cmd, "--no-color", "src/vlcsubsync/lua", "tests/lua"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
