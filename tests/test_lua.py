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
        self.messages = None  # kept in every control() write once show_all() ran

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
    def init(self, hold: bool = False):
        # Most tests are about what happens once a sync starts: they skip the wait
        # after a file opens (see the start_hold tests).
        if not hold:
            self.S.START_HOLD_US = 0
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

    def heartbeat(self, age: int = 0, **kv):
        write_kv(
            self.q / "heartbeat",
            {"time": int(time.time()) - age, "pid": 1234, "version": "0.1.0", **kv},
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
        if self.messages and "messages" not in kv:
            kv["messages"] = self.messages
        write_kv(self.q / "control", kv)

    def show_all(self):
        """Turn "Show all messages" on (by default only errors reach the OSD)."""
        self.messages = "all"
        if self.S:
            # what read_control() sets from messages=all; writing the control file
            # here would also set the sync_now baseline
            self.S.state.messages_all = True
        else:
            c = self.q / "control"
            write_kv(c, {**(parse_kv(c) if c.exists() else {}), "messages": "all"})

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
    h.show_all()
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


# ---- the wait after a file opens (START_HOLD_US) ------------------------------------


@pytest.fixture
def hold(runtime, tmp_path):
    hh = Harness(runtime, tmp_path, INTF)
    hh.heartbeat()
    hh.init(hold=True)
    hh.show_all()  # so the tests see which messages would be shown
    return hh


def miss(h, req_id):
    """The helper's answer to a cache_only request without a cached result."""
    h.status(req_id, state="miss", progress="0.000", message="Not cached")
    (h.q / "requests" / f"{req_id}.req").unlink(missing_ok=True)


def test_start_hold_asks_cache_then_waits_ten_seconds(hold):
    h = hold
    h.set_input(audio_sel=11, spu_sel=21)
    h.tick(4)  # past the debounce: only a quiet cache lookup
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["cache_only"] == "1"
    assert reqs[0]["audio_index"] == "1" and reqs[0]["sub_index"] == "1"
    assert reqs[0]["force"] == "0"
    assert not any(o.startswith("Syncing") for o in h.osd())
    assert h.intf_state()["state"] == "waiting"
    miss(h, reqs[0]["id"])
    h.tick(16)  # up to 9.5 s after the file opened
    assert h.requests() == []
    assert h.intf_state()["state"] == "waiting"
    h.tick()  # 10 s: the real request
    reqs = h.requests()
    assert len(reqs) == 1 and "cache_only" not in reqs[0]
    assert reqs[0]["sub_index"] == "1" and reqs[0]["force"] == "0"
    assert "Syncing subtitles…" in h.osd()
    out = h.finish(reqs[0]["id"])
    h.tick(2)
    assert h.added() == [(out, True)]


def test_start_hold_cache_hit_applies_at_once(hold):
    h = hold
    h.set_input()
    h.tick(4)
    probe = h.requests()[0]
    assert probe["cache_only"] == "1"
    out = h.finish(probe["id"])  # the helper had a synced result
    h.tick()
    assert h.added() == [(out, True)]
    h.tick(2)
    assert h.selected("spu-es") == h.spu_ids()[-1]
    assert "Subtitles synced: offset +2.35s, drift +4.1%" in h.osd()
    h.tick(30)
    assert h.requests() == []


def test_start_hold_user_track_change_syncs_without_waiting(hold):
    h = hold
    h.set_input(spu_sel=20)
    h.tick(6)
    miss(h, h.requests()[0]["id"])
    h.tick(2)  # 4 s in: the user picks another subtitle track
    h.select("spu-es", 21)
    h.tick(4)  # debounce only
    reqs = h.requests()
    assert len(reqs) == 1 and "cache_only" not in reqs[0] and reqs[0]["sub_index"] == "1"


def test_start_hold_vlc_initial_picks_do_not_end_it(hold):
    h = hold
    # VLC selects the subtitle track a moment after the audio one
    h.set_input(spu_sel=-1)
    h.tick(2)
    h.select("spu-es", 20)
    h.tick(4)
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["cache_only"] == "1"
    miss(h, reqs[0]["id"])
    h.tick(8)  # 7 s in
    assert h.requests() == []


def test_start_hold_tracks_already_missed_are_not_asked_again(hold):
    h = hold
    h.S.START_SETTLE_US = 5_000_000  # a slow VLC still picking tracks
    h.set_input(spu_sel=20)
    h.tick(4)
    miss(h, h.requests()[0]["id"])
    h.tick()
    h.select("spu-es", -1)
    h.tick()
    h.select("spu-es", 20)  # 3 s in, back to the tracks that already missed
    h.tick(14)  # up to 9.5 s
    assert h.requests() == []
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1 and "cache_only" not in reqs[0]


def test_start_hold_late_cache_answer_syncs_at_once(hold):
    h = hold
    h.set_input()
    h.tick(4)
    probe = h.requests()[0]
    h.tick(20)  # the helper is slow to answer; the hold is over
    assert h.requests() == [probe]
    miss(h, probe["id"])
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1 and "cache_only" not in reqs[0]


def test_start_hold_sync_now_does_not_wait(hold):
    h = hold
    h.control(sync_now=1)  # baseline
    h.set_input()
    h.tick()
    h.control(sync_now=2)
    h.tick()
    reqs = h.requests()
    assert len(reqs) == 1 and "cache_only" not in reqs[0]
    assert "Syncing subtitles…" in h.osd()


def test_start_hold_restarts_for_the_next_file(hold):
    h = hold
    h.set_input()
    h.tick(4)
    miss(h, h.requests()[0]["id"])
    h.tick(20)
    first = h.requests()[0]
    h.finish(first["id"])
    h.tick(3)
    # next playlist item: waits again
    h.set_input(uri="file:///media/next.mkv")
    h.tick(4)
    reqs = h.requests()
    assert len(reqs) == 1 and reqs[0]["cache_only"] == "1"
    assert reqs[0]["media"] == "/media/next.mkv"


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
    h.show_all()
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


def test_default_osd_shows_only_errors(h):
    """Progress and "Subtitles synced" stay off screen unless "Show all messages" is on."""
    h.set_input()
    h.settle()
    req = h.requests()[0]
    h.status(req["id"], state="running", progress="0.42", message="Transcribing 3/10")
    h.tick(8)
    out = h.finish(req["id"])
    h.tick(2)
    assert h.added() == [(out, True)]
    assert h.osd() == []
    assert any("OSD (hidden, messages=errors): Subtitles synced" in line for line in h.logs())
    assert h.intf_state()["state"] == "done"  # the Status dialog still has it all
    # a result that does not fit is an error: shown
    h.select("spu-es", 21)
    h.settle()
    req = h.requests()[0]
    h.finish(req["id"], message="not synced: too few transcript matches (3)", applied=0)
    h.tick(2)
    assert h.osd() == [
        "SubSync: not synced (not synced: too few transcript matches (3)) – keeping original timing"
    ]


def test_show_all_messages_follows_control(h):
    h.control(auto=1, sync_now=0, messages="all")
    h.set_input()
    h.settle()
    assert "Syncing subtitles…" in h.osd()
    assert any("on-screen messages: all" in line for line in h.logs())
    n = len(h.osd())
    h.control(auto=1, sync_now=0, messages="errors")
    h.select("spu-es", 21)
    h.settle()
    assert h.requests()[0]["sub_index"] == "1"  # superseded the first one
    assert len(h.osd()) == n
    h.control(auto=1, sync_now=0, messages="all")
    h.select("spu-es", 20)
    h.settle()
    assert h.requests()[0]["sub_index"] == "0"
    assert len(h.osd()) == n + 1


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
    h.show_all()
    h.set_input()
    h.settle()
    assert "mode" not in h.requests()[0]
    assert "Syncing subtitles…" in h.osd()


def test_sync_now_exhaustive_writes_mode_and_warns(h):
    h.show_all()
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
    hh.show_all()
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
    hh.show_all()
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


# ========================================================================== extension


@pytest.fixture
def ext(runtime, tmp_path):
    return Harness(runtime, tmp_path, EXT)


@pytest.mark.parametrize("script", [INTF, EXT], ids=["intf", "ext"])
def test_loads_without_rawget(runtime, script):
    # VLC's extension loader has no rawget; the script used to die on its last line
    # (the test hook), so VLC listed it in the Addons Manager but never loaded it.
    lua = importlib.import_module(f"lupa.{runtime}").LuaRuntime()
    lua.execute("rawget = nil; vlc = setmetatable({}, {__index = function() return {} end})")
    if script == EXT:
        lua.execute(f"dofile({str(script)!r})")
        assert lua.eval("descriptor().shortdesc") == "SubSync"
    else:
        lua.execute("SUBSYNC_TEST = true")
        lua.execute(f"dofile({str(script)!r})")
        assert lua.eval("type(subsync)") == "table"


def test_ext_descriptor_in_vlc_scan_state(runtime):
    # VLC reads descriptor() (with the icon) in a Lua state with no libraries at all.
    lua = importlib.import_module(f"lupa.{runtime}").LuaRuntime(encoding=None)
    lua.execute(b"for k in pairs(_G) do if k ~= '_G' then _G[k] = nil end end")
    lua.execute(EXT.read_bytes())
    d = lua.eval(b"descriptor()")
    assert d[b"shortdesc"] == b"SubSync"  # also the View menu label
    icon = d[b"icon"]
    assert icon.startswith(b"\x89PNG\r\n\x1a\n") and icon.endswith(b"IEND\xaeB`\x82")


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
        6: "Show all messages: OFF",
        7: "Use cached results: ON",
        8: "Delete cached results",
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
    ext.show_all()
    write_intf_state(ext)
    ext.lua.eval("trigger_menu(2)")
    c = parse_kv(ext.q / "control")
    assert c == {"auto": "1", "sync_now": "1", "sync_now_mode": "exhaustive", "messages": "all"}
    assert any("may take a while" in o for o in ext.osd())
    # a plain "Sync subtitles now" afterwards drops the mode again
    ext.lua.eval("trigger_menu(1)")
    assert parse_kv(ext.q / "control") == {"auto": "1", "sync_now": "2", "messages": "all"}
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
    ext.show_all()
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


def test_ext_show_all_messages_toggle(ext):
    write_intf_state(ext)
    ext.heartbeat(cache="on")
    ext.control(auto=1, sync_now=3)
    # default: the informational OSD of other items stays hidden
    ext.lua.eval("trigger_menu(3)")
    ext.lua.eval("trigger_menu(1)")
    assert ext.osd() == []
    ext.lua.eval("trigger_menu(4)")
    assert "On-screen messages:</b> errors only" in ext.mock.last_dialog.widgets[1].text
    ext.lua.eval("trigger_menu(6)")
    assert parse_kv(ext.q / "control") == {"auto": "0", "sync_now": "4", "messages": "all"}
    assert dict(ext.lua.eval("menu()").items())[6] == "Show all messages: ON"
    assert ext.osd() == ["SubSync: all messages ON"]
    ext.lua.eval("trigger_menu(4)")
    assert "On-screen messages:</b> all" in ext.mock.last_dialog.widgets[1].text
    # other control writes keep the choice
    for item in (1, 2, 3, 7):
        ext.lua.eval(f"trigger_menu({item})")
        assert parse_kv(ext.q / "control")["messages"] == "all"
    assert "SubSync: sync requested" in ext.osd()
    ext.lua.eval("trigger_menu(6)")
    assert parse_kv(ext.q / "control")["messages"] == "errors"
    assert dict(ext.lua.eval("menu()").items())[6] == "Show all messages: OFF"
    n = len(ext.osd())
    ext.lua.eval("trigger_menu(1)")
    assert len(ext.osd()) == n


def test_ext_cache_toggle_writes_control(ext):
    ext.show_all()
    write_intf_state(ext)
    ext.heartbeat(cache="on")
    ext.control(auto=0, sync_now=5)
    ext.lua.eval("trigger_menu(7)")
    assert parse_kv(ext.q / "control") == {
        "auto": "0",
        "sync_now": "5",
        "cache": "off",
        "messages": "all",
    }
    assert dict(ext.lua.eval("menu()").items())[7] == "Use cached results: OFF"
    assert "SubSync cached results OFF" in ext.osd()
    ext.lua.eval("trigger_menu(4)")
    assert "Cached results:</b> not used" in ext.mock.last_dialog.widgets[1].text
    # other control writes keep the choice
    for item in (1, 2, 3, 6):
        ext.lua.eval(f"trigger_menu({item})")
        assert parse_kv(ext.q / "control")["cache"] == "off"
    ext.lua.eval("trigger_menu(7)")
    assert parse_kv(ext.q / "control")["cache"] == "on"
    assert dict(ext.lua.eval("menu()").items())[7] == "Use cached results: ON"


def test_ext_cache_menu_reflects_helper_config(ext):
    ext.heartbeat(cache="off")  # cache=off in config.ini
    assert dict(ext.lua.eval("menu()").items())[7] == "Use cached results: OFF"
    ext.lua.eval("trigger_menu(7)")  # toggling turns it on explicitly
    assert parse_kv(ext.q / "control")["cache"] == "on"


def test_ext_cache_toggle_accepts_helper_spellings(ext):
    ext.heartbeat(cache="on")
    ext.control(auto=1, sync_now=0, cache=" 0 ")  # hand-written; the helper reads it as off
    assert dict(ext.lua.eval("menu()").items())[7] == "Use cached results: OFF"


def test_ext_cache_toggle_without_helper_explains(ext):
    ext.lua.eval("trigger_menu(7)")
    assert parse_kv(ext.q / "control")["cache"] == "off"
    assert "helper is not running" in ext.mock.last_dialog.widgets[1].text


def test_ext_delete_cached_results(ext):
    ext.show_all()
    ext.heartbeat(cache="on")
    ext.lua.eval("trigger_menu(8)")
    assert "time" in parse_kv(ext.q / "clear_cache")
    assert "SubSync: deleting cached results" in ext.osd()
    (ext.q / "clear_cache").unlink()
    ext.heartbeat(age=600)  # helper gone: it deletes them once it starts
    ext.lua.eval("trigger_menu(8)")
    assert (ext.q / "clear_cache").exists()
    assert "vlc-subsync clear-cache" in ext.mock.last_dialog.widgets[1].text


def test_ext_cache_items_with_old_helper_ask_for_restart(ext):
    ext.heartbeat()  # no cache key: a helper from before these items
    ext.control(auto=1, sync_now=2)
    ext.lua.eval("trigger_menu(7)")
    ext.lua.eval("trigger_menu(8)")
    assert parse_kv(ext.q / "control") == {"auto": "1", "sync_now": "2"}
    assert not (ext.q / "clear_cache").exists()
    assert "Restart VLC" in ext.mock.last_dialog.widgets[1].text


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
