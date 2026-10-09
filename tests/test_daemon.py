from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlcsubsync import daemon as D
from vlcsubsync import protocol as P


def wait_for(cond, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cond()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("timeout waiting for condition")


class FakeRunner:
    """Writes a fake SRT; optionally blocks until released."""

    def __init__(self, fail_for=(), block=False, applied=True, mapping_segments="default"):
        # the engine always returns its mapping when applied; None = an engine
        # from before mappings were sent
        if mapping_segments == "default":
            mapping_segments = [P.MapSegment(None, None, 1.0417, 2.35)]
        self.mapping_segments = mapping_segments
        self.calls: list[D.JobSpec] = []
        self.fail_for = set(fail_for)
        self.block = block
        self.applied = applied
        self.release = threading.Event()
        self.started = threading.Event()

    def __call__(self, spec, progress):
        self.calls.append(spec)
        self.started.set()
        progress(0.1, "Decoding audio")
        if self.block:
            while not self.release.wait(0.02):
                progress(0.2, "Transcribing 1/10")
        if spec.media in self.fail_for:
            raise RuntimeError("boom: engine failure")
        progress(0.9, "Fitting")
        Path(spec.output_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
        return SimpleNamespace(
            output_path=spec.output_path,
            method="whisper",
            offset=2.35,
            scale=1.0417,
            segments=1,
            confidence=0.93,
            anchors=40,
            applied=self.applied,
            message="offset +2.35s, drift +4.17%",
            mapping_segments=self.mapping_segments,
        )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLC_SUBSYNC_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VLC_SUBSYNC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("VLC_SUBSYNC_LOG_DIR", str(tmp_path / "log"))
    queue = tmp_path / "vlc" / "subsync"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    return SimpleNamespace(tmp=tmp_path, queue=queue, media_dir=media_dir)


def make_media(env, name="movie.mkv", content=b"fake media"):
    p = env.media_dir / name
    p.write_bytes(content)
    return str(p)


def external_resolver(req):
    return (
        D.ResolvedSource("external", path=req.media + ".srt")
        if req.sub_path
        else (D.ResolvedSource("embedded", index=req.sub_index or 0))
    )


class Harness:
    def __init__(self, env, runner, resolver=external_resolver, **kw):
        config_loader = kw.pop(
            "config_loader", lambda: SimpleNamespace(model_en="base.en", model_multi="base")
        )
        self.daemon = D.Daemon(
            [env.queue],
            use_default_queues=False,
            runner=runner,
            resolver=resolver,
            config_loader=config_loader,
            cache_dir=env.tmp / "cache" / "results",
            lock_path=env.tmp / "state" / "daemon.lock",
            poll_interval=0.02,
            heartbeat_interval=0.1,
            status_interval=0.0,
            **kw,
        )
        self.queue = env.queue
        self.thread = threading.Thread(target=self.daemon.run)
        self.rc = None

    def __enter__(self):
        self.thread.start()
        wait_for(lambda: (self.queue / "requests").is_dir())
        return self

    def __exit__(self, *exc):
        self.daemon.request_stop()
        self.thread.join(15)
        assert not self.thread.is_alive()

    def submit(self, req_id, media, **kw):
        P.write_request(self.queue, P.Request(id=req_id, media=media, **kw))

    def status(self, req_id):
        return P.read_status(self.queue, req_id)

    def wait_state(self, req_id, *states, timeout=10):
        return wait_for(
            lambda: (s := self.status(req_id)) is not None and s.state in states and s,
            timeout,
        )


def test_end_to_end_done(env):
    media = make_media(env)
    runner = FakeRunner()
    seen_states = []
    with Harness(env, runner) as h:
        h.submit("1_1", media, audio_index=1, sub_index=0)

        def watch():
            s = h.status("1_1")
            if s and (not seen_states or seen_states[-1] != s.state):
                seen_states.append(s.state)
            return s is not None and s.state == "done"

        wait_for(watch)
        st = h.status("1_1")
        assert st.applied is True and st.method == "whisper"
        assert st.offset == pytest.approx(2.35)
        assert st.confidence == pytest.approx(0.93)
        assert st.progress == 1.0
        assert Path(st.output) == env.queue / "out" / "1_1.srt"
        assert Path(st.output).is_file()
        assert not (env.queue / "requests" / "1_1.req").exists()
        hb = P.read_heartbeat(env.queue)
        assert hb.pid == os.getpid() and hb.age() < 5
        assert runner.calls[0].audio_index == 1
        assert runner.calls[0].source == D.ResolvedSource("embedded", index=0)
    assert seen_states[-1] == "done"
    # heartbeat removed on graceful shutdown
    assert P.read_heartbeat(env.queue) is None


def test_running_status_with_progress(env):
    media = make_media(env)
    runner = FakeRunner(block=True)
    with Harness(env, runner) as h:
        h.submit("r_1", media, sub_index=0)
        st = h.wait_state("r_1", "running")
        wait_for(lambda: h.status("r_1").progress >= 0.1)
        st = h.status("r_1")
        assert st.state == "running"
        assert st.message.startswith(("Transcribing", "Decoding"))
        runner.release.set()
        h.wait_state("r_1", "done")


def test_error_does_not_kill_daemon(env):
    bad = make_media(env, "bad.mkv")
    good = make_media(env, "good.mkv")
    runner = FakeRunner(fail_for={bad})
    with Harness(env, runner) as h:
        h.submit("e_1", bad, sub_index=0)
        st = h.wait_state("e_1", "error")
        assert "boom" in st.message
        h.submit("e_2", good, sub_index=0)
        assert h.wait_state("e_2", "done").applied


def test_resolver_error_and_missing_media(env):
    media = make_media(env)

    def resolver(req):
        raise D.JobError("No subtitle track selected")

    with Harness(env, FakeRunner(), resolver=resolver) as h:
        h.submit("n_1", media)
        assert h.wait_state("n_1", "error").message == "No subtitle track selected"
        h.submit("n_2", str(env.media_dir / "missing.mkv"), sub_index=0)
        assert "not found" in h.wait_state("n_2", "error").message


def test_bad_request_file(env):
    with Harness(env, FakeRunner()) as h:
        (env.queue / "requests" / "bad_1.req").write_text("id=bad_1\n")  # no media
        st = h.wait_state("bad_1", "error")
        assert "Bad request" in st.message
        wait_for(lambda: not (env.queue / "requests" / "bad_1.req").exists())


def test_expired_request(env):
    media = make_media(env)
    runner = FakeRunner()
    env.queue.joinpath("requests").mkdir(parents=True)
    path = P.write_request(env.queue, P.Request(id="old_1", media=media, sub_index=0))
    old = time.time() - 2 * D.REQUEST_EXPIRY_SECONDS
    os.utime(path, (old, old))
    with Harness(env, runner) as h:
        assert h.wait_state("old_1", "error").message == "Request expired"
    assert runner.calls == []


def test_supersede_queued_and_cancel_running(env):
    m1 = make_media(env, "a.mkv")
    m2 = make_media(env, "b.mkv")
    runner = FakeRunner(block=True)
    with Harness(env, runner) as h:
        h.submit("s_1", m1, sub_index=0)
        runner.started.wait(5)
        h.wait_state("s_1", "running")
        # two queued requests for the same media: the newer supersedes the older
        h.submit("s_2", m2, sub_index=0)
        h.wait_state("s_2", "queued")
        h.submit("s_3", m2, sub_index=1)
        st = h.wait_state("s_2", "error")
        assert "Superseded" in st.message
        # a newer request for the running media with different tracks cancels it
        h.submit("s_4", m1, sub_index=1)
        st = h.wait_state("s_1", "error")
        assert "Superseded" in st.message
        runner.release.set()
        h.wait_state("s_4", "done")
        h.wait_state("s_3", "done")
    run_ids = [c.id for c in runner.calls]
    assert "s_2" not in run_ids
    assert run_ids[0] == "s_1"


def test_same_params_does_not_cancel_running(env):
    m1 = make_media(env)
    runner = FakeRunner(block=True)
    with Harness(env, runner) as h:
        h.submit("p_1", m1, sub_index=0)
        h.wait_state("p_1", "running")
        h.submit("p_2", m1, sub_index=0)
        h.wait_state("p_2", "queued")
        time.sleep(0.2)
        assert h.status("p_1").state == "running"
        runner.release.set()
        h.wait_state("p_1", "done")
        st = h.wait_state("p_2", "done")
        assert Path(st.output).name == "p_2.srt"
    assert [c.id for c in runner.calls] == ["p_1"]  # p_2 came from the cache


@pytest.mark.parametrize(
    "first,second,cancels",
    [
        ({}, {"mode": "exhaustive"}, True),  # "Sync now (exhaustive)" during a fast run
        ({"mode": "exhaustive"}, {"force": True}, True),  # forced plain "Sync now"
        ({"mode": "exhaustive"}, {}, False),  # an automatic request waits
        ({"mode": "exhaustive"}, {"mode": "exhaustive"}, False),
    ],
)
def test_mode_switch_cancels_running(env, first, second, cancels):
    m1 = make_media(env)
    runner = FakeRunner(block=True)
    with Harness(env, runner) as h:
        h.submit("m_1", m1, sub_index=0, **first)
        h.wait_state("m_1", "running")
        h.submit("m_2", m1, sub_index=0, **second)
        h.wait_state("m_2", "queued", "running", "done")
        if cancels:
            assert "Superseded" in h.wait_state("m_1", "error").message
        else:
            time.sleep(0.2)
            assert h.status("m_1").state == "running"
        runner.release.set()
        h.wait_state("m_2", "done")


def test_cache_hit_and_force(env):
    media = make_media(env)
    runner = FakeRunner()
    with Harness(env, runner) as h:
        h.submit("c_1", media, sub_index=0)
        h.wait_state("c_1", "done")
        h.submit("c_2", media, sub_index=0)
        st = h.wait_state("c_2", "done")
        assert len(runner.calls) == 1
        assert st.applied and st.offset == pytest.approx(2.35)
        assert Path(st.output) == env.queue / "out" / "c_2.srt"
        assert Path(st.output).read_text().startswith("1\n")
        # different audio track -> different key
        h.submit("c_3", media, sub_index=0, audio_index=1)
        h.wait_state("c_3", "done")
        assert len(runner.calls) == 2
        # force bypasses the cache
        h.submit("c_4", media, sub_index=0, force=True)
        h.wait_state("c_4", "done")
        assert len(runner.calls) == 3
        # modified media -> cache miss
        Path(media).write_bytes(b"different content!")
        h.submit("c_5", media, sub_index=0)
        h.wait_state("c_5", "done")
        assert len(runner.calls) == 4


def test_cache_key_includes_mode(env):
    from vlcsubsync.config import Config

    media = make_media(env)
    d = D.Daemon(
        [env.queue], use_default_queues=False, runner=FakeRunner(),
        resolver=external_resolver, cache_dir=env.tmp / "c",
        lock_path=env.tmp / "state" / "daemon.lock",
    )  # fmt: skip
    src = D.ResolvedSource("embedded", index=0)

    def key(cfg_mode="fast", req_mode=""):
        job = D._Job(P.Request(id="k", media=media, sub_index=0, mode=req_mode), env.queue)
        return d.cache_key(job, src, Config(mode=cfg_mode))

    keys = {m: key(m) for m in ("fast", "thorough", "exhaustive")}
    assert len(set(keys.values())) == 3
    # the request's mode wins over the config's; an unknown config mode means fast
    assert key("fast", "exhaustive") == keys["exhaustive"]
    assert key("bogus") == keys["fast"]
    # lookup order: most thorough first, never a less thorough result
    job = D._Job(P.Request(id="k", media=media, sub_index=0, mode="thorough"), env.queue)
    assert d.cache_lookup_keys(job, src, Config()) == [
        ("exhaustive", keys["exhaustive"]),
        ("thorough", keys["thorough"]),
    ]


def test_request_mode_reaches_runner_and_cache(env):
    from vlcsubsync.config import Config

    media = make_media(env)
    runner = FakeRunner()
    with Harness(env, runner) as h:
        h.daemon.config_loader = lambda: Config(mode="fast")
        h.submit("m_1", media, sub_index=0)
        h.wait_state("m_1", "done")
        assert runner.calls[-1].config.mode == "fast"
        # an exhaustive request does not reuse the fast result
        h.submit("m_2", media, sub_index=0, mode="exhaustive")
        h.wait_state("m_2", "done")
        assert len(runner.calls) == 2
        assert runner.calls[-1].config.mode == "exhaustive"
        # ... but a later fast or thorough request reuses the exhaustive one
        h.submit("m_3", media, sub_index=0, mode="thorough")
        h.wait_state("m_3", "done")
        h.submit("m_4", media, sub_index=0)
        h.wait_state("m_4", "done")
        assert len(runner.calls) == 2
        # a thorough result does not satisfy an exhaustive request
        h.submit("m_5", media, sub_index=1, mode="thorough")
        h.wait_state("m_5", "done")
        h.submit("m_6", media, sub_index=1, mode="exhaustive")
        h.wait_state("m_6", "done")
        assert [c.config.mode for c in runner.calls[2:]] == ["thorough", "exhaustive"]


def test_unapplied_result_not_served_across_modes(env):
    """An unapplied exhaustive result (e.g. Whisper windows lost to CUDA OOM, then the
    VAD fallback) must not block a fast sync, but still answers exhaustive requests."""
    from vlcsubsync.config import Config

    media = make_media(env)
    runner = FakeRunner(applied=False)
    with Harness(env, runner) as h:
        h.daemon.config_loader = lambda: Config(mode="fast")
        h.submit("u_1", media, sub_index=0, mode="exhaustive")
        assert h.wait_state("u_1", "done").applied is False
        # same mode: the cached unapplied result is reused (hopeless work not redone)
        h.submit("u_2", media, sub_index=0, mode="exhaustive")
        assert h.wait_state("u_2", "done").applied is False
        assert len(runner.calls) == 1
        # fast and thorough: not answered by the unapplied exhaustive result
        runner.applied = True
        h.submit("u_3", media, sub_index=0)
        assert h.wait_state("u_3", "done").applied is True
        assert [c.config.mode for c in runner.calls] == ["exhaustive", "fast"]
        h.submit("u_4", media, sub_index=0, mode="thorough")
        h.wait_state("u_4", "done")
        assert [c.config.mode for c in runner.calls][-1] == "thorough"
        assert len(runner.calls) == 3
        # an applied exhaustive result does answer the cheaper modes
        h.submit("u_5", media, sub_index=1, mode="exhaustive")
        h.wait_state("u_5", "done")
        h.submit("u_6", media, sub_index=1)
        assert h.wait_state("u_6", "done").applied is True
        assert len(runner.calls) == 4


def test_forced_resync_wins_over_other_modes(env):
    """A forced fast re-sync replaces an older exhaustive result for later opens."""
    from vlcsubsync.config import Config

    media = make_media(env)
    runner = FakeRunner()
    cache = env.tmp / "cache" / "results"
    with Harness(env, runner) as h:
        h.daemon.config_loader = lambda: Config(mode="fast")
        h.submit("f_1", media, sub_index=0, mode="exhaustive")
        h.wait_state("f_1", "done")
        wait_for(lambda: len(list(cache.glob("*.meta"))) == 1)  # stored after "done"
        # user clicks "Sync now" (force=1) in fast mode with a different result
        runner.applied = False
        h.submit("f_2", media, sub_index=0, force=True)
        assert h.wait_state("f_2", "done").applied is False
        assert len(runner.calls) == 2

        # only the forced fast result is left; the next (fast) open gets it
        # (the cache is updated right after "done" is written)
        def only_forced_entry_left():
            metas = [P.read_kv(m) for m in cache.glob("*.meta")]
            return len(metas) == 1 and metas[0].get("applied") == "0"

        wait_for(only_forced_entry_left)
        assert len(list(cache.glob("*.srt"))) == 1
        h.submit("f_3", media, sub_index=0)
        assert h.wait_state("f_3", "done").applied is False
        assert len(runner.calls) == 2
        # an exhaustive request now runs again (its old entry is gone)
        h.submit("f_4", media, sub_index=0, mode="exhaustive")
        h.wait_state("f_4", "done")
        assert len(runner.calls) == 3
        # a non-forced run does not drop other modes' entries
        wait_for(lambda: len(list(cache.glob("*.meta"))) == 2)  # stored after "done"


def test_stale_lock_is_taken_over(env):
    lock = env.tmp / "state" / "daemon.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("999999\n")  # dead pid, nobody holds the OS lock
    assert D.daemon_running_pid(lock) is None
    il = D.InstanceLock(lock)
    assert il.acquire()
    assert D.InstanceLock.read_pid(lock) == os.getpid()
    il.release()


def test_live_lock_blocks_second_instance(env):
    lock = env.tmp / "state" / "daemon.lock"
    first = D.InstanceLock(lock)
    assert first.acquire()
    try:
        assert D.daemon_running_pid(lock) == os.getpid()
        d = D.Daemon([env.queue], use_default_queues=False, runner=FakeRunner(), lock_path=lock)
        assert d.run() == D.ALREADY_RUNNING
    finally:
        first.release()
    second = D.InstanceLock(lock)
    assert second.acquire()
    second.release()


def test_housekeeping_removes_old_files(env):
    d = D.Daemon(
        [env.queue],
        use_default_queues=False,
        runner=FakeRunner(),
        cache_dir=env.tmp / "cache",
    )
    d.scan_queue_dirs()
    old = time.time() - 8 * 86400
    files = {
        "old_status": env.queue / "jobs" / "a.status",
        "old_out": env.queue / "out" / "a.srt",
        "new_out": env.queue / "out" / "b.srt",
        "old_cache": env.tmp / "cache" / "k.meta",
        "old_tmp": env.queue / "jobs" / ".x.status.abc.tmp",
    }
    (env.tmp / "cache").mkdir()
    for p in files.values():
        p.write_text("x")
    for key in ("old_status", "old_out", "old_cache"):
        os.utime(files[key], (old, old))
    t = time.time() - 7200
    os.utime(files["old_tmp"], (t, t))
    assert d.housekeeping() == 4
    assert files["new_out"].exists()
    assert not files["old_out"].exists()


def test_scan_queue_dirs_creates_layout(tmp_path, monkeypatch):
    vlc_data = tmp_path / "vlcdata"
    vlc_data.mkdir()
    absent = tmp_path / "novlc" / "subsync"
    monkeypatch.setenv(
        "VLC_SUBSYNC_QUEUE_DIRS", os.pathsep.join([str(vlc_data / "subsync"), str(absent)])
    )
    d = D.Daemon(runner=FakeRunner())
    found = d.scan_queue_dirs()
    assert found == [vlc_data / "subsync"]
    for sub in ("requests", "jobs", "out"):
        assert (vlc_data / "subsync" / sub).is_dir()
    assert not absent.exists()


def test_default_queue_dirs_per_platform(tmp_path):
    home = tmp_path
    linux = D.default_queue_dirs("linux", home, env={})
    assert linux == [
        home / ".local/share/vlc/subsync",
        home / "snap/vlc/current/.local/share/vlc/subsync",
        home / ".var/app/org.videolan.VLC/data/vlc/subsync",
    ]
    assert D.default_queue_dirs("darwin", home, env={}) == [
        home / "Library/Application Support/org.videolan.vlc/subsync"
    ]
    assert D.default_queue_dirs("win32", home, env={"APPDATA": str(tmp_path / "AD")}) == [
        tmp_path / "AD" / "vlc" / "subsync"
    ]
    assert D.default_queue_dirs("linux", home, env={"XDG_DATA_HOME": "/x"})[0] == Path(
        "/x/vlc/subsync"
    )


# ------------------------------------------------------------------ source resolution


def test_resolve_explicit_sub_path_wins(tmp_path):
    sub = tmp_path / "x.srt"
    sub.write_text("")
    src = D.resolve_source(
        "/m.mkv", 0, str(sub), count_embedded=lambda m: 3, find_sidecars=lambda m: []
    )
    assert src == D.ResolvedSource("external", path=str(sub))
    with pytest.raises(D.JobError):
        D.resolve_source("/m.mkv", 0, str(tmp_path / "nope.srt"))


def test_resolve_embedded_then_sidecars():
    kw = dict(count_embedded=lambda m: 2, find_sidecars=lambda m: ["/a.srt", "/a.en.srt"])
    assert D.resolve_source("/m", 0, **kw) == D.ResolvedSource("embedded", index=0)
    assert D.resolve_source("/m", 1, **kw) == D.ResolvedSource("embedded", index=1)
    assert D.resolve_source("/m", 2, **kw) == D.ResolvedSource("external", path="/a.srt")
    assert D.resolve_source("/m", 3, **kw) == D.ResolvedSource("external", path="/a.en.srt")
    with pytest.raises(D.JobError, match="not found"):
        D.resolve_source("/m", 4, **kw)
    with pytest.raises(D.JobError, match="No subtitle"):
        D.resolve_source("/m", None, **kw)


def test_resolve_probe_failure_is_job_error():
    def bad(m):
        raise OSError("cannot open")

    with pytest.raises(D.JobError, match="Cannot read media"):
        D.resolve_source("/m", 0, count_embedded=bad, find_sidecars=lambda m: [])


def test_count_embedded_uses_media_probe(monkeypatch):
    import vlcsubsync.media as media

    monkeypatch.setattr(media, "probe", lambda p: SimpleNamespace(subtitles=[1, 2, 3], audio=[1]))
    assert D.count_embedded_subtitles("/x.mkv") == 3


def test_external_output_keeps_format(env):
    media = make_media(env)
    sub = Path(media).with_suffix(".ass")
    sub.write_text("[Script Info]\n")
    runner = FakeRunner()
    with Harness(env, runner, resolver=lambda r: D.ResolvedSource("external", path=str(sub))) as h:
        h.submit("f_1", media, sub_index=0)
        st = h.wait_state("f_1", "done")
    assert st.output.endswith("f_1.ass")


SEGS = [
    P.MapSegment(None, 83.71, 1.0, 2.0, ((10.0, 0.05), (60.0, -0.02))),
    P.MapSegment(83.71, None, 0.959041, 12.0),
]


def test_done_status_carries_mapping_segments_and_cache_returns_them(env):
    media = make_media(env)
    runner = FakeRunner(mapping_segments=SEGS)
    with Harness(env, runner) as h:
        h.submit("m_1", media, sub_index=0)
        st = h.wait_state("m_1", "done")
        raw = P.read_kv(P.status_path(env.queue, "m_1"))
        assert raw["segments"] == "2"
        assert raw["seg0"] == ",83.710,1.0000000,2.0000"
        assert raw["seg0_knots"] == "10.000:0.0500;60.000:-0.0200"
        assert raw["seg1"] == "83.710,,0.9590410,12.0000"
        assert "seg1_knots" not in raw
        assert raw["sync_mode"] == "track"  # config without sync_mode: the default
        assert st.segments == SEGS
        # cache hit: same mapping, without running the engine again
        h.submit("m_2", media, sub_index=0)
        st2 = h.wait_state("m_2", "done")
        assert len(runner.calls) == 1
        assert st2.segments == SEGS
        meta = [P.read_kv(m) for m in (env.tmp / "cache" / "results").glob("*.meta")]
        assert meta and meta[0]["segments"] == "2" and "seg1" in meta[0]


def test_unapplied_result_has_no_segments(env):
    media = make_media(env)
    runner = FakeRunner(applied=False, mapping_segments=SEGS)
    with Harness(env, runner) as h:
        h.submit("u_1", media, sub_index=0)
        st = h.wait_state("u_1", "done")
        assert st.segments is None
        assert "segments" not in P.read_kv(P.status_path(env.queue, "u_1"))


def test_status_reports_configured_sync_mode_also_on_cache_hit(env):
    from vlcsubsync.config import Config

    media = make_media(env)
    cfg = {"c": Config(sync_mode="delay")}
    runner = FakeRunner(mapping_segments=SEGS)
    with Harness(env, runner, config_loader=lambda: cfg["c"]) as h:
        h.submit("s_1", media, sub_index=0)
        assert h.wait_state("s_1", "done").sync_mode == "delay"
        cfg["c"] = Config(sync_mode="track")  # the user edited config.ini
        h.submit("s_2", media, sub_index=0)
        st = h.wait_state("s_2", "done")
        assert len(runner.calls) == 1  # cache hit...
        assert st.sync_mode == "track"  # ...with today's setting
        assert st.segments == SEGS


def test_applied_cache_entry_without_mapping_is_resynced(env):
    """Results cached before this change have no seg* keys: delay mode could not
    use them, so they are a miss (once: the new result replaces them)."""
    media = make_media(env)
    runner = FakeRunner(mapping_segments=None)  # like the engine before mappings
    with Harness(env, runner) as h:
        h.submit("o_1", media, sub_index=0)
        h.wait_state("o_1", "done")
        runner.mapping_segments = SEGS
        h.submit("o_2", media, sub_index=0)
        st = h.wait_state("o_2", "done")
        assert len(runner.calls) == 2  # re-synced, not served from the old entry
        assert st.segments == SEGS
        h.submit("o_3", media, sub_index=0)
        st = h.wait_state("o_3", "done")
        assert len(runner.calls) == 2  # the new entry is a normal hit
        assert st.segments == SEGS


def test_unapplied_cache_entry_without_mapping_is_still_a_hit(env):
    media = make_media(env)
    runner = FakeRunner(applied=False)
    with Harness(env, runner) as h:
        h.submit("n_1", media, sub_index=0)
        h.wait_state("n_1", "done")
        h.submit("n_2", media, sub_index=0)
        h.wait_state("n_2", "done")
        assert len(runner.calls) == 1  # hopeless work is not redone


def test_cache_key_includes_configured_mode(env):
    from vlcsubsync.config import Config

    media = make_media(env)
    d = D.Daemon(
        [env.queue], use_default_queues=False, runner=FakeRunner(),
        resolver=external_resolver, cache_dir=env.tmp / "c",
        lock_path=env.tmp / "state" / "daemon.lock",
    )  # fmt: skip
    src = D.ResolvedSource("embedded", index=0)
    job = D._Job(P.Request(id="k", media=media, sub_index=0), env.queue)
    keys = {m: d.cache_key(job, src, Config(mode=m)) for m in ("fast", "thorough", "exhaustive")}
    assert len(set(keys.values())) == 3
    assert d.cache_key(job, src, Config(mode="bogus")) == keys["fast"]


def test_unlink_retry_survives_a_transient_sharing_violation(tmp_path, monkeypatch):
    target = tmp_path / "x.meta"
    target.write_text("applied=1\n")
    real_unlink = Path.unlink
    calls = []

    def flaky_unlink(self, *a, **kw):
        calls.append(self)
        if len(calls) < 3:
            raise PermissionError(32, "being used by another process")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    D._unlink_retry(target, delay=0)
    assert not target.exists() and len(calls) == 3
