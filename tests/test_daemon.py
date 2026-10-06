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

    def __init__(self, fail_for=(), block=False):
        self.calls: list[D.JobSpec] = []
        self.fail_for = set(fail_for)
        self.block = block
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
            applied=True,
            message="offset +2.35s, drift +4.17%",
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
        self.daemon = D.Daemon(
            [env.queue],
            use_default_queues=False,
            runner=runner,
            resolver=resolver,
            config_loader=lambda: SimpleNamespace(model_en="base.en", model_multi="base"),
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
