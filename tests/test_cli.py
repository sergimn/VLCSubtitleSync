from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlcsubsync import __version__, cli
from vlcsubsync import daemon as D


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    for name in ("CACHE", "STATE", "LOG", "CONFIG"):
        monkeypatch.setenv(f"VLC_SUBSYNC_{name}_DIR", str(tmp_path / "app" / name.lower()))


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 2
    assert "sync" in capsys.readouterr().out


def test_parse_sync():
    a = parse("sync", "m.mkv", "--audio", "1", "--sub", "2", "-o", "x.srt", "--model", "small")
    assert (a.command, a.media, a.audio, a.sub, a.output, a.model) == (
        "sync",
        "m.mkv",
        1,
        2,
        "x.srt",
        "small",
    )
    a = parse("sync", "m.mkv", "--sub-file", "s.srt", "--device", "cpu")
    assert a.sub_file == "s.srt" and a.sub is None and a.device == "cpu" and a.audio == 0
    with pytest.raises(SystemExit):
        parse("sync", "m.mkv", "--sub", "1", "--sub-file", "s.srt")
    with pytest.raises(SystemExit):
        parse("sync", "m.mkv", "--device", "tpu")


def test_parse_setup_uninstall_serve():
    a = parse(
        "setup", "--dry-run", "--no-autostart", "--no-model", "--vlc-dir", "/a", "--vlc-dir", "/b"
    )
    assert a.dry_run and a.no_autostart and a.no_model and a.vlc_dir == ["/a", "/b"]
    a = parse("uninstall", "--purge")
    assert a.purge and not a.dry_run
    a = parse("serve", "--queue-dir", "/q1", "--queue-dir", "/q2", "--no-default-queues")
    assert a.queue_dir == ["/q1", "/q2"] and a.no_default_queues
    a = parse("download-models", "small", "--all")
    assert a.models == ["small"] and a.all
    assert parse("doctor").command == "doctor"


def test_default_output_path():
    assert cli.default_output_path("/v/Movie.mkv", None) == str(Path("/v/Movie.synced.srt"))
    assert cli.default_output_path("/v/Movie.mkv", "/v/Movie.en.ASS") == str(
        Path("/v/Movie.synced.ass")
    )
    assert cli.default_output_path("/v/Movie.mkv", "/v/Movie.sub") == str(
        Path("/v/Movie.synced.srt")
    )


def fake_result(output):
    return SimpleNamespace(
        output_path=output,
        method="whisper",
        offset=-1.5,
        scale=25 / 23.976,
        segments=1,
        confidence=0.88,
        anchors=31,
        applied=True,
        message="offset -1.50s, drift +4.27%",
    )


def test_cmd_sync_external(tmp_path, monkeypatch, capsys):
    import vlcsubsync.sync as sync_mod

    media = tmp_path / "Movie.mkv"
    media.write_bytes(b"x")
    sub = tmp_path / "Movie.srt"
    sub.write_text("")
    calls = {}

    def fake_sync(media_path, audio_index, subtitle, output_path, config, progress=None, **kw):
        calls.update(media=media_path, audio=audio_index, sub=subtitle, out=output_path, cfg=config)
        progress(0.5, "Transcribing")
        return fake_result(output_path)

    monkeypatch.setattr(sync_mod, "sync_subtitles", fake_sync)
    rc = cli.main(["sync", str(media), "--sub-file", str(sub), "--audio", "1", "--model", "tiny"])
    assert rc == 0
    assert calls["audio"] == 1
    assert calls["sub"].kind == "external" and calls["sub"].path == str(sub)
    assert calls["out"] == str(tmp_path / "Movie.synced.srt")
    assert calls["cfg"].model_en == calls["cfg"].model_multi == "tiny"
    out = capsys.readouterr().out
    assert "offset -1.50s" in out and "Applied:    yes" in out and "+4.27" in out


def test_cmd_sync_vlc_ordinal(tmp_path, monkeypatch):
    import vlcsubsync.sync as sync_mod

    media = tmp_path / "Movie.mkv"
    media.write_bytes(b"x")
    side = tmp_path / "Movie.en.srt"
    side.write_text("")
    monkeypatch.setattr(D, "count_embedded_subtitles", lambda m: 1)
    monkeypatch.setattr(D, "default_find_sidecars", lambda m: [str(side)])
    # resolve_source binds defaults at def time; patch through the parameters
    real = D.resolve_source

    def resolve(media, sub_index, sub_path="", **kw):
        return real(
            media,
            sub_index,
            sub_path,
            count_embedded=lambda m: 1,
            find_sidecars=lambda m: [str(side)],
        )

    monkeypatch.setattr(D, "resolve_source", resolve)
    got = {}

    def fake_sync(media_path, audio_index, subtitle, output_path, config, progress=None, **kw):
        got["sub"] = subtitle
        return fake_result(output_path)

    monkeypatch.setattr(sync_mod, "sync_subtitles", fake_sync)
    assert cli.main(["sync", str(media)]) == 0
    assert got["sub"].kind == "embedded" and got["sub"].index == 0
    assert cli.main(["sync", str(media), "--sub", "1", "-o", str(tmp_path / "o.srt")]) == 0
    assert got["sub"].kind == "external" and got["sub"].path == str(side)
    assert cli.main(["sync", str(media), "--sub", "5"]) == 2


def test_cmd_sync_missing_media(tmp_path, capsys):
    assert cli.main(["sync", str(tmp_path / "nope.mkv")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cmd_sync_engine_error(tmp_path, monkeypatch, capsys):
    import vlcsubsync.sync as sync_mod

    media = tmp_path / "m.mkv"
    media.write_bytes(b"x")
    sub = tmp_path / "m.srt"
    sub.write_text("")

    def boom(*a, **k):
        raise ValueError("bad audio")

    monkeypatch.setattr(sync_mod, "sync_subtitles", boom)
    assert cli.main(["sync", str(media), "--sub-file", str(sub)]) == 1
    assert "bad audio" in capsys.readouterr().err


def test_serve_dispatch(monkeypatch):
    got = {}

    def fake_serve(queue_dirs, **kw):
        got.update(queue_dirs=queue_dirs, **kw)
        return 0

    monkeypatch.setattr(D, "serve", fake_serve)
    assert cli.main(["serve", "--queue-dir", "/q", "--no-default-queues", "-v"]) == 0
    assert got == {
        "queue_dirs": ["/q"],
        "use_default_queues": False,
        "log_to_stderr": True,
        "verbose": True,
    }


def test_daemon_main_no_console(monkeypatch):
    got = {}
    monkeypatch.setattr(D, "serve", lambda q, **kw: got.update(q=q, **kw) or 0)
    monkeypatch.setattr(sys, "argv", ["vlc-subsync-daemon", "--queue-dir", "/x"])
    assert cli.daemon_main() == 0
    assert got["q"] == ["/x"] and got["log_to_stderr"] is False


def test_setup_dispatch(monkeypatch):
    import vlcsubsync.setup_vlc as S

    got = {}

    def fake_setup(ctx, **kw):
        got.update(dry=ctx.dry_run, **kw)
        return 0

    monkeypatch.setattr(S, "run_setup", fake_setup)
    assert cli.main(["setup", "--dry-run", "--no-model"]) == 0
    assert got["dry"] and got["model"] is False and got["autostart"] is True


def test_serve_real_daemon_already_running_exits_zero(tmp_path, monkeypatch):
    lock = D.InstanceLock(D.lock_file_path())
    assert lock.acquire()
    try:
        monkeypatch.setenv("VLC_SUBSYNC_QUEUE_DIRS", str(tmp_path / "q"))
        assert cli.main(["serve", "--no-console"]) == 0
    finally:
        lock.release()


def test_doctor_runs(tmp_path, capsys):
    import vlcsubsync.setup_vlc as S

    home = tmp_path / "home"
    (home / ".config/vlc").mkdir(parents=True)
    (home / ".config/vlc/vlcrc").write_text("extraintf=luaintf\nlua-intf=subsync\n")
    lines = []
    ctx = S.Context(
        platform="linux", home=home, env={}, root=tmp_path / "root", which=lambda n: None
    )
    rc = cli.run_doctor(out=lines.append, ctx=ctx)
    text = "\n".join(lines)
    assert "VLC (system package)" in text
    assert "lua-intf: subsync" in text
    assert "CUDA" in text
    assert rc == 1  # scripts missing, daemon not running
