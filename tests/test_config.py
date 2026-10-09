from vlcsubsync.config import Config, default_config_path


def test_defaults_match_design():
    c = Config()
    assert c.model_en == "base.en"
    assert c.model_multi == "base"
    assert c.device == "auto"
    assert c.compute_type == "int8"
    assert c.windows == "auto"
    assert c.min_confidence == 0.5
    assert c.threads == 0
    assert c.sync_mode == "track"  # delay mode is experimental and opt-in


def test_missing_file_gives_defaults(tmp_path):
    assert Config.load(tmp_path / "nope.ini") == Config()


def test_roundtrip_and_unknown_keys_preserved(tmp_path):
    p = tmp_path / "sub" / "config.ini"
    c = Config(model_en="small.en", device="cpu", threads=4, windows="12", min_confidence=0.7)
    c.extra["daemon_poll"] = "0.5"
    written = c.save(p)
    assert written == p and p.exists()
    assert not p.with_name("config.ini.tmp").exists()
    loaded = Config.load(p)
    assert loaded == c
    assert loaded.extra == {"daemon_poll": "0.5"}


def test_parsing_is_lenient(tmp_path):
    p = tmp_path / "config.ini"
    p.write_text(
        "﻿# comment\n[core]\n"
        "model_en = tiny.en   # inline comment\n"
        "device=GPU\n"  # invalid -> ignored
        "threads=abc\n"  # invalid -> ignored
        "min_confidence=1.5\n"  # out of range -> ignored
        "windows=0\n"  # invalid -> ignored
        "windows=inf\n"  # overflow -> ignored
        "threads=inf\n"  # overflow -> ignored
        "compute_type='float32'\n"
        "garbage line\n",
        encoding="utf-8",
    )
    c = Config.load(p)
    assert c.model_en == "tiny.en"
    assert c.device == "auto"
    assert c.threads == 0
    assert c.min_confidence == 0.5
    assert c.windows == "auto"
    assert c.compute_type == "float32"


def test_window_count():
    assert Config().window_count(45 * 60) == 11
    assert Config().window_count(60) == 8
    assert Config().window_count(2 * 3600) == 30
    assert Config(windows="5").window_count(3600) == 5


def test_default_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VLC_SUBSYNC_CONFIG_DIR", str(tmp_path))
    assert default_config_path() == tmp_path / "config.ini"
    Config(model_multi="small").save()
    assert Config.load().model_multi == "small"


def test_verify_windows(tmp_path):
    assert Config().verify_windows == "auto"
    assert Config().verify_budget(45 * 60) == 3
    assert Config().verify_budget(20 * 60) == 2
    assert Config(verify_windows="0").verify_budget(3600) == 0
    assert Config(verify_windows="7").verify_budget(60) == 7
    p = tmp_path / "config.ini"
    p.write_text("verify_windows=5\n", encoding="utf-8")
    assert Config.load(p).verify_windows == "5"
    p.write_text("verify_windows=-1\nverify_windows=lots\n", encoding="utf-8")
    assert Config.load(p).verify_windows == "auto"  # invalid values ignored


def test_mode_parsing(tmp_path):
    assert Config().mode == "fast"
    p = tmp_path / "config.ini"
    for raw, want in [
        ("exhaustive", "exhaustive"),
        (" Thorough ", "thorough"),
        ("FAST", "fast"),
        ("turbo", "fast"),  # invalid -> default kept
        ("", "fast"),
    ]:
        p.write_text(f"mode={raw}\n", encoding="utf-8")
        assert Config.load(p).mode == want, raw
    c = Config(mode="exhaustive", windows="12")
    c.save(p)
    assert Config.load(p) == c
    # with_mode: copy with an override; None/invalid keeps the current mode
    assert c.with_mode("thorough").mode == "thorough" and c.mode == "exhaustive"
    assert c.with_mode(None).mode == "exhaustive"
    assert c.with_mode("bogus").mode == "exhaustive"
    assert Config(mode="bogus").effective_mode == "fast"


def test_window_count_per_mode():
    d = 28.7 * 60
    fast, thorough = Config(mode="fast"), Config(mode="thorough")
    assert fast.window_count(d) == 8
    assert thorough.window_count(d) == 20  # 2.5x
    assert thorough.window_count(2 * 3600) == 75 and fast.window_count(2 * 3600) == 30
    assert thorough.window_count(60) == 20  # capped by the file length in pick_windows
    # explicit windows= still wins in fast/thorough, exhaustive covers the whole file
    assert Config(mode="thorough", windows="5").window_count(3600) == 5
    assert Config(mode="exhaustive", windows="5").window_count(3600) == 120
    assert Config(mode="exhaustive").window_count(d) == 58  # ceil(1722 / 30)


def test_verify_budget_per_mode():
    d = 45 * 60
    assert Config(mode="fast").verify_budget(d) == 3
    assert Config(mode="thorough").verify_budget(d) == 9
    assert Config(mode="exhaustive").verify_budget(d) == 0  # every window is transcribed
    assert Config(mode="exhaustive", verify_windows="4").verify_budget(d) == 4
    assert Config(mode="thorough", verify_windows="1").verify_budget(d) == 1


def test_sync_mode_parsing(tmp_path):
    p = tmp_path / "config.ini"
    p.write_text("sync_mode = Delay\n", encoding="utf-8")
    c = Config.load(p)
    assert c.sync_mode == "delay" and c.effective_sync_mode == "delay"
    p.write_text("sync_mode=sideways\n", encoding="utf-8")
    assert Config.load(p).sync_mode == "track"  # invalid: default kept
    c.save(p)
    assert "sync_mode=delay" in p.read_text(encoding="utf-8")
    assert Config.load(p).sync_mode == "delay"
    assert Config(sync_mode="bogus").effective_sync_mode == "track"


def test_cache_switch(tmp_path):
    p = tmp_path / "config.ini"
    assert Config.load(p).cache is True  # on by default
    for off in ("off", "0", "false", "No"):
        p.write_text(f"cache = {off}\n", encoding="utf-8")
        assert Config.load(p).cache is False
    for on in ("on", "1", "TRUE", "yes"):
        p.write_text(f"cache=off\ncache={on}\n", encoding="utf-8")
        assert Config.load(p).cache is True  # the last line wins
    p.write_text("cache=off\ncache=maybe\n", encoding="utf-8")
    assert Config.load(p).cache is False  # invalid: ignored
    Config(cache=False).save(p)
    assert "cache=off" in p.read_text(encoding="utf-8")
    assert Config.load(p).cache is False
