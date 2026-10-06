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
