from __future__ import annotations

import os

import pytest

from vlcsubsync import protocol as P


def test_request_roundtrip(tmp_path):
    req = P.Request(
        id="1728221234_1",
        media="/abs/path/to/movie with spaces #1.mkv",
        audio_index=1,
        audio_label="Track 2 - [English]",
        sub_index=0,
        sub_label="Track 1 - [English]",
        force=True,
    )
    path = P.write_request(tmp_path, req)
    assert path == tmp_path / "requests" / "1728221234_1.req"
    back = P.read_request(path)
    assert back == req


def test_request_file_format_matches_design(tmp_path):
    req = P.Request(id="a1", media="/m.mkv", audio_index=1, sub_index=0)
    text = (P.write_request(tmp_path, req)).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "version=1"
    assert "id=a1" in lines
    assert "media=/m.mkv" in lines
    assert "audio_index=1" in lines
    assert "sub_index=0" in lines
    assert "sub_path=" in lines
    assert "force=0" in lines


@pytest.mark.parametrize("mode", ["fast", "thorough", "exhaustive"])
def test_request_roundtrip_with_mode(tmp_path, mode):
    req = P.Request(id="m1", media="/m.mkv", sub_index=0, mode=mode)
    path = P.write_request(tmp_path, req)
    assert f"mode={mode}" in path.read_text(encoding="utf-8").splitlines()
    assert P.read_request(path) == req


def test_request_mode_optional_and_validated(tmp_path):
    # no mode -> no key written (old daemons/readers are unaffected)
    path = P.write_request(tmp_path, P.Request(id="m2", media="/m.mkv", sub_index=0))
    assert "mode" not in P.read_kv(path)
    assert P.read_request(path).mode == ""
    base = {"id": "m3", "media": "/m.mkv"}
    assert P.Request.from_dict({**base, "mode": " Exhaustive "}).mode == "exhaustive"
    assert P.Request.from_dict({**base, "mode": "turbo"}).mode == ""  # unknown: ignored
    assert P.Request.from_dict({**base, "mode": ""}).mode == ""


def test_status_roundtrip(tmp_path):
    st = P.Status(
        id="x_1",
        state="done",
        progress=1,
        message="offset +2.35s",
        output="/q/out/x_1.srt",
        applied=True,
        method="whisper",
        offset=2.35,
        scale=1.0417,
        confidence=0.93,
    )
    P.write_status(tmp_path, st)
    text = (tmp_path / "jobs" / "x_1.status").read_text()
    assert "state=done" in text
    assert "applied=1" in text
    assert "offset=2.350" in text
    assert "progress=1.000" in text
    back = P.read_status(tmp_path, "x_1")
    assert back.state == "done" and back.applied is True
    assert back.offset == pytest.approx(2.35)
    assert back.scale == pytest.approx(1.0417)
    assert back.output == "/q/out/x_1.srt"


def test_running_status_omits_result_fields(tmp_path):
    P.write_status(tmp_path, P.Status(id="r", state="running", progress=0.42, message="m"))
    data = P.read_kv(tmp_path / "jobs" / "r.status")
    assert data["progress"] == "0.420"
    assert "output" not in data and "applied" not in data


@pytest.mark.parametrize(
    "bad", ["../evil", "a/b", "a\\b", "", ".", "..", "a b", "x.req", "é", "a" * 200, "/abs"]
)
def test_invalid_ids_rejected(bad, tmp_path):
    assert not P.is_valid_id(bad)
    with pytest.raises(P.ProtocolError):
        P.status_path(tmp_path, bad)


@pytest.mark.parametrize("good", ["1728221234_1", "abc-DEF_9", "x"])
def test_valid_ids(good):
    assert P.is_valid_id(good)


def test_request_with_traversal_id_uses_filename_or_fails(tmp_path):
    d = tmp_path / "requests"
    d.mkdir()
    (d / "good_1.req").write_text("id=../../etc\nmedia=/m.mkv\n")
    assert P.read_request(d / "good_1.req").id == "good_1"
    (d / "bad..req").write_text("id=../../etc\nmedia=/m.mkv\n")
    with pytest.raises(P.ProtocolError):
        P.read_request(d / "bad..req")


def test_sanitize_values():
    assert P.sanitize_value("a\nb\r\nc") == "a b  c"
    assert P.sanitize_value("tab\there\x00") == "tab here "
    assert P.sanitize_value("line sep") == "line sep"
    assert P.sanitize_value(None) == ""
    assert P.sanitize_value(True) == "1"
    assert P.sanitize_value(False) == "0"
    text = P.format_kv({"message": "multi\nline", "k": "ünïcødé"})
    assert text == "message=multi line\nk=ünïcødé\n"


def test_format_rejects_bad_keys():
    with pytest.raises(P.ProtocolError):
        P.format_kv({"bad key": 1})
    with pytest.raises(P.ProtocolError):
        P.format_kv({"a=b": 1})


def test_tolerant_reader():
    text = (
        "﻿version=1\r\n"
        "# comment\r\n"
        "\r\n"
        "garbage line\r\n"
        "id=abc\r\n"
        "media=/path/with=equals.mkv\r\n"
        "unknown_key=whatever\r\n"
        "=novalue\r\n"
        "audio_index=notanumber\r\n"
    )
    data = P.parse_kv(text)
    assert data["version"] == "1"
    assert data["media"] == "/path/with=equals.mkv"
    req = P.Request.from_dict(data)
    assert req.id == "abc"
    assert req.audio_index == 0  # tolerant default
    assert req.sub_index is None


def test_reader_handles_invalid_utf8(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"id=x\nmessage=\xff\xfe bad\n")
    data = P.read_kv(p)
    assert data["id"] == "x"
    assert "bad" in data["message"]


def test_read_missing_returns_none(tmp_path):
    assert P.read_kv(tmp_path / "nope") is None
    assert P.read_status(tmp_path, "nope") is None
    assert P.read_heartbeat(tmp_path) is None


def test_file_uri_media_is_decoded():
    data = {"id": "a", "media": "file:///home/u/My%20Movie%20%C3%A9.mkv", "sub_index": "-1"}
    req = P.Request.from_dict(data)
    if os.name != "nt":
        assert req.media == "/home/u/My Movie é.mkv"
    assert req.sub_index is None


def test_atomic_write_leaves_no_tmp_and_replaces(tmp_path):
    target = tmp_path / "jobs" / "a.status"
    P.write_kv(target, {"state": "queued"})
    P.write_kv(target, {"state": "running"})
    assert P.read_kv(target) == {"state": "running"}
    assert [p.name for p in target.parent.iterdir()] == ["a.status"]


def test_atomic_write_failure_keeps_old_content(tmp_path, monkeypatch):
    target = tmp_path / "x"
    P.write_kv(target, {"a": "1"})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(P.os, "replace", boom)
    with pytest.raises(OSError):
        P.write_kv(target, {"a": "2"})
    assert P.read_kv(target) == {"a": "1"}
    assert [p.name for p in tmp_path.iterdir()] == ["x"]


def test_atomic_write_retries_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "x"
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        real(src, dst)

    monkeypatch.setattr(P.os, "replace", flaky)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    P.write_kv(target, {"a": "1"})
    assert P.read_kv(target) == {"a": "1"}


def test_heartbeat_roundtrip(tmp_path):
    P.write_heartbeat(tmp_path, P.Heartbeat(time=1000.7, pid=42, version="0.1.0"))
    assert (tmp_path / "heartbeat").read_text() == "time=1000\npid=42\nversion=0.1.0\n"
    hb = P.read_heartbeat(tmp_path)
    assert hb.pid == 42 and hb.age(now=1005) == 5
