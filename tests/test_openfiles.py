"""Which subtitle files a running VLC has open (vlcsubsync.openfiles)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlcsubsync import openfiles as O


def test_subtitle_extensions_are_vlcs():
    for name in ("a.srt", "A.ASS", "x.idx", "x.sub", "x.webvtt", "x.sami", "x.ttml"):
        assert O.is_subtitle_file(name), name
    for name in ("movie.mkv", "notes.txt", "font.ttf", "srt"):
        assert not O.is_subtitle_file(name), name


def test_process_name_matching():
    assert O._name_matches("vlc", ["vlc"])
    assert O._name_matches("VLC", ["vlc"])  # macOS
    assert O._name_matches(r"C:\Program Files\VideoLAN\VLC\vlc.exe", ["vlc"])
    assert not O._name_matches("vlc-subsync", ["vlc"])  # our own helper
    assert not O._name_matches("python3", ["vlc"])


def test_subtitles_of_picks_the_process_playing_the_media(tmp_path):
    media = str(tmp_path / "Movie.mp4")
    other = str(tmp_path / "Other.mp4")
    a, b, c = (str(tmp_path / n) for n in ("a.srt", "b.ass", "c.srt"))
    procs = [
        (1, "vlc", [other, c]),  # another VLC window plays another video
        (2, "vlc", ["/dev/null", media, a, str(tmp_path / "font.ttf"), b, a]),
        (3, "qbittorrent", [media, c]),  # seeding the release: not VLC
        (os.getpid(), "vlc", [media, c]),  # never ourselves
    ]
    assert O._subtitles_of(procs, media, ["vlc"]) == [a, b]
    assert O._subtitles_of(procs, str(tmp_path / "Nope.mp4"), ["vlc"]) is None
    assert O._subtitles_of([(2, "vlc", [media])], media, ["vlc"]) == []


def test_parse_lsof_output():
    out = "\n".join(
        [
            "p101",
            "cVLC",
            "fcwd",
            "n/",
            "f4",
            "n/Users/me/Am\\xc3\\xa9lie (2001).mkv",
            "f5",
            "n/Users/me/Subs/2_English.srt",
            "f6",
            "n->0x1234",  # a socket
            "p202",
            "cVLC",
            "f3",
            "n/tmp/x.srt",
            "",
        ]
    )
    assert O.parse_lsof(out) == [
        (101, "VLC", ["/", "/Users/me/Amélie (2001).mkv", "/Users/me/Subs/2_English.srt"]),
        (202, "VLC", ["/tmp/x.srt"]),
    ]
    assert O._lsof_unescape(r"/a\\b\tc") == "/a\\b\tc"


def test_darwin_runs_lsof_for_vlc():
    # lsof prints POSIX paths: no tmp_path, which is a C:\\ path on Windows CI
    media = "/Users/me/Movies/Movie.mp4"
    sub = "/Users/me/Movies/Subs/3_English.srt"
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=f"p7\ncVLC\nf4\nn{media}\nf5\nn{sub}\n")

    got = O.subtitles_open_in_vlc(media, platform="darwin", run=run)
    assert got == [sub]
    cmd = calls[0]
    assert cmd[1:] == ["-nP", "-w", "+c", "0", "-F", "pcn", "-c", "/^vlc$/i"]

    def failing(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 5)

    assert O.subtitles_open_in_vlc(media, platform="darwin", run=failing) is None
    nothing = SimpleNamespace(returncode=1, stdout="", stderr="")  # lsof found no VLC
    assert O.subtitles_open_in_vlc(media, platform="darwin", run=lambda c, **k: nothing) is None


@pytest.mark.skipif(sys.platform == "win32", reason="needs symlinks")
def test_linux_reads_proc_fd(tmp_path):
    media = tmp_path / "Movie.mp4"
    sub = tmp_path / "Subs" / "3_English.srt"
    sub.parent.mkdir()
    for f in (media, sub):
        f.write_bytes(b"")
    proc = tmp_path / "proc"

    def fake_process(pid, comm, targets):
        fd = proc / str(pid) / "fd"
        fd.mkdir(parents=True)
        (proc / str(pid) / "comm").write_text(comm + "\n", encoding="utf-8")
        for n, t in enumerate(targets, start=3):
            os.symlink(t, fd / str(n))

    fake_process(4242, "vlc", [str(media), str(sub), "/gone.srt (deleted)", "pipe:[1]"])
    fake_process(4343, "vlc-subsync", [str(media), str(tmp_path / "out.srt")])
    (proc / "self").mkdir()
    (proc / "4444").mkdir()  # a process that went away: no comm, no fd
    got = O.subtitles_open_in_vlc(str(media), platform="linux", proc=proc)
    assert got == [str(sub)]
    assert O.subtitles_open_in_vlc(str(tmp_path / "x.mp4"), platform="linux", proc=proc) is None
    assert O.subtitles_open_in_vlc(str(media), platform="linux", proc=tmp_path / "no") is None


def test_windows_checks_each_candidate_with_vlcs_pids(tmp_path):
    media, a, b = r"C:\v\Movie.mp4", r"C:\v\Subs\2_English.srt", r"C:\v\Subs\3_English.srt"
    holders = {
        media: [(10, "VLC media player"), (11, "qBittorrent")],
        a: [(11, "qBittorrent")],
        b: [(10, "VLC media player"), (11, "qBittorrent")],
    }
    images = {10: r"C:\Program Files\VideoLAN\VLC\vlc.exe", 11: r"C:\qbt\qbittorrent.exe"}

    def run(m, cands, holders_fn=holders.get, image=images.get):
        return O._windows_subtitles(m, cands, ["vlc"], 5.0, holders_fn, lambda pid: image(pid, ""))

    assert run(media, [a, b]) == [b]
    # the image path can't be read (elevated VLC): its app name still says VLC
    assert run(media, [a, b], image=lambda pid, d: "") == [b]
    assert run(r"C:\v\Other.mp4", [a, b]) is None  # no VLC has the video open
    assert run(media, [a, b], holders_fn=lambda p: None if p == b else holders.get(p)) is None


def _child_name(pid: int) -> str:
    """The process name subtitles_open_in_vlc sees for ``pid``."""
    if sys.platform.startswith("linux"):
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    if sys.platform == "darwin":
        lsof = shutil.which("lsof") or "/usr/sbin/lsof"
        out = subprocess.run(
            [lsof, "-nP", "-w", "+c", "0", "-F", "c", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        return next(ln[1:] for ln in out.splitlines() if ln.startswith("c"))
    # Windows: a venv's python.exe may start the real python.exe as a child process,
    # which then holds the files
    return "python"


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform in ("darwin", "win32")),
    reason="Linux, macOS and Windows only",
)
def test_finds_files_another_process_has_open(tmp_path):
    """The real OS mechanism, with a Python child holding a video and two subtitle
    files open the way VLC does (and a third file it doesn't open)."""
    media = tmp_path / "Movie Ü.mp4"
    subs = tmp_path / "Subs"
    subs.mkdir()
    held = [subs / "3_English.srt", tmp_path / "English.ass"]
    free = subs / "2_English.srt"
    for f in (media, free, *held):
        f.write_bytes(b"1\n")
    code = (
        "import sys\n"
        "fs = [open(p, 'rb') for p in sys.argv[1:]]\n"
        "print('ready', flush=True)\n"
        "sys.stdin.read()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(media), *map(str, held)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        name = _child_name(child.pid)
        candidates = [str(free), *map(str, held)]
        got = O.subtitles_open_in_vlc(str(media), candidates, names=(name,), timeout=30)
        assert got is not None, f"no process named {name!r} seen with the video open"
        assert sorted(map(O.same_file_key, got)) == sorted(map(O.same_file_key, map(str, held)))
        # VLC is not running here
        assert O.subtitles_open_in_vlc(str(media), candidates, timeout=30) is None
    finally:
        child.stdin.close()
        child.wait(timeout=30)
