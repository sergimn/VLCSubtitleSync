"""End-to-end tests with real Whisper models on real (TTS) speech.

Fixtures live in ``tests/fixtures`` and are produced by ``scripts/make_fixtures.py``
(see its docstring). Every cue of ``*.truth.srt`` starts exactly where the speech of
that line starts in the audio, so a synced output can be scored cue-by-cue.

Markers:
  * ``slow`` -- needs Whisper models (downloaded on first use, ~75-150 MB each).
    Run with ``pytest -m slow``.
  * ``vlc``  -- additionally needs a real VLC 3 binary. Run with ``pytest -m vlc``.

Environment knobs:
  * ``VLC_SUBSYNC_TEST_MODELS`` comma-separated English models for the accuracy
    matrix (default ``tiny.en,base.en``).
  * ``VLC_SUBSYNC_TEST_FAST_MODEL`` model for CLI/daemon/VLC tests (default ``tiny.en``).
  * ``VLC_SUBSYNC_PERF_BUDGET`` seconds allowed for the 12-minute perf file (default 300).
  * ``VLC_BIN`` VLC executable (default: ``vlc`` on PATH).
  * ``VLC_SUBSYNC_E2E_ALLOW_SNAP=1`` allow the VLC test with a snap VLC (which ignores
    ``XDG_DATA_HOME``, so it uses the *real* ``~/snap/vlc/current`` queue dir).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
MANIFEST = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))

MEDIAN_MAX = 0.2
P95_MAX = 0.5

EN_MODELS = [
    m.strip()
    for m in os.environ.get("VLC_SUBSYNC_TEST_MODELS", "tiny.en,base.en").split(",")
    if m.strip()
]
FAST_MODEL = os.environ.get("VLC_SUBSYNC_TEST_FAST_MODEL", "tiny.en")

EN_VARIANTS = sorted(
    name.split(".")[1] for name in MANIFEST["files"]["en_dialogue.mkv"]["variants"]
)


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "make_fixtures", ROOT / "scripts" / "make_fixtures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("make_fixtures", mod)
    spec.loader.exec_module(mod)
    return mod


mf = _load_fixture_module()


# ----------------------------------------------------------------------------- helpers


def engine():
    """Import the engine lazily; skip (not fail) while it isn't importable."""
    try:
        from vlcsubsync import sync as sync_mod
        from vlcsubsync.config import Config
    except ImportError as exc:  # pragma: no cover - depends on concurrent work
        pytest.skip(f"vlcsubsync engine not importable yet: {exc}")
    for name in ("sync_subtitles", "SubtitleSource"):
        if not hasattr(sync_mod, name):
            pytest.skip(f"vlcsubsync.sync.{name} not implemented yet")
    return sync_mod, Config


_model_ok: dict[str, bool] = {}


def require_model(name: str) -> None:
    """Make sure a Whisper model is available (download once); skip if impossible."""
    if name not in _model_ok:
        try:
            from faster_whisper import download_model

            download_model(name)
            _model_ok[name] = True
        except Exception as exc:  # noqa: BLE001 - offline, HF outage, ...
            _model_ok[name] = False
            pytest.skip(f"Whisper model {name!r} unavailable: {exc}")
    if not _model_ok[name]:
        pytest.skip(f"Whisper model {name!r} unavailable")


def make_config(Config, model_en: str = FAST_MODEL, model_multi: str = "base"):
    cfg = Config()
    cfg.model_en = model_en
    cfg.model_multi = model_multi
    cfg.device = "cpu"  # deterministic and identical on every CI runner
    return cfg


def score(output: Path, truth: Path) -> dict:
    """Per-cue start error of ``output`` vs ``truth`` (matched by index, else by text)."""
    out, ref = mf.parse_srt(output), mf.parse_srt(truth)
    assert out, f"{output} has no cues"
    if len(out) == len(ref):
        pairs = list(zip(out, ref, strict=True))
    else:  # engine merged/dropped something: match on normalized text
        by_text = {" ".join(c.text.split()).lower(): c for c in ref}
        pairs = [(c, by_text[k]) for c in out if (k := " ".join(c.text.split()).lower()) in by_text]
        assert len(pairs) >= 0.9 * len(ref), f"only {len(pairs)}/{len(ref)} cues matched"
    errs = sorted(abs(o.start - r.start) for o, r in pairs)
    p95 = errs[min(len(errs) - 1, int(round(0.95 * (len(errs) - 1))))]
    return {
        "n": len(errs),
        "median": statistics.median(errs),
        "p95": p95,
        "max": errs[-1],
        "worst": sorted(((abs(o.start - r.start), r.start, r.text[:40]) for o, r in pairs))[-3:],
    }


def assert_accurate(stats: dict, median_max: float = MEDIAN_MAX, p95_max: float = P95_MAX):
    msg = (
        f"median={stats['median']:.3f}s p95={stats['p95']:.3f}s max={stats['max']:.3f}s "
        f"n={stats['n']} worst={stats['worst']}"
    )
    print(msg)
    assert stats["median"] < median_max, msg
    assert stats["p95"] < p95_max, msg


def isolated_env(tmp: Path, model: str = FAST_MODEL) -> dict[str, str]:
    """Environment with all vlc-subsync per-user dirs redirected into ``tmp``."""
    env = dict(os.environ)
    dirs = {
        "VLC_SUBSYNC_CONFIG_DIR": tmp / "config",
        "VLC_SUBSYNC_CACHE_DIR": tmp / "cache",
        "VLC_SUBSYNC_STATE_DIR": tmp / "state",
        "VLC_SUBSYNC_LOG_DIR": tmp / "log",
    }
    for k, p in dirs.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p)
    (dirs["VLC_SUBSYNC_CONFIG_DIR"] / "config.ini").write_text(
        f"model_en={model}\nmodel_multi=base\ndevice=cpu\n", encoding="utf-8"
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "vlcsubsync.cli", *args]


def copy_fixtures(dst: Path, *names: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy2(FIX / n, dst / n)


def write_request(q: Path, req_id: str, fields: dict[str, object]) -> Path:
    """Write a .req exactly like the Lua side does (tmp + rename), independent of
    vlcsubsync.protocol so the on-disk contract itself is tested."""
    body = "".join(f"{k}={v}\n" for k, v in {"version": 1, "id": req_id, **fields}.items())
    final = q / "requests" / f"{req_id}.req"
    tmp = final.with_name(final.name + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, final)
    return final


def read_kv(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return {}
    return dict(ln.split("=", 1) for ln in text.splitlines() if "=" in ln)


def wait_status(q: Path, req_id: str, proc: subprocess.Popen, timeout: float) -> dict:
    path = q / "jobs" / f"{req_id}.status"
    deadline = time.monotonic() + timeout
    st: dict[str, str] = {}
    while time.monotonic() < deadline:
        st = read_kv(path)
        if st.get("state") in ("done", "error"):
            return st
        if proc.poll() is not None:
            pytest.fail(f"daemon exited early with rc={proc.returncode}; last status {st}")
        time.sleep(0.25)
    pytest.fail(f"job {req_id} not finished after {timeout}s; last status {st}")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)


# ----------------------------------------------------------------------------- fixtures sanity


def test_fixture_manifest_consistent():
    """Cheap (unmarked) check that committed fixtures match their manifest."""
    truth = mf.parse_srt(FIX / "en_dialogue.truth.srt")
    assert len(truth) == len(MANIFEST["cues"]["en"]) >= 25
    for name, spec in MANIFEST["files"]["en_dialogue.mkv"]["variants"].items():
        fn = mf.variant_fn(spec["params"])
        got = mf.parse_srt(FIX / name)
        assert len(got) == len(truth)
        for g, t in zip(got, truth, strict=True):
            assert abs(g.start - fn(t.start)) < 0.0015, name
    for f in MANIFEST["files"]:
        assert (FIX / f).stat().st_size < 1_500_000
    assert sum(p.stat().st_size for p in FIX.iterdir() if p.is_file()) < 3_000_000


# ----------------------------------------------------------------------------- engine (slow)


@pytest.mark.slow
@pytest.mark.parametrize("model", EN_MODELS)
@pytest.mark.parametrize("variant", EN_VARIANTS)
def test_sync_variant_accuracy(variant: str, model: str, tmp_path: Path):
    sync_mod, Config = engine()
    require_model(model)
    src = FIX / f"en_dialogue.{variant}.srt"
    out = tmp_path / "out.srt"
    t0 = time.monotonic()
    res = sync_mod.sync_subtitles(
        str(FIX / "en_dialogue.mkv"),
        0,
        sync_mod.SubtitleSource(kind="external", path=str(src)),
        str(out),
        make_config(Config, model_en=model),
    )
    print(f"{variant}/{model}: {res.message} [{res.method}] in {time.monotonic() - t0:.1f}s")
    assert res.applied, res
    assert res.method == "whisper", res
    assert Path(res.output_path).is_file()
    if variant == "cut":
        assert res.segments >= 2, res
    assert_accurate(score(Path(res.output_path), FIX / "en_dialogue.truth.srt"))


@pytest.mark.slow
def test_sync_truth_is_left_alone(tmp_path: Path):
    """Already-correct subtitles must come back (almost) unchanged."""
    sync_mod, Config = engine()
    require_model(FAST_MODEL)
    res = sync_mod.sync_subtitles(
        str(FIX / "en_dialogue.mkv"),
        0,
        sync_mod.SubtitleSource(kind="external", path=str(FIX / "en_dialogue.truth.srt")),
        str(tmp_path / "out.srt"),
        make_config(Config),
    )
    stats = score(Path(res.output_path), FIX / "en_dialogue.truth.srt")
    assert_accurate(stats, 0.1, 0.3)


@pytest.mark.slow
def test_sync_spanish_multilingual(tmp_path: Path):
    """Non-English subtitles use the multilingual model with the guessed language."""
    sync_mod, Config = engine()
    require_model("base")
    res = sync_mod.sync_subtitles(
        str(FIX / "es_dialogue.mkv"),
        0,
        sync_mod.SubtitleSource(kind="external", path=str(FIX / "es_dialogue.offset_plus_3_2.srt")),
        str(tmp_path / "out.srt"),
        make_config(Config, model_multi="base"),
    )
    print(res)
    assert res.applied and res.method == "whisper", res
    assert_accurate(score(Path(res.output_path), FIX / "es_dialogue.truth.srt"))


@pytest.mark.slow
def test_sync_language_mismatch_uses_vad(tmp_path: Path):
    """Spanish subtitle text on English audio -> VAD fallback (looser tolerance)."""
    sync_mod, Config = engine()
    require_model(FAST_MODEL)
    require_model("base")
    res = sync_mod.sync_subtitles(
        str(FIX / "en_dialogue.mkv"),
        0,
        sync_mod.SubtitleSource(
            kind="external", path=str(FIX / "en_dialogue.es_text.offset_plus_3_2.srt")
        ),
        str(tmp_path / "out.srt"),
        make_config(Config),
    )
    print(res)
    assert res.method == "vad", res
    assert res.applied, res
    # timing is the English truth; text is Spanish -> score by index
    out = mf.parse_srt(Path(res.output_path))
    ref = mf.parse_srt(FIX / "en_dialogue.truth.srt")
    assert len(out) == len(ref)
    errs = sorted(abs(o.start - r.start) for o, r in zip(out, ref, strict=True))
    assert statistics.median(errs) < 0.3, errs
    assert errs[int(0.95 * (len(errs) - 1))] < 0.8, errs


@pytest.mark.slow
@pytest.mark.parametrize(
    "source, truth_offset",
    [("embedded", None), ("sidecar", None)],
    ids=["embedded-sub0", "sidecar-en"],
)
def test_sync_second_audio_track(source: str, truth_offset, tmp_path: Path):
    """multi_audio.mkv: audio#0 is a Spanish dub with different timing, audio#1 English.
    Syncing English subs to audio#1 must match the English truth."""
    sync_mod, Config = engine()
    require_model(FAST_MODEL)
    if source == "embedded":
        sub = sync_mod.SubtitleSource(kind="embedded", index=0)
    else:
        sub = sync_mod.SubtitleSource(kind="external", path=str(FIX / "multi_audio.en.srt"))
    res = sync_mod.sync_subtitles(
        str(FIX / "multi_audio.mkv"), 1, sub, str(tmp_path / "out.srt"), make_config(Config)
    )
    print(res)
    assert res.applied and res.method == "whisper", res
    assert_accurate(score(Path(res.output_path), FIX / "en_dialogue.truth.srt"))


@pytest.mark.slow
def test_perf_long_file(tmp_path: Path):
    """~14 min file built on the fly from the committed audio (never committed)."""
    sync_mod, Config = engine()
    require_model(FAST_MODEL)
    paths = mf.make_long(tmp_path, minutes=12.0)
    budget = float(os.environ.get("VLC_SUBSYNC_PERF_BUDGET", "300"))
    stages: list[tuple[float, float, str]] = []
    t0 = time.monotonic()
    res = sync_mod.sync_subtitles(
        str(paths["media"]),
        0,
        sync_mod.SubtitleSource(kind="external", path=str(paths["offset"])),
        str(tmp_path / "out.srt"),
        make_config(Config),
        progress=lambda p, m: stages.append((time.monotonic() - t0, p, m)),
    )
    elapsed = time.monotonic() - t0
    print(f"long file: {elapsed:.1f}s, {res}")
    for s in stages[:: max(1, len(stages) // 20)]:
        print(f"  {s[0]:7.1f}s {s[1]:5.2f} {s[2]}")
    assert res.applied, res
    assert elapsed < budget, f"took {elapsed:.0f}s > budget {budget:.0f}s"
    assert [p for _, p, _ in stages] == sorted(p for _, p, _ in stages), "progress not monotonic"
    # Lines repeat every ~3 min loop, so only require loose accuracy here.
    assert_accurate(score(Path(res.output_path), paths["truth"]), 0.3, 1.0)


# ----------------------------------------------------------------------------- CLI (slow)


@pytest.mark.slow
@pytest.mark.parametrize("sub, truth_note", [("0", "embedded"), ("1", "sidecar")])
def test_cli_sync_track_ordinals(sub: str, truth_note: str, tmp_path: Path):
    engine()
    require_model(FAST_MODEL)
    media_dir = tmp_path / "media"
    copy_fixtures(media_dir, "multi_audio.mkv", "multi_audio.en.srt")
    out = tmp_path / "synced.srt"
    env = isolated_env(tmp_path / "home")
    r = subprocess.run(
        cli("sync", str(media_dir / "multi_audio.mkv"), "--audio", "1", "--sub", sub)
        + ["-o", str(out), "--model", FAST_MODEL, "--device", "cpu"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    print(r.stdout, r.stderr[-3000:])
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    expected_src = "multi_audio.en.srt" if truth_note == "sidecar" else "embedded subtitle track 0"
    assert expected_src in r.stdout
    assert "Applied:    yes" in r.stdout
    assert_accurate(score(out, FIX / "en_dialogue.truth.srt"))


@pytest.mark.slow
def test_cli_default_output_path(tmp_path: Path):
    engine()
    require_model(FAST_MODEL)
    copy_fixtures(tmp_path, "en_dialogue.mkv", "en_dialogue.offset_plus_3_2.srt")
    r = subprocess.run(
        cli("sync", str(tmp_path / "en_dialogue.mkv"), "--sub-file")
        + [str(tmp_path / "en_dialogue.offset_plus_3_2.srt"), "--device", "cpu"],
        env=isolated_env(tmp_path / "home"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "en_dialogue.synced.srt"
    assert out.is_file(), list(tmp_path.iterdir())
    assert_accurate(score(out, FIX / "en_dialogue.truth.srt"))


def test_cli_errors_cleanly_on_missing_media(tmp_path: Path):
    """Unmarked: no model needed; bad input must give rc!=0 and a message, no traceback."""
    try:
        import vlcsubsync.cli  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"cli not importable: {exc}")
    r = subprocess.run(
        cli("sync", str(tmp_path / "nope.mkv"), "--sub", "0"),
        env=isolated_env(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert "not found" in r.stderr.lower()


# ----------------------------------------------------------------------------- daemon (slow)


@pytest.fixture
def daemon(tmp_path: Path):
    """``vlc-subsync serve`` on a private queue dir with isolated user dirs."""
    engine()
    q = tmp_path / "q"
    for d in ("requests", "jobs", "out"):
        (q / d).mkdir(parents=True)
    env = isolated_env(tmp_path / "home")
    log = (tmp_path / "daemon.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cli("serve", "--queue-dir", str(q), "--no-default-queues", "-v"),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 60
        while not read_kv(q / "heartbeat").get("time"):
            if proc.poll() is not None or time.monotonic() > deadline:
                log.flush()
                pytest.fail(
                    "daemon did not write a heartbeat:\n"
                    + (tmp_path / "daemon.log").read_text(encoding="utf-8")
                )
            time.sleep(0.2)
        yield q, proc
    finally:
        stop(proc)
        log.close()
        print((tmp_path / "daemon.log").read_text(encoding="utf-8")[-4000:])


@pytest.mark.slow
def test_daemon_round_trip(daemon, tmp_path: Path):
    q, proc = daemon
    require_model(FAST_MODEL)
    hb = read_kv(q / "heartbeat")
    assert int(hb["pid"]) == proc.pid
    assert abs(time.time() - float(hb["time"])) <= 10

    media_dir = tmp_path / "media"
    copy_fixtures(media_dir, "multi_audio.mkv", "multi_audio.en.srt")
    media = media_dir / "multi_audio.mkv"
    fields = {
        "media": str(media),
        "audio_index": 1,
        "audio_label": "Track 2 - [English]",
        "sub_index": 0,
        "sub_label": "Track 1 - [English]",
        "sub_path": "",
        "force": 0,
    }
    req = write_request(q, "1728221234_1", fields)
    t0 = time.monotonic()
    st = wait_status(q, "1728221234_1", proc, timeout=600)
    first = time.monotonic() - t0
    print("status:", st)
    assert st["state"] == "done", st
    assert st.get("applied") == "1", st
    assert st.get("method") == "whisper", st
    assert not req.exists(), ".req must be deleted once picked up"
    out = Path(st["output"])
    assert out.is_file() and out.parent == q / "out", out
    assert abs(float(st["offset"]) + 4.5) < 0.3, st  # embedded track was +4.5 s late
    assert_accurate(score(out, FIX / "en_dialogue.truth.srt"))

    # same request again -> served from the result cache, fast
    write_request(q, "1728221234_2", fields)
    t0 = time.monotonic()
    st2 = wait_status(q, "1728221234_2", proc, timeout=120)
    assert st2["state"] == "done" and st2.get("applied") == "1", st2
    assert time.monotonic() - t0 < max(5.0, first / 3), "result cache not used"
    assert_accurate(score(Path(st2["output"]), FIX / "en_dialogue.truth.srt"))

    # sub ordinal 1 == first sidecar (1 embedded text track before it)
    write_request(q, "1728221234_3", {**fields, "sub_index": 1, "sub_label": "Track 2"})
    st3 = wait_status(q, "1728221234_3", proc, timeout=600)
    assert st3["state"] == "done" and st3.get("applied") == "1", st3
    assert abs(float(st3["scale"]) - 1 / mf.DRIFT) < 0.002, st3
    assert_accurate(score(Path(st3["output"]), FIX / "en_dialogue.truth.srt"))


@pytest.mark.slow
def test_daemon_reports_errors(daemon, tmp_path: Path):
    q, proc = daemon
    write_request(
        q, "bad_1", {"media": str(tmp_path / "missing.mkv"), "audio_index": 0, "sub_index": 0}
    )
    st = wait_status(q, "bad_1", proc, timeout=60)
    assert st["state"] == "error", st
    assert st.get("message"), st
    assert proc.poll() is None, "daemon must survive a failing job"


# ----------------------------------------------------------------------------- real VLC


def _vlc_binary() -> str | None:
    cand = os.environ.get("VLC_BIN") or shutil.which("vlc")
    if not cand and sys.platform == "darwin":
        mac = "/Applications/VLC.app/Contents/MacOS/VLC"
        cand = mac if os.path.exists(mac) else None
    return cand


def _is_snap(vlc: str) -> bool:
    return os.path.realpath(vlc).startswith("/snap/") or vlc.startswith("/snap/")


@pytest.mark.slow
@pytest.mark.vlc
def test_vlc_end_to_end(tmp_path: Path):
    """Headless VLC + our Lua intf + daemon: playing a file with a subtitle track
    selected must end with the synced subtitle loaded as a new track.

    Isolation: the Lua scripts are found through ``VLC_DATA_PATH`` (VLC searches
    ``$VLC_DATA_PATH/lua/intf``); the interface is enabled on the command line
    (``--extraintf luaintf --lua-intf subsync``) so no vlcrc is touched; HOME and the
    XDG dirs point into ``tmp_path`` so ``vlc.config.userdatadir()`` (=> queue dir)
    is private. Note ``VLC_DATA_PATH`` *replaces* VLC's share dir; that's fine for
    ``-I dummy`` (no skins / http interface needed).
    """
    vlc = _vlc_binary()
    if not vlc:
        pytest.skip("VLC not installed")
    engine()
    require_model(FAST_MODEL)
    snap = _is_snap(vlc)
    if snap and os.environ.get("VLC_SUBSYNC_E2E_ALLOW_SNAP") != "1":
        pytest.skip(
            "snap VLC ignores XDG_DATA_HOME (queue dir would be the real "
            "~/snap/vlc/current/...); set VLC_SUBSYNC_E2E_ALLOW_SNAP=1 to run anyway"
        )

    # snap confinement: /tmp is private and hidden dirs in $HOME are blocked, so put
    # everything under the (non-hidden) repo dir in that case.
    base = tmp_path
    if snap:
        base = ROOT / ".cache" / "vlc-e2e" / tmp_path.name
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
    home = base / "home"
    data_home = home / ".local" / "share"
    lua_root = base / "vlcdata"
    (lua_root / "lua" / "intf").mkdir(parents=True)
    (lua_root / "lua" / "extensions").mkdir(parents=True)
    pkg_lua = ROOT / "src" / "vlcsubsync" / "lua"
    shutil.copy2(pkg_lua / "intf" / "subsync.lua", lua_root / "lua" / "intf")
    shutil.copy2(pkg_lua / "extensions" / "subsync_ext.lua", lua_root / "lua" / "extensions")

    if snap:
        q = Path.home() / "snap" / "vlc" / "current" / ".local" / "share" / "vlc" / "subsync"
    elif sys.platform == "darwin":
        q = home / "Library" / "Application Support" / "org.videolan.vlc" / "subsync"
    else:
        q = data_home / "vlc" / "subsync"
    for d in ("requests", "jobs", "out"):
        (q / d).mkdir(parents=True, exist_ok=True)

    media_dir = base / "media"
    copy_fixtures(media_dir, "multi_audio.mkv", "multi_audio.en.srt")

    env = isolated_env(base / "subsync")
    env.update(
        HOME=str(home),
        XDG_DATA_HOME=str(data_home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        VLC_DATA_PATH=str(lua_root),
    )
    # the daemon must use the real HOME for the HF model cache
    denv = dict(env, HOME=os.environ.get("HOME", str(home)))
    denv.pop("XDG_CACHE_HOME")
    dlog = (base / "daemon.log").open("w", encoding="utf-8")
    daemon_proc = subprocess.Popen(
        cli("serve", "--queue-dir", str(q), "--no-default-queues", "-v"),
        env=denv,
        stdout=dlog,
        stderr=subprocess.STDOUT,
    )
    vlog_path = base / "vlc.log"
    vlog = vlog_path.open("w", encoding="utf-8", errors="replace")
    vlc_cmd = [
        vlc, "-I", "dummy", "--extraintf", "luaintf", "--lua-intf", "subsync",
        "-vv", "--no-metadata-network-access",
        "--vout", "dummy", "--aout", "dummy",
        "--audio-track", "1", "--sub-track", "0",
        "--play-and-exit",
        str(media_dir / "multi_audio.mkv"),
    ]  # fmt: skip
    vlc_proc = None
    try:
        deadline = time.monotonic() + 60
        while not read_kv(q / "heartbeat").get("time"):
            assert daemon_proc.poll() is None, "daemon died"
            assert time.monotonic() < deadline, "no daemon heartbeat"
            time.sleep(0.2)
        vlc_proc = subprocess.Popen(vlc_cmd, env=env, stdout=vlog, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        text = ""
        while time.monotonic() < deadline:
            text = vlog_path.read_text(encoding="utf-8", errors="replace")
            if "[subsync] synced track es=" in text:
                break
            if vlc_proc.poll() is not None:
                break
            time.sleep(0.5)
        print("\n".join(ln for ln in text.splitlines() if "subsync" in ln.lower())[-6000:])
        assert "[subsync] started, queue dir" in text, "Lua intf did not start (see log)"
        assert "[subsync] request " in text, "Lua intf never submitted a request"
        assert "[subsync] synced track es=" in text, "synced subtitle was not added"
        outs = sorted((q / "out").glob("*.srt"), key=lambda p: p.stat().st_mtime)
        assert outs, "daemon produced no output"
        assert any(o.name in text for o in outs), "VLC log never mentions the synced file"
        assert_accurate(score(outs[-1], FIX / "en_dialogue.truth.srt"))
    finally:
        if vlc_proc is not None:
            stop(vlc_proc)
        stop(daemon_proc)
        vlog.close()
        dlog.close()
        print((base / "daemon.log").read_text(encoding="utf-8")[-3000:])
        if snap:
            for sub in ("requests", "jobs", "out"):
                for p in (q / sub).glob("*"):
                    p.unlink(missing_ok=True)
            shutil.rmtree(base, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.vlc
def test_vlc_delay_mode_end_to_end(tmp_path: Path):
    """Experimental delay mode in a real VLC: the original (drifting) sidecar stays
    selected, no track is added, and the intf moves the input's spu-delay along the
    mapping as playback advances.

    Same isolation as :func:`test_vlc_end_to_end`, plus: the intf is installed under
    another name (``subsync_e2e``; VLC prefers a ``subsync.lua`` in the user data
    dir, i.e. a real installation) and uses its own queue dir name, so an installed
    SubSync helper watching the real snap queue never sees these requests.
    """
    import re

    vlc = _vlc_binary()
    if not vlc:
        pytest.skip("VLC not installed")
    engine()
    require_model(FAST_MODEL)
    snap = _is_snap(vlc)
    if snap and os.environ.get("VLC_SUBSYNC_E2E_ALLOW_SNAP") != "1":
        pytest.skip("snap VLC: set VLC_SUBSYNC_E2E_ALLOW_SNAP=1 to run anyway")

    qname = "subsync-e2e"
    base = tmp_path
    if snap:
        base = ROOT / ".cache" / "vlc-e2e" / tmp_path.name
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
    home = base / "home"
    data_home = home / ".local" / "share"
    lua_root = base / "vlcdata"
    (lua_root / "lua" / "intf").mkdir(parents=True)
    src = (ROOT / "src" / "vlcsubsync" / "lua" / "intf" / "subsync.lua").read_text("utf-8")
    patched = src.replace('return join(base, "subsync")', f'return join(base, "{qname}")')
    assert patched != src
    (lua_root / "lua" / "intf" / "subsync_e2e.lua").write_text(patched, encoding="utf-8")

    if snap:
        q = Path.home() / "snap" / "vlc" / "current" / ".local" / "share" / "vlc" / qname
    elif sys.platform == "darwin":
        q = home / "Library" / "Application Support" / "org.videolan.vlc" / qname
    else:
        q = data_home / "vlc" / qname
    for d in ("requests", "jobs", "out"):
        (q / d).mkdir(parents=True, exist_ok=True)

    media_dir = base / "media"
    copy_fixtures(media_dir, "multi_audio.mkv", "multi_audio.en.srt")
    env = isolated_env(base / "subsync")
    with (base / "subsync" / "config" / "config.ini").open("a", encoding="utf-8") as fh:
        fh.write("sync_mode=delay\n")  # the experimental mode, from the helper's config
    env.update(
        HOME=str(home),
        XDG_DATA_HOME=str(data_home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        VLC_DATA_PATH=str(lua_root),
    )
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    denv = dict(env, HOME=os.environ.get("HOME", str(home)))
    denv.pop("XDG_CACHE_HOME")
    dlog = (base / "daemon.log").open("w", encoding="utf-8")
    daemon_proc = subprocess.Popen(
        cli("serve", "--queue-dir", str(q), "--no-default-queues", "-v"),
        env=denv,
        stdout=dlog,
        stderr=subprocess.STDOUT,
    )
    vlog_path = base / "vlc.log"
    vlog = vlog_path.open("w", encoding="utf-8", errors="replace")
    vlc_cmd = [
        vlc, "-I", "dummy", "--extraintf", "luaintf", "--lua-intf", "subsync_e2e",
        "-vv", "--no-metadata-network-access",
        "--vout", "dummy", "--aout", "dummy",
        # English audio, and the sidecar: timed for 25 fps on 23.976 audio, -1.5 s
        "--audio-track", "1", "--sub-track", "1",
        "--play-and-exit",
        str(media_dir / "multi_audio.mkv"),
    ]  # fmt: skip
    pat = re.compile(r"\[subsync\] spu-delay=(-?\d+) us \(time=([0-9.]+)s")
    vlc_proc = None
    try:
        deadline = time.monotonic() + 60
        while not read_kv(q / "heartbeat").get("time"):
            assert daemon_proc.poll() is None, "daemon died"
            assert time.monotonic() < deadline, "no daemon heartbeat"
            time.sleep(0.2)
        vlc_proc = subprocess.Popen(vlc_cmd, env=env, stdout=vlog, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        text, sets = "", []
        while time.monotonic() < deadline:
            text = vlog_path.read_text(encoding="utf-8", errors="replace")
            sets = [(float(t), int(d) / 1e6) for d, t in pat.findall(text)]
            if len(sets) >= 12 or vlc_proc.poll() is not None:
                break
            time.sleep(0.5)
        print("\n".join(ln for ln in text.splitlines() if "[subsync]" in ln)[-6000:])
        assert "[subsync] delay mode: start" in text, "delay mode never started"
        assert "synced track es=" not in text and "added subtitle" not in text
        assert len(sets) >= 12, f"spu-delay was set only {len(sets)} times"
        # the sidecar is truth * 25/23.976 - 1.5, so at playback time T the delay is
        # T - (T * 25/23.976 - 1.5), aimed (lookahead) at T + 1 s + the negative part
        k = 25 / 23.976

        def want(t):
            d = t - (t * k - 1.5)
            t2 = t + 1.0 + max(0.0, -d)
            return t2 - (t2 * k - 1.5)

        errs = [abs(d - want(t)) for t, d in sets]
        assert max(errs) < 0.3, list(zip(sets, errs, strict=True))
        delays = [d for _t, d in sets]
        assert delays == sorted(delays, reverse=True)  # drift: it keeps decreasing
        assert delays[0] - delays[-1] > 0.3
    finally:
        if vlc_proc is not None:
            stop(vlc_proc)
        stop(daemon_proc)
        vlog.close()
        dlog.close()
        print((base / "daemon.log").read_text(encoding="utf-8")[-3000:])
        if snap:
            shutil.rmtree(q, ignore_errors=True)  # our own queue dir, nothing else
            shutil.rmtree(base, ignore_errors=True)
