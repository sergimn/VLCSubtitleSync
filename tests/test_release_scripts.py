"""scripts/version.py, release_notes.py and release_assets.py, used by the release workflow."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from vlcsubsync import __version__

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("version_script", ROOT / "scripts" / "version.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


V = _load()


def test_versions_agree():
    assert V.current_version() == __version__


@pytest.fixture
def tree(tmp_path):
    for rel in V.FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_set_version_rewrites_every_file(tree):
    V.set_version("1.2.3", tree)
    assert set(V.read_versions(tree).values()) == {"1.2.3"}
    # only the version line changes
    for rel in V.FILES:
        old = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        new = (tree / rel).read_text(encoding="utf-8").splitlines()
        assert len(old) == len(new)
        assert sum(a != b for a, b in zip(old, new, strict=True)) == (__version__ != "1.2.3")


@pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0-rc.1", "1.0.0\n"])
def test_set_version_rejects_non_semver(tree, bad):
    with pytest.raises(SystemExit):
        V.set_version(bad, tree)
    assert V.current_version(tree) == __version__


def test_mismatch_is_reported(tree):
    init = tree / "src/vlcsubsync/__init__.py"
    init.write_text(init.read_text().replace(__version__, "9.9.9"))
    with pytest.raises(SystemExit, match="version mismatch"):
        V.current_version(tree)


def _load_notes():
    spec = importlib.util.spec_from_file_location(
        "release_notes_script", ROOT / "scripts" / "release_notes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


N = _load_notes()

GENERATED = """<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
* Fix {braces} by @sergimn in https://github.com/o/r/pull/1

**Full Changelog**: https://github.com/o/r/commits/v1.0.0
"""


def test_untouched_skeleton_renders_only_install_and_changes():
    out = N.render(N.skeleton("1.0.0", GENERATED))
    assert "<!--" not in out
    assert "## Highlights" not in out  # left empty
    assert "releases/download/v1.0.0/install.sh" in out
    assert out.index("## Install or upgrade") < out.index("## Changes")
    # GitHub's own headings are nested under "Changes"
    assert "### What's Changed\n* Fix {braces} by @sergimn" in out
    assert "\n\n\n" not in out and out.endswith("\n")


def test_filled_in_highlights_are_kept():
    text = N.skeleton("1.0.0", GENERATED).replace(
        "<!-- What a user notices in this release, in a few bullets. -->", "- First release."
    )
    out = N.render(text)
    assert out.startswith("## Highlights\n\n- First release.\n\n## Install or upgrade")


def _load_assets():
    spec = importlib.util.spec_from_file_location(
        "release_assets_script", ROOT / "scripts" / "release_assets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


A = _load_assets()
BASE = "https://github.com/o/r/releases/download/v1.2.3"
WHEEL = "vlc_subsync-1.2.3-py3-none-any.whl"


def test_released_installers_point_at_the_release(tmp_path):
    A.write_installers(BASE, WHEEL, tmp_path)
    expected = {
        "install.sh": [
            f'DEFAULT_SOURCE="{BASE}/{WHEEL}"',
            f'DEFAULT_CONSTRAINTS="{BASE}/constraints.txt"',
        ],
        "install.ps1": [
            f"$DefaultSource = '{BASE}/{WHEEL}'",
            f"$DefaultConstraints = '{BASE}/constraints.txt'",
        ],
        "install.cmd": [f"{BASE}/install.ps1"],
    }
    for name, wanted in expected.items():
        old_raw, new_raw = (ROOT / name).read_bytes(), (tmp_path / name).read_bytes()
        assert old_raw.count(b"\r\n") == new_raw.count(b"\r\n")  # line endings kept
        old, new = old_raw.decode().splitlines(), new_raw.decode().splitlines()
        assert len(old) == len(new)
        changed = [b for a, b in zip(old, new, strict=True) if a != b]
        assert len(changed) == len(wanted)
        for line, want in zip(changed, wanted, strict=True):
            assert want in line


def test_installer_rewrite_must_match(tmp_path):
    for name in A.REWRITES:
        shutil.copyfile(ROOT / name, tmp_path / name)
    sh = tmp_path / "install.sh"
    sh.write_text(sh.read_text().replace('DEFAULT_CONSTRAINTS=""\n', ""))
    with pytest.raises(SystemExit, match="install.sh"):
        A.write_installers(BASE, WHEEL, tmp_path / "out", root=tmp_path)


@pytest.mark.parametrize(
    ("base", "wheel"),
    [("http://example.com/x", WHEEL), (BASE + "/", WHEEL), (BASE, "x.tar.gz"), (BASE, "a b.whl")],
)
def test_installer_rewrite_rejects_bad_input(tmp_path, base, wheel):
    with pytest.raises(SystemExit):
        A.write_installers(base, wheel, tmp_path)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_constraints_are_the_locked_runtime_versions(tmp_path):
    out = tmp_path / "constraints.txt"
    A.export_constraints(out)
    pins = [ln for ln in out.read_text().splitlines() if ln and not ln.startswith((" ", "#"))]
    names = {ln.split("==")[0] for ln in pins}
    assert all("==" in ln for ln in pins)
    assert {"faster-whisper", "ctranslate2", "onnxruntime", "av", "numpy"} <= names
    assert not names & {"vlc-subsync", "pytest", "ruff", "piper-tts"}  # runtime only


FAKE_UV = """#!/bin/sh
case "$1" in
    --version) echo "uv ${FAKE_UV_VERSION:-0.11.32} (fake)" ;;
    tool) printf '%s\\n' "$@" > "$FAKE_UV_LOG"; exit 3 ;;
esac
"""


@pytest.fixture
def run_install_sh(tmp_path, monkeypatch):
    """Run an install.sh with a fake uv that records its `uv tool install` arguments."""
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("needs a POSIX sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "uv-args"
    for var in ("VLC_SUBSYNC_SOURCE", "VLC_SUBSYNC_CONSTRAINTS", "VLC_SUBSYNC_PYTHON"):
        monkeypatch.delenv(var, raising=False)

    def run(script: Path, **env: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
        log.unlink(missing_ok=True)
        full = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
        full.update(HOME=str(tmp_path), FAKE_UV_LOG=str(log), **env)
        proc = subprocess.run(
            ["sh", str(script), "--no-model"], env=full, capture_output=True, text=True
        )
        args = log.read_text().splitlines() if log.exists() else None
        return proc, args

    return run


def _constraints_arg(args):
    return args[args.index("--constraints") + 1] if "--constraints" in args else None


def test_install_sh_uses_the_release_constraints(tmp_path, run_install_sh):
    released = tmp_path / "released"
    A.write_installers(BASE, WHEEL, released)
    script = released / "install.sh"

    _, args = run_install_sh(script)
    assert _constraints_arg(args) == f"{BASE}/constraints.txt"
    assert args[-1] == f"vlc-subsync @ {BASE}/{WHEEL}"

    _, args = run_install_sh(script, VLC_SUBSYNC_CONSTRAINTS="none")
    assert _constraints_arg(args) is None

    _, args = run_install_sh(script, VLC_SUBSYNC_CONSTRAINTS="/x/c.txt")
    assert _constraints_arg(args) == "/x/c.txt"

    # another source does not get this release's versions unless asked
    _, args = run_install_sh(script, VLC_SUBSYNC_SOURCE=str(ROOT))
    assert _constraints_arg(args) is None
    url = args[-1].removeprefix("vlc-subsync @ ")
    assert url.startswith("file:///") and " " not in url
    assert os.path.samefile(unquote(urlsplit(url).path), ROOT)


def test_install_sh_from_main_installs_unpinned_unless_asked(run_install_sh):
    _, args = run_install_sh(ROOT / "install.sh")
    assert _constraints_arg(args) is None
    _, args = run_install_sh(ROOT / "install.sh", VLC_SUBSYNC_CONSTRAINTS="c.txt")
    assert _constraints_arg(args) == "c.txt"


def test_install_sh_stops_on_a_uv_too_old_for_constraints(run_install_sh):
    proc, args = run_install_sh(
        ROOT / "install.sh", VLC_SUBSYNC_CONSTRAINTS="c.txt", FAKE_UV_VERSION="0.5.5"
    )
    assert proc.returncode == 1 and args is None
    assert "uv 0.5.5 (fake) is too old" in proc.stderr
    proc, args = run_install_sh(ROOT / "install.sh", FAKE_UV_VERSION="0.5.5")
    assert args is not None  # no pins asked: any uv will do
    # 0.5.6 is the first uv with `uv tool install --constraints`
    proc, args = run_install_sh(
        ROOT / "install.sh", VLC_SUBSYNC_CONSTRAINTS="c.txt", FAKE_UV_VERSION="0.5.6"
    )
    assert _constraints_arg(args) == "c.txt", proc.stderr


def test_install_sh_passes_a_local_constraints_file_as_a_url(tmp_path, run_install_sh):
    # uv splits a --constraints value at spaces, so a local file goes in as a file:// URL
    folder = tmp_path / "My Files #1"
    folder.mkdir()
    pins = folder / "pins 100%.txt"
    pins.write_text("six==1.16.0\n")
    proc, args = run_install_sh(ROOT / "install.sh", VLC_SUBSYNC_CONSTRAINTS=str(pins))
    url = _constraints_arg(args)
    assert url is not None, proc.stderr
    assert url.startswith("file:///") and " " not in url
    assert os.path.samefile(unquote(urlsplit(url).path), pins)
    # a URL is passed on as it is
    _, args = run_install_sh(ROOT / "install.sh", VLC_SUBSYNC_CONSTRAINTS=url)
    assert _constraints_arg(args) == url

    uv = shutil.which("uv")
    if uv is None:
        return
    # and the real uv reads it back as that file (offline: nothing to resolve)
    (tmp_path / "empty.in").write_text("")
    proc = subprocess.run(
        [uv, "pip", "compile", "--no-config", "--offline", "-q", "empty.in", "-c", url],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
