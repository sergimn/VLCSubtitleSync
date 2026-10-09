"""scripts/version.py and scripts/release_notes.py, used by the release workflow."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

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
