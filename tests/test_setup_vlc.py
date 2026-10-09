from __future__ import annotations

import base64
import contextlib
import plistlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from vlcsubsync import protocol as P
from vlcsubsync import setup_vlc as S

# ------------------------------------------------------------------ vlcrc text editing

VLC_STYLE = (
    "﻿###\n###  vlc 3.0.24\n###\n\n"
    "[core] # core program\n\n"
    "# Extra interface modules (string)\n"
    "#extraintf=\n\n"
    "# Control interfaces (string)\n"
    "#control=\n\n"
    "[lua] # Lua interpreter\n\n"
    "# Lua interface (string)\n"
    "#lua-intf=dummy\n"
)


def active(text, key):
    vals = [ln.split("=", 1)[1] for ln in text.splitlines() if ln.startswith(f"{key}=")]
    return vals


def test_commented_lines_are_uncommented_in_place():
    edit = S.apply_vlcrc(VLC_STYLE)
    assert edit.changed and edit.added_luaintf and edit.previous_lua_intf is None
    lines = edit.text.splitlines()
    assert "extraintf=luaintf" in lines
    assert "lua-intf=subsync" in lines
    assert "#extraintf=" not in lines and "#lua-intf=dummy" not in lines
    # stays in its section, same position
    assert lines.index("extraintf=luaintf") < lines.index("[lua] # Lua interpreter")
    assert lines.index("lua-intf=subsync") > lines.index("[lua] # Lua interpreter")
    assert edit.text.startswith("﻿###")
    assert edit.text.endswith("\n")


def test_existing_extraintf_preserved():
    text = "[core]\nextraintf=http:telnet\n[lua]\nlua-intf=cli\n"
    edit = S.apply_vlcrc(text)
    assert active(edit.text, "extraintf") == ["http:telnet:luaintf"]
    assert active(edit.text, "lua-intf") == ["subsync"]
    assert edit.added_luaintf and edit.previous_lua_intf == "cli"
    back, changed = S.revert_vlcrc(edit.text, True, "cli")
    assert changed
    assert back == text


def test_missing_keys_are_inserted_in_sections():
    text = "[core]\nfoo=1\n[lua]\nbar=2\n"
    out = S.apply_vlcrc(text).text.splitlines()
    assert out[out.index("[core]") + 1] == "extraintf=luaintf"
    assert out[out.index("[lua]") + 1] == "lua-intf=subsync"


def test_no_sections_appends():
    out = S.apply_vlcrc("foo=1").text
    assert active(out, "extraintf") == ["luaintf"]
    assert active(out, "lua-intf") == ["subsync"]
    assert out.startswith("foo=1\n")


def test_apply_is_idempotent():
    once = S.apply_vlcrc(VLC_STYLE)
    twice = S.apply_vlcrc(once.text)
    assert twice.text == once.text
    assert not twice.changed and not twice.added_luaintf
    assert twice.previous_lua_intf == "subsync"


def test_luaintf_already_present_not_marked_added():
    text = "extraintf=luaintf\nlua-intf=other\n"
    edit = S.apply_vlcrc(text)
    assert not edit.added_luaintf
    back, _ = S.revert_vlcrc(edit.text, edit.added_luaintf, edit.previous_lua_intf)
    assert back == text


def test_revert_to_commented_defaults():
    edit = S.apply_vlcrc(VLC_STYLE)
    back, changed = S.revert_vlcrc(edit.text, True, None)
    assert changed
    assert active(back, "extraintf") == [] and active(back, "lua-intf") == []
    assert "#extraintf=" in back.splitlines() and "#lua-intf=dummy" in back.splitlines()


def test_revert_without_state_uses_heuristic():
    text = "extraintf=http:luaintf\nlua-intf=subsync\n"
    back, _ = S.revert_vlcrc(text, None, None)
    assert active(back, "extraintf") == ["http"]
    # not ours -> untouched
    text2 = "extraintf=luaintf\nlua-intf=mine\n"
    assert S.revert_vlcrc(text2, None, None) == (text2, False)


def test_crlf_preserved():
    text = "[core]\r\n#extraintf=\r\n[lua]\r\n#lua-intf=dummy\r\n"
    out = S.apply_vlcrc(text).text
    assert out == "[core]\r\nextraintf=luaintf\r\n[lua]\r\nlua-intf=subsync\r\n"


def test_minimal_vlcrc_is_vlc_parsable():
    """VLC 3 skips lines starting with '#', '[' or empty and needs key=value."""
    out = S.apply_vlcrc(S.MINIMAL_VLCRC).text
    opts = {}
    for line in out.split("\n"):
        if not line or line[0] in "#[" or "=" not in line:
            continue
        k, v = line.split("=", 1)
        opts[k] = v
    assert opts == {"extraintf": "luaintf", "lua-intf": "subsync"}
    assert out.endswith("\n")


# ------------------------------------------------------------------ fake systems


class FakeSystem:
    def __init__(self, tmp_path, platform, *, systemd=True, which=None):
        self.home = tmp_path / "home"
        self.root = tmp_path / "root"
        self.home.mkdir()
        self.root.mkdir()
        self.platform = platform
        self.systemd = systemd
        self.commands: list[list[str]] = []
        self.popens: list[list[str]] = []
        self.lines: list[str] = []
        self._which = which or {}
        self.env = (
            {"APPDATA": str(self.home / "AppData" / "Roaming")} if platform == "windows" else {}
        )

    def which(self, name):
        if name == "systemctl" and self.systemd:
            return "/usr/bin/systemctl"
        return self._which.get(name)

    def run(self, args, **kw):
        self.commands.append(list(args))
        if args[:3] == ["systemctl", "--user", "show-environment"]:
            return subprocess.CompletedProcess(args, 0 if self.systemd else 1, "", "")
        if args[:3] == ["systemctl", "--user", "is-enabled"]:
            return subprocess.CompletedProcess(args, 0, "enabled\n", "")
        if args[:3] == ["systemctl", "--user", "is-active"]:
            state = "active" if args[3].endswith(".path") else "inactive"
            return subprocess.CompletedProcess(args, 0, state + "\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    def popen(self, cmd, **kw):
        self.popens.append(list(cmd))

    def ctx(self, dry_run=False):
        return S.Context(
            platform=self.platform,
            home=self.home,
            env=self.env,
            root=self.root,
            dry_run=dry_run,
            run=self.run,
            which=self.which,
            out=self.lines.append,
            popen=self.popen,
            vlc_running=lambda: False,
        )


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    for name in ("CACHE", "STATE", "LOG", "CONFIG"):
        monkeypatch.setenv(f"VLC_SUBSYNC_{name}_DIR", str(tmp_path / "app" / name.lower()))


def scripts_installed(data_dir: Path) -> bool:
    return (data_dir / "lua/intf/subsync.lua").is_file() and (
        data_dir / "lua/extensions/subsync_ext.lua"
    ).is_file()


def test_packaged_scripts_present():
    scripts = S.packaged_scripts()
    assert [n for n, _ in scripts["intf"]] == ["subsync.lua"]
    assert [n for n, _ in scripts["extensions"]] == ["subsync_ext.lua"]


def test_packaged_icons_present():
    assert S.asset_path("icon.ico") is not None
    assert S.asset_path("icon-256.png") is not None


VLC_CATALOG = """<?xml version="1.0" encoding="UTF-8"?>
<videolan xmlns="http://videolan.org/ns/vlc/addons/1.0">
\t<addons>
\t\t<addon source="" type="extension" id="0123456789abcdef0123456789abcdef"
\t\t\tdownloads="3" score="5" version="1.2">
\t\t\t<name>Other</name>
\t\t\t<description><![CDATA[Someone else's <b>add-on</b>]]></description>
\t\t\t<authorship>
\t\t\t</authorship>
\t\t\t<resource type="extension">other.lua</resource>
\t\t</addon>
\t\t<addon source="" type="extension" id="{our_id}" downloads="0" score="0" version="0.0.1">
\t\t\t<name>SubSync</name>
\t\t\t<resource type="extension">subsync_ext.lua</resource>
\t\t</addon>
\t</addons>
</videolan>
"""


def _vlc_style_id(uid):
    # how VLC writes an id back (vlc_addons.h addons_uuid_to_psz)
    h = uid.replace("-", "")
    return f"{h[:8]}-{h[8:14]}-{h[14:18]}-{h[18:22]}-{h[22:]}"


def _catalog_addons(path):
    ns = {"v": S.ADDONS_NS}
    root = ET.parse(path).getroot()
    return {a.findtext("v:name", namespaces=ns): a for a in root.iterfind("v:addons/v:addon", ns)}


def test_setup_lists_addon_in_vlc_catalog(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    path = fs.home / ".local/share/vlc/catalog.xml"
    text = path.read_text(encoding="utf-8")
    # VLC compares element names with their prefix: the namespace must be the default one
    assert '<videolan xmlns="http://videolan.org/ns/vlc/addons/1.0">' in text
    ns = {"v": S.ADDONS_NS}
    [addon] = _catalog_addons(path).values()
    assert addon.get("id") == S.ADDON_ID and addon.get("type") == "extension"
    assert addon.get("version") == S.__version__
    assert addon.findtext("v:summary", namespaces=ns) == S.ADDON_SUMMARY
    assert addon.findtext("v:description", namespaces=ns) == S.ADDON_DESCRIPTION
    image = base64.b64decode(addon.findtext("v:image", namespaces=ns))
    assert image.startswith(b"\x89PNG")
    resources = [(r.get("type"), r.text) for r in addon.iterfind("v:resource", ns)]
    assert resources == [("interface", "subsync.lua"), ("extension", "subsync_ext.lua")]

    lines = []
    ctx = fs.ctx()
    ctx.out = lines.append
    S.run_setup(ctx, model=False, autostart=False)
    assert path.read_text(encoding="utf-8") == text
    assert any("lists SubSync" in line for line in lines)

    assert S.run_uninstall(fs.ctx()) == 0
    assert not path.exists()


def test_vlc_catalog_keeps_other_addons(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    path = fs.home / ".local/share/vlc/catalog.xml"
    path.parent.mkdir(parents=True)
    path.write_text(VLC_CATALOG.format(our_id=_vlc_style_id(S.ADDON_ID)), encoding="utf-8")
    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    addons = _catalog_addons(path)
    assert list(addons) == ["Other", "SubSync"]  # the stale entry was replaced
    other = addons["Other"]
    assert other.get("downloads") == "3"
    assert other.findtext(f"{{{S.ADDONS_NS}}}description") == "Someone else's <b>add-on</b>"
    assert addons["SubSync"].get("version") == S.__version__

    assert S.run_uninstall(fs.ctx()) == 0
    assert list(_catalog_addons(path)) == ["Other"]


def test_unreadable_vlc_catalog_is_left_alone(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    path = fs.home / ".local/share/vlc/catalog.xml"
    path.parent.mkdir(parents=True)
    path.write_text("<videolan><addons>", encoding="utf-8")
    ctx = fs.ctx()
    assert S.run_setup(ctx, model=False, autostart=False) == 0
    assert path.read_text(encoding="utf-8") == "<videolan><addons>"
    assert any("catalog" in w for w in ctx.warnings)
    assert scripts_installed(fs.home / ".local/share/vlc")
    assert S.run_uninstall(fs.ctx()) == 0
    assert path.read_text(encoding="utf-8") == "<videolan><addons>"


def test_linux_detection(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    ctx = fs.ctx()
    # nothing installed -> default native location, flagged not detected
    [inst] = S.detect_vlc_installs(ctx)
    assert inst.kind == "native" and not inst.detected
    assert S.detect_vlc_installs(ctx, include_default=False) == []

    (fs.root / "usr/bin").mkdir(parents=True)
    (fs.root / "usr/bin/vlc").write_text("")
    (fs.root / "snap/vlc").mkdir(parents=True)
    (fs.home / ".var/app/org.videolan.VLC").mkdir(parents=True)
    installs = {i.kind: i for i in S.detect_vlc_installs(ctx)}
    assert set(installs) == {"native", "snap", "flatpak"}
    assert installs["native"].vlcrc == fs.home / ".config/vlc/vlcrc"
    assert installs["native"].lua_dir == fs.home / ".local/share/vlc/lua"
    assert not installs["snap"].usable  # never started: no ~/snap/vlc/current
    (fs.home / "snap/vlc/current").mkdir(parents=True)
    snap = {i.kind: i for i in S.detect_vlc_installs(ctx)}["snap"]
    assert snap.usable
    # The snap launcher runs `vlc --config=$SNAP_USER_COMMON/vlcrc`.
    assert snap.vlcrc == fs.home / "snap/vlc/common/vlcrc"
    assert snap.queue_dir == fs.home / "snap/vlc/current/.local/share/vlc/subsync"
    fp = installs["flatpak"]
    assert fp.vlcrc == fs.home / ".var/app/org.videolan.VLC/config/vlc/vlcrc"
    assert fp.data_dir == fs.home / ".var/app/org.videolan.VLC/data/vlc"


def test_snap_binary_on_path_is_not_native(tmp_path):
    fs = FakeSystem(tmp_path, "linux", which={"vlc": "/snap/bin/vlc"})
    (fs.home / "snap/vlc/current").mkdir(parents=True)
    kinds = [i.kind for i in S.detect_vlc_installs(fs.ctx())]
    assert kinds == ["snap"]


def test_user_saved_prefs_then_rerun(tmp_path):
    """VLC rewrites vlcrc on 'Save preferences'; setup must cope."""
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    (fs.home / ".config/vlc").mkdir(parents=True)
    rc_path = fs.home / ".config/vlc/vlcrc"
    rc_path.write_text(VLC_STYLE, encoding="utf-8")
    S.run_setup(fs.ctx(), model=False, autostart=False)
    # user removed luaintf via prefs and VLC rewrote the file
    rc_path.write_text(VLC_STYLE.replace("#extraintf=", "extraintf=http"), encoding="utf-8")
    S.run_setup(fs.ctx(), model=False, autostart=False)
    assert active(rc_path.read_text(encoding="utf-8"), "extraintf") == ["http:luaintf"]
    S.run_uninstall(fs.ctx())
    assert active(rc_path.read_text(encoding="utf-8"), "extraintf") == ["http"]


def test_unusable_snap_is_skipped(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.root / "snap/vlc").mkdir(parents=True)
    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    assert any("never started" in w for w in fs.ctx().warnings) or any(
        "never started" in line for line in fs.lines
    )
    assert not (fs.home / "snap").exists()


def test_dry_run_changes_nothing(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / ".config/vlc").mkdir(parents=True)
    (fs.home / ".config/vlc/vlcrc").write_text(VLC_STYLE)
    before = sorted(p for p in tmp_path.rglob("*"))
    assert S.run_setup(fs.ctx(dry_run=True)) == 0
    assert sorted(p for p in tmp_path.rglob("*")) == before
    assert any("[plan]" in line for line in fs.lines)
    assert fs.popens == []
    assert not any(c[:3] == ["systemctl", "--user", "enable"] for c in fs.commands)


def test_custom_vlc_dir(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    custom = tmp_path / "portable"
    rc = tmp_path / "portable-rc"
    assert (
        S.run_setup(fs.ctx(), model=False, autostart=False, vlc_dirs=[str(custom)], vlcrc=str(rc))
        == 0
    )
    assert scripts_installed(custom)
    assert active(rc.read_text(encoding="utf-8-sig"), "lua-intf") == ["subsync"]


def test_model_download_failure_is_warning(tmp_path, monkeypatch):
    fs = FakeSystem(tmp_path, "linux", systemd=False)

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(S, "download_models", boom)
    ctx = fs.ctx()
    assert S.run_setup(ctx, autostart=False) == 0
    assert any("offline" in w for w in ctx.warnings)


def test_vlc_running_warning(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    ctx = fs.ctx()
    ctx.vlc_running = lambda: True
    S.run_setup(ctx, model=False, autostart=False)
    assert any("restart VLC" in w for w in ctx.warnings)


def test_daemon_command_prefers_venv_script(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "vlc-subsync").write_text("")
    monkeypatch.setattr(S, "_bin_dir", lambda: bindir)
    ctx = S.Context(platform="linux", home=tmp_path, env={}, which=lambda n: None)
    assert S.daemon_command(ctx) == [str(bindir / "vlc-subsync"), "serve"]
    (bindir / "vlc-subsync").unlink()
    assert S.daemon_command(ctx)[1:] == ["-m", "vlcsubsync.cli", "serve"]
    wctx = S.Context(platform="windows", home=tmp_path, env={}, which=lambda n: None)
    (bindir / "vlc-subsync-daemon.exe").write_text("")
    assert S.daemon_command(wctx, gui=True) == [str(bindir / "vlc-subsync-daemon.exe")]


def test_setup_cleans_legacy_snap_vlcrc(tmp_path):
    """0.1.0 wrote the snap vlcrc VLC never reads; setup moves the settings to common/."""
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / "snap/vlc/current").mkdir(parents=True)
    legacy_dir = fs.home / "snap/vlc/current/.config/vlc"
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / "vlcrc"
    legacy.write_text("[core]\nextraintf=luaintf\n[lua]\nlua-intf=subsync\n", encoding="utf-8")
    (legacy_dir / "vlcrc.subsync-state").write_text(
        "added_luaintf=1\nprevious_lua_intf=\ncreated_vlcrc=1\n", encoding="utf-8"
    )

    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    assert not legacy.exists()
    assert not (legacy_dir / "vlcrc.subsync-state").exists()
    text = (fs.home / "snap/vlc/common/vlcrc").read_text(encoding="utf-8")
    assert active(text, "extraintf") == ["luaintf"]
    assert active(text, "lua-intf") == ["subsync"]


def _legacy_snap(fs, text, *, created, marker=True):
    (fs.home / "snap/vlc/current").mkdir(parents=True, exist_ok=True)
    d = fs.home / "snap/vlc/current/.config/vlc"
    d.mkdir(parents=True, exist_ok=True)
    (d / "vlcrc").write_text(text, encoding="utf-8")
    if marker:
        (d / "vlcrc.subsync-state").write_text(
            f"added_luaintf=1\nprevious_lua_intf=\ncreated_vlcrc={int(created)}\n",
            encoding="utf-8",
        )
    return d / "vlcrc"


OURS = "[core]\nextraintf=luaintf\n[lua]\nlua-intf=subsync\n"


def test_legacy_created_vlcrc_with_user_settings_is_kept(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    legacy = _legacy_snap(fs, OURS + "[core]\nvolume=200\n", created=True)
    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    text = legacy.read_text(encoding="utf-8")
    assert "volume=200" in text
    assert active(text, "lua-intf") == []  # our settings are reverted all the same


def test_legacy_not_created_by_us_is_reverted_not_removed(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    legacy = _legacy_snap(fs, OURS, created=False)
    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    assert legacy.exists()
    assert active(legacy.read_text(encoding="utf-8"), "lua-intf") == []


def test_legacy_without_marker_is_untouched(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    legacy = _legacy_snap(fs, OURS, created=True, marker=False)
    assert S.run_setup(fs.ctx(), model=False, autostart=False) == 0
    assert legacy.read_text(encoding="utf-8") == OURS


def test_legacy_cleanup_dry_run_touches_nothing(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    legacy = _legacy_snap(fs, OURS, created=True)
    ctx = fs.ctx()
    ctx.dry_run = True
    assert S.run_setup(ctx, model=False, autostart=False) == 0
    assert legacy.read_text(encoding="utf-8") == OURS
    assert (legacy.parent / "vlcrc.subsync-state").exists()
    assert not (fs.home / "snap/vlc/common/vlcrc").exists()


def test_uninstall_cleans_legacy_snap_vlcrc(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    legacy = _legacy_snap(fs, OURS, created=True)
    assert S.run_uninstall(fs.ctx()) == 0
    assert not legacy.exists()
    assert not (legacy.parent / "vlcrc.subsync-state").exists()


# ------------------------------------------------------------------ helper lifecycle


def _snap_with_vlcrc(fs):
    (fs.home / "snap/vlc/current").mkdir(parents=True)
    (fs.home / "snap/vlc/common").mkdir(parents=True)
    rc_path = fs.home / "snap/vlc/common/vlcrc"
    rc_path.write_text(VLC_STYLE.replace("#extraintf=", "extraintf=http"), encoding="utf-8")
    return rc_path


def test_setup_and_uninstall_linux_systemd(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    rc_path = _snap_with_vlcrc(fs)
    original = rc_path.read_bytes()

    assert S.run_setup(fs.ctx(), model=False) == 0
    data = fs.home / "snap/vlc/current/.local/share/vlc"
    q = data / "subsync"
    assert scripts_installed(data)
    text = rc_path.read_text(encoding="utf-8")
    assert active(text, "extraintf") == ["http:luaintf"]
    assert active(text, "lua-intf") == ["subsync"]
    assert (rc_path.parent / "vlcrc.subsync-backup").read_bytes() == original
    for sub in ("requests", "jobs", "out"):
        assert (q / sub).is_dir()

    units = fs.home / ".config/systemd/user"
    service = (units / "vlc-subsync.service").read_text()
    path_unit = (units / "vlc-subsync.path").read_text().splitlines()
    assert "ExecStart=" in service and " serve" in service
    assert "[Install]" not in service  # never started at login by itself
    assert f"PathModified={q / 'intf_state'}" in path_unit
    assert f"DirectoryNotEmpty={q / 'requests'}" in path_unit
    assert ["systemctl", "--user", "enable", "--now", "vlc-subsync.path"] in fs.commands
    # nothing is started or enabled for login
    assert not any(c[-1] == "vlc-subsync.service" and "enable" in c for c in fs.commands)
    assert not any("restart" in c or "start" in c for c in fs.commands)
    assert P.read_kv(q / "launcher")["mode"] == "service"
    assert fs.popens == []

    # idempotent re-run
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert rc_path.read_text(encoding="utf-8") == text
    assert (rc_path.parent / "vlcrc.subsync-backup").read_bytes() == original
    assert P.read_kv(rc_path.parent / "vlcrc.subsync-state")["added_luaintf"] == "1"

    fs.commands.clear()
    assert S.run_uninstall(fs.ctx()) == 0
    assert rc_path.read_bytes() == original
    assert not scripts_installed(data)
    assert not q.exists()
    assert not (units / "vlc-subsync.service").exists()
    assert not (units / "vlc-subsync.path").exists()
    assert not (rc_path.parent / "vlcrc.subsync-state").exists()
    assert not (rc_path.parent / "vlcrc.subsync-backup").exists()
    assert ["systemctl", "--user", "disable", "--now", "vlc-subsync.path"] in fs.commands
    assert ["systemctl", "--user", "stop", "vlc-subsync.service"] in fs.commands
    assert ["systemctl", "--user", "daemon-reload"] in fs.commands


def test_setup_without_systemd_uses_lua_spawn(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    (fs.home / ".config/vlc").mkdir(parents=True)
    _snap_with_vlcrc(fs)
    ctx = fs.ctx()
    assert S.run_setup(ctx, model=False) == 0
    rc_path = fs.home / ".config/vlc/vlcrc"
    text = rc_path.read_text(encoding="utf-8-sig")
    assert active(text, "extraintf") == ["luaintf"]
    assert active(text, "lua-intf") == ["subsync"]
    native_q = fs.home / ".local/share/vlc/subsync"
    launcher = P.read_kv(native_q / "launcher")
    assert launcher["mode"] == "spawn"
    assert launcher["exe"] and launcher["args"].endswith("serve")
    # no login autostart of any kind, nothing started now
    assert not (fs.home / ".config/autostart/vlc-subsync.desktop").exists()
    assert not (fs.home / ".config/systemd/user/vlc-subsync.path").exists()
    assert fs.popens == []
    # the sandboxed snap cannot spawn the helper: warned
    assert any("sandboxed" in w for w in ctx.warnings)

    assert S.run_uninstall(fs.ctx()) == 0
    text = rc_path.read_text(encoding="utf-8-sig")
    assert active(text, "extraintf") == [] and active(text, "lua-intf") == []
    assert not native_q.exists()


def test_systemd_unit_rendering():
    text = S.systemd_service_text(["/home/a b/bin/vlc-subsync", "serve", "50%"])
    assert 'ExecStart="/home/a b/bin/vlc-subsync" serve 50%%' in text
    for line in ("Type=simple", "Nice=10", "IOSchedulingClass=idle", "CPUSchedulingPolicy=batch"):
        assert line in text.splitlines()
    assert "Restart" not in text and "[Install]" not in text
    env = S.systemd_service_text(["x"], environment={"VLC_SUBSYNC_STATE_DIR": "/t/s"})
    assert "Environment=VLC_SUBSYNC_STATE_DIR=/t/s" in env

    qs = [Path("/h/.local/share/vlc/subsync"), Path("/h/snap/vlc/current/.local/share/vlc/subsync")]
    lines = S.systemd_path_text(qs).splitlines()
    assert lines.index("[Path]") < lines.index("Unit=vlc-subsync.service")
    assert f"PathModified={qs[1] / 'intf_state'}" in lines
    assert f"DirectoryNotEmpty={qs[0] / 'requests'}" in lines
    assert "WantedBy=default.target" in lines
    custom = S.systemd_path_text([Path("/q%x")], service="vlc-subsync-test.service")
    assert "Unit=vlc-subsync-test.service" in custom
    assert "PathModified=" + str(Path("/q%%x") / "intf_state") in custom


def test_launchd_plist_rendering(tmp_path):
    qs = [tmp_path / "q1", tmp_path / "q2"]
    plist = plistlib.loads(S.launchd_plist(["/bin/vlc-subsync", "serve"], tmp_path, qs))
    assert plist["WatchPaths"] == [str(q / "intf_state") for q in qs]
    assert plist["QueueDirectories"] == [str(q / "requests") for q in qs]
    assert plist["ProcessType"] == "Background" and plist["LowPriorityIO"] is True
    assert "KeepAlive" not in plist and "RunAtLoad" not in plist
    assert plist["ProgramArguments"] == ["/bin/vlc-subsync", "serve"]


def test_setup_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_uid", lambda: 501)
    fs = FakeSystem(tmp_path, "macos")
    (fs.root / "Applications/VLC.app").mkdir(parents=True)
    [inst] = S.detect_vlc_installs(fs.ctx())
    assert inst.kind == "macos" and inst.detected
    assert S.run_setup(fs.ctx(), model=False) == 0
    data = fs.home / "Library/Application Support/org.videolan.vlc"
    assert scripts_installed(data)
    rc = fs.home / "Library/Preferences/org.videolan.vlc/vlcrc"
    assert active(rc.read_text(encoding="utf-8-sig"), "lua-intf") == ["subsync"]
    assert (data / "subsync/requests").is_dir()
    plist_path = fs.home / "Library/LaunchAgents" / f"{S.LAUNCHD_LABEL}.plist"
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["Label"] == S.LAUNCHD_LABEL
    assert plist["ProgramArguments"][-1] == "serve"
    assert "RunAtLoad" not in plist and "KeepAlive" not in plist
    assert plist["WatchPaths"] == [str(data / "subsync" / "intf_state")]
    assert ["launchctl", "bootstrap", "gui/501", str(plist_path)] in fs.commands
    assert P.read_kv(data / "subsync" / "launcher")["mode"] == "service"

    assert S.run_uninstall(fs.ctx()) == 0
    assert not plist_path.exists()
    assert not scripts_installed(data)
    assert ["launchctl", "bootout", f"gui/501/{S.LAUNCHD_LABEL}"] in fs.commands


def test_setup_macos_replaces_keepalive_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_uid", lambda: 501)
    fs = FakeSystem(tmp_path, "macos")
    (fs.root / "Applications/VLC.app").mkdir(parents=True)
    plist_path = fs.home / "Library/LaunchAgents" / f"{S.LAUNCHD_LABEL}.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(
        plistlib.dumps({"Label": S.LAUNCHD_LABEL, "RunAtLoad": True, "KeepAlive": {}})
    )
    assert S.legacy_autostart_artifacts(fs.ctx()) == [plist_path]
    assert S.run_setup(fs.ctx(), model=False) == 0
    plist = plistlib.loads(plist_path.read_bytes())
    assert "KeepAlive" not in plist and "RunAtLoad" not in plist
    assert S.legacy_autostart_artifacts(fs.ctx()) == []


def test_setup_windows_launcher_and_no_startup_shortcut(tmp_path, monkeypatch):
    bindir = tmp_path / "Scripts"
    bindir.mkdir()
    exe = bindir / "vlc-subsync-daemon.exe"
    exe.write_text("")
    monkeypatch.setattr(S, "_bin_dir", lambda: bindir)
    fs = FakeSystem(tmp_path, "windows")
    appdata = fs.home / "AppData/Roaming"
    (appdata / "vlc").mkdir(parents=True)
    [inst] = S.detect_vlc_installs(fs.ctx())
    assert inst.kind == "windows" and inst.vlcrc == appdata / "vlc" / "vlcrc"
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert scripts_installed(appdata / "vlc")
    assert active((appdata / "vlc/vlcrc").read_text(encoding="utf-8-sig"), "extraintf") == [
        "luaintf"
    ]
    assert (appdata / "vlc/subsync/out").is_dir()
    launcher = P.read_kv(appdata / "vlc/subsync/launcher")
    assert launcher == {"version": "1", "mode": "spawn", "exe": str(exe), "args": ""}
    # no login autostart, nothing started now
    assert not (S.windows_startup_dir(fs.ctx()) / S.WINDOWS_SHORTCUT).exists()
    assert not any(c[0] == "powershell" for c in fs.commands)
    assert fs.popens == []
    rows = S.lifecycle_status(fs.ctx(), [inst.queue_dir])
    assert rows and rows[0][2] is True and str(exe) in rows[0][1]

    assert S.run_uninstall(fs.ctx()) == 0
    assert not scripts_installed(appdata / "vlc")
    assert not (appdata / "vlc/subsync").exists()


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_ALL_ACCESS = 0xF003F

    def __init__(self, values):
        self.values = values

    def OpenKey(self, root, path, reserved, access):  # noqa: N802
        assert path.endswith(r"CurrentVersion\Run")
        return contextlib.nullcontext(self)

    def DeleteValue(self, key, name):  # noqa: N802
        if name not in self.values:
            raise FileNotFoundError(name)
        del self.values[name]


def test_setup_windows_removes_startup_shortcut_and_run_key(tmp_path, monkeypatch):
    reg = FakeWinreg({"SubSync": "C:\\x\\vlc-subsync-daemon.exe", "Other": "keep"})
    monkeypatch.setitem(sys.modules, "winreg", reg)
    fs = FakeSystem(tmp_path, "windows")
    (fs.home / "AppData/Roaming/vlc").mkdir(parents=True)
    lnk = S.windows_startup_dir(fs.ctx()) / S.WINDOWS_SHORTCUT
    lnk.parent.mkdir(parents=True)
    lnk.write_bytes(b"lnk")
    assert S.legacy_autostart_artifacts(fs.ctx()) == [lnk]
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert not lnk.exists()
    assert reg.values == {"Other": "keep"}
    assert any("Run" in line and "removed" in line for line in fs.lines)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_setup_linux_migrates_always_on_service(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / ".config/vlc").mkdir(parents=True)
    units = fs.home / ".config/systemd/user"
    (units / "default.target.wants").mkdir(parents=True)
    old = units / "vlc-subsync.service"
    old.write_text(
        "[Unit]\nDescription=old\n[Service]\nExecStart=/x serve\nRestart=on-failure\n"
        "[Install]\nWantedBy=default.target\n"
    )
    (units / "default.target.wants" / "vlc-subsync.service").symlink_to(old)
    desktop = fs.home / ".config/autostart/vlc-subsync.desktop"
    desktop.parent.mkdir(parents=True)
    desktop.write_text("[Desktop Entry]\nExec=/x serve\n")
    assert len(S.legacy_autostart_artifacts(fs.ctx())) == 2

    assert S.run_setup(fs.ctx(), model=False) == 0
    disable = ["systemctl", "--user", "disable", "--now", "vlc-subsync.service"]
    enable = ["systemctl", "--user", "enable", "--now", "vlc-subsync.path"]
    # the old service is stopped and disabled before its unit file is replaced
    assert fs.commands.index(disable) < fs.commands.index(enable)
    assert not (units / "default.target.wants" / "vlc-subsync.service").is_symlink()
    assert not desktop.exists()
    assert "[Install]" not in old.read_text() and "Restart" not in old.read_text()
    assert S.legacy_autostart_artifacts(fs.ctx()) == []


def test_uninstall_removes_legacy_artifacts(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    (fs.home / ".config/vlc").mkdir(parents=True)
    units = fs.home / ".config/systemd/user"
    units.mkdir(parents=True)
    (units / "vlc-subsync.service").write_text("[Service]\n[Install]\nWantedBy=default.target\n")
    desktop = fs.home / ".config/autostart/vlc-subsync.desktop"
    desktop.parent.mkdir(parents=True)
    desktop.write_text("[Desktop Entry]\n")
    assert S.run_uninstall(fs.ctx()) == 0
    assert not desktop.exists()
    assert not (units / "vlc-subsync.service").exists()


def test_dry_run_migration_touches_nothing(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    units = fs.home / ".config/systemd/user"
    units.mkdir(parents=True)
    (units / "vlc-subsync.service").write_text("[Service]\n[Install]\nWantedBy=default.target\n")
    before = sorted(p for p in tmp_path.rglob("*"))
    assert S.run_setup(fs.ctx(dry_run=True), model=False) == 0
    assert sorted(p for p in tmp_path.rglob("*")) == before
    assert not any("disable" in c for c in fs.commands)


def test_lifecycle_status_linux(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / ".config/vlc").mkdir(parents=True)
    rows = S.lifecycle_status(fs.ctx())
    assert rows[0][2] is False and "not configured" in rows[0][1]
    S.run_setup(fs.ctx(), model=False)
    rows = {label: (value, ok) for label, value, ok in S.lifecycle_status(fs.ctx())}
    assert rows["vlc-subsync.path"][1] is True
    assert rows["vlc-subsync.service"] == ("inactive (normal while VLC is closed)", None)


def test_launcher_data_windows_and_posix():
    w = S.launcher_data("spawn", [r"C:\Py\Scripts\vlc-subsync-daemon.exe"], "windows")
    assert w == {
        "version": 1,
        "mode": "spawn",
        "exe": r"C:\Py\Scripts\vlc-subsync-daemon.exe",
        "args": "",
    }
    p = S.launcher_data("service", ["/usr/bin/python3", "-m", "vlcsubsync.cli", "serve"], "linux")
    assert p["exe"] == "/usr/bin/python3" and p["args"] == "-m vlcsubsync.cli serve"


def test_service_has_start_limit():
    lines = S.systemd_service_text(["/x", "serve"]).splitlines()
    unit = lines[: lines.index("[Service]")]
    assert "StartLimitIntervalSec=600" in unit and "StartLimitBurst=20" in unit


def test_resetup_restarts_path_unit_when_watches_change(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / ".config/vlc").mkdir(parents=True)
    restart = ["systemctl", "--user", "restart", "vlc-subsync.path"]
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert restart not in fs.commands  # first install: enable --now is enough

    fs.commands.clear()
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert restart not in fs.commands  # unchanged watches

    # a new VLC (the snap) shows up: the path unit must watch it too
    (fs.home / "snap/vlc/current").mkdir(parents=True)
    fs.commands.clear()
    assert S.run_setup(fs.ctx(), model=False) == 0
    path_unit = (fs.home / ".config/systemd/user/vlc-subsync.path").read_text()
    assert str(Path("snap/vlc/current")) in path_unit
    reload_ = ["systemctl", "--user", "daemon-reload"]
    assert reload_ in fs.commands and restart in fs.commands
    assert fs.commands.index(reload_) < fs.commands.index(restart)


def test_windows_percent_in_exe_path_warns(tmp_path, monkeypatch):
    bindir = tmp_path / "50%off" / "Scripts"
    bindir.mkdir(parents=True)
    (bindir / "vlc-subsync-daemon.exe").write_text("")
    monkeypatch.setattr(S, "_bin_dir", lambda: bindir)
    fs = FakeSystem(tmp_path, "windows")
    (fs.home / "AppData/Roaming/vlc").mkdir(parents=True)
    ctx = fs.ctx()
    assert S.run_setup(ctx, model=False) == 0
    assert any("'%'" in w for w in ctx.warnings)


# ------------------------------------------------------------------ setup --vlc-dir


def _queue_dir_pairs(args):
    return [args[i + 1] for i, a in enumerate(args[:-1]) if a == "--queue-dir"]


def test_daemon_command_passes_queue_dirs(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "vlc-subsync").write_text("")
    monkeypatch.setattr(S, "_bin_dir", lambda: bindir)
    ctx = S.Context(platform="linux", home=tmp_path, env={})
    q = tmp_path / "my vlc" / "subsync"
    assert S.daemon_command(ctx, queue_dirs=[q]) == [
        str(bindir / "vlc-subsync"),
        "serve",
        "--queue-dir",
        str(q),
    ]
    assert S.daemon_command(ctx) == [str(bindir / "vlc-subsync"), "serve"]


def test_setup_vlc_dir_systemd_keeps_default_and_watches_custom(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / ".config/vlc").mkdir(parents=True)
    native_q = fs.home / ".local/share/vlc/subsync"
    custom = tmp_path / "portable"
    custom_q = custom / "subsync"
    assert S.run_setup(fs.ctx(), model=False) == 0  # the detected VLC first
    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(custom)]) == 0

    units = fs.home / ".config/systemd/user"
    service = (units / "vlc-subsync.service").read_text()
    [exec_start] = [ln for ln in service.splitlines() if ln.startswith("ExecStart=")]
    assert exec_start.endswith(f" serve --queue-dir {S._systemd_quote(str(custom_q))}")
    path_unit = (units / "vlc-subsync.path").read_text().splitlines()
    for q in (native_q, custom_q):
        assert f"PathModified={q / 'intf_state'}" in path_unit
        assert f"DirectoryNotEmpty={q / 'requests'}" in path_unit
    launcher = P.read_kv(custom_q / "launcher")
    assert launcher["mode"] == "service"
    assert _queue_dir_pairs(launcher["args"].split()) == [str(custom_q)]
    rows = {label: (value, ok) for label, value, ok in S.lifecycle_status(fs.ctx(), [native_q])}
    assert rows["--vlc-dir queue dir"] == (str(custom_q), True)

    # removing only the custom dir keeps the helper starting for the detected VLC
    assert S.run_uninstall(fs.ctx(), vlc_dirs=[str(custom)]) == 0
    assert not custom_q.exists()
    service = (units / "vlc-subsync.service").read_text()
    assert "--queue-dir" not in service
    path_unit = (units / "vlc-subsync.path").read_text()
    assert str(native_q / "intf_state") in path_unit and str(custom_q) not in path_unit
    assert native_q.is_dir() and P.read_kv(native_q / "launcher")["mode"] == "service"
    assert ["systemctl", "--user", "restart", "vlc-subsync.path"] in fs.commands


@pytest.mark.parametrize("systemd", [True, False])
def test_vlc_dirs_accumulate_across_setups(tmp_path, systemd):
    """A later `setup` (plain, as the installer runs on upgrade, or for another
    --vlc-dir) keeps the --vlc-dir dirs set up earlier; `uninstall --vlc-dir`
    removes only its own."""
    fs = FakeSystem(tmp_path, "linux", systemd=systemd)
    (fs.home / ".config/vlc").mkdir(parents=True)
    native_q = fs.home / ".local/share/vlc/subsync"
    x_q, y_q = tmp_path / "x" / "subsync", tmp_path / "y" / "subsync"

    def passed(q):
        return _queue_dir_pairs(P.read_kv(q / "launcher")["args"].split())

    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(x_q.parent)]) == 0
    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(y_q.parent)]) == 0
    assert S.run_setup(fs.ctx(), model=False) == 0
    for q in (native_q, x_q, y_q):
        assert sorted(passed(q)) == sorted([str(x_q), str(y_q)])
    assert S.installed_queue_dir_args(fs.ctx(), [native_q]) == [y_q, x_q]

    assert S.run_uninstall(fs.ctx(), vlc_dirs=[str(x_q.parent)]) == 0
    assert not x_q.exists()
    assert passed(native_q) == [str(y_q)] and passed(y_q) == [str(y_q)]
    assert S.run_uninstall(fs.ctx(), vlc_dirs=[str(y_q.parent)]) == 0
    assert passed(native_q) == []


def test_setup_vlc_dir_only_custom_when_nothing_detected(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    custom_q = tmp_path / "portable" / "subsync"
    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(custom_q.parent)]) == 0
    path_unit = (fs.home / ".config/systemd/user/vlc-subsync.path").read_text()
    assert path_unit.count("PathModified=") == 1 and str(custom_q) in path_unit
    # uninstalling the only configured dir removes the units
    assert S.run_uninstall(fs.ctx(), vlc_dirs=[str(custom_q.parent)]) == 0
    assert not (fs.home / ".config/systemd/user/vlc-subsync.path").exists()


def test_setup_vlc_dir_skips_unusable_detected_install(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.root / "snap/vlc").mkdir(parents=True)  # snap installed, never started
    custom_q = tmp_path / "portable" / "subsync"
    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(custom_q.parent)]) == 0
    path_unit = (fs.home / ".config/systemd/user/vlc-subsync.path").read_text()
    assert "snap" not in path_unit and str(custom_q) in path_unit


def test_setup_vlc_dir_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_uid", lambda: 501)
    fs = FakeSystem(tmp_path, "macos")
    (fs.root / "Applications/VLC.app").mkdir(parents=True)
    data_q = fs.home / "Library/Application Support/org.videolan.vlc/subsync"
    custom_q = tmp_path / "portable vlc" / "subsync"
    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(custom_q.parent)]) == 0
    plist_path = fs.home / "Library/LaunchAgents" / f"{S.LAUNCHD_LABEL}.plist"
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["ProgramArguments"][-3:] == ["serve", "--queue-dir", str(custom_q)]
    assert plist["WatchPaths"] == [str(custom_q / "intf_state"), str(data_q / "intf_state")]
    assert plist["QueueDirectories"] == [str(custom_q / "requests"), str(data_q / "requests")]
    assert S.installed_queue_dir_args(fs.ctx()) == [custom_q]


def test_setup_vlc_dir_without_systemd_spawns_with_queue_dir(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    (fs.home / ".config/vlc").mkdir(parents=True)
    native_q = fs.home / ".local/share/vlc/subsync"
    custom_q = tmp_path / "portable" / "subsync"
    spaced_q = tmp_path / "my vlc" / "subsync"
    ctx = fs.ctx()
    dirs = [str(custom_q.parent), str(spaced_q.parent)]
    assert S.run_setup(ctx, model=False, vlc_dirs=dirs) == 0
    for q in (native_q, custom_q):
        launcher = P.read_kv(q / "launcher")
        assert launcher["mode"] == "spawn"
        # the Lua side splits args on whitespace: the spaced path cannot be passed
        assert _queue_dir_pairs(launcher["args"].split()) == [str(custom_q)]
    assert not (spaced_q / "launcher").exists()
    assert any("spaces" in w and str(spaced_q) in w for w in ctx.warnings)


def test_setup_vlc_dir_windows_launchers(tmp_path, monkeypatch):
    bindir = tmp_path / "Scripts"
    bindir.mkdir()
    exe = bindir / "vlc-subsync-daemon.exe"
    exe.write_text("")
    monkeypatch.setattr(S, "_bin_dir", lambda: bindir)
    fs = FakeSystem(tmp_path, "windows")
    appdata_q = fs.home / "AppData/Roaming/vlc/subsync"
    appdata_q.parent.mkdir(parents=True)
    custom_q = tmp_path / "portable" / "subsync"
    assert S.run_setup(fs.ctx(), model=False, vlc_dirs=[str(custom_q.parent)]) == 0
    for q in (appdata_q, custom_q):
        assert P.read_kv(q / "launcher") == {
            "version": "1",
            "mode": "spawn",
            "exe": str(exe),
            "args": f"--queue-dir {custom_q}",
        }
