from __future__ import annotations

import plistlib
import subprocess
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
    def __init__(self, tmp_path, platform, *, systemd=True, which=None, lnk_ok=True):
        self.home = tmp_path / "home"
        self.root = tmp_path / "root"
        self.home.mkdir()
        self.root.mkdir()
        self.platform = platform
        self.systemd = systemd
        self.lnk_ok = lnk_ok
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
        if args and args[0] == "powershell" and self.lnk_ok:
            lnk = S.windows_startup_dir(self.ctx()) / S.WINDOWS_SHORTCUT
            lnk.write_bytes(b"lnk")
            return subprocess.CompletedProcess(args, 0, "", "")
        if args and args[0] == "powershell":
            return subprocess.CompletedProcess(args, 1, "", "COM error")
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


def test_setup_and_uninstall_linux_systemd(tmp_path):
    fs = FakeSystem(tmp_path, "linux")
    (fs.home / "snap/vlc/current").mkdir(parents=True)
    (fs.home / "snap/vlc/common").mkdir(parents=True)
    rc_path = fs.home / "snap/vlc/common/vlcrc"
    rc_path.write_text(VLC_STYLE.replace("#extraintf=", "extraintf=http"), encoding="utf-8")
    original = rc_path.read_bytes()

    assert S.run_setup(fs.ctx(), model=False) == 0
    data = fs.home / "snap/vlc/current/.local/share/vlc"
    assert scripts_installed(data)
    text = rc_path.read_text(encoding="utf-8")
    assert active(text, "extraintf") == ["http:luaintf"]
    assert active(text, "lua-intf") == ["subsync"]
    assert (rc_path.parent / "vlcrc.subsync-backup").read_bytes() == original
    for sub in ("requests", "jobs", "out"):
        assert (data / "subsync" / sub).is_dir()
    unit = fs.home / ".config/systemd/user/vlc-subsync.service"
    assert "ExecStart=" in unit.read_text() and " serve" in unit.read_text()
    assert ["systemctl", "--user", "enable", "vlc-subsync.service"] in fs.commands
    assert ["systemctl", "--user", "restart", "vlc-subsync.service"] in fs.commands

    # idempotent re-run
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert rc_path.read_text(encoding="utf-8") == text
    assert (rc_path.parent / "vlcrc.subsync-backup").read_bytes() == original
    state = P.read_kv(rc_path.parent / "vlcrc.subsync-state")
    assert state["added_luaintf"] == "1"

    assert S.run_uninstall(fs.ctx()) == 0
    assert rc_path.read_bytes() == original
    assert not scripts_installed(data)
    assert not (data / "subsync").exists()
    assert not unit.exists()
    assert not (rc_path.parent / "vlcrc.subsync-state").exists()
    assert not (rc_path.parent / "vlcrc.subsync-backup").exists()
    assert ["systemctl", "--user", "disable", "--now", "vlc-subsync.service"] in fs.commands


def test_setup_creates_missing_vlcrc_and_uninstall(tmp_path):
    fs = FakeSystem(tmp_path, "linux", systemd=False)
    (fs.home / ".config/vlc").mkdir(parents=True)
    assert S.run_setup(fs.ctx(), model=False) == 0
    rc_path = fs.home / ".config/vlc/vlcrc"
    text = rc_path.read_text(encoding="utf-8-sig")
    assert active(text, "extraintf") == ["luaintf"]
    assert active(text, "lua-intf") == ["subsync"]
    assert not (rc_path.parent / "vlcrc.subsync-backup").exists()
    # no systemd -> XDG autostart + detached start
    desktop = fs.home / ".config/autostart/vlc-subsync.desktop"
    content = desktop.read_text()
    assert "Exec=" in content and "Name=SubSync" in content and "Icon=" in content
    assert fs.popens and fs.popens[-1][-1] == "serve"

    assert S.run_uninstall(fs.ctx()) == 0
    text = rc_path.read_text(encoding="utf-8-sig")
    assert active(text, "extraintf") == [] and active(text, "lua-intf") == []
    assert not desktop.exists()


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
    assert plist["RunAtLoad"] is True
    assert ["launchctl", "bootstrap", "gui/501", str(plist_path)] in fs.commands

    assert S.run_uninstall(fs.ctx()) == 0
    assert not plist_path.exists()
    assert not scripts_installed(data)
    assert ["launchctl", "bootout", f"gui/501/{S.LAUNCHD_LABEL}"] in fs.commands


def test_setup_windows(tmp_path):
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
    ps = [c for c in fs.commands if c[0] == "powershell"]
    assert ps and "WScript.Shell" in ps[0][-1] and "IconLocation" in ps[0][-1]
    lnk = S.windows_startup_dir(fs.ctx()) / S.WINDOWS_SHORTCUT
    assert lnk.exists()
    assert fs.popens  # daemon started detached

    assert S.run_uninstall(fs.ctx()) == 0
    assert not lnk.exists()
    assert not scripts_installed(appdata / "vlc")


def test_windows_shortcut_failure_falls_back(tmp_path, monkeypatch):
    fs = FakeSystem(tmp_path, "windows", lnk_ok=False)
    (fs.home / "AppData/Roaming/vlc").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(S, "_set_run_key", lambda ctx, cmd: calls.append(cmd) or True)
    assert S.run_setup(fs.ctx(), model=False) == 0
    assert calls and calls[0]


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


def test_systemd_unit_quoting():
    text = S.systemd_unit_text(["/home/a b/bin/vlc-subsync", "serve"])
    assert 'ExecStart="/home/a b/bin/vlc-subsync" serve' in text


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
