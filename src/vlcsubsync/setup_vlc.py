"""Install / uninstall the VLC integration.

* detect VLC installs (Linux native / snap / flatpak, macOS, Windows)
* copy the Lua scripts (package data) into ``<userdatadir>/lua/{intf,extensions}``
* edit ``vlcrc`` (``extraintf`` += ``luaintf``, ``lua-intf=subsync``) with a backup and a
  small state file so ``uninstall`` can revert exactly what we changed
* create the queue dirs
* register autostart (systemd user unit / XDG autostart / launchd agent / Windows
  Startup shortcut or HKCU Run key) and start the daemon
* pre-download the default Whisper model

Everything takes a :class:`Context` so tests can redirect home, platform, env,
filesystem root and subprocess calls.
"""

from __future__ import annotations

import contextlib
import os
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from . import daemon as D
from . import protocol as P

SERVICE_NAME = "vlc-subsync.service"
LAUNCHD_LABEL = "io.github.sergimn.vlc-subsync"
DISPLAY_NAME = "SubSync"
WINDOWS_SHORTCUT = "SubSync.lnk"
WINDOWS_RUN_VALUE = "SubSync"
BACKUP_SUFFIX = ".subsync-backup"
STATE_SUFFIX = ".subsync-state"
LUA_INTF_NAME = "subsync"
EXTRAINTF_MODULE = "luaintf"
KNOWN_SCRIPTS = {"intf": ("subsync.lua",), "extensions": ("subsync_ext.lua",)}

MINIMAL_VLCRC = (
    "﻿###\n"
    "###  vlc 3.0 (created by vlc-subsync; VLC rewrites this file when you save preferences)\n"
    "###\n"
    "\n"
    "###\n"
    "### lines beginning with a '#' character are comments\n"
    "###\n"
    "\n"
    "[core] # core program\n"
    "\n"
    "[lua] # Lua interpreter\n"
)


# --------------------------------------------------------------------------- context


@dataclass
class Context:
    platform: str = field(default_factory=lambda: D.platform_key())
    home: Path = field(default_factory=Path.home)
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    root: Path = Path("/")
    dry_run: bool = False
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None
    which: Callable[[str], str | None] = shutil.which
    out: Callable[[str], None] = print
    popen: Callable[..., Any] | None = None
    vlc_running: Callable[[], bool] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.platform = D.platform_key(self.platform)
        self.home = Path(self.home)
        self.root = Path(self.root)

    # -- reporting
    def ok(self, msg: str) -> None:
        self.out(f"  [ok]   {msg}")

    def info(self, msg: str) -> None:
        self.out(f"         {msg}")

    def plan(self, msg: str) -> None:
        self.out(f"  [plan] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.out(f"  [warn] {msg}")

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.out(f"  [fail] {msg}")

    def step(self, title: str) -> None:
        self.out(f"\n{title}")

    def do(self, description: str) -> bool:
        """Report an action; returns False in dry-run mode (caller skips it)."""
        if self.dry_run:
            self.plan(description)
            return False
        return True

    def sh(self, args: Sequence[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
        runner = self.run or subprocess.run
        try:
            return runner(
                list(args),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(list(args), 127, "", str(exc))

    # -- standard dirs
    @property
    def config_home(self) -> Path:
        if self.platform == "linux":
            return Path(self.env.get("XDG_CONFIG_HOME") or self.home / ".config")
        return self.home / ".config"

    @property
    def appdata(self) -> Path:
        return Path(self.env.get("APPDATA") or self.home / "AppData" / "Roaming")


# --------------------------------------------------------------------------- VLC installs


@dataclass
class VlcInstall:
    kind: str  # native | snap | flatpak | macos | windows | custom
    data_dir: Path  # vlc.config.userdatadir()
    vlcrc: Path
    reason: str = ""
    detected: bool = True
    usable: bool = True  # False e.g. for a snap that has never been started

    @property
    def lua_dir(self) -> Path:
        return self.data_dir / "lua"

    @property
    def queue_dir(self) -> Path:
        return self.data_dir / "subsync"

    @property
    def state_file(self) -> Path:
        return self.vlcrc.with_name(self.vlcrc.name + STATE_SUFFIX)

    @property
    def backup_file(self) -> Path:
        return self.vlcrc.with_name(self.vlcrc.name + BACKUP_SUFFIX)

    def label(self) -> str:
        return {
            "native": "VLC (system package)",
            "snap": "VLC (snap)",
            "flatpak": "VLC (flatpak)",
            "macos": "VLC (macOS)",
            "windows": "VLC (Windows)",
        }.get(self.kind, f"VLC ({self.kind})")


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def _under_root(ctx: Context, abs_path: str) -> Path:
    return ctx.root / abs_path.lstrip("/\\")


def detect_vlc_installs(ctx: Context, *, include_default: bool = True) -> list[VlcInstall]:
    """Return every VLC user profile we can configure on this machine."""
    installs: list[VlcInstall] = []
    dirs = dict(D.vlc_data_dirs(ctx.platform, ctx.home, ctx.env))
    home = ctx.home

    if ctx.platform == "linux":
        # native (distro package, PPA, self-built)
        data = dirs["native"]
        cfg = ctx.config_home / "vlc"
        reasons = []
        which_vlc = ctx.which("vlc")
        if which_vlc and "/snap/" not in which_vlc and "flatpak" not in which_vlc:
            reasons.append(f"binary {which_vlc}")
        for b in ("usr/bin/vlc", "usr/local/bin/vlc"):
            if _exists(_under_root(ctx, b)) and f"binary /{b}" not in reasons:
                if not (which_vlc and Path(which_vlc) == Path("/" + b)):
                    reasons.append(f"binary /{b}")
        if _exists(data):
            reasons.append(f"data dir {data}")
        if _exists(cfg):
            reasons.append(f"config dir {cfg}")
        if reasons:
            installs.append(VlcInstall("native", data, cfg / "vlcrc", ", ".join(reasons)))

        # snap
        snap_user = home / "snap" / "vlc"
        if _exists(_under_root(ctx, "snap/vlc")) or _exists(snap_user):
            current = snap_user / "current"
            inst = VlcInstall(
                "snap",
                dirs["snap"],
                current / ".config" / "vlc" / "vlcrc",
                f"snap ({current})",
            )
            if not _exists(current):
                inst.usable = False
                inst.reason = "snap installed but never started (no ~/snap/vlc/current)"
            installs.append(inst)

        # flatpak
        fp_user = home / ".var" / "app" / "org.videolan.VLC"
        fp_markers = [
            _under_root(ctx, "var/lib/flatpak/app/org.videolan.VLC"),
            home / ".local" / "share" / "flatpak" / "app" / "org.videolan.VLC",
            fp_user,
        ]
        found = [m for m in fp_markers if _exists(m)]
        if found:
            installs.append(
                VlcInstall(
                    "flatpak",
                    dirs["flatpak"],
                    fp_user / "config" / "vlc" / "vlcrc",
                    f"flatpak ({found[0]})",
                )
            )
        if not installs and include_default:
            installs.append(
                VlcInstall(
                    "native",
                    data,
                    cfg / "vlcrc",
                    "VLC not found; configuring the default location",
                    detected=False,
                )
            )
    elif ctx.platform == "macos":
        data = dirs["macos"]
        prefs = home / "Library" / "Preferences" / "org.videolan.vlc"
        markers = [
            _under_root(ctx, "Applications/VLC.app"),
            home / "Applications" / "VLC.app",
            data,
            prefs,
        ]
        found = [m for m in markers if _exists(m)]
        if found or include_default:
            installs.append(
                VlcInstall(
                    "macos",
                    data,
                    prefs / "vlcrc",
                    str(found[0]) if found else "VLC not found; configuring the default location",
                    detected=bool(found),
                )
            )
    else:  # windows
        data = dirs["windows"]
        markers = [data]
        for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            base = ctx.env.get(var)
            if base:
                markers.append(Path(base) / "VideoLAN" / "VLC" / "vlc.exe")
        found = [m for m in markers if _exists(m)]
        if found or include_default:
            installs.append(
                VlcInstall(
                    "windows",
                    data,
                    data / "vlcrc",
                    str(found[0]) if found else "VLC not found; configuring the default location",
                    detected=bool(found),
                )
            )
    return installs


def custom_install(data_dir: str | Path, vlcrc: str | Path | None = None) -> VlcInstall:
    data = Path(data_dir).expanduser()
    rc = Path(vlcrc).expanduser() if vlcrc else data / "vlcrc"
    return VlcInstall("custom", data, rc, "given with --vlc-dir")


# --------------------------------------------------------------------------- vlcrc editing


@dataclass
class VlcrcEdit:
    text: str
    changed: bool
    added_luaintf: bool
    previous_lua_intf: str | None  # active value before our edit (None = not set)


def _split_lines(text: str) -> tuple[str, list[str], str]:
    bom = ""
    if text.startswith("﻿"):
        bom, text = "﻿", text[1:]
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    return bom, lines, newline


def _join_lines(bom: str, lines: list[str], newline: str) -> str:
    return bom + newline.join(lines) + newline


def _key_re(key: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    k = re.escape(key)
    return (
        re.compile(rf"^\s*{k}\s*=(.*)$"),
        re.compile(rf"^\s*#\s*{k}\s*=(.*)$"),
    )


def _insert_in_section(lines: list[str], section: str, line: str) -> None:
    sec_re = re.compile(rf"^\s*\[{re.escape(section)}\]")
    for i, ln in enumerate(lines):
        if sec_re.match(ln):
            j = i + 1
            if j < len(lines) and not lines[j].strip():
                lines.insert(j + 1, line)
            else:
                lines.insert(j, line)
            return
    while lines and not lines[-1].strip():
        lines.pop()
    lines += ["", f"[{section}]", "", line]


def _modules(value: str) -> list[str]:
    return [m.strip() for m in value.split(":") if m.strip()]


def apply_vlcrc(text: str) -> VlcrcEdit:
    """Enable our Lua interface in vlcrc text (idempotent)."""
    bom, lines, nl = _split_lines(text)
    original = list(lines)
    added = False

    active_re, comment_re = _key_re("extraintf")
    actives = [i for i, ln in enumerate(lines) if active_re.match(ln)]
    if actives:
        for i in actives:
            mods = _modules(active_re.match(lines[i]).group(1))  # type: ignore[union-attr]
            if EXTRAINTF_MODULE not in mods:
                mods.append(EXTRAINTF_MODULE)
                added = True
            lines[i] = "extraintf=" + ":".join(mods)
    else:
        added = True
        commented = [i for i, ln in enumerate(lines) if comment_re.match(ln)]
        if commented:
            lines[commented[0]] = f"extraintf={EXTRAINTF_MODULE}"
        else:
            _insert_in_section(lines, "core", f"extraintf={EXTRAINTF_MODULE}")

    active_re, comment_re = _key_re("lua-intf")
    previous: str | None = None
    actives = [i for i, ln in enumerate(lines) if active_re.match(ln)]
    if actives:
        previous = active_re.match(lines[actives[-1]]).group(1).strip()  # type: ignore[union-attr]
        for i in actives:
            lines[i] = f"lua-intf={LUA_INTF_NAME}"
    else:
        commented = [i for i, ln in enumerate(lines) if comment_re.match(ln)]
        if commented:
            lines[commented[0]] = f"lua-intf={LUA_INTF_NAME}"
        else:
            _insert_in_section(lines, "lua", f"lua-intf={LUA_INTF_NAME}")

    new_text = _join_lines(bom, lines, nl)
    return VlcrcEdit(
        text=new_text,
        changed=lines != original or not text.endswith(("\n", "\r\n")),
        added_luaintf=added,
        previous_lua_intf=previous,
    )


def revert_vlcrc(
    text: str, added_luaintf: bool | None, previous_lua_intf: str | None
) -> tuple[str, bool]:
    """Undo :func:`apply_vlcrc`.

    ``added_luaintf=None`` means "unknown" (state file lost): then ``luaintf`` is removed
    only if ``lua-intf`` still points at our script.
    """
    bom, lines, nl = _split_lines(text)
    original = list(lines)

    lua_active, _ = _key_re("lua-intf")
    current_lua = None
    for ln in lines:
        m = lua_active.match(ln)
        if m:
            current_lua = m.group(1).strip()
    ours = current_lua == LUA_INTF_NAME
    remove_luaintf = added_luaintf if added_luaintf is not None else ours

    if remove_luaintf:
        ext_active, _ = _key_re("extraintf")
        for i, ln in enumerate(lines):
            m = ext_active.match(ln)
            if m:
                mods = [x for x in _modules(m.group(1)) if x != EXTRAINTF_MODULE]
                lines[i] = "extraintf=" + ":".join(mods) if mods else "#extraintf="

    if ours:
        restore = previous_lua_intf if previous_lua_intf not in (None, "", LUA_INTF_NAME) else None
        for i, ln in enumerate(lines):
            if lua_active.match(ln):
                lines[i] = f"lua-intf={restore}" if restore else "#lua-intf=dummy"

    return _join_lines(bom, lines, nl), lines != original


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def configure_vlcrc(ctx: Context, inst: VlcInstall) -> None:
    rc = inst.vlcrc
    existed = rc.exists()
    state = P.read_kv(inst.state_file) or {}
    text = _read_text(rc) if existed else MINIMAL_VLCRC
    edit = apply_vlcrc(text)

    if not existed:
        if ctx.do(f"create {rc} with extraintf=luaintf, lua-intf=subsync"):
            P.write_text_atomic(rc, edit.text)
            ctx.ok(f"created {rc}")
    elif not edit.changed:
        ctx.ok(f"{rc} already configured")
    else:
        if not inst.backup_file.exists() and ctx.do(f"back up {rc} -> {inst.backup_file.name}"):
            shutil.copy2(rc, inst.backup_file)
        if ctx.do(f"edit {rc}: extraintf+=luaintf, lua-intf=subsync"):
            P.write_text_atomic(rc, edit.text)
            ctx.ok(f"updated {rc} (backup: {inst.backup_file.name})")

    if state:
        added = state.get("added_luaintf") == "1" or edit.added_luaintf
        previous = state.get("previous_lua_intf", "") if "previous_lua_intf" in state else None
        created = state.get("created_vlcrc") == "1"
    else:
        added = edit.added_luaintf
        previous = edit.previous_lua_intf
        created = not existed
    new_state = {
        "version": __version__,
        "added_luaintf": added,
        "created_vlcrc": created,
    }
    if previous is not None:
        new_state["previous_lua_intf"] = previous
    if not ctx.dry_run:
        P.write_kv(inst.state_file, new_state)


def unconfigure_vlcrc(ctx: Context, inst: VlcInstall) -> None:
    rc = inst.vlcrc
    state = P.read_kv(inst.state_file)
    if rc.exists():
        added: bool | None = None
        previous: str | None = None
        if state is not None:
            added = state.get("added_luaintf") == "1"
            previous = state.get("previous_lua_intf")
        new_text, changed = revert_vlcrc(_read_text(rc), added, previous)
        if changed and ctx.do(f"revert subsync settings in {rc}"):
            P.write_text_atomic(rc, new_text)
            ctx.ok(f"restored {rc}")
        elif not changed:
            ctx.ok(f"{rc} has no subsync settings")
    for extra in (inst.state_file, inst.backup_file):
        if extra.exists() and ctx.do(f"remove {extra}"):
            with contextlib.suppress(OSError):
                extra.unlink()


# --------------------------------------------------------------------------- Lua scripts


def packaged_scripts() -> dict[str, list[tuple[str, bytes]]]:
    """{"intf": [(name, bytes)], "extensions": [...]} from package data."""
    out: dict[str, list[tuple[str, bytes]]] = {"intf": [], "extensions": []}
    base = resources.files("vlcsubsync").joinpath("lua")
    for sub in out:
        d = base.joinpath(sub)
        try:
            entries = list(d.iterdir())
        except (FileNotFoundError, NotADirectoryError, OSError):
            continue
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.name.endswith(".lua") and entry.is_file():
                out[sub].append((entry.name, entry.read_bytes()))
    return out


def install_scripts(ctx: Context, inst: VlcInstall, scripts: dict[str, list[tuple[str, bytes]]]):
    if not any(scripts.values()):
        ctx.error("no Lua scripts found in the package (broken installation?)")
        return
    for sub, files in scripts.items():
        dest_dir = inst.lua_dir / sub
        for name, data in files:
            dest = dest_dir / name
            if dest.exists() and dest.read_bytes() == data:
                ctx.ok(f"{dest} up to date")
                continue
            if ctx.do(f"copy {name} -> {dest}"):
                dest_dir.mkdir(parents=True, exist_ok=True)
                P.write_text_atomic(dest, data.decode("utf-8"))
                ctx.ok(f"installed {dest}")


def remove_scripts(ctx: Context, inst: VlcInstall) -> None:
    names = {k: set(v) for k, v in KNOWN_SCRIPTS.items()}
    for sub, files in packaged_scripts().items():
        names.setdefault(sub, set()).update(n for n, _ in files)
    for sub, files in names.items():
        for name in sorted(files):
            p = inst.lua_dir / sub / name
            if p.exists() and ctx.do(f"remove {p}"):
                p.unlink()
                ctx.ok(f"removed {p}")


def create_queue_dir(ctx: Context, inst: VlcInstall) -> None:
    q = inst.queue_dir
    if all((q / s).is_dir() for s in (P.REQUESTS_DIR, P.JOBS_DIR, P.OUT_DIR)):
        ctx.ok(f"queue dir {q} exists")
        return
    if ctx.do(f"create queue dir {q}"):
        D.ensure_queue_layout(q)
        ctx.ok(f"created queue dir {q}")


def remove_queue_dir(ctx: Context, inst: VlcInstall) -> None:
    q = inst.queue_dir
    if q.exists() and ctx.do(f"remove queue dir {q}"):
        shutil.rmtree(q, ignore_errors=True)
        ctx.ok(f"removed {q}")


# --------------------------------------------------------------------------- daemon command


def asset_path(name: str) -> Path | None:
    """Filesystem path of a packaged asset (icon), or None if unavailable."""
    try:
        res = resources.files("vlcsubsync").joinpath("assets").joinpath(name)
        if res.is_file():
            p = Path(str(res))
            return p if p.is_file() else None
    except (OSError, TypeError, ValueError):
        pass
    return None


def _bin_dir() -> Path:
    return Path(sys.executable).parent


def daemon_command(ctx: Context, *, gui: bool = False) -> list[str]:
    """Command line that starts the daemon (absolute paths)."""
    bindir = _bin_dir()
    if ctx.platform == "windows":
        if gui:
            for cand in (
                bindir / "vlc-subsync-daemon.exe",
                bindir / "Scripts" / "vlc-subsync-daemon.exe",
            ):
                if cand.exists():
                    return [str(cand)]
            found = ctx.which("vlc-subsync-daemon")
            if found:
                return [found]
            pythonw = bindir / "pythonw.exe"
            exe = str(pythonw) if pythonw.exists() else sys.executable
            return [exe, "-m", "vlcsubsync.cli", "serve", "--no-console"]
        cand = bindir / "vlc-subsync.exe"
        if cand.exists():
            return [str(cand), "serve"]
    else:
        cand = bindir / "vlc-subsync"
        if cand.exists():
            return [str(cand), "serve"]
        found = ctx.which("vlc-subsync")
        if found:
            return [found, "serve"]
    return [sys.executable, "-m", "vlcsubsync.cli", "serve"]


def _systemd_quote(arg: str) -> str:
    if re.match(r"^[A-Za-z0-9_@%+=:,./-]+$", arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _desktop_quote(arg: str) -> str:
    if re.match(r"^[A-Za-z0-9_@%+=:,./-]+$", arg):
        return arg
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped}"'


def systemd_unit_text(cmd: Sequence[str]) -> str:
    return (
        "[Unit]\n"
        f"Description={DISPLAY_NAME} daemon (automatic subtitle synchronisation for VLC)\n"
        "Documentation=https://github.com/sergimn/VLCSubtitleSync\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={' '.join(_systemd_quote(a) for a in cmd)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "Nice=10\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def desktop_entry_text(cmd: Sequence[str], icon: Path | None = None) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={DISPLAY_NAME}\n"
        + (f"Icon={icon}\n" if icon else "")
        + "Comment=Automatic subtitle synchronisation for VLC\n"
        f"Exec={' '.join(_desktop_quote(a) for a in cmd)}\n"
        "Terminal=false\n"
        "NoDisplay=true\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def launchd_plist(cmd: Sequence[str], log_dir: Path) -> bytes:
    return plistlib.dumps(
        {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": list(cmd),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ProcessType": "Background",
            "StandardOutPath": str(log_dir / "launchd.out.log"),
            "StandardErrorPath": str(log_dir / "launchd.err.log"),
        }
    )


# --------------------------------------------------------------------------- autostart


def _systemd_user_available(ctx: Context) -> bool:
    if not ctx.which("systemctl"):
        return False
    res = ctx.sh(["systemctl", "--user", "show-environment"], timeout=10)
    return res.returncode == 0


def _uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else 0


def systemd_unit_path(ctx: Context) -> Path:
    return ctx.config_home / "systemd" / "user" / SERVICE_NAME


def desktop_autostart_path(ctx: Context) -> Path:
    return ctx.config_home / "autostart" / "vlc-subsync.desktop"


def launch_agent_path(ctx: Context) -> Path:
    return ctx.home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def windows_startup_dir(ctx: Context) -> Path:
    return ctx.appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def install_autostart(ctx: Context) -> str:
    """Register autostart and (re)start the daemon.  Returns the mechanism used."""
    if ctx.platform == "linux":
        cmd = daemon_command(ctx)
        if _systemd_user_available(ctx):
            unit = systemd_unit_path(ctx)
            if ctx.do(f"write systemd user unit {unit} (ExecStart={' '.join(cmd)})"):
                P.write_text_atomic(unit, systemd_unit_text(cmd))
                ctx.ok(f"wrote {unit}")
            if ctx.do(f"systemctl --user enable + restart {SERVICE_NAME}"):
                for args in (
                    ["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", SERVICE_NAME],
                    ["systemctl", "--user", "restart", SERVICE_NAME],
                ):
                    res = ctx.sh(args)
                    if res.returncode != 0:
                        ctx.warn(f"{' '.join(args)} failed: {(res.stderr or '').strip()}")
                        break
                else:
                    ctx.ok(f"daemon enabled and started ({SERVICE_NAME})")
                    return "systemd"
                start_daemon_detached(ctx)
            return "systemd"
        entry = desktop_autostart_path(ctx)
        if ctx.do(f"write XDG autostart entry {entry}"):
            P.write_text_atomic(entry, desktop_entry_text(cmd, asset_path("icon-256.png")))
            ctx.ok(f"wrote {entry}")
        start_daemon_detached(ctx)
        return "xdg-autostart"

    if ctx.platform == "macos":
        cmd = daemon_command(ctx)
        plist = launch_agent_path(ctx)
        log_dir = D.user_log_dir()
        if ctx.do(f"write LaunchAgent {plist}"):
            log_dir.mkdir(parents=True, exist_ok=True)
            plist.parent.mkdir(parents=True, exist_ok=True)
            plist.write_bytes(launchd_plist(cmd, log_dir))
            ctx.ok(f"wrote {plist}")
        if ctx.do(f"launchctl bootstrap gui/{_uid()} {plist}"):
            ctx.sh(["launchctl", "bootout", f"gui/{_uid()}/{LAUNCHD_LABEL}"])
            res = ctx.sh(["launchctl", "bootstrap", f"gui/{_uid()}", str(plist)])
            if res.returncode != 0:
                ctx.warn(f"launchctl bootstrap failed: {(res.stderr or '').strip()}")
                start_daemon_detached(ctx)
            else:
                ctx.ok("daemon loaded with launchd")
        return "launchd"

    # windows
    cmd = daemon_command(ctx, gui=True)
    lnk = windows_startup_dir(ctx) / WINDOWS_SHORTCUT
    if ctx.do(f"create Startup shortcut {lnk} -> {cmd[0]}"):
        lnk.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
            + _ps_quote(str(lnk))
            + ");"
            + f"$s.TargetPath={_ps_quote(cmd[0])};"
            + f"$s.Arguments={_ps_quote(subprocess.list2cmdline(cmd[1:]))};"
            + f"$s.WorkingDirectory={_ps_quote(str(Path(cmd[0]).parent))};"
            + f"$s.Description='{DISPLAY_NAME} daemon';$s.WindowStyle=7;"
        )
        icon = asset_path("icon.ico")
        if icon is not None:
            script += f"$s.IconLocation={_ps_quote(str(icon) + ',0')};"
        script += "$s.Save()"
        res = ctx.sh(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ]
        )
        if res.returncode == 0 and lnk.exists():
            ctx.ok(f"created {lnk}")
            mechanism = "startup-shortcut"
        else:
            ctx.warn(f"shortcut creation failed ({(res.stderr or '').strip()}); using HKCU Run key")
            mechanism = "run-key"
            _set_run_key(ctx, subprocess.list2cmdline(cmd))
    else:
        mechanism = "startup-shortcut"
    start_daemon_detached(ctx)
    return mechanism


def _set_run_key(ctx: Context, command: str | None) -> bool:
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return False
    path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_ALL_ACCESS) as key:
            if command is None:
                with contextlib.suppress(FileNotFoundError):
                    winreg.DeleteValue(key, WINDOWS_RUN_VALUE)
            else:
                winreg.SetValueEx(key, WINDOWS_RUN_VALUE, 0, winreg.REG_SZ, command)
        return True
    except OSError as exc:
        ctx.warn(f"registry update failed: {exc}")
        return False


def remove_autostart(ctx: Context) -> None:
    if ctx.platform == "linux":
        unit = systemd_unit_path(ctx)
        if unit.exists():
            if ctx.do(f"systemctl --user disable --now {SERVICE_NAME}"):
                ctx.sh(["systemctl", "--user", "disable", "--now", SERVICE_NAME])
            if ctx.do(f"remove {unit}"):
                unit.unlink()
                ctx.sh(["systemctl", "--user", "daemon-reload"])
                ctx.ok(f"removed {unit}")
        entry = desktop_autostart_path(ctx)
        if entry.exists() and ctx.do(f"remove {entry}"):
            entry.unlink()
            ctx.ok(f"removed {entry}")
    elif ctx.platform == "macos":
        plist = launch_agent_path(ctx)
        if ctx.do(f"launchctl bootout gui/{_uid()}/{LAUNCHD_LABEL}"):
            ctx.sh(["launchctl", "bootout", f"gui/{_uid()}/{LAUNCHD_LABEL}"])
        if plist.exists() and ctx.do(f"remove {plist}"):
            plist.unlink()
            ctx.ok(f"removed {plist}")
    else:
        lnk = windows_startup_dir(ctx) / WINDOWS_SHORTCUT
        if lnk.exists() and ctx.do(f"remove {lnk}"):
            lnk.unlink()
            ctx.ok(f"removed {lnk}")
        if ctx.do("remove HKCU Run entry (if any)"):
            _set_run_key(ctx, None)


def autostart_status(ctx: Context) -> str:
    if ctx.platform == "linux":
        if systemd_unit_path(ctx).exists():
            return f"systemd user unit {systemd_unit_path(ctx)}"
        if desktop_autostart_path(ctx).exists():
            return f"XDG autostart {desktop_autostart_path(ctx)}"
    elif ctx.platform == "macos":
        if launch_agent_path(ctx).exists():
            return f"LaunchAgent {launch_agent_path(ctx)}"
    else:
        lnk = windows_startup_dir(ctx) / WINDOWS_SHORTCUT
        if lnk.exists():
            return f"Startup shortcut {lnk}"
    return "not configured"


# --------------------------------------------------------------------------- daemon control


def stop_daemon(ctx: Context, timeout: float = 10.0) -> bool:
    """Stop a running daemon (from the lock file pid).  True if one was stopped."""
    pid = D.daemon_running_pid()
    if not pid or pid == os.getpid():
        return False
    if not ctx.do(f"stop running daemon (pid {pid})"):
        return False
    try:
        if ctx.platform == "windows":
            ctx.sh(["taskkill", "/PID", str(pid), "/T", "/F"])
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        ctx.warn(f"could not stop daemon pid {pid}: {exc}")
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not D.pid_alive(pid):
            ctx.ok(f"stopped daemon (pid {pid})")
            return True
        time.sleep(0.2)
    ctx.warn(f"daemon pid {pid} did not exit in {timeout:.0f}s")
    return False


def start_daemon_detached(ctx: Context) -> None:
    cmd = daemon_command(ctx, gui=ctx.platform == "windows")
    if not ctx.do(f"start daemon: {' '.join(cmd)}"):
        return
    popen = ctx.popen or subprocess.Popen
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if ctx.platform == "windows":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED|NEWGROUP|NOWIN
    else:
        kwargs["start_new_session"] = True
    try:
        popen(cmd, **kwargs)
        ctx.ok("daemon started")
    except OSError as exc:
        ctx.error(f"could not start the daemon: {exc}")


def is_vlc_running(ctx: Context) -> bool:
    if ctx.vlc_running is not None:
        return ctx.vlc_running()
    try:
        if ctx.platform == "linux":
            proc = Path("/proc")
            for p in proc.iterdir():
                if p.name.isdigit():
                    with contextlib.suppress(OSError):
                        if (p / "comm").read_text().strip() == "vlc":
                            return True
            return False
        if ctx.platform == "macos":
            return ctx.sh(["pgrep", "-x", "VLC"], timeout=5).returncode == 0
        res = ctx.sh(["tasklist", "/FI", "IMAGENAME eq vlc.exe", "/NH"], timeout=10)
        return "vlc.exe" in (res.stdout or "").lower()
    except OSError:
        return False


# --------------------------------------------------------------------------- models


def download_models(names: Sequence[str] | None = None, *, out: Callable[[str], None] = print):
    """Download Whisper models (default: the configured English model)."""
    from .config import Config
    from .transcribe import download_model

    cfg = Config.load()
    if not names:
        names = [cfg.model_en]
    root = cfg.extra.get("model_dir") or None
    for name in names:
        out(f"  downloading Whisper model '{name}' (first time only, may take a minute)...")
        path = download_model(name, download_root=root)
        out(f"  [ok]   model '{name}' ready ({path})")


# --------------------------------------------------------------------------- top level


def _target_installs(
    ctx: Context, vlc_dirs: Sequence[str] = (), vlcrc: str | None = None
) -> list[VlcInstall]:
    if vlc_dirs:
        return [custom_install(d, vlcrc if len(vlc_dirs) == 1 else None) for d in vlc_dirs]
    return detect_vlc_installs(ctx)


def run_setup(
    ctx: Context,
    *,
    autostart: bool = True,
    model: bool = True,
    vlc_dirs: Sequence[str] = (),
    vlcrc: str | None = None,
) -> int:
    ctx.out(
        f"{DISPLAY_NAME} {__version__} setup"
        + (" (dry run, nothing is changed)" if ctx.dry_run else "")
    )
    installs = _target_installs(ctx, vlc_dirs, vlcrc)
    scripts = packaged_scripts()

    for inst in installs:
        ctx.step(f"{inst.label()}: {inst.data_dir}")
        ctx.info(f"detected via: {inst.reason}")
        if not inst.usable:
            ctx.warn(f"skipping: {inst.reason}. Start VLC once, then re-run `vlc-subsync setup`.")
            continue
        try:
            install_scripts(ctx, inst, scripts)
            configure_vlcrc(ctx, inst)
            create_queue_dir(ctx, inst)
        except OSError as exc:
            ctx.error(f"could not configure {inst.label()}: {exc}")

    if autostart:
        ctx.step("Background service")
        try:
            stop_daemon(ctx)
            mech = install_autostart(ctx)
            ctx.info(f"autostart: {mech}")
        except OSError as exc:
            ctx.error(f"autostart setup failed: {exc}")
    else:
        ctx.step("Background service")
        ctx.info("autostart skipped (--no-autostart); run `vlc-subsync serve` manually")

    if model:
        ctx.step("Speech model")
        if ctx.do("download the default Whisper model"):
            try:
                download_models(out=ctx.out)
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"model download failed ({exc}); it will be downloaded on first use")

    if is_vlc_running(ctx):
        ctx.step("Note")
        ctx.warn("VLC is running: restart VLC so the new settings take effect")

    ctx.out("")
    if ctx.errors:
        ctx.out(f"Setup finished with {len(ctx.errors)} error(s).")
        return 1
    if ctx.dry_run:
        ctx.out("Dry run complete.")
    else:
        ctx.out(
            "Done! Open a video in VLC and pick a subtitle track: it will be synced "
            "automatically.\nUse View > 'SubSync' in VLC for manual control."
        )
    return 0


def run_uninstall(
    ctx: Context, *, purge: bool = False, vlc_dirs: Sequence[str] = (), vlcrc: str | None = None
) -> int:
    ctx.out(f"{DISPLAY_NAME} {__version__} uninstall" + (" (dry run)" if ctx.dry_run else ""))
    ctx.step("Background service")
    remove_autostart(ctx)
    stop_daemon(ctx)
    for inst in _target_installs(ctx, vlc_dirs, vlcrc):
        if not inst.usable:
            continue
        ctx.step(f"{inst.label()}: {inst.data_dir}")
        try:
            remove_scripts(ctx, inst)
            unconfigure_vlcrc(ctx, inst)
            remove_queue_dir(ctx, inst)
        except OSError as exc:
            ctx.error(f"could not clean {inst.label()}: {exc}")
    if purge:
        ctx.step("Purge")
        for d in (D.user_cache_dir(), D.user_state_dir(), D.user_log_dir(), D.user_config_dir()):
            if d.exists() and ctx.do(f"remove {d}"):
                shutil.rmtree(d, ignore_errors=True)
                ctx.ok(f"removed {d}")
        ctx.info("Whisper models in the Hugging Face cache (~/.cache/huggingface) are kept")
    ctx.out("")
    if ctx.errors:
        ctx.out(f"Uninstall finished with {len(ctx.errors)} error(s).")
        return 1
    ctx.out(f"{DISPLAY_NAME} removed. To remove the program itself: uv tool uninstall vlc-subsync")
    return 0
