"""Install / uninstall the VLC integration.

* detect VLC installs (Linux native / snap / flatpak, macOS, Windows)
* copy the Lua scripts (package data) into ``<userdatadir>/lua/{intf,extensions}``
* edit ``vlcrc`` (``extraintf`` += ``luaintf``, ``lua-intf=subsync``) with a backup and a
  small state file so ``uninstall`` can revert exactly what we changed
* create the queue dirs
* make the helper start *with VLC*, and only then (systemd user path unit / launchd
  WatchPaths agent / a ``launcher`` file the Lua interface uses to spawn it), removing
  the login autostart of earlier versions; the helper exits by itself after VLC
* pre-download the default Whisper model

Everything takes a :class:`Context` so tests can redirect home, platform, env,
filesystem root and subprocess calls.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import plistlib
import re
import shlex
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
PATH_UNIT_NAME = "vlc-subsync.path"
LAUNCHER_FILE = "launcher"
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
            # The snap's launcher (vlc-snap-wrapper.sh) runs
            # `vlc --config=$SNAP_USER_COMMON/vlcrc`, so VLC reads ~/snap/vlc/common/vlcrc,
            # not the XDG path inside the revision directory.
            inst = VlcInstall(
                "snap",
                dirs["snap"],
                snap_user / "common" / "vlcrc",
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


def cleanup_legacy_snap_vlcrc(ctx: Context, inst: VlcInstall) -> None:
    """Undo what versions <= 0.1.0 wrote to the snap vlcrc that VLC never reads.

    Those versions edited ~/snap/vlc/current/.config/vlc/vlcrc, but the snap launcher
    passes --config=~/snap/vlc/common/vlcrc. Only files carrying our state marker are
    touched. A vlcrc we created ourselves is removed only if, after reverting our keys,
    no active settings remain (the user or VLC may have added some since).
    """
    if inst.kind != "snap":
        return
    legacy = dataclasses.replace(
        inst, vlcrc=ctx.home / "snap" / "vlc" / "current" / ".config" / "vlc" / "vlcrc"
    )
    if legacy.vlcrc == inst.vlcrc or not legacy.state_file.exists():
        return
    state = P.read_kv(legacy.state_file) or {}
    ctx.info(f"cleaning up settings from an earlier version in {legacy.vlcrc}")
    unconfigure_vlcrc(ctx, legacy)
    if state.get("created_vlcrc") != "1" or not legacy.vlcrc.exists():
        return
    if ctx.dry_run:
        ctx.do(f"remove {legacy.vlcrc} if no other settings remain")
        return
    if _has_active_settings(_read_text(legacy.vlcrc)):
        ctx.info(f"keeping {legacy.vlcrc}: it now contains other settings")
        return
    if ctx.do(f"remove {legacy.vlcrc}"):
        with contextlib.suppress(OSError):
            legacy.vlcrc.unlink()


def _has_active_settings(text: str) -> bool:
    """True if a vlcrc has any `key=value` line that isn't commented out."""
    _bom, lines, _nl = _split_lines(text)
    return any("=" in ln and not ln.lstrip().startswith(("#", ";", "[")) for ln in lines)


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


def daemon_command(
    ctx: Context, *, gui: bool = False, queue_dirs: Sequence[Path | str] = ()
) -> list[str]:
    """Command line that starts the daemon (absolute paths).

    ``queue_dirs`` are passed as ``--queue-dir`` (the daemon watches them as well as
    the default VLC dirs): needed for ``setup --vlc-dir``, whose queue dir the
    daemon would not find by itself.
    """
    extra = [a for q in queue_dirs for a in ("--queue-dir", str(q))]
    return _daemon_base_command(ctx, gui=gui) + extra


def _daemon_base_command(ctx: Context, *, gui: bool = False) -> list[str]:
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
    arg = arg.replace("%", "%%")  # systemd specifier escaping
    if re.match(r"^[A-Za-z0-9_@%+=:,./-]+$", arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _systemd_path_value(path: Path | str) -> str:
    return str(path).replace("%", "%%")


def systemd_service_text(
    cmd: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    description: str | None = None,
    path_unit: str = PATH_UNIT_NAME,
) -> str:
    """The on-demand service: started by the .path unit, exits by itself after VLC.

    No ``[Install]`` section (it is never enabled on its own, so nothing starts it at
    login) and no ``Restart=``: the path unit starts it again on VLC's next write.
    """
    env_lines = "".join(
        f"Environment={_systemd_quote(f'{k}={v}')}\n" for k, v in (environment or {}).items()
    )
    return (
        "[Unit]\n"
        f"Description={description or DISPLAY_NAME + ' helper (subtitle sync for VLC)'}\n"
        "Documentation=https://github.com/sergimn/VLCSubtitleSync\n"
        f"# Started on demand by {path_unit} when VLC starts or queues a job;\n"
        "# exits by itself ~15 s after VLC closes. Nothing runs while VLC is closed.\n"
        "# A crash loop (VLC rewrites intf_state every 5 s) stops after 20 starts in\n"
        "# 10 min; `vlc-subsync setup` or `systemctl --user reset-failed` clears it.\n"
        "StartLimitIntervalSec=600\n"
        "StartLimitBurst=20\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={' '.join(_systemd_quote(a) for a in cmd)}\n" + env_lines + "Nice=10\n"
        "IOSchedulingClass=idle\n"
        "CPUSchedulingPolicy=batch\n"
    )


def systemd_path_text(
    queue_dirs: Sequence[Path],
    *,
    service: str = SERVICE_NAME,
    description: str | None = None,
) -> str:
    """Path unit watching every queue dir: VLC's ``intf_state`` and ``requests/``.

    systemd resolves symlinks in the watched paths (inotify follows them) and also
    watches every parent directory, so when the snap's ``~/snap/vlc/current`` symlink
    is switched to a new revision the event on ``~/snap/vlc`` makes it re-resolve
    the path (see DESIGN.md "Lifecycle").
    """
    watches = []
    for q in queue_dirs:
        watches.append(f"PathModified={_systemd_path_value(Path(q) / P.INTF_STATE_FILE)}\n")
        watches.append(f"DirectoryNotEmpty={_systemd_path_value(Path(q) / P.REQUESTS_DIR)}\n")
    return (
        "[Unit]\n"
        f"Description={description or 'Start the ' + DISPLAY_NAME + ' helper when VLC starts'}\n"
        "Documentation=https://github.com/sergimn/VLCSubtitleSync\n"
        "\n"
        "[Path]\n" + "".join(watches) + f"Unit={service}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def launchd_plist(cmd: Sequence[str], log_dir: Path, queue_dirs: Sequence[Path]) -> bytes:
    """LaunchAgent started on demand by launchd: no RunAtLoad, no KeepAlive."""
    return plistlib.dumps(
        {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": list(cmd),
            # VLC rewrites intf_state when it starts (and every few seconds) ...
            "WatchPaths": [str(Path(q) / P.INTF_STATE_FILE) for q in queue_dirs],
            # ... and launchd also starts us while a request is waiting
            "QueueDirectories": [str(Path(q) / P.REQUESTS_DIR) for q in queue_dirs],
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "StandardOutPath": str(log_dir / "launchd.out.log"),
            "StandardErrorPath": str(log_dir / "launchd.err.log"),
        }
    )


# --------------------------------------------------------------------------- launcher file


def _windows_ascii_path(path: str, *, spaces: bool = False) -> str:
    """8.3 short form of a non-ASCII path (VLC's Lua ``os.execute`` uses the ANSI code
    page, so a non-ASCII path would be mangled). ``spaces``: also shorten a path with
    whitespace (for launcher ``args``, which are whitespace-separated)."""
    needed = not path.isascii() or (spaces and any(c.isspace() for c in path))
    if not needed or os.name != "nt":
        return path
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))  # type: ignore[attr-defined]
        if 0 < n < len(buf) and buf.value.isascii():
            if not (spaces and any(c.isspace() for c in buf.value)):
                return buf.value
    except (OSError, AttributeError):
        pass
    return path


def launcher_data(mode: str, cmd: Sequence[str], platform: str) -> dict[str, object]:
    """Contents of ``<q>/launcher``, read by the Lua interface (see DESIGN.md).

    ``mode=service``: a service manager starts the helper (systemd path unit /
    launchd), the Lua side must not. ``mode=spawn``: the Lua side starts ``exe`` with
    ``args`` (whitespace-separated) itself when it sees no fresh heartbeat.
    """
    exe = _windows_ascii_path(cmd[0]) if platform == "windows" else cmd[0]
    return {"version": 1, "mode": mode, "exe": exe, "args": " ".join(cmd[1:])}


def _spawn_queue_dir_args(ctx: Context, queue_dirs: Sequence[Path]) -> dict[Path, Path | None]:
    """``{queue dir: argument}`` for a launcher's ``args`` (``None``: cannot be passed).

    The Lua side splits ``args`` on whitespace (see :func:`launcher_data`), so a path
    with whitespace cannot be passed: it is left out with a warning (the systemd unit
    and the LaunchAgent pass arguments one by one and have no such limit). On Windows
    a non-ASCII or spaced path is first replaced by its 8.3 short form when there is
    one.
    """
    usable: dict[Path, Path | None] = {}
    for q in queue_dirs:
        arg = str(q)
        if ctx.platform == "windows":
            arg = _windows_ascii_path(arg, spaces=True)
            if "%" in arg:
                ctx.warn(
                    f"the queue dir {q} contains '%'; cmd.exe may expand it when VLC "
                    "starts the helper. Use a VLC dir without '%'"
                )
        if any(c.isspace() for c in arg):
            ctx.warn(
                f"VLC cannot pass a path with spaces to the helper it starts, so {q} "
                "is not watched by it: use a --vlc-dir without spaces, or run "
                f'`vlc-subsync serve --persistent --queue-dir "{q}"` yourself'
            )
            usable[q] = None
            continue
        usable[q] = Path(arg)
    return usable


def write_launchers(
    ctx: Context, queue_dirs: Sequence[Path], mode: str, cmd: Sequence[str]
) -> None:
    data = launcher_data(mode, cmd, ctx.platform)
    for q in queue_dirs:
        path = Path(q) / LAUNCHER_FILE
        if P.read_kv(path) == {k: P.sanitize_value(v) for k, v in data.items()}:
            ctx.ok(f"{path} up to date (mode={mode})")
            continue
        if ctx.do(f"write {path} (mode={mode}, exe={data['exe']})"):
            path.parent.mkdir(parents=True, exist_ok=True)
            P.write_kv(path, data)
            ctx.ok(f"wrote {path} (mode={mode})")


# --------------------------------------------------------------------------- start with VLC


def _systemd_user_available(ctx: Context) -> bool:
    if not ctx.which("systemctl"):
        return False
    res = ctx.sh(["systemctl", "--user", "show-environment"], timeout=10)
    return res.returncode == 0


def _uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else 0


def systemd_user_dir(ctx: Context) -> Path:
    return ctx.config_home / "systemd" / "user"


def systemd_unit_path(ctx: Context) -> Path:
    return systemd_user_dir(ctx) / SERVICE_NAME


def systemd_path_unit_path(ctx: Context) -> Path:
    return systemd_user_dir(ctx) / PATH_UNIT_NAME


def legacy_service_wants_link(ctx: Context) -> Path:
    """``default.target.wants`` link of the always-on service of earlier versions."""
    return systemd_user_dir(ctx) / "default.target.wants" / SERVICE_NAME


def desktop_autostart_path(ctx: Context) -> Path:
    return ctx.config_home / "autostart" / "vlc-subsync.desktop"


def launch_agent_path(ctx: Context) -> Path:
    return ctx.home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def windows_startup_dir(ctx: Context) -> Path:
    return ctx.appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _systemctl(ctx: Context, *args: str, warn: bool = True) -> bool:
    cmd = ["systemctl", "--user", *args]
    res = ctx.sh(cmd)
    if res.returncode != 0 and warn:
        ctx.warn(f"{' '.join(cmd)} failed: {(res.stderr or '').strip()}")
    return res.returncode == 0


def _is_link_or_exists(p: Path) -> bool:
    return p.is_symlink() or p.exists()


def _legacy_service_installed(ctx: Context) -> bool:
    """The always-on (login) service of earlier versions is installed or enabled."""
    if _is_link_or_exists(legacy_service_wants_link(ctx)):
        return True
    unit = systemd_unit_path(ctx)
    try:
        return unit.exists() and "[Install]" in unit.read_text(encoding="utf-8")
    except OSError:
        return False


def _legacy_launch_agent(ctx: Context) -> bool:
    plist = launch_agent_path(ctx)
    try:
        data = plistlib.loads(plist.read_bytes()) if plist.exists() else {}
    except (OSError, plistlib.InvalidFileException, ValueError):
        return False
    return bool(data.get("RunAtLoad") or data.get("KeepAlive"))


def legacy_autostart_artifacts(ctx: Context) -> list[Path]:
    """Login autostart entries written by earlier versions that are still present."""
    found: list[Path] = []
    if ctx.platform == "linux":
        if _legacy_service_installed(ctx):
            link = legacy_service_wants_link(ctx)
            found.append(link if _is_link_or_exists(link) else systemd_unit_path(ctx))
        if desktop_autostart_path(ctx).exists():
            found.append(desktop_autostart_path(ctx))
    elif ctx.platform == "macos":
        if _legacy_launch_agent(ctx):
            found.append(launch_agent_path(ctx))
    else:
        lnk = windows_startup_dir(ctx) / WINDOWS_SHORTCUT
        if lnk.exists():
            found.append(lnk)
    return found


def _delete_run_key(ctx: Context) -> bool:
    """Delete the HKCU ``Run`` value of earlier versions. True if one was removed."""
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return False
    path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_ALL_ACCESS) as key:
            winreg.DeleteValue(key, WINDOWS_RUN_VALUE)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        ctx.warn(f"registry update failed: {exc}")
        return False


def remove_legacy_autostart(ctx: Context) -> None:
    """Remove the login autostart of earlier versions (always-on service, shortcut)."""
    if ctx.platform == "linux":
        if _legacy_service_installed(ctx) and ctx.do(
            f"stop and disable the always-on {SERVICE_NAME} of an earlier version"
        ):
            if ctx.which("systemctl"):
                # before the unit file is rewritten: disable needs its [Install] section
                _systemctl(ctx, "disable", "--now", SERVICE_NAME, warn=False)
            link = legacy_service_wants_link(ctx)
            if _is_link_or_exists(link):
                with contextlib.suppress(OSError):
                    link.unlink()
            with contextlib.suppress(OSError):
                systemd_unit_path(ctx).unlink()
            if ctx.which("systemctl"):
                _systemctl(ctx, "daemon-reload", warn=False)
            ctx.ok(f"removed the always-on {SERVICE_NAME} (started at login)")
        entry = desktop_autostart_path(ctx)
        if entry.exists() and ctx.do(f"remove login autostart entry {entry}"):
            entry.unlink()
            ctx.ok(f"removed {entry}")
    elif ctx.platform == "macos":
        if _legacy_launch_agent(ctx) and ctx.do(
            f"unload and remove the always-on LaunchAgent {launch_agent_path(ctx)}"
        ):
            ctx.sh(["launchctl", "bootout", f"gui/{_uid()}/{LAUNCHD_LABEL}"])
            with contextlib.suppress(OSError):
                launch_agent_path(ctx).unlink()
            ctx.ok("removed the always-on LaunchAgent (KeepAlive/RunAtLoad)")
    else:
        lnk = windows_startup_dir(ctx) / WINDOWS_SHORTCUT
        if lnk.exists() and ctx.do(f"remove Startup shortcut {lnk}"):
            lnk.unlink()
            ctx.ok(f"removed {lnk}")
        if ctx.do(rf"remove HKCU\...\Run\{WINDOWS_RUN_VALUE} (if any)") and _delete_run_key(ctx):
            ctx.ok(rf"removed HKCU\...\Run\{WINDOWS_RUN_VALUE}")


def watched_installs(
    ctx: Context, installs: Sequence[VlcInstall], *, keep_registered: bool = True
) -> list[VlcInstall]:
    """The usable installs the helper must start for.

    The start-with-VLC units are global (one per user), so a ``setup --vlc-dir``
    adds to them instead of replacing them: with custom installs the detected
    (usable) VLC installs are kept too, and ``keep_registered`` keeps the
    ``--vlc-dir`` dirs of an earlier setup whose queue dir still exists (until
    ``uninstall --vlc-dir`` removes them).
    """
    out = [i for i in installs if i.usable]
    detected = [i for i in detect_vlc_installs(ctx, include_default=False) if i.usable]

    def add(inst: VlcInstall) -> None:
        if inst.queue_dir not in {i.queue_dir for i in out}:
            out.append(inst)

    if keep_registered:
        known = [i.queue_dir for i in out + detected]
        for q in installed_queue_dir_args(ctx, known):
            if q.is_dir():
                add(custom_install(q.parent))
    if any(i.kind == "custom" for i in out):
        for det in detected:
            add(det)
    return out


def install_autostart(
    ctx: Context, installs: Sequence[VlcInstall] = (), *, keep_registered: bool = True
) -> str:
    """Make the helper start with VLC, and only then. Returns the mechanism used.

    Nothing is started now: the helper comes up when VLC starts and exits after it.
    For ``--vlc-dir`` (``kind="custom"``) installs the detected VLC installs keep
    being watched too (the units and LaunchAgent are global, one per user), and the
    helper is started with ``--queue-dir`` for each custom queue dir.
    """
    watched = watched_installs(ctx, installs, keep_registered=keep_registered)
    queue_dirs = [i.queue_dir for i in watched]
    custom_dirs = [i.queue_dir for i in watched if i.kind == "custom"]
    remove_legacy_autostart(ctx)
    if not queue_dirs:
        ctx.warn("no usable VLC installation; nothing to start the helper for")
        return "none"
    given = {i.queue_dir for i in installs if i.usable}
    if any(i.queue_dir not in given for i in watched):
        ctx.info(
            "the helper also keeps starting for: "
            + ", ".join(str(i.data_dir) for i in watched if i.queue_dir not in given)
        )

    def spawn_setup(gui: bool = False) -> tuple[list[str], list[Path]]:
        """(command, launcher dirs) for mode=spawn. A dir that cannot be passed in
        ``args`` gets no launcher: its VLC would start a helper that ignores it."""
        args = _spawn_queue_dir_args(ctx, custom_dirs)
        cmd = daemon_command(ctx, gui=gui, queue_dirs=[a for a in args.values() if a])
        return cmd, [q for q in queue_dirs if args.get(q, q) is not None]

    if ctx.platform == "linux":
        cmd = daemon_command(ctx, queue_dirs=custom_dirs)
        if _systemd_user_available(ctx):
            service, path_unit = systemd_unit_path(ctx), systemd_path_unit_path(ctx)
            path_text = systemd_path_text(queue_dirs)
            try:
                old_path_text: str | None = path_unit.read_text(encoding="utf-8")
            except OSError:
                old_path_text = None
            # an already running path unit keeps its old watches until restarted
            watches_changed = old_path_text is not None and old_path_text != path_text
            if ctx.do(f"write systemd user units {path_unit.name} + {service.name}"):
                P.write_text_atomic(service, systemd_service_text(cmd))
                P.write_text_atomic(path_unit, path_text)
                ctx.ok(f"wrote {path_unit} (watches {len(queue_dirs)} VLC queue dir(s))")
                ctx.ok(f"wrote {service} (ExecStart={' '.join(cmd)})")
            if ctx.do(f"systemctl --user enable --now {PATH_UNIT_NAME}"):
                _systemctl(ctx, "daemon-reload")
                _systemctl(ctx, "reset-failed", PATH_UNIT_NAME, SERVICE_NAME, warn=False)
                if _systemctl(ctx, "enable", "--now", PATH_UNIT_NAME):
                    if watches_changed and _systemctl(ctx, "restart", PATH_UNIT_NAME):
                        ctx.ok(f"restarted {PATH_UNIT_NAME}: it watches the new paths")
                    ctx.ok(f"{PATH_UNIT_NAME} active: the helper starts when VLC starts")
            write_launchers(ctx, queue_dirs, "service", cmd)
            return "systemd-path"
        cmd, spawn_dirs = spawn_setup()
        write_launchers(ctx, spawn_dirs, "spawn", cmd)
        for inst in installs:
            if inst.usable and inst.kind in ("snap", "flatpak"):
                ctx.warn(
                    f"{inst.label()} is sandboxed and cannot start the helper itself, and "
                    "there is no systemd user session: run `vlc-subsync serve --persistent` "
                    "yourself to use it with this VLC"
                )
        ctx.ok("no systemd user session: VLC starts the helper itself when it opens")
        return "vlc-spawn"

    if ctx.platform == "macos":
        cmd = daemon_command(ctx, queue_dirs=custom_dirs)
        plist = launch_agent_path(ctx)
        log_dir = D.user_log_dir()
        if ctx.do(f"write LaunchAgent {plist} (WatchPaths + QueueDirectories, on demand)"):
            log_dir.mkdir(parents=True, exist_ok=True)
            plist.parent.mkdir(parents=True, exist_ok=True)
            plist.write_bytes(launchd_plist(cmd, log_dir, queue_dirs))
            ctx.ok(f"wrote {plist}")
        if ctx.do(f"launchctl bootstrap gui/{_uid()} {plist}"):
            ctx.sh(["launchctl", "bootout", f"gui/{_uid()}/{LAUNCHD_LABEL}"])
            res = ctx.sh(["launchctl", "bootstrap", f"gui/{_uid()}", str(plist)])
            if res.returncode != 0:
                ctx.warn(f"launchctl bootstrap failed: {(res.stderr or '').strip()}")
            else:
                ctx.ok("LaunchAgent loaded: the helper starts when VLC starts")
        write_launchers(ctx, queue_dirs, "service", cmd)
        return "launchd"

    # windows: VLC's Lua interface launches the GUI exe itself (see DESIGN.md)
    cmd, spawn_dirs = spawn_setup(gui=True)
    if "%" in cmd[0]:
        ctx.warn(
            f"the helper's path contains '%' ({cmd[0]}); cmd.exe may expand it and VLC "
            "then fails to start the helper. Reinstall SubSync to a path without '%'"
        )
    write_launchers(ctx, spawn_dirs, "spawn", cmd)
    ctx.ok(f"VLC starts the helper itself when it opens ({Path(cmd[0]).name})")
    return "vlc-spawn"


def remove_autostart(ctx: Context) -> None:
    remove_legacy_autostart(ctx)
    if ctx.platform == "linux":
        path_unit, service = systemd_path_unit_path(ctx), systemd_unit_path(ctx)
        if path_unit.exists() or service.exists():
            if ctx.do(f"systemctl --user disable --now {PATH_UNIT_NAME}; stop {SERVICE_NAME}"):
                _systemctl(ctx, "disable", "--now", PATH_UNIT_NAME, warn=False)
                _systemctl(ctx, "stop", SERVICE_NAME, warn=False)
            for unit in (path_unit, service):
                if unit.exists() and ctx.do(f"remove {unit}"):
                    unit.unlink()
                    ctx.ok(f"removed {unit}")
            if ctx.do("systemctl --user daemon-reload"):
                _systemctl(ctx, "daemon-reload", warn=False)
    elif ctx.platform == "macos":
        plist = launch_agent_path(ctx)
        if ctx.do(f"launchctl bootout gui/{_uid()}/{LAUNCHD_LABEL}"):
            ctx.sh(["launchctl", "bootout", f"gui/{_uid()}/{LAUNCHD_LABEL}"])
        if plist.exists() and ctx.do(f"remove {plist}"):
            plist.unlink()
            ctx.ok(f"removed {plist}")
    # windows: only the legacy entries; the launcher files go with the queue dirs


def _queue_dir_args(args: Sequence[str]) -> list[Path]:
    return [Path(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "--queue-dir"]


def installed_queue_dir_args(ctx: Context, queue_dirs: Sequence[Path] = ()) -> list[Path]:
    """The ``--queue-dir`` dirs (``setup --vlc-dir``) the installed helper command
    passes: from the systemd unit, the LaunchAgent, or the launchers in
    ``queue_dirs``."""
    found: list[Path] = []
    if ctx.platform == "linux" and systemd_unit_path(ctx).exists():
        with contextlib.suppress(OSError, ValueError):
            for line in systemd_unit_path(ctx).read_text(encoding="utf-8").splitlines():
                if line.startswith("ExecStart="):
                    found += _queue_dir_args(
                        shlex.split(line[len("ExecStart=") :].replace("%%", "%"))
                    )
    elif ctx.platform == "macos" and launch_agent_path(ctx).exists():
        with contextlib.suppress(OSError, plistlib.InvalidFileException, ValueError):
            data = plistlib.loads(launch_agent_path(ctx).read_bytes())
            found += _queue_dir_args([str(a) for a in data.get("ProgramArguments", [])])
    for q in queue_dirs:
        data = P.read_kv(Path(q) / LAUNCHER_FILE) or {}
        found += _queue_dir_args(data.get("args", "").split())
    return list(dict.fromkeys(found))


def lifecycle_status(
    ctx: Context, queue_dirs: Sequence[Path] = ()
) -> list[tuple[str, str, bool | None]]:
    """``(label, value, ok)`` rows describing how the helper starts (for ``doctor``)."""
    rows: list[tuple[str, str, bool | None]] = []
    for p in legacy_autostart_artifacts(ctx):
        rows.append(
            ("login autostart (earlier version)", f"{p}: re-run `vlc-subsync setup`", False)
        )
    for q in installed_queue_dir_args(ctx, queue_dirs):
        rows.append(
            (
                "--vlc-dir queue dir",
                f"{q}" + ("" if q.is_dir() else " (missing: re-run setup --vlc-dir)"),
                q.is_dir(),
            )
        )
    if ctx.platform == "windows":
        launchers = [(q, P.read_kv(Path(q) / LAUNCHER_FILE)) for q in queue_dirs]
        for q, data in launchers:
            if not data:
                rows.append(
                    (f"launcher {Path(q) / LAUNCHER_FILE}", "missing (re-run setup)", False)
                )
                continue
            exe = data.get("exe", "")
            rows.append(
                (
                    f"launcher {Path(q) / LAUNCHER_FILE}",
                    f"VLC starts {exe}" + ("" if Path(exe).exists() else " (not found!)"),
                    Path(exe).exists(),
                )
            )
        if not launchers:
            rows.append(("start with VLC", "no queue dir", False))
        return rows
    if ctx.platform == "macos":
        plist = launch_agent_path(ctx)
        if not plist.exists():
            rows.append(("LaunchAgent", "missing (re-run setup)", False))
            return rows
        res = ctx.sh(["launchctl", "print", f"gui/{_uid()}/{LAUNCHD_LABEL}"], timeout=10)
        loaded = res.returncode == 0
        running = loaded and "state = running" in (res.stdout or "")
        rows.append(
            (
                "LaunchAgent",
                f"{plist}: "
                + ("not loaded (re-run setup)" if not loaded else "loaded, starts with VLC")
                + ("; helper running" if running else ""),
                loaded,
            )
        )
        return rows
    path_unit = systemd_path_unit_path(ctx)
    if path_unit.exists():

        def state(*args: str) -> str:
            res = ctx.sh(["systemctl", "--user", *args], timeout=10)
            return (res.stdout or "").strip() or (res.stderr or "").strip() or "unknown"

        enabled = state("is-enabled", PATH_UNIT_NAME)
        active = state("is-active", PATH_UNIT_NAME)
        rows.append(
            (
                PATH_UNIT_NAME,
                f"{enabled}, {active} (starts the helper when VLC starts)",
                enabled == "enabled" and active == "active",
            )
        )
        service = state("is-active", SERVICE_NAME)
        rows.append(
            (
                SERVICE_NAME,
                service
                + (" (VLC is open)" if service == "active" else " (normal while VLC is closed)"),
                None,
            )
        )
        return rows
    spawn = [
        q for q in queue_dirs if (P.read_kv(Path(q) / LAUNCHER_FILE) or {}).get("mode") == "spawn"
    ]
    if spawn:
        exe = (P.read_kv(Path(spawn[0]) / LAUNCHER_FILE) or {}).get("exe", "?")
        rows.append(("start with VLC", f"VLC starts {exe} (no systemd user session)", True))
    else:
        rows.append(("start with VLC", "not configured (run `vlc-subsync setup`)", False))
    return rows


def autostart_status(ctx: Context, queue_dirs: Sequence[Path] = ()) -> str:
    """One-line summary of :func:`lifecycle_status`."""
    rows = lifecycle_status(ctx, queue_dirs)
    return "; ".join(f"{label}: {value}" for label, value, _ in rows) or "not configured"


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
            cleanup_legacy_snap_vlcrc(ctx, inst)
            configure_vlcrc(ctx, inst)
            create_queue_dir(ctx, inst)
        except OSError as exc:
            ctx.error(f"could not configure {inst.label()}: {exc}")

    ctx.step("Start with VLC")
    if autostart:
        try:
            stop_daemon(ctx)  # e.g. the always-on daemon of an earlier version
            mech = install_autostart(ctx, installs)
            ctx.info(f"helper lifecycle: {mech} (starts with VLC, exits after it)")
        except OSError as exc:
            ctx.error(f"start-with-VLC setup failed: {exc}")
    else:
        ctx.info("skipped (--no-autostart); run `vlc-subsync serve --persistent` yourself")

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


def _same_dir(q: Path, others: Sequence[Path] | set[Path]) -> bool:
    for o in others:
        if q == o:
            return True
        with contextlib.suppress(OSError):
            if os.path.samefile(q, o):
                return True
    return False


def _installs_kept(ctx: Context, targets: Sequence[VlcInstall]) -> list[VlcInstall]:
    """Configured installs not being removed by ``uninstall --vlc-dir``, if the
    start-with-VLC mechanism is installed for them (else nothing to keep)."""
    removed = {i.queue_dir for i in targets}
    detected = [i for i in detect_vlc_installs(ctx, include_default=False) if i.usable]
    custom = [
        custom_install(q.parent)
        for q in installed_queue_dir_args(ctx, [i.queue_dir for i in [*detected, *targets]])
    ]
    keep: list[VlcInstall] = []
    for inst in custom + detected:
        q = inst.queue_dir
        if not q.is_dir() or q in {i.queue_dir for i in keep} or _same_dir(q, removed):
            continue
        keep.append(inst)
    installed = (
        systemd_path_unit_path(ctx).exists()
        or launch_agent_path(ctx).exists()
        or any((i.queue_dir / LAUNCHER_FILE).exists() for i in keep)
    )
    return keep if installed else []


def run_uninstall(
    ctx: Context, *, purge: bool = False, vlc_dirs: Sequence[str] = (), vlcrc: str | None = None
) -> int:
    ctx.out(f"{DISPLAY_NAME} {__version__} uninstall" + (" (dry run)" if ctx.dry_run else ""))
    targets = _target_installs(ctx, vlc_dirs, vlcrc)
    ctx.step("Start with VLC")
    keep = _installs_kept(ctx, targets) if vlc_dirs else []
    if keep:
        # only some dirs are removed: the helper keeps starting for the others
        ctx.info(
            "other VLC installs keep using the helper: " + ", ".join(str(i.data_dir) for i in keep)
        )
        install_autostart(ctx, keep, keep_registered=False)
    else:
        remove_autostart(ctx)
    stop_daemon(ctx)
    for inst in targets:
        if not inst.usable:
            continue
        ctx.step(f"{inst.label()}: {inst.data_dir}")
        try:
            remove_scripts(ctx, inst)
            cleanup_legacy_snap_vlcrc(ctx, inst)
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
