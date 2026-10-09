"""Command line interface: ``vlc-subsync sync|serve|setup|uninstall|doctor|download-models|
clear-cache``."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import MODES

DISPLAY_NAME = "SubSync"


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlc-subsync",
        description=f"{DISPLAY_NAME}: automatically sync VLC subtitles to the audio (Whisper).",
    )
    parser.add_argument("--version", action="version", version=f"vlc-subsync {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("sync", help="sync one subtitle track/file to a media file")
    p.add_argument("media", help="video/audio file")
    p.add_argument("--audio", type=int, default=0, metavar="N", help="audio track (0-based)")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--sub",
        type=int,
        default=None,
        metavar="N",
        help="subtitle track (0-based, VLC order: embedded tracks first, then "
        "external files next to the media); default 0",
    )
    g.add_argument("--sub-file", metavar="PATH", help="external subtitle file to sync")
    p.add_argument(
        "-o", "--output", metavar="OUT", help="output file (default: <media stem>.synced.<ext>)"
    )
    p.add_argument("--model", help="Whisper model name (overrides config for all languages)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="inference device")
    p.add_argument(
        "--mode",
        choices=MODES,
        help="fast (default): sampled windows; thorough: ~2.5x more windows; "
        "exhaustive: transcribe every 30 s with speech (slow on CPU). Overrides config",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    p = sub.add_parser("serve", help="run the background daemon used by VLC")
    p.add_argument(
        "--queue-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="extra queue dir to watch (repeatable)",
    )
    p.add_argument(
        "--no-default-queues",
        action="store_true",
        help="only watch --queue-dir dirs, not the VLC user-data dirs",
    )
    p.add_argument("--no-console", action="store_true", help="log to file only")
    p.add_argument(
        "--persistent",
        action="store_true",
        help="keep running while VLC is closed (default: exit ~15 s after VLC closes)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    def add_vlc_dir_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--vlc-dir",
            action="append",
            default=[],
            metavar="DIR",
            help="VLC user data dir to use instead of auto-detection (repeatable)",
        )
        sp.add_argument("--vlcrc", metavar="PATH", help="vlcrc file for --vlc-dir")
        sp.add_argument("--dry-run", action="store_true", help="show what would be done")

    p = sub.add_parser(
        "setup", help="install the VLC integration (scripts, settings, start-with-VLC)"
    )
    add_vlc_dir_opts(p)
    p.add_argument(
        "--no-autostart",
        action="store_true",
        help="don't register the start of the helper with VLC",
    )
    p.add_argument("--no-model", action="store_true", help="don't pre-download the Whisper model")

    p = sub.add_parser("uninstall", help="remove the VLC integration")
    add_vlc_dir_opts(p)
    p.add_argument("--purge", action="store_true", help="also delete cache, logs, config and state")

    sub.add_parser("doctor", help="diagnose the installation")

    sub.add_parser(
        "clear-cache",
        help="delete the stored sync results (to disable the cache, set cache=off in config.ini)",
    )

    p = sub.add_parser("download-models", help="download Whisper models")
    p.add_argument("models", nargs="*", help="model names (default: configured English model)")
    p.add_argument(
        "--all", action="store_true", help="download both the English and multilingual models"
    )
    return parser


# --------------------------------------------------------------------------- commands


def default_output_path(media: str, source_path: str | None) -> str:
    ext = ".srt"
    if source_path:
        e = Path(source_path).suffix.lower()
        if e in (".srt", ".ass", ".ssa", ".vtt"):
            ext = e
    m = Path(media)
    return str(m.with_name(f"{m.stem}.synced{ext}"))


def _progress_printer():
    tty = sys.stderr is not None and sys.stderr.isatty()
    last = [0.0, ""]

    def progress(frac: float, message: str = "") -> None:
        if sys.stderr is None:
            return
        line = f"[{int(max(0.0, min(1.0, frac)) * 100):3d}%] {message}"
        if tty:
            sys.stderr.write("\r\x1b[2K" + line[:150])
            sys.stderr.flush()
        elif message != last[1] or time.monotonic() - last[0] > 5:
            sys.stderr.write(line + "\n")
            last[0], last[1] = time.monotonic(), message

    def done() -> None:
        if tty and sys.stderr is not None:
            sys.stderr.write("\r\x1b[2K")
            sys.stderr.flush()

    return progress, done


def cmd_sync(args: argparse.Namespace) -> int:
    from .config import Config
    from .daemon import JobError, resolve_source

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    media = os.path.abspath(args.media)
    if not os.path.isfile(media):
        print(f"error: media file not found: {args.media}", file=sys.stderr)
        return 2
    try:
        if args.sub_file:
            source = resolve_source(media, None, os.path.abspath(args.sub_file))
        else:
            source = resolve_source(media, args.sub if args.sub is not None else 0)
    except JobError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = Config.load()
    if args.model:
        config.model_en = config.model_multi = args.model
    if args.device:
        config.device = args.device
    if args.mode:
        config = config.with_mode(args.mode)
    output = (
        os.path.abspath(args.output) if args.output else default_output_path(media, source.path)
    )

    from .sync import SubtitleSource, sync_subtitles

    src = SubtitleSource(kind=source.kind, index=source.index, path=source.path)
    what = source.path if source.kind == "external" else f"embedded subtitle track {source.index}"
    print(f"Syncing {what}\n     to audio track {args.audio} of {media}", flush=True)
    progress, done = _progress_printer()
    try:
        result = sync_subtitles(media, args.audio, src, output, config, progress=progress)
    except KeyboardInterrupt:
        done()
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        done()
        if args.verbose:
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    done()
    print(format_result(result))
    return 0


def format_result(result) -> str:
    drift = (float(result.scale) - 1.0) * 100.0
    lines = [
        f"Result:     {result.message}",
        f"Output:     {result.output_path}",
        f"Applied:    {'yes' if result.applied else 'no (confidence too low; timing unchanged)'}",
        f"Method:     {result.method}",
        f"Offset:     {float(result.offset):+.3f} s",
        f"Drift:      {drift:+.3f} % (scale {float(result.scale):.6f})",
        f"Segments:   {getattr(result, 'segments', 1)}",
        f"Anchors:    {getattr(result, 'anchors', 0)}",
        f"Confidence: {float(result.confidence):.2f}",
    ]
    return "\n".join(lines)


def cmd_serve(args: argparse.Namespace) -> int:
    from .daemon import serve

    return serve(
        args.queue_dir,
        use_default_queues=not args.no_default_queues,
        log_to_stderr=not args.no_console,
        verbose=args.verbose,
        persistent=args.persistent,
    )


def cmd_setup(args: argparse.Namespace) -> int:
    from .setup_vlc import Context, run_setup

    ctx = Context(dry_run=args.dry_run)
    return run_setup(
        ctx,
        autostart=not args.no_autostart,
        model=not args.no_model,
        vlc_dirs=args.vlc_dir,
        vlcrc=args.vlcrc,
    )


def cmd_uninstall(args: argparse.Namespace) -> int:
    from .setup_vlc import Context, run_uninstall

    ctx = Context(dry_run=args.dry_run)
    return run_uninstall(ctx, purge=args.purge, vlc_dirs=args.vlc_dir, vlcrc=args.vlcrc)


def cmd_download_models(args: argparse.Namespace) -> int:
    from .config import Config
    from .setup_vlc import download_models

    names = list(args.models)
    if args.all:
        cfg = Config.load()
        names += [cfg.model_en, cfg.model_multi]
    try:
        download_models(list(dict.fromkeys(names)) or None)
    except Exception as exc:  # noqa: BLE001
        print(f"error: model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    from . import daemon as D

    d = D.results_cache_dir()
    removed, left = D.clear_results_cache(d)
    print(f"Deleted {removed} cached result(s) from {d}")
    if left:
        print(f"error: could not delete {left} file(s) in {d}", file=sys.stderr)
        return 1
    return 0


def _vlcrc_value(text: str, key: str) -> str | None:
    import re

    value = None
    for ln in text.splitlines():
        m = re.match(rf"^\s*{re.escape(key)}\s*=(.*)$", ln)
        if m:
            value = m.group(1).strip()
    return value


def run_doctor(out=print, ctx=None) -> int:
    from . import daemon as D
    from . import lifecycle as L
    from . import protocol as P
    from . import setup_vlc as S

    ctx = ctx or S.Context()
    problems = 0

    def item(label: str, value: object, ok: bool | None = None) -> None:
        nonlocal problems
        tag = "      " if ok is None else ("[ok]  " if ok else "[!!]  ")
        if ok is False:
            problems += 1
        out(f"  {tag}{label}: {value}")

    out(f"{DISPLAY_NAME} (vlc-subsync) {__version__}")
    out(f"  Python {platform.python_version()} ({sys.executable}) on {platform.platform()}")

    packaged = S.packaged_scripts()
    out("\nVLC installations")
    installs = S.detect_vlc_installs(ctx, include_default=False)
    if not installs:
        item("VLC", "not detected", False)
    for inst in installs:
        out(f"  {inst.label()}  [{inst.reason}]")
        if not inst.usable:
            item("status", "not usable (start VLC once, then re-run setup)", False)
            continue
        for sub, files in packaged.items():
            for name, data in files:
                p = inst.lua_dir / sub / name
                if not p.exists():
                    item(f"lua/{sub}/{name}", "missing", False)
                else:
                    current = p.read_bytes() == data
                    item(
                        f"lua/{sub}/{name}",
                        "installed" if current else "installed (outdated, re-run setup)",
                        current,
                    )
        if inst.vlcrc.exists():
            text = S._read_text(inst.vlcrc)
            ext = _vlcrc_value(text, "extraintf")
            lua = _vlcrc_value(text, "lua-intf")
            item(f"vlcrc {inst.vlcrc}", "present", True)
            item("extraintf", ext, bool(ext and "luaintf" in ext.split(":")))
            item("lua-intf", lua, lua == "subsync")
        else:
            item(f"vlcrc {inst.vlcrc}", "missing", False)
        q = inst.queue_dir
        item("queue dir", q, q.is_dir())
        launcher = P.read_kv(q / S.LAUNCHER_FILE) if q.is_dir() else None
        if launcher:
            item("launcher", f"mode={launcher.get('mode', '?')} exe={launcher.get('exe', '?')}")
        vlc_open = L.read_intf_state(q, time.time()).alive if q.is_dir() else False
        hb = P.read_heartbeat(q) if q.is_dir() else None
        if hb is None:
            # the helper only runs while VLC does
            item(
                "daemon heartbeat",
                "none" + ("" if vlc_open else " (VLC is closed)"),
                False if vlc_open else None,
            )
        else:
            age = hb.age()
            item(
                "daemon heartbeat",
                f"{age:.0f}s ago (pid {hb.pid}, v{hb.version})",
                age <= 10 or (None if not vlc_open else False),
            )

    out("\nHelper lifecycle (starts with VLC, exits ~15 s after it)")
    queue_dirs = [i.queue_dir for i in installs if i.usable]
    for label, value, ok in S.lifecycle_status(ctx, queue_dirs):
        item(label, value, ok)
    pid = D.daemon_running_pid()
    item("process", f"running (pid {pid})" if pid else "not running (normal while VLC is closed)")
    item("lock file", D.lock_file_path())
    item("log file", D.user_log_dir() / "daemon.log")
    item("result cache", D.results_cache_dir())

    out("\nConfiguration & models")
    try:
        from .config import Config, default_config_path

        cfg = Config.load()
        item("config", f"{default_config_path()}")
        item(
            "settings",
            f"model_en={cfg.model_en} model_multi={cfg.model_multi} device={cfg.device} "
            f"compute_type={cfg.compute_type} cache={'on' if cfg.cache else 'off'}",
        )
        try:
            from faster_whisper import download_model

            for name in dict.fromkeys([cfg.model_en, cfg.model_multi]):
                try:
                    path = download_model(
                        name,
                        local_files_only=True,
                        output_dir=cfg.extra.get("model_dir") or None,
                    )
                    item(f"model {name}", f"cached ({path})", True)
                except Exception:  # noqa: BLE001
                    item(
                        f"model {name}",
                        "not downloaded (run `vlc-subsync download-models`)",
                        None,
                    )
        except ImportError as exc:
            item("faster-whisper", f"not importable: {exc}", False)
    except Exception as exc:  # noqa: BLE001
        item("config", f"error: {exc}", False)
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        item("CUDA", f"{n} device(s)" if n else "not available (CPU will be used)")
    except Exception as exc:  # noqa: BLE001
        item("CUDA", f"unknown ({exc})")

    out("")
    out("No problems found." if problems == 0 else f"{problems} problem(s) found.")
    return 0 if problems == 0 else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    return run_doctor()


COMMANDS = {
    "sync": cmd_sync,
    "serve": cmd_serve,
    "setup": cmd_setup,
    "uninstall": cmd_uninstall,
    "doctor": cmd_doctor,
    "download-models": cmd_download_models,
    "clear-cache": cmd_clear_cache,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130


def daemon_main() -> int:
    """GUI-script entry point (``vlc-subsync-daemon``): serve without a console."""
    from .daemon import serve

    argv = sys.argv[1:]
    queue_dirs = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--queue-dir"]
    return serve(
        queue_dirs,
        log_to_stderr=False,
        verbose="-v" in argv or "--verbose" in argv,
        persistent="--persistent" in argv,
    )


if __name__ == "__main__":
    sys.exit(main())
