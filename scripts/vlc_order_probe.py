"""Check SubSync's copy of VLC's subtitle autodetection against a real VLC.

Makes a few videos with subtitle files next to them, plays each in VLC (no window,
no sound) and reads the order VLC loads the subtitle files in from its debug log,
then compares it with what :mod:`vlcsubsync.sidecars` works out on this machine.

    python scripts/vlc_order_probe.py [--vlc PATH] [--locale NAME ...]

Exits 1 if any order differs. VLC for Windows never sets a locale, so --locale only
matters on Linux and macOS (VLC sets LC_ALL from the environment there).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlcsubsync import sidecars as S  # noqa: E402

SRT = "1\n00:00:01,000 --> 00:00:02,000\nhello\n"
IDX = "# VobSub index file, v7\nsize: 720x480\nid: en, index: 0\n"

# name -> (video name, subtitle files, extra startup file or None)
LAYOUTS: dict[str, tuple[str, list[str], str | None]] = {
    "matrix": (
        "The.Matrix.wav",
        [
            "The.Matrix.srt",
            "The.Matrix.smi",
            "The.Matrix.en.srt",
            "the_matrix_de.ass",
            "The.Matrix-ZZ.ssa",
            "The Matrix - Spanish.srt",
            "The.Matrix (2).srt",
            "Best of The Matrix.srt",
            "English.srt",
            ".hidden.srt",
            "Subs/The Matrix.srt",
            "Subs/The.Matrix.fr.srt",
            "Subs/3_English.srt",
            "subtitles/The.Matrix.it.srt",
            "Sub/The.Matrix.pt.srt",
        ],
        None,
    ),
    "two": ("Movie.wav", ["Movie.en.srt", "Movie.es.srt"], None),
    "three": ("Movie.wav", ["Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"], None),
    "mixed": (
        "Movie.wav",
        ["Movie.srt", "Movie.en.srt", "Movie.es.srt", "Subs/Movie.de.srt", "Movie.ass"],
        None,
    ),
    "nine": ("Movie.wav", [f"Movie.{c}.srt" for c in "qhbxmdkat"], None),
    "twelve": ("Movie.wav", [f"Movie.{c}{c}.srt" for c in "plmokijnuhby"], None),
    "vobsub": (
        "Movie.wav",
        ["Movie.idx", "Movie.sub", "Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"],
        None,
    ),
    "case": (
        "Movie.wav",
        ["movie.EN.srt", "Movie.de.srt", "Movie_fr.srt", "Movie-es.srt", "MOVIE.it.srt"],
        None,
    ),
    "subfile": ("Movie.wav", ["Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"], "Extra.srt"),
    "subfile9": ("Movie.wav", [f"Movie.{c}.srt" for c in "qhbxmdkat"], "Extra.srt"),
}

LOADED = re.compile(r"loading spu-es slave: (\S+) \(forced")


def default_vlc() -> str:
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")):
            exe = Path(base, "VideoLAN", "VLC", "vlc.exe")
            if exe.is_file():
                return str(exe)
    if sys.platform == "darwin":
        return "/Applications/VLC.app/Contents/MacOS/VLC"
    return shutil.which("vlc") or "vlc"


def make_layout(root: Path, video: str, files: list[str], extra: str | None) -> Path:
    root.mkdir(parents=True)
    media = root / video
    with wave.open(str(media), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 4000)
    for name in files + ([extra] if extra else []):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(IDX if name.endswith(".idx") else SRT, encoding="utf-8")
    return media


def run_vlc(vlc: str, media: Path, log: Path, locale: str | None, sub_file: Path | None) -> str:
    env = dict(os.environ)
    if locale:
        env["LC_ALL"] = locale
    cmd = [
        vlc,
        "-I",
        "dummy",
        *(["--dummy-quiet"] if sys.platform == "win32" else []),  # no console window
        "--no-video",
        "--aout=dummy",
        "--vout=dummy",
        "--no-media-library",
        "--play-and-exit",
        "--run-time=5",
        "-vv",
        "--file-logging",
        f"--logfile={log}",
        "--log-verbose=2",
    ]
    if sub_file:
        cmd.append(f"--sub-file={sub_file}")
    cmd.append(str(media))
    subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    return log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlc", default=default_vlc())
    ap.add_argument("--locale", action="append", default=[])
    ap.add_argument("--work", default="vlc-order-probe")
    args = ap.parse_args()
    locales = args.locale or [None]
    work = Path(args.work).resolve()
    shutil.rmtree(work, ignore_errors=True)
    print(f"python {sys.version.split()[0]} on {sys.platform}; VLC: {args.vlc}")
    print(f"C library sort here is VLC's: {S._sorter()[1]}")
    bad = 0
    for locale in locales:
        for name, (video, files, extra) in LAYOUTS.items():
            root = work / f"{(locale or 'env').replace('.', '_')}-{name}"
            media = make_layout(root, video, files, extra)
            sub_file = root / extra if extra else None
            text = run_vlc(args.vlc, media, root / "vlc.log", locale, sub_file)
            version = re.search(r"VLC media player - (\S+)", text)
            loaded = LOADED.findall(text)
            collate = locale or ("C" if sys.platform == "win32" else S._env_collate())
            if sys.platform == "win32":
                collate = "C"
            det = S.VlcSidecars(
                str(media), S.Autodetect(True, 3, S.default_paths(sys.platform == "win32"), collate)
            )
            uri_of = {os.path.normpath(e.path): e.uri for e in det.entries}
            extra_uri = S.path_to_uri(str(sub_file), sys.platform == "win32") if sub_file else None
            seen = [u for u in loaded if u != extra_uri]
            got = det.orders(1 if extra else 0)
            ours = [[uri_of[p] for p in o] for o in got.orders]
            ok = seen in ours and (extra is None or loaded[:1] == [extra_uri])
            likeliest = seen == ours[0]
            bad += not ok
            print(
                f"\n== {name} (locale {locale or 'from env'}, VLC {version and version.group(1)}):"
                f" {'MATCH' if ok else 'DIFFERENT'}"
                f"{'' if likeliest or not ok else ' (not the likeliest order)'};"
                f" {len(ours)} possible order(s), exact={got.exact} {got.why}"
            )
            print("  VLC:  " + "\n        ".join(loaded or ["(nothing loaded)"]))
            if not ok:
                for i, o in enumerate(ours):
                    print(f"  ours{i}: " + "\n         ".join(o or ["(none)"]))
                if not loaded:
                    print("  log tail:\n" + "\n".join(text.splitlines()[-40:]))
    print(f"\n{bad} layout(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
