#!/usr/bin/env python3
"""Measure sync accuracy on a real file against an independent full-file transcript.

Ground truth: a Whisper model (default ``small.en`` on CUDA, else ``base.en``) is run
over the *whole* audio track with word timestamps (cached in ``OUT/truth_*.json``).
The transcript is aligned to the subtitle text (token-level diff), and for every cue
whose first token is matched inside a run of >= ``--min-run`` consecutive tokens the
reference time is that word's start. The per-cue error is
``|retimed cue start - reference word start|``.

Each run syncs with whatever ``vlcsubsync`` is importable, so two code versions can be
compared on the same ground truth::

    # this checkout
    python scripts/measure_real.py movie.mkv --audio 0 --sub 0 --out /tmp/m --label new
    # another checkout (e.g. a worktree of main)
    PYTHONPATH=/path/to/main/src python scripts/measure_real.py movie.mkv \\
        --audio 0 --sub 0 --out /tmp/m --label main
    # table of every run stored in /tmp/m
    python scripts/measure_real.py movie.mkv --audio 0 --sub 0 --out /tmp/m --report

Nothing is written next to the media; keep ``--out`` outside the repository.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


def _truth_path(out: Path, media: str, audio: int, model: str) -> Path:
    st = os.stat(media)
    key = f"{Path(media).stem}.a{audio}.{model}.{st.st_size}"
    return out / f"truth_{key}.json"


def ground_truth(media: str, audio: int, out: Path, model: str, device: str) -> list[dict]:
    """Full-file word timestamps (cached)."""
    path = _truth_path(out, media, audio, model)
    if path.exists():
        return json.loads(path.read_text())["words"]
    from faster_whisper import WhisperModel

    from vlcsubsync.media import decode_audio

    t0 = time.monotonic()
    pcm = decode_audio(media, audio)
    candidates = [("cuda", "int8_float16"), ("cpu", "int8")] if device == "auto" else []
    if device == "cuda":
        candidates = [("cuda", "int8_float16")]
    elif device == "cpu":
        candidates = [("cpu", "int8")]
    last: Exception | None = None
    words: list[dict] = []
    for dev, ct in candidates:
        try:
            wm = WhisperModel(model, device=dev, compute_type=ct)
            segs, _info = wm.transcribe(
                pcm,
                language="en" if model.endswith(".en") else None,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            words = []
            for seg in segs:
                for w in seg.words or ():
                    if w.word.strip():
                        words.append({"s": float(w.start), "e": float(w.end), "w": w.word.strip()})
            break
        except Exception as e:  # CUDA libs missing → CPU
            print(f"ground truth on {dev} failed: {e}", file=sys.stderr)
            last = e
    else:
        raise RuntimeError(f"ground-truth transcription failed: {last}")
    path.write_text(json.dumps({"model": model, "words": words}))
    print(f"ground truth: {len(words)} words in {time.monotonic() - t0:.0f}s → {path}")
    return words


def load_original(media: str, sub: int, out: Path):
    from vlcsubsync.media import extract_subtitles
    from vlcsubsync.sync import resolve_subtitle_source

    src = resolve_subtitle_source(media, sub)
    if src.kind == "embedded":
        subs = extract_subtitles(media, src.index)
    else:
        import pysubs2

        subs = pysubs2.load(src.path)
    subs.save(str(out / "original.srt"), format_="srt")
    return src


def reference_times(original_srt: Path, words: list[dict], min_run: int):
    """Per original cue: reference audio time of its first spoken token, or NaN."""
    import pysubs2

    from vlcsubsync.subtitles import tokenize

    subs = pysubs2.load(str(original_srt))
    sub_tok: list[str] = []
    cue_first: list[int] = []  # index of each cue's first token (-1 = no tokens)
    for ev in subs:
        toks = tokenize(ev.plaintext)
        cue_first.append(len(sub_tok) if toks else -1)
        sub_tok.extend(toks)
    tr_tok: list[str] = []
    tr_time: list[float] = []
    for w in words:
        parts = tokenize(w["w"], drop_annotations=False)
        dur = max(w["e"] - w["s"], 0.0)
        for k, p in enumerate(parts):
            tr_tok.append(p)
            tr_time.append(w["s"] + dur * k / len(parts))
    sm = difflib.SequenceMatcher(None, sub_tok, tr_tok, autojunk=False)
    match = np.full(len(sub_tok), -1)
    for a, b, n in sm.get_matching_blocks():
        if n >= min_run:
            match[a : a + n] = np.arange(b, b + n)
    ref = np.full(len(subs), np.nan)
    for i, f in enumerate(cue_first):
        if f >= 0 and match[f] >= 0:
            ref[i] = tr_time[match[f]]
    return ref, len(subs)


class CountingTranscriber:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.multilingual = getattr(inner, "multilingual", True)

    def transcribe(self, audio, sr, language, *, start=0.0):
        self.calls += 1
        return self.inner.transcribe(audio, sr, language, start=start)

    def detect_language(self, audio, sr):
        return self.inner.detect_language(audio, sr)


def run_sync(args, out: Path) -> dict:
    import vlcsubsync
    from vlcsubsync.config import Config
    from vlcsubsync.sync import SubtitleSource, sync_subtitles
    from vlcsubsync.transcribe import get_transcriber

    cfg = Config()
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        cfg._set(k.strip().lower(), v.strip())
    tr = get_transcriber(cfg, "en")
    tr.load()  # the daemon keeps the model warm: don't time loading
    counter = CountingTranscriber(tr)
    src = SubtitleSource("embedded", index=args.sub) if not args.sub_path else None
    if src is None:
        src = SubtitleSource("external", path=args.sub_path)
    t0 = time.monotonic()
    r = sync_subtitles(
        args.media, args.audio, src, str(out / f"{args.label}.srt"), cfg, transcriber=counter
    )
    dt = time.monotonic() - t0
    info = {
        "label": args.label,
        "code": str(Path(vlcsubsync.__file__).parent),
        "settings": args.set or [],
        "runtime_s": round(dt, 1),
        "windows": counter.calls,
        "device": tr.device,
        "applied": r.applied,
        "method": r.method,
        "scale": r.scale,
        "offset": r.offset,
        "segments": r.segments,
        "confidence": r.confidence,
        "message": r.message,
        "output": r.output_path,
    }
    (out / f"{args.label}.json").write_text(json.dumps(info, indent=1))
    return info


def evaluate(srt: str, ref: np.ndarray) -> dict:
    import pysubs2

    out = pysubs2.load(srt)
    starts = np.array([e.start / 1000.0 for e in out])
    if starts.size != ref.size:
        raise SystemExit(f"{srt}: {starts.size} cues, original has {ref.size}")
    ok = ~np.isnan(ref)
    d = starts[ok] - ref[ok]
    e = np.abs(d)
    return {
        "cues": int(ok.sum()),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "signed_median": float(np.median(d)),
        "within_0.5": float(np.mean(e <= 0.5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("media")
    ap.add_argument("--audio", type=int, default=0, help="audio stream ordinal")
    ap.add_argument("--sub", type=int, default=0, help="embedded subtitle stream ordinal")
    ap.add_argument("--sub-path", help="external subtitle file instead of --sub")
    ap.add_argument("--out", required=True, help="output directory (results, cache)")
    ap.add_argument("--label", default="current", help="name of this run")
    ap.add_argument("--set", action="append", help="config key=value for the sync run")
    ap.add_argument("--truth-model", default=None, help="default: small.en on CUDA, else base.en")
    ap.add_argument("--truth-device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--min-run", type=int, default=3, help="min matched tokens in a row")
    ap.add_argument("--report", action="store_true", help="only print the table of all runs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = args.truth_model
    if model is None:
        try:
            import ctranslate2

            cuda = ctranslate2.get_cuda_device_count() > 0 and args.truth_device != "cpu"
        except Exception:
            cuda = False
        model = "small.en" if cuda else "base.en"
    if args.sub_path:
        import shutil

        shutil.copy(args.sub_path, out / "original.srt")  # same cue order as the output
    elif not (out / "original.srt").exists():
        load_original(args.media, args.sub, out)
    words = ground_truth(args.media, args.audio, out, model, args.truth_device)
    ref, n_cues = reference_times(out / "original.srt", words, args.min_run)

    if not args.report:
        info = run_sync(args, out)
        print(json.dumps(info, indent=1))

    rows = []
    orig = evaluate(str(out / "original.srt"), ref)
    rows.append(("original (unsynced)", None, orig))
    for jf in sorted(out.glob("*.json")):
        if jf.name.startswith("truth_"):
            continue
        info = json.loads(jf.read_text())
        if not Path(info["output"]).exists():
            continue
        rows.append((info["label"], info, evaluate(info["output"], ref)))
    print(f"\nground truth: {model}, {len(words)} words; {orig['cues']}/{n_cues} cues measured\n")
    hdr = "| run | median | p90 | p95 | signed med | <=0.5s | runtime | windows | result |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for label, info, m in rows:
        extra = (
            f"{info['runtime_s']:.1f}s | {info['windows']} | "
            f"scale {info['scale']:.4f}, {info['segments']} seg, conf {info['confidence']:.2f}"
            if info
            else "- | - | -"
        )
        print(
            f"| {label} | {m['median']:.3f}s | {m['p90']:.3f}s | {m['p95']:.3f}s | "
            f"{m['signed_median']:+.3f}s | {m['within_0.5'] * 100:.0f}% | {extra} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
