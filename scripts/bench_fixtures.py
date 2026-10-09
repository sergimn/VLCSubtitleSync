#!/usr/bin/env python3
"""Sync accuracy benchmark on the committed TTS fixtures (plus the long perf file).

Every ``tests/fixtures/*.truth.srt`` cue starts exactly where its speech starts, so the
per-cue start error of a synced output is exact. CI runs this on every push to ``main``
(results in the step summary and the ``benchmark-results`` artifact) and on every PR,
where ``compare`` turns the PR's results and the latest ``main`` results into a comment::

    python scripts/bench_fixtures.py run --out /tmp/bench            # results.json + summary.md
    python scripts/bench_fixtures.py compare /tmp/bench/results.json --base main.json

Models run on CPU (as in the integration tests) so numbers are comparable across runners.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fixtures as mf  # noqa: E402
from measure_real import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
MARKER = "<!-- vlcsubsync-benchmark -->"
MULTI_MODEL = "base"
# A case counts as better/worse only past these margins (seconds).
MEDIAN_DELTA = 0.020
P95_DELTA = 0.050


@dataclass
class Case:
    id: str
    media: Path
    audio: int
    sub: Path | int  # external file, or embedded subtitle ordinal
    model_en: str
    truth: Path


def build_cases(models: list[str], long_paths: dict[str, Path] | None) -> list[Case]:
    manifest = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    mkv, truth = FIX / "en_dialogue.mkv", FIX / "en_dialogue.truth.srt"
    multi = FIX / "multi_audio.mkv"
    fast = models[0]
    cases = [
        Case(f"en/{name.split('.')[1]}/{m}", mkv, 0, FIX / name, m, truth)
        for m in models
        for name in sorted(manifest["files"]["en_dialogue.mkv"]["variants"])
    ]
    cases += [
        Case(f"en/truth-noop/{fast}", mkv, 0, truth, fast, truth),
        Case(
            f"es/offset_plus_3_2/{MULTI_MODEL}",
            FIX / "es_dialogue.mkv",
            0,
            FIX / "es_dialogue.offset_plus_3_2.srt",
            fast,
            FIX / "es_dialogue.truth.srt",
        ),
        Case(
            "es-text-on-en-audio/vad",
            mkv,
            0,
            FIX / "en_dialogue.es_text.offset_plus_3_2.srt",
            fast,
            truth,
        ),
        Case(f"multi_audio/a1-embedded-s0/{fast}", multi, 1, 0, fast, truth),
        Case(f"multi_audio/a1-sidecar/{fast}", multi, 1, FIX / "multi_audio.en.srt", fast, truth),
    ]
    if long_paths:
        cases.append(
            Case(
                f"long-12min/offset_plus_3_2/{fast}",
                long_paths["media"],
                0,
                long_paths["offset"],
                fast,
                long_paths["truth"],
            )
        )
    return cases


def run_case(case: Case, out_dir: Path) -> dict:
    from vlcsubsync.config import Config
    from vlcsubsync.sync import SubtitleSource, sync_subtitles
    from vlcsubsync.transcribe import get_model

    cfg = Config()
    cfg.model_en = case.model_en
    cfg.model_multi = MULTI_MODEL
    cfg.device = "cpu"
    # No transcriber is injected (that changes sync's language detection), so count
    # windows on the cached models sync will pick, loaded up front to keep load time out.
    calls = [0]
    models = {id(m): m for m in (get_model(cfg, cfg.model_en), get_model(cfg, MULTI_MODEL))}
    if isinstance(case.sub, int):
        src = SubtitleSource(kind="embedded", index=case.sub)
    else:
        src = SubtitleSource(kind="external", path=str(case.sub))
    rec: dict = {"id": case.id}
    try:
        for m in models.values():
            m.load()
            orig = m.transcribe

            def counted(*a, _orig=orig, **kw):
                calls[0] += 1
                return _orig(*a, **kw)

            m.transcribe = counted
        t0 = time.monotonic()
        r = sync_subtitles(
            str(case.media), case.audio, src, str(out_dir / f"{case.id.replace('/', '_')}.srt"), cfg
        )
        rec["runtime_s"] = round(time.monotonic() - t0, 1)
        rec.update(
            windows=calls[0],
            method=r.method,
            applied=r.applied,
            scale=r.scale,
            offset=r.offset,
            segments=r.segments,
            confidence=r.confidence,
        )
        ref = np.array([c.start for c in mf.parse_srt(case.truth)])
        rec.update(evaluate(r.output_path, ref))
        import pysubs2

        starts = np.array([e.start / 1000.0 for e in pysubs2.load(r.output_path)])
        rec["max"] = float(np.max(np.abs(starts - ref)))
        rec["errors"] = [round(float(x), 3) for x in np.abs(starts - ref)]
    except (Exception, SystemExit) as e:  # evaluate() exits on a cue-count mismatch
        rec["error"] = str(e) or type(e).__name__
    finally:
        for m in models.values():
            vars(m).pop("transcribe", None)
    return rec


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def overall(cases: list[dict], ids: set[str] | None = None) -> dict:
    errs = np.array(
        [e for c in cases if ids is None or c["id"] in ids for e in c.get("errors", ())]
    )
    if not errs.size:
        return {}
    return {
        "cues": int(errs.size),
        "median": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)),
        "within_0.5": float(np.mean(errs <= 0.5)),
    }


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")[:120]


def ms(x: float) -> str:
    return f"{x * 1000:.0f} ms"


def details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *body, "", "</details>"]


def results_summary(res: dict) -> list[str]:
    """Overall line + per-case table collapsed behind a dropdown."""
    o, n = res.get("overall"), len(res["cases"])
    failed = sum("error" in c for c in res["cases"])
    head = (
        f"**All {o['cues']} cues: median {ms(o['median'])} · p95 {ms(o['p95'])} · "
        f"{o['within_0.5']:.0%} ≤0.5 s**"
        if o
        else "**No case could be scored.**"
    )
    if failed:
        head += f" ❌ {failed} of {n} cases failed."
    return [head, "", *details(f"Per-case results ({n} cases)", results_table(res))]


def results_table(res: dict) -> list[str]:
    lines = [
        "| case | median | p95 | ≤0.5 s | method | windows | runtime |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in res["cases"]:
        if "error" in c:
            lines.append(f"| `{c['id']}` | ❌ {cell(c['error'])} | | | | | |")
            continue
        lines.append(
            f"| `{c['id']}` | {ms(c['median'])} | {ms(c['p95'])} | {c['within_0.5']:.0%} "
            f"| {c['method']} | {c['windows']} | {c['runtime_s']:.1f} s |"
        )
    return lines


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "srt").mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        long_paths = None if args.no_long else mf.make_long(Path(tmp), minutes=12.0)
        cases = build_cases(models, long_paths)
        records = []
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.id} ...", file=sys.stderr, flush=True)
            rec = run_case(case, out / "srt")
            print(
                f"    {json.dumps({k: v for k, v in rec.items() if k != 'errors'})}",
                file=sys.stderr,
                flush=True,
            )
            records.append(rec)
    res = {
        "sha": args.sha or git_sha(),
        "models": models,
        "cases": records,
        "overall": overall(records),
    }
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    summary = [f"### Subtitle sync benchmark — `{res['sha'][:7]}`", "", *results_summary(res)]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 1 if all("error" in c for c in records) else 0


def classify(h: dict, b: dict | None) -> tuple[str, str]:
    """(status, note) of a head case against its base case."""
    if b is None:
        return "🆕", "new case"
    if "error" in h:
        return ("❌", "still failing") if "error" in b else ("⚠️", "now fails")
    if "error" in b:
        return "✅", "fixed (failed on main)"
    note = f"method {b['method']} → {h['method']}" if h["method"] != b["method"] else ""
    dm, dp = h["median"] - b["median"], h["p95"] - b["p95"]
    if dm > MEDIAN_DELTA or dp > P95_DELTA:
        return "⚠️", note
    if dm < -MEDIAN_DELTA or dp < -P95_DELTA:
        return "✅", note
    return "≈", note


def delta(h: float, b: float) -> str:
    d = (h - b) * 1000
    return f"{h * 1000:.0f} ms ({d:+.0f})" if round(d) else f"{h * 1000:.0f} ms (=)"


def cmd_compare(args: argparse.Namespace) -> int:
    head = json.loads(Path(args.head).read_text(encoding="utf-8"))
    base = None
    if args.base and Path(args.base).is_file():
        base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    lines = [MARKER, "### Subtitle sync benchmark", ""]
    if base is None:
        lines += [
            f"No baseline from `main` yet. Results for `{head['sha'][:7]}` only.",
            "",
            *results_summary(head),
        ]
        print("\n".join(lines))
        return 0

    by_id = {c["id"]: c for c in base["cases"]}
    rows, worse, counts = [], [], {"✅": 0, "⚠️": 0, "≈": 0, "🆕": 0, "❌": 0}
    for h in head["cases"]:
        b = by_id.get(h["id"])
        status, note = classify(h, b)
        counts[status] += 1
        if status == "⚠️":
            worse.append(h["id"])
        if "error" in h:
            rows.append(f"| {status} | `{h['id']}` | {cell(h['error'])} | | | | | {note} |")
            continue
        if b is None or "error" in b:
            rows.append(
                f"| {status} | `{h['id']}` | {ms(h['median'])} | {ms(h['p95'])} "
                f"| {h['within_0.5']:.0%} | {h['method']} | {h['windows']} "
                f"| {h['runtime_s']:.1f} s {note} |"
            )
            continue
        dw = (h["within_0.5"] - b["within_0.5"]) * 100
        rows.append(
            f"| {status} | `{h['id']}` | {delta(h['median'], b['median'])} "
            f"| {delta(h['p95'], b['p95'])} | {h['within_0.5']:.0%} ({dw:+.0f} pp) "
            f"| {h['method']} | {h['windows']} ({h['windows'] - b['windows']:+d}) "
            f"| {h['runtime_s']:.1f} s ({h['runtime_s'] - b['runtime_s']:+.1f}) {note} |"
        )
    removed = sorted(set(by_id) - {c["id"] for c in head["cases"]})
    base_ref = f"`{base['sha'][:7]}`"
    if args.base_url:
        base_ref = f"[{base_ref}]({args.base_url})"
    parts = [f"{counts['✅']} better", f"{counts['⚠️']} worse", f"{counts['≈']} unchanged"]
    parts += [f"{counts[k]} {w}" for k, w in (("🆕", "new"), ("❌", "failing")) if counts[k]]
    parts += [f"{len(removed)} removed"] if removed else []
    lines += [
        f"{'⚠️ ' if counts['⚠️'] else ''}**{' · '.join(parts)}**: PR head "
        f"`{head['sha'][:7]}` (merged with its base) vs `main` at {base_ref}",
        "",
    ]
    ok = {c["id"] for c in head["cases"] if "error" not in c} & {
        c["id"] for c in base["cases"] if "error" not in c
    }
    ho, bo = overall(head["cases"], ok), overall(base["cases"], ok)
    if ho and bo:
        lines += [
            f"All {ho['cues']} cues of the cases scored in both: median "
            f"{delta(ho['median'], bo['median'])} · p95 {delta(ho['p95'], bo['p95'])} · "
            f"{ho['within_0.5']:.0%} ≤0.5 s",
            "",
        ]
    if worse:
        lines += ["Worse: " + ", ".join(f"`{w}`" for w in worse), ""]
    if removed:
        lines += ["Cases on `main` missing here: " + ", ".join(f"`{r}`" for r in removed), ""]
    table = [
        "|   | case | median Δ | p95 Δ | ≤0.5 s | method | windows | runtime |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Errors are per-cue start errors against the exact TTS ground truth. "
        f"⚠️/✅ = median moved by more than {MEDIAN_DELTA * 1000:.0f} ms or p95 by more "
        f"than {P95_DELTA * 1000:.0f} ms. Runtimes come from different runners and are "
        "only indicative. This comment is informational and never fails the check.",
    ]
    lines += details(f"Per-case results ({len(head['cases'])} cases)", table)
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every case, write results.json and summary.md")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--models", default="tiny.en,base.en", help="English models (first = fast)")
    r.add_argument("--sha", default=os.environ.get("BENCH_SHA"), help="commit being measured")
    r.add_argument("--no-long", action="store_true", help="skip the 12-minute file")
    c = sub.add_parser("compare", help="Markdown comparison of two results.json files")
    c.add_argument("head")
    c.add_argument("--base", help="baseline results.json (missing file = no baseline)")
    c.add_argument("--base-url", help="link to the baseline run")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
