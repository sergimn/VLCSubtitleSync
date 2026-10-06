#!/usr/bin/env python3
"""Generate the speech test fixtures in ``tests/fixtures/`` (and the long perf file).

Two sub-commands:

``all`` (default) -- needs the optional ``fixtures`` extra (``piper-tts``) and an
``ffmpeg`` binary with libopus + libx264. Synthesizes an English two-speaker dialogue
and a Spanish dub of it with Piper TTS, places every line on a timeline with
seeded-random gaps, measures the exact speech onset/offset of each line from the
synthesized samples and writes:

* ``en_dialogue.mkv``    tiny black video + English Opus mono audio
* ``en_dialogue.truth.srt`` ground truth (cue == spoken line, exact sample timing)
* ``en_dialogue.<variant>.srt`` corrupted copies (see ``EN_VARIANTS``)
* ``es_dialogue.mkv`` / ``es_dialogue.truth.srt`` / ``es_dialogue.offset_plus_3_2.srt``
* ``en_dialogue.es_text.offset_plus_3_2.srt`` Spanish *text* timed to the English
  audio (+3.2 s) -> language mismatch, exercises the VAD fallback
* ``multi_audio.mkv``   video + audio#0 Spanish dub + audio#1 English + embedded
  English SRT shifted by ``MULTI_EMBEDDED_OFFSET``; plus sidecar ``multi_audio.en.srt``
  (English, drifted 25/23.976 and shifted by ``MULTI_SIDECAR_OFFSET``)
* ``manifest.json``     machine-readable description of everything above

``long`` -- needs only PyAV (a runtime dependency). Builds a >= N-minute Opus
``.mka`` by concatenating the committed ``en_dialogue.mkv`` audio with varying
silences, plus its truth SRT and an offset copy. Used for performance checks; it is
generated on the fly (tests / CI), never committed.

Everything is deterministic: Piper runs with ``noise_scale=0, noise_w_scale=0``,
gaps come from a fixed-seed RNG, room tone from a fixed-seed numpy generator, and
ffmpeg is run single-threaded with ``bitexact`` flags. Ground truth is always
measured from the audio that is actually encoded, so even if a different Piper /
onnxruntime build produces slightly different waveforms the truth stays exact.

Regenerate::

    uv pip install --python .venv piper-tts      # or: pip install -e .[fixtures]
    .venv/bin/python scripts/make_fixtures.py all
    .venv/bin/python scripts/make_fixtures.py long --out /tmp/long --minutes 12

Voices are downloaded once from Hugging Face (rhasspy/piper-voices) into
``$VLC_SUBSYNC_FIXTURE_CACHE`` (default ``~/.cache/vlc-subsync-fixtures/piper``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

SR = 22050  # Piper "medium"/"high" voices all synthesize at 22.05 kHz
SEED = 20241006
LEAD_IN = 9.0  # first line starts after this -> the "-7 s" variant stays >= 0
TAIL = 3.0
GAP_RANGE = (0.3, 4.0)
ROOM_TONE_DBFS = -58.0
ONSET_THRESHOLD_DBFS = -42.0  # 5 ms frame RMS relative to full scale
OPUS_BITRATE = "24k"

FPS_A, FPS_B = 25.0, 24000 / 1001  # 23.976...
DRIFT = FPS_A / FPS_B  # 1.04270833...

MULTI_EMBEDDED_OFFSET = 4.5
MULTI_SIDECAR_OFFSET = -1.5

VOICES = {
    "en": ("en_US-lessac-medium", "en_US-ryan-medium"),
    "es": ("es_ES-davefx-medium", "es_MX-claude-high"),
}

# Speaker index (0/1), English line, Spanish translation.
# fmt: off
LINES: list[tuple[int, str, str]] = [
    (0, "Theo, wake up. The train leaves in forty minutes.",
        "Theo, despierta. El tren sale en cuarenta minutos."),
    (1, "Forty minutes? You told me we had the whole morning.",
        "¿Cuarenta minutos? Me dijiste que teníamos toda la mañana."),
    (0, "I said that yesterday, before the station changed the schedule.",
        "Eso lo dije ayer, antes de que la estación cambiara el horario."),
    (1, "Fine. Where did you put my blue jacket?",
        "Vale. ¿Dónde pusiste mi chaqueta azul?"),
    (0, "It's hanging behind the kitchen door, next to the umbrella.",
        "Está colgada detrás de la puerta de la cocina, junto al paraguas."),
    (1, "Did you remember to pack the tickets and the passports?",
        "¿Te acordaste de guardar los billetes y los pasaportes?"),
    (0, "Both of them are in the small pocket of my backpack.",
        "Los dos están en el bolsillo pequeño de mi mochila."),
    (1, "Good, because last time we almost missed the ferry.",
        "Bien, porque la última vez casi perdimos el ferri."),
    (0, "That was your fault. You wanted one more coffee.",
        "Eso fue culpa tuya. Querías otro café."),
    (1, "And it was the best coffee I have ever had.",
        "Y fue el mejor café que he tomado nunca."),
    (0, "Hurry up. The taxi driver is already waiting outside.",
        "Date prisa. El taxista ya está esperando fuera."),
    (1, "Tell him we need two more minutes, please.",
        "Dile que necesitamos dos minutos más, por favor."),
    (0, "Excuse me, is this the platform for the northern coast?",
        "Perdone, ¿es este el andén para la costa norte?"),
    (1, "I think so. The sign says platform number seven.",
        "Creo que sí. El cartel dice andén número siete."),
    (0, "Look at the mountains. I can't believe how green everything is.",
        "Mira las montañas. No puedo creer lo verde que está todo."),
    (1, "My grandmother grew up in a village just like that one.",
        "Mi abuela creció en un pueblo justo como ese."),
    (0, "Really? You never told me anything about her.",
        "¿En serio? Nunca me contaste nada de ella."),
    (1, "She was a fisherman's daughter. She could read the weather in the clouds.",
        "Era hija de un pescador. Sabía leer el tiempo en las nubes."),
    (0, "So what do the clouds say about this afternoon?",
        "¿Y qué dicen las nubes sobre esta tarde?"),
    (1, "They say you should have brought a warmer sweater.",
        "Dicen que deberías haber traído un jersey más abrigado."),
    (0, "Very funny. Let's find something to eat when we arrive.",
        "Muy gracioso. Busquemos algo de comer cuando lleguemos."),
    (1, "There is a little restaurant near the harbour that serves grilled sardines.",
        "Hay un restaurante pequeño cerca del puerto que sirve sardinas a la plancha."),
    (0, "Perfect. And after lunch we can walk to the lighthouse.",
        "Perfecto. Y después de comer podemos ir andando al faro."),
    (1, "The lighthouse is closed on Mondays, remember?",
        "El faro cierra los lunes, ¿recuerdas?"),
    (0, "Then we will climb the hill behind the old church instead.",
        "Entonces subiremos la colina detrás de la iglesia vieja."),
    (1, "Wait. Did you hear that announcement?",
        "Espera. ¿Has oído ese aviso?"),
    (0, "Something about a delay because of a broken signal.",
        "Algo sobre un retraso por una señal averiada."),
    (1, "How long do they expect us to wait here?",
        "¿Cuánto tiempo esperan que nos quedemos aquí?"),
    (0, "The conductor said about twenty minutes, maybe less.",
        "El revisor dijo unos veinte minutos, quizá menos."),
    (1, "Then I am going to buy a newspaper and a sandwich.",
        "Entonces voy a comprar un periódico y un bocadillo."),
    (0, "Bring me a bottle of water and some chocolate.",
        "Tráeme una botella de agua y algo de chocolate."),
    (1, "Dark or milk chocolate?",
        "¿Chocolate negro o con leche?"),
    (0, "Dark, obviously. I am not a child.",
        "Negro, obviamente. No soy una niña."),
    (1, "We are moving again. Finally.",
        "Ya nos movemos otra vez. Por fin."),
    (0, "Next stop is ours. Grab your bag and don't forget the camera.",
        "La próxima parada es la nuestra. Coge tu bolsa y no olvides la cámara."),
    (1, "I never forget the camera. That was you, in Lisbon.",
        "Yo nunca olvido la cámara. Fuiste tú, en Lisboa."),
]
# fmt: on


# ----------------------------------------------------------------------------- data


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class Track:
    lang: str
    audio: np.ndarray  # float32 mono @ SR
    cues: list[Cue]

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


# ----------------------------------------------------------------------------- SRT


def fmt_ts(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, cues: Sequence[Cue]) -> None:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{fmt_ts(c.start)} --> {fmt_ts(c.end)}\n{c.text}\n")
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")


def parse_srt(path: Path) -> list[Cue]:
    """Minimal SRT reader (shared with tests, which import this module)."""

    def ts(s: str) -> float:
        hms, ms = s.strip().replace(".", ",").split(",")
        h, m, sec = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    cues: list[Cue] = []
    blocks = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip().split("\n\n")
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        arrow = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if arrow is None:
            continue
        a, b = lines[arrow].split("-->")
        cues.append(Cue(ts(a), ts(b.split()[0]), "\n".join(lines[arrow + 1 :])))
    return cues


def transform(cues: Sequence[Cue], fn: Callable[[float], float]) -> list[Cue]:
    return [Cue(round(fn(c.start), 3), round(fn(c.end), 3), c.text) for c in cues]


def cut_point(cues: Sequence[Cue]) -> float:
    """Middle of the inter-cue gap closest to the temporal midpoint of the dialogue."""
    mid = (cues[0].start + cues[-1].end) / 2
    gaps = [((a.end + b.start) / 2, a, b) for a, b in zip(cues, cues[1:], strict=False)]
    return min(gaps, key=lambda g: abs(g[0] - mid))[0]


def en_variants(truth: Sequence[Cue]) -> dict[str, dict]:
    """name -> {"fn": time mapping truth->corrupted, "desc": ...}."""
    cp = cut_point(truth)
    return {
        "offset_plus_3_2": {"desc": "t + 3.2", "params": {"offset": 3.2}},
        "offset_minus_7": {"desc": "t - 7", "params": {"offset": -7.0}},
        "drift_25_23976": {"desc": "t * 25/23.976", "params": {"scale": DRIFT}},
        "offset_drift": {
            "desc": "t * 23.976/25 + 4.0",
            "params": {"scale": 1 / DRIFT, "offset": 4.0},
        },
        "cut": {
            "desc": f"t + 2 for t < {cp:.3f}, t + 12 after (simulated ad-break cut)",
            "params": {"cut_at": round(cp, 3), "offset_before": 2.0, "offset_after": 12.0},
        },
    }


def variant_fn(params: dict) -> Callable[[float], float]:
    if "cut_at" in params:
        c, a, b = params["cut_at"], params["offset_before"], params["offset_after"]
        return lambda t: t + (a if t < c else b)
    s, o = params.get("scale", 1.0), params.get("offset", 0.0)
    return lambda t: t * s + o


# ----------------------------------------------------------------------------- audio


def speech_bounds(x: np.ndarray) -> tuple[int, int]:
    """First/last sample of the region whose 5 ms RMS exceeds the onset threshold."""
    frame = int(SR * 0.005)
    n = len(x) // frame
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = np.nonzero(db > ONSET_THRESHOLD_DBFS)[0]
    if len(loud) == 0:
        raise RuntimeError("synthesized line is silent?")
    return int(loud[0] * frame), int((loud[-1] + 1) * frame)


def synthesize(lang: str, voices_dir: Path) -> list[np.ndarray]:
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError:  # pragma: no cover
        sys.exit("piper-tts is required: uv pip install --python .venv piper-tts")

    loaded = []
    for name in VOICES[lang]:
        onnx = voices_dir / f"{name}.onnx"
        if not onnx.exists():
            voices_dir.mkdir(parents=True, exist_ok=True)
            from piper.download_voices import download_voice

            print(f"downloading Piper voice {name} -> {voices_dir}")
            download_voice(name, voices_dir)
        v = PiperVoice.load(onnx)
        # single-threaded ONNX session: multi-threaded reductions are not bit-reproducible
        import onnxruntime

        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        v.session = onnxruntime.InferenceSession(
            str(onnx), sess_options=so, providers=["CPUExecutionProvider"]
        )
        if v.config.sample_rate != SR:
            raise RuntimeError(f"{name}: unexpected sample rate {v.config.sample_rate}")
        loaded.append(v)

    cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0, length_scale=1.0)
    clips = []
    for speaker, en, es in LINES:
        text = en if lang == "en" else es
        chunks = list(loaded[speaker].synthesize(text, cfg))
        x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        a, b = speech_bounds(x)
        x = x[a:b]
        # 10 ms fade in/out so the cut edges don't click
        f = int(SR * 0.01)
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        x[:f] *= ramp
        x[-f:] *= ramp[::-1]
        clips.append(x * np.float32(0.7 / max(1e-6, float(np.abs(x).max()))))
    return clips


def build_track(lang: str, clips: list[np.ndarray], seed: int) -> Track:
    rng = random.Random(seed)
    gaps = [rng.uniform(*GAP_RANGE) for _ in clips]
    total = LEAD_IN + sum(len(c) / SR for c in clips) + sum(gaps[:-1]) + TAIL
    audio = np.zeros(int(round(total * SR)) + SR, dtype=np.float32)
    cues: list[Cue] = []
    t = LEAD_IN
    for i, (clip, gap) in enumerate(zip(clips, gaps, strict=True)):
        s = int(round(t * SR))
        audio[s : s + len(clip)] += clip
        # measure from the placed samples (exactness), in seconds
        a, b = speech_bounds(clip)
        text = LINES[i][1] if lang == "en" else LINES[i][2]
        cues.append(Cue(round((s + a) / SR, 3), round((s + b) / SR, 3), text))
        t = (s + len(clip)) / SR + gap
    audio = audio[: int(round((cues[-1].end + TAIL) * SR))]
    noise = np.random.default_rng(seed).standard_normal(len(audio)).astype(np.float32)
    audio += noise * np.float32(10 ** (ROOM_TONE_DBFS / 20))
    return Track(lang, np.clip(audio, -1, 1), cues)


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------- ffmpeg


def find_ffmpeg() -> str:
    for cand in (os.environ.get("FFMPEG"), shutil.which("ffmpeg"), str(Path.home() / "bin/ffmpeg")):
        if cand and Path(cand).exists():
            return cand
    sys.exit("ffmpeg not found (set $FFMPEG)")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def mux(
    ffmpeg: str,
    out: Path,
    duration: float,
    audios: list[tuple[Path, str, str]],  # (wav, iso639-2 lang, title)
    subs: Sequence[tuple[Path, str, str]] = (),
) -> None:
    """Encode each elementary stream separately, then stream-copy mux.

    Encoding everything in one ffmpeg graph is *not* reproducible (ffmpeg's threaded
    scheduler changes how the Opus encoder is fed); separate single-input encodes are.
    """
    base = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    exact = ["-fflags", "+bitexact", "-map_metadata", "-1"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        video = tmp / "v.mkv"
        run(
            base
            + ["-f", "lavfi", "-i", f"color=c=black:s=64x36:r=2:d={duration:.3f}"]
            + ["-c:v", "libx264", "-preset", "veryslow", "-tune", "stillimage", "-crf", "40"]
            + ["-g", "20", "-pix_fmt", "yuv420p", "-flags:v", "+bitexact"]
            + ["-x264-params", "threads=1:lookahead-threads=1:sliced-threads=0"]
            + exact
            + [str(video)]
        )
        encoded = []
        for i, (wav, _, _) in enumerate(audios):
            enc = tmp / f"a{i}.mka"
            run(
                base
                + ["-i", str(wav), "-c:a", "libopus", "-b:a", OPUS_BITRATE, "-ac", "1"]
                + ["-ar", "48000", "-application", "voip", "-frame_duration", "20"]
                + ["-flags:a", "+bitexact"]
                + exact
                + [str(enc)]
            )
            encoded.append(enc)
        inputs = [video, *encoded, *(s for s, _, _ in subs)]
        cmd = list(base)
        for p in inputs:
            cmd += ["-i", str(p)]
        cmd += ["-map", "0:v"]
        cmd += [x for i in range(len(audios)) for x in ("-map", f"{i + 1}:a")]
        cmd += [x for i in range(len(subs)) for x in ("-map", f"{len(audios) + 1 + i}:s")]
        cmd += ["-c:v", "copy", "-c:a", "copy"]
        if subs:
            cmd += ["-c:s", "srt"]
        for kind, streams in (("a", audios), ("s", subs)):
            for i, (_, lang, title) in enumerate(streams):
                spec = f"-metadata:s:{kind}:{i}"
                cmd += [spec, f"language={lang}", spec, f"title={title}"]
        cmd += ["-disposition:a:0", "default"]
        cmd += exact + [str(out)]
        run(cmd)


# ----------------------------------------------------------------------------- commands


def cmd_all(args: argparse.Namespace) -> None:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    voices_dir = Path(
        args.voices
        or os.environ.get("VLC_SUBSYNC_FIXTURE_CACHE", "~/.cache/vlc-subsync-fixtures/piper")
    ).expanduser()

    en = build_track("en", synthesize("en", voices_dir), SEED)
    es = build_track("es", synthesize("es", voices_dir), SEED + 1)
    manifest: dict = {
        "generator": "scripts/make_fixtures.py",
        "sample_rate_measured": SR,
        "onset_threshold_dbfs": ONSET_THRESHOLD_DBFS,
        "files": {},
    }

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        en_wav, es_wav = tmp / "en.wav", tmp / "es.wav"
        write_wav(en_wav, en.audio)
        write_wav(es_wav, es.audio)

        # --- English single-track file + truth + variants
        write_srt(out / "en_dialogue.truth.srt", en.cues)
        mux(ffmpeg, out / "en_dialogue.mkv", en.duration, [(en_wav, "eng", "English")])
        variants = en_variants(en.cues)
        for name, v in variants.items():
            write_srt(out / f"en_dialogue.{name}.srt", transform(en.cues, variant_fn(v["params"])))
        manifest["files"]["en_dialogue.mkv"] = {
            "duration": round(en.duration, 3),
            "audio": [{"index": 0, "lang": "en"}],
            "truth": "en_dialogue.truth.srt",
            "variants": {f"en_dialogue.{k}.srt": v for k, v in variants.items()},
            "mismatch_variants": {
                "en_dialogue.es_text.offset_plus_3_2.srt": {
                    "desc": "Spanish text on English timing, t + 3.2 (language mismatch)",
                    "params": {"offset": 3.2},
                }
            },
        }
        # Spanish text, English timing -> language mismatch -> VAD fallback
        es_text_on_en = [Cue(c.start, c.end, LINES[i][2]) for i, c in enumerate(en.cues)]
        write_srt(
            out / "en_dialogue.es_text.offset_plus_3_2.srt",
            transform(es_text_on_en, lambda t: t + 3.2),
        )

        # --- Spanish single-track file
        write_srt(out / "es_dialogue.truth.srt", es.cues)
        write_srt(out / "es_dialogue.offset_plus_3_2.srt", transform(es.cues, lambda t: t + 3.2))
        mux(ffmpeg, out / "es_dialogue.mkv", es.duration, [(es_wav, "spa", "Español")])
        manifest["files"]["es_dialogue.mkv"] = {
            "duration": round(es.duration, 3),
            "audio": [{"index": 0, "lang": "es"}],
            "truth": "es_dialogue.truth.srt",
            "variants": {"es_dialogue.offset_plus_3_2.srt": {"params": {"offset": 3.2}}},
        }

        # --- two audio tracks + embedded sub + sidecar
        emb = tmp / "embedded.srt"
        write_srt(emb, transform(en.cues, lambda t: t + MULTI_EMBEDDED_OFFSET))
        dur = max(en.duration, es.duration)
        mux(
            ffmpeg,
            out / "multi_audio.mkv",
            dur,
            [(es_wav, "spa", "Español (doblaje)"), (en_wav, "eng", "English")],
            [(emb, "eng", "English")],
        )
        sidecar = {"scale": DRIFT, "offset": MULTI_SIDECAR_OFFSET}
        write_srt(out / "multi_audio.en.srt", transform(en.cues, variant_fn(sidecar)))
        manifest["files"]["multi_audio.mkv"] = {
            "duration": round(dur, 3),
            "audio": [
                {"index": 0, "lang": "es", "truth": "es_dialogue.truth.srt"},
                {"index": 1, "lang": "en", "truth": "en_dialogue.truth.srt"},
            ],
            "embedded_subs": [
                {
                    "index": 0,
                    "lang": "en",
                    "matches_audio": 1,
                    "params": {"offset": MULTI_EMBEDDED_OFFSET},
                }
            ],
            "sidecars": {
                "multi_audio.en.srt": {"lang": "en", "matches_audio": 1, "params": sidecar}
            },
        }

    manifest["cues"] = {"en": [asdict(c) for c in en.cues], "es": [asdict(c) for c in es.cues]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")

    total = 0
    for p in sorted(out.iterdir()):
        if p.is_file():
            total += p.stat().st_size
            print(f"{p.stat().st_size:>9}  {p.name}")
    print(f"{total:>9}  total  (en {en.duration:.1f}s, es {es.duration:.1f}s, {len(en.cues)} cues)")


def make_long(
    out_dir: Path,
    minutes: float = 12.0,
    source: Path = FIXTURES / "en_dialogue.mkv",
    truth: Path = FIXTURES / "en_dialogue.truth.srt",
    offset: float = 3.2,
    seed: int = SEED,
) -> dict[str, Path]:
    """Loop the committed English audio with varying silences into a long Opus .mka.

    Only needs PyAV. Returns paths: media, truth, offset (truth + ``offset``).
    Note: lines repeat every loop, so this file is for performance checks; n-gram
    anchors are ambiguous across loops (a realistic stress for the matcher too).
    """
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    rate = 16000
    with av.open(str(source)) as c:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        parts = []
        for frame in c.decode(audio=0):
            for f in resampler.resample(frame):
                parts.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            parts.append(f.to_ndarray().reshape(-1))
    pcm = np.concatenate(parts).astype(np.float32)
    base_cues = parse_srt(truth)
    seg_len = len(pcm) / rate

    rng = random.Random(seed)
    chunks, cues, t = [], [], 0.0
    while t < minutes * 60:
        gap = rng.uniform(2.0, 20.0)
        chunks.append(np.zeros(int(gap * rate), dtype=np.float32))
        t += int(gap * rate) / rate
        chunks.append(pcm)
        cues += [Cue(round(c.start + t, 3), round(c.end + t, 3), c.text) for c in base_cues]
        t += seg_len
    audio = np.concatenate(chunks)

    media = out_dir / "long_en.mka"
    with av.open(str(media), "w", format="matroska") as oc:
        st = oc.add_stream("libopus", rate=48000)
        st.layout = "mono"
        st.bit_rate = 24000
        st.metadata["language"] = "eng"
        up = av.AudioResampler(format="s16", layout="mono", rate=48000)
        hop = rate  # 1 s blocks
        for i in range(0, len(audio), hop):
            block = (np.clip(audio[i : i + hop], -1, 1) * 32767).astype(np.int16)
            fr = av.AudioFrame.from_ndarray(block.reshape(1, -1), format="s16", layout="mono")
            fr.sample_rate = rate
            fr.pts = None
            for rf in up.resample(fr):
                for pkt in st.encode(rf):
                    oc.mux(pkt)
        for rf in up.resample(None):
            for pkt in st.encode(rf):
                oc.mux(pkt)
        for pkt in st.encode(None):
            oc.mux(pkt)

    truth_out, off_out = out_dir / "long_en.truth.srt", out_dir / f"long_en.offset_{offset:+g}.srt"
    write_srt(truth_out, cues)
    write_srt(off_out, transform(cues, lambda x: x + offset))
    return {"media": media, "truth": truth_out, "offset": off_out}


def cmd_long(args: argparse.Namespace) -> None:
    paths = make_long(args.out, args.minutes)
    for k, p in paths.items():
        print(f"{k:7} {p} ({p.stat().st_size} bytes)")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("all", help="synthesize committed fixtures (needs piper-tts + ffmpeg)")
    a.add_argument("--out", type=Path, default=FIXTURES)
    a.add_argument("--voices", help="Piper voice cache dir")
    lg = sub.add_parser("long", help="build the long perf file from committed fixtures (PyAV)")
    lg.add_argument("--out", type=Path, required=True)
    lg.add_argument("--minutes", type=float, default=12.0)
    args = ap.parse_args(argv)
    if args.cmd in (None, "all"):
        if args.cmd is None:
            args = ap.parse_args(["all", *(argv or sys.argv[1:])])
        cmd_all(args)
    else:
        cmd_long(args)


if __name__ == "__main__":
    main()
