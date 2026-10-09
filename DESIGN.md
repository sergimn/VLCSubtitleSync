# VLC SubSync — Design

Automatically re-times the **currently selected subtitle track** to the **currently
selected audio track** in VLC, fixing both constant delays and progressive drift
(e.g. 23.976↔25 fps or 29.97↔23.976 fps conversions, ad-break cuts) with zero user
input.

## Why this architecture

* VLC 3.x scripting is Lua only; Lua cannot run Whisper. Heavy lifting happens in a
  Python helper (`vlcsubsync`) that uses **faster-whisper** (CTranslate2) and **PyAV**
  (bundles FFmpeg libs, so no system ffmpeg is needed).
* Spawning processes from VLC Lua is unreliable: `os.execute` flashes a console window
  on Windows, blocks the interface, and is denied entirely for sandboxed VLC
  (snap/flatpak). So the Lua side and Python side talk through a **file-based job
  queue** in VLC's user-data dir, and the helper runs as a small per-user **daemon**
  started at login (it also keeps the Whisper model warm). Lua still *tries* to
  start the daemon itself (non-Windows, non-sandboxed) if no heartbeat is seen.
* Fully automatic mode needs a **Lua interface script** (`lua/intf/subsync.lua`,
  enabled via `vlcrc`: `extraintf=luaintf`, `lua-intf=subsync`), because VLC
  extensions must be activated manually each session. A companion **extension**
  (`lua/extensions/subsync_ext.lua`, View menu) provides "Sync now", auto on/off and
  status.

```
VLC ── intf/subsync.lua ──writes──► <q>/requests/<id>.req
   ▲                                     │
   │ add_subtitle(out)            vlc-subsync serve (daemon)
   │                                     │ decode audio (PyAV) → VAD → sample windows
   └──reads── <q>/jobs/<id>.status ◄─────┘ → Whisper words → anchors → piecewise fit
                <q>/out/<id>.srt               → write retimed subtitles
```

`<q>` (the *queue dir*) = `vlc.config.userdatadir() .. "/subsync"`, i.e.
| platform | queue dir |
|---|---|
| Linux native | `~/.local/share/vlc/subsync` |
| Linux snap | `~/snap/vlc/current/.local/share/vlc/subsync` |
| Linux flatpak | `~/.var/app/org.videolan.VLC/data/vlc/subsync` |
| macOS | `~/Library/Application Support/org.videolan.vlc/subsync` |
| Windows | `%APPDATA%\vlc\subsync` |

The daemon watches **every one of these that exists** (plus any given with
`--queue-dir`). Installer creates them (with `requests/`, `jobs/`, `out/`).

## Repository layout

```
pyproject.toml                 # package "vlc-subsync", module "vlcsubsync", hatchling
src/vlcsubsync/
  __init__.py                  # __version__
  cli.py                       # argparse entry: sync | serve | setup | uninstall | doctor | download-models
  config.py                    # Config dataclass, load/save (TOML-ish key=value in <config dir>/config.ini)
  media.py                     # PyAV: probe streams, decode audio track → 16k mono float32, extract embedded text subs
  subtitles.py                 # load/save via pysubs2; sidecar discovery; language guess
  transcribe.py                # faster-whisper wrapper (Transcriber protocol) + model mgmt
  vad.py                       # speech-activity (faster-whisper's silero VAD) → speech mask / segments
  align.py                     # anchors from words↔cues, robust piecewise-linear fit, VAD fallback
  sync.py                      # orchestration: sync_subtitles(...) -> SyncResult
  protocol.py                  # key=value file format, request/status dataclasses, atomic writes
  daemon.py                    # queue watcher, heartbeat, job runner, result cache
  setup_vlc.py                 # install/uninstall Lua scripts, vlcrc edits, autostart, dirs
  lua/intf/subsync.lua         # shipped as package data, copied into VLC by `setup`
  lua/extensions/subsync_ext.lua
tests/                         # pytest; tests/lua/ runs Lua via `lupa` with a mocked `vlc` module
tests/fixtures/                # small generated media + ground-truth subs (committed)
scripts/                       # fixture generation, dev helpers
install.sh  install.ps1  install.cmd   # one-liners: get uv → uv tool install → vlc-subsync setup
.github/workflows/             # ci.yml (lint, unit, lua, integration), release.yml
```

## Contracts

### Python API (core ↔ daemon)

```python
# vlcsubsync/sync.py
@dataclass
class SubtitleSource:
    kind: Literal["embedded", "external"]
    index: int | None = None      # ordinal among the container's subtitle streams (0-based), for embedded
    path: str | None = None       # for external

@dataclass
class SyncResult:
    output_path: str
    method: str                   # "whisper" | "vad" | "none"
    offset: float                 # seconds, at t=0 (new = old*scale + offset for the dominant segment)
    scale: float                  # drift factor of the dominant segment (1.0 = no drift)
    segments: int                 # number of piecewise segments
    confidence: float             # 0..1
    anchors: int
    applied: bool                 # False if confidence too low → output == original timing
    message: str                  # human summary, e.g. "offset +2.35s, drift +4.1%"

def sync_subtitles(media_path: str, audio_index: int, subtitle: SubtitleSource,
                   output_path: str, config: Config,
                   progress: Callable[[float, str], None] = lambda p, m: None,
                   transcriber: Transcriber | None = None) -> SyncResult
```
`audio_index` = ordinal among the container's **audio** streams (0-based), which is
the order VLC lists them. External subs: if VLC's sub ordinal ≥ number of embedded
text sub streams, the daemon resolves `path` via sidecar discovery
(`subtitles.find_sidecars(media_path)`: same stem / stem.*.ext in the media dir and
`Subs/`, `Subtitles/` subdirs; exts srt, ass, ssa, vtt, sub; sorted like VLC:
exact-stem match first, then alphabetical) and picks `ordinal - n_embedded`.

### File protocol (Lua ↔ daemon) — all files UTF-8 `key=value` lines

Writers always write `<name>.tmp` then `os.rename` (atomic). Readers ignore unknown keys.
Values are single-line (newlines replaced by spaces).

`<q>/requests/<id>.req` (Lua → daemon). `id` = `[0-9A-Za-z_-]+`, Lua uses `os.time()`+counter.
```
version=1
id=1728221234_1
media=/abs/path/to/movie.mkv          # local filesystem path (decoded from file:// URI)
audio_index=1                          # ordinal among audio tracks (0-based)
audio_label=Track 2 - [English]
sub_index=0                            # ordinal among subtitle tracks VLC lists (0-based, our own synced tracks excluded)
sub_label=Track 1 - [English]
sub_path=                              # optional explicit external subtitle path
force=0                                # 1 = ignore result cache
```
`<q>/jobs/<id>.status` (daemon → Lua), rewritten on each update:
```
id=...
state=queued|running|done|error
progress=0.42
message=Transcribing 3/10
output=/abs/path/<q>/out/<id>.srt      # when done
applied=1
method=whisper
offset=2.350
scale=1.0417
confidence=0.93
```
`<q>/heartbeat` (daemon): `time=<unix seconds>\npid=<pid>\nversion=<x.y.z>`; refreshed every ≤2 s.
Lua considers the daemon alive if `os.time() - time <= 10`.

`<q>/control` (extension → intf): `sync_now=<counter>`, `auto=1|0`. intf reacts when
`sync_now` increases. `<q>/intf_state` (intf → extension, display only): `state`,
`message`, `last_result`.

Daemon housekeeping: delete `.req` once picked up; delete jobs/out older than 7 days.
Result cache key = sha1(media path, size, mtime, audio_index, sub source identity,
model, version) → reuse previous output instantly unless `force=1`.

### Lua behaviour (intf)
Loop every ~500 ms (`vlc.misc.mwait`); VLC 3 has no `should_die()` — `mwait` raises "Interrupted." when the interface is closing, which ends the loop.
Trigger a sync when, for the current input, (a) playback started with a sub track
selected, (b) `spu-es` changed to a non-disabled track that isn't one we added, or
(c) `audio-es` changed while a sub track is selected; debounce 1.5 s; only one job in
flight per input (newer request supersedes). Track ordinals come from
`vlc.var.get_list(input, "audio-es"/"spu-es")` (skip value -1 "Disable"; skip ES ids we
added). On `done` + `applied=1`: `vlc.input.add_subtitle(output, true)` (try path, then
`vlc.strings.make_uri(output)`), remember the new ES id(s) as "ours → source", OSD
`"Subtitles synced: <message>"`. On error / not applied: short OSD message, keep
original. Remember the result per (input, audio, source) so re-selecting doesn't resync.
OSD messages via `vlc.osd.message(text, channel, "top-right", 3000000)`.

## Alignment algorithm (align.py)
1. Decode selected audio stream → 16 kHz mono float32.
2. VAD → speech segments over whole file (cheap).
3. Choose K windows (default ~ max(8, duration/240s), 30 s each) evenly across the
   file, snapped to speech-dense regions; transcribe with word timestamps.
   Language: guess from subtitle text (stopword heuristic); English → `*.en` model,
   else multilingual model with `language=<guess>`; if audio language (whisper
   detect) ≠ subtitle language → skip to VAD fallback.
4. Anchors: normalize words (lowercase, strip punctuation, numbers→digits); subtitle
   tokens get times inside each cue: `cue start + min(lead / scale, cap)`, where
   `lead` = characters before the token at a nominal 15 chars/s and `cap` = the same
   fraction of the cue's duration. The speaking rate is nominal in the *audio* clock,
   so under drift the in-cue offset is divided by the fitted scale (without this, a
   1.25× file biases later words by up to 25% of their in-cue offset; fit_mapping
   refits once with the first line's scale). Match rare-ish n-grams (n=3, then 2)
   between transcript windows and the *whole* subtitle token stream (no assumption on
   offset magnitude) → (sub_time, audio_time) pairs, weighted by n-gram uniqueness.
5. Fit `audio = scale*sub + offset` robustly (RANSAC + IRLS refine). Scales are bounded
   to 0.78–1.28 (−22% / +28%). Long segments use a free fit that snaps to a known
   framerate ratio (`align.SCALE_CANDIDATES`) when within 0.0015 of it; short or sparse
   segments pick the best-fitting ratio, with a prior towards 1.0 that grows with the
   ratio's size. Snapping is judged per anchor *or* per transcription window: a
   window's word timestamps share a bias of 0.2–0.4 s on real audio, so one biased
   window must not tilt the line off an exact ratio. The ratio is taken if the
   per-window median residuals are no worse, or if the free slope is within 2
   standard errors of it, with each window one observation (SE = robust spread of
   the window medians / (√windows · spread of window positions)). Ratios beyond ±10%
   also need at least 10 anchors:

   | ratio | typical cause |
   |---|---|
   | 1 | same release, only an offset |
   | 25/23.976, 23.976/25 | PAL ↔ film/NTSC-film (+4.3% / −4.1%) |
   | 24/23.976, 23.976/24 | 24 ↔ 23.976 fps (±0.1%) |
   | 25/24, 24/25 | PAL ↔ 24 fps (+4.2% / −4.0%) |
   | 29.97/23.976, 23.976/29.97 | NTSC TV ↔ film (+25% / −20%), seen on real TV episodes |
   | 29.97/25, 25/29.97 | NTSC TV ↔ PAL (+19.9% / −16.6%) |

   If residuals show
   ≥2 clusters (cuts), fit piecewise-linear with DP over time-sorted anchors with a
   segment penalty (segments share `scale` unless evidence otherwise). If anchors are
   too sparse in a region, transcribe more windows there (adaptive, bounded).
6. VAD fallback (language mismatch / too few anchors): cross-correlate the speech mask
   with the subtitle-on mask at 10 ms resolution over the same candidate scales
   (FFT), pick best.
7. Confidence: (1 − e^(−inliers/12)) · (0.35 + 0.65·inlier weight ratio) ·
   (0.4 + 0.6·window coverage) · e^(−max(0, residual − 0.25)/0.4) · 0.95^(segments−1).
   The residual is the median |inlier residual| after removing up to 0.2 s of each
   window's median (its shared timestamp bias is not misfit; a line off by more still
   shows the excess).
8. Quality gate: apply only if confidence ≥ threshold; clamp cue overlaps; write
   output in the source format when possible (ASS keeps styles), else SRT.

## Config (`config.ini` in platformdirs user config dir `vlc-subsync`)
`model_en=base.en`, `model_multi=base`, `device=auto|cpu|cuda`, `compute_type=int8`,
`windows=auto`, `min_confidence=0.5`, `threads=0`.
`device=auto` tries CUDA and silently falls back to CPU on any load error.

## Installation UX
* Linux/macOS: `curl -LsSf https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.sh | sh`
* Windows: double-click `install.cmd` (or `irm …/install.ps1 | iex`).
Steps: install `uv` if missing → `uv tool install --python 3.12 vlc-subsync@<archive url>`
→ `vlc-subsync setup` which: copies Lua scripts into every detected VLC (native,
snap, flatpak), edits each `vlcrc` (`extraintf` append `luaintf` preserving existing
entries, `lua-intf=subsync`) with a backup, creates queue dirs, registers autostart
(systemd user unit / launchd agent / Windows Startup shortcut to the GUI exe
`vlc-subsync-daemon`), starts the daemon, pre-downloads the default English model.
`vlc-subsync uninstall` reverses everything. `vlc-subsync doctor` diagnoses.
