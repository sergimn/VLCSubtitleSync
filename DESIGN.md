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
  (`lua/extensions/subsync_ext.lua`, View menu) provides "Sync now", "Sync now
  (exhaustive)", auto on/off and status.

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
    mapping_segments: list[MapSegment]  # full mapping (protocol.MapSegment), [] if not applied

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
mode=exhaustive                        # optional: fast|thorough|exhaustive, overrides config
```
`mode` is omitted for the default; unknown values are ignored (config mode used).
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
segments=2                             # done + applied: the whole mapping, see below
seg0=,1200.000,1.0000000,2.3500
seg0_knots=120.000:0.0500;240.000:-0.0200
seg1=1200.000,,1.0000000,32.3500
sync_mode=track                        # the helper's configured sync_mode (track|delay)
```
**Mapping segments.** In the subtitle clock (seconds), segment *i* maps
`audio = scale*sub + offset + c(sub)` for `sub_start <= sub < sub_end`:
`seg<i>=<sub_start>,<sub_end>,<scale>,<offset>`, time-ordered, an empty bound is open
(the first segment starts at −∞, the last ends at +∞), `sub_end` of one segment is
`sub_start` of the next. `c` is the local refinement (`align.Segment.knots`, at most
±0.5 s): `seg<i>_knots=<sub_time>:<seconds>;…`, linearly interpolated and flat outside
the knots; absent = 0. This is exactly the mapping `retime` applies to the output file
(except its overlap/negative-time clean-up of individual cues). At most 256 segments
and 4096 knots in total: the writer logs a warning and drops the extra segments, or the
knots of the segments past the cap; readers apply the same knot cap.
A reader drops the whole mapping if any line is malformed. Written only when
`applied=1`; the result cache meta stores the same keys, so a cache hit returns them.
An *applied* cache entry without `segments` (stored before mappings were sent) is
treated as a miss and re-synced once; the new result replaces it. Track mode ignores
the keys and still loads `output`.
`<q>/heartbeat` (daemon): `time=<unix seconds>\npid=<pid>\nversion=<x.y.z>`; refreshed every ≤2 s.
Lua considers the daemon alive if `os.time() - time <= 10`.

`<q>/control` (extension → intf): `sync_now=<counter>`, `auto=1|0`, and optionally
`sync_now_mode=exhaustive`. intf reacts when `sync_now` increases; the
`sync_now_mode` present in that same write becomes the request's `mode=` (absent =
no `mode` key, i.e. the configured mode). "Sync now (exhaustive)" in the extension
writes it; "Sync subtitles now" does not. `<q>/intf_state` (intf → extension,
display only, except `modes`): `state`, `message`, `last_result`, `modes` (the
`sync_now_mode` values it understands; an intf from before sync modes lacks it, and
the extension then asks for a VLC restart instead of claiming an exhaustive sync).

Daemon housekeeping: delete `.req` once picked up; delete jobs/out older than 7 days.
Result cache key = sha1(media path, size, mtime, audio_index, sub source identity,
model, version, effective mode) → reuse previous output instantly unless `force=1`.
Effective mode = the request's `mode=`, else the config's. A lookup tries the
exhaustive key, then thorough, down to the job's own mode: a result of a *more*
thorough mode also answers a later cheaper request for the same file and tracks
(it used at least the same evidence), never the other way round. So after
"Sync now (exhaustive)", reopening the file reuses the exhaustive result.
* Another mode's entry is only used if it was applied (`applied=1`). Example: an
  exhaustive run that lost its windows to CUDA OOM and fell back to VAD with
  `applied=0` must not block a fast sync that might succeed. The job's own mode
  reuses unapplied results as before, so hopeless work is not redone.
* A request for the running job's media and tracks cancels it when the user asked
  for another mode: an explicit different `mode=` ("Sync now (exhaustive)" during a
  fast run), or `force=1` in another mode ("Sync subtitles now" during an exhaustive
  run). Automatic requests (no `mode`, no `force`) wait for it.
* A forced run (`force=1`) stores its result and deletes the other modes' entries
  for the same file and tracks, so the newest forced result wins on the next open.
* The order assumes the mode defaults. Explicit `windows=N` / `verify_windows=N` are
  not part of the key and can make "thorough" sample less than "fast". This is
  accepted: such settings are rare and were never in the key.

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
OSD messages via `vlc.osd.message(text, channel, "top-right", 3000000)`. A job with
`mode=exhaustive` (or thorough) says so in its OSD/progress text: "Syncing subtitles
(exhaustive, may take a while)… 42% – Transcribing 20/58 (exhaustive)".

## Alignment algorithm (align.py)
1. Decode selected audio stream → 16 kHz mono float32.
2. VAD → speech segments over whole file (cheap).
3. Choose K windows (default ~ max(8, duration/240s), 30 s each) evenly across the
   file, snapped to speech-dense regions; transcribe with word timestamps. The
   count and the later verification budget depend on the sync mode (see "Sync
   modes"); exhaustive mode transcribes consecutive windows instead.
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
6. Bisection verification (`align.subdivide`; differences below the 1 s inlier
   threshold, e.g. 0.6 s steps that the fit absorbs into a tilted compromise line):
   * Each segment is checked against the window nearest its midpoint (transcribed
     there if none lies within `clamp(len/8, 30, 90)` s and the budget
     `verify_windows` allows: auto = 2 + 1 per 30 min) and against every window
     already inside it (free). A window's own consensus is the mode of the residuals
     of *all* its candidate anchors (however far from the line), kept if ≥ 6
     transcript tokens support it. The window *disagrees* if that consensus is
     > 0.25 s off the line (10 s off is disagreement, not missing evidence), or the
     line explains < 50% of what the consensus does; it is *unknown* only without
     a coherent consensus (music, silence, garbled words). A segment verifies only on
     positive evidence: at least one window agrees and none disagrees.
   * A probe window without coherent evidence is not agreement: other positions
     (±¼ of the segment) are tried while the budget lasts. A segment with no
     evidence at all is left as it is and counted in `verify_unverified`; empty
     probes in `verify_empty_probes`.
   * On disagreement two cuts are tried: the best cue gap near the midpoint, and the
     cue gap at the L1 change point of the per-window median residuals (sections at
     1/3 and 2/3 leave both midpoint halves mixed). Each half is refitted with known
     ratios only (dominant scale, parent scale, 1.0, or `_best_candidate_scale`; a free
     scale over a few windows follows local wobble and extrapolates badly), judged by
     the mean |window median| with 1.0, then the dominant scale, winning near-ties.
   * A split is kept if the halves differ by > 0.25 s + 2·SE (SE from the spread of
     per-window medians, so one window's timestamp bias is not a "section") and it
     explains the disagreement: the per-window error halves, or one half verifies on
     its own and drops it ≥ 20% (the other half is left to the recursion). Smooth
     wobble fails this test and is left to step 7.
   * A segment that is not split may still be replaced by a refit of its whole range
     that verifies (or halves its error): the initial dominant line can be a
     compromise. Recursion stops at 2×120 s, depth 5 or when the probe has nothing new.
   * Segments too short to split get one probe at their point farthest from any
     window (independent evidence) and are folded into a neighbour when at least as
     many audio regions vote for the neighbour as for them (ties favour fewer
     segments). A fold compares two lines, so each window votes for the line its
     consensus is closer to (if within 0.5 s); overlapping windows (centres < 30 s
     apart) share their audio and its timestamp bias and cast one vote. The safety
     net below leaves the outvoted windows out of its comparison. Real cases: with
     beam-1 CPU decoding the cold-open window(s) came back 1.6 s late and created a
     false 90 s first segment; in thorough mode an adaptive window overlapped the
     biased one and outvoted the probe 2:0 until overlapping windows voted once.
   * Adjacent segments merge when one refit of both *positively* verifies against all
     their windows (a genuine 10 s cut holding < 10% of the anchors stays: its windows
     disagree with any merged line). Boundaries of new splits
     use the usual cue-gap / speech-overlap placement. Safety net: the result is
     dropped if it explains < 90% of the inliers or its median residual grows > 0.05 s.
   * Known limit: a section needs two windows of evidence (one window's timestamps
     can be off by more than a second), so sections shorter than the window spacing
     (~4 min by default) can be missed.
7. Local refinement (`align.refine_local`), three steps that each work on the
   residual of the previous one, so they cannot fight, assembled into **one
   continuous correction curve relative to the fitted lines, bounded by ±0.5 s in
   total** (`_assemble_correction`): each segment contributes its shift (steps 1–2)
   and wobble knots (step 3), every segment boundary gets a shared knot (mean of both
   sides), and all values are clipped to ±0.5 s. The curve is linear between knots
   and flat outside them, so the total never exceeds the bound, and the mapping jumps
   at a boundary exactly as much as the fitted lines do. Refinement alone never
   creates a backward jump (which `retime` would resolve by squeezing earlier cues);
   genuine cuts keep theirs.
   1. per segment: shift by the weighted median residual of its inlier anchors
      (one candidate per transcript token);
   2. global: `refine_with_speech` snaps mapped cue starts to the nearest speech
      onset (±0.5 s) and applies the median shift when it is *precise*: standard
      error of the median ≤ 0.04 s and MAD ≤ 0.25 s (real dialogue has a 0.1–0.2 s
      MAD; the old MAD ≤ 0.12 gate rejected useful shifts). Speech onsets are
      independent of Whisper's timestamp bias, so they own the absolute level;
   3. per segment: smooth correction knots every 120 s (subtitle clock), from the
      ±120 s neighbourhood: per-window median anchor residuals (variance
      SE² + 0.2² for the window's shared timestamp bias) and median speech-onset
      deltas (SE²), each relative to its own segment baseline, combined by
      precision, shrunk towards 0 with a 0.2 s prior, smoothed [¼ ½ ¼] and linearly
      interpolated (`Segment.knots`). Slow wobble is followed; single lines are not
      moved individually.
8. VAD fallback (language mismatch / too few anchors): cross-correlate the speech mask
   with the subtitle-on mask at 10 ms resolution over the same candidate scales
   (FFT), pick best.
9. Confidence: (1 − e^(−inliers/12)) · (0.35 + 0.65·inlier weight ratio) ·
   (0.4 + 0.6·window coverage) · e^(−max(0, residual − 0.25)/0.4) · 0.95^(segments−1).
   The residual is the median |inlier residual| after removing up to 0.2 s of each
   window's median (its shared timestamp bias is not misfit; a line off by more still
   shows the excess). Verification probe windows count in the coverage denominator
   only if they contribute inliers.
10. Quality gate: apply only if confidence ≥ threshold; clamp cue overlaps; write
   output in the source format when possible (ASS keeps styles), else SRT.

## Sync modes (`mode=` in config, per-request `mode=`, `vlc-subsync sync --mode`)

| mode | windows (step 3) | adaptive (step 5) | verification budget (step 6) |
|---|---|---|---|
| `fast` (default) | `max(8, d/240 s)` sampled | ≤ `max(4, K/2)` | `2 + 1 per 30 min` |
| `thorough` | `max(20, d/96 s)` sampled (~2.5×) | ≤ `max(4, K/2)` | 3× fast |
| `exhaustive` | consecutive `[0,30) [30,60) …` up to `d`, minus silent ones | none | 0 |

* Exhaustive skips a window with < 0.5 s of VAD speech (`sync.MIN_SPEECH_SECONDS`);
  the last window may be shorter than 30 s. Every window with speech is transcribed,
  so adaptive and verification probes have nothing left to add (they could only pick
  silent windows); the fit, bisection verification (with the windows already
  inside each segment) and local refinement run on all anchors as usual.
* Explicit `windows=N` / `verify_windows=N` keep overriding the per-mode defaults
  (`windows` is ignored by exhaustive, which covers everything).
* Cost is roughly linear in the number of transcribed windows: exhaustive is one
  Whisper pass over all dialogue. On a 28.7 min episode it transcribed 50 windows,
  against fast's 12–13 (8 sampled + adaptive). That took 1.9× fast's runtime on GPU
  and 2.3× on CPU, because decoding and VAD (~7.5 s) are fixed. README has the table.

## Config (`config.ini` in platformdirs user config dir `vlc-subsync`)
`mode=fast|thorough|exhaustive`, `model_en=base.en`, `model_multi=base`,
`device=auto|cpu|cuda`, `compute_type=int8`, `windows=auto`, `verify_windows=auto`,
`min_confidence=0.5`, `threads=0`, `sync_mode=track|delay` (how VLC applies a result;
reported to the intf in each done status, default track). Invalid values are ignored
(default kept).
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
