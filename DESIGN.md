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
  queue** in VLC's user-data dir. The helper is a small per-user **daemon** that
  **starts with VLC and exits ~15 s after it** (never at login): a systemd path unit
  or a launchd agent starts it when VLC writes its `intf_state`; on Windows and on
  Linux without systemd the Lua intf launches it itself. See "Lifecycle".
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
  lifecycle.py                 # idle exit (VLC gone), model unloading, process priority
  setup_vlc.py                 # install/uninstall Lua scripts, vlcrc edits, start-with-VLC, dirs
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
`sub_start` of the next. `c` is #8's local refinement (`align.Segment.knots`, at most
±0.5 s): `seg<i>_knots=<sub_time>:<seconds>;…`, linearly interpolated and flat outside
the knots; absent = 0. This is exactly the mapping `retime` applies to the output file
(except its overlap/negative-time clean-up of individual cues). At most 256 segments
and 4096 knots in total: the writer logs a warning and drops the extra segments, or the
knots of the segments past the cap; readers (Python and Lua) apply the same knot cap.
A reader drops the whole mapping if any line is malformed. Written only when
`applied=1`; the result cache meta stores the same keys, so a cache hit returns them.
An *applied* cache entry without `segments` (stored before mappings were sent) is
treated as a miss and re-synced once; the new result replaces it. Track mode ignores
the keys and still loads `output`.
`<q>/heartbeat` (daemon): `time=<unix seconds>\npid=<pid>\nversion=<x.y.z>`; refreshed every ≤2 s.
Lua considers the daemon alive if `os.time() - time <= 10`.

`<q>/control` (extension → intf): `sync_now=<counter>`, `auto=1|0`, optionally
`sync_now_mode=exhaustive`, and `sync_mode=delay|track` once the "Experimental: no extra
track (live delay)" toggle was used (every control write keeps it; absent = the helper's
config decides, from the `sync_mode=` of the last done status). intf reacts when `sync_now` increases; the
`sync_now_mode` present in that same write becomes the request's `mode=` (absent =
no `mode` key, i.e. the configured mode). "Sync now (exhaustive)" in the extension
writes it; "Sync subtitles now" does not. `<q>/intf_state` (intf → extension,
display only, except `modes`/`sync_modes`): `state`, `message`, `last_result`, `modes` (the
`sync_now_mode` values it understands; an intf from before sync modes lacks it, and
the extension then asks for a VLC restart instead of claiming an exhaustive sync),
`sync_mode` (in effect), `sync_modes=track,delay` (an intf without it has no delay
mode: the toggle asks for a restart) and `delay_active=1|0`.

`<q>/launcher` (setup → intf), see "Lifecycle":
```
version=1
mode=spawn                             # spawn: intf starts exe; service: systemd/launchd does
exe=C:\Users\me\...\Scripts\vlc-subsync-daemon.exe   # absolute (8.3 short form if non-ASCII)
args=                                  # whitespace-separated arguments (POSIX: `serve`)
```

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
* A forced run (`force=1`) stores its result and deletes the other modes' entries
  for the same file and tracks, so the newest forced result wins on the next open.
* The order assumes the mode defaults. Explicit `windows=N` / `verify_windows=N` are
  not part of the key and can make "thorough" sample less than "fast". This is
  accepted: such settings are rare and were never in the key.

### Lua behaviour (intf)
Loop every ~500 ms (`vlc.misc.mwait`); VLC 3 has no `should_die()` — `mwait` raises "Interrupted." when the interface is closing, which ends the loop.
`intf_state` is rewritten at least every 5 s while VLC runs and gets `state=stopped`
when the loop ends; the daemon's idle exit relies on both. With `launcher` `mode=spawn`
the intf starts the helper at startup and whenever the heartbeat is stale (checked
every 2 s), at most once per 60 s and 3 times per session; `mode=service` never
spawns. Without a `launcher` file (set up by an older version) it tries once, when a
job finds no heartbeat, from the usual install paths (non-Windows, non-sandboxed).
Trigger a sync when, for the current input, (a) playback started with a sub track
selected, (b) `spu-es` changed to a non-disabled track that isn't one we added, or
(c) `audio-es` changed while a sub track is selected; debounce 1.5 s; only one job in
flight per input (newer request supersedes). Track ordinals come from
`vlc.var.get_list(input, "audio-es"/"spu-es")` (skip value -1 "Disable"; skip ES ids we
added). On `done` + `applied=1`: `vlc.input.add_subtitle(output, true)` (try path, then
`vlc.strings.make_uri(output)`), remember the new ES id(s) as "ours → source", OSD
`"Subtitles synced: <message>"`. On error / not applied: short OSD message, keep
original. Remember the result per (input, audio, source) so re-selecting doesn't resync.
In delay mode (below) a done result never calls `add_subtitle`.
OSD messages via `vlc.osd.message(text, channel, "top-right", 3000000)`. A job with
`mode=exhaustive` (or thorough) says so in its OSD/progress text: "Syncing subtitles
(exhaustive, may take a while)… 42% – Transcribing 20/58 (exhaustive)".

### Delay mode (experimental, off by default)

`sync_mode=delay` (config, or the extension toggle via `control`): instead of loading
a synced copy, the intf keeps the user's **original** track selected and corrects it
live through the input variable `spu-delay`. Default `track` behaviour is unchanged.

VLC 3.0.x semantics (checked in the 3.0.x sources):
* `spu-delay` and `time` are `VLC_VAR_INTEGER` in **µs** (`src/input/var.c`
  `input_ControlVarInit`: `var_Create(p_input, "spu-delay"|"time", VLC_VAR_INTEGER)`;
  `input.c`: initialised from `sub-delay` (1/10 s) × 100000). `vlc.var.set` takes an
  integer (`modules/lua/libs/variables.c`), so we pass rounded µs.
* **Positive = later**: `var.c` `EsDelayCallback` → `INPUT_CONTROL_SET_SPU_DELAY` →
  `UpdatePtsDelay` → `es_out_SetDelay(SPU_ES)` → `EsOutDecoderChangeDelay` →
  `input_DecoderChangeDelay`; `decoder.c` `DecoderFixTs` adds it to the subpicture's
  start/stop (`*pi_ts0 += i_es_delay`) in `DecoderPlaySpu`, i.e. **when the subtitle is
  decoded**, before it is queued to the vout. A subtitle already queued (or on screen)
  keeps its timing: no redraw, no flicker; a change applies from the next decoded one.
* **Negative values pause playback.** `input.c` `UpdatePtsDelay` adds
  `-min(audio-delay, spu-delay)` to the input's `pts_delay`, and `clock.c`
  `input_clock_SetJitter` only ever raises the clock's `pts_delay` ("TODO when
  increasing -> force rebuffering"). Every new minimum shifts all output later by the
  difference, i.e. a stall of that length; the total equals the most negative delay
  reached. `time` (`es_out.c` `ES_OUT_SET_TIMES`) is the demux time minus the
  buffering, i.e. the playback position.

Mapping inversion, each 500 ms tick while the synced combination is selected: read
`time` T (s). Pick the segment whose *audio-domain* span `[audio(sub_start),
audio(sub_end))` contains T (the later one if spans overlap, i.e. a backward cut; in a
gap, a forward cut, the upcoming one; after the end the last). Subtitle time
`s = (T − offset)/scale`, refined by 4 fixed-point steps `s = (T − offset − c(s))/scale`
when knots exist (c is bounded and slow); delay `d = T − s`. Because a subtitle takes
its delay when decoded, d is evaluated at `T + lead`, `lead = 1 s + max(0, −d(T))`
(the caching plus the extra buffering of a negative delay; measured: median error
+0.19 s without the lookahead, +0.06 s with it, same as track mode's +0.07 s). The
lead assumes the buffering of the *current* delay, but VLC never shrinks `pts_delay`:
after a backward seek (or once the delay rises again) the real buffering stays at the
historical low, so the lookahead is too short by that difference and the error is
about drift × difference (≈ 0.2 s after seeking back over −5 s at −4% drift). Set
`spu-delay = round(d·1e6) + bias` when it differs from the value we last set by
> 40 ms, at once on a seek (T moved > 2 s away from wall-clock progress), and on
start. Every set logs `[subsync] spu-delay=<µs> us (time=<T>s seg=<i> bias=<µs>)`
(dbg).

User bias: at start `bias` = the current `spu-delay` (normally 0). If the variable
differs from what we last set by ≥ 1 ms (hotkeys g/h, Track Synchronization), the
difference is added to `bias` and kept on top of every later correction; smaller
differences are ignored. On a switch of the subtitle or audio track, `spu-delay` is
set back to `bias`; re-selecting the synced combination re-applies the remembered
mapping (memo per media/audio/sub, no new request). Stopping (track switch, mode
toggle, a replacing result, intf exit) first folds a change the user made since the
last tick into `bias`. When the interface closes (`M.shutdown`, from `M.run`), the input
gets `bias` back, since VLC keeps the variable until the input ends.

Input identity: the intf creates a string variable `subsync-delay` on the input object
(`vlc.var.create`) holding `<token>|<bias>|<last set>`, updated on every set. A new
input object for the *same* URI (repeat one, a one-item playlist loop, stop and play
again) lacks it, so the intf drops its delay state without restoring or folding
anything (that input's `spu-delay` is fresh, from `sub-delay`) and starts over as for a
new input; the remembered mapping is applied again without a request. A restarted intf
that finds the variable with `last set` equal to the current `spu-delay` takes the
recorded bias instead of capturing its predecessor's correction as the user's. A
different URI or no input resets the state as before (nothing to restore).

Toggling the mode while playing moves the result over (delay → track: restore the bias,
load `output`; track → delay: select the original track). "Sync now" with the
corrected track selected forces a re-sync. A done status without segments is re-requested
once with `force=1` (bypassing a stale cache entry); if that one has none either (a helper
older than this protocol), it is not applied, the OSD says so, and nothing is remembered,
so a later selection asks again. OSD on start: `"Subtitles synced (live delay,
experimental): <message>"`, plus "– may pause playback up to N s in total" when the
mapping needs a delay below −0.5 s over `[0, length]`. With an unknown length (0) only 0
and the segment boundaries are evaluated, so an open last segment that keeps drifting
negative is underestimated.

Cost per tick: segment spans in the audio domain are computed once when the status is
parsed; `segment_at` is O(segments) and the knot lookup a binary search, so with the caps
(256 segments, 4096 knots) a tick is cheap; `min_delay` (once per start) is
O(segments²).

Real VLC 3.0.24 (snap) check, `tests/test_integration.py::test_vlc_delay_mode_end_to_end`
plus a screen-recorded run (Xvfb + x11grab, testsrc video, sidecar with −4% drift):
`spu-delay` went from +1.20 s to −5.36 s over the file; median line start error
+0.06 s (track mode +0.07 s, unsynced up to 5.2 s); total stall ≈ 5.4 s (wall clock
minus media time), as predicted; hotkey bias (+1 s via 20×h) kept for 50 s of
corrections, the line on screen when it changed was not redrawn or moved; when the
bias was removed again (−1 s) the line already queued kept its +1 s start but was cut
short (0.6 s of 1.5 s shown, cause not investigated); seeks ±60 s corrected on the next
tick (the first `time` after a seek is ~2 s off while VLC rebuffers, so two "seek"
sets happen). VLC prefers an installed `subsync.lua` in the user data
dir over `VLC_DATA_PATH`, so the test installs the intf as `subsync_e2e` with its own
queue name.

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
   (FFT). Each scale's peak height above its own baseline is divided by the same
   prior as `_best_candidate_scale` (1.0: ×1/0.98, others: 1 + 2·|s − 1|), and the
   winner must stand out against the best rival scale that maps the file > 3 s
   differently (`cross_scale`): more candidates must not mean more chances for a
   noise maximum to win.
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
`min_confidence=0.5`, `threads=0` (= `max(1, min(8, cpu_count // 2))`, leaving half
the cores to VLC), `sync_mode=track|delay` (experimental delay mode, default track;
reported to the intf in each done status). Invalid values are ignored (default kept). The daemon loads the
config for every job.
`device=auto` tries CUDA and silently falls back to CPU on any load error.

## Installation UX
* Linux/macOS: `curl -LsSf https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.sh | sh`
* Windows: double-click `install.cmd` (or `irm …/install.ps1 | iex`).
Steps: install `uv` if missing → `uv tool install --python 3.12 vlc-subsync@<archive url>`
→ `vlc-subsync setup` which: copies Lua scripts into every detected VLC (native,
snap, flatpak), edits each `vlcrc` (`extraintf` append `luaintf` preserving existing
entries, `lua-intf=subsync`) with a backup, creates queue dirs, registers the
start-with-VLC mechanism and the `launcher` files (see "Lifecycle"; nothing is
started now), removes the login autostart of earlier versions, pre-downloads the
default English model. `vlc-subsync uninstall` reverses everything (including any
leftover earlier-version autostart). `vlc-subsync doctor` diagnoses, including the
lifecycle state (path unit enabled/active, LaunchAgent loaded, launcher exe present,
leftover login autostart).

## Lifecycle (the helper runs only while VLC runs)

Owner requirement: the helper is **not** a login service.

**Exit** (`lifecycle.IdleExitPolicy`, checked every second; injectable monotonic and
wall clocks). VLC counts as alive if any watched queue's `intf_state` has
`time` ≤ 20 s old and `state != stopped`. When no VLC is alive and no job is queued
or running for 15 s, the daemon exits (code 0). A fresh daemon that has seen neither
VLC nor a job waits up to 60 s first (VLC may still be starting, or a request
started it); a `state=stopped` written in the last 60 s cuts that wait. So: VLC
closed → exit ~15–17 s later; VLC killed → ~35 s (20 s staleness + 15 s); a job in
flight always finishes first. Before exiting it polls `requests/` once more and then
empties `requests/` (else `DirectoryNotEmpty` / `QueueDirectories` would restart it
forever): non-`.req` entries older than 60 s (files and directories) are deleted, and
`.req` files still there after 10 s are ones it could not read or delete. Whatever it
cannot delete is moved to `<q>/rejected/` (cleaned after 7 days); if even that fails it
logs it once. At startup only the non-`.req` junk is cleaned. `serve --persistent` disables all this.
If a new instance starts while the old one is exiting, it waits up to 3 s for the lock.

**Models**: 60 s after the last job that ran the engine, the *worker thread* calls
`transcribe.clear_cache()` (drops each model under its transcriber lock), drops the
Silero VAD, runs `gc.collect()` and `malloc_trim(0)`, and logs the RSS before/after.
Running on the worker means it can never overlap a job. The 7-day result cache is
unchanged.

**Priority** (`lifecycle.lower_priority`, at `serve` start before any thread, since
Linux nice/ioprio are per thread and inherited): nice ≥ 10 (never lowered), Linux
`SCHED_BATCH` + idle I/O class via `ioprio_set` (raw syscall through ctypes, known
arches only), Windows `BELOW_NORMAL_PRIORITY_CLASS`. Every step is best effort and
never raises. `VLC_SUBSYNC_PRIORITY=normal` skips it (tests).

**Linux (systemd user session)**: `~/.config/systemd/user/vlc-subsync.path`, enabled
(`WantedBy=default.target`: only the inotify watch inside the user manager exists at
login, no process), with `PathModified=<q>/intf_state` and
`DirectoryNotEmpty=<q>/requests` for each usable queue dir, `Unit=vlc-subsync.service`.
The service is `Type=simple`, `Nice=10`, `IOSchedulingClass=idle`,
`CPUSchedulingPolicy=batch`, no `Restart=` (VLC's next 5 s write starts it again) and
no `[Install]` (never enabled by itself). `StartLimitIntervalSec=600` +
`StartLimitBurst=20` stop a crash loop (a VLC session normally starts it once);
re-running setup does `reset-failed`. When re-setup changes the watched paths (e.g. a
new snap install), it runs `daemon-reload` and then `restart vlc-subsync.path`, since
a running path unit keeps its old watches. The Lua intf's tmp+rename write triggers
`PathModified` (the watched inode is replaced).
*Snap*: the queue dir is `~/snap/vlc/current/.local/share/vlc/subsync`, and
`current` is a symlink to the revision dir that snapd switches on refresh. systemd
resolves the symlink when it sets the watch (inotify follows it), but it also watches
every parent directory, so the switch of `current` in `~/snap/vlc` makes it re-set
the watches on the new target. Verified with systemd 249: after swapping
`current` from `x1` to `x2`, writes to the old revision no longer trigger, and writes
through `current` (now `x2`) do. The snap queue dir is therefore written with
`current`, never with a revision number.
*No systemd user session*: `launcher` `mode=spawn`; the intf runs
`'<exe>' 'serve' >/dev/null 2>&1 &`. Snap/flatpak VLC cannot do that (sandbox), so
setup warns that those need a manually started `vlc-subsync serve --persistent`.
Migration: an old always-on unit (has `[Install]` or a `default.target.wants`
link) is `disable --now`'d *before* its file is replaced, and an XDG autostart
`.desktop` is removed.

**macOS**: `~/Library/LaunchAgents/io.github.sergimn.vlc-subsync.plist` with
`WatchPaths` = each `<q>/intf_state`, `QueueDirectories` = each `<q>/requests`,
`ProcessType=Background`, `LowPriorityIO=true`, no `KeepAlive`, no `RunAtLoad`; setup
`bootout`s and `bootstrap`s it. An old plist with `KeepAlive`/`RunAtLoad` is booted
out and replaced. Not verified on a Mac in this change. launchd also throttles
relaunches (10 s by default), which only delays a restart right after an exit.

**Windows**: setup writes `<q>/launcher` (`mode=spawn`, absolute path of the GUI-subsystem
`vlc-subsync-daemon.exe`, as an 8.3 short path if it has non-ASCII characters, since
Lua's `os.execute` uses the ANSI code page). The intf runs `start "" /B "<exe>"`.
Startup shortcut and `HKCU\…\Run\SubSync` of earlier versions are removed.
Options considered (VLC 3 Lua has no process API besides `os.execute`/`io.popen`, both
C `system()`/`_popen()`, i.e. `cmd.exe /c`):
* `os.execute('"<exe>"')`: cmd.exe gets a console (VLC has none to share), which may
  flash. cmd.exe does not wait for a GUI program, but `start` makes that explicit.
* `os.execute('start "" /B "<exe>"')`: same single cmd.exe flash, returns at once;
  the GUI exe opens no window. **Chosen.**
* `io.popen`: same cmd.exe, plus a pipe; no benefit.
* `wscript //B launcher.vbs` (`WshShell.Run cmd, 0`): still launched through cmd.exe,
  so the same flash. It only helps to hide *console* programs, adds a process, often
  triggers antivirus heuristics, and VBScript is deprecated by Microsoft.
Residual behaviour (from documentation, **not verified on Windows**): one brief
console flash when VLC starts and the helper is not already running; at most 3 per
session if it keeps dying. A `%` in the exe path could be expanded by cmd.exe (the
default install paths have none).
