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

`<q>/launcher` (setup → intf), see "Lifecycle":
```
version=1
mode=spawn                             # spawn: intf starts exe; service: systemd/launchd does
exe=C:\Users\me\...\Scripts\vlc-subsync-daemon.exe   # absolute (8.3 short form if non-ASCII)
args=                                  # whitespace-separated arguments (POSIX: `serve`)
```

Daemon housekeeping: delete `.req` once picked up; delete jobs/out older than 7 days.
Result cache key = sha1(media path, size, mtime, audio_index, sub source identity,
model, version) → reuse previous output instantly unless `force=1`.

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
   tokens get times by interpolating within each cue. Match rare-ish n-grams (n=3, then
   2) between transcript windows and the *whole* subtitle token stream (no assumption
   on offset magnitude) → (sub_time, audio_time) pairs, weighted by n-gram uniqueness.
5. Fit `audio = scale*sub + offset` robustly (RANSAC + IRLS refine). Scales are bounded
   to 0.78–1.28 (−22% / +28%). Long segments use a free fit that snaps to a known
   framerate ratio (`align.SCALE_CANDIDATES`) when within 0.0015 of it; short or sparse
   segments pick the best-fitting ratio, with a prior towards 1.0 that grows with the
   ratio's size. Ratios beyond ±10% also need at least 10 anchors:

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
7. Quality gate: apply only if confidence ≥ threshold; clamp cue overlaps; write
   output in the source format when possible (ASS keeps styles), else SRT.

## Config (`config.ini` in platformdirs user config dir `vlc-subsync`)
`model_en=base.en`, `model_multi=base`, `device=auto|cpu|cuda`, `compute_type=int8`,
`windows=auto`, `min_confidence=0.5`, `threads=0` (= `max(1, min(8, cpu_count // 2))`,
leaving half the cores to VLC). The daemon loads the config for every job.
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
Started by systemd (`INVOCATION_ID` set) while another instance keeps the lock (e.g. a
manual `serve --persistent`), it stays up while VLC runs, retrying the lock, instead
of exiting: every exit would count toward the unit's start limit as VLC keeps
triggering the path unit, and hitting it fails the path unit until `reset-failed`.

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
