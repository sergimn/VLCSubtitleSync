<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg"><img src="assets/logo.svg" alt="SubSync — subtitles locked to the audio" width="560"></picture></p>

<p align="center">
  <a href="https://github.com/sergimn/VLCSubtitleSync/actions/workflows/ci.yml"><img src="https://github.com/sergimn/VLCSubtitleSync/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

**SubSync** makes VLC re-time your subtitles to the audio automatically. Pick a
subtitle track, keep watching, and a few seconds later the subtitles are swapped for a
synced copy. It fixes:

- **constant delays** (subtitles always 3 s early or late),
- **drift** (subtitles that slowly slide out of sync, e.g. a 25 fps subtitle on a
  23.976 fps video, or subtitles made for a 29.97 fps TV broadcast, which drift
  by 25%: minutes off by the end of an episode),
- **cuts** (the subtitle was made for a version with an extra scene or ad break, so
  it jumps out of sync halfway through).

It listens to the audio track you have selected with OpenAI's Whisper speech model
(running locally on your computer, offline after the first download) and matches
what is said to the subtitle text. No account, no upload, nothing to click.

## Install

You need [VLC 3](https://www.videolan.org/vlc/). The installer sets up everything else
(including its own Python) and adds SubSync to VLC. A small helper program then
starts together with VLC and quits shortly after VLC closes: nothing runs at login.

**Linux / macOS**: paste in a terminal:

```sh
curl -LsSf https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.sh | sh
```

**Windows**: download [`install.cmd`](https://github.com/sergimn/VLCSubtitleSync/releases/latest/download/install.cmd)
and double-click it, or paste in PowerShell:

```powershell
powershell -c "irm https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.ps1 | iex"
```

Then **restart VLC**. The first install downloads about 200 MB (Python runtime,
speech-recognition libraries and the English model).

The installer finds VLC wherever it is installed: the regular system package, the
Linux **snap** and **flatpak**, the macOS app and the Windows installer. Your existing
VLC settings are kept. Before it changes `vlcrc` it saves a backup copy next to it.

## Usage

Open a video and choose a subtitle track (embedded, or a `.srt`/`.ass`/`.vtt` file next
to the video). That's all. You'll see *"Syncing subtitles…"* in the top-right corner, then
*"Subtitles synced: offset +2.35s"*. A new subtitle track is added and selected. The
original track is still in the **Subtitle → Sub Track** menu if you want it back.

A sync runs again when you pick a different subtitle track or a different audio
track (for example after switching to another language's dub). Results are cached, so
reopening the same file is instant.

From the menu **View → SubSync** you can:

| Menu item | What it does |
|---|---|
| Sync subtitles now | force a re-sync of the current tracks (ignores the cache) |
| Sync now (exhaustive) | re-sync by transcribing the whole file, 30 s at a time: for files the normal sync gets wrong. Much slower, especially on a CPU (see [Sync modes](#sync-modes)) |
| Auto-sync: ON/OFF | turn automatic syncing on or off (remembered) |
| Status… | show what SubSync is doing and the last result |
| Experimental: no extra track (live delay): ON/OFF | correct the subtitle track you picked instead of adding a synced copy (remembered; see [below](#experimental-no-extra-subtitle-track-live-delay)) |

> The extension has to be enabled once per VLC session from the View menu. Automatic
> syncing does not need it.

### How long does it take?

The first sync of a file usually takes **30–90 seconds on a typical CPU** (seconds
with an NVIDIA GPU). SubSync only transcribes a few 30-second samples spread over the
file, not the whole movie. Playback continues normally meanwhile.

The section check (see *How it works*) may transcribe a few extra samples: at most
2, plus 1 per 30 minutes of video, by default. Often none are needed, but on a CPU each
one adds a second or so (one extra sample on a 28-minute episode: 19.2 s instead of
19.1 s). Set `verify_windows=0` in `config.ini` to never transcribe extra samples
for it.

## How it works

```
 VLC  ── SubSync Lua script ──writes request──►  queue folder  ◄──watches── vlc-subsync helper
  ▲      (tracks your audio/subtitle choice)                         (runs only while VLC does)
  │                                                                               │
  │                                         1. decode the selected audio track    │
  │                                         2. find speech (voice activity)       │
  └──── loads the synced subtitle ◄──────── 3. Whisper: words + timestamps        │
                                            4. match words to subtitle lines      │
                                            5. fit offset / drift / cuts, write   ┘
```

VLC can only run Lua scripts, so the speech recognition runs in a separate helper
program, `vlc-subsync`. The two sides talk through small files in VLC's own data
folder, which also works for the sandboxed snap and flatpak builds of VLC.

The helper is not a login service. It starts when VLC starts and exits about 15
seconds after VLC closes (or after the last sync finishes, if one is still running):

| System | What starts the helper |
|---|---|
| Linux with systemd | `vlc-subsync.path` watches the SubSync files in VLC's folders and starts `vlc-subsync.service` when VLC writes them. Only that file watch is active while VLC is closed; no process runs. |
| Linux without systemd | the SubSync script inside VLC starts the helper (not possible for snap/flatpak VLC, see Troubleshooting) |
| macOS | a LaunchAgent that launchd starts when VLC writes its status file (no `KeepAlive`, no `RunAtLoad`) |
| Windows | the SubSync script inside VLC starts `vlc-subsync-daemon.exe` |

While it runs, the helper uses low priority (nice 10 and idle disk priority on Linux,
background priority on macOS, "below normal" on Windows) and at most half your CPU
cores, so playback stays smooth. It frees the speech model from memory a minute after
the last sync.

Step 5 in more detail: SubSync first fits one timing line (offset and, if the
subtitles were made for another frame rate, a drift factor) per section of the video.
It then checks each section in the middle, transcribing one more sample there if
needed. Where the check disagrees by more than a quarter second, the section is split
in two and each half is refitted, recursively. Finally it fine-tunes each section:
the median gap between the spoken words and the subtitle lines, the moments where
speech starts, and a gentle correction that follows slow local wobble (never more
than half a second, and smooth from line to line).

If the subtitle language and the spoken language differ (say English audio with
Spanish subtitles), there is no text to compare. SubSync then lines up the subtitle
timing with the moments where people are speaking. That method is less precise but
still fixes delays and frame-rate drift.

## Command line

The helper doubles as a command-line tool:

```sh
# sync one file (writes movie.synced.srt next to the video)
vlc-subsync sync movie.mkv                       # audio track 0, subtitle track 0
vlc-subsync sync movie.mkv --audio 1 --sub 0     # 2nd audio track, 1st subtitle track
vlc-subsync sync movie.mkv --sub-file movie.en.srt -o fixed.srt
vlc-subsync sync movie.mkv --model small.en --device cuda
vlc-subsync sync movie.mkv --mode exhaustive     # transcribe everything (slow)

vlc-subsync doctor            # check the installation
vlc-subsync setup             # (re)install the VLC integration
vlc-subsync download-models --all
vlc-subsync serve             # run the helper in the foreground (for debugging);
                              # it exits ~15 s after VLC closes, add --persistent to keep it
vlc-subsync uninstall [--purge]
```

Track numbers start at 0 and follow VLC's order: audio tracks as listed in **Audio →
Audio Track**; subtitle tracks are the embedded ones first, then subtitle files next
to the video (same name, e.g. `movie.srt`, `movie.en.srt`, also in `Subs/` or
`Subtitles/` folders).

## Configuration

Optional. Create or edit `config.ini` in the SubSync config folder:

| OS | file |
|---|---|
| Linux | `~/.config/vlc-subsync/config.ini` |
| macOS | `~/Library/Application Support/vlc-subsync/config.ini` |
| Windows | `%LOCALAPPDATA%\vlc-subsync\config.ini` |

<!-- TODO(lead): verify macOS/Windows paths = platformdirs.user_config_dir("vlc-subsync", appauthor=False). `vlc-subsync doctor` prints the actual path. -->

```ini
mode=fast               # fast | thorough | exhaustive (see "Sync modes" below)
model_en=base.en        # Whisper model for English subtitles (tiny.en, base.en, small.en, ...)
model_multi=base        # model for every other language (tiny, base, small, medium, ...)
device=auto             # auto | cpu | cuda   (auto falls back to CPU if CUDA fails)
compute_type=int8       # CTranslate2 compute type (int8, int8_float16, float16, float32)
windows=auto            # number of 30 s audio samples to transcribe, or auto (per mode);
                        # a number is ignored in exhaustive mode, which covers everything
verify_windows=auto     # extra samples the section check may transcribe (0 = none)
min_confidence=0.5      # below this the original timing is kept (0..1)
threads=0               # CPU threads, 0 = automatic (half the cores, at most 8)
sync_mode=track         # track | delay (EXPERIMENTAL, see below)
```

`vlc-subsync doctor` prints the location it uses. The helper reads the file again for
every sync, so a change applies to the next one.

### Sync modes

| mode | what it transcribes |
|---|---|
| `fast` (default) | about one 30 s sample per 4 minutes (at least 8), plus a few extra samples to check each section |
| `thorough` | about 2.5× more samples (one per ~96 s, at least 20) and a 3× larger budget for the section check |
| `exhaustive` | the whole file in consecutive 30 s pieces, skipping the ones with no speech. All matches feed the fit and the fine-tuning. The section check uses those pieces and transcribes nothing extra |

Set the default with `mode=` in `config.ini`. You can also choose a mode for one run:
**View → SubSync → Sync now (exhaustive)** in VLC, or `vlc-subsync sync --mode …` on
the command line. A cached exhaustive result is reused when the file is opened again,
including by later fast syncs, but only if it was applied. A forced re-sync ("Sync
subtitles now" on an already synced track) replaces the cached results of the other
modes, so the newest result is what you get next time.

After upgrading SubSync, **restart VLC** before using "Sync now (exhaustive)". Until
then the old interface script keeps running and can't do it. The menu says so instead
of running a normal sync.

Measured on a real 28.7 min TV episode (`--audio 0 --sub 0`). Its subtitles were timed
for 29.97 fps, so they drift +25% and start almost 3 minutes off. Hardware: RTX 3050 Ti
Laptop GPU and i7-12700H CPU (20 threads), base.en model, already loaded. Accuracy is
the per-line start error against a full `small.en` transcript (240 of 464 lines
measured, `scripts/measure_real.py`):

| mode | windows | GPU (CUDA) | CPU | median | p90 | p95 |
|---|---|---|---|---|---|---|
| unsynced | – | – | – | 173 s | 280 s | 298 s |
| fast | 12 / 13 | 13.5 s | 19.7 s | 0.31 s | 0.79–0.80 s | 1.20–1.22 s |
| thorough | 22 / 24 | 17.4 s | 28.3 s | 0.30–0.31 s | 0.79–0.82 s | 1.19–1.21 s |
| exhaustive | 50 (8 silent skipped) | 25.8 s | 45.2 s | 0.31 s | 0.82–0.83 s | 1.18–1.21 s |

Ranges and "GPU / CPU" pairs cover the GPU and CPU runs. Runtimes include about 7.5 s
of audio decoding and speech detection. `vlc-subsync sync --mode …` writes the same
output; started cold, it adds about 1 s to load the model. All modes find the same
single +25% line, so the extra windows do not improve accuracy on this file. The
roughly 0.3 s median looks like the limit of this measurement: subtitle lines usually
start a little before the first word is spoken. Exhaustive mode is meant for harder files, such as ones with short
sections the samples miss or few matching words.

Rough exhaustive-mode cost per hour of video: about 1 minute on this GPU and about
2 minutes on this 20-thread CPU. Expect roughly 5–10 minutes per hour on an older
4-core laptop CPU (an estimate, not measured). Transcription time grows with the
number of windows with speech. Fast mode stays under a minute per hour on either
device.

### Experimental: no extra subtitle track (live delay)

**Off by default.** Normally SubSync adds the synced subtitles as a new track. With
`sync_mode=delay` in `config.ini`, or **View → SubSync → Experimental: no extra track
(live delay)**, it keeps the track you picked selected and corrects it while you
watch, through VLC's own subtitle delay (the one the <kbd>G</kbd>/<kbd>H</kbd> keys
change). The helper sends the timing it found (offset, drift and sections); twice a
second the script sets the delay that fits the current playback position. The menu
toggle wins over `config.ini` once you have used it.

You see *"Subtitles synced (live delay, experimental): …"*. Your own
<kbd>G</kbd>/<kbd>H</kbd> adjustments are kept on top of the correction. Switching to
another subtitle or audio track gives the delay back to your own value (normally 0).

Checked in a real VLC 3.0.24 (snap) on a 3-minute test video whose subtitles drift
−4% (25 fps subtitles on 23.976 fps audio): the median line appeared 0.06 s after
its speech started (0.07 s in the default mode, measured the same way), seeking
forward and back was corrected at once, and changing the delay while a line was on
screen never made it flicker.

Limitations, why it is experimental:

- **Subtitles that must appear earlier make VLC pause.** A negative subtitle delay
  makes VLC buffer that much more, and every time the delay reaches a new low VLC
  pauses playback (sound and picture) for the difference. With an offset that's a
  single pause; with drift towards earlier subtitles it is a short pause every second
  or two, adding up to the largest correction needed (about 5 s over the 3-minute test
  above; at −4% a 2-hour film would add up to almost 5 minutes). The message says
  *"may pause playback up to N s in total"* when this applies. Drift the other way
  (subtitles that must appear later) and positive offsets cost nothing.
- A line that VLC has already prepared keeps its old timing, so a correction takes
  effect from the next line or two, not on the line on screen. In the test, a line
  prepared just before a large change (your own −1 s) was cut short.
- At a cut (an ad break missing from the subtitles) the switch happens about a second
  early, so a line next to the cut can be off by the size of the cut.
- The delay applies to every subtitle track of the file, and VLC's track
  synchronization dialog shows SubSync's value, not yours.
- When the helper finds nothing to fix confidently, the original timing is kept, as
  in the default mode.

## Languages

- **English** subtitles use the English-only model (`base.en`), which is downloaded
  during installation.
- **Other languages** (Spanish, French, German, … anything Whisper knows) use the
  multilingual `base` model, downloaded automatically the first time it is needed
  (~150 MB). The subtitle language is guessed from its text.
- **Subtitle language ≠ audio language**: falls back to voice-activity alignment
  (see *How it works*).

Larger models (`small`, `medium`) are more robust on noisy soundtracks but slower; set
them in `config.ini`.

## Limitations

- **Text subtitles only.** Image-based subtitles (DVD VobSub, Blu-ray PGS) can't be
  re-timed. Convert them to SRT first (e.g. with Subtitle Edit).
- **Local files only.** Streams, network shares mounted as URLs and DVD/Blu-ray discs
  are not supported.
- **The first sync of a file takes ~30–90 s on CPU.** The synced track appears when
  it's ready, and results are cached afterwards.
- **Small residual offsets are possible.** The fit is one timing line per section of
  the video, built from a sample of the audio and fine-tuned locally, so individual
  lines can still be a fraction of a second early or late (on a real TV episode the
  typical line is within 0.3 s of the spoken words). Sections shorter than about four
  minutes, or a timing change between two samples that no check landed on, may be
  missed. Drift outside 0.78–1.28× (−22% / +28%) isn't handled.
- The live delay mode (no extra track) is experimental and off by default; see its
  section above for what it cannot do.
- Subtitles that don't match the dialogue at all (a different cut with re-edited
  lines, or a heavily paraphrased translation) may not be accepted. SubSync then keeps
  the original timing and tells you so instead of making things worse.

## Troubleshooting

Run:

```sh
vlc-subsync doctor
```

It checks every VLC installation it finds (Lua scripts, `vlcrc` settings, queue
folder), how the helper is started with VLC and whether it is running, the models and
CUDA. The helper not running is normal while VLC is closed. Things to try:

- **Nothing happens when I pick a subtitle**: restart VLC after installing. The
  script is loaded when VLC starts. Check *Tools → Messages* (verbosity 2) for lines
  starting with `[subsync]`.
- **"SubSync helper not running"**: re-run `vlc-subsync setup`, then check
  `vlc-subsync doctor`. On Linux, `systemctl --user status vlc-subsync.path
  vlc-subsync.service` and `journalctl --user -u vlc-subsync.service` show whether it
  was started and why it stopped. To see errors directly, close VLC and run
  `vlc-subsync serve --persistent`, then open VLC. Its log is in the SubSync log
  folder (`doctor` prints the path).
- **The helper keeps running after VLC closed**: it waits for a running sync to
  finish, then exits after about 15 s. If VLC crashed, it exits about 35 s later
  (VLC's status file goes stale after 20 s). A helper started with `--persistent`
  never exits by itself.
- **Snap VLC (Ubuntu)**: snap VLC cannot start programs, so it relies on systemd
  starting the helper (`vlc-subsync.path`). This works across snap updates: the
  watch follows `~/snap/vlc/current` to the new revision. If you install the snap
  after SubSync, run `vlc-subsync setup` again. Start VLC once before running setup
  so that `~/snap/vlc/current` exists. Without systemd, run
  `vlc-subsync serve --persistent` yourself before using snap VLC.
- **Flatpak VLC**: same as the snap. Files opened through the flatpak file chooser
  may have a `/run/user/…/doc/…` path, which the helper can still read.
  <!-- TODO(lead): verify document-portal paths with flatpak VLC -->
- **Windows: a console window flashes when VLC starts**: VLC can start programs only
  through `cmd.exe`, which gets a window for a moment. SubSync uses `start "" /B`, so
  `cmd.exe` exits at once and the helper itself (`vlc-subsync-daemon.exe`) has no
  window. Expect at most one brief flash per VLC session, and none if the helper is
  already running. This is based on Windows documentation and has not been tested
  on Windows yet. If your antivirus quarantined the helper, allow it and re-run the
  installer.

## Uninstall

```sh
curl -LsSf https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.sh | sh -s -- --uninstall
```

Windows: run `install.cmd --uninstall`, or in PowerShell
`$env:VLC_SUBSYNC_UNINSTALL=1; irm https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.ps1 | iex`.

If the program is still installed you can also run `vlc-subsync uninstall`, which
removes the VLC scripts, restores your `vlcrc` settings and removes what starts the
helper with VLC (`vlc-subsync.path` and `vlc-subsync.service` on Linux, the
LaunchAgent on macOS). It also removes the login autostart entries of older versions
if any are left. Add `--purge` to also delete the cache, logs, config and state. The
Whisper models stay in the shared Hugging Face cache (`~/.cache/huggingface`).

## Development

```sh
git clone https://github.com/sergimn/VLCSubtitleSync && cd VLCSubtitleSync
uv venv && uv pip install -e ".[dev]"
.venv/bin/pre-commit install       # ruff, shellcheck, actionlint, luacheck (if installed) on every commit
.venv/bin/pytest                 # unit tests (fast; excludes slow/vlc)
.venv/bin/pytest -m slow         # real Whisper models on real speech (downloads tiny.en, base.en, base)
.venv/bin/pytest -m vlc          # headless VLC + Lua script + helper, end to end
.venv/bin/pre-commit run --all-files
```

Dependency updates are proposed by [Renovate](https://docs.renovatebot.com/) (`renovate.json`): CI tooling and dev extras are grouped and automerged when CI passes; runtime dependencies always get a PR for manual review.

Test markers:

| marker | needs | CI job |
|---|---|---|
| *(none)* | nothing | `unit` (Linux/macOS/Windows × Python 3.10/3.12) |
| `slow` | Whisper models (cached in `~/.cache/huggingface`) | `integration` |
| `vlc` | a VLC 3 binary (`VLC_BIN` to override) | `e2e-vlc` (Ubuntu, apt VLC) |

The VLC test runs `vlc -I dummy --extraintf luaintf --lua-intf subsync` with the
scripts found through `VLC_DATA_PATH`, and with `HOME`/`XDG_*` pointing to a temp
dir, so your own VLC profile is never touched. A snap VLC ignores `XDG_DATA_HOME`,
so with a snap you must opt in (`VLC_SUBSYNC_E2E_ALLOW_SNAP=1`) and the test then uses
the real `~/snap/vlc/current/.local/share/vlc/subsync` queue folder, cleaning up after
itself.

### Test fixtures

`tests/fixtures/` holds about 2 MB of generated media with exact ground truth:

| file | content |
|---|---|
| `en_dialogue.mkv` | 2.7 min, 36-line two-voice English dialogue (Opus 24 kbps mono + tiny black video) |
| `en_dialogue.truth.srt` | cue start/end = measured speech onset/offset of each line |
| `en_dialogue.{offset_plus_3_2,offset_minus_7,drift_25_23976,offset_drift,cut}.srt` | corrupted copies (see `manifest.json` for exact parameters) |
| `en_dialogue.es_text.offset_plus_3_2.srt` | Spanish text on English timing (+3.2 s): language-mismatch / VAD-fallback case |
| `es_dialogue.mkv` + `.truth.srt` + `.offset_plus_3_2.srt` | the Spanish version, for the multilingual model |
| `multi_audio.mkv` | audio #0 Spanish dub, audio #1 English, embedded English SRT (+4.5 s) |
| `multi_audio.en.srt` | sidecar: English, drifted 25/23.976 and −1.5 s |
| `manifest.json` | machine-readable description + all truth cues |

They are synthesized with [Piper](https://github.com/OHF-Voice/piper1-gpl) TTS
(voices `en_US-lessac/ryan-medium`, `es_ES-davefx-medium`, `es_MX-claude-high`) and are
bit-for-bit reproducible on the same toolchain. To regenerate:

```sh
uv pip install -e ".[fixtures]"            # piper-tts (+ onnxruntime)
.venv/bin/python scripts/make_fixtures.py  # needs ffmpeg with libopus + libx264 (or set $FFMPEG)
```

The ~14-minute performance file is never committed. The test builds it from the
committed audio (PyAV only), or you can run
`python scripts/make_fixtures.py long --out /tmp/long --minutes 12`.

## License

MIT. See [LICENSE](LICENSE). Speech recognition by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) using OpenAI Whisper
models (MIT).
