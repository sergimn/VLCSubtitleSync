"""User configuration (``config.ini`` in the platformdirs user config dir).

The file is a flat ``key=value`` list (``#``/``;`` comments, optional ``[section]``
headers are ignored).  Unknown keys are preserved on save so that other components
(or newer versions) can store their own settings in the same file.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

APP_NAME = "vlc-subsync"
CONFIG_FILENAME = "config.ini"

_VALID_DEVICES = ("auto", "cpu", "cuda")

# Sync modes, cheapest first (see DESIGN.md "Sync modes"):
#   fast        sampled 30 s windows (default ~1 per 4 min, at least 8) plus a small
#               bisection-verification budget (2 + 1 per 30 min).
#   thorough    ~2.5x more sampled windows (~1 per 96 s, at least 20) and a 3x larger
#               verification budget.
#   exhaustive  consecutive 30 s windows over the whole file ([0, 30), [30, 60), ...),
#               skipping windows without detected speech; every anchor feeds the fit,
#               subdivision and local refinement.  Costs roughly one Whisper pass over
#               all dialogue (minutes on CPU for an episode).
MODES = ("fast", "thorough", "exhaustive")
DEFAULT_MODE = "fast"


def normalize_mode(value: object) -> str | None:
    """``value`` as one of :data:`MODES` (case-insensitive), or None if invalid/empty."""
    v = str(value or "").strip().lower()
    return v if v in MODES else None


def mode_rank(mode: str) -> int:
    """0 for fast, 1 for thorough, 2 for exhaustive (unknown → fast)."""
    return MODES.index(mode) if mode in MODES else 0


def config_dir() -> Path:
    """Per-user configuration directory (``VLC_SUBSYNC_CONFIG_DIR`` overrides)."""
    env = os.environ.get("VLC_SUBSYNC_CONFIG_DIR")
    if env:
        return Path(env)
    from platformdirs import user_config_dir

    return Path(user_config_dir(APP_NAME, appauthor=False))


def default_config_path() -> Path:
    return config_dir() / CONFIG_FILENAME


@dataclass
class Config:
    mode: str = DEFAULT_MODE  # fast | thorough | exhaustive (see MODES)
    model_en: str = "base.en"
    model_multi: str = "base"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "int8"
    # Sampled 30 s windows: "auto" (per mode) or a positive integer, which overrides the
    # mode's count in fast/thorough. Exhaustive mode ignores it (it covers everything).
    windows: str = "auto"
    # Extra 30 s windows the bisection verification may transcribe: "auto" (per mode) or
    # an integer >= 0 (0 = verify with the windows already transcribed only).
    verify_windows: str = "auto"
    min_confidence: float = 0.5
    threads: int = 0  # 0 = auto: half the CPU cores, 1..8 (transcribe.default_threads)
    extra: dict[str, str] = field(default_factory=dict)  # unknown keys, preserved on save

    # -- derived helpers -------------------------------------------------------------
    @property
    def effective_mode(self) -> str:
        return normalize_mode(self.mode) or DEFAULT_MODE

    def with_mode(self, mode: str | None) -> Config:
        """Copy of this config with ``mode`` replaced (None/invalid → unchanged copy)."""
        m = normalize_mode(mode)
        return replace(self, mode=m or self.effective_mode, extra=dict(self.extra))

    def window_count(self, duration: float) -> int:
        """Number of sampled transcription windows for a file of ``duration`` seconds.

        Exhaustive mode does not sample (see ``sync.plan_windows``); this returns the
        number of consecutive 30 s windows covering the file, before silent ones are
        skipped.
        """
        mode = self.effective_mode
        if mode == "exhaustive":
            return max(1, math.ceil(max(0.0, duration) / 30.0))
        w = str(self.windows).strip().lower()
        if w and w != "auto":
            try:
                return max(1, int(float(w)))
            except (ValueError, OverflowError):
                pass
        if mode == "thorough":
            return max(20, round(duration / 96.0))
        return max(8, round(duration / 240.0))

    def verify_budget(self, duration: float) -> int:
        """Window budget of the bisection verification for a ``duration`` s file.

        Exhaustive mode defaults to 0: every window with speech is already transcribed,
        so a probe could only add a silent one.
        """
        v = str(self.verify_windows).strip().lower()
        if v and v != "auto":
            try:
                return max(0, int(float(v)))
            except (ValueError, OverflowError):
                pass
        mode = self.effective_mode
        base = 2 + int(duration // 1800)
        if mode == "thorough":
            return 3 * base
        if mode == "exhaustive":
            return 0
        return base

    # -- persistence -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> Config:
        """Load from ``path`` (default location if None). Missing file → defaults.

        Invalid values are ignored (the default is kept) rather than raising, so a
        hand-edited typo never prevents the daemon from starting.
        """
        p = Path(path) if path is not None else default_config_path()
        cfg = cls()
        try:
            text = p.read_text(encoding="utf-8-sig")
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            return cfg
        except UnicodeDecodeError:
            text = p.read_bytes().decode("latin-1")
        known = {f.name: f for f in fields(cls) if f.name != "extra"}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line[0] in "#;[":
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = _strip_inline_comment(value).strip().strip('"').strip("'")
            if key in known:
                cfg._set(key, value)
            else:
                cfg.extra[key] = value
        return cfg

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        """Write atomically (``.tmp`` + rename). Returns the path written."""
        p = Path(path) if path is not None else default_config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# vlc-subsync configuration (key=value)"]
        for f in fields(self):
            if f.name == "extra":
                continue
            lines.append(f"{f.name}={getattr(self, f.name)}")
        for k, v in sorted(self.extra.items()):
            lines.append(f"{k}={v}")
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, p)
        return p

    def _set(self, key: str, value: str) -> None:
        try:
            if key == "min_confidence":
                v = float(value)
                if 0.0 <= v <= 1.0:
                    self.min_confidence = v
            elif key == "threads":
                self.threads = max(0, int(float(value)))
            elif key == "mode":
                m = normalize_mode(value)
                if m:
                    self.mode = m
            elif key == "device":
                if value.lower() in _VALID_DEVICES:
                    self.device = value.lower()
            elif key == "verify_windows":
                v = value.lower()
                if v == "auto" or int(float(v)) >= 0:
                    self.verify_windows = v if v == "auto" else str(int(float(v)))
            elif key == "windows":
                w = value.lower()
                if w == "auto" or int(float(w)) > 0:
                    self.windows = w if w == "auto" else str(int(float(w)))
            elif value:
                setattr(self, key, value)
        except (ValueError, OverflowError):
            pass


def _strip_inline_comment(value: str) -> str:
    for marker in (" #", " ;", "\t#", "\t;"):
        i = value.find(marker)
        if i >= 0:
            value = value[:i]
    return value
