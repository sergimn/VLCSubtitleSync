"""User configuration (``config.ini`` in the platformdirs user config dir).

The file is a flat ``key=value`` list (``#``/``;`` comments, optional ``[section]``
headers are ignored).  Unknown keys are preserved on save so that other components
(or newer versions) can store their own settings in the same file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

APP_NAME = "vlc-subsync"
CONFIG_FILENAME = "config.ini"

_VALID_DEVICES = ("auto", "cpu", "cuda")


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
    model_en: str = "base.en"
    model_multi: str = "base"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "int8"
    windows: str = "auto"  # "auto" or a positive integer (number of 30 s windows)
    # Extra 30 s windows the bisection verification may transcribe: "auto" or an
    # integer >= 0 (0 = verify with the windows already transcribed only).
    verify_windows: str = "auto"
    min_confidence: float = 0.5
    threads: int = 0  # 0 = let CTranslate2 decide
    extra: dict[str, str] = field(default_factory=dict)  # unknown keys, preserved on save

    # -- derived helpers -------------------------------------------------------------
    def window_count(self, duration: float) -> int:
        """Number of transcription windows for a file of ``duration`` seconds."""
        w = str(self.windows).strip().lower()
        if w and w != "auto":
            try:
                return max(1, int(float(w)))
            except (ValueError, OverflowError):
                pass
        return max(8, round(duration / 240.0))

    def verify_budget(self, duration: float) -> int:
        """Window budget of the bisection verification for a ``duration`` s file."""
        v = str(self.verify_windows).strip().lower()
        if v and v != "auto":
            try:
                return max(0, int(float(v)))
            except (ValueError, OverflowError):
                pass
        return 2 + int(duration // 1800)

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
