"""Release assets for the release workflow (.github/workflows/release.yml) and CI.

    python scripts/release_assets.py constraints release/constraints.txt
    python scripts/release_assets.py installers <download URL> <wheel file name> release/

`constraints` exports the runtime dependency versions pinned in uv.lock (direct and
indirect, with platform markers) as a constraints file for `uv tool install`. It fails
if uv.lock is out of date with pyproject.toml. CI installs with the same file, so a
release installs exactly what its CI tested.

`installers` writes the released copies of install.sh, install.ps1 and install.cmd.
The ones in the repository install the main branch. The released ones install that
release's wheel with its constraints.txt, both from <download URL>
(https://github.com/<repo>/releases/download/<tag>), and install.cmd fetches that
release's install.ps1. Each rewritten line must be found exactly once.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hashes are left out: `uv tool install` does not check hashes in a constraints file.
EXPORT = "uv export --locked --no-dev --no-emit-project --no-hashes --format requirements.txt -q"

# installer -> [(pattern, replacement)]; {base} and {wheel} are filled in
REWRITES = {
    "install.sh": [
        (r"(?m)^DEFAULT_SOURCE=[^\r\n]*", 'DEFAULT_SOURCE="{base}/{wheel}"'),
        (r"(?m)^DEFAULT_CONSTRAINTS=[^\r\n]*", 'DEFAULT_CONSTRAINTS="{base}/constraints.txt"'),
    ],
    "install.ps1": [
        (r"(?m)^\$DefaultSource = '[^'\r\n]*'", "$DefaultSource = '{base}/{wheel}'"),
        (
            r"(?m)^\$DefaultConstraints = '[^'\r\n]*'",
            "$DefaultConstraints = '{base}/constraints.txt'",
        ),
    ],
    "install.cmd": [
        (
            r"https://raw\.githubusercontent\.com/[^/\s]+/[^/\s]+/main/install\.ps1",
            "{base}/install.ps1",
        ),
    ],
}


def export_constraints(out: Path, root: Path = ROOT) -> None:
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # relative when it can be, since uv writes the command into the file's header
    dest = os.path.relpath(out, root) if out.is_relative_to(root.resolve()) else str(out)
    subprocess.run([*EXPORT.split(), "--output-file", dest], cwd=root, check=True)


def write_installers(base: str, wheel: str, out_dir: Path, root: Path = ROOT) -> list[Path]:
    if not re.fullmatch(r"https://[^\s'\"]+[^/]", base):
        raise SystemExit(f"not an https download URL: {base!r}")
    if not re.fullmatch(r"[\w.+-]+\.whl", wheel):
        raise SystemExit(f"not a wheel file name: {wheel!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rules in REWRITES.items():
        text = (root / name).read_bytes().decode("utf-8")
        for pattern, template in rules:
            line = template.format(base=base, wheel=wheel)
            text, n = re.subn(pattern, lambda _m, line=line: line, text)
            if n != 1:
                raise SystemExit(f"{name}: expected one match for {pattern!r}, found {n}")
        path = out_dir / name
        path.write_bytes(text.encode("utf-8"))  # keep the file's line endings
        written.append(path)
    return written


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "constraints":
        export_constraints(Path(argv[1]))
    elif len(argv) == 4 and argv[0] == "installers":
        for path in write_installers(argv[1], argv[2], Path(argv[3])):
            print(path)
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
