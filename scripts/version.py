"""Read, check or set the SubSync version in every file that carries it.

The version lives in four places that must agree: pyproject.toml, the Python
package (__version__) and the two Lua scripts (shown in VLC's extension list
and written to the heartbeat). The release workflow uses this to bump them.

    python scripts/version.py              # print the version (fails if the files disagree)
    python scripts/version.py set 1.0.0    # write 1.0.0 into every file
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"\d+\.\d+\.\d+")

# file -> pattern whose group 2 is the version (group 1 and 3 are kept as is)
FILES = {
    "pyproject.toml": r'(?m)^(version = ")([^"]+)(")',
    "src/vlcsubsync/__init__.py": r'(?m)^(__version__ = ")([^"]+)(")',
    "src/vlcsubsync/lua/extensions/subsync_ext.lua": r'(?m)^(E\.VERSION = ")([^"]+)(")',
    "src/vlcsubsync/lua/intf/subsync.lua": r'(?m)^(M\.VERSION = ")([^"]+)(")',
}


def read_versions(root: Path = ROOT) -> dict[str, str]:
    out = {}
    for rel, pattern in FILES.items():
        found = re.findall(pattern, (root / rel).read_bytes().decode("utf-8"))
        if len(found) != 1:
            raise SystemExit(f"{rel}: expected one version line, found {len(found)}")
        out[rel] = found[0][1]
    return out


def current_version(root: Path = ROOT) -> str:
    versions = read_versions(root)
    if len(set(versions.values())) != 1:
        lines = "\n".join(f"  {rel}: {v}" for rel, v in versions.items())
        raise SystemExit(f"version mismatch:\n{lines}")
    return next(iter(versions.values()))


def set_version(version: str, root: Path = ROOT) -> None:
    if not SEMVER.fullmatch(version):
        raise SystemExit(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    for rel, pattern in FILES.items():
        path = root / rel
        text = path.read_bytes().decode("utf-8")
        new, n = re.subn(pattern, rf"\g<1>{version}\g<3>", text)
        if n != 1:
            raise SystemExit(f"{rel}: expected one version line, found {n}")
        path.write_bytes(new.encode("utf-8"))  # keep the file's line endings
    if current_version(root) != version:
        raise SystemExit("version was not written everywhere")


def main(argv: list[str]) -> int:
    if not argv:
        print(current_version())
    elif len(argv) == 2 and argv[0] == "set":
        set_version(argv[1])
        print(argv[1])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
