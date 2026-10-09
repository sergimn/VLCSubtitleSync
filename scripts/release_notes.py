"""Release notes for the release workflow (.github/workflows/release.yml).

    python scripts/release_notes.py skeleton 1.0.0 changes.md > release-notes/v1.0.0.md
    python scripts/release_notes.py render release-notes/v1.0.0.md > notes.md

`skeleton` writes the file the release PR asks a human to fill in; changes.md is
the list of merged PRs GitHub generates. `render` turns the edited file into the
GitHub Release text: HTML comments (the guidance) and sections left empty are
dropped.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "sergimn/VLCSubtitleSync")

SKELETON = """\
<!--
Release notes for SubSync {version}. Edit this file in the release PR; when the
PR is merged it becomes the text of the GitHub Release. Comments like this one
are dropped, and so is any section left empty.
-->

## Highlights

<!-- What a user notices in this release, in a few bullets. -->

## Install or upgrade

You need [VLC 3](https://www.videolan.org/vlc/). Close VLC, then:

**Linux / macOS**: paste in a terminal:

```sh
curl -LsSf https://github.com/{repo}/releases/download/v{version}/install.sh | sh
```

**Windows**: download `install.cmd` below and double-click it.

Then restart VLC. Running the installer again upgrades an existing install.

## Changes

<!-- Generated from the pull requests merged since the last release. Trim or regroup. -->

{changes}
"""


def skeleton(version: str, changes: str) -> str:
    # GitHub's list has its own "## What's Changed" etc.: nest them under "Changes"
    changes = re.sub(r"(?m)^## ", "### ", changes.strip())
    return SKELETON.format(version=version, repo=REPO, changes=changes)


def render(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # split into a preamble and "## " sections; drop sections with no content
    parts = re.split(r"(?m)^(?=## )", text)
    kept = []
    for part in parts:
        heading, _, body = part.partition("\n")
        if heading.startswith("## ") and not body.strip():
            continue
        kept.append(part.strip())
    out = "\n\n".join(p for p in kept if p)
    return re.sub(r"\n{3,}", "\n\n", out) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "skeleton":
        changes = Path(argv[2]).read_text(encoding="utf-8")
        sys.stdout.write(skeleton(argv[1], changes))
    elif len(argv) == 2 and argv[0] == "render":
        sys.stdout.write(render(Path(argv[1]).read_text(encoding="utf-8")))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
