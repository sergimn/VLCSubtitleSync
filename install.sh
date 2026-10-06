#!/bin/sh
# SubSync (vlc-subsync) installer for Linux and macOS.
#
#   curl -LsSf https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.sh | sh
#   curl -LsSf .../install.sh | sh -s -- --uninstall
#
# Options (anything else is passed on to `vlc-subsync setup`, e.g. --no-model):
#   --uninstall   remove the VLC integration and the program
#   --help        show this help
#
# Environment:
#   VLC_SUBSYNC_SOURCE   package source (default: GitHub main branch archive);
#                        may be a local checkout directory or archive for testing
#   VLC_SUBSYNC_PYTHON   Python version for the tool environment (default 3.12)
#   VLC_SUBSYNC_MIN_AGE_DAYS
#                        only install dependency releases at least this many days
#                        old (default 14, supply-chain protection; 0 disables)

set -eu

DEFAULT_SOURCE="https://github.com/sergimn/VLCSubtitleSync/archive/refs/heads/main.zip"
SOURCE="${VLC_SUBSYNC_SOURCE:-$DEFAULT_SOURCE}"
PYVER="${VLC_SUBSYNC_PYTHON:-3.12}"
MIN_AGE_DAYS="${VLC_SUBSYNC_MIN_AGE_DAYS:-14}"
ACTION=install

if [ -t 1 ]; then
    BOLD="$(printf '\033[1m')"; GREEN="$(printf '\033[32m')"
    RED="$(printf '\033[31m')"; YELLOW="$(printf '\033[33m')"; RESET="$(printf '\033[0m')"
else
    BOLD=""; GREEN=""; RED=""; YELLOW=""; RESET=""
fi

say() { printf '%s==>%s %s\n' "$BOLD$GREEN" "$RESET" "$*"; }
warn() { printf '%swarning:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die() { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
    sed -n '2,15p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' || true
}

# Collect pass-through args for `vlc-subsync setup` in "$@".
n=$#
while [ "$n" -gt 0 ]; do
    arg="$1"; shift; n=$((n - 1))
    case "$arg" in
        --uninstall) ACTION=uninstall ;;
        -h|--help) usage; exit 0 ;;
        *) set -- "$@" "$arg" ;;
    esac
done

download() { # url -> stdout
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        die "need curl or wget to download $1"
    fi
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for c in "${XDG_BIN_HOME:-}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$c" ]; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}

UV="$(find_uv || true)"

if [ "$ACTION" = uninstall ]; then
    say "Uninstalling SubSync"
    [ -n "$UV" ] || die "uv not found; nothing to uninstall?"
    BIN_DIR="$("$UV" tool dir --bin --color never 2>/dev/null || true)"
    if [ -n "$BIN_DIR" ] && [ -x "$BIN_DIR/vlc-subsync" ]; then
        "$BIN_DIR/vlc-subsync" uninstall "$@" || warn "vlc-subsync uninstall reported problems"
    elif command -v vlc-subsync >/dev/null 2>&1; then
        vlc-subsync uninstall "$@" || warn "vlc-subsync uninstall reported problems"
    else
        warn "vlc-subsync command not found; skipping VLC cleanup"
    fi
    "$UV" tool uninstall vlc-subsync || warn "uv tool uninstall failed"
    say "SubSync has been removed."
    exit 0
fi

say "Installing SubSync (automatic subtitle sync for VLC)"

if [ -z "$UV" ]; then
    say "Installing uv (Python package manager from astral.sh)"
    download https://astral.sh/uv/install.sh | sh || die "uv installation failed"
    UV="$(find_uv || true)"
    [ -n "$UV" ] || die "uv was installed but cannot be found; open a new terminal and re-run"
fi
say "Using uv: $UV"

# A local directory/file source is turned into a PEP 508 file:// reference.
SPEC="vlc-subsync @ $SOURCE"
if [ -e "$SOURCE" ]; then
    ABS="$(cd "$(dirname "$SOURCE")" && pwd)/$(basename "$SOURCE")"
    SPEC="vlc-subsync @ file://$ABS"
fi

# Supply-chain protection: ignore dependency releases newer than MIN_AGE_DAYS, so a
# freshly published malicious version can't reach users before it's noticed. uv
# records the cutoff in the tool receipt, so 'uv tool upgrade' keeps honouring it.
case "$MIN_AGE_DAYS" in
    '' | *[!0-9]*) die "VLC_SUBSYNC_MIN_AGE_DAYS must be a whole number of days" ;;
esac
AGE_ARGS=""
if [ "$MIN_AGE_DAYS" -gt 0 ]; then
    CUTOFF_EPOCH=$(( $(date -u +%s) - MIN_AGE_DAYS * 86400 ))
    # GNU/busybox date take -d @epoch; BSD/macOS date takes -r epoch.
    CUTOFF="$(date -u -d "@$CUTOFF_EPOCH" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
        || date -u -r "$CUTOFF_EPOCH" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)" \
        || die "cannot compute the dependency age cutoff (set VLC_SUBSYNC_MIN_AGE_DAYS=0 to skip)"
    AGE_ARGS="$CUTOFF"
    say "Using only dependency releases published before $CUTOFF ($MIN_AGE_DAYS days ago)"
fi

say "Installing vlc-subsync with Python $PYVER (this downloads ~200 MB the first time)"
if [ -n "$AGE_ARGS" ]; then
    "$UV" tool install --force --python "$PYVER" --exclude-newer "$AGE_ARGS" "$SPEC" \
        || die "package installation failed"
else
    "$UV" tool install --force --python "$PYVER" "$SPEC" || die "package installation failed"
fi

BIN_DIR="$("$UV" tool dir --bin --color never)"
EXE="$BIN_DIR/vlc-subsync"
[ -x "$EXE" ] || die "vlc-subsync was not found in $BIN_DIR after installation"

say "Configuring VLC"
"$EXE" setup "$@" || die "vlc-subsync setup failed (run '$EXE doctor' for details)"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH; run '$UV tool update-shell' to use 'vlc-subsync' directly" ;;
esac

printf '\n%sSubSync is installed.%s Restart VLC, open a video and choose a subtitle track.\n' "$BOLD" "$RESET"
