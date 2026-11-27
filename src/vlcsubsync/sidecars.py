"""VLC 3's own subtitle autodetection, so SubSync knows which subtitle files VLC loads
by itself for a video ("sidecars") and in which order it lists them.

A port of ``subtitles_Detect`` (src/input/subtitles.c) and of the sorts in
``LoadSlaves`` (src/input/input.c) of VLC 3.0 (VLC 4 has the same code), checked
against a real VLC on Linux, macOS and Windows by scripts/vlc_order_probe.py:

* VLC looks in the video's folder, then in the folders of its ``sub-autodetect-path``
  setting (``./Subtitles, ./subtitles, ./Subs, ./subs``; ``.\\subtitles, .\\subs`` on
  Windows; a path starting with ``.`` is relative to the video's folder). It reads each
  as spelled, so on a case-insensitive file system ``./Subs`` and ``./subs`` are the
  same folder, read twice: its files become two tracks each.
* It takes the files with a subtitle extension (:data:`VLC_SUBTITLE_EXTS`) that are not
  hidden, and gives each a priority from its name compared with the video's (both
  without the extension, as :func:`vlc_trim` gives them): the same name 4, the video's
  name followed by more words 3 ("Movie.en", "Movie - Spanish"), only words in front
  of it 2 ("Best of Movie"), any other name in the video's folder 1, in a subfolder 0.
  It keeps the files whose priority is at least ``sub-autodetect-fuzzy`` (3 by
  default; 0 turns it off, as does ``sub-autodetect-file``). It then drops a ``.sub``
  that an ``.idx`` of the same name goes with (VobSub), and a ``.cdg`` that hasn't the
  video's exact name.
* It sorts them twice with the C library's ``qsort``: by their ``file://`` URIs, then
  by priority, highest first. The URIs are compared with ``strcoll`` in VLC's
  ``LC_COLLATE`` locale on Linux, byte by byte on Windows and macOS (VLC for those is
  cross-compiled, and then its configure leaves ``strcoll`` out).
  ``qsort`` need not keep the order of equal items (on Windows it doesn't), and a
  dropped file stays in the first sort as an item equal to every other. So the order
  is the C library's: :class:`VlcSidecars` runs this machine's ``qsort`` and
  ``strcoll`` the way VLC does, which matches VLC when both run on the same machine.

Files given when the video opens take part in those sorts too, at the highest
priority (5), so VLC lists them first. The ``--sub-file`` (also the Open dialog's
subtitle file) is in the URI sort; right after it VLC deletes every item from the
second one on with the ``--sub-file``'s URI: its own copy among the sidecars if it
sorted first, else the ``--sub-file`` itself (and with it the string it compares
with, so on glibc and macOS nothing else matches after that). The files added during
an earlier play of the same playlist item come after the URI sort. Where they all
sit can change the order of equal sidecars: :meth:`VlcSidecars.lists` works it out
for a known ``--sub-file`` and tries every place an unknown one can be.

Each file is one track, except a VobSub ``.idx``: one track per language (``id:``
line) it lists, together at its place (:func:`vobsub_tracks`).
"""

from __future__ import annotations

import functools
import itertools
import logging
import os
import platform
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import quote

from .openfiles import VLC_SUBTITLE_EXTS, same_file_key

log = logging.getLogger("vlcsubsync.sidecars")

PRIORITY_ALL, PRIORITY_LEFT, PRIORITY_RIGHT, PRIORITY_NONE = 4, 3, 2, 1
PRIORITY_USER = 5  # files given when the video opens

WINDOWS_PATHS = ".\\subtitles, .\\subs"
POSIX_PATHS = "./Subtitles, ./subtitles, ./Subs, ./subs"

__all__ = [
    "VLC_SUBTITLE_EXTS",
    "Autodetect",
    "VlcSidecars",
    "may_be_autoloaded",
    "vlc_sidecars",
    "vlc_trim",
]


@dataclass(frozen=True)
class Autodetect:
    """VLC's subtitle autodetection settings for a video (the input's variables).

    None: not known (an older SubSync VLC script, or no VLC at all); VLC's defaults
    are assumed, and for the locale both the C locale and this process's."""

    enabled: bool | None = None  # sub-autodetect-file
    fuzzy: int | None = None  # sub-autodetect-fuzzy
    paths: str | None = None  # sub-autodetect-path
    collate: str | None = None  # VLC's LC_COLLATE locale

    @property
    def known(self) -> bool:
        return None not in (self.enabled, self.fuzzy, self.paths, self.collate)


def default_paths(windows: bool) -> str:
    """VLC 3's default ``sub-autodetect-path`` (src/libvlc-module.c)."""
    return WINDOWS_PATHS if windows else POSIX_PATHS


def vlc_trim(name: str) -> str:
    """``name`` the way VLC 3 compares subtitle and video names (``strcpy_trim``):
    its ASCII letters and digits, lowercased, as words separated by single spaces."""
    return " ".join(w.lower() for w in re.findall(r"[A-Za-z0-9]+", name))


def _strip_ext(name: str) -> str:
    dot = name.rfind(".")
    return name[:dot] if dot >= 0 else name


def _ext(name: str) -> str:
    dot = name.rfind(".")
    return name[dot + 1 :].lower() if dot >= 0 else ""


def match_priority(media_name: str, sub_name: str, in_media_dir: bool) -> int:
    """VLC's priority for subtitle file ``sub_name`` of video ``media_name`` (both file
    names)."""
    movie = vlc_trim(_strip_ext(media_name))
    sub = vlc_trim(_strip_ext(sub_name))
    if sub == movie:
        return PRIORITY_ALL
    at = sub.find(movie)
    if at >= 0:
        rest = sub[at + len(movie) :]
        return PRIORITY_LEFT if re.search(r"[A-Za-z0-9]", rest) else PRIORITY_RIGHT
    return PRIORITY_NONE if in_media_dir else 0


def search_dirs(media_dir: str, paths: str) -> list[str]:
    """VLC's ``paths_to_list``: the folders of a ``sub-autodetect-path`` value.

    ``media_dir`` ends with the path separator. Entries are separated by ``,`` (the
    spaces after a comma skipped); one starting with ``.`` is relative to
    ``media_dir``."""
    out = []
    for i, entry in enumerate(paths.split(",")):
        if i:
            entry = entry.lstrip(" ")
        if entry:
            out.append(media_dir + entry if entry.startswith(".") else entry)
    return out


def path_to_uri(path: str, windows: bool = False) -> str:
    """VLC's ``vlc_path2uri`` for an absolute path: each component percent-encoded (all
    but ASCII letters, digits and ``-._~``, upper-case hex)."""
    sep = "\\" if windows else "/"
    prefix = "file://"
    if windows and len(path) >= 2 and path[1] == ":" and path[0].isascii() and path[0].isalpha():
        prefix, path = f"file:///{path[0]}:", path[2:]
    elif windows and path.startswith("\\\\"):
        host, _, rest = path[2:].partition(sep)
        prefix, path = "file://" + host, sep + rest
    raw = (lambda p: p.encode("utf-8")) if windows else os.fsencode
    return prefix + "".join("/" + quote(raw(p), safe="") for p in path.split(sep)[1:])


def _split(media_path: str, windows: bool) -> tuple[str, str] | None:
    cut = media_path.rfind("\\" if windows else "/")
    return None if cut < 0 else (media_path[: cut + 1], media_path[cut + 1 :])


@dataclass(frozen=True)
class Entry:
    """A subtitle file VLC's scan found."""

    path: str  # as VLC builds it: folder as configured + name
    uri: str
    priority: int
    dropped: bool = False  # VLC drops it after the scan (the .sub of a VobSub pair, .cdg)
    tracks: int | None = 1  # how many tracks VLC makes of it (None: can't tell)


def scan(media_path: str, settings: Autodetect, windows: bool) -> list[Entry]:
    """VLC's ``subtitles_Detect`` scan: the files it keeps, in the order it finds them
    (folder by folder, each in the order the OS lists it)."""
    return _mark_dropped(_found(media_path, settings, windows))


def _found(media_path: str, settings: Autodetect, windows: bool) -> list[Entry]:
    if settings.enabled is False or settings.fuzzy == 0:
        return []
    fuzzy = 3 if settings.fuzzy is None else settings.fuzzy
    parts = _split(media_path, windows)
    if parts is None:
        return []
    media_dir, media_name = parts
    sep = "\\" if windows else "/"
    paths = default_paths(windows) if settings.paths is None else settings.paths
    found: list[Entry] = []
    for j, folder in enumerate([media_dir, *search_dirs(media_dir, paths)]):
        if j and folder == media_dir:
            continue
        try:
            with os.scandir(folder) as it:
                names = [e.name for e in it]
        except (OSError, ValueError):
            continue
        for name in names:
            if name.startswith(".") or "." + _ext(name) not in VLC_SUBTITLE_EXTS:
                continue
            prio = match_priority(media_name, name, j == 0)
            if prio < fuzzy:
                continue
            path = folder + name if folder.endswith(sep) else folder + sep + name
            if path == media_path or not os.path.isfile(path):
                continue
            found.append(Entry(path, path_to_uri(path, windows), prio, tracks=file_tracks(path)))
    return found


_VOBSUB_ID = re.compile(r"id:\s*(?:[^ ,]{1,32})?,\s*index:\s*[+-]?\d")


def vobsub_tracks(idx_path: str) -> int | None:
    """How many tracks VLC 3's VobSub reader makes of an ``.idx``: one per ``id:`` line
    it can read. One if it isn't VobSub (no "# VobSub index file" first line: another
    reader takes it, as any subtitle file). None if VLC can't open the ``.sub`` (the
    name with ``.sub`` for the last four characters, as VLC spells it): then it's
    unclear what VLC makes of it."""
    try:
        with open(idx_path, "rb") as f:
            text = f.read(4 << 20).decode("latin-1")
    except OSError:
        return None
    lines = text.split("\n")
    if "# vobsub index file" not in lines[0].lower():
        return 1
    if not os.path.isfile(idx_path[:-4] + ".sub"):
        return None
    return sum(1 for line in lines if line.startswith("id:") and _VOBSUB_ID.match(line))


def file_tracks(path: str) -> int | None:
    """How many subtitle tracks VLC makes of a subtitle file (see :func:`vobsub_tracks`)."""
    return vobsub_tracks(path) if _ext(path) == "idx" else 1


def _mark_dropped(found: Sequence[Entry]) -> list[Entry]:
    """VLC drops a ``.sub`` that an ``.idx`` of the same name goes with and a ``.cdg``
    without the video's exact name."""
    out = []
    for e in found:
        ext = _ext(e.uri)
        drop = ext == "cdg" and e.priority < PRIORITY_ALL
        if ext == "sub":
            stem = e.uri[:-3].lower()
            drop = any(o.uri.lower().startswith(stem) and _ext(o.uri) == "idx" for o in found)
        out.append(Entry(e.path, e.uri, e.priority, dropped=drop, tracks=e.tracks))
    return out


def startup_file(sub_file: str, windows: bool) -> tuple[str, str] | None:
    """The path and URI VLC uses for a ``--sub-file`` (``input_SubtitleFile2Uri``: a
    ``.sub`` with an ``.idx`` next to it is taken as the ``.idx``), or None for a
    relative path (VLC makes it absolute with its own working folder)."""
    path = sub_file
    if path.endswith(".sub") and os.path.isfile(path[:-4] + ".idx"):
        path = path[:-4] + ".idx"
    if windows:
        absolute = path.startswith("\\\\") or (
            len(path) >= 3 and path[0].isascii() and path[0].isalpha() and path[1:3] == ":\\"
        )
    else:
        absolute = path.startswith("/")
    return (path, path_to_uri(path, windows)) if absolute else None


# --------------------------------------------------------------------------------------
# The C library's sort and compare, as VLC uses them
# --------------------------------------------------------------------------------------


def _libc():  # pragma: no cover - depends on the OS
    import ctypes

    if sys.platform == "win32":
        return ctypes.cdll.msvcrt  # VLC 3 for Windows is built against msvcrt.dll
    return ctypes.CDLL(None)


@functools.cache
def _c_qsort() -> Callable[[list, Callable[[object, object], int]], list] | None:
    """``qsort`` of this machine's C library, sorting a Python list (None if
    unavailable)."""
    try:
        import ctypes

        qsort = _libc().qsort
        cmp_t = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
        )
        qsort.restype = None
        qsort.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, cmp_t]
    except (OSError, AttributeError, ImportError):
        return None

    def sort(items: list, cmp: Callable[[object, object], int]) -> list:
        n = len(items)
        if n < 2:
            return list(items)
        # pointer-sized items, as VLC sorts an array of pointers
        arr = (ctypes.c_size_t * n)(*range(n))
        cb = cmp_t(lambda a, b: cmp(items[a[0]], items[b[0]]))
        qsort(arr, n, ctypes.sizeof(ctypes.c_size_t), cb)
        return [items[i] for i in arr]

    return sort


def _msort(items: list, cmp: Callable[[object, object], int]) -> list:
    """glibc's ``qsort`` (``msort_with_tmp``): a top-down merge sort."""
    if len(items) <= 1:
        return list(items)
    n1 = len(items) // 2
    a, b = _msort(items[:n1], cmp), _msort(items[n1:], cmp)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if cmp(a[i], b[j]) <= 0:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]


def _sorter() -> tuple[Callable[[list, Callable[[object, object], int]], list], bool]:
    """The sort VLC's ``qsort`` does here, and whether it is exactly VLC's."""
    c = _c_qsort()
    if c is not None:
        return c, True
    return _msort, sys.platform.startswith("linux")  # glibc's, if this is glibc


# VLC compares the URIs with strcoll only where it was built on the OS it runs on:
# VLC for Windows and macOS is cross-compiled, so configure's AC_FUNC_STRCOLL can't run
# its test and VLC uses strcmp (checked with VLC 3.0.24 from videolan.org).
_VLC_USES_STRCOLL = sys.platform not in ("win32", "darwin")
_LC_COLLATE_MASK = 1 << 3  # glibc and musl: 1 << LC_COLLATE (3)


@functools.lru_cache(maxsize=8)
def _strcoll(name: str) -> Callable[[str, str], int] | None:
    """``strcoll`` in locale ``name``, without changing this process's locale (None if
    this OS or the locale is not available). "C" is a byte compare everywhere."""
    if name in ("C", "POSIX"):
        return lambda a, b: (a > b) - (a < b)  # URIs are ASCII
    if not _VLC_USES_STRCOLL:  # pragma: no cover - not needed there
        return None
    try:  # pragma: no cover - depends on the C library
        import ctypes

        libc = _libc()
        newlocale, strcoll_l = libc.newlocale, libc.strcoll_l
        newlocale.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p]
        newlocale.restype = ctypes.c_void_p
        strcoll_l.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        strcoll_l.restype = ctypes.c_int
        loc = newlocale(_LC_COLLATE_MASK, name.encode(), None)
    except (OSError, AttributeError, ImportError):
        return None
    if not loc:
        return None
    return lambda a, b: strcoll_l(a.encode(), b.encode(), loc)  # loc is kept for good


def env_collate() -> str:
    for var in ("LC_ALL", "LC_COLLATE", "LANG"):
        if os.environ.get(var):
            return os.environ[var]
    return "C"


def _ranks(uris: Sequence[str], coll: Callable[[str, str], int]) -> tuple[dict[int, int], int]:
    """Index -> rank of ``uris`` in ``coll`` order (equal ones share a rank), and the
    highest rank."""
    rank: dict[int, int] = {}
    prev, r = None, -1
    for i in sorted(
        range(len(uris)), key=functools.cmp_to_key(lambda a, b: coll(uris[a], uris[b]))
    ):
        if prev is None or coll(uris[prev], uris[i]) != 0:
            r += 1
        rank[i], prev = r, i
    return rank, r


@functools.cache
def _free_clobbers() -> tuple[bool, ...]:
    """Whether freeing a string overwrites its first bytes here, which VLC's duplicate
    check relies on without knowing it: once VLC deleted the ``--sub-file`` itself, it
    goes on comparing with the freed URI, which matches nothing on glibc (its tcache
    and bins keep their links there) and macOS. Elsewhere (Windows' heap) both may
    happen."""
    linux_glibc = sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc"
    return (True,) if sys.platform == "darwin" or linux_glibc else (True, False)


@dataclass(frozen=True)
class _Head:
    """The ``--sub-file`` in VLC's sorts: a known file (``path``, ``uri``), an unknown
    one that is sidecar number ``same``, or (neither) an unknown other file."""

    path: str = ""
    uri: str = ""
    same: int = -1

    @property
    def unseen(self) -> bool:  # a file SubSync doesn't know
        return not self.path and self.same < 0


_HEAD, _TAIL = object(), object()  # the --sub-file, another file given at startup


def _order(
    entries: Sequence[Entry],
    rank: dict[int, float],
    sort: Callable[[list, Callable[[object, object], int]], list],
    head: _Head | None,
    head_rank: float,
    tails: int,
    clobber: bool,
) -> tuple[str | None, ...]:
    """VLC's list of the files it loads as the video opens (``LoadSlaves``): the paths
    of ``entries`` (``rank``: index -> strcoll rank of its URI) and of the ``--sub-file``
    ``head`` (strcoll rank ``head_rank``), None for one SubSync doesn't know (an
    unknown ``--sub-file``, the ``tails`` other files given at startup). ``clobber``:
    see :func:`_free_clobbers`."""
    items: list = ([_HEAD] if head is not None else []) + [
        None if e.dropped else i for i, e in enumerate(entries)
    ]

    def uri_rank(x) -> float:
        return head_rank if x is _HEAD else rank[x]

    def by_uri(a, b) -> int:  # slave_strcmp
        if a is None or b is None:
            return 0
        ra, rb = uri_rank(a), uri_rank(b)
        return (ra > rb) - (ra < rb)

    items = sort(items, by_uri)
    if head is not None:  # every item with the --sub-file's URI from the second one on
        matching = True
        for i in range(1, len(items)):
            x = items[i]
            if x is None or not matching:
                continue
            if x is _HEAD or x == head.same or (head.uri and entries[x].uri == head.uri):
                items[i] = None
                if x is _HEAD and clobber:
                    matching = False  # the URI it compares with is gone
    items += [_TAIL] * tails

    def prio(x) -> int:
        return PRIORITY_USER if x is _HEAD or x is _TAIL else entries[x].priority

    def by_priority(a, b) -> int:  # SlaveCompare
        if a is None or b is None:
            return 1 if a is None else -1 if b is None else 0
        pa, pb = prio(a), prio(b)
        return -1 if pa > pb else 1 if pa < pb else 0

    out: list[str | None] = []
    for x in sort(items, by_priority):
        if x is None:
            break  # VLC stops at the first deleted one
        if x is _TAIL or (x is _HEAD and head is not None and head.unseen):
            out.append(None)
            continue
        if x is _HEAD and head is not None:
            path = head.path or entries[head.same].path
            n = file_tracks(path) if head.path else entries[head.same].tracks
        else:
            path, n = entries[x].path, entries[x].tracks
        out += [os.path.normpath(path)] * (1 if n is None else n)  # an .idx: one per language
    return tuple(out)


MAX_ABSENT = 6  # files VLC hasn't open whose presence at startup is tried both ways


@dataclass(frozen=True)
class Orders:
    """The lists VLC may have made of the files it loaded as the video opened."""

    orders: list[tuple[str | None, ...]]  # the likeliest first; None: a file SubSync can't see
    exact: bool  # False: VLC's sort or locale couldn't be redone here, so not sure at all
    why: str = ""  # why there is more than one, or why it isn't exact


class VlcSidecars:
    """The sidecars VLC 3 loads for a video (see the module docstring), a sequence of
    their paths in the likeliest order.

    ``sub_file``: VLC's ``sub-file`` variable for the video ("" for none, None if not
    known): the ``--sub-file`` or the Open dialog's subtitle file."""

    def __init__(
        self,
        media_path: str | os.PathLike[str],
        settings: Autodetect | None = None,
        *,
        windows: bool | None = None,
        sub_file: str | None = "",
    ):
        self.media = os.fspath(media_path)
        self.settings = settings or Autodetect()
        self.windows = sys.platform == "win32" if windows is None else windows
        self._found = _found(self.media, self.settings, self.windows)
        self.entries = _mark_dropped(self._found)
        self.startup = startup_file(sub_file, self.windows) if sub_file else None
        # where SubSync can't tell what VLC made of it: an unknown or relative --sub-file
        self.sub_file_known = sub_file is not None and (not sub_file or self.startup is not None)
        first = self.lists(first_only=True).orders
        self.paths: list[str] = (
            list(dict.fromkeys(p for p in first[0] if p is not None)) if first else []
        )
        live = [os.path.normpath(e.path) for e in self.entries if not e.dropped]
        startup = [os.path.normpath(self.startup[0])] if self.startup else []
        # every file that can be in VLC's list, and every file the scan found
        self.group: list[str] = list(dict.fromkeys([*self.paths, *live, *startup]))
        self.files: list[str] = list(
            dict.fromkeys([*self.group, *(os.path.normpath(e.path) for e in self.entries)])
        )

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        return self.paths[i]

    def lists(
        self,
        tails: Callable[[int], Iterable[int]] = lambda n: (0,),
        *,
        maybe_absent: Sequence[Iterable[str]] = (),
        is_open: Callable[[str], bool] | None = None,
        gone: Iterable[str] = (),
        first_only: bool = False,
    ) -> Orders:
        """Every list VLC may have made of the files it loaded as the video opened, one
        path per track (an ``.idx`` once per language), None for a file SubSync can't
        see.

        ``tails(n)``: how many other files may have been given at startup (files added
        during an earlier play of the same playlist item), when VLC listed ``n`` tracks
        besides them. ``maybe_absent``: the paths of files (one group each) that may
        have appeared after the video opened (VLC hasn't them open); both are tried.
        ``gone``: files that weren't there when the video opened. ``is_open``: whether
        VLC has a file open; the others are left out of the lists (VLC couldn't load
        them)."""
        sort, exact = _sorter()
        why = [] if exact or len(self.entries) < 2 else ["can't sort like VLC's C library"]
        startup_tracks = file_tracks(self.startup[0]) if self.startup else 1
        if startup_tracks is None or any(e.tracks is None for e in self.entries if not e.dropped):
            exact = False
            why.append("VLC may not read a VobSub .idx as SubSync does")
        live_prios = [e.priority for e in self.entries if not e.dropped]
        collate = self.settings.collate
        if not _VLC_USES_STRCOLL:
            names = ["C"]
        elif collate:
            names = [collate]
        else:
            names = list(dict.fromkeys([env_collate(), "C"]))
            # the order depends on the locale unless the priorities alone decide it
            if (
                len(set(live_prios)) < len(live_prios)
                or self.startup is not None
                or (not self.sub_file_known and live_prios)
            ):
                exact = False
                why.append("VLC's locale isn't known")
        groups = [g for g in (frozenset(map(os.path.normpath, g)) for g in maybe_absent) if g]
        gone_set = frozenset(map(os.path.normpath, gone))
        if first_only:
            subsets: list[frozenset[str]] = [frozenset()]
        elif len(groups) <= MAX_ABSENT:
            subsets = [
                frozenset().union(*c)
                for r in range(len(groups) + 1)
                for c in itertools.combinations(groups, r)
            ]
        else:
            subsets = [frozenset(), frozenset().union(*groups)]
            exact = False
            why.append(f"VLC hasn't {len(groups)} of the subtitle files open")
        out: list[tuple[str | None, ...]] = []
        for absent in (a | gone_set for a in subsets):
            entries = (
                self.entries
                if not absent
                else _mark_dropped(
                    [e for e in self._found if os.path.normpath(e.path) not in absent]
                )
            )
            live = [i for i, e in enumerate(entries) if not e.dropped]
            for name in names:
                coll = _strcoll(name)
                if coll is None:
                    exact = False
                    why.append(f"can't compare names like VLC's locale {name!r}")
                    coll = _strcoll("C")
                uris = [entries[i].uri for i in live] + ([self.startup[1]] if self.startup else [])
                ranked, r = _ranks(uris, coll)
                rank: dict[int, float] = {live[j]: ranked[j] for j in range(len(live))}
                heads: list[tuple[_Head | None, float]]
                if self.startup:
                    heads = [(_Head(*self.startup), ranked[len(live)])]
                elif self.sub_file_known:
                    heads = [(None, 0.0)]
                else:  # it may be one of the sidecars, or another file at any rank
                    heads = [(None, 0.0)]
                    heads += [(_Head(same=i), rank[i]) for i in live]
                    heads += [(_Head(), h - 0.5) for h in range(r + 2)]
                for head, head_rank in heads[:1] if first_only else heads:
                    for clobber in _free_clobbers()[:1] if head is None else _free_clobbers():
                        base = _order(entries, rank, sort, head, head_rank, 0, clobber)
                        n = sum(1 for x in base if x is None or is_open is None or is_open(x))
                        for t in tails(n):
                            seq = (
                                base
                                if t == 0
                                else _order(entries, rank, sort, head, head_rank, t, clobber)
                            )
                            if is_open is not None:
                                seq = tuple(x for x in seq if x is None or is_open(x))
                            if seq not in out:
                                out.append(seq)
                        if first_only and out:
                            return Orders(out, exact, "; ".join(dict.fromkeys(why)))
        if len(out) > 1:
            why.append("VLC's order depends on things SubSync can't see")
        return Orders(out, exact, "; ".join(dict.fromkeys(why)))

    def autoloads(self, path: str) -> bool:
        """Whether VLC's scan takes subtitle file ``path``."""
        s = self.settings
        if s.enabled is False or s.fuzzy == 0:
            return False
        parts = _split(self.media, self.windows)
        if parts is None:
            return False
        media_dir, media_name = parts
        name = os.path.basename(path)
        if name.startswith(".") or "." + _ext(name) not in VLC_SUBTITLE_EXTS:
            return False
        paths = default_paths(self.windows) if s.paths is None else s.paths
        dirs = [media_dir, *search_dirs(media_dir, paths)]
        folder = os.path.dirname(path)
        fuzzy = 3 if s.fuzzy is None else s.fuzzy
        return any(
            _same_dir(d, folder) and match_priority(media_name, name, j == 0) >= fuzzy
            for j, d in enumerate(dirs)
            if not (j and d == media_dir)
        )


def _same_dir(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)  # also where case or symlinks differ
    except (OSError, ValueError):
        return same_file_key(a) == same_file_key(b)


def vlc_sidecars(
    media_path: str | os.PathLike[str], settings: Autodetect | None = None
) -> list[str]:
    """The subtitle files VLC 3 loads by itself for ``media_path``, in the order it
    lists them (the likeliest one)."""
    return VlcSidecars(media_path, settings).paths


def may_be_autoloaded(media_path: str | os.PathLike[str], path: str) -> bool:
    """Whether VLC 3 may have loaded subtitle file ``path`` by itself for the video,
    whatever its settings: the file is in the video's folder or a ``Subs/``-style
    subfolder of it and its name contains the video's (as :func:`vlc_trim` gives
    them). For when VLC's settings are not known."""
    media_dir = os.path.dirname(os.fspath(media_path))
    folder = os.path.dirname(path)
    home = same_file_key(media_dir)
    if same_file_key(folder) != home and not (
        os.path.basename(folder).casefold() in ("subtitles", "subs")
        and same_file_key(os.path.dirname(folder)) == home
    ):
        return False
    media_name = os.path.basename(os.fspath(media_path))
    return vlc_trim(_strip_ext(media_name)) in vlc_trim(_strip_ext(os.path.basename(path)))
