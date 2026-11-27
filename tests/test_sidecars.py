"""SubSync's copy of VLC 3's subtitle autodetection (vlcsubsync.sidecars).

The expected orders are what a real VLC loaded (scripts/vlc_order_probe.py): VLC
3.0.20 on Linux, VLC 3.0.24 on macOS and Windows. The order of equal files is the C
library's, so each OS has its own; CI runs these tests on all three.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

from vlcsubsync import sidecars as S
from vlcsubsync.sidecars import Autodetect, VlcSidecars, may_be_autoloaded, vlc_trim

LINUX, MACOS, WINDOWS = (
    sys.platform.startswith("linux"),
    sys.platform == "darwin",
    sys.platform == "win32",
)


def layout(root: Path, *names: str, media: str = "Movie.mkv") -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / media).write_bytes(b"")
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
    return str(root / media)


def names(root: Path, paths) -> list[str]:
    return [os.path.relpath(p, root).replace("\\", "/") for p in paths]


def C() -> Autodetect:
    return Autodetect(True, 3, S.default_paths(WINDOWS), "C")


def test_vlc_trim_is_vlcs_strcpy_trim():
    assert vlc_trim("The.Matrix (1999)") == "the matrix 1999"
    assert vlc_trim("Movie - Spanish") == "movie spanish"
    assert vlc_trim("Am\u00e9lie_EN") == "am lie en"  # VLC compares ASCII letters only
    assert vlc_trim("\u0130stanbul") == "stanbul"  # no ASCII from lower() of others
    assert vlc_trim("...") == ""


@pytest.mark.parametrize(
    "sub, in_media_dir, prio",
    [
        ("The.Matrix.srt", True, 4),
        ("the matrix.ass", False, 4),
        ("THE_MATRIX.en.srt", True, 3),
        ("The Matrix - Spanish.srt", False, 3),
        ("Best of The Matrix.srt", True, 2),
        ("Best of The Matrix.srt", False, 2),
        ("English.srt", True, 1),
        ("English.srt", False, 0),
        ("TheMatrix.srt", True, 1),  # "thematrix" doesn't contain "the matrix"
    ],
)
def test_match_priority(sub, in_media_dir, prio):
    assert S.match_priority("The.Matrix.mkv", sub, in_media_dir) == prio


def test_search_dirs_is_vlcs_paths_to_list():
    assert S.search_dirs("/m/", "./Subtitles, ./subtitles, ./Subs, ./subs") == [
        "/m/./Subtitles",
        "/m/./subtitles",
        "/m/./Subs",
        "/m/./subs",
    ]
    # spaces after a comma are skipped, others kept; empty entries skipped
    assert S.search_dirs("/m/", " ./a,,  /abs/subs ,./b") == [" ./a", "/abs/subs ", "/m/./b"]
    assert S.search_dirs("/m/", "") == []
    assert S.search_dirs("C:\\m\\", ".\\subtitles, .\\subs") == [
        "C:\\m\\.\\subtitles",
        "C:\\m\\.\\subs",
    ]


def test_path_to_uri_is_vlcs_path2uri():
    assert (
        S.path_to_uri("/m/./Subs/The Matrix (2) é~_-.srt")
        == "file:///m/./Subs/The%20Matrix%20%282%29%20%C3%A9~_-.srt"
    )
    assert S.path_to_uri("C:\\m\\.\\subs\\a b.srt", windows=True) == "file:///C:/m/./subs/a%20b.srt"
    assert S.path_to_uri("\\\\host\\share\\a.srt", windows=True) == "file://host/share/a.srt"


def test_scan_takes_what_vlc_takes(tmp_path):
    media = layout(
        tmp_path,
        "Movie.srt",
        "Movie.en.srt",
        "Best of Movie.srt",  # priority 2
        "English.srt",  # 1
        "Subs/English.srt",  # 0
        "Subs/Movie.de.srt",
        "Sub/Movie.fr.srt",  # not a folder VLC looks in
        ".Movie.srt",  # hidden
        "Movie.txt",
        "Movie.idx",
        "Movie.sub",  # the .idx's other half
        "Movie.cdg",
        "Movie.x.cdg",  # .cdg needs the exact name
    )
    (tmp_path / "Movie.dir.srt").mkdir()
    got = {
        names(tmp_path, [e.path])[0]: (e.priority, e.dropped) for e in S.scan(media, C(), WINDOWS)
    }
    # VLC reads each folder as its setting spells it: on macOS ./Subs and ./subs both
    subs = ["subs"] if WINDOWS else ["Subs", "subs"] if MACOS else ["Subs"]
    assert got == {
        "Movie.srt": (4, False),
        "Movie.en.srt": (3, False),
        "Movie.idx": (4, False),
        "Movie.sub": (4, True),
        "Movie.cdg": (4, False),
        "Movie.x.cdg": (3, True),
        **{f"{d}/Movie.de.srt": (3, False) for d in subs},
    }
    # lower sub-autodetect-fuzzy settings take more; 0 or autodetection off: none
    two = {Path(e.path).name for e in S.scan(media, Autodetect(True, 2, None, "C"), WINDOWS)}
    assert "Best of Movie.srt" in two and "English.srt" not in two
    one = {os.path.normpath(e.path) for e in S.scan(media, Autodetect(True, 1, None, "C"), WINDOWS)}
    assert str(tmp_path / "English.srt") in one
    assert str(tmp_path / "Subs" / "English.srt") not in one
    assert S.scan(media, Autodetect(True, 0, None, "C"), WINDOWS) == []
    assert S.scan(media, Autodetect(False, 3, None, "C"), WINDOWS) == []
    # no subfolders when the setting is empty
    assert {Path(e.path).name for e in S.scan(media, Autodetect(True, 3, "", "C"), WINDOWS)} == {
        "Movie.srt",
        "Movie.en.srt",
        "Movie.idx",
        "Movie.sub",
        "Movie.cdg",
        "Movie.x.cdg",
    }


# what VLC loaded, per OS (scripts/vlc_order_probe.py layouts)
TWO = ["Movie.en.srt", "Movie.es.srt"]
THREE = ["Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"]
CASE = ["movie.EN.srt", "Movie.de.srt", "Movie_fr.srt", "Movie-es.srt", "MOVIE.it.srt"]
NINE = [f"Movie.{c}.srt" for c in "qhbxmdkat"]
VOBSUB = ["Movie.idx", "Movie.sub", "Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"]
EXPECTED = {
    "linux": {
        "two": ["Movie.en.srt", "Movie.es.srt"],
        "three": ["Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"],
        "case": ["MOVIE.it.srt", "Movie-es.srt", "Movie.de.srt", "Movie_fr.srt", "movie.EN.srt"],
        "nine": sorted(NINE),
        "vobsub": ["Movie.idx", "Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"],
    },
    "win32": {
        # Windows' qsort doesn't keep equal items in order: two equal ones swap
        "two": ["Movie.es.srt", "Movie.en.srt"],
        "three": ["Movie.es.srt", "Movie.fr.srt", "Movie.en.srt"],
        "case": ["Movie-es.srt", "Movie.de.srt", "Movie_fr.srt", "movie.EN.srt", "MOVIE.it.srt"],
        "nine": sorted(NINE),
        "vobsub": ["Movie.idx", "Movie.es.srt", "Movie.fr.srt", "Movie.en.srt"],
    },
    "darwin": {
        "two": ["Movie.en.srt", "Movie.es.srt"],
        "three": ["Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"],
        "case": ["MOVIE.it.srt", "Movie-es.srt", "Movie.de.srt", "Movie_fr.srt", "movie.EN.srt"],
        "nine": [f"Movie.{c}.srt" for c in "xbdhkmqta"],
        "vobsub": ["Movie.idx", "Movie.es.srt", "Movie.fr.srt", "Movie.en.srt"],
    },
}
LAYOUTS = {"two": TWO, "three": THREE, "case": CASE, "nine": NINE, "vobsub": VOBSUB}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_order_is_what_vlc_loads_on_this_os(tmp_path, name, listed_in):
    plat = "linux" if LINUX else sys.platform
    if plat not in EXPECTED:
        pytest.skip(f"no VLC order known for {sys.platform}")
    if LINUX:
        listed_in(LAYOUTS[name])  # the order the files were made in
    media = layout(tmp_path, *LAYOUTS[name])
    det = VlcSidecars(media, C())
    assert names(tmp_path, det.paths) == EXPECTED[plat][name]
    got = det.lists()
    assert got.exact and len(got.orders) == 1 and not got.why


MATRIX = [
    "The.Matrix.srt",
    "The.Matrix.smi",
    "The.Matrix.en.srt",
    "the_matrix_de.ass",
    "The.Matrix-ZZ.ssa",
    "The Matrix - Spanish.srt",
    "The.Matrix (2).srt",
    "Best of The Matrix.srt",
    "English.srt",
    ".hidden.srt",
    "Subs/The Matrix.srt",
    "Subs/The.Matrix.fr.srt",
    "Subs/3_English.srt",
    "subtitles/The.Matrix.it.srt",
    "Sub/The.Matrix.pt.srt",
]
MATRIX_HEAD = [
    "Subs/The Matrix.srt",
    "The.Matrix.smi",
    "The.Matrix.srt",
    "Subs/The.Matrix.fr.srt",
    "subtitles/The.Matrix.it.srt",
    "The Matrix - Spanish.srt",
    "The.Matrix (2).srt",
]


@pytest.mark.skipif(not LINUX, reason="VLC sorts with the locale on Linux only")
@pytest.mark.parametrize(
    "locale, tail",
    [
        ("C", ["The.Matrix-ZZ.ssa", "The.Matrix.en.srt", "the_matrix_de.ass"]),
        ("en_US.UTF-8", ["the_matrix_de.ass", "The.Matrix.en.srt", "The.Matrix-ZZ.ssa"]),
        ("es_ES.UTF-8", ["the_matrix_de.ass", "The.Matrix.en.srt", "The.Matrix-ZZ.ssa"]),
    ],
)
def test_linux_order_follows_vlcs_locale(tmp_path, locale, tail):
    if S._strcoll(locale) is None:
        pytest.skip(f"no {locale} locale here")
    media = layout(tmp_path, *MATRIX, media="The.Matrix.mp4")
    det = VlcSidecars(media, Autodetect(True, 3, S.POSIX_PATHS, locale))
    assert names(tmp_path, det.paths) == MATRIX_HEAD + tail


@pytest.mark.skipif(not LINUX, reason="VLC sorts with the locale on Linux only")
def test_unknown_locale_gives_every_candidate(tmp_path, monkeypatch):
    if S._strcoll("en_US.UTF-8") is None:
        pytest.skip("no en_US.UTF-8 locale here")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    media = layout(tmp_path, *MATRIX, media="The.Matrix.mp4")
    got = VlcSidecars(media, Autodetect()).lists()  # an older intf: no settings
    assert [names(tmp_path, o)[-3:] for o in got.orders] == [
        ["the_matrix_de.ass", "The.Matrix.en.srt", "The.Matrix-ZZ.ssa"],  # this process's
        ["The.Matrix-ZZ.ssa", "The.Matrix.en.srt", "the_matrix_de.ass"],  # C
    ]
    # ... and VLC's may be another one
    assert not got.exact and "locale isn't known" in got.why
    got = VlcSidecars(media, Autodetect(True, 3, None, "xx_NOPE.UTF-8")).lists()
    assert not got.exact and "xx_NOPE" in got.why


@pytest.mark.skipif(not LINUX, reason="VLC sorts with the locale on Linux only")
def test_unknown_locale_is_no_matter_where_the_priorities_decide(tmp_path):
    media = layout(tmp_path, "Movie.srt", "Movie.EN.srt", "Best of Movie.srt")
    got = VlcSidecars(media, Autodetect(True, 2, None, None)).lists()
    assert got.exact and len(got.orders) == 1
    two = layout(tmp_path / "two", "Movie.EN.srt", "Movie.de.srt")
    assert not VlcSidecars(two, Autodetect(True, 3, None, None)).lists().exact


def test_files_given_at_startup_can_change_the_order(tmp_path):
    # VLC lists four files: the three sidecars and one SubSync doesn't know about (an
    # unknown --sub-file, or a file added during an earlier play)
    media = layout(tmp_path, *THREE)
    det = VlcSidecars(media, C(), sub_file=None)
    got = det.lists(lambda n: (4 - n,) if n <= 4 else ())
    assert got.exact
    sidecars = names(tmp_path, det.paths)
    clobbers = S._free_clobbers() == (True,)
    for order in got.orders:
        files = names(tmp_path, [p for p in order if p])
        assert len(order) == 4 and len(set(files)) == len(files) and set(files) <= set(sidecars)
        if clobbers:  # the same files, maybe in another order
            assert sorted(files) == sorted(sidecars)
    if not clobbers:  # Windows: if the unknown file is a sidecar and VLC deleted it first,
        # the freed URI may still match the sidecar's copy, which goes too
        assert any(len([p for p in o if p]) < len(sidecars) for o in got.orders)
    if WINDOWS:  # where the --sub-file sorts among them changes Windows' qsort order
        assert len(got.orders) > 1 and got.why
    if LINUX and clobbers:  # glibc's merge sort keeps equal ones in order
        assert {tuple(names(tmp_path, [p for p in o if p])) for o in got.orders} == {
            tuple(names(tmp_path, det.paths))
        }


def test_a_dropped_sub_makes_the_order_depend_on_the_folder_listing(tmp_path, listed_in):
    # what the ext4 CI runners showed: the same files, listed in another order
    media = layout(tmp_path, *VOBSUB)
    want = {
        ("Movie.idx", "Movie.sub", "Movie.en.srt", "Movie.es.srt", "Movie.fr.srt"): [
            "Movie.idx",
            "Movie.en.srt",
            "Movie.es.srt",
            "Movie.fr.srt",
        ],
        ("Movie.idx", "Movie.fr.srt", "Movie.sub", "Movie.en.srt", "Movie.es.srt"): [
            "Movie.idx",
            "Movie.fr.srt",
            "Movie.en.srt",
            "Movie.es.srt",
        ],
    }
    if not LINUX:
        pytest.skip("glibc's merge sort only")
    for order, expected in want.items():
        listed_in(order)
        assert names(tmp_path, VlcSidecars(media, C()).paths) == expected


def test_c_qsort_is_this_c_librarys(monkeypatch):
    # the C library's sort through ctypes, and glibc's merge sort redone in Python
    sort, exact = S._sorter()
    assert exact
    rng = random.Random(7)
    for _ in range(200):
        items = [rng.choice([None, 1, 2, 3]) for _ in range(rng.randrange(0, 15))]

        def cmp(a, b):  # VLC's slave_strcmp: a dropped item is equal to every other
            if a is None or b is None:
                return 0
            return (a > b) - (a < b)

        got = sort(items, cmp)
        assert sorted(x for x in got if x is not None) == sorted(x for x in items if x is not None)
        if LINUX:
            assert got == S._msort(items, cmp)


def test_without_the_c_library_sort_the_order_is_not_sure(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_c_qsort", lambda: None)
    media = layout(tmp_path, *TWO)
    got = VlcSidecars(media, C()).lists()
    assert got.exact == LINUX
    assert names(tmp_path, got.orders[0]) == TWO


def test_autoloads_is_vlcs_rule(tmp_path):
    media = layout(tmp_path, "Movie.srt")
    det = VlcSidecars(media, C())
    for name in ("Movie.en.srt", "Subs/Movie.de.srt", "subtitles/MOVIE.ass"):
        assert det.autoloads(str(tmp_path / name)), name
    for name in ("Best of Movie.srt", "English.srt", "Sub/Movie.srt", ".Movie.srt", "Movie.txt"):
        assert not det.autoloads(str(tmp_path / name)), name
    low = VlcSidecars(media, Autodetect(True, 1, None, "C"))
    assert low.autoloads(str(tmp_path / "English.srt"))
    assert not low.autoloads(str(tmp_path / "Subs" / "English.srt"))
    assert not VlcSidecars(media, Autodetect(False, 3, None, "C")).autoloads(
        str(tmp_path / "Movie.en.srt")
    )


def test_may_be_autoloaded_whatever_the_settings(tmp_path):
    media = str(tmp_path / "The.Matrix.mkv")
    yes = [
        "The Matrix - Spanish.srt",
        "the_matrix_en.smi",
        "Subs/The.Matrix.fr.srt",
        "subtitles/THE MATRIX.ass",
        "Best of The Matrix.srt",  # VLC loads it with a lower sub-autodetect-fuzzy
    ]
    no = ["Subs/2_English.srt", "English.srt", "Sub/The.Matrix.srt", "Extras/The.Matrix.srt"]
    for name in yes:
        assert may_be_autoloaded(media, str(tmp_path / name)), name
    for name in no:
        assert not may_be_autoloaded(media, str(tmp_path / name)), name
    assert not may_be_autoloaded(media, str(tmp_path.parent / "The.Matrix.srt"))
    # a name without ASCII letters: VLC matches every subtitle file
    assert may_be_autoloaded(str(tmp_path / "\u65e5\u672c.mkv"), str(tmp_path / "x.srt"))
