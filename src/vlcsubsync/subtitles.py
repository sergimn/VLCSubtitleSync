"""Subtitle I/O (pysubs2), sidecar discovery, tokenisation and language guessing."""

from __future__ import annotations

import codecs
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pysubs2

SIDECAR_EXTS = (".srt", ".ass", ".ssa", ".vtt", ".sub")
SIDECAR_SUBDIRS = ("subs", "subtitles", "sub", "subtitle")

# Formats we write back as-is; everything else becomes SRT.
_KEEP_FORMATS = {"ass": ".ass", "ssa": ".ssa", "vtt": ".vtt"}


class SubtitleError(Exception):
    """A subtitle file could not be read."""


class UnsupportedSubtitle(SubtitleError):
    """Subtitle exists but cannot be synced (image-based, binary, ...)."""


# --------------------------------------------------------------------------------------
# Loading / saving
# --------------------------------------------------------------------------------------


def decode_text(data: bytes) -> tuple[str, str]:
    """Decode subtitle bytes. Returns ``(text, encoding)``.

    Chain: BOM sniffing (utf-8-sig/utf-16/utf-32) → BOM-less UTF-16 heuristic (many
    NULs) → strict UTF-8 → cp1252 → latin-1 (never fails).
    """
    boms = (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    )
    for bom, enc in boms:
        if data.startswith(bom):
            try:
                return data.decode(enc), enc
            except UnicodeDecodeError:
                break
    sample = data[:4000]
    if sample and sample.count(b"\x00") > len(sample) // 4:
        even_nuls = sample[0::2].count(b"\x00")
        odd_nuls = sample[1::2].count(b"\x00")
        enc = "utf-16-le" if odd_nuls > even_nuls else "utf-16-be"
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8", "cp1252"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def load_subtitles(path: str | os.PathLike[str]) -> tuple[pysubs2.SSAFile, str]:
    """Load a subtitle file with encoding fallbacks. Returns ``(subs, format)``."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise SubtitleError(f"cannot read subtitle file {p}: {e}") from e
    if p.suffix.lower() == ".sub" and (
        data[:4] == b"\x00\x00\x01\xba" or p.with_suffix(".idx").exists()
    ):
        raise UnsupportedSubtitle(f"{p.name} is a VobSub (image) subtitle; cannot sync it")
    text, _enc = decode_text(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    fmt_hint = pysubs2.formats.FILE_EXTENSION_TO_FORMAT_IDENTIFIER.get(p.suffix.lower())
    attempts: list[dict] = [{}]
    if fmt_hint and fmt_hint not in ("json", "tmp"):
        attempts.insert(0, {"format_": fmt_hint})
    last_err: Exception | None = None
    for kwargs in attempts:
        for extra in ({}, {"fps": 23.976}):
            try:
                subs = pysubs2.SSAFile.from_string(text, **kwargs, **extra)
            except Exception as e:  # pysubs2 raises various errors on bad input
                last_err = e
                continue
            if len(subs) == 0 and kwargs:
                last_err = SubtitleError("no events parsed")
                continue
            return subs, (subs.format or kwargs.get("format_") or "srt")
    raise SubtitleError(f"cannot parse subtitle file {p.name}: {last_err}")


def output_format(source_format: str | None) -> tuple[str, str]:
    """``(pysubs2 format, extension)`` to write a re-timed copy of ``source_format``."""
    fmt = (source_format or "").lower()
    if fmt in _KEEP_FORMATS:
        return fmt, _KEEP_FORMATS[fmt]
    return "srt", ".srt"


def save_subtitles(
    subs: pysubs2.SSAFile, path: str | os.PathLike[str], source_format: str | None = None
) -> str:
    """Write ``subs`` atomically. The extension of ``path`` is adjusted to the output
    format (source format for ASS/SSA/VTT, SRT otherwise). Returns the path written."""
    fmt, ext = output_format(source_format)
    p = Path(path)
    if p.suffix.lower() != ext:
        p = p.with_suffix(ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = subs.to_string(fmt)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return str(p)


# --------------------------------------------------------------------------------------
# Sidecar discovery
# --------------------------------------------------------------------------------------


def find_sidecars(media_path: str | os.PathLike[str]) -> list[str]:
    """External subtitle files VLC would auto-load for ``media_path``.

    Same stem or ``stem.*.ext`` in the media dir and its ``Subs/``/``Subtitles/``
    subdirs (case-insensitive). Sorted like VLC: exact-stem match first, then
    alphabetical.
    """
    media = Path(media_path)
    folder = media.parent
    stem = media.stem.casefold()
    dirs: list[tuple[Path, int]] = [(folder, 0)]
    try:
        for child in folder.iterdir():
            if child.is_dir() and child.name.casefold() in SIDECAR_SUBDIRS:
                dirs.append((child, 1))
    except OSError:
        return []
    found: list[tuple[int, int, str, str]] = []
    for d, depth in dirs:
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for f in entries:
            if f.suffix.lower() not in SIDECAR_EXTS or not f.is_file():
                continue
            fstem = f.stem.casefold()
            if fstem == stem:
                exact = 0
            elif fstem.startswith(stem + "."):
                exact = 1
            else:
                continue
            found.append((exact, depth, f.name.casefold(), str(f)))
    found.sort()
    return [x[3] for x in found]


# --------------------------------------------------------------------------------------
# Text normalisation / tokenisation
# --------------------------------------------------------------------------------------

_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")
_ASS_DRAWING = re.compile(r"\\p[1-9]")
_HTML_TAG = re.compile(r"<[^>]{0,200}>")
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪[^♪]*♪")
_SPEAKER = re.compile(r"(^|\s|-)[A-Z][A-Z0-9 .'-]{1,30}:\s")
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}  # fmt: skip


def is_drawing(text: str) -> bool:
    """True for ASS vector drawings (``{\\p1}...``), which carry no dialogue."""
    for tag in _ASS_OVERRIDE.findall(text):
        if _ASS_DRAWING.search(tag):
            return True
    return False


def plain_text(text: str) -> str:
    """Strip ASS override blocks / HTML tags and line breaks (for matching only)."""
    t = _ASS_OVERRIDE.sub("", text)
    t = t.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    t = _HTML_TAG.sub("", t)
    return " ".join(t.split())


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_word(w: str) -> str:
    w = strip_accents(w.casefold()).replace("'", "").replace("’", "")
    w = "".join(ch for ch in w if ch.isalnum())
    return _NUMBER_WORDS.get(w, w)


def tokenize(text: str, *, drop_annotations: bool = True) -> list[str]:
    """Normalised word tokens of a (plain) text.

    With ``drop_annotations`` hearing-impaired annotations (``[door slams]``,
    ``(laughs)``, ``♪ lyrics ♪``) and ``SPEAKER:`` labels are removed first.
    """
    if drop_annotations:
        text = _BRACKETED.sub(" ", text)
        text = _SPEAKER.sub(" ", text)
    out = []
    for m in _TOKEN.finditer(text):
        tok = normalize_word(m.group(0))
        if tok:
            out.append(tok)
    return out


def dialogue_events(subs: pysubs2.SSAFile) -> list[tuple[int, pysubs2.SSAEvent, str]]:
    """``(index, event, plain_text)`` for events that carry dialogue text."""
    out = []
    for i, ev in enumerate(subs):
        if ev.is_comment or not ev.text or is_drawing(ev.text):
            continue
        txt = plain_text(ev.text)
        if txt:
            out.append((i, ev, txt))
    return out


# --------------------------------------------------------------------------------------
# Language guess
# --------------------------------------------------------------------------------------

_STOPWORDS: dict[str, set[str]] = {
    "en": set(
        "the and you to of is it that what this was have are with he she we they dont im "
        "your just know be for not my me do can there here youre its thats whats would "
        "will about all get got if him her them were been how why who when"
        .split()
    ),
    "es": set(
        "el la los las que de y es en un una no por para con lo se me te mi pero esta "
        "como mas eso esto aqui bien si yo tu usted muy hay todo nada porque donde cuando "
        "estoy tengo puedo vamos ella del al ya senor gracias quiero sabes tambien hacer"
        .split()
    ),
    "pt": set(
        "o a os as que de e em um uma nao voce com para por isso isto aqui mas meu minha "
        "ele ela eu tem esta sim muito bem entao fazer vai ja tudo obrigado do da dos das "
        "no na foi estou vamos agora sei quero"
        .split()
    ),
    "fr": set(
        "le la les de des et est un une je tu il elle nous vous pas ne que qui ce en du au "
        "avec pour sur mais oui non moi toi suis bien tout sais fait ca rien cest jai quil "
        "nest cetait quoi sont etait ici vais peux veux faire"
        .split()
    ),
    "de": set(
        "der die das und ist ich du nicht ein eine es sie wir ihr zu mit den dem auf fur "
        "ja nein was wie mich mir dich dir aber auch hier noch sind bin hast habe kann "
        "schon gut doch mal bitte haben wird wenn sich nur"
        .split()
    ),
    "it": set(
        "il lo la gli le che di e un una non per con mi ti ci si sono ma cosa questo "
        "questa qui bene io tu lui lei noi voi come perche anche ho hai ha sei del della "
        "nel alla cosi molto grazie fatto adesso"
        .split()
    ),
    "nl": set(
        "de het een en van ik je jij is niet dat die wat we wij zijn op te met voor maar "
        "ook er hij zij ze hebben heb kan nog wel geen naar hier mijn jullie waar goed dit "
        "weet moet gaan heeft"
        .split()
    ),
}  # fmt: skip

_WORD_LANGS: dict[str, list[str]] = {}
for _lang, _words in _STOPWORDS.items():
    for _w in _words:
        _WORD_LANGS.setdefault(_w, []).append(_lang)


def guess_language(texts: Iterable[str] | str, *, max_tokens: int = 20000) -> str | None:
    """Stopword-based guess (ISO 639-1) among en, es, fr, de, it, pt, nl, or None."""
    if isinstance(texts, str):
        texts = [texts]
    tokens: list[str] = []
    for t in texts:
        # keep contractions as separate words for French/English: c'est -> c est / cest
        tokens.extend(tokenize(t))
        if len(tokens) >= max_tokens:
            break
    if not tokens:
        return None
    scores: Counter[str] = Counter()
    for tok in tokens:
        langs = _WORD_LANGS.get(tok)
        if langs:
            share = 1.0 / len(langs)
            for lang in langs:
                scores[lang] += share
    if not scores:
        return None
    ranked = scores.most_common(2)
    best, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 3 or best_score / len(tokens) < 0.08 or best_score < 1.25 * second:
        return None
    return best
