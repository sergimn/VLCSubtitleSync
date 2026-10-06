import codecs
import os

import pysubs2
import pytest

from vlcsubsync.subtitles import (
    UnsupportedSubtitle,
    decode_text,
    dialogue_events,
    find_sidecars,
    guess_language,
    load_subtitles,
    output_format,
    plain_text,
    save_subtitles,
    tokenize,
)

SRT = (
    "1\n00:00:01,000 --> 00:00:02,500\nCafé déjà vu, señor!\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nBye\n"
)

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Sign,Comic Sans MS,30,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,2,8,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,Bob,0,0,0,,{\\i1}Hello{\\i0} there\\Nfriend
Dialogue: 1,0:00:03.00,0:00:04.00,Sign,,0,0,0,,{\\pos(320,50)}EXIT
Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\\p1}m 0 0 l 100 0 100 100{\\p0}
Comment: 0,0:00:07.00,0:00:08.00,Default,,0,0,0,,a comment
"""  # noqa: E501


@pytest.mark.parametrize(
    "encoding,prefix",
    [
        ("utf-8", b""),
        ("utf-8", codecs.BOM_UTF8),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-16-be", codecs.BOM_UTF16_BE),
        ("utf-16-le", b""),  # BOM-less UTF-16
        ("cp1252", b""),
        ("latin-1", b""),
    ],
)
def test_encoding_fallbacks(tmp_path, encoding, prefix):
    p = tmp_path / "x.srt"
    p.write_bytes(prefix + SRT.encode(encoding))
    subs, fmt = load_subtitles(p)
    assert fmt == "srt"
    assert len(subs) == 2
    assert subs[0].plaintext == "Café déjà vu, señor!"


def test_decode_text_cp1252_specials():
    text, enc = decode_text("“quoted” – dash €".encode("cp1252"))
    assert enc == "cp1252"
    assert text == "“quoted” – dash €"


def test_latin1_last_resort():
    # 0x81 is undefined in cp1252, so only latin-1 can decode it.
    text, enc = decode_text(b"abc\x81")
    assert enc == "latin-1"
    assert text.startswith("abc")


def test_crlf_and_vtt(tmp_path):
    p = tmp_path / "x.vtt"
    p.write_text("WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.000\r\nHi\r\n", encoding="utf-8")
    subs, fmt = load_subtitles(p)
    assert fmt == "vtt"
    assert output_format(fmt) == ("vtt", ".vtt")
    out = save_subtitles(subs, tmp_path / "out.srt", fmt)
    assert out.endswith(".vtt")
    assert pysubs2.load(out)[0].start == 1000


def test_microdvd_becomes_srt(tmp_path):
    p = tmp_path / "x.sub"
    p.write_text("{25}{50}Hello there\n{75}{100}Second|line\n", encoding="utf-8")
    subs, fmt = load_subtitles(p)
    assert len(subs) == 2
    assert output_format(fmt) == ("srt", ".srt")
    out = save_subtitles(subs, tmp_path / "out.ass", fmt)
    assert out.endswith(".srt")


def test_vobsub_rejected(tmp_path):
    p = tmp_path / "movie.sub"
    p.write_bytes(b"\x00\x00\x01\xba" + b"\x00" * 100)
    with pytest.raises(UnsupportedSubtitle):
        load_subtitles(p)


def test_ass_roundtrip_keeps_styles(tmp_path):
    p = tmp_path / "x.ass"
    p.write_text(ASS, encoding="utf-8")
    subs, fmt = load_subtitles(p)
    assert fmt == "ass"
    for ev in subs:
        ev.start += 1000
        ev.end += 1000
    out = save_subtitles(subs, tmp_path / "o.srt", fmt)
    assert out.endswith(".ass")
    again = pysubs2.load(out)
    assert set(again.styles) == {"Default", "Sign"}
    assert again.styles["Sign"].fontname == "Comic Sans MS"
    assert again[0].text == "{\\i1}Hello{\\i0} there\\Nfriend"
    assert again[0].name == "Bob"
    assert again[1].style == "Sign" and again[1].layer == 1
    assert again[0].start == 2000


def test_dialogue_events_and_plain_text(tmp_path):
    subs = pysubs2.SSAFile.from_string(ASS)
    ev = dialogue_events(subs)
    # drawing and comment skipped
    assert [i for i, _e, _t in ev] == [0, 1]
    assert ev[0][2] == "Hello there friend"
    assert plain_text("<i>a</i>\\Nb{\\b1}c\\hd") == "a bc d"


def test_tokenize_normalisation():
    assert tokenize("Don't STOP! Café, 20 dollars; twenty-one.") == [
        "dont", "stop", "cafe", "20", "dollars", "20", "1",
    ]  # fmt: skip
    assert tokenize("[door slams] JOHN: Hello (laughs) ♪ la la ♪ there") == ["hello", "there"]


@pytest.mark.parametrize(
    "lang,text",
    [
        ("en", "I don't know what you are talking about, but we have to get out of here now."),
        ("es", "No sé de qué estás hablando, pero tenemos que salir de aquí ahora mismo."),
        ("fr", "Je ne sais pas de quoi tu parles, mais nous devons partir d'ici tout de suite."),
        ("de", "Ich weiß nicht, wovon du sprichst, aber wir müssen sofort hier raus, ja."),
        ("it", "Non so di cosa stai parlando, ma dobbiamo andarcene da qui subito, per favore."),
        ("pt", "Eu não sei do que você está falando, mas nós temos que sair daqui agora."),
        ("nl", "Ik weet niet waar je het over hebt, maar we moeten hier nu weg, dat is zeker."),
    ],
)
def test_guess_language(lang, text):
    assert guess_language([text] * 5) == lang


def test_guess_language_unknown():
    assert guess_language(["Привет, как дела?"] * 10) is None
    assert guess_language([]) is None
    assert guess_language("12345 67890") is None


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
    return p


def test_find_sidecars_ordering(tmp_path):
    media = tmp_path / "Movie.Name.2020.mkv"
    media.write_bytes(b"")
    exact_srt = _touch(tmp_path / "Movie.Name.2020.srt")
    exact_ass = _touch(tmp_path / "Movie.Name.2020.ass")
    en = _touch(tmp_path / "Movie.Name.2020.en.srt")
    es = _touch(tmp_path / "movie.name.2020.ES.forced.srt")  # case-insensitive
    sub_dir = _touch(tmp_path / "Subs" / "Movie.Name.2020.fr.srt")
    sub_dir2 = _touch(tmp_path / "subtitles" / "Movie.Name.2020.vtt")
    _touch(tmp_path / "Other.Movie.srt")
    _touch(tmp_path / "Movie.Name.2020.txt")
    _touch(tmp_path / "Movie.Name.2020-sample.srt")
    got = find_sidecars(str(media))
    assert got == [
        str(exact_ass),
        str(exact_srt),
        str(sub_dir2),
        str(en),
        str(es),
        str(sub_dir),
    ]
    assert all(os.path.isabs(p) for p in got)


def test_find_sidecars_missing_dir(tmp_path):
    assert find_sidecars(str(tmp_path / "nope" / "x.mkv")) == []
