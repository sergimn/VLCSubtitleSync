"""Build the SubSync logo assets (SVG with text outlined, PNG, ICO) in assets/.

Source design: the "SubSync Logo" design canvas (icon + lockups). Text is
converted to outlines with Space Grotesk so the SVGs render without web fonts.

    pip install fonttools cairosvg pillow
    python scripts/make_logo.py path/to/SpaceGrotesk[wght].ttf
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cairosvg
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image

ACCENT = "#FF7A1A"
ACCENT_TEXT = "#D45F06"
INK = "#15171C"
PAPER = "#F4F1EA"
MUTED_LIGHT = "#4A4D55"
MUTED_DARK = "#B9BBC2"
TAGLINE = "Subtitles locked to the audio · a VLC plugin"

OUT = Path(__file__).resolve().parent.parent / "assets"


def icon_body(dark: bool = False) -> str:
    """The 100x100 icon artwork (shared by every artboard)."""
    if dark:
        tile = (
            '<rect x="0.5" y="0.5" width="99" height="99" rx="24" fill="#23262E" '
            'stroke="#34373F" stroke-width="1"/>'
        )
    else:
        tile = f'<rect x="0" y="0" width="100" height="100" rx="24" fill="{INK}"/>'
    bars = [
        (18, 29, 10, 1),
        (25, 23, 22, 1),
        (32, 27, 14, 1),
        (39, 19, 30, 1),
        (57, 24, 20, 0.35),
        (64, 20, 28, 0.35),
        (71, 28, 12, 0.35),
        (78, 25, 18, 0.35),
    ]
    parts = [tile]
    for x, y, h, op in bars:
        o = "" if op == 1 else f' opacity="{op}"'
        parts.append(f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{PAPER}"{o}/>')
    parts += [
        f'<rect x="18" y="60" width="64" height="7" rx="3.5" fill="{PAPER}" opacity="0.35"/>',
        f'<rect x="18" y="60" width="32" height="7" rx="3.5" fill="{ACCENT}"/>',
        f'<rect x="28" y="72" width="44" height="7" rx="3.5" fill="{PAPER}" opacity="0.35"/>',
        f'<rect x="49" y="14" width="2" height="70" rx="1" fill="{ACCENT}"/>',
        f'<circle cx="50" cy="14" r="4.5" fill="{ACCENT}"/>',
    ]
    return "".join(parts)


class Outliner:
    def __init__(self, ttf: Path, weight: int):
        font = TTFont(ttf)
        self.font = instantiateVariableFont(font, {"wght": weight})
        self.glyphs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        hhea = self.font["hhea"]
        self.ascent, self.descent = hhea.ascent, hhea.descent

    def run(self, text: str, size: float, x: float, baseline: float, tracking_em: float = 0.0):
        """Return (svg path data, advance width) for text at the given baseline."""
        scale = size / self.upm
        pen = SVGPathPen(self.glyphs)
        cursor = 0.0
        for i, ch in enumerate(text):
            name = self.cmap[ord(ch)]
            tp = TransformPen(pen, (scale, 0, 0, -scale, x + cursor, baseline))
            self.glyphs[name].draw(tp)
            cursor += self.glyphs[name].width * scale
            if i < len(text) - 1:
                cursor += tracking_em * size
        return pen.getCommands(), cursor

    def width(self, text: str, size: float, tracking_em: float = 0.0) -> float:
        return self.run(text, size, 0, 0, tracking_em)[1]


def lockup(bold: Outliner, medium: Outliner, dark: bool) -> str:
    icon = 184
    gap = 44
    pad = 24
    title_size, tag_size = 116, 23
    track_title = -0.04
    # Column: title line (line-height 1) + 6px gap + tagline (normal line height).
    tag_lh = tag_size * (medium.ascent - medium.descent) / medium.upm
    col_h = title_size + 6 + tag_lh
    sub_w = bold.width("Sub", title_size, track_title) + track_title * title_size
    title_w = sub_w + bold.width("Sync", title_size, track_title)
    tag_w = 6 + medium.width(TAGLINE, tag_size, 0.01)
    col_w = max(title_w, tag_w)
    width = pad + icon + gap + col_w + pad
    height = pad + icon + pad
    tx = pad + icon + gap
    col_top = pad + (icon - col_h) / 2

    # Baselines: center glyph ascent/descent box inside each line box like CSS does.
    def baseline(top, line_h, f, size):
        content = size * (f.ascent - f.descent) / f.upm
        return top + (line_h - content) / 2 + size * f.ascent / f.upm

    tb = baseline(col_top, title_size, bold, title_size)
    gb = baseline(col_top + title_size + 6, tag_lh, medium, tag_size)
    sub_d, _ = bold.run("Sub", title_size, tx, tb, track_title)
    sync_d, _ = bold.run("Sync", title_size, tx + sub_w, tb, track_title)
    tag_d, _ = medium.run(TAGLINE, tag_size, tx + 6, gb, 0.01)
    ink, muted, accent = (PAPER, MUTED_DARK, ACCENT) if dark else (INK, MUTED_LIGHT, ACCENT_TEXT)
    s = icon / 100
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" aria-label="SubSync">'
        f'<g transform="translate({pad} {pad}) scale({s})">{icon_body(dark)}</g>'
        f'<path d="{sub_d}" fill="{ink}"/><path d="{sync_d}" fill="{accent}"/>'
        f'<path d="{tag_d}" fill="{muted}"/></svg>\n'
    )


def main() -> None:
    ttf = Path(sys.argv[1])
    OUT.mkdir(exist_ok=True)
    icon_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="512" '
        f'height="512" role="img" aria-label="SubSync icon">{icon_body()}</svg>\n'
    )
    (OUT / "icon.svg").write_text(icon_svg)
    bold, medium = Outliner(ttf, 700), Outliner(ttf, 500)
    for name, dark in (("logo.svg", False), ("logo-dark.svg", True)):
        (OUT / name).write_text(lockup(bold, medium, dark))
    for size in (256, 512):
        cairosvg.svg2png(
            bytestring=icon_svg.encode(),
            write_to=str(OUT / f"icon-{size}.png"),
            output_width=size,
            output_height=size,
        )
    big = Image.open(
        io.BytesIO(
            cairosvg.svg2png(bytestring=icon_svg.encode(), output_width=256, output_height=256)
        )
    )
    big.save(
        OUT / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    cairosvg.svg2png(url=str(OUT / "logo.svg"), write_to=str(OUT / "logo.png"), scale=2)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
