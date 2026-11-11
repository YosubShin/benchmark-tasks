"""The danger tag's printed faces (PNG), in the layout of a Brady 65502 / Master Lock 497A style lockout tag: a black
header with a red "DANGER" oval (ANSI Z535 danger header), "DO NOT OPERATE" in bold black, red-and-white hazard
stripes and write-on lines on a white laminate. 5 px per mm (76 x 146 mm -> 380 x 730 px). The grommet hole is left
blank at the top centre (the brass grommet is drawn as geometry)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PX_PER_MM = 5
W_MM, H_MM = 76, 146
GROMMET_FROM_TOP_MM = 13
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FILE_FRONT = "locktag_tag_front.png"
FILE_BACK = "locktag_tag_back.png"


def _mm(v: float) -> int:
    return int(round(v * PX_PER_MM))


def _header(d: ImageDraw.ImageDraw, w: int) -> None:
    top, bottom = _mm(24), _mm(46)
    d.rectangle([0, top, w, bottom], fill=(15, 15, 15))
    d.rounded_rectangle([_mm(6), top + _mm(3), w - _mm(6), bottom - _mm(3)], radius=_mm(8), fill=(210, 25, 30),
                        outline=(245, 245, 245), width=_mm(0.8))
    d.text((w // 2, (top + bottom) // 2), "DANGER", fill=(250, 250, 250), font=ImageFont.truetype(FONT, _mm(11)),
           anchor="mm")


def _stripes(d: ImageDraw.ImageDraw, x0: int, x1: int, y0: int, y1: int) -> None:
    step = _mm(8)
    for k in range(-20, 40):
        x = x0 + k * step
        d.polygon([(x, y1), (x + step // 2, y1), (x + step // 2 + (y1 - y0), y0), (x + (y1 - y0), y0)],
                  fill=(210, 25, 30))


def write(out_dir: Path) -> None:
    w, h = _mm(W_MM), _mm(H_MM)
    big = ImageFont.truetype(FONT, _mm(10))
    small = ImageFont.truetype(FONT, _mm(4))
    for name, back in ((FILE_FRONT, False), (FILE_BACK, True)):
        img = Image.new("RGB", (w, h), (246, 246, 244))
        d = ImageDraw.Draw(img)
        # Hazard stripes along the lower part (front) / a plain band (back), then the white text field over them.
        _stripes(d, 0, w, _mm(46), _mm(H_MM))
        d.rectangle([_mm(5), _mm(50), w - _mm(5), _mm(H_MM - 5)], fill=(246, 246, 244))
        _header(d, w)
        if not back:
            d.text((w // 2, _mm(62)), "DO NOT", fill=(10, 10, 10), font=big, anchor="mm")
            d.text((w // 2, _mm(76)), "OPERATE", fill=(10, 10, 10), font=big, anchor="mm")
            d.text((_mm(8), _mm(92)), "This lock/tag may only", fill=(20, 20, 20), font=small, anchor="lm")
            d.text((_mm(8), _mm(98)), "be removed by:", fill=(20, 20, 20), font=small, anchor="lm")
            for k, label in enumerate(("Name", "Dept", "Date")):
                y = _mm(110 + 10 * k)
                d.text((_mm(8), y), label, fill=(20, 20, 20), font=small, anchor="lb")
                d.line([_mm(22), y, w - _mm(8), y], fill=(20, 20, 20), width=_mm(0.4))
        else:
            d.text((w // 2, _mm(60)), "LOCKED OUT", fill=(10, 10, 10), font=ImageFont.truetype(FONT, _mm(7)),
                   anchor="mm")
            for k in range(5):
                y = _mm(78 + 12 * k)
                d.line([_mm(8), y, w - _mm(8), y], fill=(20, 20, 20), width=_mm(0.4))
        # Grommet area: blank (the grommet is geometry).
        c = (w // 2, _mm(GROMMET_FROM_TOP_MM))
        r = _mm(10.5)
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(246, 246, 244))
        img.save(out_dir / name)


if __name__ == "__main__":
    import sys

    write(Path(sys.argv[1]))
