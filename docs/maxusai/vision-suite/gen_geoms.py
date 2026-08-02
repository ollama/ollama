#!/usr/bin/env python3
"""Render the eight token-budget geometries measure.py expects into testimgs/.

measure.py reads testimgs/<W>x<H>.png for each size in its ladder; nothing else in the
suite produces them (gen_scenes.py writes the three ground-truth scenes into visimgs/).
Without this the token-budget protocol is not reproducible from a clean checkout.

The images only need to be valid PNGs of an exact pixel size — the protocol measures
prompt_eval_count, which depends on geometry alone — but a flat colour field can be
compressed to a trivially small payload, so each frame gets deterministic per-pixel noise
plus gridlines. That keeps the encoded bytes realistic and makes a mis-scaled preprocessor
visible in the coherence sample at the end of measure.py.

Usage: gen_geoms.py [outdir]   (default: testimgs/ beside this script)
Needs Pillow, same as gen_scenes.py.
"""
import os
import sys

from PIL import Image, ImageDraw

SIZES = ["320x240", "640x480", "896x896", "1568x1568", "1920x1080",
         "2048x1664", "3000x2000", "3200x32"]

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "testimgs")


def render(w, h):
    img = Image.new("RGB", (w, h))
    px = img.load()
    # Deterministic pseudo-noise: no RNG, so regenerating is byte-stable.
    for y in range(h):
        for x in range(w):
            px[x, y] = (((x * 7 + y * 13) % 256),
                        ((x * 3 + y * 29) % 256),
                        ((x * 17 + y * 5) % 256))
    d = ImageDraw.Draw(img)
    for gx in range(0, w, 64):
        d.line([(gx, 0), (gx, h)], fill=(255, 255, 255), width=1)
    for gy in range(0, h, 64):
        d.line([(0, gy), (w, gy)], fill=(255, 255, 255), width=1)
    # Corner markers make letterboxing/cropping obvious in a caption smoke test.
    d.rectangle([0, 0, 31, 31], fill=(255, 0, 0))
    d.rectangle([w - 32, h - 32, w - 1, h - 1], fill=(0, 0, 255))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in SIZES:
        w, h = (int(v) for v in name.split("x"))
        path = os.path.join(OUT, f"{name}.png")
        render(w, h).save(path)
        print(f"{path}  {w}x{h}  {os.path.getsize(path)} B")


if __name__ == "__main__":
    main()
