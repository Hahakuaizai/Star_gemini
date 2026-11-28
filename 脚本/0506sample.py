import os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
D = os.path.join(OUT, "_diag0506")
TH = 170


def find(prefix):
    return [p for p in glob.glob(os.path.join(OUT, prefix + "*", "*.webp")) if "sheet" not in os.path.basename(p)] or \
           [p for p in glob.glob(os.path.join(OUT, "0*", "*.webp")) if os.path.basename(p).startswith(prefix) and "sheet" not in os.path.basename(p)]


for pre in ("05thinking（", "06working_busy（"):
    hits = [p for p in glob.glob(os.path.join(OUT, "0*", "*.webp"))
            if os.path.basename(p).startswith(pre) and "sheet" not in os.path.basename(p)]
    p = hits[0]
    im = Image.open(p)
    n = im.n_frames
    idx = [0, 2, 4, 7] + list(range(10, n - 1, max(1, n // 12)))[:9] + [n - 2, n - 1]
    idx = sorted(set([min(i, n - 1) for i in idx]))
    tiles = []
    for i in idx:
        im.seek(i)
        f = im.convert("RGBA")
        c = Image.new("RGBA", f.size, (247, 247, 247, 255))
        c.alpha_composite(f)
        tiles.append((i + 1, c.convert("RGB").resize((TH, TH), Image.LANCZOS)))
    cols = 7
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (TH * cols, (TH + 16) * rows), (90, 90, 90))
    dr = ImageDraw.Draw(sheet)
    for k, (i, t) in enumerate(tiles):
        x, y = (k % cols) * TH, (k // cols) * (TH + 16)
        sheet.paste(t, (x, y + 16))
        dr.text((x + 4, y + 2), "f%d" % i, fill=(20, 20, 20))
    sp = os.path.join(D, pre[:2] + "_built_sample.png")
    sheet.save(sp)
    print(pre, n, "->", sp)
