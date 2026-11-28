import os
import numpy as np
from PIL import Image, ImageDraw

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
SETS = (("05thinking", [1, 3, 4, 12, 29, 33, 41]), ("06busy", [1, 18, 29, 32, 39, 43, 44]))
TH = 250
for act, idxs in SETS:
    tiles = []
    for i in idxs:
        a = Image.open(os.path.join(OUT, act, "_norm", "norm_%02d.png" % i)).convert("RGBA")
        for bg in ((255, 255, 255, 255), (0, 0, 0, 255)):
            c = Image.new("RGBA", a.size, bg)
            c.alpha_composite(a)
            tiles.append((i, bg[0], c.convert("RGB").resize((TH, TH), Image.LANCZOS)))
    cols = len(idxs)
    sheet = Image.new("RGB", (cols * TH, 2 * (TH + 20)), (90, 90, 90))
    dr = ImageDraw.Draw(sheet)
    for k, (i, bg, t) in enumerate(tiles):
        r = k // cols
        c = k % cols
        x, y = c * TH, r * (TH + 20)
        sheet.paste(t, (x, y + 20))
        dr.text((x + 5, y + 4), "f%d bg%s" % (i, "W" if bg else "K"), fill=(255, 225, 0))
    sp = os.path.join(OUT, "_diag0506", act + "_norm_check.png")
    sheet.save(sp)
    print("->", sp, sheet.size)
