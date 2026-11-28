import os, glob
import numpy as np
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审", "_diag0506")


def cell(act, i):
    idx = i - 1
    p = sorted(glob.glob(os.path.join(G, "素材", act, "part*.webp")))[idx // 12]
    im = Image.open(p).convert("RGB")
    w, h = im.size
    cw, ch = w // 4, h // 3
    k = idx % 12
    r, c = divmod(k, 4)
    return im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))


tiles = []
for i in (28, 29, 31, 32, 39, 43):
    im = cell("06busy", i)
    crop = im.crop((0, int(im.height * 0.55), im.width, im.height)).resize((448, 200), Image.LANCZOS)
    tiles.append((i, crop))
sheet = Image.new("RGB", (448 * 3, (200 + 18) * 2), (60, 60, 60))
from PIL import ImageDraw
dr = ImageDraw.Draw(sheet)
for k, (i, t) in enumerate(tiles):
    x = (k % 3) * 448
    y = (k // 3) * (218)
    sheet.paste(t, (x, y + 18))
    dr.text((x + 5, y + 3), "06 f%d 底部" % i, fill=(255, 220, 0))
sheet.save(os.path.join(OUT, "06busy_bottom_check.png"))
print("ok")
