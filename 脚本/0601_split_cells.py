import os, glob
import numpy as np
from PIL import Image, ImageDraw

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(G, "素材", "06busy")
OUT = os.path.join(G, "待审", "06busy_fix", "_cells")
os.makedirs(OUT, exist_ok=True)

cells = []
for p in sorted(glob.glob(os.path.join(SRC, "part*.webp"))):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    cw, ch = w // 4, h // 3
    for r in range(3):
        for c in range(4):
            cells.append(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
print("cells", len(cells), cells[0].size)

rep = []
for i, c in enumerate(cells):
    a = np.asarray(c, dtype=np.float32)
    L = a.max(axis=2)
    m = L > 18
    ys, xs = np.where(m)
    bottom = int(ys.max()) if len(ys) else -1
    edge = int(m[440:].sum())
    rep.append((i + 1, bottom, edge))
    c.save(os.path.join(OUT, "c%02d.png" % (i + 1)))

print("idx bottom_y bottom_12rows_px")
for i, b, e in rep:
    print("%02d %3d %5d %s" % (i, b, e, "<= 贴底" if b >= 440 else ""))
