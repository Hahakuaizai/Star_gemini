import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
D = os.path.join(OUT, "_diag0506")
CANVAS = 384
PAIRS = [("05thinking", 3, 4), ("05thinking", 16, 17), ("05thinking", 28, 29), ("05thinking", 44, 45),
         ("06busy", 17, 18), ("06busy", 28, 29), ("06busy", 39, 40), ("06busy", 43, 44)]


def load(act, i):
    return np.array(Image.open(os.path.join(OUT, act, "_norm", "norm_%02d.png" % i)).convert("RGBA"),
                    dtype=np.float32)


def sdf_of(al, thr=0.5):
    m = al > thr
    if not m.any():
        return np.full(al.shape, 999.0, np.float32)
    din = ndimage.distance_transform_edt(m)
    dout = ndimage.distance_transform_edt(~m)
    return (dout - din).astype(np.float32)


def morph(a, b, t):
    aa = a[..., 3] / 255.0
    ab = b[..., 3] / 255.0
    sa = sdf_of(aa)
    sb = sdf_of(ab)
    sm = (1.0 - t) * sa + t * sb
    al = np.clip(0.5 - sm / 1.5, 0.0, 1.0)
    al[al < 0.10] = 0.0
    wa = aa * (1.0 - t)
    wb = ab * t
    den = np.maximum(wa + wb, 1e-4)
    col = (a[..., :3] * wa[..., None] + b[..., :3] * wb[..., None]) / den[..., None]
    out = np.zeros((CANVAS, CANVAS, 4), np.float32)
    out[..., :3] = np.clip(col, 0, 255)
    out[..., 3] = al * 255.0
    return out, sm


tiles = []
for act, ia, ib in PAIRS:
    a, b = load(act, ia), load(act, ib)
    for t in (0.34, 0.5, 0.66):
        m, sm = morph(a, b, t)
        c = Image.new("RGBA", (CANVAS, CANVAS), (248, 248, 248, 255))
        c.alpha_composite(Image.fromarray(np.clip(m, 0, 255).astype(np.uint8), "RGBA"))
        tiles.append((f"{act[:2]} {ia}->{ib} t{t}", c.convert("RGB").resize((186, 186), Image.LANCZOS)))
cols = 6
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (186 * cols, (186 + 17) * rows), (85, 85, 85))
dr = ImageDraw.Draw(sheet)
for k, (txt, im) in enumerate(tiles):
    x, y = (k % cols) * 186, (k // cols) * 203
    sheet.paste(im, (x, y + 17))
    dr.text((x + 3, y + 3), txt, fill=(255, 232, 0))
sheet.save(os.path.join(D, "sdf_morph.png"))
print("ok", sheet.size)
