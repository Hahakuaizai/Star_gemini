import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
D = os.path.join(OUT, "_diag0506")
CANVAS = 384
PAIRS = [("05thinking", 4, 5), ("05thinking", 16, 17), ("05thinking", 28, 29), ("05thinking", 36, 37),
         ("06busy", 17, 18), ("06busy", 28, 29), ("06busy", 39, 40), ("06busy", 43, 44)]


def load(act, i):
    return np.array(Image.open(os.path.join(OUT, act, "_norm", "norm_%02d.png" % i)).convert("RGBA"),
                    dtype=np.float32)


def premul(a):
    o = a.copy()
    o[..., :3] = a[..., :3] * (a[..., 3:4] / 255.0)
    return o


def unpremul(a):
    o = a.copy()
    al = a[..., 3:4] / 255.0
    o[..., :3] = np.clip(np.where(al > 1e-4, a[..., :3] / np.maximum(al, 1e-4), 0.0), 0, 255)
    return o


def gray(a):
    g = a[..., 3:4] / 255.0 * 255.0
    lum = 0.3 * a[..., 0:1] + 0.59 * a[..., 1:2] + 0.11 * a[..., 2:3]
    return np.clip(0.55 * g + 0.45 * lum, 0, 255).astype(np.uint8)[..., 0]


DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flow_dis(a, b):
    DIS.setUseSpatialPropagation(True)
    return DIS.calc(gray(a), gray(b), None)


def flow_fb(a, b):
    return cv2.calcOpticalFlowFarneback(gray(a), gray(b), None, 0.5, 4, 25, 5, 7, 1.5, 0)


def warp_flow(img, fl, t):
    h, w = fl.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = x + fl[..., 0] * t
    my = y + fl[..., 1] * t
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def one(a, b, t, ff):
    fab = ff(a, b)
    fba = ff(b, a)
    if t < 0.5:
        return unpremul(warp_flow(premul(a), fab, t))
    return unpremul(warp_flow(premul(b), fba, 1.0 - t))


tiles = []
for act, ia, ib in PAIRS:
    a, b = load(act, ia), load(act, ib)
    for name, ff in (("FB", flow_fb), ("DIS", flow_dis)):
        for t in (0.34, 0.66):
            m = one(a, b, t, ff)
            c = Image.new("RGBA", (CANVAS, CANVAS), (250, 250, 250, 255))
            c.alpha_composite(Image.fromarray(np.clip(m, 0, 255).astype(np.uint8), "RGBA"))
            tiles.append((f"{act[:2]} {ia}->{ib} {name} t{t}", c.convert("RGB").resize((190, 190), Image.LANCZOS)))
cols = 6
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (190 * cols, (190 + 17) * rows), (80, 80, 80))
dr = ImageDraw.Draw(sheet)
for k, (txt, im) in enumerate(tiles):
    x = (k % cols) * 190
    y = (k // cols) * 207
    sheet.paste(im, (x, y + 17))
    dr.text((x + 3, y + 3), txt, fill=(255, 232, 0))
sheet.save(os.path.join(D, "interp_single.png"))
print("ok", sheet.size)
