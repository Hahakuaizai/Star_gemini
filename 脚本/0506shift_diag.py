import os
import cv2
import numpy as np
from PIL import Image
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
CANVAS = 384
def load(d, i):
    return np.array(Image.open(os.path.join(OUT, d, "_norm", "norm_%02d.png" % i)).convert("RGBA"), dtype=np.float32)
def shift_diff(a, b):
    ala = a[..., 3] / 255.0
    alb = b[..., 3] / 255.0
    (dx, dy), _ = cv2.phaseCorrelate(ala.astype(np.float32), alb.astype(np.float32))
    moved = cv2.warpAffine(alb, np.float32([[1, 0, -dx], [0, 1, -dy]]), (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderValue=0)
    rgba_d = float(np.abs(a - cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderValue=0)).mean() / 255.0 * 100)
    return float(np.abs(ala - moved).mean() * 100), rgba_d, dx, dy
for d in ("05thinking", "06busy"):
    ds, dr, dxs = [], [], []
    for i in range(1, 48):
        a, b = load(d, i), load(d, i + 1)
        sd, rd, dx, dy = shift_diff(a, b)
        ds.append(sd); dr.append(rd); dxs.append((dx, dy))
    ds = np.array(ds)
    print("==", d, "对齐alpha差 mean %.2f median %.2f min %.2f max %.2f" % (ds.mean(), np.median(ds), ds.min(), ds.max()))
    print("   分档 <=2.5:%d  <=5:%d  >5:%d" % ((ds <= 2.5).sum(), ((ds > 2.5) & (ds <= 5)).sum(), (ds > 5).sum()))
    print("   位移 dx 范围 %.1f~%.1f dy %.1f~%.1f" % (min(x for x, y in dxs), max(x for x, y in dxs), min(y for x, y in dxs), max(y for x, y in dxs)))
    print("   ", [round(float(x), 1) for x in ds])
