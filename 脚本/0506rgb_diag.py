import os
import cv2
import numpy as np
from PIL import Image
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
CANVAS = 384
def load(d, i):
    return np.array(Image.open(os.path.join(OUT, d, "_norm", "norm_%02d.png" % i)).convert("RGBA"), dtype=np.float32)
def pm(a):
    o = a.copy(); o[..., :3] = a[..., :3] * (a[..., 3:4] / 255.0); return o
def shift(a, b, cap=3.0):
    ala, alb = a[..., 3] / 255.0, b[..., 3] / 255.0
    r0, r1 = int(0.62 * CANVAS), int(0.95 * CANVAS)
    (dx, dy), _ = cv2.phaseCorrelate(ala[r0:r1].astype(np.float32), alb[r0:r1].astype(np.float32))
    if abs(dx) > cap or abs(dy) > cap: dx, dy = 0.0, 0.0
    mb = cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return float(np.abs(pm(a) - pm(mb)).mean() / 255.0 * 100), dx, dy
for d in ("05thinking", "06busy"):
    ds = []
    for i in range(1, 48):
        ds.append(shift(load(d, i), load(d, i + 1))[0])
    ds = np.array(ds)
    print(d, "rgb差 mean %.2f median %.2f" % (ds.mean(), np.median(ds)))
    print("   <=3:%d 3-6:%d 6-10:%d >10:%d" % ((ds <= 3).sum(), ((ds > 3) & (ds <= 6)).sum(), ((ds > 6) & (ds <= 10)).sum(), (ds > 10).sum()))
    print("   ", [round(float(x), 1) for x in ds])
