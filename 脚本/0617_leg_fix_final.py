import os, json
import numpy as np
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
BOX = (163, 328, 272, 447)          # c46 腿部（含大腿到靴底）
FLOORS = {25: 430, 26: 447, 27: 447, 28: 447}

def face_cx(im):
    r, g, b = [im[..., k].astype(np.int16) for k in range(3)]
    m = ((r > 225) & (g > 185) & (b > 170) & ((r - b) > 18) & ((r - b) < 90)).astype(np.uint8)
    m[int(im.shape[0] * 0.55):] = 0
    import cv2
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cn = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 210.0
    return float(cn[1 + int(np.argmax(st[1:, 4]))][0])

sp = np.asarray(Image.open(os.path.join(F, "_cells", "c46.png")).crop(BOX)).astype(np.float32)
h, w = sp.shape[:2]
al = np.clip((sp.max(axis=2) - 10) / 26.0, 0, 1)
col = np.clip(sp / np.maximum(al, 0.6)[..., None], 0, 255)

rep = []
for i in range(25, 45):
    im = np.asarray(Image.open(os.path.join(F, "_cells", "c%02d.png" % i)).convert("RGB")).astype(np.float32)
    fl = FLOORS.get(i, 445)
    cx = face_cx(im.astype(np.uint8))
    x0 = int(round(cx + 14 - w / 2)); y0 = fl - h + 1
    a = al.copy()
    a[:14] *= np.linspace(0.12, 1, 14)[:, None]
    reg = im[y0:y0 + h, x0:x0 + w]
    out = im.copy()
    out[y0:y0 + h, x0:x0 + w] = a[..., None] * col + (1 - a[..., None]) * reg
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(F, "_fixed", "c%02d.png" % i))
    rep.append(dict(i=i, floor=fl, cx=round(cx, 1), x0=x0, y0=y0))
    print("c%02d floor=%d cx=%.1f paste=(%d,%d)" % (i, fl, cx, x0, y0))
json.dump(rep, open(os.path.join(F, "_fix_plate.json"), "w"), indent=1)
print("plate", w, h)
