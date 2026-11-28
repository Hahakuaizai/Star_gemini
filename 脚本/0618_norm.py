import os, json
import numpy as np, cv2
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
OUT = os.path.join(F, "_norm")
os.makedirs(OUT, exist_ok=True)
S, CAN, BODY_H, FOOT_Y = 448, 384, 305, 372

def load(i):
    p = os.path.join(F, "_fixed" if i >= 25 else "_cells", "c%02d.png" % i)
    if i >= 25 and not os.path.exists(p):
        p = os.path.join(F, "_cells", "c%02d.png" % i)
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    L = a.max(axis=2)
    al = np.clip((L - 10) / 30.0, 0, 1)
    return a, al

body = []
for i in range(1, 49):
    rgb, al = load(i)
    m = (al > 0.15).astype(np.uint8)
    n, lab, st, cn = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        body.append(None); continue
    # 最大的连通域＝角色本体（图标/星星是碎块）
    k = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, area = st[k]
    body.append((x, y, w, h, k))
    print("c%02d body bbox=(%d,%d,%d,%d) h=%d area=%d" % (i, x, y, w, h, h, area))
HS = [b[3] for b in body if b]
HEMED = float(np.median(HS))
SC = BODY_H / HEMED
print("median body h", HEMED, "scale", round(SC, 4))

meta = []
for i in range(1, 49):
    rgb, al = load(i)
    b = body[i - 1]
    x, y, w, h, k = b
    cx = x + w / 2
    src = np.dstack([rgb, (al * 255).astype(np.float32)])
    M = np.array([[SC, 0, CAN / 2 - cx * SC], [0, SC, FOOT_Y - (y + h) * SC]], np.float32)
    dst = cv2.warpAffine(src, M, (CAN, CAN), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out = np.clip(dst, 0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(os.path.join(OUT, "f%02d.png" % i))
    meta.append(dict(i=i, bbox=[int(x), int(y), int(w), int(h)], cx=round(float(cx), 2)))
json.dump(dict(scale=round(SC, 4), body_h=HEMED, frames=meta), open(os.path.join(F, "_norm_meta.json"), "w"), indent=1)
print("done", len(meta))
