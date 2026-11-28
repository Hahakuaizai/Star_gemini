import os, json
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
SPRITE = os.path.join(F, "_ai", "c29.png")
BOX = (620, 1170, 960, 1505)
S = 0.342
FLOORS = {25: 430, 26: 447, 27: 447, 28: 447, 29: 447, 30: 447, 31: 447, 32: 447}

def face_cx(im):
    r, g, b = [im[..., k].astype(np.int16) for k in range(3)]
    m = ((r > 225) & (g > 185) & (b > 170) & ((r - b) > 18) & ((r - b) < 90)).astype(np.uint8)
    m[int(im.shape[0] * 0.55):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cn = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 210.0
    return float(cn[1 + int(np.argmax(st[1:, 4]))][0])

sp = np.asarray(Image.open(SPRITE).crop(BOX).resize(
    (int(round((BOX[2]-BOX[0])*S)), int(round((BOX[3]-BOX[1])*S))), Image.LANCZOS)).astype(np.float32)
h, w = sp.shape[:2]
al = np.clip((sp.max(axis=2) - 14) / 34.0, 0, 1)
col = np.clip(sp / np.maximum(al, 0.55)[..., None], 0, 255)

info = []
for i in range(25, 45):
    im = np.asarray(Image.open(os.path.join(F, "_cells", "c%02d.png" % i)).convert("RGB")).astype(np.float32)
    fl = FLOORS.get(i, 445)
    y0 = fl - h + 1
    cx = face_cx(im.astype(np.uint8))
    x0 = int(round(cx + 11 - w / 2))

    a = al.copy()
    a[:8] *= np.linspace(0, 1, 8)[:, None]
    a = gaussian_filter(a, 0.8)

    # 只清除贴图覆盖范围附近的原图肤色/橙块残留
    r, g, b = [im[..., k] for k in range(3)]
    sk = (r > 185) & (g > 145) & (b > 135) & ((r - b) < 60) & ((r - b) > -25)
    orn = (r > 110) & (g > 45) & (g < 175) & (b < 95)
    cover = np.zeros((448, 448), np.float32)
    cover[max(0, y0):y0 + h, max(0, x0):x0 + w] = (a > 0.15)
    cover = binary_dilation(cover > 0, iterations=10).astype(np.float32)
    cover = gaussian_filter(cover, 2.0)
    lim = np.zeros((448, 448), bool); lim[335:, :] = True
    canvas = np.where(((sk | orn) & lim & (cover > 0.15))[..., None], im * 0.05, im)

    reg = canvas[y0:y0 + h, x0:x0 + w]
    canvas[y0:y0 + h, x0:x0 + w] = a[..., None] * col + (1 - a[..., None]) * reg

    out = np.clip(canvas, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(F, "_fixed", "c%02d.png" % i))
    info.append(dict(i=i, floor=fl, cx=round(cx, 1), x0=x0, y0=y0))
    print("c%02d floor=%d cx=%.1f box=(%d,%d,%d,%d)" % (i, fl, cx, x0, y0, w, h))
print("sprite", w, h)
json.dump(info, open(os.path.join(F, "_paste.json"), "w"), indent=1)
