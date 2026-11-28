import os
import numpy as np, cv2
from scipy.ndimage import gaussian_filter

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S, HEM, FLOOR = 448, 344, 446
SCALE = 0.342

def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def face_cx(im):
    r, g, b = [im[..., k].astype(np.int16) for k in range(3)]
    m = ((r > 225) & (g > 185) & (b > 170) & ((r - b) > 18) & ((r - b) < 90)).astype(np.uint8)
    m[int(im.shape[0] * 0.55):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cn = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 210
    k = 1 + int(np.argmax(st[1:, 4]))
    return float(cn[k][0])

def ai_leg(im):
    L = im.max(axis=2); m = L > 18
    ys = np.where(m.any(axis=1))[0]
    sole = int(ys.max())
    w = m.sum(axis=1)
    top = None
    for y in range(sole, 700, -1):
        if w[y] > 400:
            top = y
            break
    top = (top or int(im.shape[0]*0.78)) + 6
    band = m[top:sole + 1]
    col = band.sum(axis=0)
    xs = np.where(col >= 12)[0]
    x0, x1 = int(xs.min()), int(xs.max())
    return top, sole, x0, x1

for i in range(25, 45):
    orig = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    ai = rd(os.path.join(F, "_ai", "c%02d.png" % i))
    top, sole, ax0, ax1 = ai_leg(ai)
    leg = ai[top:sole + 1, ax0:ax1 + 1]
    lh, lw = leg.shape[:2]
    tw = int(round(lw * SCALE))
    th = FLOOR - HEM
    rs = cv2.resize(leg, (tw, th), interpolation=cv2.INTER_AREA)
    Lr = rs.max(axis=2).astype(np.float32)
    a = np.clip((Lr - 10) / 30.0, 0, 1)
    col = np.clip(rs.astype(np.float32) / np.maximum(a, 0.55)[..., None], 0, 255)

    cx = face_cx(orig) + 11.0
    left = int(round(cx - tw / 2))
    canvas = orig.astype(np.float32).copy()
    # 清掉原始区域（腿根残留、旧断腿）
    canvas[HEM - 2:FLOOR + 2, left - 2:left + tw + 2] = canvas[HEM - 2:FLOOR + 2, left - 2:left + tw + 2] * 0.02

    mask = np.zeros((S, S), np.float32)
    ys0, ys1 = HEM, HEM + th
    xs0, xs1 = left, left + tw
    pad = np.clip(mask[ys0:ys1, xs0:xs1].shape, 0, None)
    mask[ys0:ys1, xs0:xs1] = 1.0
    mask = gaussian_filter(mask, 2.0)
    # 顶部羽化 6px（藏在裙下）
    ramp = np.ones((th, tw), np.float32)
    ramp[:8] = np.linspace(0, 1, 8)[:, None]
    mask[ys0:ys1, xs0:xs1] = np.minimum(mask[ys0:ys1, xs0:xs1], 1.0) * ramp if False else mask[ys0:ys1, xs0:xs1]
    m3 = np.zeros((S, S, 1), np.float32)
    m3[ys0:ys1, xs0:xs1, 0] = a * ramp
    m3[ys0:ys1, xs0:xs1, 0] = gaussian_filter(m3[ys0:ys1, xs0:xs1, 0], 1.0)

    comp = canvas.copy()
    comp[ys0:ys1, xs0:xs1] = m3[ys0:ys1, xs0:xs1] * col + (1 - m3[ys0:ys1, xs0:xs1]) * canvas[ys0:ys1, xs0:xs1]
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(F, "_fixed", "c%02d.png" % i), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print("c%02d ai_leg y%4d..%4d x%4d..%4d -> fit (%d,%d) w=%d h=%d" % (i, top, sole, ax0, ax1, left, HEM, tw, th))
