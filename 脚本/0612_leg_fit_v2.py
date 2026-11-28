import os, json
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, binary_dilation

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S = 448
SCALE = 0.342
LEGH = 102

def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def face_cx(im):
    r, g, b = [im[..., k].astype(np.int16) for k in range(3)]
    m = ((r > 225) & (g > 185) & (b > 170) & ((r - b) > 18) & ((r - b) < 90)).astype(np.uint8)
    m[int(im.shape[0] * 0.55):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cn = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 210.0
    k = 1 + int(np.argmax(st[1:, 4]))
    return float(cn[k][0])

def ai_leg(im):
    L = im.max(axis=2); m = L > 18
    sole = int(np.where(m.any(axis=1))[0].max())
    w = m.sum(axis=1)
    top = None
    for y in range(sole, 700, -1):
        if w[y] > 400:
            top = y; break
    top = (top or int(im.shape[0] * 0.78)) + 6
    cx = face_cx(im)
    band = m[top:sole + 1]
    col = band.sum(axis=0)
    lo, hi = int(max(0, cx - 190)), int(min(im.shape[1], cx + 150))
    xs = np.where(col[lo:hi] >= 12)[0] + lo
    return top, sole, int(xs.min()), int(xs.max())

def floor_of(i):
    if i == 25: return 430
    if i <= 32: return 447
    return 445

out = []
for i in range(25, 45):
    orig = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    ai = rd(os.path.join(F, "_ai", "c%02d.png" % i))
    top, sole, ax0, ax1 = ai_leg(ai)
    leg = ai[top:sole + 1, ax0:ax1 + 1]
    tw = max(40, int(round((ax1 - ax0 + 1) * SCALE)))
    rs = cv2.resize(leg, (tw, LEGH), interpolation=cv2.INTER_AREA)
    Lr = rs.max(axis=2).astype(np.float32)
    a = np.clip((Lr - 10) / 30.0, 0, 1)
    col = np.clip(rs.astype(np.float32) / np.maximum(a, 0.55)[..., None], 0, 255)

    fl = floor_of(i)
    y0, y1 = fl - LEGH + 1, fl + 1
    cx = face_cx(orig) + 11.0
    x0 = int(round(cx - tw / 2)); x1 = x0 + tw

    comp = orig.astype(np.float32).copy()
    M = np.zeros((LEGH, tw), np.float32)
    M[:, :] = a
    ramp = np.ones(LEGH, np.float32); ramp[:9] = np.linspace(0.15, 1, 9)
    M *= ramp[:, None]
    M = gaussian_filter(M, 1.0)

    region = comp[y0:y1, x0:x1]
    comp[y0:y1, x0:x1] = M[..., None] * col + (1 - M[..., None]) * region

    # 清掉原图残留（肤色断腿、橙块）在腿框内 alpha 低的位置
    r, g, b = [orig[..., k].astype(np.int16) for k in range(3)]
    skin = (r > 190) & (g > 150) & (b > 140) & ((r - b) < 55)
    org = (orig.max(axis=2) > 12)
    kill = skin | ((r > 120) & (g > 60) & (g < 180) & (b < 90))   # 橙色残留
    killmask = kill[y0:y1, x0:x1] & (M < 0.30) & org[y0:y1, x0:x1]
    comp[y0:y1, x0:x1][killmask] = comp[y0:y1, x0:x1][killmask] * 0.05

    comp = np.clip(comp, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(F, "_fixed", "c%02d.png" % i), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    out.append(dict(i=i, floor=fl, tw=tw, x0=x0, x1=x1))
    print("c%02d floor=%d tw=%d x=%d..%d" % (i, fl, tw, x0, x1))
json.dump(out, open(os.path.join(F, "_fit.json"), "w"), indent=1)
