import os, json
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, binary_dilation

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S = 448
SCALE = 0.342
LEGH = 96

def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def face_cx(im):
    r, g, b = [im[..., k].astype(np.int16) for k in range(3)]
    m = ((r > 225) & (g > 185) & (b > 170) & ((r - b) > 18) & ((r - b) < 90)).astype(np.uint8)
    m[int(im.shape[0] * 0.55):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cn = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 210.0
    return float(cn[1 + int(np.argmax(st[1:, 4]))][0])

def ai_leg(im):
    L = im.max(axis=2); m = L > 18
    sole = int(np.where(m.any(axis=1))[0].max())
    w = m.sum(axis=1)
    skirt = int(im.shape[0] * 0.78)
    for y in range(sole, 700, -1):
        if w[y] > 400:
            skirt = y; break
    top = skirt + 12
    cx = face_cx(im)
    band = m[top:sole + 1]; col = band.sum(axis=0)
    lo, hi = int(max(0, cx - 190)), int(min(im.shape[1], cx + 150))
    xs = np.where(col[lo:hi] >= 12)[0] + lo
    return top, sole, int(xs.min()), int(xs.max())

def floor_of(i):
    return 430 if i == 25 else (447 if i <= 32 else 445)

for i in range(25, 45):
    orig = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    ai = rd(os.path.join(F, "_ai", "c%02d.png" % i))
    top, sole, ax0, ax1 = ai_leg(ai)
    leg = ai[top:sole + 1, ax0:ax1 + 1]
    tw = max(40, int(round((ax1 - ax0 + 1) * SCALE)))
    rs = cv2.resize(leg, (tw, LEGH), interpolation=cv2.INTER_AREA)
    a = np.clip((rs.max(axis=2).astype(np.float32) - 14) / 34.0, 0, 1)
    col = np.clip(rs.astype(np.float32) / np.maximum(a, 0.55)[..., None], 0, 255)

    fl = floor_of(i); y0 = fl - LEGH + 1
    cx = face_cx(orig) + 11.0
    x0 = int(round(cx - tw / 2))

    base = orig.astype(np.float32)
    whole = np.zeros((S, S), np.float32); whole[y0:y0 + LEGH, x0:x0 + tw] = a
    whole = binary_dilation(whole > 0.10, iterations=8).astype(np.float32)
    whole = gaussian_filter(whole, 2.0)

    r, g, b = [base[..., k] for k in range(3)]
    skin = (r > 185) & (g > 145) & (b > 135) & ((r - b) < 60) & ((r - b) > -25)
    orange = (r > 110) & (g > 45) & (g < 175) & (b < 95)
    below = np.zeros((S, S), bool); below[340:, :] = True
    kill = (skin | orange) & below & (whole > 0.05)
    base = np.where(kill[..., None], base * 0.06, base)

    M = a.copy()
    ramp = np.ones(LEGH, np.float32); ramp[:14] = np.linspace(0.15, 1, 14)
    M *= ramp[:, None]
    M = gaussian_filter(M, 1.1)
    reg = base[y0:y0 + LEGH, x0:x0 + tw]
    base[y0:y0 + LEGH, x0:x0 + tw] = M[..., None] * col + (1 - M[..., None]) * reg

    comp = np.clip(base, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(F, "_fixed", "c%02d.png" % i), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print("c%02d floor=%d box=(%d,%d,%d,%d) crop=(%d,%d,%d,%d)" % (i, fl, x0, y0, tw, LEGH, top, sole, ax0, ax1))
