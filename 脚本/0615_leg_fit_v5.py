import os, json
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, binary_dilation

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S, LEGH = 448, 96

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
    cx = face_cx(im)
    lo, hi = int(max(0, cx - 210)), int(min(im.shape[1], cx + 170))
    band = m[sole - 45:sole - 3]
    xs = np.where(band.sum(axis=0)[lo:hi] >= 3)[0] + lo
    bx0, bx1 = int(xs.min()), int(xs.max())
    w = m.sum(axis=1)
    skirt = int(im.shape[0] * 0.78)
    for y in range(max(700, sole - 260), 600, -1):
        if w[y] > 400:
            skirt = y; break
    top = skirt + 10
    sub = m[top:sole + 1]
    xs2 = np.where(sub.sum(axis=0)[max(0, bx0 - 25):bx1 + 26] >= 5)[0] + max(0, bx0 - 25)
    return top, sole, int(xs2.min()), int(xs2.max()), bx1 - bx0 + 1

def floor_of(i): return 430 if i == 25 else (447 if i <= 32 else 445)

def skin_mask(a, ylim=340):
    r, g, b = [a[..., k] for k in range(3)]
    m = (r > 150) & (g > 110) & (b > 100) & ((r - b) < 70) & ((r - b) > 0) & (r >= g)
    m[:ylim] = False
    return m

ref = rd(os.path.join(F, "_cells", "c46.png")).astype(np.float32)
rm = skin_mask(ref); rm[:, :110] = False; rm[:, 310:] = False
tgt = np.array([ref[..., k][rm].mean() for k in range(3)])

res = []
for i in range(25, 45):
    orig = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    ai = rd(os.path.join(F, "_ai", "c%02d.png" % i))
    top, sole, ax0, ax1, bspan = ai_leg(ai)
    leg = ai[top:sole + 1, ax0:ax1 + 1]
    s = 80.0 / bspan
    tw = max(40, int(round((ax1 - ax0 + 1) * s)))
    rs = cv2.resize(leg, (tw, LEGH), interpolation=cv2.INTER_AREA)
    a = np.clip((rs.max(axis=2).astype(np.float32) - 14) / 34.0, 0, 1)
    col = np.clip(rs.astype(np.float32) / np.maximum(a, 0.55)[..., None], 0, 255)

    sm = skin_mask(col, ylim=0) & (a > 0.6)
    src = np.array([col[..., k][sm].mean() for k in range(3)])
    col = np.clip(col * (tgt / np.maximum(src, 1)), 0, 255)

    fl = floor_of(i); y0 = fl - LEGH + 1
    cx = face_cx(orig) + 11.0
    x0 = int(round(cx - tw / 2))

    base = orig.astype(np.float32)
    whole = np.zeros((S, S), np.float32); whole[y0:y0 + LEGH, x0:x0 + tw] = a
    whole = binary_dilation(whole > 0.10, iterations=8).astype(np.float32)
    whole = gaussian_filter(whole, 2.0)

    r, g, b = [base[..., k] for k in range(3)]
    sk = (r > 185) & (g > 145) & (b > 135) & ((r - b) < 60) & ((r - b) > -25)
    orange = (r > 110) & (g > 45) & (g < 175) & (b < 95)
    below = np.zeros((S, S), bool); below[340:, :] = True
    kill = (sk | orange) & below & (whole > 0.05)
    base = np.where(kill[..., None], base * 0.06, base)

    M = a.copy()
    ramp = np.ones(LEGH, np.float32); ramp[:12] = np.linspace(0.2, 1, 12)
    M *= ramp[:, None]
    M = gaussian_filter(M, 1.1)
    reg = base[y0:y0 + LEGH, x0:x0 + tw]
    base[y0:y0 + LEGH, x0:x0 + tw] = M[..., None] * col + (1 - M[..., None]) * reg

    comp = np.clip(base, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(F, "_fixed", "c%02d.png" % i), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    res.append(dict(i=i, floor=fl, s=round(float(s), 4), tw=tw, x0=x0,
                    crop=[top, sole, ax0, ax1], src_tone=[round(float(v), 1) for v in src]))
    print("c%02d floor=%d s=%.3f tw=%3d x0=%3d crop=%s" % (i, fl, s, tw, x0, [top, sole, ax0, ax1]))
json.dump(res, open(os.path.join(F, "_fit5.json"), "w"), indent=1)
