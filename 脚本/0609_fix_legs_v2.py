import os, json
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, binary_dilation

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S = 448
BAND = 330        # 裙摆下缘以下才开始贴
SIDE = (112, 340) # 只贴腿区，保护两侧头发/尾巴
FLOOR = 446       # 脚底目标线
IDX = list(range(25, 45))

def rd(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def align(orig, ai):
    tpl = orig[110:330, 95:335]
    best = (-1, 0.35, (0, 0))
    for s in np.arange(0.26, 0.421, 0.008):
        n = int(round(1536 * s))
        rs = cv2.resize(ai, (n, n), interpolation=cv2.INTER_AREA)
        if rs.shape[0] <= tpl.shape[0]:
            continue
        res = cv2.matchTemplate(rs, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, s, ml)
    sc = best[1]
    for s in np.arange(sc - 0.010, sc + 0.0105, 0.002):
        n = int(round(1536 * s))
        rs = cv2.resize(ai, (n, n), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(rs, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, s, ml)
    return best

def warp(ai, s, x0, y0, k=1.0):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ax = (xx + x0) / s
    ay = (yy + y0) / s
    band = yy >= BAND
    ay = np.where(band, (BAND + y0) / s + (yy - BAND) / (s * k), ay)
    return cv2.remap(ai, ax, ay, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

report = []
for i in IDX:
    orig = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    ai = rd(os.path.join(F, "_ai", "c%02d.png" % i))
    score, s, (x0, y0) = align(orig, ai)

    w1 = warp(ai, s, x0, y0, 1.0)
    L1 = w1.max(axis=2).astype(np.float32)
    a1 = np.clip((L1 - 12) / 34.0, 0, 1)
    col1 = np.where(a1[..., None] > 0.02, w1.astype(np.float32) / np.maximum(a1, 0.55)[..., None], 0)
    sole = None
    for y in range(S - 1, BAND, -1):
        row = a1[y, 160:300]
        if (row > 0.12).sum() >= 22:
            sole = y
            break
    k = 1.0
    if sole and sole > BAND + 40:
        k = float((FLOOR - BAND) / (sole - BAND))
        k = float(np.clip(k, 1.0, 1.45))

    w2 = warp(ai, s, x0, y0, k)
    L = w2.max(axis=2).astype(np.float32)
    a = np.clip((L - 12) / 34.0, 0, 1)
    col = np.clip(w2.astype(np.float32) / np.maximum(a, 0.55)[..., None], 0, 255)

    yy, xx = np.mgrid[0:S, 0:S]
    side = (xx >= SIDE[0]) & (xx <= SIDE[1])
    m = ((a > 0.06) & (yy >= BAND) & side).astype(np.float32)
    m = gaussian_filter(m, 1.3)
    m = np.clip(m * 1.6, 0, 1)[..., None]

    base = orig.astype(np.float32).copy()
    # 清掉原图残留的断腿（肤色像素，位于腿区、且 AI 有内容的膨胀范围内）
    legmask = ((a > 0.10) & (yy >= BAND) & side)
    legmask = binary_dilation(legmask, iterations=7)
    skin = (orig[..., 0] > 190) & (orig[..., 1] > 150) & (orig[..., 2] > 140) & \
           ((orig[..., 0].astype(np.int16) - orig[..., 2]) < 55)
    kill = (legmask & skin & (yy >= BAND))[..., None]
    base = np.where(kill, base * 0.10, base)

    comp = m * col + (1 - m) * base
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    up = np.abs(comp[:250].astype(np.int16) - orig[:250].astype(np.int16)).mean()
    cv2.imwrite(os.path.join(F, "_fixed", "c%02d.png" % i), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    report.append(dict(i=i, score=round(float(score), 3), s=round(float(s), 4), sole=sole, k=round(k, 3), up=round(float(up), 3)))
    print("c%02d score=%.3f s=%.3f sole=%s k=%.3f up=%.3f" % (i, score, s, sole, k, up))
json.dump(report, open(os.path.join(F, "_align2.json"), "w"), indent=1)
