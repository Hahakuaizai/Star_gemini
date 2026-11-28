import os, json
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S = 448
IDX = list(range(25, 45))
SEAM = 258          # 0.576 * 448

def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

report = []
for i in IDX:
    orig = load(os.path.join(F, "_cells", "c%02d.png" % i))
    ai = load(os.path.join(F, "_ai", "c%02d.png" % i))
    tpl = orig[110:330, 95:335]
    best = (-1, None, None)
    for s in np.arange(0.26, 0.421, 0.01):
        n = int(round(ai.shape[0] * s))
        rs = cv2.resize(ai, (n, n), interpolation=cv2.INTER_AREA)
        if rs.shape[0] <= tpl.shape[0]:
            continue
        res = cv2.matchTemplate(rs, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, s, ml)
    # refine
    sc0 = best[1]
    for s in np.arange(sc0 - 0.012, sc0 + 0.0125, 0.003):
        n = int(round(ai.shape[0] * s))
        rs = cv2.resize(ai, (n, n), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(rs, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, s, ml)
    score, s, (x0, y0) = best
    M = np.array([[1.0 / s, 0, x0 / s], [0, 1.0 / s, y0 / s]], dtype=np.float32)
    wai = cv2.warpAffine(ai, M, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    L = wai.max(axis=2).astype(np.float32)
    alpha = np.clip((L - 12.0) / 34.0, 0, 1)
    color = wai.astype(np.float32) / np.maximum(alpha, 0.55)[..., None]
    color = np.clip(color, 0, 255)

    yy = np.arange(S)[:, None]
    mask = ((alpha > 0.06) & (yy >= SEAM)).astype(np.float32)
    mask = gaussian_filter(mask, 1.2)

    comp = mask[..., None] * color + (1 - mask[..., None]) * orig.astype(np.float32)
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    up = np.abs(comp[:240].astype(np.int16) - orig[:240].astype(np.int16)).mean()
    cv2.imwrite(os.path.join(F, "_fixed", "c%02d.png" % i), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    report.append((i, round(float(score), 3), round(float(s), 4), int(x0), int(y0), round(float(up), 3)))
    print("c%02d score=%.3f s=%.4f off=(%d,%d) upperdiff=%.3f" % (i, score, s, x0, y0, up))

json.dump(report, open(os.path.join(F, "_align.json"), "w"), ensure_ascii=False, indent=1)
print("mean score", round(np.mean([r[1] for r in report]), 3))
