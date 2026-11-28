import os, sys
import numpy as np, cv2
from PIL import Image
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
N = os.path.join(F, "_norm")
S = 384

def rd(i):
    return np.asarray(Image.open(os.path.join(N, "f%02d.png" % i))).astype(np.float32)

def calc_flow(a, b):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    g1 = cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(b[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fab = dis.calc(g1, g2, None)
    fba = dis.calc(g2, g1, None)
    return fab[..., 0], fab[..., 1], fba[..., 0], fba[..., 1]

def warp(img, fx, fy, t):
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    return cv2.remap(img, xx + t * fx, yy + t * fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def mid(a, b, t, flow=None, mode='blend'):
    fx, fy, gx, gy = flow if flow else calc_flow(a, b)
    fx, fy = cv2.GaussianBlur(fx, (0, 0), 2), cv2.GaussianBlur(fy, (0, 0), 2)
    gx, gy = cv2.GaussianBlur(gx, (0, 0), 2), cv2.GaussianBlur(gy, (0, 0), 2)
    aw = warp(a, fx, fy, t)
    bw_ = warp(b, gx, gy, 1 - t)
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    bx = cv2.remap(gx, xx + fx, yy + fy, cv2.INTER_LINEAR)
    by = cv2.remap(gy, xx + fx, yy + fy, cv2.INTER_LINEAR)
    err = cv2.GaussianBlur(np.abs(bx + fx) + np.abs(by + fy), (0, 0), 4)
    cons = np.clip(1 - err / 1.6, 0, 1)
    if mode == 'nearest':
        cons = np.zeros_like(cons)

    aa = aw[..., 3] / 255.0
    ba = bw_[..., 3] / 255.0
    hard = 1 - cons
    wA = (1 - t) * cons
    wB = t * cons
    if t < 0.5:
        wA = wA + hard
        al = (cons * (aa * (1 - t) + ba * t) + hard * aa) * 255
    else:
        wB = wB + hard
        al = (cons * (aa * (1 - t) + ba * t) + hard * ba) * 255
    den = (wA + wB)[..., None] + 1e-4
    col = (aw[..., :3] * wA[..., None] + bw_[..., :3] * wB[..., None]) / den
    out = np.concatenate([col, al[..., None]], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    for i, j, tag in ((19, 20, "19_20"), (45, 46, "45_46")):
        a, b = rd(i), rd(j)
        fl = calc_flow(a, b)
        m1 = mid(a, b, 0.34, fl, 'nearest'); m2 = mid(a, b, 0.67, fl, 'nearest')
        Image.fromarray(np.hstack([a.astype(np.uint8), m1, m2, b.astype(np.uint8)]), "RGBA").save(
            os.path.join(F, "_interp2_%s.png" % tag))
    print("ok")
