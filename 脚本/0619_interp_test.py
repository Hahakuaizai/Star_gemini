import os
import numpy as np, cv2
from PIL import Image
F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "待审", "06busy_fix")
N = os.path.join(F, "_norm")

def rd(i):
    a = np.asarray(Image.open(os.path.join(N, "f%02d.png" % i))).astype(np.uint8)
    return a

def flow(a, b):
    g1 = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(b[..., :3], cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    return dis.calc(g1, g2, None), dis.calc(g2, g1, None)

def bw(fx, fy):
    return cv2.GaussianBlur(fx, (0, 0), 2), cv2.GaussianBlur(fy, (0, 0), 2)

def warp(img, fx, fy, t):
    H, W = fx.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xx + t * fx; mapy = yy + t * fy
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def mid(a, b, t):
    fab, fba = flow(a, b)
    fx, fy = fab[..., 0], fab[..., 1]
    gx, gy = fba[..., 0], fba[..., 1]
    fx, fy = bw(fx, fy); gx, gy = bw(gx, gy)
    fab = (fx, fy); fba = (gx, gy)
    aw = warp(a.astype(np.float32), fab[0], fab[1], t)
    bw_ = warp(b.astype(np.float32), fba[0], fba[1], 1 - t)
    # 前后一致性
    x, y = np.meshgrid(np.arange(384, dtype=np.float32), np.arange(384, dtype=np.float32))
    mabx = x + fab[0]; maby = y + fab[1]
    back = cv2.remap(fba[0], mabx, maby, cv2.INTER_LINEAR)
    backy = cv2.remap(fba[1], mabx, maby, cv2.INTER_LINEAR)
    err = np.abs(back + fab[0]) + np.abs(backy + fab[1])
    err = cv2.GaussianBlur(err, (0, 0), 3)
    consist = np.clip(1.0 - err / 3.0, 0, 1)[..., None]
    w = (1 - t) * np.ones_like(consist)
    out = consist * (w * aw + (1 - w) * bw_) + (1 - consist) * (aw if t < 0.5 else bw_)
    return np.clip(out, 0, 255).astype(np.uint8)

for i, j, tag in ((19, 20, "19_20"), (44, 45, "44_45")):
    a, b = rd(i), rd(j)
    m1 = mid(a, b, 0.33); m2 = mid(a, b, 0.67)
    row = np.hstack([a, m1, m2, b])
    Image.fromarray(row).save(os.path.join(F, "_interp_%s.png" % tag))
print("ok")
