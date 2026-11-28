import os, json
import numpy as np, cv2
from PIL import Image
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
N = os.path.join(F, "_norm")
SEQ = os.path.join(F, "_seq")
os.makedirs(SEQ, exist_ok=True)
S = 384
DUR_KEY = 140
DUR_MID = 95

def rd(i):
    return np.asarray(Image.open(os.path.join(N, "f%02d.png" % i))).astype(np.float32)

def calc_flow(a, b):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    g1 = cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(b[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fab = dis.calc(g1, g2, None); fba = dis.calc(g2, g1, None)
    return fab[..., 0], fab[..., 1], fba[..., 0], fba[..., 1]

def warp(img, fx, fy, t):
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    return cv2.remap(img, xx + t * fx, yy + t * fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def mid(a, b, t, flow):
    fx, fy, gx, gy = flow
    fx, fy = cv2.GaussianBlur(fx, (0, 0), 2), cv2.GaussianBlur(fy, (0, 0), 2)
    gx, gy = cv2.GaussianBlur(gx, (0, 0), 2), cv2.GaussianBlur(gy, (0, 0), 2)
    aw = warp(a, fx, fy, t); bw_ = warp(b, gx, gy, 1 - t)
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    bx = cv2.remap(gx, xx + fx, yy + fy, cv2.INTER_LINEAR)
    by = cv2.remap(gy, xx + fx, yy + fy, cv2.INTER_LINEAR)
    err = cv2.GaussianBlur(np.abs(bx + fx) + np.abs(by + fy), (0, 0), 4)
    cons = np.clip(1 - err / 2.2, 0, 1)
    hard = 1 - cons
    aa = aw[..., 3] / 255.0; ba = bw_[..., 3] / 255.0
    wA = (1 - t) * cons; wB = t * cons
    if t < 0.5:
        wA = wA + hard
        al = (cons * (aa * (1 - t) + ba * t) + hard * aa) * 255
    else:
        wB = wB + hard
        al = (cons * (aa * (1 - t) + ba * t) + hard * ba) * 255
    den = (wA + wB)[..., None] + 1e-4
    col = (aw[..., :3] * wA[..., None] + bw_[..., :3] * wB[..., None]) / den
    return np.clip(np.concatenate([col, al[..., None]], axis=2), 0, 255).astype(np.uint8)

frames = []
for i in range(1, 49):
    a = rd(i)
    frames.append(a.astype(np.uint8))
    b = rd(i % 48 + 1)
    fl = calc_flow(a, b)
    for t in (1 / 3, 2 / 3):
        frames.append(mid(a, b, t, fl))
print("total frames", len(frames))

for k, fr in enumerate(frames, 1):
    Image.fromarray(fr, "RGBA").save(os.path.join(SEQ, "s%03d.png" % k))

imgs = [Image.open(os.path.join(SEQ, "s%03d.png" % k)) for k in range(1, len(frames) + 1)]
anim = os.path.join(F, "06working_busy（忙碌_动画）.webp")
durs = []
for _ in range(len(imgs) // 3):
    durs += [DUR_KEY, DUR_MID, DUR_MID]
imgs[0].save(anim, save_all=True, append_images=imgs[1:], duration=durs, loop=0, quality=95, method=4, alpha_quality=100)

cols = 8
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new("RGBA", (S * cols, S * rows), (0, 0, 0, 0))
for k, im in enumerate(imgs):
    sheet.paste(im, ((k % cols) * S, (k // cols) * S))
sheet.save(os.path.join(F, "06working_busy_sheet（忙碌_图集）.webp"), quality=95, method=4, alpha_quality=100)

meta = dict(frames=len(imgs), duration_ms=dict(key=DUR_KEY, mid=DUR_MID), total_ms=sum(durs), size=[S, S],
            sheet=[S * cols, S * rows], cols=cols, rows=rows, canvas=S,
            rhythm="非匀速：关键姿态 140ms、中间光流帧 95ms", note="腿脚按 c46 腿部原画补全（part3 #25–36 / part4 #37–44 及 part2 边缘），48 姿态保留，每对姿态插 2 帧")
json.dump(meta, open(os.path.join(F, "06busy_meta.json"), "w"), ensure_ascii=False, indent=1)
print("anim", os.path.getsize(anim) // 1024, "KB")
print("sheet", os.path.getsize(os.path.join(F, "06working_busy_sheet（忙碌_图集）.webp")) // 1024, "KB", sheet.size)
