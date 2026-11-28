import os, glob
import numpy as np
from PIL import Image
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审", "_diag0506")


def cells_of(act):
    cs = []
    for p in sorted(glob.glob(os.path.join(G, "素材", act, "part*.webp"))):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        cw, ch = w // 4, h // 3
        for r in range(3):
            for c in range(4):
                cs.append(np.asarray(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)), dtype=np.float32))
    return cs


def matte(rgb, lo=12.0, span=34.0, klow=0.55):
    L = rgb.max(axis=2)
    al = np.clip((L - lo) / span, 0.0, 1.0)
    al = ndimage.gaussian_filter(al, 0.6)
    al[al < 0.06] = 0.0
    color = np.clip(rgb / np.maximum(al, klow)[..., None], 0, 255)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = color.astype(np.uint8)
    out[..., 3] = (al * 255).astype(np.uint8)
    return out


def body_bbox(a):
    m = a >= 110
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    body = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(body)
    return xs.min(), xs.max(), ys.min(), ys.max(), int(sizes.max())


def face_cx(a):
    rgb = a[..., :3].astype(np.int16)
    al = a[..., 3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    skin = (r > 228) & (g > 198) & (b > 188) & (r >= g) & (g >= b) & (al > 120)
    h = a.shape[0]
    skin[: int(h * 0.15)] = False
    skin[int(h * 0.75):] = False
    skin = ndimage.binary_opening(skin, np.ones((3, 3)))
    skin = ndimage.binary_fill_holes(skin)
    lab, n = ndimage.label(skin)
    if n == 0:
        return None
    sizes = ndimage.sum(skin, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    if sizes.max() < 900:
        return None
    ys, xs = np.where(m)
    return (xs.min() + xs.max()) / 2.0


for act in ("05thinking", "06busy"):
    cs = cells_of(act)
    hs, fcs, bottoms, areas, cx, wids = [], [], [], [], [], []
    for i, c in enumerate(cs):
        a = matte(c.copy())
        bb = body_bbox(a[..., 3])
        hs.append(bb[3] - bb[2] + 1)
        bottoms.append(bb[3])
        areas.append(bb[4])
        fcs.append(face_cx(a))
        cx.append((bb[0] + bb[1]) / 2.0)
        wids.append(bb[1] - bb[0] + 1)
    hs = np.array(hs)
    print("==", act)
    print(" h:", [int(x) for x in hs])
    print(" median h", np.median(hs), " min", hs.min(), " max", hs.max())
    print(" scale(306/median)", round(306 / np.median(hs), 4))
    print(" face_cx:", [None if x is None else int(x) for x in fcs])
    print(" bottom y:", [int(x) for x in bottoms])
    print(" body area:", [int(x) for x in areas])
    cxs = np.array(cx)
    print(" bbox cx:", [int(x) for x in cxs])
    print(" bb cx median %.1f  范围 %.1f~%.1f  抖动(相邻均值) %.2f" % (
        np.median(cxs), cxs.min(), cxs.max(), np.abs(np.diff(cxs)).mean()))
    print(" bbox w:", [int(x) for x in wids])
