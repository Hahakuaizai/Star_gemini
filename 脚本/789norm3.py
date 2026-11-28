import os
import numpy as np
from PIL import Image
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
CANVAS = 384
GROUND_Y = 366
CENTER_X = 192
N = {"07sleep_start": 36, "08sleep_loop": 48, "09sleep_wake": 36}
REF_H = 303.0


def body_bbox(a):
    m = a >= 110
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    body = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(body)
    return xs.min(), xs.max(), ys.min(), ys.max()


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


def fill_outliers(vals):
    v = np.array([np.nan if x is None else x for x in vals], dtype=np.float64)
    idx = np.arange(len(v))
    good = ~np.isnan(v)
    if good.sum() < 2:
        v[:] = CENTER_X
        return v
    v[~good] = np.interp(idx[~good], idx[good], v[good])
    return ndimage.median_filter(v, size=3, mode="nearest")


def norm(a, sc, ox, oy):
    img = Image.fromarray(a, "RGBA")
    nw = max(2, int(round(img.width * sc)))
    arr = np.array(img.resize((nw, nw), Image.LANCZOS), dtype=np.float32)
    out = np.zeros((CANVAS, CANVAS, 4), dtype=np.float32)
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw = min(nw - sx0, CANVAS - dx0)
    ch = min(nw - sy0, CANVAS - dy0)
    out[dy0:dy0 + ch, dx0:dx0 + cw] = arr[sy0:sy0 + ch, sx0:sx0 + cw]
    rgb = out[..., :3]
    bl = ndimage.gaussian_filter(rgb, sigma=(1.0, 1.0, 0))
    out[..., :3] = np.clip(rgb + 1.5 * (rgb - bl), 0, 255)
    return out


if __name__ == "__main__":
    raws, hs = {}, {}
    for d, n in N.items():
        rs, hh = [], []
        for i in range(n):
            a = np.array(Image.open(os.path.join(OUT, d, "_raw", "raw_%02d.png" % (i + 1))).convert("RGBA"))
            rs.append(a)
            bb = body_bbox(a[..., 3])
            hh.append(bb[3] - bb[2] + 1)
        raws[d], hs[d] = rs, np.array(hh, dtype=np.float64)
    s07 = 306.0 / np.median(hs["07sleep_start"][:12])
    s09 = 306.0 / np.median(hs["09sleep_wake"][-8:])
    s08 = hs["07sleep_start"][-1] * s07 / hs["08sleep_loop"][0]
    S = {"07sleep_start": s07, "08sleep_loop": s08, "09sleep_wake": s09}
    print("base scales", {k: round(v, 4) for k, v in S.items()})

    for d, n in N.items():
        od = os.path.join(OUT, d, "_norm")
        os.makedirs(od, exist_ok=True)
        sc = []
        for i in range(n):
            H = min(max(hs[d][i] * S[d], REF_H * 0.95), REF_H * 1.05)
            sc.append(H / hs[d][i])
        sc = np.array(sc)
        for i in range(1, n):
            sc[i] = min(max(sc[i], sc[i - 1] * 0.978), sc[i - 1] * 1.022)
        sc = ndimage.uniform_filter1d(sc, 3, mode="nearest")
        fcs = fill_outliers([face_cx(a) for a in raws[d]])
        print("==", d, "face cx", [int(x) for x in fcs])
        for i, a in enumerate(raws[d]):
            bb = body_bbox(a[..., 3])
            nw = int(round(a.shape[1] * sc[i]))
            x0f = bb[0] * sc[i]
            ox = int(round(CENTER_X - fcs[i] * sc[i]))
            oy = int(round(GROUND_Y - bb[3] * sc[i]))
            out = norm(a, sc[i], ox, oy)
            m = out[..., 3] > 110
            ys, xs = np.where(m)
            fc = face_cx(out)
            Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA").save(
                os.path.join(od, "norm_%02d.png" % (i + 1)))
            print(i + 1, "scale", round(float(sc[i]), 3), "facecx", None if fc is None else int(fc),
                  "bbox", int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))

