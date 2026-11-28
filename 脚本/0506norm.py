import os, glob, json
import numpy as np
from PIL import Image
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
CANVAS, GROUND_Y, CENTER_X = 384, 366, 192
ACTS = ["05thinking", "06busy"]


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


def cleanup_bottom(a, cell=448):
    m = a[..., 3] > 20
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    kill = np.zeros_like(m)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.max() > 0.90 * cell and (ys.max() - ys.min()) < 52 and sizes[i - 1] < 0.045 * cell * cell:
            kill |= lab == i
    if kill.any():
        a[kill] = 0
    return a


def matte(rgb, lo=12.0, span=34.0, klow=0.55):
    L = rgb.max(axis=2)
    al = np.clip((L - lo) / span, 0.0, 1.0)
    al = ndimage.gaussian_filter(al, 0.6)
    al[al < 0.06] = 0.0
    color = np.clip(rgb / np.maximum(al, klow)[..., None], 0, 255)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = color.astype(np.uint8)
    out[..., 3] = (al * 255).astype(np.uint8)
    return cleanup_bottom(out)


def body_of(a):
    m = a >= 110
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def torso_cx(a):
    m = body_of(a[..., 3])
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    band = (ys >= y0 + 0.55 * h) & (ys <= y0 + 0.88 * h)
    wgt = a[..., 3][ys[band], xs[band]] / 255.0
    if wgt.sum() < 1:
        return None
    return float((xs[band] * wgt).sum() / wgt.sum())


def foot_y(a):
    m = body_of(a[..., 3])
    ys, xs = np.where(m)
    return int(ys.max())


def refine(a):
    al = a[..., 3].astype(np.float32) / 255.0
    edge = np.clip((al - 0.35) / 0.4, 0, 1)
    a2 = a.copy()
    a2[..., 3] = (np.where(al > 0.02, edge, 0.0) * 255).astype(np.uint8)
    a2[a2[..., 3] > 0, 3] = np.maximum(a2[a2[..., 3] > 0, 3], 26)
    a2[a2[..., 3] < 40] = 0
    return a2


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
    out[..., :3] = np.clip(rgb + 1.4 * (rgb - bl), 0, 255)
    return out


if __name__ == "__main__":
    report = {}
    for act in ACTS:
        cs = cells_of(act)
        raws = [matte(c.copy()) for c in cs]
        hs2 = []
        for r in raws:
            m = body_of(r[..., 3])
            ys = np.where(m)[0]
            hs2.append(int(ys.max() - ys.min() + 1))
        sc = 306.0 / float(np.median(hs2))
        tcx = [torso_cx(r) for r in raws]
        tcx_v = np.array([x for x in tcx if x is not None])
        fx = float(np.median(tcx_v))
        fy = float(np.median([foot_y(r) for r in raws]))
        ox = int(round(CENTER_X - fx * sc))
        oy = int(round(GROUND_Y - fy * sc))
        od = os.path.join(OUT, act, "_norm")
        os.makedirs(od, exist_ok=True)
        print("==", act, "frames", len(raws), "scale", round(sc, 4),
              "torso_cx median", round(fx, 1), "foot_y median", fy, "offset", ox, oy)
        print("   torso_cx:", [None if x is None else int(x) for x in tcx])
        rows = []
        for i, a in enumerate(raws):
            out = norm(a, sc, ox, oy)
            im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")
            im.save(os.path.join(od, "norm_%02d.png" % (i + 1)))
            al = np.array(im)[..., 3]
            ys, xs = np.where(al > 110)
            rows.append(dict(i=i + 1, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
        report[act] = dict(scale=sc, offset=[ox, oy], frames=len(raws), bbox=rows)
        tight = [r for r in rows if min(r["bbox"][0], r["bbox"][1], 383 - r["bbox"][2], 383 - r["bbox"][3]) < 3]
        print("   贴边帧:", tight[:5], "共", len(tight))
    with open(os.path.join(OUT, "_diag0506", "norm_report.json"), "w") as fh:
        json.dump(report, fh)
