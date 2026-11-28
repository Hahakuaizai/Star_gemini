import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
CANVAS = 384
COLS = 8

ACTS = [
    ("05thinking", "05thinking（思考_动画）.webp",
     "05thinking_sheet（思考_图集）.webp", 10.5, "think"),
    ("06busy", "06working_busy（忙碌_动画）.webp",
     "06working_busy_sheet（忙碌_图集）.webp", 9.0, "busy"),
]

DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
DIS.setUseSpatialPropagation(True)


def load(d, i):
    return np.array(Image.open(os.path.join(OUT, d, "_norm", "norm_%02d.png" % i)).convert("RGBA"),
                    dtype=np.float32)


def premul(a):
    o = a.copy()
    o[..., :3] = a[..., :3] * (a[..., 3:4] / 255.0)
    return o


def unpremul(a):
    o = a.copy()
    al = a[..., 3:4] / 255.0
    o[..., :3] = np.clip(np.where(al > 1e-4, a[..., :3] / np.maximum(al, 1e-4), 0.0), 0, 255)
    return o


def gray(a):
    g = a[..., 3:4] / 255.0 * 255.0
    lum = 0.3 * a[..., 0:1] + 0.59 * a[..., 1:2] + 0.11 * a[..., 2:3]
    return np.clip(0.55 * g + 0.45 * lum, 0, 255).astype(np.uint8)[..., 0]


def warp_flow(img, fl, t):
    h, w = fl.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = x + fl[..., 0] * t
    my = y + fl[..., 1] * t
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def pair_diff(a, b):
    pa, pb = premul(a), premul(b)
    return float(np.abs(pa - pb).mean() / 255.0 * 100)


def clean(a, min_area=14):
    al = a[..., 3]
    m = al > 12
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        for i in range(1, n + 1):
            if sizes[i - 1] < min_area:
                a[lab == i] = 0
    a[a[..., 3] < 10] = 0
    return a


def mid(a, b, t, fab, fba):
    if t < 0.5:
        out = unpremul(warp_flow(premul(a), fab, t))
    else:
        out = unpremul(warp_flow(premul(b), fba, 1.0 - t))
    return clean(np.clip(out, 0, 255))


def dur_profile(kind, n):
    if kind == "think":
        def f(i):
            p = i / (n - 1.0)
            if p < 0.12:
                return 150
            if p < 0.55:
                return 74
            if p < 0.72:
                return 96
            return 108
    else:
        def f(i):
            p = i / (n - 1.0)
            if p < 0.18:
                return 78
            if p < 0.55:
                return 58
            if p < 0.80:
                return 66
            return 92
    return [int(round(f(i))) for i in range(n)]


def to_img(a):
    a = np.clip(a, 0, 255).astype(np.uint8)
    a[a[..., 3] < 8] = 0
    return Image.fromarray(a, "RGBA")


def build(d, anim_name, sheet_name, target_sec, kind):
    n0 = 48
    frames = [load(d, i) for i in range(1, n0 + 1)]
    diffs = [pair_diff(frames[i], frames[i + 1]) for i in range(n0 - 1)]
    seq = [to_img(frames[0])]
    seg = []
    for i in range(n0 - 1):
        a, b = frames[i], frames[i + 1]
        fab = DIS.calc(gray(a), gray(b), None)
        fba = DIS.calc(gray(b), gray(a), None)
        df = diffs[i]
        if df <= 5.0:
            use = 2
        elif df <= 12.0:
            use = 3
        else:
            use = 3
        for k in range(1, use):
            seq.append(to_img(mid(a, b, k / float(use), fab, fba)))
        seq.append(to_img(b))
        seg.extend([df] * use)
    tail_gap = pair_diff(frames[-1], frames[0])
    if tail_gap > 7.0:
        tfab = DIS.calc(gray(frames[-1]), gray(frames[0]), None)
        tfba = DIS.calc(gray(frames[0]), gray(frames[-1]), None)
        for t in (0.34, 0.66):
            seq.append(to_img(mid(frames[-1], frames[0], t, tfab, tfba)))
            seg.append(tail_gap / 3.0)
    seq.append(to_img(frames[0]))
    seg.append(tail_gap)

    total = int(target_sec * 1000)
    base = np.array(dur_profile(kind, len(seq)), dtype=np.float64)
    base = base / base.sum() * total
    durs = [int(max(46, round(x))) for x in base]

    n = len(seq)
    rows = (n + COLS - 1) // COLS
    sheet = Image.new("RGBA", (COLS * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, f in enumerate(seq):
        r, c = divmod(i, COLS)
        sheet.paste(f, (c * CANVAS, r * CANVAS), f)
    anim = os.path.join(OUT, d, anim_name)
    sheetp = os.path.join(OUT, d, sheet_name)
    seq[0].save(anim, format="WEBP", save_all=True, append_images=seq[1:], duration=durs, loop=0,
                lossless=False, quality=95, alpha_quality=100, disposal=2, minimize_size=False)
    sheet.save(sheetp, format="WEBP", quality=95, alpha_quality=100, method=6)
    for tone, bg in (("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255))):
        canvas = Image.new("RGBA", sheet.size, bg)
        canvas.alpha_composite(sheet)
        out = canvas.convert("RGB")
        dr = ImageDraw.Draw(out)
        for i in range(n):
            r, c = divmod(i, COLS)
            dr.text((c * CANVAS + 8, r * CANVAS + 6), str(i + 1), fill=(200, 30, 50))
        out.save(os.path.join(OUT, d, d + "_%s.png" % tone))
        out.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(
            os.path.join(OUT, d, d + "_%s_half.png" % tone))
    with open(os.path.join(OUT, d, d + "_meta.json"), "w") as fh:
        json.dump({"cols": COLS, "rows": rows, "frames": n, "cellSize": CANVAS,
                   "total_ms": sum(durs), "durations": durs}, fh)
    print(d, "frames", n, "rows", rows, "total_ms", sum(durs),
          "grid", "%.0fx%.0f" % (COLS * CANVAS, rows * CANVAS),
          "diff中位", round(float(np.median(diffs)), 1), "尾差", round(tail_gap, 1))
    return anim


if __name__ == "__main__":
    for d, an, sn, sec, kind in ACTS:
        build(d, an, sn, sec, kind)
