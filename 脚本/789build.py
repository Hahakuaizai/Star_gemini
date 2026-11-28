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
    ("07sleep_start", "07sleep_start\uff08\u5165\u7761_\u52a8\u753b\uff09.webp",
     "07sleep_start_sheet\uff08\u5165\u7761_\u56fe\u96c6\uff09.webp", 8.8),
    ("08sleep_loop", "08sleep_loop\uff08\u7761\u7720\u5faa\u73af_\u52a8\u753b\uff09.webp",
     "08sleep_loop_sheet\uff08\u7761\u7720\u5faa\u73af_\u56fe\u96c6\uff09.webp", 9.0),
    ("09sleep_wake", "09sleep_wake\uff08\u7761\u9192_\u52a8\u753b\uff09.webp",
     "09sleep_wake_sheet\uff08\u7761\u9192_\u56fe\u96c6\uff09.webp", 8.0),
]


def to_np(img):
    return np.array(img.convert("RGBA"), dtype=np.float32)


def to_img(a):
    a = np.clip(a, 0, 255).astype(np.uint8)
    a[a[..., 3] < 4] = 0
    return Image.fromarray(a, "RGBA")


def premul(a):
    o = a.copy()
    o[..., :3] = a[..., :3] * (a[..., 3:4] / 255.0)
    return o


def unpremul(a):
    o = a.copy()
    al = a[..., 3:4] / 255.0
    o[..., :3] = np.where(al > 1e-4, a[..., :3] / np.maximum(al, 1e-4), 0.0)
    return o


def body_weight(a):
    m = a[..., 3] > 128
    m = ndimage.binary_fill_holes(m)
    return ndimage.gaussian_filter(m.astype(np.float32), 1.4)


def mid_body(pa, pb, t, wa, wb, kb=1.0):
    al1 = (pa[..., 3:4] / 255.0) * wa[..., None]
    al2 = (pb[..., 3:4] / 255.0) * wb[..., None] * kb
    alpha = np.maximum(al1, al2)
    rgb = (pa[..., :3] * al1 * (1.0 - t) + pb[..., :3] * al2 * t) / np.maximum(al1 * (1.0 - t) + al2 * t, 1e-4)
    out = np.zeros_like(pa)
    out[..., :3] = rgb
    out[..., 3:4] = alpha * 255.0
    return out


def mid_particle(pa, pb, t, wa, wb):
    A = pa.copy()
    A[..., 3] *= (1.0 - wa)
    B = pb.copy()
    B[..., 3] *= (1.0 - wb)
    return unpremul(premul(A) * (1.0 - t) + premul(B) * t)


def combine(body, part):
    out = body.copy()
    a1 = body[..., 3:4] / 255.0
    a2 = part[..., 3:4] / 255.0
    out[..., :3] = body[..., :3] * a1 + part[..., :3] * a2 * (1.0 - a1)
    out[..., 3:4] = (a1 + a2 * (1.0 - a1)) * 255.0
    return out


def interp(a, b, t, wa, wb, kb=1.0):
    return combine(mid_body(a, b, t, wa, wb, kb), mid_particle(a, b, t, wa, wb))


def alpha_diff(a, b):
    ala = a[..., 3] / 255.0
    alb = b[..., 3] / 255.0
    return float(np.abs(ala - alb).mean() * 100)


def pair_shift(a, b):
    ala = a[..., 3] / 255.0
    alb = b[..., 3] / 255.0
    try:
        (dx, dy), _ = cv2.phaseCorrelate(ala.astype(np.float32), alb.astype(np.float32))
    except Exception:
        dx, dy = 0.0, 0.0
    moved = cv2.warpAffine(alb, np.float32([[1, 0, -dx], [0, 1, -dy]]), (CANVAS, CANVAS),
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return float(np.abs(ala - moved).mean() * 100), dx, dy


def pair_diff(a, b):
    return pair_shift(a, b)[0]


def warp(a, dx, dy):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (CANVAS, CANVAS),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def interp_at(a, b, t, wa, wb, dx, dy, kb=1.0):
    if kb <= 0.0:
        if t < 0.5:
            return warp(a, t * dx, t * dy)
        return warp(b, -(1.0 - t) * dx, -(1.0 - t) * dy)
    at = warp(a, t * dx, t * dy)
    bt = warp(b, -(1.0 - t) * dx, -(1.0 - t) * dy)
    return interp(at, bt, t, wa, wb, kb)


def dur_profile(name, n):
    if name == "07sleep_start":
        def f(i):
            p = i / (n - 1.0)
            return 105 + 85 * p
        return [int(round(f(i))) for i in range(n)]
    if name == "08sleep_loop":
        return [100] * n
    def g(i):
        p = i / (n - 1.0)
        return 155 - 55 * p
    return [int(round(g(i))) for i in range(n)]


def build_dir(d, anim_name, sheet_name, target_sec):
    dimg = os.path.join(OUT, d)
    n0 = {"07sleep_start": 36, "08sleep_loop": 48, "09sleep_wake": 36}[d]
    frames = [to_np(Image.open(os.path.join(dimg, "_norm", "norm_%02d.png" % (i + 1))))
              for i in range(n0)]
    ws = [body_weight(f) for f in frames]
    root = frames
    diffs = [pair_diff(frames[i], frames[i + 1]) for i in range(n0 - 1)]

    seq, seg_diff = [], []
    for i in range(n0 - 1):
        a, b = frames[i], frames[i + 1]
        diff = diffs[i]
        _, sdx, sdy = pair_shift(a, b)
        if diff <= 2.5:
            use = 3
            kb = 1.0
        elif diff <= 5.0:
            use = 2
            kb = 0.7
        else:
            use = 3
            kb = 0.0
        sub = max(2, int(round(100.0 / use)))
        seq.append(a)
        seg_diff.append(diff)
        for k in range(1, use):
            seq.append(interp_at(a, b, k / float(use), ws[i], ws[i + 1], sdx, sdy, kb))
            seg_diff.append(diff)
    seq.append(frames[-1])
    seg_diff.append(diffs[-1])

    if d == "08sleep_loop":
        f48, f1 = frames[-1], frames[0]
        w48, w1 = ws[-1], ws[0]
        seq.append(interp(f48, f1, 0.5, w48, w1))
        seg_diff.append(pair_diff(f48, f1))
        seq.append(f1)
        seg_diff.append(0.0)
    if d == "07sleep_start":
        nxt = to_np(Image.open(os.path.join(OUT, "08sleep_loop", "_norm", "norm_01.png")))
        wn = body_weight(nxt)
        seq.append(interp(seq[-1], nxt, 0.34, ws[-1], wn))
        seg_diff.append(pair_diff(seq[-2], nxt))
        seq.append(interp(seq[-1], nxt, 0.67, ws[-1], wn))
        seg_diff.append(pair_diff(seq[-3], nxt))
        seq.append(nxt)
        seg_diff.append(0.0)
    if d == "09sleep_wake":
        prev = to_np(Image.open(os.path.join(OUT, "08sleep_loop", "_norm", "norm_01.png")))
        wp = body_weight(prev)
        head = [prev, interp(prev, frames[0], 0.5, wp, ws[0])]
        seq = head + seq
        seg_diff = [0.0, pair_diff(prev, frames[0])] + seg_diff

    total = int(target_sec * 1000)
    base = np.array(dur_profile(d, len(seq)), dtype=np.float64)
    base = base / base.sum() * total
    durs = [int(max(55, round(x))) for x in base]

    n = len(seq)
    rows = (n + COLS - 1) // COLS
    sheet = Image.new("RGBA", (COLS * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, f in enumerate(seq):
        fi = to_img(f)
        r, c = divmod(i, COLS)
        sheet.paste(fi, (c * CANVAS, r * CANVAS), fi)
        seq[i] = fi
    anim = os.path.join(dimg, anim_name)
    sheetp = os.path.join(dimg, sheet_name)
    seq[0].save(anim, format="WEBP", save_all=True, append_images=seq[1:], duration=durs, loop=0,
                lossless=False, quality=95, alpha_quality=100, disposal=2, minimize_size=False)
    sheet.save(sheetp, format="WEBP", quality=95, alpha_quality=100, method=6)
    base_name = d
    for tone, bg in (("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255))):
        canvas = Image.new("RGBA", sheet.size, bg)
        canvas.alpha_composite(sheet)
        out = canvas.convert("RGB")
        dr = ImageDraw.Draw(out)
        for i in range(n):
            r, c = divmod(i, COLS)
            dr.text((c * CANVAS + 8, r * CANVAS + 6), str(i + 1), fill=(210, 30, 50))
        out.save(os.path.join(dimg, base_name + "_%s.png" % tone))
    with open(os.path.join(dimg, base_name + "_meta.json"), "w") as fh:
        json.dump({"cols": COLS, "rows": rows, "frames": n, "cellSize": CANVAS,
                   "total_ms": sum(durs), "durations": durs}, fh)
    print(d, "frames", n, "rows", rows, "total_ms", sum(durs),
          "diffs>12", sum(1 for x in seg_diff if x > 12),
          "max", round(max(seg_diff), 1))
    return anim, sheetp


if __name__ == "__main__":
    for d, anim, sheet, sec in ACTS:
        build_dir(d, anim, sheet, sec)

