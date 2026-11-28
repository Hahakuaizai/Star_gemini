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
N0 = 48

ACTS = [
    ("05thinking", "05thinking（思考_动画）.webp", "05thinking_sheet（思考_图集）.webp", 10.5, "think"),
    ("06busy", "06working_busy（忙碌_动画）.webp", "06working_busy_sheet（忙碌_图集）.webp", 9.0, "busy"),
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
    if np.isscalar(kb) and kb < 0:
        al1 = (pa[..., 3:4] / 255.0) * wa[..., None]
        al2 = (pb[..., 3:4] / 255.0) * wb[..., None]
        alpha = np.maximum(al1, al2)
        rgb = np.where(al1 >= al2, pa[..., :3], pb[..., :3])
        out = np.zeros_like(pa)
        out[..., :3] = rgb
        out[..., 3:4] = alpha * 255.0
        return out
    kk = kb if np.isscalar(kb) else kb[..., None]
    al1 = (pa[..., 3:4] / 255.0) * wa[..., None]
    al2 = (pb[..., 3:4] / 255.0) * wb[..., None] * kk
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
    if t < 0.5:
        return A
    return B


def combine(body, part):
    out = body.copy()
    a1 = body[..., 3:4] / 255.0
    a2 = part[..., 3:4] / 255.0
    out[..., :3] = body[..., :3] * a1 + part[..., :3] * a2 * (1.0 - a1)
    out[..., 3:4] = (a1 + a2 * (1.0 - a1)) * 255.0
    return out


def interp(a, b, t, wa, wb, kb=1.0):
    return combine(mid_body(a, b, t, wa, wb, kb), mid_particle(a, b, t, wa, wb))


def pair_shift(a, b, cap=3.0):
    ala = a[..., 3] / 255.0
    alb = b[..., 3] / 255.0
    r0, r1 = int(0.62 * CANVAS), int(0.95 * CANVAS)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(ala[r0:r1].astype(np.float32), alb[r0:r1].astype(np.float32))
    except Exception:
        dx, dy = 0.0, 0.0
    if abs(dx) > cap or abs(dy) > cap:
        dx, dy = 0.0, 0.0
    moved = cv2.warpAffine(alb, np.float32([[1, 0, -dx], [0, 1, -dy]]), (CANVAS, CANVAS),
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    mb = cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), (CANVAS, CANVAS),
                        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    ad = float(np.abs(ala - moved).mean() * 100)
    cd = float(np.abs(premul(a) - premul(mb)).mean() / 255.0 * 100)
    return max(ad * 1.6, cd), dx, dy


def warp(a, dx, dy):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (CANVAS, CANVAS),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def interp_at(a, b, t, wa, wb, dx, dy, kb=1.0):
    if np.isscalar(kb) and kb <= 0.0:
        if t < 0.5:
            return warp(a, t * dx, t * dy)
        return warp(b, -(1.0 - t) * dx, -(1.0 - t) * dy)
    at = warp(a, t * dx, t * dy)
    bt = warp(b, -(1.0 - t) * dx, -(1.0 - t) * dy)
    return interp(at, bt, t, wa, wb, kb)


def dur_profile(kind, n):
    if kind == "think":
        def f(i):
            p = i / (n - 1.0)
            if p < 0.10:
                return 138
            if p < 0.60:
                return 76
            if p < 0.76:
                return 92
            return 104
    else:
        def f(i):
            p = i / (n - 1.0)
            if p < 0.16:
                return 84
            if p < 0.52:
                return 62
            if p < 0.78:
                return 70
            return 96
    return [int(round(f(i))) for i in range(n)]


def build(d, anim_name, sheet_name, target_sec, kind):
    dimg = os.path.join(OUT, d)
    frames = [to_np(Image.open(os.path.join(dimg, "_norm", "norm_%02d.png" % (i + 1)))) for i in range(N0)]
    ws = [body_weight(f) for f in frames]
    seq, seg_diff = [], []
    for i in range(N0 - 1):
        a, b = frames[i], frames[i + 1]
        diff, dx, dy = pair_shift(a, b)
        if diff <= 3.0:
            use, kb = 3, 1.0
        elif diff <= 6.0:
            use, kb = 3, 0.7
        else:
            use = 3
            bm = warp(b, -dx, -dy)
            mm = np.abs(premul(a) - premul(bm)).max(axis=2)
            mm = ndimage.gaussian_filter(mm, 2.0)
            kb = np.clip(1.0 - (mm - 14.0) / 26.0, 0.0, 1.0)
        seq.append(a)
        seg_diff.append(diff)
        for k in range(1, use):
            seq.append(interp_at(a, b, k / float(use), ws[i], ws[i + 1], dx, dy, kb))
            seg_diff.append(diff)
    seq.append(frames[-1])
    seg_diff.append(pair_shift(frames[-2], frames[-1])[0])

    tail, tdx, tdy = pair_shift(frames[-1], frames[0])
    if tail > 7.0:
        for t in (0.34, 0.67):
            seq.append(interp_at(frames[-1], frames[0], t, ws[-1], ws[0], tdx, tdy, 0.6))
            seg_diff.append(tail / 3.0)
    seq.append(frames[0])
    seg_diff.append(0.0)

    total = int(target_sec * 1000)
    base = np.array(dur_profile(kind, len(seq)), dtype=np.float64)
    base = base / base.sum() * total
    durs = [int(max(46, round(x))) for x in base]

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
    for tone, bg in (("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255))):
        canvas = Image.new("RGBA", sheet.size, bg)
        canvas.alpha_composite(sheet)
        out = canvas.convert("RGB")
        dr = ImageDraw.Draw(out)
        for i in range(n):
            r, c = divmod(i, COLS)
            dr.text((c * CANVAS + 8, r * CANVAS + 6), str(i + 1), fill=(200, 30, 50))
        out.save(os.path.join(dimg, d + "_%s.png" % tone))
        out.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(
            os.path.join(dimg, d + "_%s_half.png" % tone))
    with open(os.path.join(dimg, d + "_meta.json"), "w") as fh:
        json.dump({"cols": COLS, "rows": rows, "frames": n, "cellSize": CANVAS,
                   "total_ms": sum(durs), "durations": durs}, fh)
    print(d, "frames", n, "rows", rows, "total_ms", sum(durs), "grid %dx%d" % (COLS * CANVAS, rows * CANVAS),
          "diff>5:", sum(1 for x in seg_diff if x > 5), "max", round(max(seg_diff), 1))
    return anim


if __name__ == "__main__":
    for d, an, sn, sec, kind in ACTS:
        build(d, an, sn, sec, kind)
