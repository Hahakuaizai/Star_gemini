import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label, map_coordinates, zoom

G = r"C:\Users\30894\Desktop\workspace\帧动画\gemini"
SRC_DIR = os.path.join(G, "待审", "04roam")
SRC_SHEET = os.path.join(SRC_DIR, "36帧基准", "04roam_sheet（踏步_图集）.webp")
OUT_DIR = os.path.join(SRC_DIR)
CANVAS = 384
COLS, ROWS = 8, 6
TOTAL_MS = 7000
TARGET = 48
DROP_COUNT = 4
MIN_FRAME_MS = 95


def load36():
    sheet = Image.open(SRC_SHEET).convert("RGBA")
    out = []
    for i in range(36):
        r, c = divmod(i, 6)
        out.append(sheet.crop((c * CANVAS, r * CANVAS, c * CANVAS + CANVAS, r * CANVAS + CANVAS)))
    return out


def body_metrics(img):
    a = np.array(img, dtype=np.uint8)
    al = a[:, :, 3]
    lab, n = label(al > 60)
    if n == 0:
        return 0, 0, 0, 0, 0
    s = np.bincount(lab.ravel())
    s[0] = 0
    m = lab == int(s.argmax())
    ys, xs = np.where(m)
    return int(ys.max() - ys.min() + 1), int(ys.max()), int(xs.max() - xs.min() + 1), int((al > 20).sum()), int(xs.min() + xs.max()) // 2


def feature(arr):
    al = arr[:, :, 3].astype(np.float32) / 255.0
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
    return lum * al + 0.45 * al


def flow(i1, i2, levels=4, win_sigma=3.5, iters=6):
    p1 = [i1.astype(np.float32)]
    p2 = [i2.astype(np.float32)]
    for _ in range(1, levels):
        p1.append(gaussian_filter(p1[-1], 1.0)[::2, ::2])
        p2.append(gaussian_filter(p2[-1], 1.0)[::2, ::2])
    vx = np.zeros_like(p1[-1], dtype=np.float32)
    vy = np.zeros_like(p1[-1], dtype=np.float32)
    for lvl in range(levels - 1, -1, -1):
        a1 = p1[lvl]
        a2 = p2[lvl]
        lh, lw = a1.shape
        if lvl < levels - 1:
            sy = lh / vx.shape[0]
            sx = lw / vx.shape[1]
            vx = zoom(vx, (sy, sx), order=1) * sx
            vy = zoom(vy, (sy, sx), order=1) * sy
        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        for _ in range(iters):
            wx = np.clip(xx + vx, 0, lw - 1)
            wy = np.clip(yy + vy, 0, lh - 1)
            w2 = map_coordinates(a2, [wy, wx], order=1, mode="nearest")
            ix = np.gradient(w2, axis=1)
            iy = np.gradient(w2, axis=0)
            it = w2 - a1
            sxx = gaussian_filter(ix * ix, win_sigma) + 1e-3
            syy = gaussian_filter(iy * iy, win_sigma) + 1e-3
            sxy = gaussian_filter(ix * iy, win_sigma)
            sxt = gaussian_filter(ix * it, win_sigma)
            syt = gaussian_filter(iy * it, win_sigma)
            det = sxx * syy - sxy * sxy + 1e-4
            dvx = np.clip(-(syy * sxt - sxy * syt) / det, -2.5, 2.5)
            dvy = np.clip(-(sxx * syt - sxy * sxt) / det, -2.5, 2.5)
            vx = gaussian_filter(vx + dvx, 1.2)
            vy = gaussian_filter(vy + dvy, 1.2)
    return vx, vy


def warp(arr, dx, dy):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sy = np.clip(yy + dy, 0, h - 1)
    sx = np.clip(xx + dx, 0, w - 1)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    out = np.zeros((h, w, 4), dtype=np.float32)
    wa = map_coordinates(alpha, [sy, sx], order=1, mode="constant", cval=0.0)
    for c in range(3):
        prem = arr[:, :, c].astype(np.float32) * alpha
        wc = map_coordinates(prem, [sy, sx], order=1, mode="constant", cval=0.0)
        out[:, :, c] = np.where(wa > 1e-3, np.clip(wc / np.maximum(wa, 1e-3), 0, 255), 0.0)
    out[:, :, 3] = np.clip(wa * 255.0, 0, 255)
    return out


def interp(im_a, im_b, t, flows, near=None, seam=258):
    a = np.array(im_a, dtype=np.uint8)
    b = np.array(im_b, dtype=np.uint8)
    vx_ab, vy_ab, vx_ba, vy_ba = flows
    wa = warp(a, -t * vx_ab, -t * vy_ab)
    wb = warp(b, -(1.0 - t) * vx_ba, -(1.0 - t) * vy_ba)

    rgb_d = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
    alpha_d = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
    disagree = rgb_d * 0.7 + alpha_d * 0.9
    agree = np.clip(1.0 - disagree * 3.2, 0.0, 1.0)
    pick_a = (wa[:, :, 3] >= wb[:, :, 3]).astype(np.float32)
    w_a = t * agree + pick_a * (1.0 - agree)
    w_b = 1.0 - w_a

    aa = wa[:, :, 3] / 255.0
    ab = wb[:, :, 3] / 255.0
    out_a = w_a * aa + w_b * ab
    out = np.zeros_like(wa)
    for c in range(3):
        num = w_a * wa[:, :, c] * aa + w_b * wb[:, :, c] * ab
        out[:, :, c] = np.where(out_a > 1e-3, np.clip(num / np.maximum(out_a, 1e-3), 0, 255), 0.0)

    if near is not None:
        n = np.array(near, dtype=np.float32)
        na = n[:, :, 3] / 255.0
        h = out.shape[0]
        yg = np.arange(h, dtype=np.float32)[:, None]
        band = np.clip((yg - (seam - 36)) / 36.0, 0.0, 1.0)
        band = band * band * (3.0 - 2.0 * band)
        pw = band
        oa = out_a * (1.0 - pw) + na * pw
        for c in range(3):
            out[:, :, c] = np.where(oa > 1e-3,
                                    np.clip((out[:, :, c] * out_a * (1.0 - pw) + n[:, :, c] * na * pw) / np.maximum(oa, 1e-3), 0, 255),
                                    0.0)
        out_a = oa

    a8 = np.clip(out_a * 255.0, 0, 255)
    out[:, :, 3] = np.where(a8 < 26, 0.0, np.where(a8 > 226, 255.0, (a8 - 26.0) / 200.0 * 255.0))
    res = out.astype(np.uint8)
    res[res[:, :, 3] < 5] = 0
    return Image.fromarray(res, "RGBA")


def leg_lock(mid, near, seam=300, feather=18):
    a = np.array(mid, dtype=np.float32)
    b = np.array(near, dtype=np.float32)
    h = a.shape[0]
    wgt = np.zeros(h, dtype=np.float32)
    top = seam - feather
    bot = seam + feather
    for y in range(h):
        if y <= top:
            wgt[y] = 0.0
        elif y >= bot:
            wgt[y] = 1.0
        else:
            t = (y - top) / float(bot - top)
            wgt[y] = t * t * (3.0 - 2.0 * t)
    wgt = wgt[:, None, None]
    ab = a[:, :, 3:4] / 255.0
    bb = b[:, :, 3:4] / 255.0
    out_a = ab * (1.0 - wgt) + bb * wgt
    rgb = (a[:, :, :3] * ab * (1.0 - wgt) + b[:, :, :3] * bb * wgt) / np.maximum(out_a, 1e-4)
    out = np.empty_like(a)
    out[:, :, :3] = np.clip(rgb, 0, 255)
    out[:, :, 3] = np.clip(out_a * 255.0, 0, 255)
    res = out.astype(np.uint8)
    res[res[:, :, 3] < 5] = 0
    return Image.fromarray(res, "RGBA")


def motion(img_a, img_b):
    a = np.array(img_a, dtype=np.float32)
    b = np.array(img_b, dtype=np.float32)
    return float(np.abs(a[:, :, :3] - b[:, :, :3]).mean() * (a[:, :, 3] > 20).mean())


def pick_drops(n, diffs):
    cand = sorted((diffs[i - 1], i) for i in range(1, n - 3))
    return sorted(i for _, i in cand[:DROP_COUNT])


def weights(n):
    w = []
    for i in range(n):
        if i == 0 or i == n - 1:
            w.append(1.30)
        elif i < 6:
            w.append(1.04 - 0.03 * i)
        elif i >= n - 6:
            w.append(1.02 + 0.05 * (i - (n - 6)))
        else:
            w.append(1.0 + 0.06 * math.sin(i * 0.7))
    return w


def schedule(n_native, plan):
    base = weights(n_native)
    nat = [max(120, int(round(w * TOTAL_MS / sum(base)))) for w in base]
    nat[-1] += TOTAL_MS - sum(nat)
    pieces = []
    for i, t in enumerate(nat):
        k = len(plan.get(i, []))
        share = max(MIN_FRAME_MS, t // (k + 1))
        pieces.append(share)
        pieces.extend([share] * k)
    scaled = [max(MIN_FRAME_MS, int(round(p * TOTAL_MS / sum(pieces)))) for p in pieces]
    diff = TOTAL_MS - sum(scaled)
    idx = 0
    while diff != 0:
        j = idx % len(scaled)
        if diff > 0:
            scaled[j] += 1
            diff -= 1
        elif scaled[j] > MIN_FRAME_MS + 1:
            scaled[j] -= 1
            diff += 1
        idx += 1
    return scaled


def lowdiff(a, b, top=300):
    x = np.array(a, dtype=np.float32)[top:, :, :]
    y = np.array(b, dtype=np.float32)[top:, :, :]
    msk = (x[:, :, 3] > 20) | (y[:, :, 3] > 20)
    if not msk.any():
        return 0.0
    return float(np.abs(x[:, :, :3] - y[:, :, :3]).mean(2)[msk].mean())


if __name__ == "__main__":
    src = load36()
    diffs = [motion(src[i], src[i + 1]) for i in range(len(src) - 1)]
    drops = pick_drops(len(src), diffs)
    keep = [f for i, f in enumerate(src) if i not in set(drops)]
    keep[-1] = keep[0].copy()

    meta = [body_metrics(f) for f in keep]
    scored = []
    for i in range(len(keep) - 1):
        scored.append((motion(keep[i], keep[i + 1]), lowdiff(keep[i], keep[i + 1]), i))
    scored.sort(reverse=True)

    need = TARGET - len(keep)
    gaps = need
    plan = {}
    for m, ld, i in scored[:gaps]:
        plan[i] = [0.5]

    print("dropped", [i + 1 for i in drops])
    print("kept", len(keep), "insert", sum(len(v) for v in plan.values()))
    for m, ld, i in scored[:gaps]:
        print("  gap after kept#%d motion=%.2f lowdiff=%.1f" % (i + 1, m, ld))

    flows_cache = {}
    seq = []
    for i, f in enumerate(keep):
        seq.append(f)
        for t in plan.get(i, []):
            key = i
            if key not in flows_cache:
                fa = feature(np.array(keep[i], dtype=np.uint8))
                fb = feature(np.array(keep[i + 1], dtype=np.uint8))
                flows_cache[key] = flow(fa, fb) + flow(fb, fa)
            seq.append(interp(keep[i], keep[i + 1], t, flows_cache[key], near=keep[i] if t <= 0.5 else keep[i + 1]))

    assert len(seq) == TARGET, len(seq)

    bad = []
    for i, f in enumerate(seq):
        hm, fm, wm, am, cm = body_metrics(f)
        if abs(hm - 306) > 3 or abs(fm - 366) > 2:
            bad.append((i + 1, hm, fm))
    print("frames off anchor:", bad)

    durs = schedule(len(keep), plan)
    assert sum(durs) == TOTAL_MS, sum(durs)

    os.makedirs(OUT_DIR, exist_ok=True)
    sheet = Image.new("RGBA", (COLS * CANVAS, ROWS * CANVAS), (0, 0, 0, 0))
    for i, f in enumerate(seq):
        r, c = divmod(i, COLS)
        sheet.paste(f, (c * CANVAS, r * CANVAS), f)

    anim_path = os.path.join(OUT_DIR, "04roam（踏步_动画）.webp")
    sheet_path = os.path.join(OUT_DIR, "04roam_sheet（踏步_图集）.webp")
    seq[0].save(anim_path, format="WEBP", save_all=True, append_images=seq[1:],
                duration=durs, loop=0, lossless=True, quality=100, disposal=2, minimize_size=False)
    sheet.save(sheet_path, format="WEBP", lossless=True, quality=100, method=6)

    for tone, bg in (("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255))):
        comp = Image.new("RGBA", sheet.size, bg)
        comp.alpha_composite(sheet)
        comp = comp.convert("RGB")
        d = ImageDraw.Draw(comp)
        for i in range(TARGET):
            r, c = divmod(i, COLS)
            d.text((c * CANVAS + 8, r * CANVAS + 6), str(i + 1), fill=(210, 30, 30))
        comp.save(os.path.join(OUT_DIR, "04roam_%s.png" % tone))

    print("durations min/max/total", min(durs), max(durs), sum(durs))
    print(anim_path)
