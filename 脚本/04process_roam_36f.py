import math
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter, label

G = r"C:\Users\30894\Desktop\workspace\帧动画\gemini"
sys.path.insert(0, os.path.join(G, "脚本"))
from build_webp import find_cell_boxes

PARTS = [os.path.join(G, "素材", "04roam", "part%d.webp" % i) for i in (1, 2, 3)]
OUT_DIR = os.path.join(G, "待审", "04roam", "36帧基准")
CANVAS = 384
TARGET_H = 306
GROUND_Y = 366
CENTER_X = 192
BG_T = 30
AA_SIGMA = 0.60
TOTAL_MS = 7000


def flood_from_border(dark):
    h, w = dark.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if dark[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if dark[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def drop_edge_specks(fg, min_area=260):
    lab, n = label(fg)
    if n == 0:
        return fg
    out = fg.copy()
    h, w = fg.shape
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_area:
            if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
                out[m] = False
    return out


def matte(rgb):
    arr = rgb.astype(np.float32)
    mr = arr.max(axis=2)
    chroma = mr - arr.min(axis=2)

    bg = flood_from_border(mr <= BG_T)
    bg |= (mr <= 14) & (chroma <= 12)

    fg = drop_edge_specks(~bg)
    lab, n = label(fg)
    if n == 0:
        return np.zeros(arr.shape[:2] + (4,), dtype=np.uint8)

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main_id = int(sizes.argmax())
    main = lab == main_id

    core = binary_erosion(main, iterations=4)
    if not core.any():
        core = binary_erosion(main, iterations=2)
    num = gaussian_filter(np.where(core, mr, 0.0), 2.5)
    den = gaussian_filter(core.astype(np.float32), 2.5)
    ref = np.where(den > 1e-4, num / np.maximum(den, 1e-6), 0.0)
    ref = np.maximum(ref, 46.0)

    soft = np.clip((mr - BG_T * 0.25) / (ref * 0.92), 0.0, 1.0)
    rim = main & ~binary_erosion(main, iterations=3)
    alpha = np.where(main, np.where(rim, soft, 1.0), 0.0)

    part_mask = fg & ~main
    if part_mask.any():
        pa = np.clip((mr - 38.0) / 190.0, 0.0, 1.0)
        alpha = np.where(part_mask, pa, alpha)

    alpha = gaussian_filter(alpha, AA_SIGMA)
    alpha = np.where(alpha < 0.03, 0.0, alpha)
    alpha = np.clip(alpha, 0.0, 1.0)

    color = arr.copy()
    soft_area = fg & (alpha > 0.05) & (alpha < 0.995)
    color[soft_area] = np.clip(arr[soft_area] / np.maximum(alpha[soft_area, None], 0.5), 0.0, 255.0)

    outline = np.array([68.0, 48.0, 98.0], dtype=np.float32)
    edge = fg & ~binary_erosion(fg, iterations=1)
    dark_edge = edge & (mr < 74.0)
    wgt = np.clip((74.0 - mr[dark_edge]) / 74.0, 0.0, 1.0)[:, None] * 0.75
    color[dark_edge] = color[dark_edge] * (1.0 - wgt) + outline * wgt

    out = np.zeros(arr.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(color, 0.0, 255.0).astype(np.uint8)
    out[..., 3] = (alpha * 255.0).astype(np.uint8)
    out[out[..., 3] < 4] = 0
    return out


def body_metrics(rgba):
    a = rgba[..., 3]
    solid = a > 96
    lab, n = label(solid)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = lab == int(sizes.argmax())
    ys, xs = np.where(main)
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    band = main[y0:y0 + max(8, int(h * 0.34))]
    bx = np.where(band)[1]
    cx = float(bx.mean())
    return y0, y1, h, cx, main


def load_frames():
    frames = []
    for path in PARTS:
        sheet = Image.open(path).convert("RGB")
        for box in find_cell_boxes(sheet, 4, 3):
            frames.append(matte(np.array(sheet.crop(box), dtype=np.uint8)))
    return frames


def align(frames):
    metrics = [body_metrics(f) for f in frames]
    heights = np.array([m[2] for m in metrics], dtype=np.float32)
    base = float(TARGET_H) / float(np.median(heights))
    out = []
    report = []
    for f, m in zip(frames, metrics):
        y0, y1, h, cx, main = m
        scale = TARGET_H / float(h)
        scale = float(np.clip(scale, base * 0.97, base * 1.03))
        img = Image.fromarray(f, "RGBA")
        nw = max(1, int(round(img.width * scale)))
        nh = max(1, int(round(img.height * scale)))
        rs = img.resize((nw, nh), Image.Resampling.LANCZOS)
        arr = np.array(rs, dtype=np.float32)
        ma = arr[..., 3]
        lab2, n2 = label(ma > 40)
        sizes2 = np.bincount(lab2.ravel())
        sizes2[0] = 0
        mm = lab2 == int(sizes2.argmax())
        ys, xs = np.where(mm)
        nb_y1 = int(ys.max())
        bh = int(ys.max() - ys.min() + 1)
        band = mm[ys.min():ys.min() + max(8, int(bh * 0.34))]
        nb_cx = float(np.where(band)[1].mean())

        canvas = np.zeros((CANVAS, CANVAS, 4), dtype=np.float32)
        dx = int(round(CENTER_X - nb_cx))
        dy = int(round(GROUND_Y - nb_y1))
        sx0 = max(0, -dx)
        sy0 = max(0, -dy)
        dx0 = max(0, dx)
        dy0 = max(0, dy)
        wcopy = min(nw - sx0, CANVAS - dx0)
        hcopy = min(nh - sy0, CANVAS - dy0)
        if wcopy > 0 and hcopy > 0:
            piece = arr[sy0:sy0 + hcopy, sx0:sx0 + wcopy]
            alpha_p = piece[..., 3:4] / 255.0
            dst = canvas[dy0:dy0 + hcopy, dx0:dx0 + wcopy]
            rgb_pre = piece[..., :3] * alpha_p + dst[..., :3] * (1.0 - alpha_p)
            a_out = alpha_p + dst[..., 3:4] / 255.0 * (1.0 - alpha_p)
            dst[..., :3] = np.where(a_out > 0, rgb_pre / np.maximum(a_out, 1e-4), 0)
            dst[..., 3:4] = a_out * 255.0
        res = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
        res[res[..., 3] < 4] = 0
        out.append(Image.fromarray(res, "RGBA"))
        report.append((round(scale, 3), bh, int(nb_y1), round(nb_cx, 1)))
    return out, report


def durations(n):
    weights = []
    for i in range(n):
        if i == 0 or i == n - 1:
            weights.append(1.32)
        elif i < 6:
            weights.append(1.05 - 0.03 * i)
        elif i >= n - 6:
            weights.append(1.02 + 0.05 * (i - (n - 6)))
        else:
            weights.append(1.0 + 0.06 * math.sin(i * 0.7))
    base = TOTAL_MS / sum(weights)
    ms = [max(120, int(round(w * base))) for w in weights]
    diff = TOTAL_MS - sum(ms)
    idx = 0
    while diff != 0:
        if diff > 0:
            ms[idx % n] += 1
            diff -= 1
        else:
            if ms[idx % n] > 130:
                ms[idx % n] -= 1
                diff += 1
        idx += 1
    return ms


def build(frames):
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(frames)
    frames = list(frames)
    frames[-1] = frames[0].copy()
    cols, rows = 6, 6
    sheet = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet.paste(f, (c * CANVAS, r * CANVAS), f)

    anim_path = os.path.join(OUT_DIR, "04roam（踏步_动画）.webp")
    sheet_path = os.path.join(OUT_DIR, "04roam_sheet（踏步_图集）.webp")
    durs = durations(n)
    frames[0].save(anim_path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, lossless=True, quality=100, disposal=2, minimize_size=False)
    sheet.save(sheet_path, format="WEBP", lossless=True, quality=100, method=6)

    for tone, bg in (("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255))):
        comp = Image.new("RGBA", sheet.size, bg)
        comp.alpha_composite(sheet)
        comp = comp.convert("RGB")
        d = ImageDraw.Draw(comp)
        for i in range(n):
            r, c = divmod(i, cols)
            d.text((c * CANVAS + 8, r * CANVAS + 6), str(i + 1), fill=(210, 30, 30))
        comp.save(os.path.join(OUT_DIR, "04roam_%s.png" % tone))

    return anim_path, sheet_path, durs


def compare_strip(frames):
    old = Image.open(os.path.join(G, "成品", "04roam（踏步_动画）.webp"))
    idxs = [0, 8, 17, 26, 35]
    w = 300
    strip = Image.new("RGBA", (w * len(idxs), w * 2 + 24), (255, 255, 255, 255))
    for k, i in enumerate(idxs):
        old.seek(i)
        o = old.convert("RGBA").resize((w, w), Image.Resampling.LANCZOS)
        strip.alpha_composite(o, (k * w, 0))
        nf = frames[i].resize((w, w), Image.Resampling.LANCZOS)
        strip.alpha_composite(nf, (k * w, w + 24))
    strip.convert("RGB").save(os.path.join(OUT_DIR, "对照_帧位_旧vs36.png"))


if __name__ == "__main__":
    raw = load_frames()
    aligned, rep = align(raw)
    print("frames:", len(aligned))
    for i, r in enumerate(rep):
        print(i + 1, r)
    anim, sheet, durs = build(aligned)
    print("durations:", durs, "total", sum(durs))
    compare_strip(aligned)
    print(anim)
    print(sheet)
