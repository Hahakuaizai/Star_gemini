import numpy as np
from PIL import Image
from scipy import ndimage
import os, json

HOME = os.path.expanduser("~")
BASE = f"{HOME}/workspace/star_cat_girl/roam"
SHEETS = [
    "media-generation-roam-seg1-0-bb854903-0688-42db-be95-ace51de0ec76.webp",
    "media-generation-roam-seg2-0-56a63e27-ed95-4e0e-bd51-89e871dcd5a1.webp",
    "media-generation-roam-seg3b-0-c83e4440-89a0-4330-a5b7-5dab20c8a93a.webp",
]

def cutout(cell):
    """Black-bg cutout: border flood (max<46) + pocket rule; keep detached sparkles."""
    rgb = np.asarray(cell.convert("RGB")).astype(np.int32)
    mx = rgb.max(axis=2)
    sat = (mx - rgb.min(axis=2)) / np.maximum(mx, 1)
    dark = mx < 46
    # border-connected dark
    lab, n = ndimage.label(dark)
    border_ids = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - {0}
    alpha = np.ones(mx.shape, np.uint8) * 255
    for i in border_ids:
        alpha[lab == i] = 0
    # pockets: enclosed dark components
    for i in range(1, n + 1):
        if i in border_ids:
            continue
        m = lab == i
        if m.sum() < 4:
            alpha[m] = 0
        elif mx[m].mean() < 40 and sat[m].mean() < 0.16:
            alpha[m] = 0
    # remove specks < 8 px among opaque
    lab2, n2 = ndimage.label(alpha > 0)
    for i in range(1, n2 + 1):
        m = lab2 == i
        if m.sum() < 8:
            alpha[m] = 0
    out = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA"), lab2, alpha

def main_stats(alpha):
    lab, n = ndimage.label(alpha > 0)
    sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
    main_id = 1 + int(np.argmax(sizes))
    m = lab == main_id
    ys, xs = np.where(m)
    return m, xs.min(), xs.max(), ys.min(), ys.max()

# --- slice + cutout ---
cells = []
for si, name in enumerate(SHEETS):
    im = Image.open(f"{BASE}/{name}")
    w, h = im.size
    cw, ch = w // 4, h // 3
    for r in range(3):
        for c in range(4):
            cells.append(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
print("cells:", len(cells))

cut = []
heights = []
for i, cell in enumerate(cells):
    rgba, lab2, alpha = cutout(cell)
    m, x0, x1, y0, y1 = main_stats(alpha)
    heights.append(y1 - y0 + 1)
    cut.append((rgba, alpha))
    rgba.save(f"{BASE}/frames/r{i+1:02d}_raw.png")
med = float(np.median(heights))
seg_meds = [float(np.median(heights[i*12:(i+1)*12])) for i in range(3)]
print("main-comp heights:", heights)
print("median:", med, "per-segment medians:", seg_meds)

# --- per-segment scale + placement ---
TARGET_H, FEET_Y = 305, 372
seg_scales = [TARGET_H / m for m in seg_meds]
print("per-segment scales:", [round(s, 4) for s in seg_scales])
scale = seg_scales[0]

def place(rgba, alpha, sc):
    im = rgba.resize((round(rgba.width * sc), round(rgba.height * sc)), Image.LANCZOS)
    a = np.array(im.split()[3])
    a = np.where(a >= 128, 255, 0).astype(np.uint8)
    im.putalpha(Image.fromarray(a))
    m, x0, x1, y0, y1 = main_stats(a)
    # head centre-x: centroid of main-comp pixels in top 30% of bbox
    hgt = y1 - y0 + 1
    band = m.copy()
    band[y0 + int(hgt * 0.30):, :] = False
    ys, xs = np.where(band)
    head_cx = xs.mean()
    return im, head_cx, (x0, x1, y0, y1)

placed = [place(rgba, alpha, seg_scales[i // 12]) for i, (rgba, alpha) in enumerate(cut)]
# target head-cx: centre base frame's bbox at 192
im0, hcx0, (bx0, bx1, by0, by1) = placed[0]
TARGET_HCX = hcx0 + (192 - (bx0 + bx1) / 2)
print("target head-cx:", round(TARGET_HCX, 1))

finals = []
report = []
for i, (im, hcx, (x0, x1, y0, y1)) in enumerate(placed):
    canvas = Image.new("RGBA", (384, 384), (0, 0, 0, 0))
    dx = int(round(TARGET_HCX - hcx))
    dy = int(round(FEET_Y - y1))
    canvas.alpha_composite(im, (dx, dy))
    finals.append(canvas)
    a = np.array(canvas.split()[3])
    ys, xs = np.where(a > 0)
    semi = int(((a > 0) & (a < 255)).sum())
    report.append(dict(frame=i + 1, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       margin=[int(xs.min()), int(ys.min()), int(383 - xs.max()), int(383 - ys.max())],
                       semi=semi))
    canvas.save(f"{BASE}/proc/r{i+1:02d}.png")

# frame 36 := frame 1 (doc: first == last pixel-identical)
finals[35] = finals[0].copy()
finals[35].save(f"{BASE}/proc/r36.png")
d = np.abs(np.asarray(finals[35]).astype(int) - np.asarray(finals[0]).astype(int)).max()
print("seam frame36 vs frame1 max diff:", d)
print("semi-transparent px total:", sum(r["semi"] for r in report))
for r in report:
    if min(r["margin"]) < 6:
        print("TIGHT MARGIN:", r)
json.dump(report, open(f"{BASE}/proc/report.json", "w"), indent=1)
print("done")
