import numpy as np
from PIL import Image
from scipy import ndimage
import os, json

HOME = os.path.expanduser("~")
BASE = f"{HOME}/workspace/star_cat_girl/busy"
SHEETS = {
    0: "media-generation-busy-seg1b-0-1bf42d6e-f2ad-427d-a650-6b4ecac9879e.webp",
    1: "media-generation-busy-seg2-0-30773041-fb61-446f-9897-a8d82401fb12.webp",
    3: "media-generation-busy-seg4-0-697e6f74-43e7-436b-8c5c-6ebb0daaf416.webp",
}
PER = 12

def cutout(cell):
    rgb = np.asarray(cell.convert("RGB")).astype(np.int32)
    mx = rgb.max(axis=2)
    sat = (mx - rgb.min(axis=2)) / np.maximum(mx, 1)
    dark = mx < 46
    lab, n = ndimage.label(dark)
    border_ids = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - {0}
    alpha = np.ones(mx.shape, np.uint8) * 255
    for i in border_ids:
        alpha[lab == i] = 0
    for i in range(1, n + 1):
        if i in border_ids:
            continue
        m = lab == i
        if m.sum() < 4:
            alpha[m] = 0
        elif mx[m].mean() < 40 and sat[m].mean() < 0.16:
            alpha[m] = 0
    lab2, n2 = ndimage.label(alpha > 0)
    for i in range(1, n2 + 1):
        if (lab2 == i).sum() < 8:
            alpha[lab2 == i] = 0
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA"), alpha

def main_stats(alpha):
    lab, n = ndimage.label(alpha > 0)
    sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(m)
    return m, xs.min(), xs.max(), ys.min(), ys.max()

# gather 48 cells: seg3 from pre-cut component cells
cut = [None] * 48
for si, name in SHEETS.items():
    im = Image.open(f"{BASE}/{name}")
    w, h = im.size
    cw, ch = w // 4, h // 3
    for r in range(3):
        for c in range(4):
            idx = si * PER + r * 4 + c
            cut[idx] = cutout(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
for i in range(PER):
    im = Image.open(f"{BASE}/seg3_cells/c{i+1:02d}.png")
    cut[24 + i] = (im, np.array(im.split()[3]))

heights = []
for i, (rgba, alpha) in enumerate(cut):
    m, x0, x1, y0, y1 = main_stats(alpha)
    heights.append(y1 - y0 + 1)
    rgba.save(f"{BASE}/frames/b{i+1:02d}_raw.png")
seg_meds = [float(np.median(heights[i * PER:(i + 1) * PER])) for i in range(4)]
print("heights:", heights)
print("per-segment medians:", seg_meds)

TARGET_H, FEET_Y = 305, 372

def place(rgba, sc):
    im = rgba.resize((max(1, round(rgba.width * sc)), max(1, round(rgba.height * sc))), Image.LANCZOS)
    a = np.array(im.split()[3])
    a = np.where(a >= 128, 255, 0).astype(np.uint8)
    im.putalpha(Image.fromarray(a))
    m, x0, x1, y0, y1 = main_stats(a)
    hgt = y1 - y0 + 1
    band = m.copy(); band[y0 + int(hgt * 0.30):, :] = False
    ys, xs = np.where(band)
    return im, xs.mean(), (x0, x1, y0, y1)

placed = []
for i, (rgba, alpha) in enumerate(cut):
    sc = TARGET_H / heights[i]              # per-cell normalize: feet-planted action, doc demands constant proportions
    placed.append(place(rgba, sc))
im0, hcx0, (bx0, bx1, by0, by1) = placed[0]
TARGET_HCX = hcx0 + (192 - (bx0 + bx1) / 2)
print("target head-cx:", round(TARGET_HCX, 1))

finals, report = [], []
for i, (im, hcx, (x0, x1, y0, y1)) in enumerate(placed):
    canvas = Image.new("RGBA", (384, 384), (0, 0, 0, 0))
    canvas.alpha_composite(im, (int(round(TARGET_HCX - hcx)), int(round(FEET_Y - y1))))
    finals.append(canvas)
    a = np.array(canvas.split()[3])
    ys, xs = np.where(a > 0)
    report.append(dict(frame=i + 1, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       margin=[int(xs.min()), int(ys.min()), int(383 - xs.max()), int(383 - ys.max())],
                       semi=int(((a > 0) & (a < 255)).sum())))
    canvas.save(f"{BASE}/proc/b{i+1:02d}.png")

finals[-1] = finals[0].copy()
finals[-1].save(f"{BASE}/proc/b48.png")
d = np.abs(np.asarray(finals[-1]).astype(int) - np.asarray(finals[0]).astype(int)).max()
print("seam last vs first max diff:", d)
print("semi-transparent px total:", sum(r["semi"] for r in report))
for r in report:
    if min(r["margin"]) < 6:
        print("TIGHT MARGIN:", r)
json.dump(report, open(f"{BASE}/proc/report.json", "w"), indent=1)
print("done")
