import numpy as np
from PIL import Image
from scipy import ndimage
import os, json

HOME = os.path.expanduser("~")
BASE = f"{HOME}/workspace/star_cat_girl/thinking"
SHEETS = [
    "media-generation-thinking-seg1-0-aa3c8326-dd5f-43be-b85c-752a180dc30f.webp",
    "media-generation-thinking-seg2b-0-c33c2781-8492-4561-8a3d-287edc363e3b.webp",
    "media-generation-thinking-seg3-0-220f68a6-103e-4e72-8f08-678741ace3b9.webp",
    "media-generation-thinking-seg4-0-3f4c7c00-3ef3-4324-9c89-6b84697ca0dc.webp",
]
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
        m = lab2 == i
        if m.sum() < 8:
            alpha[m] = 0
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA"), alpha

def main_stats(alpha):
    lab, n = ndimage.label(alpha > 0)
    sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(m)
    return m, xs.min(), xs.max(), ys.min(), ys.max()

cells = []
for name in SHEETS:
    im = Image.open(f"{BASE}/{name}")
    w, h = im.size
    cw, ch = w // 4, h // 3
    for r in range(3):
        for c in range(4):
            cells.append(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
print("cells:", len(cells))

cut, heights = [], []
for i, cell in enumerate(cells):
    rgba, alpha = cutout(cell)
    m, x0, x1, y0, y1 = main_stats(alpha)
    heights.append(y1 - y0 + 1)
    cut.append((rgba, alpha))
    rgba.save(f"{BASE}/frames/t{i+1:02d}_raw.png")
nseg = len(cells) // PER
seg_meds = [float(np.median(heights[i * PER:(i + 1) * PER])) for i in range(nseg)]
print("heights:", heights)
print("per-segment medians:", seg_meds)

TARGET_H, FEET_Y = 305, 372
seg_scales = [TARGET_H / m for m in seg_meds]
print("per-segment scales:", [round(s, 4) for s in seg_scales])

def place(rgba, sc):
    im = rgba.resize((round(rgba.width * sc), round(rgba.height * sc)), Image.LANCZOS)
    a = np.array(im.split()[3])
    a = np.where(a >= 128, 255, 0).astype(np.uint8)
    im.putalpha(Image.fromarray(a))
    m, x0, x1, y0, y1 = main_stats(a)
    hgt = y1 - y0 + 1
    band = m.copy(); band[y0 + int(hgt * 0.30):, :] = False
    ys, xs = np.where(band)
    return im, xs.mean(), (x0, x1, y0, y1)

placed = [place(rgba, seg_scales[i // PER]) for i, (rgba, alpha) in enumerate(cut)]
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
    canvas.save(f"{BASE}/proc/t{i+1:02d}.png")

finals[-1] = finals[0].copy()
finals[-1].save(f"{BASE}/proc/t{len(finals):02d}.png")
d = np.abs(np.asarray(finals[-1]).astype(int) - np.asarray(finals[0]).astype(int)).max()
print("seam last vs first max diff:", d)
print("semi-transparent px total:", sum(r["semi"] for r in report))
for r in report:
    if min(r["margin"]) < 6:
        print("TIGHT MARGIN:", r)
json.dump(report, open(f"{BASE}/proc/report.json", "w"), indent=1)
print("done")
