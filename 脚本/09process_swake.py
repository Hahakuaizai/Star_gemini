import numpy as np
from PIL import Image
from scipy import ndimage
import os, json

HOME = os.path.expanduser("~")
B = f"{HOME}/workspace/star_cat_girl/sleep_wake"
SHEETS = [
    "media-generation-sleep-wake-seg1-0-3eff7442-8269-486a-9294-5f902406c531.webp",
    "media-generation-sleep-wake-seg2-0-18f08b74-3a5a-4cd1-ac22-398c6b47eced.webp",
    "media-generation-sleep-wake-seg3-0-01df3242-9957-481d-a809-cb3f3563f635.webp",
]

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
        if i in border_ids: continue
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

def face_band(alpha):
    m, x0, x1, y0, y1 = main_stats(alpha)
    h = y1 - y0 + 1
    spans = []
    for yy in range(y0 + int(h * 0.13), y0 + int(h * 0.33)):
        xs = np.where(m[yy])[0]
        if len(xs) > 3:
            spans.append(xs.max() - xs.min() + 1)
    return float(np.median(spans))

cells = []
for name in SHEETS:
    im = Image.open(f"{B}/{name}")
    w, h = im.size
    cw, ch = w // 4, h // 3
    for r in range(3):
        for c in range(4):
            cells.append(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
print("cells:", len(cells))

cut, heights, bands = [], [], []
for i, cell in enumerate(cells):
    rgba, alpha = cutout(cell)
    m, x0, x1, y0, y1 = main_stats(alpha)
    heights.append(y1 - y0 + 1)
    bands.append(face_band(alpha))
    cut.append((rgba, alpha))
    rgba.save(f"{B}/frames/w{i+1:02d}_raw.png")
print("heights:", heights)
print("face bands:", bands)

# rolling-median (window 5) to kill sparkle/arm band spikes while keeping real drift
bands = [float(np.median(bands[max(0, i - 2):i + 3])) for i in range(len(bands))]
print("smoothed bands:", [round(b) for b in bands])
# base scale from frame 36 (final standing) -> height 305; damped face-band correction
base_sc = 305 / heights[35]
B0 = bands[35]
print("base scale:", round(base_sc, 4), "B0:", B0)
FEET_Y = 372

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

placed = [place(rgba, base_sc * (B0 / bands[i]) ** 0.6) for i, (rgba, alpha) in enumerate(cut)]
im0, hcx_last, (bx0, bx1, by0, by1) = placed[35]
TARGET_HCX = hcx_last + (192 - (bx0 + bx1) / 2)
print("target head-cx:", round(TARGET_HCX, 1))

finals, report = [], []
for i, (im, hcx, (x0, x1, y0, y1)) in enumerate(placed):
    canvas = Image.new("RGBA", (384, 384), (0, 0, 0, 0))
    canvas.alpha_composite(im, (int(round(TARGET_HCX - hcx)), int(round(FEET_Y - y1))))
    # remove detached comps hanging below the main bottom (floor glints)
    ca = np.array(canvas.split()[3])
    lab, n = ndimage.label(ca > 0)
    if n > 1:
        sizes = ndimage.sum(ca > 0, lab, range(1, n + 1))
        mid = 1 + int(np.argmax(sizes))
        mys, _ = np.where(lab == mid)
        arr = np.array(canvas)
        changed = False
        for cid in range(1, n + 1):
            if cid == mid: continue
            m2 = lab == cid
            ys2, _ = np.where(m2)
            if ys2.mean() > mys.max():
                arr[m2, 3] = 0; changed = True
        if changed:
            canvas = Image.fromarray(arr)
    finals.append(canvas)
    a = np.array(canvas.split()[3])
    ys, xs = np.where(a > 0)
    report.append(dict(frame=i + 1, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       margin=[int(xs.min()), int(ys.min()), int(383 - xs.max()), int(383 - ys.max())],
                       semi=int(((a > 0) & (a < 255)).sum())))
    canvas.save(f"{B}/proc/w{i+1:02d}.png")
print("semi-transparent px total:", sum(r["semi"] for r in report))
for r in report:
    if min(r["margin"]) < 6:
        print("TIGHT MARGIN:", r)
json.dump(report, open(f"{B}/proc/report.json", "w"), indent=1)
print("done")
