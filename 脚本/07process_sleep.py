import numpy as np
from PIL import Image
from scipy import ndimage
import os, json

HOME = os.path.expanduser("~")
BASE = f"{HOME}/workspace/star_cat_girl/sleep_start"
SHEETS = [
    "media-generation-sleep-start-seg1b-0-e071c5cc-8ca3-4311-9afe-e295b7d5a521.webp",
    "media-generation-sleep-start-seg2-0-91cf51ec-c633-4cc9-9bc8-62834990541a.webp",
    "media-generation-sleep-start-seg3-0-39d596c1-6561-4d89-a47b-0b9ba4891a02.webp",
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
        if (lab2 == i).sum() < 8:
            alpha[lab2 == i] = 0
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
    rgba.save(f"{BASE}/frames/s{i+1:02d}_raw.png")
print("heights:", heights)

# global scale from frame 1 (standing); seam corrections per segment from seam frame pairs
H1 = heights[0]
base_sc = 305 / H1

# face-band widths (13-33% of main height): seated drawings drift larger; damp-correct per frame
def face_band(alpha):
    m, x0, x1, y0, y1 = main_stats(alpha)
    h = y1 - y0 + 1
    band = m.copy()
    band[:y0 + int(h * 0.13), :] = False
    band[y0 + int(h * 0.33):, :] = False
    ys, xs = np.where(band)
    return xs.max() - xs.min() + 1
bands = [face_band(alpha) for _, alpha in cut]
print("face bands:", bands)
B0 = bands[0]
seam_pairs = [(11, 12), (23, 24)]   # last frame of seg k vs first frame of seg k+1 (same pose family)
factors = [1.0, 1.0, 1.0]
run = 1.0
for k, (a, b) in enumerate(seam_pairs):
    # heights after applying running factor to segment k should match segment k+1 raw start
    ratio = (heights[a] * run) / heights[b]
    run = run * (1.0 / ratio) if abs(ratio - 1) > 0.04 else run
    factors[k + 1] = run
    print(f"seam {a+1}->{b+1}: heights {heights[a]} vs {heights[b]}, ratio {ratio:.3f}, factor seg{k+2} = {factors[k+1]:.3f}")
print("segment factors:", [round(f, 3) for f in factors], "base scale:", round(base_sc, 4))

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

placed = [place(rgba, base_sc * factors[i // PER] * (B0 / bands[i]) ** 0.6) for i, (rgba, alpha) in enumerate(cut)]
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
    canvas.save(f"{BASE}/proc/s{i+1:02d}.png")
print("semi-transparent px total:", sum(r["semi"] for r in report))
for r in report:
    if min(r["margin"]) < 6:
        print("TIGHT MARGIN:", r)
json.dump(report, open(f"{BASE}/proc/report.json", "w"), indent=1)
print("done")
