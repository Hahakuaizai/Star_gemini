import numpy as np
from PIL import Image
from scipy import ndimage
import os, json

HOME = os.path.expanduser("~")
B = f"{HOME}/workspace/star_cat_girl/sleep_loop"
TARGET_H, FEET_Y, TARGET_HCX = 265, 372, 183.1

def char_geom(alpha):
    rows = (alpha > 0).sum(axis=1)
    nz = np.where(rows > 0)[0]
    bot = int(nz.max())
    top = int(nz[rows[nz] > 0.35 * rows.max()].min())
    return top, bot

def band_med_x(alpha, top, bot):
    h = bot - top + 1
    band = alpha[top:top + int(h * 0.30), :]
    ys, xs = np.where(band > 0)
    return float(np.median(xs))

cells = [Image.open(f"{B}/cells/f{i:02d}.png") for i in range(1, 49)]
finals, report = [], []
for i, c in enumerate(cells):
    a0 = np.array(c.split()[3])
    t0, b0 = char_geom(a0)
    sc = TARGET_H / (b0 - t0 + 1)
    im = c.resize((max(1, round(c.width * sc)), max(1, round(c.height * sc))), Image.LANCZOS)
    a = np.array(im.split()[3])
    a = np.where(a >= 128, 255, 0).astype(np.uint8)
    im.putalpha(Image.fromarray(a))
    t1, b1 = char_geom(a)
    mx = band_med_x(a, t1, b1)
    canvas = Image.new("RGBA", (384, 384), (0, 0, 0, 0))
    canvas.alpha_composite(im, (int(round(TARGET_HCX - mx)), int(round(FEET_Y - b1))))
    finals.append(canvas)
    fa = np.array(canvas.split()[3])
    t2, b2 = char_geom(fa)
    fys, fxs = np.where(fa > 0)
    report.append(dict(frame=i + 1, char_top=int(t2), char_bot=int(b2),
                       margin=[int(fxs.min()), int(fys.min()), int(383 - fxs.max()), int(383 - fys.max())],
                       semi=int(((fa > 0) & (fa < 255)).sum())))
    canvas.save(f"{B}/proc/f{i+1:02d}.png")
print("char_tops:", [r["char_top"] for r in report])
print("char_bots:", [r["char_bot"] for r in report])
print("semi-transparent px total:", sum(r["semi"] for r in report))
for r in report:
    if min(r["margin"]) < 6:
        print("TIGHT MARGIN:", r)
json.dump(report, open(f"{B}/proc/report.json", "w"), indent=1)

for i in range(1, 49):
    p = f"{B}/proc/f{i:02d}.png"
    a = np.array(Image.open(p))
    alpha = a[:, :, 3]
    lab, n = ndimage.label(alpha > 0)
    if n <= 1: continue
    sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
    mid = 1 + int(np.argmax(sizes))
    mys, _ = np.where(lab == mid)
    rgb = a[:, :, :3].astype(int)
    mx = rgb.max(axis=2)
    rem = 0
    for cid in range(1, n + 1):
        if cid == mid: continue
        sz = sizes[cid - 1]
        if sz < 20 or sz > 250: continue
        m = lab == cid
        ys, xs = np.where(m)
        if ys.max() < mys.min() - 2 and mx[m].mean() < 145:
            a[m, 3] = 0; rem += int(sz)
    if rem:
        Image.fromarray(a).save(p)
        finals[i - 1] = Image.open(p)
        print(f"frame {i}: dash-clean {rem} px")
finals[47] = finals[0]
finals[47].save(f"{B}/proc/f48.png")
print("frame48 := frame1; diff:",
      int(np.abs(np.array(finals[47]).astype(int) - np.array(finals[0]).astype(int)).max()))
