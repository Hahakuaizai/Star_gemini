from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
import os

HOME = os.path.expanduser("~")
B = f"{HOME}/workspace/star_cat_girl/sleep_loop"
SHEETS = [
    "media-generation-sleep-loop-seg1-0-7aa188b1-39ac-4818-8f25-75f16094eb50.webp",
    "media-generation-sleep-loop-seg2-0-afd3e915-d451-4980-b945-832ff2da9877.webp",
    "media-generation-sleep-loop-seg3-0-478ba454-0c77-4f05-8c1f-1c914df622b7.webp",
    "media-generation-sleep-loop-seg4-0-4f9a3c45-df7c-4405-9c7a-d16e8a203590.webp",
]

def sheet_cells(path):
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im).astype(np.int32)
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
        if m.sum() >= 4 and mx[m].mean() < 40 and sat[m].mean() < 0.16:
            alpha[m] = 0
    lab2, n2 = ndimage.label(alpha > 0)
    sizes = ndimage.sum(alpha > 0, lab2, range(1, n2 + 1))
    mains = []
    for i in range(1, n2 + 1):
        if sizes[i - 1] > 20000:
            ys, xs = np.where(lab2 == i)
            mains.append((i, xs.mean(), ys.mean()))
    mains.sort(key=lambda t: (round(t[2] / 448), t[1]))
    assert len(mains) == 12, f"{path}: {len(mains)} mains"
    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    out = []
    for mid, mcx, mcy in mains:
        assign = [mid]
        for i in range(1, n2 + 1):
            if i == mid or sizes[i - 1] < 8: continue
            ys, xs = np.where(lab2 == i)
            d = min((xs.mean() - a[1]) ** 2 + (ys.mean() - a[2]) ** 2 for a in mains)
            if (xs.mean() - mcx) ** 2 + (ys.mean() - mcy) ** 2 <= d:
                assign.append(i)
        mask = np.isin(lab2, assign)
        ys, xs = np.where(mask)
        x0, x1 = max(0, xs.min() - 8), min(rgba.shape[1], xs.max() + 9)
        y0, y1 = max(0, ys.min() - 8), min(rgba.shape[0], ys.max() + 9)
        cell = rgba[y0:y1, x0:x1].copy()
        cell[~mask[y0:y1, x0:x1], 3] = 0
        out.append(Image.fromarray(cell, "RGBA"))
    return out

all_cells = []
for si, name in enumerate(SHEETS):
    cells = sheet_cells(f"{B}/{name}")
    for i, c in enumerate(cells):
        c.save(f"{B}/cells/f{si * 12 + i + 1:02d}.png")
        all_cells.append(c)
    print(f"seg{si + 1}: 12 cells extracted")

# frame 16 fix: delete the relapsed Z (any detached comp above the main comp)
p = f"{B}/cells/f16.png"
a = np.array(Image.open(p))
alpha = a[:, :, 3]
lab, n = ndimage.label(alpha > 0)
sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
mid = 1 + int(np.argmax(sizes))
mys, _ = np.where(lab == mid)
removed = 0
for cid in range(1, n + 1):
    if cid == mid: continue
    m = lab == cid
    ys, xs = np.where(m)
    if ys.max() < mys.min():
        a[m, 3] = 0
        removed += int(sizes[cid - 1])
Image.fromarray(a).save(p)
print(f"frame 16: removed {removed} px (relapsed Z)")
all_cells[15] = Image.open(p)

sheet = Image.new("RGB", (8 * 150, 6 * 165), (24, 24, 32))
d = ImageDraw.Draw(sheet)
for i, c in enumerate(all_cells):
    cc = c.copy(); cc.thumbnail((140, 148))
    tmp = Image.new("RGBA", cc.size, (24, 24, 32, 255)); tmp.alpha_composite(cc)
    x, y = (i % 8) * 150, (i // 8) * 165
    sheet.paste(tmp.convert("RGB"), (x, y))
    d.text((x + 3, y + 150), str(i + 1), fill=(255, 80, 80))
sheet.save(f"{B}/cells/qc.png")
print("qc saved", sheet.size)
