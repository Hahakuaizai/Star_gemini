import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
SETS = {
    "07sleep_start": ["part1.webp", "part2.webp", "part3.webp"],
    "08sleep_loop": ["part1.webp", "part2.webp", "part3.webp", "part4.webp"],
    "09sleep_wake": ["part1.webp", "part2.webp", "part3.webp"],
}


def load_cells(name, files):
    cells = []
    for f in files:
        im = Image.open(os.path.join(G, "素材", name, f)).convert("RGB")
        w, h = im.size
        cw, ch = w // 4, h // 3
        for r in range(3):
            for c in range(4):
                cells.append(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
    return cells


def matte(cell, cell_size=448):
    a0 = np.array(cell.convert("RGB"), dtype=np.float32)
    den = cv2.bilateralFilter(a0.astype(np.uint8), 5, 26, 3).astype(np.float32)
    L = den.max(axis=2)
    seed = L > 55
    lab, n = ndimage.label(seed)
    body = np.zeros_like(seed)
    for i, s in enumerate(ndimage.sum(seed, lab, range(1, n + 1)), start=1):
        if s >= 0.08 * cell_size * cell_size:
            ys, xs = np.where(lab == i)
            if ys.max() - ys.min() >= 0.45 * cell_size:
                body |= (lab == i)
    body = ndimage.binary_fill_holes(body)
    w = ndimage.gaussian_filter(body.astype(np.float32), 1.6)
    ac = 1.0 / (1.0 + np.exp(-(L - 42.0) / 8.0))
    ac = ndimage.gaussian_filter(ac, 0.7)
    ag = np.clip((L - 12.0) / 235.0, 0.0, 0.95)
    ag = ndimage.gaussian_filter(ag, 0.5)
    al = w * ac + (1.0 - w) * ag
    al[al < 0.05] = 0.0
    solid = al > 0.72
    idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    col = den[idx[0], idx[1]]
    col = np.where(solid[..., None], den, col)
    out = np.zeros((L.shape[0], L.shape[1], 4), dtype=np.uint8)
    out[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
    out[..., 3] = (al * 255.0).astype(np.uint8)
    return out


if __name__ == "__main__":
    for name, files in SETS.items():
        cells = load_cells(name, files)
        od = os.path.join(OUT, name, "_raw")
        os.makedirs(od, exist_ok=True)
        print("==", name, len(cells))
        for i, cell in enumerate(cells):
            a = matte(cell, cell.width)
            m = a[..., 3] > 110
            ys, xs = np.where(m)
            Image.fromarray(a, "RGBA").save(os.path.join(od, "raw_%02d.png" % (i + 1)))
            print(i + 1, "w", xs.max() - xs.min() + 1, "h", ys.max() - ys.min() + 1,
                  "cx", round((xs.min() + xs.max()) / 2), "cy0", ys.min(), "cy1", ys.max(),
                  "mass", int(a[..., 3].sum() // 255))

