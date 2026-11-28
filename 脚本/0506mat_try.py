import os, glob
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审", "_diag0506")
os.makedirs(OUT, exist_ok=True)


def cells_of(act):
    cs = []
    for p in sorted(glob.glob(os.path.join(G, "素材", act, "part*.webp"))):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        cw, ch = w // 4, h // 3
        for r in range(3):
            for c in range(4):
                cs.append(np.asarray(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)), dtype=np.float32))
    return cs


def matte(rgb, lo=12.0, span=34.0, klow=0.55):
    L = rgb.max(axis=2)
    al = np.clip((L - lo) / span, 0.0, 1.0)
    al = ndimage.gaussian_filter(al, 0.6)
    al[al < 0.06] = 0.0
    color = rgb / np.maximum(al, klow)[..., None]
    color = np.clip(color, 0, 255)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = color.astype(np.uint8)
    out[..., 3] = (al * 255).astype(np.uint8)
    return out


def preview_cell(rgba, bg):
    c = Image.new("RGBA", (rgba.shape[1], rgba.shape[0]), bg)
    c.alpha_composite(Image.fromarray(rgba, "RGBA"))
    return c.convert("RGB")


for act, idxs in (("05thinking", [1, 3, 4, 12, 29, 33]), ("06busy", [1, 18, 29, 32, 39, 44])):
    cs = cells_of(act)
    rows = []
    for i in idxs:
        rgba = matte(cs[i - 1].copy())
        a = rgba[..., 3]
        semi = int(((a > 0) & (a < 255)).sum())
        m = a > 100
        ys, xs = np.where(m)
        stat = f"f{i} semi={semi} w={xs.max()-xs.min()+1} h={ys.max()-ys.min()+1}"
        left = preview_cell(rgba, (255, 255, 255, 255)).resize((224, 224), Image.LANCZOS)
        right = preview_cell(rgba, (0, 0, 0, 255)).resize((224, 224), Image.LANCZOS)
        rows.append((left, right, stat))
        print(act, stat)
    W = 224 * 2 + 6
    sheet = Image.new("RGB", (W * 3, (224 + 22) * 2), (128, 128, 128))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for k, (l, r, s) in enumerate(rows):
        x = (k % 3) * W
        y = (k // 3) * (224 + 22)
        sheet.paste(l, (x, y + 22))
        sheet.paste(r, (x + 224 + 6, y + 22))
        dr.text((x + 4, y + 5), s.encode("ascii", "ignore").decode(), fill=(255, 255, 0))
    sp = os.path.join(OUT, f"{act}_matte_test.png")
    sheet.save(sp)
    print("->", sp)
