import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
D = os.path.join(OUT, "_diag0506")
CANVAS = 384
PAIRS = [("05thinking", 4, 5), ("05thinking", 16, 17), ("05thinking", 28, 29), ("05thinking", 36, 37),
         ("06busy", 17, 18), ("06busy", 28, 29), ("06busy", 39, 40), ("06busy", 43, 44)]


def load(act, i):
    return np.array(Image.open(os.path.join(OUT, act, "_norm", "norm_%02d.png" % i)).convert("RGBA"),
                    dtype=np.float32)


def premul(a):
    o = a.copy()
    o[..., :3] = a[..., :3] * (a[..., 3:4] / 255.0)
    return o


def unpremul(a):
    o = a.copy()
    al = a[..., 3:4] / 255.0
    o[..., :3] = np.where(al > 1e-4, a[..., :3] / np.maximum(al, 1e-4), 0.0)
    o[..., :3] = np.clip(o[..., :3], 0, 255)
    return o


def gray(a):
    g = a[..., 3:4] / 255.0 * 255.0
    lum = 0.3 * a[..., 0:1] + 0.59 * a[..., 1:2] + 0.11 * a[..., 2:3]
    return np.clip(0.55 * g + 0.45 * lum, 0, 255).astype(np.uint8)[..., 0]


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(gray(a), gray(b), None, 0.5, 4, 25, 5, 7, 1.5, 0)


def warp_flow(img, fl, t, border=0):
    h, w = fl.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = x + fl[..., 0] * t
    my = y + fl[..., 1] * t
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def warp_field(fl, fl2, t):
    h, w = fl.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = x + fl2[..., 0] * t
    my = y + fl2[..., 1] * t
    out = np.zeros_like(fl)
    out[..., 0] = cv2.remap(fl[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[..., 1] = cv2.remap(fl[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def interp(a, b, t):
    pa, pb = premul(a), premul(b)
    fab = flow(a, b)
    fba = flow(b, a)
    fba_at_a = warp_field(fba, fab, 1.0)
    err = np.linalg.norm(fab + fba_at_a, axis=2)
    occ = err > 1.3
    pa_t = warp_flow(pa, fab, t, 0)
    pb_t = warp_flow(pb, fba, 1.0 - t, 0)
    mix = pa_t * (1.0 - t) + pb_t * t
    pick = np.where(occ[..., None], np.where(t < 0.5, pa_t, pb_t), mix)
    out = unpremul(pick)
    return np.clip(out, 0, 255)


tiles = []
for act, ia, ib in PAIRS:
    a, b = load(act, ia), load(act, ib)
    for t in (0.34, 0.5, 0.66):
        m = interp(a, b, t)
        for tag, img in (("m", m),):
            c = Image.new("RGBA", (CANVAS, CANVAS), (250, 250, 250, 255))
            c.alpha_composite(Image.fromarray(img.astype(np.uint8), "RGBA").resize((CANVAS, CANVAS), Image.LANCZOS))
            tiles.append((f"{act} {ia}->{ib} t{t}", c.convert("RGB").resize((200, 200), Image.LANCZOS)))
sheet = Image.new("RGB", (200 * 4, (200 + 18) * (len(tiles) // 4 + 1)), (80, 80, 80))
dr = ImageDraw.Draw(sheet)
for k, (txt, im) in enumerate(tiles):
    x = (k % 4) * 200
    y = (k // 4) * 218
    sheet.paste(im, (x, y + 18))
    dr.text((x + 3, y + 3), txt, fill=(255, 230, 0))
sheet.save(os.path.join(D, "interp_try.png"))
print("ok", sheet.size, len(tiles))
