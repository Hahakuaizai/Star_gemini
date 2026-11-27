#!/usr/bin/env python3
"""Action 11 poke_shock v5: same-pose = same drawing.
21 distinct drawings from ONE relay family (seg1 -> seg1s -> seg2, all relayed
from the anchor standing). Standing frames reuse the identical master drawing;
trembling is the same shock drawing with +-3px offsets, not new drawings.
Atlas is 2 rows x 18 per the user's layout instruction."""
import sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/hatch/workspace/star_cat_girl/tsundere")
src = open("/home/hatch/workspace/star_cat_girl/tsundere/process_tsundere.py").read()
base = {}
exec(src.split("def main")[0], base)
comps = base["comps"]

HERE = os.path.dirname(os.path.abspath(__file__))
CANVAS, CX, GROUND, TARGET_H = base["CANVAS"], base["CX"], base["GROUND"], base["TARGET_H"]

# (sheet, panel, dx) in playback order; f36 := f01 appended at build time
SEQ = [
    ("s1", 1, 0), ("s1", 2, 0), ("s1", 1, 0), ("s1", 4, 0),                       # A
    ("ss", 1, 0), ("ss", 1, -3), ("ss", 2, 0), ("ss", 2, 3), ("ss", 4, 0),
    ("ss", 4, -3), ("ss", 8, 0), ("ss", 8, 3), ("ss", 2, -2), ("s1", 11, 0),
    ("s1", 11, 2), ("ss", 12, 0), ("ss", 12, -2), ("ss", 12, 1),                  # B
    ("s2", 1, 0), ("s2", 2, 0), ("s2", 5, 0), ("s2", 4, 0), ("s2", 6, 0),
    ("s2", 7, 0), ("s2", 8, 0), ("s1", 3, 0), ("s1", 12, 0), ("s1", 2, 0),        # C
    ("s2", 10, 0), ("s2", 11, 0), ("s2", 12, 0), ("s1", 1, 0), ("s1", 2, 0),
    ("s1", 1, 0), ("s1", 3, 0),                                                   # D
]
DUR = ([250, 170, 170, 130]
       + [115, 115, 150, 150, 115, 115, 150, 150, 170, 190, 190, 205, 205, 205]
       + [200, 200, 210, 200, 210, 200, 210, 200, 210, 200]
       + [170, 170, 180, 180, 180, 180, 190]
       + [280])
assert len(SEQ) == 35 and len(DUR) == 36

EYE_FIX = {
    ("s1", 4):  (283, 272, 17, 16, +53, 265, 360),   # L eye pink -> gold
    ("ss", 4):  (283, 272, 17, 16, -76, 8, 78),      # R eye gold -> pink
    ("ss", 8):  (276, 262, 17, 16, -76, 8, 78),      # R eye gold -> pink
    ("s2", 5):  (368, 255, 20, 17, -76, 8, 78),      # R eye gold -> pink
}


def load_cell(tag, idx):
    return np.asarray(Image.open(os.path.join(HERE, f"cell_{tag}_{idx:02d}.png"))).copy()


def fix_eye(rgba, cx, cy, rx, ry, delta, lo, hi):
    out = rgba.copy()
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    sel = ell & (rgba[:, :, 3] > 0)
    px = rgba[:, :, :3].astype(np.float32) / 255.0
    r, g, b = px[:, :, 0], px[:, :, 1], px[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    hue = np.zeros_like(mx)
    m = (mx == r) & (df > 0); hue[m] = (60 * ((g - b)[m] / df[m]) + 360) % 360
    m = (mx == g) & (df > 0); hue[m] = 60 * ((b - r)[m] / df[m]) + 120
    m = (mx == b) & (df > 0); hue[m] = 60 * ((r - g)[m] / df[m]) + 240
    sat = np.where(mx > 0, df / np.maximum(mx, 1e-9), 0)
    gate = sel & (hue >= lo) & (hue <= hi) & (sat > 0.18) & (mx > 0.15)
    if gate.sum() == 0:
        print("  eye fix: no gated px!")
        return out
    newh = (hue[gate] + delta) % 360
    hh = newh / 60.0
    ss_ = sat[gate]; vv = mx[gate]
    ii = np.floor(hh).astype(int) % 6
    ff = hh - np.floor(hh)
    p = vv * (1 - ss_); q = vv * (1 - ss_ * ff); t = vv * (1 - ss_ * (1 - ff))
    rr = np.choose(ii, [vv, q, p, p, t, vv])
    gg = np.choose(ii, [t, vv, vv, q, p, p])
    bb = np.choose(ii, [p, p, t, vv, vv, q])
    out[:, :, 0][gate] = (rr * 255).astype(np.uint8)
    out[:, :, 1][gate] = (gg * 255).astype(np.uint8)
    out[:, :, 2][gate] = (bb * 255).astype(np.uint8)
    print(f"  eye fix: {int(gate.sum())} px rotated {delta:+d} deg")
    return out


def clean(rgba):
    alpha = rgba[:, :, 3].astype(np.float32)
    rgb = rgba[:, :, :3].astype(np.int16)
    mean_max = rgb.max(axis=2).astype(np.float32)
    lab, cs = comps(rgba[:, :, 3])
    if not cs:
        return rgba
    main = max(cs, key=lambda c: c["size"])
    keep = np.zeros(alpha.shape, dtype=bool)
    for c in cs:
        drop = False
        if c["id"] != main["id"]:
            if 20 <= c["size"] <= 250 and mean_max[c["mask"]].mean() < 145 and c["cy"] < main["y0"]:
                drop = True
            wdt = c["x1"] - c["x0"] + 1; hgt = c["y1"] - c["y0"] + 1
            if wdt <= 14 and hgt >= 20 and (c["x1"] < 75 or c["x0"] > 629):
                drop = True
        if not drop:
            keep |= c["mask"]
    out = rgba.copy()
    out[:, :, 3] = np.where(keep, rgba[:, :, 3], 0).astype(np.uint8)
    return out


def place(rgba, scale, dx=0):
    im = Image.fromarray(rgba)
    nw, nh = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    arr = np.asarray(im).copy()
    arr[:, :, 3] = np.where(arr[:, :, 3] >= 128, 255, 0).astype(np.uint8)
    alpha = arr[:, :, 3]
    lab, cs = comps(alpha)
    main = max(cs, key=lambda c: c["size"])
    grounds = [c for c in cs if c["size"] >= 300
               and c["x0"] <= main["cx"] + 40 and c["x1"] >= main["cx"] - 40
               and c["y0"] <= main["y1"] + 60]
    gbot = max(c["y1"] for c in grounds) if grounds else main["y1"]
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    px = round(CX - main["cx"]) + dx
    py = round(GROUND - gbot)
    canvas.alpha_composite(Image.fromarray(arr), (px, py))
    out = np.asarray(canvas)
    ys, xs = np.nonzero(out[:, :, 3] > 0)
    return Image.fromarray(out), (ys.max() - ys.min() + 1)


def main():
    cells = {}
    for tag, idx, _ in SEQ:
        if (tag, idx) not in cells:
            cells[(tag, idx)] = load_cell(tag, idx)
    for key, (cx, cy, rx, ry, delta, lo, hi) in EYE_FIX.items():
        print("fix", key)
        cells[key] = fix_eye(cells[key], cx, cy, rx, ry, delta, lo, hi)
    # per-sheet scale from the USED panels' median main height
    scales = {}
    for tag in ("s1", "ss", "s2"):
        hs = []
        for (t, idx) in cells:
            if t == tag:
                lab, cs = comps(cells[(t, idx)][:, :, 3])
                m = max(cs, key=lambda c: c["size"])
                hs.append(m["y1"] - m["y0"] + 1)
        med = float(np.median(hs))
        scales[tag] = TARGET_H / med
        print(tag, "used mains", hs, "median", med, "scale", round(scales[tag], 4))
    frames = []
    for tag, idx, dx in SEQ:
        fr, h = place(clean(cells[(tag, idx)]), scales[tag], dx)
        frames.append(fr)
        print(tag, idx, "dx", dx, "placed h", h)
    frames.append(frames[0].copy())
    os.makedirs(os.path.join(HERE, "proc5"), exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(os.path.join(HERE, "proc5", f"frame_{i+1:02d}.png"))
    # atlas: 2 rows x 18 (user's layout instruction)
    atlas = Image.new("RGBA", (CANVAS * 18, CANVAS * 2), (0, 0, 0, 0))
    for i, fr in enumerate(frames):
        r, c = divmod(i, 18)
        atlas.alpha_composite(fr, (c * CANVAS, r * CANVAS))
    atlas.save(os.path.join(HERE, "poke_shock_atlas_v5.webp"), lossless=True)
    frames[0].save(os.path.join(HERE, "poke_shock_anim_v5.webp"), save_all=True,
                   append_images=frames[1:], duration=DUR, loop=0, lossless=True)
    print("TOTAL", sum(DUR), "ms")


if __name__ == "__main__":
    main()
