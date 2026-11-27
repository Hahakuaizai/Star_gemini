#!/usr/bin/env python3
"""11_poke_shock: whole-sheet extraction; all frames grounded (feet 372);
face-band-constant scale; f01/f36 := 10's standing master (cross-action chaining);
dash/sliver/floor-glint clean; atlas+anim."""
import os
import numpy as np
from PIL import Image
from scipy import ndimage

HOME = os.path.expanduser("~")
WD = f"{HOME}/workspace/star_cat_girl/shy_praise"
OUT = f"{HOME}/workspace/star_cat_girl"
MASTER = f"{HOME}/workspace/star_cat_girl/happy_jump/standing_master.png"
SHEETS = [
    f"{WD}/media-generation-shy-praise-seg1c-0-cc6caf46-b551-4b82-a540-b276f168ada2.webp",
    f"{WD}/media-generation-shy-praise-seg2b-0-809dd5fa-f96a-4745-8d90-d16c153904a5.webp",
    f"{WD}/media-generation-shy-praise-seg3-0-22a70cda-5fb8-4714-abff-14fa35cb1a4e.webp",
]
CANVAS = 384
TARGET_H = 305
GROUND = 372
HCX = 183
VC = 704
VO = 128

DURATIONS = ([260] + [145] * 6 + [230] * 8 + [145] + [230] * 3 + [145]
             + [230] * 4 + [190] * 6 + [190] * 5 + [280])


def cutout(arr):
    mx = arr.max(axis=2)
    dark = mx < 46
    lab, n = ndimage.label(dark)
    border_labels = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    border_labels.discard(0)
    bg = np.isin(lab, list(border_labels)) if border_labels else np.zeros_like(dark)
    sat = (mx - arr.min(axis=2)) / np.maximum(mx, 1)
    alpha = np.ones(mx.shape, dtype=np.float32)
    alpha[bg] = 0.0
    for i in range(1, n + 1):
        if i in border_labels:
            continue
        m = lab == i
        if m.sum() < 8:
            alpha[m] = 0.0
        elif dark[m].all() and mx[m].mean() < 40 and sat[m].mean() < 0.16:
            alpha[m] = 0.0
    op = alpha > 0
    lab2, n2 = ndimage.label(op)
    for i in range(1, n2 + 1):
        m = lab2 == i
        if m.sum() < 8:
            alpha[m] = 0.0
    return np.dstack([arr, (alpha * 255).astype(np.uint8)])


def comps(alpha):
    lab, n = ndimage.label(alpha > 0)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        out.append(dict(id=i, mask=m, size=len(xs), x0=xs.min(), x1=xs.max(),
                        y0=ys.min(), y1=ys.max(), cx=xs.mean(), cy=ys.mean()))
    return lab, out


def split_fused(cs):
    out = []
    for c in cs:
        hgt = c["y1"] - c["y0"]
        if c["size"] > 45000 and hgt > 500:
            m = c["mask"]
            lo = c["y0"] + int(0.25 * hgt)
            hi = c["y0"] + int(0.60 * hgt)
            widths = [(y, int(m[y].sum())) for y in range(lo, hi)]
            ystar, wstar = min(widths, key=lambda t: t[1])
            if wstar < 60:
                print(f"split fused comp at y={ystar} (neck width {wstar})")
                top = m.copy(); top[ystar + 1:, :] = False
                bot = m.copy(); bot[:ystar + 1, :] = False
                for pm in (top, bot):
                    ys, xs = np.nonzero(pm)
                    if len(xs) > 500:
                        out.append(dict(mask=pm, size=len(xs), x0=xs.min(), x1=xs.max(),
                                        y0=ys.min(), y1=ys.max(), cx=xs.mean(), cy=ys.mean()))
                continue
        out.append(c)
    return out


def extract_sheet(path):
    im = np.asarray(Image.open(path).convert("RGB"))
    rgba = cutout(im)
    alpha = rgba[:, :, 3]
    lab, cs = comps(alpha)
    cs = split_fused(cs)
    mains = sorted([c for c in cs if c["size"] > 20000], key=lambda d: d["cy"])
    assert len(mains) == 12, f"{os.path.basename(path)}: {len(mains)} mains"
    rows = []
    for r in range(3):
        rows.append(sorted(mains[r * 4:(r + 1) * 4], key=lambda d: d["cx"]))
    ordered = [m for row in rows for m in row]
    main_ids = {id(m) for m in ordered}
    owner = {}
    for c in cs:
        if id(c) in main_ids:
            continue
        d2 = [((c["cx"] - m["cx"]) ** 2 + (c["cy"] - m["cy"]) ** 2, k) for k, m in enumerate(ordered)]
        owner[id(c)] = min(d2)[1]
    cells = []
    H, W = alpha.shape
    for k, m in enumerate(ordered):
        r, cc = divmod(k, 4)
        ox, oy = cc * (W // 4), r * (H // 3)
        cell = np.zeros((VC, VC, 4), dtype=np.uint8)
        parts = [m] + [c for c in cs if owner.get(id(c)) == k]
        mask = np.zeros(alpha.shape, dtype=bool)
        for p in parts:
            mask |= p["mask"]
        ys, xs = np.nonzero(mask)
        vx, vy = xs - ox + VO, ys - oy + VO
        ok = (vx >= 0) & (vx < VC) & (vy >= 0) & (vy < VC)
        cell[vy[ok], vx[ok]] = rgba[ys[ok], xs[ok]]
        cells.append(cell)
    return cells


def face_metrics(alpha, main):
    # measure on the MAIN comp's mask only — detached hearts/sparkles beside the
    # head must not inflate the face band
    m = main["mask"]
    y0, y1 = main["y0"], main["y1"]
    h = y1 - y0 + 1
    spans = []
    for y in range(y0 + int(0.08 * h), y0 + int(0.38 * h)):
        xs = np.nonzero(m[y])[0]
        if len(xs) > 3:
            spans.append(xs.max() - xs.min())
    band = float(np.median(spans)) if spans else float(main["x1"] - main["x0"])
    top_rows = m[y0:y0 + int(0.30 * h)]
    xs = np.nonzero(top_rows)[1]
    hcx = float(np.median(xs)) if len(xs) else main["cx"]
    return band, hcx


def main():
    assert len(DURATIONS) == 36, len(DURATIONS)
    cells = []
    for sp in SHEETS:
        cells.extend(extract_sheet(sp))
    assert len(cells) == 36

    info = []
    for rgba in cells:
        alpha = rgba[:, :, 3]
        lab, cs = comps(alpha)
        m = max(cs, key=lambda d: d["size"])
        band, hcx = face_metrics(alpha, m)
        # ground bottom: boots are separate comps in this art style (strap gap) —
        # anchor on the lowest sizeable comp near the main body, not the main comp itself
        gbot = m["y1"]
        for c in cs:
            if (c["size"] >= 300 and m["x0"] - 40 <= c["cx"] <= m["x1"] + 40
                    and c["y0"] <= m["y1"] + 60):
                gbot = max(gbot, c["y1"])
        info.append(dict(main=m, band=band, hcx=hcx, h=m["y1"] - m["y0"] + 1, bot=gbot))
    bands = np.array([d["band"] for d in info])
    sm = np.array([np.median(bands[max(0, i - 2):min(36, i + 3)]) for i in range(36)])
    B0 = sm[0]
    s1 = TARGET_H / info[0]["h"]
    scales = [float(np.clip(s1 * (B0 / sm[i]) ** 1.0, 0.62, 0.88)) for i in range(36)]
    print(f"B0={B0:.0f} s1={s1:.3f} band range {bands.min():.0f}..{bands.max():.0f}")

    frames = []
    for i in range(36):
        d = info[i]
        s = scales[i]
        im = Image.fromarray(cells[i])
        nw, nh = max(1, round(VC * s)), max(1, round(VC * s))
        im2 = im.resize((nw, nh), Image.LANCZOS)
        ox = int(round(HCX - d["hcx"] * s))
        oy = int(round(GROUND - d["bot"] * s))
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(im2, (ox, oy), im2)
        arr = np.array(canvas)
        arr[:, :, 3] = np.where(arr[:, :, 3] >= 128, 255, 0).astype(np.uint8)
        frames.append(Image.fromarray(arr))

    # cleans
    total_clean = 0
    for i, fr in enumerate(frames):
        arr = np.array(fr)
        alpha = arr[:, :, 3]
        lab, cs = comps(alpha)
        if not cs:
            continue
        main_c = max(cs, key=lambda d: d["size"])
        rgbmax = arr[:, :, :3].max(axis=2)
        gbot_f = main_c["y1"]
        for c in cs:
            if (c["size"] >= 300 and main_c["x0"] - 40 <= c["cx"] <= main_c["x1"] + 40
                    and c["y0"] <= main_c["y1"] + 60):
                gbot_f = max(gbot_f, c["y1"])
        for c in cs:
            if c["id"] == main_c["id"]:
                continue
            wdt = c["x1"] - c["x0"] + 1
            hgt = c["y1"] - c["y0"] + 1
            kill = False
            if c["y0"] > gbot_f + 2:
                kill = True
            if 20 <= c["size"] <= 250 and c["cy"] < main_c["y0"] and rgbmax[c["mask"]].mean() < 145:
                kill = True
            if wdt <= 14 and hgt >= 20 and (c["x1"] < 75 or c["x0"] > 309):
                kill = True
            if kill:
                arr[c["mask"], 3] = 0
                total_clean += c["size"]
        frames[i] = Image.fromarray(arr)
    print("cleaned px:", total_clean)

    # doc rule: first == last frame, this action's OWN standing frame
    frames[35] = frames[0].copy()
    d36 = np.abs(np.array(frames[35]).astype(int) - np.array(frames[0]).astype(int)).max()
    print("f36 := own f01, diff:", d36)

    semi = sum(int(((np.array(f)[:, :, 3] > 0) & (np.array(f)[:, :, 3] < 255)).sum()) for f in frames)
    print("semi-transparent px:", semi)
    for i in range(36):
        a = np.array(frames[i])[:, :, 3]
        ys = np.nonzero(a.any(axis=1))[0]
        xs = np.nonzero(a.any(axis=0))[0]
        if ys.max() > 380 or ys.min() < 2:
            print(f"EDGE f{i+1:02d} top={ys.min()} bot={ys.max()}")
    print("total duration:", sum(DURATIONS), "ms")
    atlas = Image.new("RGBA", (CANVAS * 6, CANVAS * 6), (0, 0, 0, 0))
    for i, fr in enumerate(frames):
        atlas.paste(fr, ((i % 6) * CANVAS, (i // 6) * CANVAS))
    atlas.save(f"{OUT}/shy_praise_atlas.webp", lossless=True)
    frames[0].save(f"{OUT}/shy_praise_anim.webp", save_all=True, append_images=frames[1:],
                   duration=DURATIONS, loop=0, lossless=True)
    os.makedirs(f"{WD}/proc", exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(f"{WD}/proc/f{i+1:02d}.png")
    print("saved atlas+anim+proc")


if __name__ == "__main__":
    main()
