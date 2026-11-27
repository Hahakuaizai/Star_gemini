#!/usr/bin/env python3
"""11_poke_shock v3 (redo): full-legs rebuild.
User rejected v2: middle frames shrank to ~260px with legs/feet unreadable.
v3 sheets were generated with a hard full-height/full-legs lock in every
panel; processing mirrors tsundere v6 (per-segment constant scale -> 305,
feet y=372, binary alpha, standard cleans). 36 frames = 35 drawings +
doc bookend f36 := f01. Pacing per doc S11: burst 115, tremble ~205,
relief 200, settle 170."""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/hatch/workspace/star_cat_girl/tsundere")
import process_tsundere as base

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "proc")
os.makedirs(OUT, exist_ok=True)

CANVAS, CX, GROUND, VC = base.CANVAS, base.CX, base.GROUND, base.VC
TARGET_H = base.TARGET_H

SEG1 = "media-generation-poke-shock-v3-seg1-0-75dd6351-2a13-43c0-9c8c-1ad7bc5ef597.webp"
SEG2 = "media-generation-poke-shock-v3-seg2-0-d88d79f4-c900-4ed2-b004-0fdabdb49150.webp"
SEG3 = "media-generation-poke-shock-v3-seg3-0-40e76bd0-9539-4819-ad4c-079a78115179.webp"


def cell_info(rgba):
    alpha = rgba[:, :, 3]
    lab, cs = base.comps(alpha)
    m = max(cs, key=lambda d: d["size"])
    gbot = m["y1"]
    for c in cs:
        if (c["size"] >= 300 and m["x0"] - 40 <= c["cx"] <= m["x1"] + 40
                and c["y0"] <= m["y1"] + 60):
            gbot = max(gbot, c["y1"])
    return dict(main=m, h=m["y1"] - m["y0"] + 1, bot=gbot,
                bcx=(m["x0"] + m["x1"]) / 2.0)


def main():
    seg1 = base.extract_sheet(os.path.join(HERE, SEG1))
    seg2 = base.extract_sheet(os.path.join(HERE, SEG2))
    seg3 = base.extract_sheet(os.path.join(HERE, SEG3))

    SEQ = ([(c, "seg1") for c in seg1]
           + [(c, "seg2") for c in seg2]
           + [(c, "seg3") for c in seg3[:11]])  # seg3 p12 replaced by bookend
    assert len(SEQ) == 35, len(SEQ)

    DUR = ([250, 170, 170, 130]          # standing -> shiver
           + [115, 115]                  # shock burst
           + [205] * 12                  # frozen trembling
           + [200] * 10                  # relief, blush, tongue
           + [170] * 7                   # settle
           + [280])                      # bookend
    assert len(DUR) == 36, len(DUR)

    infos = [cell_info(rgba) for rgba, _ in SEQ]
    groups = {}
    for (_, grp), inf in zip(SEQ, infos):
        groups.setdefault(grp, []).append(inf["h"])
    gscale = {}
    for g, hs in groups.items():
        if g == "seg1":
            gscale[g] = TARGET_H / infos[0]["h"]
        else:
            gscale[g] = TARGET_H / float(np.median(hs))
    print("group scales:", {g: round(s, 3) for g, s in sorted(gscale.items())})

    frames = []
    for (rgba, grp), inf in zip(SEQ, infos):
        s = gscale[grp]
        im = Image.fromarray(rgba)
        nw, nh = max(1, round(VC * s)), max(1, round(VC * s))
        im2 = im.resize((nw, nh), Image.LANCZOS)
        ox = int(round(CX - inf["bcx"] * s))
        oy = int(round(GROUND - inf["bot"] * s))
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(im2, (ox, oy), im2)
        arr = np.array(canvas)
        arr[:, :, 3] = np.where(arr[:, :, 3] >= 128, 255, 0).astype(np.uint8)
        frames.append(Image.fromarray(arr))

    total_clean = 0
    for i, fr in enumerate(frames):
        arr = np.array(fr)
        alpha = arr[:, :, 3]
        lab, cs = base.comps(alpha)
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

    frames.append(frames[0].copy())  # f36 := f01 (doc bookend)
    d_last = np.abs(np.array(frames[-1]).astype(int) - np.array(frames[0]).astype(int)).max()
    print("f36 := own f01, diff:", d_last)
    semi = sum(int(((np.array(f)[:, :, 3] > 0) & (np.array(f)[:, :, 3] < 255)).sum()) for f in frames)
    print("semi-transparent px:", semi)
    print("total duration:", sum(DUR), "ms  frames:", len(frames))
    hs = []
    for f in frames:
        a = np.array(f)[:, :, 3]
        ys = np.where(a > 0)[0]
        hs.append(int(ys.max() - ys.min() + 1))
    print("frame heights:", hs)

    for i, fr in enumerate(frames):
        fr.save(os.path.join(OUT, f"f{i + 1:02d}.png"))
    atlas = Image.new("RGBA", (CANVAS * 6, CANVAS * 6), (0, 0, 0, 0))
    for i, fr in enumerate(frames):
        atlas.paste(fr, ((i % 6) * CANVAS, (i // 6) * CANVAS))
    atlas.save(os.path.join(HERE, "poke_shock_atlas.webp"), lossless=True)
    frames[0].save(os.path.join(HERE, "poke_shock_anim.webp"), save_all=True,
                   append_images=frames[1:], duration=DUR, loop=0, lossless=True)
    print("saved atlas (6x6) + anim + proc")


if __name__ == "__main__":
    main()
