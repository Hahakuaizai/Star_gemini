import importlib.util
import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask

# Reuse anti-aliased black-bg & nebula-removal & smooth optical flow from 02process_sweeping.py
_spec = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned
interpolate_smooth_no_burr = _swp.interpolate_smooth_no_burr

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366


def align_gomoku_frame(f: Image.Image, ref_scale: float, ground_y: int = GROUND_Y) -> Image.Image:
    """
    Measure character body height excluding the wide floating wooden gomoku board on the right/front,
    locking character height consistently to TARGET_H and grounding boots at ground_y.
    """
    arr = np.array(f, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    top_mask = mask[y0 + int(0.06 * h): y0 + int(0.32 * h), :]
    top_xs = np.nonzero(top_mask)[1]
    head_cx = float(np.median(top_xs)) if len(top_xs) > 0 else float(np.median(xs))

    band_half = int(f.width * 0.16)
    x_lo = max(0, int(round(head_cx - band_half)))
    x_hi = min(f.width, int(round(head_cx + band_half)))
    c_ys = np.nonzero(mask[:, x_lo:x_hi])[0]
    char_h = max(1, int(c_ys.max() - c_ys.min() + 1)) if len(c_ys) > 0 else h

    scale = float(np.clip(TARGET_H / char_h, ref_scale * 0.96, ref_scale * 1.04))
    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    r_arr = np.array(resized, dtype=np.uint8)
    rmask = get_main_component_mask(r_arr[:, :, 3])
    rys, rxs = np.nonzero(rmask)
    ry0, ry1 = int(rys.min()), int(rys.max())
    rh = ry1 - ry0 + 1
    rtop = rmask[ry0 + int(0.06 * rh): ry0 + int(0.32 * rh), :]
    rtop_xs = np.nonzero(rtop)[1]
    rhead_cx = float(np.median(rtop_xs)) if len(rtop_xs) > 0 else float(np.median(rxs))

    r_half = int(resized.width * 0.16)
    rx_lo = max(0, int(round(rhead_cx - r_half)))
    rx_hi = min(resized.width, int(round(rhead_cx + r_half)))
    rc_ys = np.nonzero(rmask[:, rx_lo:rx_hi])[0]
    rboot_y1 = int(rc_ys.max()) if len(rc_ys) > 0 else ry1

    paste_x = int(round(CANVAS * 0.5 - rhead_cx))
    paste_y = int(round(ground_y - rboot_y1))

    bbox = resized.getbbox()
    if bbox and paste_y + bbox[1] < 8:
        paste_y = 8 - bbox[1]

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def main() -> dict:
    s1 = Image.open("gemini/素材/19gomoku/part1.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/19gomoku/part2.jpg").convert("RGBA")

    p1 = [clean_black_bg_rich_particles(s1.crop(b)) for b in find_cell_boxes(s1, 4, 3)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b)) for b in find_cell_boxes(s2, 4, 3)]

    # Measure ref_scale from p1[0]
    arr0 = np.array(p1[0], dtype=np.uint8)
    m0 = get_main_component_mask(arr0[:, :, 3])
    ys0 = np.nonzero(m0)[0]
    ref_scale = TARGET_H / max(1.0, float(ys0.max() - ys0.min() + 1))

    # Align Part 1 (with 6px celebration hop on p1[11]) & Part 2[:11]
    a1 = [
        remove_nebula_blobs_aligned(
            align_gomoku_frame(f, ref_scale, ground_y=(GROUND_Y - 6 if idx == 11 else GROUND_Y))
        )
        for idx, f in enumerate(p1)
    ]
    a2 = [
        remove_nebula_blobs_aligned(align_gomoku_frame(f, ref_scale, ground_y=GROUND_Y))
        for f in p2[:11]
    ]

    keys_24 = a1 + a2 + [a1[0].copy()]
    assert len(keys_24) == 24

    # Expand 24 keyframes -> 48 frames (8x6) with anti-aliased optical flow
    frames_48: List[Image.Image] = []
    for i in range(24):
        frames_48.append(keys_24[i])
        if i < 23:
            frames_48.append(interpolate_smooth_no_burr(keys_24[i], keys_24[i + 1]))
    frames_48.append(a1[0].copy())
    assert len(frames_48) == 48

    heights = []
    for fr in frames_48:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "19gomoku_sheet（下五子棋_图集）.webp")
    anim_path = os.path.join(out_dir, "19gomoku（下五子棋_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Doc target ~9.0-9.5s: stone placements 145ms, holds (push glasses/5-in-a-row/victory/take-off/ends) 235ms, rest 180ms -> ~9.07s
    fast_set = {8, 9, 10, 11, 14, 15, 16, 17, 18, 19}
    hold_set = {0, 1, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 36, 37, 46, 47}
    durations = [145 if i in fast_set else (235 if i in hold_set else 180) for i in range(48)]

    frames_48[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_48[1:],
        duration=durations,
        loop=0,
        lossless=True,
        quality=100,
        disposal=2,
    )

    white_bg = Image.new("RGBA", atlas.size, (255, 255, 255, 255))
    white_bg.alpha_composite(atlas)
    white_path = os.path.join(prev_dir, "19gomoku_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "19gomoku_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "19gomoku",
        "frames": 48,
        "grid": "8x6",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "height_min": min(heights),
        "height_max": max(heights),
        "total_seconds": round(sum(durations) / 1000.0, 2),
        "sheet_path": sheet_path,
        "anim_path": anim_path,
        "preview_white": white_path,
        "preview_black": black_path,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    main()
