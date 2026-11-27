import importlib.util
import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import clean_black_bg_and_glow, find_cell_boxes
from expand_48 import get_main_component_mask, interpolate_midpoint_no_ghost, measure_body_stats

# Load white-bg slicer & cleaner from 17build_wave_hello.py
_spec = importlib.util.spec_from_file_location("bwh", "gemini/脚本/17build_wave_hello.py")
_bwh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bwh)
find_white_grid_boxes = _bwh.find_white_grid_boxes
clean_white_bg = _bwh.clean_white_bg

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366


def extract_sheet_auto(path: str) -> List[Image.Image]:
    """Auto-detect pure white vs pure black background, slice 4x3 cells, and matte cleanly."""
    sheet = Image.open(path).convert("RGBA")
    arr = np.array(sheet.convert("RGB"), dtype=np.uint8)
    border_mean = float(np.concatenate([arr[:10, :, :].ravel(), arr[-10:, :, :].ravel()]).mean())

    if border_mean > 200.0:
        boxes = find_white_grid_boxes(sheet, cols=4, rows=3, thresh=242)
        return [clean_white_bg(sheet.crop(b), bg_thresh=236) for b in boxes]
    else:
        bg = int(np.percentile(arr[:12, :, :].max(axis=2), 95))
        t = max(22, bg + 14)
        boxes = find_cell_boxes(sheet, cols=4, rows=3, thresh=t)
        return [
            clean_black_bg_and_glow(
                sheet.crop(b),
                bg_thresh=max(24, bg + 12),
                hole_thresh=max(20, bg + 8),
                glow_cutoff=max(34, bg + 16),
            )
            for b in boxes
        ]


def align_main_body(f: Image.Image, ground_y: int = GROUND_Y) -> Image.Image:
    """
    Strictly lock the main character body height to TARGET_H (304px)
    and anchor boot soles to ground_y, centered horizontally at X=50%.
    Particles (!, sweat drops, storm cloud, sparkles) scale proportionally without shrinking the body.
    """
    bx0, by0, bx1, by1, _ = measure_body_stats(f)
    body_h = max(1, by1 - by0)
    scale = TARGET_H / body_h

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    r_arr = np.array(resized, dtype=np.uint8)
    mask = get_main_component_mask(r_arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    my0, my1 = int(ys.min()), int(ys.max())
    top_cut = my0 + max(4, int((my1 - my0 + 1) * 0.30))
    head_cx = float(np.median(xs[ys <= top_cut]))

    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(ground_y - my1))

    # Clamp only if top particle touches very top edge
    bbox = resized.getbbox()
    if bbox and paste_y + bbox[1] < 8:
        paste_y = 8 - bbox[1]

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def main() -> dict:
    p1_raw = extract_sheet_auto("gemini/素材/16error_panic/part1.jpg")
    p2_raw = extract_sheet_auto("gemini/素材/16error_panic/part2.jpg")

    # Align all 12 frames of Part 1 (with 4px startled hop on frame 1) + first 11 frames of Part 2
    aligned_p1 = [
        align_main_body(f, ground_y=(GROUND_Y - 4 if idx == 1 else GROUND_Y))
        for idx, f in enumerate(p1_raw)
    ]
    aligned_p2 = [align_main_body(f, ground_y=GROUND_Y) for f in p2_raw[:11]]

    # 24 keyframes: 12 (Part 1) + 11 (Part 2) + exact Frame 0 closure
    keys_24 = aligned_p1 + aligned_p2 + [aligned_p1[0].copy()]
    assert len(keys_24) == 24

    # Expand 24 keyframes -> 48 frames (24 keys + 23 optical-flow midpoints + 1 final hold)
    frames_48: List[Image.Image] = []
    for i in range(24):
        frames_48.append(keys_24[i])
        if i < 23:
            frames_48.append(interpolate_midpoint_no_ghost(keys_24[i], keys_24[i + 1]))
    frames_48.append(aligned_p1[0].copy())
    assert len(frames_48) == 48, len(frames_48)

    # Measure main-body height consistency across all 48 frames
    body_heights = []
    for fr in frames_48:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        body_heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "16error_panic_sheet（报错慌张_图集）.webp")
    anim_path = os.path.join(out_dir, "16error_panic（报错慌张_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Variable-speed durations (~7.39s total: fast panic waving 115ms, holds 205ms, normal 150ms)
    fast_set = set(range(4, 17))
    hold_set = {0, 1, 2, 18, 20, 22, 24, 28, 30, 34, 36, 46, 47}
    durations = [115 if i in fast_set else (205 if i in hold_set else 150) for i in range(48)]

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
    white_path = os.path.join(prev_dir, "16error_panic_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "16error_panic_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "16error_panic",
        "frames": 48,
        "grid": "8x6",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "body_height_min": min(body_heights),
        "body_height_max": max(body_heights),
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
