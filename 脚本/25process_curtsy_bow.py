import importlib.util
import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask

_spec = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned

_spec17 = importlib.util.spec_from_file_location("w17", "gemini/脚本/17build_wave_hello.py")
_w17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(_w17)
find_white_grid_boxes = _w17.find_white_grid_boxes
clean_white_bg = _w17.clean_white_bg

_spec18 = importlib.util.spec_from_file_location("n18", "gemini/脚本/18process_note_taking.py")
_n18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(_n18)
measure_head_metrics = _n18.measure_head_metrics

_spec20 = importlib.util.spec_from_file_location("t20", "gemini/脚本/20process_thread_needle.py")
_t20 = importlib.util.module_from_spec(_spec20)
_spec20.loader.exec_module(_t20)
align_full_body_at = _t20.align_full_body_at

_spec22 = importlib.util.spec_from_file_location("c22", "gemini/脚本/22process_eat_cookie.py")
_c22 = importlib.util.module_from_spec(_spec22)
_spec22.loader.exec_module(_c22)
interpolate_flow_multi = _c22.interpolate_flow_multi

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366


def align_curtsy_by_head(f: Image.Image, ref_hw: float) -> Image.Image:
    """
    Align curtsy/kneel frames by locking head width 1:1 to `ref_hw` and grounding boots at `GROUND_Y`.
    Prevents the character from inflating during knee-bend poses.
    """
    arr = np.array(f, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys = np.nonzero(mask)[0]
    body_h = max(1, int(ys.max() - ys.min() + 1))

    hw, _, _, _ = measure_head_metrics(f, body_ratio=1.0)
    scale_hw = ref_hw / max(1.0, hw)
    scale_h = TARGET_H / body_h
    # Clamp head-locked scale within reasonable bounds
    scale = float(np.clip(scale_hw, scale_h * 0.88, scale_h * 1.06))

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    _, head_cx, _, my1 = measure_head_metrics(resized, body_ratio=1.0)
    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(GROUND_Y - my1))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def main() -> dict:
    s1 = Image.open("gemini/素材/25curtsy_bow/part1.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/25curtsy_bow/part2.jpg").convert("RGBA")

    # Part 1 is pure white bg; Part 2 is pure black bg
    b1 = find_white_grid_boxes(s1, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    p1 = [clean_white_bg(s1.crop(b1[i]), bg_thresh=234) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    # Use p2[11] (pure black bg clean stand) as ref_stand for crispest outlines & head width reference
    ref_stand = remove_nebula_blobs_aligned(align_full_body_at(p2[11], GROUND_Y))
    ref_hw, _, _, _ = measure_head_metrics(ref_stand, body_ratio=1.0)

    a1: List[Image.Image] = []
    for i in range(12):
        if i == 0:
            a1.append(ref_stand)
        elif 4 <= i <= 11:
            # Curtsy knee-bend poses -> lock head width to ref_hw so character lowers naturally
            a1.append(align_curtsy_by_head(p1[i], ref_hw))
        else:
            a1.append(align_full_body_at(p1[i], GROUND_Y))

    # Select clean Part 2 keyframes: p2[0] (rising curtsy), p2[1], p2[3], p2[5] (blink), p2[6]
    a2_sel = [
        remove_nebula_blobs_aligned(align_curtsy_by_head(p2[0], ref_hw)),
        remove_nebula_blobs_aligned(align_full_body_at(p2[1], GROUND_Y)),
        remove_nebula_blobs_aligned(align_full_body_at(p2[3], GROUND_Y)),
        remove_nebula_blobs_aligned(align_full_body_at(p2[5], GROUND_Y)),
        remove_nebula_blobs_aligned(align_full_body_at(p2[6], GROUND_Y)),
    ]

    # 18 keyframes -> 17 intervals: 12 triple-intervals + 5 double-intervals = 18 + 36 + 10 = 64 frames (8x8)
    keys_18 = a1 + a2_sel + [ref_stand.copy()]
    assert len(keys_18) == 18

    double_intervals = {0, 7, 13, 15, 16}
    frames_64: List[Image.Image] = []
    for i in range(18):
        frames_64.append(keys_18[i])
        if i < 17:
            if i in double_intervals:
                frames_64.extend(interpolate_flow_multi(keys_18[i], keys_18[i + 1], [1.0 / 3.0, 2.0 / 3.0]))
            else:
                frames_64.extend(interpolate_flow_multi(keys_18[i], keys_18[i + 1], [0.25, 0.5, 0.75]))
    assert len(frames_64) == 64

    heights = []
    for fr in frames_64:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 8
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_64):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "25curtsy_bow_sheet（屈膝礼仪_图集）.webp")
    anim_path = os.path.join(out_dir, "25curtsy_bow（屈膝礼仪_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 64 frames variable durations (~8.45s): deep curtsy hold + shoulder stars 175ms, blink/ends 185ms, normal 120ms
    hold_set = {0, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 55, 56, 63}
    fast_set = set(range(4, 15)) | set(range(36, 48))
    durations = [175 if i in hold_set else (115 if i in fast_set else 130) for i in range(64)]

    frames_64[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_64[1:],
        duration=durations,
        loop=0,
        lossless=True,
        quality=100,
        disposal=2,
    )

    white_bg = Image.new("RGBA", atlas.size, (255, 255, 255, 255))
    white_bg.alpha_composite(atlas)
    white_path = os.path.join(prev_dir, "25curtsy_bow_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "25curtsy_bow_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "25curtsy_bow",
        "frames": 64,
        "grid": "8x8",
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
