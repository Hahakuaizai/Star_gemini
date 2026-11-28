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
interpolate_smooth_no_burr = _swp.interpolate_smooth_no_burr

_spec18 = importlib.util.spec_from_file_location("n18", "gemini/脚本/18process_note_taking.py")
_n18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(_n18)
measure_head_metrics = _n18.measure_head_metrics
align_cropped_with_ref_boots = _n18.align_cropped_with_ref_boots

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366


def align_full_body_at(f: Image.Image, ground_y: int = GROUND_Y) -> Image.Image:
    """Align full-body frame to TARGET_H (304px) with boots grounded at `ground_y`."""
    arr = np.array(f, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys = np.nonzero(mask)[0]
    body_h = max(1, int(ys.max() - ys.min() + 1))
    scale = TARGET_H / body_h

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    _, head_cx, _, my1 = measure_head_metrics(resized, body_ratio=1.0)
    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(ground_y - my1))

    bbox = resized.getbbox()
    if bbox and paste_y + bbox[1] < 8:
        paste_y = 8 - bbox[1]

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def fix_fading_moons(a2: List[Image.Image]) -> None:
    """
    Replace dim/greyish fading moon regions (x >= 265, y < 130) in a2[2..4]
    with clean scaled-alpha copies of the crisp warm-yellow moon from a2[0],
    eliminating dark-grey halos on white backgrounds.
    """
    src_arr = np.array(a2[0], dtype=np.uint8)
    moon_tpl = src_arr[:130, 265:CANVAS, :].astype(np.float32)

    fade_scales = {2: 0.62, 3: 0.34, 4: 0.14}
    for idx, scale in fade_scales.items():
        arr = np.array(a2[idx], dtype=np.uint8)
        arr[:130, 265:CANVAS, :] = 0
        m = moon_tpl.copy()
        m[:, :, 3] = np.clip(m[:, :, 3] * scale, 0.0, 255.0)
        arr[:130, 265:CANVAS, :] = m.astype(np.uint8)
        a2[idx] = Image.fromarray(arr, "RGBA")


def main() -> dict:
    s1 = Image.open("gemini/素材/20thread_needle/part1_noglasses.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/20thread_needle/part2_noglasses.jpg").convert("RGBA")

    b1 = find_cell_boxes(s1, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    p1 = [clean_black_bg_rich_particles(s1.crop(b1[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    ref_stand = remove_nebula_blobs_aligned(align_full_body_at(p1[0], GROUND_Y))

    a1: List[Image.Image] = []
    for i in range(12):
        if i == 0:
            a1.append(ref_stand)
        elif i == 3:
            # Skip p1[3] (had left-tail & swapped-eye mirror drift)
            a1.append(None)  # type: ignore[arg-type]
        elif i == 9:
            # Celebration small vertical hop: raise by 12px while keeping jumping boots untouched
            a1.append(remove_nebula_blobs_aligned(align_full_body_at(p1[i], GROUND_Y - 12)))
        elif i in (8, 10, 11):
            # Row 3 sitting close to bottom edge -> safely ensure full boots & tail via head-locked composite
            a1.append(remove_nebula_blobs_aligned(align_cropped_with_ref_boots(p1[i], ref_stand, body_ratio=0.97)))
        else:
            a1.append(remove_nebula_blobs_aligned(align_full_body_at(p1[i], GROUND_Y)))

    a2: List[Image.Image] = []
    for i in range(11):
        if i >= 8:
            # Row 3 of part2 sits near bottom edge -> ensure complete boots without clipping
            a2.append(remove_nebula_blobs_aligned(align_cropped_with_ref_boots(p2[i], ref_stand, body_ratio=0.97)))
        else:
            a2.append(remove_nebula_blobs_aligned(align_full_body_at(p2[i], GROUND_Y)))

    fix_fading_moons(a2)

    # 23 clean keyframes (skipping p1[3]) -> 48 frames (20 single-midpoints + 2 double-midpoints + 1 final hold)
    keys_23 = a1[0:3] + a1[4:12] + a2[0:11] + [ref_stand.copy()]
    assert len(keys_23) == 23

    double_intervals = {1, 10}
    frames_48: List[Image.Image] = []
    for i in range(23):
        frames_48.append(keys_23[i])
        if i < 22:
            if i in double_intervals:
                mid = interpolate_smooth_no_burr(keys_23[i], keys_23[i + 1])
                frames_48.append(interpolate_smooth_no_burr(keys_23[i], mid))
                frames_48.append(interpolate_smooth_no_burr(mid, keys_23[i + 1]))
            else:
                frames_48.append(interpolate_smooth_no_burr(keys_23[i], keys_23[i + 1]))
    frames_48.append(ref_stand.copy())
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
        # Direct RGBA copy (no mask) preserves straight alpha RGB on semi-transparent fading moons
        atlas.paste(fr, (c * CANVAS, r * CANVAS))

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "20thread_needle_sheet（穿针乞巧_图集）.webp")
    anim_path = os.path.join(out_dir, "20thread_needle（穿针乞巧_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Doc target ~9.0s: thread-success & hop 145ms, holds (summon/display/wink/blink/ends) 235ms, rest 180ms -> ~9.17s
    fast_set = {12, 13, 14, 16, 17, 18}
    hold_set = {0, 1, 5, 6, 19, 21, 24, 25, 26, 27, 40, 41, 46, 47}
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
    white_path = os.path.join(prev_dir, "20thread_needle_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "20thread_needle_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "20thread_needle",
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
