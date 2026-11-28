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

_spec20 = importlib.util.spec_from_file_location("t20", "gemini/脚本/20process_thread_needle.py")
_t20 = importlib.util.module_from_spec(_spec20)
_spec20.loader.exec_module(_t20)
align_full_body_at = _t20.align_full_body_at

CANVAS = 384
GROUND_Y = 366


def main() -> dict:
    s1 = Image.open("gemini/素材/21drink_milktea/part1.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/21drink_milktea/part2.jpg").convert("RGBA")

    b1 = find_cell_boxes(s1, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    p1 = [clean_black_bg_rich_particles(s1.crop(b1[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    ref_stand = remove_nebula_blobs_aligned(align_full_body_at(p1[0], GROUND_Y))

    a1: List[Image.Image] = []
    for i in range(12):
        if i == 0:
            a1.append(ref_stand)
        elif i == 8:
            # Skip p1[8] (missing milk tea cup)
            a1.append(None)  # type: ignore[arg-type]
        else:
            a1.append(remove_nebula_blobs_aligned(align_full_body_at(p1[i], GROUND_Y)))

    a2: List[Image.Image] = [
        remove_nebula_blobs_aligned(align_full_body_at(p2[i], GROUND_Y)) for i in range(11)
    ]

    # 23 clean keyframes (skipping p1[8]) -> 48 frames
    keys_23 = a1[0:8] + a1[9:12] + a2[0:11] + [ref_stand.copy()]
    assert len(keys_23) == 23

    # Double midpoint on summoning cup (idx 1) and cup dissolving into stardust (idx 15)
    double_intervals = {1, 15}
    frames_48: List[Image.Image] = []
    for i in range(23):
        frames_48.append(keys_23[i])
        if i < 22:
            if i in double_intervals:
                frames_48.append(interpolate_smooth_no_burr(keys_23[i], keys_23[i + 1], t=1.0 / 3.0))
                frames_48.append(interpolate_smooth_no_burr(keys_23[i], keys_23[i + 1], t=2.0 / 3.0))
            else:
                frames_48.append(interpolate_smooth_no_burr(keys_23[i], keys_23[i + 1], t=0.5))
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
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "21drink_milktea_sheet（喝奶茶_图集）.webp")
    anim_path = os.path.join(out_dir, "21drink_milktea（喝奶茶_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Sipping boba fast 150ms, satisfied holds / wipe mouth / stardust / ends 225ms, normal 175ms (~8.55s)
    fast_set = set(range(10, 19))
    hold_set = {0, 1, 20, 21, 22, 23, 24, 25, 31, 32, 33, 40, 41, 46, 47}
    durations = [150 if i in fast_set else (225 if i in hold_set else 175) for i in range(48)]

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
    white_path = os.path.join(prev_dir, "21drink_milktea_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "21drink_milktea_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "21drink_milktea",
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
