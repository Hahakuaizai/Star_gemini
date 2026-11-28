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

_spec20 = importlib.util.spec_from_file_location("t20", "gemini/脚本/20process_thread_needle.py")
_t20 = importlib.util.module_from_spec(_spec20)
_spec20.loader.exec_module(_t20)
align_full_body_at = _t20.align_full_body_at

_spec22 = importlib.util.spec_from_file_location("c22", "gemini/脚本/22process_eat_cookie.py")
_c22 = importlib.util.module_from_spec(_spec22)
_spec22.loader.exec_module(_c22)
interpolate_flow_multi = _c22.interpolate_flow_multi

CANVAS = 384
GROUND_Y = 366


def main() -> dict:
    s1 = Image.open("gemini/素材/24spirit_butterfly/part1.jpg").convert("RGBA")
    b1 = find_cell_boxes(s1, 4, 3)

    # Preserve delicate luminous violet-pink spirit butterfly glow on black bg with clean alpha
    p1 = [clean_black_bg_rich_particles(s1.crop(b1[i]), bg_thresh=22, hole_thresh=16, glow_cutoff=16) for i in range(12)]

    ref_stand = remove_nebula_blobs_aligned(align_full_body_at(p1[0], GROUND_Y))
    a1 = [ref_stand if i == 0 else remove_nebula_blobs_aligned(align_full_body_at(p1[i], GROUND_Y)) for i in range(12)]

    # Build complete closed-loop 14-keyframe sequence:
    # stand -> brooch glow (a1[1]) -> butterfly orbit & finger perch (a1[2..11]) -> merge back to brooch glow (a1[1]) -> stand
    keys_14 = a1 + [a1[1].copy(), ref_stand.copy()]
    assert len(keys_14) == 14

    # 13 intervals: 11 intervals * 4 intermediates + 2 intervals * 3 intermediates = 14 + 44 + 6 = 64 frames (8x8)
    triple_intervals = {0, 12}
    frames_64: List[Image.Image] = []
    for i in range(14):
        frames_64.append(keys_14[i])
        if i < 13:
            if i in triple_intervals:
                frames_64.extend(interpolate_flow_multi(keys_14[i], keys_14[i + 1], [0.25, 0.5, 0.75]))
            else:
                frames_64.extend(interpolate_flow_multi(keys_14[i], keys_14[i + 1], [0.2, 0.4, 0.6, 0.8]))
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

    sheet_path = os.path.join(out_dir, "24spirit_butterfly_sheet（灵蝶环绕_图集）.webp")
    anim_path = os.path.join(out_dir, "24spirit_butterfly（灵蝶环绕_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 64 frames variable durations (~8.85s): smooth butterfly orbit 130ms, cheek/finger perch holds & ends 185ms
    hold_set = {0, 4, 34, 44, 49, 54, 59, 63}
    durations = [185 if i in hold_set else 130 for i in range(64)]

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
    white_path = os.path.join(prev_dir, "24spirit_butterfly_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "24spirit_butterfly_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "24spirit_butterfly",
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
