import importlib.util
import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask, pyramidal_optical_flow, warp_rgba

_spec = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned

_spec20 = importlib.util.spec_from_file_location("t20", "gemini/脚本/20process_thread_needle.py")
_t20 = importlib.util.module_from_spec(_spec20)
_spec20.loader.exec_module(_t20)
align_full_body_at = _t20.align_full_body_at

CANVAS = 384
GROUND_Y = 366


def interpolate_flow_multi(img_a: Image.Image, img_b: Image.Image, ts: List[float]) -> List[Image.Image]:
    """Compute bidirectional optical flow once and synthesize smooth anti-aliased intermediates at fractions `ts`."""
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa, fb = feature_map(a), feature_map(b)
    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=4, win_sigma=3.5, iters=6)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=4, win_sigma=3.5, iters=6)

    out_frames: List[Image.Image] = []
    for t in ts:
        wa = warp_rgba(a, -t * vx_ab, -t * vy_ab)
        wb = warp_rgba(b, -(1.0 - t) * vx_ba, -(1.0 - t) * vy_ba)

        rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
        alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
        disagree = rgb_diff * 0.7 + alpha_diff * 0.9
        agree_w = np.clip(1.0 - disagree * 2.6, 0.0, 1.0)

        prefer_a = ((wa[:, :, 3] >= wb[:, :, 3]) if t <= 0.5 else (wa[:, :, 3] > wb[:, :, 3])).astype(np.float32)
        w_a = (1.0 - t) * agree_w + prefer_a * (1.0 - agree_w)
        w_b = 1.0 - w_a

        out = np.zeros_like(wa, dtype=np.float32)
        out[:, :, :3] = wa[:, :, :3] * w_a[:, :, None] + wb[:, :, :3] * w_b[:, :, None]
        out_a = wa[:, :, 3] * w_a + wb[:, :, 3] * w_b

        # Smooth anti-aliased alpha curve [28..228] (zero jagged burrs)
        out_a_u8 = np.clip(out_a, 0, 255)
        out[:, :, 3] = np.where(
            out_a_u8 < 28.0,
            0.0,
            np.where(out_a_u8 > 228.0, 255.0, (out_a_u8 - 28.0) / 200.0 * 255.0),
        )
        out_frames.append(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA"))
    return out_frames


def main() -> dict:
    s1 = Image.open("gemini/素材/22eat_cookie/part1.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/22eat_cookie/part2.jpg").convert("RGBA")

    b1 = find_cell_boxes(s1, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    p1 = [clean_black_bg_rich_particles(s1.crop(b1[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    ref_stand = remove_nebula_blobs_aligned(align_full_body_at(p1[0], GROUND_Y))

    a1 = [ref_stand if i == 0 else remove_nebula_blobs_aligned(align_full_body_at(p1[i], GROUND_Y)) for i in range(12)]
    a2 = [remove_nebula_blobs_aligned(align_full_body_at(p2[i], GROUND_Y)) for i in range(9)]

    # 22 keyframes -> 21 intervals * 2 intermediates each = 64 frames (8x8)
    keys_22 = a1 + a2 + [ref_stand.copy()]
    assert len(keys_22) == 22

    frames_64: List[Image.Image] = []
    for i in range(22):
        frames_64.append(keys_22[i])
        if i < 21:
            frames_64.extend(interpolate_flow_multi(keys_22[i], keys_22[i + 1], [1.0 / 3.0, 2.0 / 3.0]))
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

    sheet_path = os.path.join(out_dir, "22eat_cookie_sheet（吃曲奇_图集）.webp")
    anim_path = os.path.join(out_dir, "22eat_cookie（吃曲奇_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 64 frames variable durations (~8.8s): chewing fast 110ms, holds/licking/blink/ends 185ms, normal 135ms
    fast_set = set(range(12, 34))
    hold_set = {0, 6, 7, 36, 39, 40, 42, 43, 57, 58, 63}
    durations = [110 if i in fast_set else (185 if i in hold_set else 135) for i in range(64)]

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
    white_path = os.path.join(prev_dir, "22eat_cookie_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "22eat_cookie_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "22eat_cookie",
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
