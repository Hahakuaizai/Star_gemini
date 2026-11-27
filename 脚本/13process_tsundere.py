import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import clean_black_bg_and_glow, find_cell_boxes
from expand_48 import measure_body_stats, pyramidal_optical_flow, warp_rgba

CANVAS = 384
TARGET_H = 310  # Strictly locked standing height (+/- 2px across all frames)
GROUND_Y = 366


def align_strict_height(raw_frames: List[Image.Image]) -> List[Image.Image]:
    """
    Strictly lock every standing frame's main-body vertical height to TARGET_H (310px)
    and anchor boots to GROUND_Y (366px), centered horizontally at X=50%.
    """
    aligned: List[Image.Image] = []
    for f in raw_frames:
        bx0, by0, bx1, by1, _ = measure_body_stats(f)
        body_h = max(1, by1 - by0)
        scale = TARGET_H / body_h

        nw = max(1, int(round(f.width * scale)))
        nh = max(1, int(round(f.height * scale)))
        resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

        body_cx = ((bx0 + bx1) * 0.5) * scale
        body_bot = by1 * scale
        paste_x = int(round(CANVAS * 0.5 - body_cx))
        paste_y = int(round(GROUND_Y - body_bot))

        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y), resized)
        aligned.append(canvas)
    return aligned


def interpolate_midpoint(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """Single crisp t=0.5 bidirectional optical-flow midpoint with sharp alpha."""
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa, fb = feature_map(a), feature_map(b)
    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=4, win_sigma=3.5, iters=6)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=4, win_sigma=3.5, iters=6)

    wa = warp_rgba(a, -0.5 * vx_ab, -0.5 * vy_ab)
    wb = warp_rgba(b, -0.5 * vx_ba, -0.5 * vy_ba)

    rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
    alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
    disagree = rgb_diff * 0.7 + alpha_diff * 0.9

    agree_weight = np.clip(1.0 - disagree * 2.8, 0.0, 1.0)
    pick_a = (wa[:, :, 3] >= wb[:, :, 3]).astype(np.float32)
    w_a = 0.5 * agree_weight + pick_a * (1.0 - agree_weight)
    w_b = 1.0 - w_a

    aa = wa[:, :, 3] / 255.0
    ab = wb[:, :, 3] / 255.0
    out_a = w_a * aa + w_b * ab

    out = np.zeros_like(wa)
    for c in range(3):
        num = w_a * wa[:, :, c] * aa + w_b * wb[:, :, c] * ab
        out[:, :, c] = np.where(out_a > 1e-3, np.clip(num / np.maximum(out_a, 1e-3), 0, 255), 0.0)

    out_a_u8 = np.clip(out_a * 255.0, 0, 255)
    out[:, :, 3] = np.where(
        out_a_u8 < 60, 0.0, np.where(out_a_u8 > 200, 255.0, (out_a_u8 - 60) / 140.0 * 255.0)
    )
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def main() -> dict:
    p1_sheet = Image.open("gemini/素材/13tsundere/part1.jpg").convert("RGBA")
    pt_sheet = Image.open("gemini/素材/13tsundere/part_turns.jpg").convert("RGBA")
    p2_sheet = Image.open("gemini/素材/13tsundere/part2.jpg").convert("RGBA")

    p1 = [clean_black_bg_and_glow(p1_sheet.crop(b)) for b in find_cell_boxes(p1_sheet, 4, 3)]
    pt = [
        clean_black_bg_and_glow(pt_sheet.crop(b), bg_thresh=34, hole_thresh=28, glow_cutoff=38)
        for b in find_cell_boxes(pt_sheet, 4, 3, thresh=34)
    ]
    p2 = [clean_black_bg_and_glow(p2_sheet.crop(b)) for b in find_cell_boxes(p2_sheet, 4, 3)]

    # 29 genuinely unique hand-drawn poses with full multi-angle turn-in, return-turn & arm-uncrossing!
    raw_29 = [
        p1[0],   # 0: Clean front standard standing
        p1[1],   # 1: Brows furrow, hands lifting
        p1[2],   # 2: Puffed cheeks, arms folding
        p1[3],   # 3: Arms firmly crossed + 💢
        p1[4],   # 4: Slight turn start (~10°) + 💢
        pt[0],   # 5: Turning ~20° right + 💢 (tail right)
        pt[1],   # 6: 35° three-quarter turn (tail tucked behind legs!) + 💢
        pt[2],   # 7: 55° turn (tail emerging left!) + 💢
        pt[3],   # 8: 70° side turn (tail left) + 💢
        p1[6],   # 9: Full 75° side profile > < + 💢 + huff cloud
        p1[7],   # 10: Side closed eyes + huff expanding + foot lift
        p1[8],   # 11: Side foot tap 1 + huff cloud
        p1[9],   # 12: Side haughty glare + tail swish
        p1[10],  # 13: Side puffed cheeks + 💢
        p1[11],  # 14: Side 💢 gone, peeking sideways
        p2[0],   # 15: Angled side peek
        p2[1],   # 16: Blushing side peek
        p2[2],   # 17: Return turn start
        p2[4],   # 18: Return turn ~50°
        pt[5],   # 19: Return turn ~38°
        p2[5],   # 20: Return turn ~22°, softening
        pt[4],   # 21: Front-facing ~5°, arms crossed, tail cleanly right
        pt[7],   # 22: Front-facing, arms loosely crossed, gentle smile, star clip right
        pt[8],   # 23: Hands separating from elbows (uncrossing 1)
        pt[9],   # 24: Arms opening outward (uncrossing 2)
        pt[10],  # 25: Arms lowering calmly (uncrossing 3)
        pt[11],  # 26: Warm smile, arms at sides
        p2[10],  # 27: Settling peacefully
        p1[0].copy(),  # 28: Exact Frame 0 clean standard standing
    ]

    aligned_29 = align_strict_height(raw_29)
    aligned_29[-1] = aligned_29[0].copy()

    # Insert ONLY 7 single optical-flow midpoints on the 7 widest same-side transitions -> 36 crisp unique frames!
    interp_after = {0, 1, 9, 10, 15, 23, 24}
    frames_36: List[Image.Image] = []
    for i in range(len(aligned_29)):
        frames_36.append(aligned_29[i])
        if i in interp_after:
            frames_36.append(interpolate_midpoint(aligned_29[i], aligned_29[i + 1]))

    assert len(frames_36) == 36, len(frames_36)

    heights = []
    for fr in frames_36:
        a = np.array(fr)[:, :, 3]
        ys = np.nonzero(a.any(axis=1))[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 6, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_36):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "13tsundere_sheet（傲娇抱胸_图集）.webp")
    anim_path = os.path.join(out_dir, "13tsundere（傲娇抱胸_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Variable-speed durations (~7.30s total, crisp turns 145ms, normal 195ms, expressive holds 270ms)
    fast_set = {1, 2, 3, 4, 6, 7, 8, 9, 10, 22, 23, 24, 25, 29, 30, 31}
    hold_set = {0, 5, 11, 13, 15, 19, 20, 21, 27, 35}
    durations = [145 if i in fast_set else (270 if i in hold_set else 195) for i in range(36)]

    frames_36[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_36[1:],
        duration=durations,
        loop=0,
        lossless=True,
        quality=100,
        disposal=2,
    )

    white_bg = Image.new("RGBA", atlas.size, (255, 255, 255, 255))
    white_bg.alpha_composite(atlas)
    white_path = os.path.join(prev_dir, "13tsundere_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "13tsundere_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "13tsundere",
        "frames": 36,
        "hand_drawn_unique": 29,
        "optical_flow_midpoints": 7,
        "grid": "6x6",
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
