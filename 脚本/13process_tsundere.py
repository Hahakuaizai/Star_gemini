import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import clean_black_bg_and_glow, find_cell_boxes
from expand_48 import measure_body_stats, pyramidal_optical_flow, warp_rgba

CANVAS = 384
TARGET_H = 310  # Strictly locked standing height (+/- 1px across all frames)
GROUND_Y = 366


def align_strict_height(raw_frames: List[Image.Image]) -> List[Image.Image]:
    """
    Strictly lock every standing frame's main-body vertical height to TARGET_H (310px)
    and anchor boots to GROUND_Y (366px), centered horizontally at X=50%.
    Eliminates any front-vs-side profile size jitter.
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


def interpolate_between(img_a: Image.Image, img_b: Image.Image, ts: List[float]) -> List[Image.Image]:
    """
    Compute bidirectional pyramidal optical flow ONCE for pair (A, B) and synthesize
    crisp, ghost-free intermediate frames at arbitrary fractions t in ts.
    """
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa = feature_map(a)
    fb = feature_map(b)
    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=4, win_sigma=3.5, iters=6)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=4, win_sigma=3.5, iters=6)

    out_frames: List[Image.Image] = []
    for t in ts:
        wa = warp_rgba(a, -t * vx_ab, -t * vy_ab)
        wb = warp_rgba(b, -(1.0 - t) * vx_ba, -(1.0 - t) * vy_ba)

        rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
        alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
        disagree = rgb_diff * 0.7 + alpha_diff * 0.9

        agree_weight = np.clip(1.0 - disagree * 2.8, 0.0, 1.0)
        # When warps disagree, snap cleanly toward whichever keyframe is temporally closer
        prefer_a = 1.0 if t <= 0.5 else 0.0
        w_a = (1.0 - t) * agree_weight + prefer_a * (1.0 - agree_weight)
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
        out_frames.append(Image.fromarray(out.astype(np.uint8), "RGBA"))
    return out_frames


def main() -> dict:
    p1_sheet = Image.open("gemini/素材/13tsundere/part1.jpg").convert("RGBA")
    p2_sheet = Image.open("gemini/素材/13tsundere/part2.jpg").convert("RGBA")

    p1 = [clean_black_bg_and_glow(p1_sheet.crop(b)) for b in find_cell_boxes(p1_sheet, 4, 3)]
    p2 = [clean_black_bg_and_glow(p2_sheet.crop(b)) for b in find_cell_boxes(p2_sheet, 4, 3)]

    # 19 chronologically ordered keyframes
    raw_19 = [
        p1[0], p1[1], p1[2], p1[3], p1[4], p1[5],
        p1[6], p1[7], p1[8], p1[9], p1[10], p1[11],
        p2[1], p2[4], p2[5], p2[6], p2[7], p2[10], p1[0].copy(),
    ]
    aligned_19 = align_strict_height(raw_19)
    aligned_19[-1] = aligned_19[0].copy()

    # Expand 19 keyframes -> EXACTLY 64 frames (8x8 ultra-smooth!)
    # 18 transitions:
    # - Skip optical-flow warp on the 2 tail-side-flip transitions (i=4 and i=15): 0 in-betweens (crisp snap)
    # - Insert 3 in-betweens (t=0.25, 0.5, 0.75) on 13 expressive transitions
    # - Insert 2 in-betweens (t=0.33, 0.67) on 3 shorter transitions (i=0, 16, 17)
    # Total: 19 keyframes + 13*3 + 3*2 = 19 + 39 + 6 = 64 frames!
    short_pairs = {0, 16, 17}
    snap_pairs = {4, 15}

    frames_64: List[Image.Image] = []
    for i in range(18):
        frames_64.append(aligned_19[i])
        if i in snap_pairs:
            continue
        elif i in short_pairs:
            frames_64.extend(interpolate_between(aligned_19[i], aligned_19[i + 1], [1 / 3, 2 / 3]))
        else:
            frames_64.extend(interpolate_between(aligned_19[i], aligned_19[i + 1], [0.25, 0.5, 0.75]))
    frames_64.append(aligned_19[0].copy())
    assert len(frames_64) == 64, len(frames_64)

    # Verify height consistency across all 64 frames
    heights = []
    for fr in frames_64:
        a = np.array(fr)[:, :, 3]
        ys = np.nonzero(a.any(axis=1))[0]
        heights.append(int(ys.max() - ys.min() + 1))

    # Export 8x8 lossless transparent sheet (3072 x 3072)
    cols, rows = 8, 8
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_64):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "13tsundere_sheet（傲娇抱胸_图集）.webp")
    anim_path = os.path.join(out_dir, "13tsundere（傲娇抱胸_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Variable-speed durations for 64 frames (~7.4s total, ~9 FPS buttery smooth!)
    # Fast crisp turns/arm folds: 85ms; Normal transitions: 115ms; Key expressive holds: 165ms
    fast_set = set(range(4, 18)) | set(range(48, 56))
    hold_set = {0, 1, 21, 22, 25, 26, 41, 42, 45, 46, 62, 63}
    durations = [85 if i in fast_set else (165 if i in hold_set else 115) for i in range(64)]

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
    white_path = os.path.join(prev_dir, "13tsundere_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "13tsundere_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "13tsundere",
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
