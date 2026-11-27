import json
import math
import os
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from build_webp import clean_black_bg_and_glow, find_cell_boxes
from expand_48 import (
    get_main_component_mask,
    interpolate_midpoint_no_ghost,
    measure_body_stats,
    warp_rgba,
)

CANVAS = 384
HEAD_TOP_Y = 58
PIVOT_X = 192.0
PIVOT_Y = 30.0


def extract_pure_black_rows(path: str, keep_rows: int = 3) -> List[Image.Image]:
    """Slice a 4x3 sheet and clean pure-black background for the first `keep_rows` rows."""
    sheet = Image.open(path).convert("RGBA")
    arr = np.array(sheet.convert("RGB"), dtype=np.uint8)
    # Measure background level from the top row border only (safe even if Row 3 has non-black bg)
    top_strip = arr[:12, :, :].max(axis=2)
    bg = int(np.percentile(top_strip, 95))
    t = max(22, bg + 14)

    # Crop only the pure-black top rows if keep_rows < 3 so non-black rows never affect column/row splits
    if keep_rows < 3:
        h, w = arr.shape[:2]
        # Find exact bottom boundary of row `keep_rows` by checking where non-black bg or gap sits
        cut_h = int(round(h * keep_rows / 3.0))
        # Refine cut_h slightly upward if a bright gradient starts just above cut_h
        row_bg_edge = arr[:, :15, :].max(axis=2).mean(axis=1)
        bright_rows = np.where(row_bg_edge > bg + 20)[0]
        if len(bright_rows) > 0 and bright_rows[0] > int(h * 0.55):
            cut_h = int(bright_rows[0]) - 2
        sub = sheet.crop((0, 0, w, cut_h))
        boxes = find_cell_boxes(sub, cols=4, rows=keep_rows, thresh=t)
        src = sub
    else:
        boxes = find_cell_boxes(sheet, cols=4, rows=3, thresh=t)
        src = sheet

    cells = [
        clean_black_bg_and_glow(
            src.crop(b),
            bg_thresh=max(24, bg + 12),
            hole_thresh=max(20, bg + 8),
            glow_cutoff=max(34, bg + 16),
        )
        for b in boxes
    ]
    return cells


def align_hanging_poses(
    poses: List[Image.Image],
    non_kick_indices: List[int],
    target_h: int = 262,
) -> tuple[List[Image.Image], int, int]:
    """
    Uniformly align vertical hanging poses to CANVAS x CANVAS with strict size consistency.
    Uses median non-kick body height + gently clamped head-width scaling so kicks never shrink/grow the character.
    """
    stats = [measure_body_stats(f) for f in poses]
    nk_heights = [stats[i][3] - stats[i][1] for i in non_kick_indices]
    nk_hws = [stats[i][4] for i in non_kick_indices]
    s_ref = target_h / max(1.0, float(np.median(nk_heights)))
    hw_ref = float(np.median(nk_hws))

    aligned: List[Image.Image] = []
    out_nk_heights: List[int] = []

    for idx, (f, (bx0, by0, bx1, by1, hw)) in enumerate(zip(poses, stats)):
        if idx in non_kick_indices:
            # Lock non-kick hanging height strictly to target_h (+/- 1px)
            s_i = target_h / max(1.0, float(by1 - by0))
        else:
            s_i = float(np.clip(s_ref * hw_ref / max(1.0, hw), 0.98 * s_ref, 1.02 * s_ref))
        nw = max(1, int(round(f.width * s_i)))
        nh = max(1, int(round(f.height * s_i)))
        resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

        # Head horizontal center from top 25% of main connected component
        r_arr = np.array(resized, dtype=np.uint8)
        main_mask = get_main_component_mask(r_arr[:, :, 3])
        ys, xs = np.nonzero(main_mask)
        my0, my1 = int(ys.min()), int(ys.max())
        top_cut = my0 + max(4, int((my1 - my0 + 1) * 0.25))
        head_xs = xs[ys <= top_cut]
        head_cx = float(np.median(head_xs)) if len(head_xs) > 0 else float(np.median(xs))

        paste_x = int(round(CANVAS * 0.5 - head_cx))
        paste_y = int(round(HEAD_TOP_Y - my0))

        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y), resized)
        aligned.append(canvas)

        if idx in non_kick_indices:
            out_nk_heights.append(my1 - my0 + 1)

    return aligned, min(out_nk_heights), max(out_nk_heights)


def apply_secondary_lag_shear(img: Image.Image, lag_px: float) -> Image.Image:
    """
    Apply smooth phase-lagged secondary sway to lower hair/skirt/tail/legs below eye level,
    keeping head & dot-eyes 100% rigid and undistorted while creating realistic pendulum inertia.
    """
    if abs(lag_px) < 0.05:
        return img.copy()
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth S-curve ramp: 0 above y=125 (head & face rigid), ramping smoothly down to 1.0 at y=310
    t = np.clip((yy - 125.0) / (310.0 - 125.0), 0.0, 1.0)
    ramp = t * t * (3.0 - 2.0 * t)
    dx = -lag_px * ramp  # inverse mapping offset so content shifts by +lag_px
    dy = np.zeros_like(dx)

    warped = warp_rgba(arr, dx, dy)
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")


def draw_lifting_lines(img: Image.Image) -> Image.Image:
    """Draw 3 crisp bright-white lifting motion lines with subtle lavender outline above head."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    lines = [
        ((192, 16), (192, 42)),
        ((172, 22), (180, 44)),
        ((212, 22), (204, 44)),
    ]
    # Subtle lavender outline so lines stay crisp on pure white AND pure black backgrounds
    for p0, p1 in lines:
        draw.line([p0, p1], fill=(195, 175, 255, 255), width=7)
    for p0, p1 in lines:
        draw.line([p0, p1], fill=(255, 255, 255, 255), width=4)
    return out


def rotate_around_pivot(img: Image.Image, theta_deg: float) -> Image.Image:
    """
    Rotate RGBA image around collar pivot (PIVOT_X, PIVOT_Y) using premultiplied warp_rgba.
    Positive theta_deg swings her feet toward VIEWER'S LEFT (clockwise in screen coords).
    """
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Screen coords (x right, y down): clockwise rotation by +theta moves bottom (y > cy) to LEFT (x < cx).
    # Inverse mapping (dest -> src) rotates counter-clockwise by -theta:
    # sx = cx + cos(theta)*(x-cx) + sin(theta)*(y-cy)
    # sy = cy - sin(theta)*(x-cx) + cos(theta)*(y-cy)
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    rx = xx - PIVOT_X
    ry = yy - PIVOT_Y
    sx = PIVOT_X + c * rx + s * ry
    sy = PIVOT_Y - s * rx + c * ry
    dx = sx - xx
    dy = sy - yy

    warped = warp_rgba(arr, dx, dy)
    # Crisp edge cleanup to prevent faint semi-transparent dust
    a = warped[:, :, 3]
    warped[:, :, 3] = np.where(a < 18, 0.0, a)
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")


def build_cycle_frames(
    aligned_p1: List[Image.Image],
    aligned_p2: List[Image.Image],
) -> tuple[List[Image.Image], List[Image.Image]]:
    """
    Build 24 vertical states for Cycle 1 (all 12 big-eyes Set A keys + 12 optical-flow midpoints)
    and 24 vertical states for Cycle 2 (all 12 big-eyes Set B keys + 12 optical-flow midpoints).
    """
    def expand_12_to_24(keys: List[Image.Image], wrap_target: Image.Image) -> List[Image.Image]:
        out: List[Image.Image] = []
        for k in range(12):
            a = keys[k]
            b = keys[k + 1] if k < 11 else wrap_target
            out.append(a)
            # Avoid optical-flow smear into/out of the 1-beat comic dot-eye cell (index 9 in Set B)
            out.append(interpolate_midpoint_no_ghost(a, b))
        return out

    return expand_12_to_24(aligned_p1, aligned_p2[0]), expand_12_to_24(aligned_p2, aligned_p1[0])


def has_border_clip(frames: List[Image.Image], margin: int = 3) -> bool:
    for fr in frames:
        a = np.array(fr)[:, :, 3]
        if (
            a[:margin, :].any()
            or a[-margin:, :].any()
            or a[:, :margin].any()
            or a[:, -margin:].any()
        ):
            return True
    return False


def main() -> dict:
    # Big-eyes Set A & Set B: ALL 3 rows (12 cells each) on pure solid #000000 black background!
    p1_cells = extract_pure_black_rows("gemini/素材/14drag_float/part1_bigeyes.jpg", keep_rows=3)
    p2_cells = extract_pure_black_rows("gemini/素材/14drag_float/part2_bigeyes.jpg", keep_rows=3)

    all_poses = p1_cells + p2_cells  # 24 total poses
    # Kick cells are p2 indices 1, 2, 6, 7 -> global indices 13, 14, 18, 19
    kick_indices = {13, 14, 18, 19}
    non_kick_indices = [i for i in range(24) if i not in kick_indices]

    frames_48: List[Image.Image] = []
    chosen_target_h = 262
    nk_h_min, nk_h_max = 0, 0

    for target_h in (262, 252, 244):
        aligned_all, nk_h_min, nk_h_max = align_hanging_poses(all_poses, non_kick_indices, target_h=target_h)
        aligned_p1 = aligned_all[:12]
        aligned_p2 = aligned_all[12:]

        states_a, states_b = build_cycle_frames(aligned_p1, aligned_p2)

        candidate: List[Image.Image] = []
        for i in range(48):
            cyc = i % 24
            phase = 2.0 * math.pi * i / 24.0
            theta = 15.0 * math.sin(phase)          # +15° at i=6 (feet left), -15° at i=18 (feet right)
            # Phase-lagged secondary hair/tail sway (~90° lag behind angular velocity -> trails opposite motion)
            lag_px = 13.0 * math.cos(phase)

            base_state = states_a[cyc] if i < 24 else states_b[cyc]
            swayed = apply_secondary_lag_shear(base_state, lag_px)
            with_lines = draw_lifting_lines(swayed)
            rotated = rotate_around_pivot(with_lines, theta)
            candidate.append(rotated)

        if not has_border_clip(candidate, margin=3):
            frames_48 = candidate
            chosen_target_h = target_h
            break
        frames_48 = candidate
        chosen_target_h = target_h

    # Verify swing direction at frame 6 (theta = +15°): bottom pixels should be LEFT of center (x < 192)
    f6_a = np.array(frames_48[6])[:, :, 3]
    ys6, xs6 = np.nonzero(f6_a > 128)
    bot_xs6 = xs6[ys6 > int(np.percentile(ys6, 85))]
    f6_foot_cx = float(np.mean(bot_xs6))

    # Variable-speed pendulum durations (slower at apexes, faster through bottom center)
    durations = [int(round(125 + 45 * abs(math.sin(2.0 * math.pi * i / 24.0)))) for i in range(48)]

    # Export 8x6 lossless transparent sprite sheet (3072 x 2304)
    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "14drag_float_sheet（拎起晃动_图集）.webp")
    anim_path = os.path.join(out_dir, "14drag_float（拎起晃动_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

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
    white_path = os.path.join(prev_dir, "14drag_float_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "14drag_float_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "14drag_float",
        "frames": 48,
        "grid": "8x6",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "target_h": chosen_target_h,
        "non_kick_height_range": [nk_h_min, nk_h_max],
        "frame6_foot_cx": round(f6_foot_cx, 1),
        "border_clip": has_border_clip(frames_48, margin=3),
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
