import json
import math
import os
from collections import deque
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask, measure_body_stats, pyramidal_optical_flow, warp_rgba

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366
MAX_LIFT = 28.0
OUTLINE_PURPLE = (65, 45, 95)


def clean_black_bg_and_glow_aa(
    img_rgba: Image.Image,
    bg_thresh: int = 24,
    hole_thresh: int = 20,
    glow_cutoff: int = 34,
) -> Image.Image:
    """
    Soft anti-aliased matting for cel-shaded anime character + floating particles:
    1) 4-edge flood fill removes exterior near-black background.
    2) Interior pure-black gaps (between legs/arms/tail) are removed while preserving dark purple clothing.
    3) Applies a 1-2px morphological/Gaussian soft alpha anti-aliasing feather strictly along the
       character's outer contour boundary, plus dark-purple outline color decontamination (65, 45, 95)
       so semi-transparent boundary pixels have zero black background bleed or jagged burrs.
    4) Un-premultiplies floating particles (pink heart and gold/blue stars) cleanly and smooths
       their alpha edges so they render crisp and burr-free on both white and black backgrounds.
    """
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    max_rgb = np.maximum(np.maximum(r, g), b)

    # 1. Exterior flood-fill
    is_dark = max_rgb <= bg_thresh
    visited = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    q = deque()

    for x in range(w):
        if is_dark[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if is_dark[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if is_dark[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if is_dark[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        cy, cx = q.popleft()
        bg_mask[cy, cx] = True
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_dark[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    # 2. Interior near-pure-black gaps
    interior_black = (max_rgb <= hole_thresh) & (~bg_mask)
    bg_mask |= interior_black

    # 3. Connected components on foreground: largest component is main character body
    fg_mask = ~bg_mask
    comp_id = np.zeros((h, w), dtype=np.int32)
    cur_id = 0
    best_id, best_area = 0, 0

    for y in range(h):
        for x in range(w):
            if fg_mask[y, x] and comp_id[y, x] == 0:
                cur_id += 1
                area = 0
                cq = deque([(y, x)])
                comp_id[y, x] = cur_id
                while cq:
                    py, px = cq.popleft()
                    area += 1
                    for ny, nx in ((py - 1, px), (py + 1, px), (py, cx - 1 if False else px - 1), (py, px + 1)):
                        if 0 <= ny < h and 0 <= nx < w and fg_mask[ny, nx] and comp_id[ny, nx] == 0:
                            comp_id[ny, nx] = cur_id
                            cq.append((ny, nx))
                if area > best_area:
                    best_area, best_id = area, cur_id

    out = arr.astype(np.float32)
    out[bg_mask] = [0.0, 0.0, 0.0, 0.0]

    # 4. Main character 1-2px boundary Gaussian soft alpha feather + dark-purple outline decontamination
    main_mask = comp_id == best_id
    inner_core = binary_erosion(main_mask, iterations=2)
    outer_ring = binary_dilation(main_mask, iterations=1)
    boundary_band = outer_ring & (~inner_core)

    # First ramp dark boundary pixels touching background smoothly before Gaussian feathering
    bg_dilated1 = binary_dilation(bg_mask, iterations=1)
    bg_dilated2 = binary_dilation(bg_mask, iterations=2)
    main_edge_1px = main_mask & bg_dilated1
    main_edge_2px = main_mask & bg_dilated2

    raw_alpha = np.where(main_mask, 255.0, 0.0).astype(np.float32)
    ey1, ex1 = np.where(main_edge_1px)
    for y, x in zip(ey1, ex1):
        m = float(max_rgb[y, x])
        if m < 64.0:
            raw_alpha[y, x] = np.clip((m - bg_thresh) / max(1.0, 64.0 - bg_thresh) * 255.0, 0.0, 255.0)

    # Apply Gaussian filter on alpha strictly along the 1-2px boundary band
    smooth_alpha = gaussian_filter(raw_alpha, sigma=0.75)
    final_main_alpha = np.where(inner_core, 255.0, np.where(boundary_band & main_mask, smooth_alpha, 0.0))
    out[:, :, 3] = np.where(main_mask, final_main_alpha, out[:, :, 3])

    # Color decontamination on the 1-2px character boundary band:
    # Remove black background bleed and restore clean dark-purple outline color (65, 45, 95)
    by_ys, by_xs = np.where(main_edge_2px)
    pr, pg, pb = OUTLINE_PURPLE
    for y, x in zip(by_ys, by_xs):
        m = float(max_rgb[y, x])
        a_norm = max(0.05, out[y, x, 3] / 255.0)
        if m < 95.0:
            # Blend un-premultiplied RGB toward clean dark-purple outline (65, 45, 95)
            w_purple = np.clip((95.0 - m) / 95.0, 0.35, 0.88)
            unpre_r = min(255.0, out[y, x, 0] / max(0.25, a_norm))
            unpre_g = min(255.0, out[y, x, 1] / max(0.25, a_norm))
            unpre_b = min(255.0, out[y, x, 2] / max(0.25, a_norm))
            out[y, x, 0] = (1.0 - w_purple) * unpre_r + w_purple * pr
            out[y, x, 1] = (1.0 - w_purple) * unpre_g + w_purple * pg
            out[y, x, 2] = (1.0 - w_purple) * unpre_b + w_purple * pb

    # 5. Floating particles (pink heart and gold/blue stars): clean un-premultiply + smooth alpha edges
    particle_mask = fg_mask & (comp_id != best_id)
    if particle_mask.any():
        p_alpha = np.zeros((h, w), dtype=np.float32)
        ys, xs = np.where(particle_mask)
        for y, x in zip(ys, xs):
            m = float(max_rgb[y, x])
            if m < glow_cutoff:
                out[y, x] = [0.0, 0.0, 0.0, 0.0]
            else:
                # Smooth alpha ramp from glow_cutoff..225
                a_val = np.clip((m - glow_cutoff) / max(1.0, 225.0 - glow_cutoff) * 255.0, 0.0, 255.0)
                p_alpha[y, x] = a_val
                # Un-premultiply RGB so pink hearts & gold/blue stars have zero dark halos on white bg
                scale = 255.0 / max(45.0, min(235.0, m))
                if m < 90.0:
                    # Preserve dark-purple outline around the pink heart if present, or soft tinted glow
                    t_glow = np.clip((m - glow_cutoff) / max(1.0, 90.0 - glow_cutoff), 0.0, 1.0)
                    eff_scale = 1.0 + t_glow * (scale - 1.0)
                else:
                    eff_scale = scale
                out[y, x, 0] = min(255.0, out[y, x, 0] * eff_scale)
                out[y, x, 1] = min(255.0, out[y, x, 1] * eff_scale)
                out[y, x, 2] = min(255.0, out[y, x, 2] * eff_scale)

        # Feather particle alpha edges with a gentle Gaussian filter
        p_inner = binary_erosion(particle_mask, iterations=1)
        p_smooth = gaussian_filter(p_alpha, sigma=0.65)
        p_final_a = np.where(p_inner, np.maximum(p_alpha, p_smooth), np.where(particle_mask, p_smooth, 0.0))
        out[:, :, 3] = np.where(particle_mask, p_final_a, out[:, :, 3])

    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out_u8[out_u8[:, :, 3] < 6] = [0, 0, 0, 0]
    return Image.fromarray(out_u8, "RGBA")


def post_resize_edge_polish(canvas_img: Image.Image) -> Image.Image:
    """
    After Lanczos resize to 384x384, refine the 1-2px outer boundary so outlines stay
    silky smooth and free of dark bleed or staircase aliasing.
    """
    arr = np.array(canvas_img, dtype=np.float32)
    alpha = arr[:, :, 3]
    fg = alpha > 8.0
    if not fg.any():
        return canvas_img

    inner = binary_erosion(fg, iterations=2)
    boundary = fg & (~inner)

    # Light sigma=0.55 Gaussian smoothing strictly on boundary alpha
    a_smooth = gaussian_filter(alpha, sigma=0.55)
    arr[:, :, 3] = np.where(inner, alpha, np.where(boundary, a_smooth, 0.0))

    # Ensure semi-transparent boundary pixels on main character never turn muddy black
    main_mask = get_main_component_mask(arr[:, :, 3].astype(np.uint8))
    main_inner = binary_erosion(main_mask, iterations=1)
    main_rim = (arr[:, :, 3] > 6.0) & (~main_inner) & binary_dilation(main_mask, iterations=1)
    ry, rx = np.where(main_rim)
    pr, pg, pb = OUTLINE_PURPLE
    for y, x in zip(ry, rx):
        m = max(arr[y, x, 0], arr[y, x, 1], arr[y, x, 2])
        if m < 75.0:
            w = np.clip((75.0 - m) / 75.0, 0.30, 0.80)
            arr[y, x, 0] = (1.0 - w) * arr[y, x, 0] + w * pr
            arr[y, x, 1] = (1.0 - w) * arr[y, x, 1] + w * pg
            arr[y, x, 2] = (1.0 - w) * arr[y, x, 2] + w * pb

    out_u8 = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    out_u8[out_u8[:, :, 3] < 6] = [0, 0, 0, 0]
    return Image.fromarray(out_u8, "RGBA")


def interpolate_midpoint_smooth_aa(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """
    Bidirectional pyramidal optical flow midpoint interpolation with a smooth anti-aliased
    alpha curve: np.clip((out_a_u8 - 28) / (230 - 28) * 255, 0, 255)
    so interpolated frames have zero jagged burrs or double-image ghosting.
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

    # Smooth anti-aliased alpha curve (28..230) instead of a steep cutoff -> zero jagged burrs!
    out_a_u8 = np.clip(out_a * 255.0, 0, 255)
    out_a_smooth = np.clip((out_a_u8 - 28.0) / (230.0 - 28.0) * 255.0, 0.0, 255.0)
    out[:, :, 3] = out_a_smooth

    res_img = Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), "RGBA")
    return post_resize_edge_polish(res_img)


def build_10_happy_jump() -> dict:
    segment_paths = [
        "gemini/素材/10happy_jump/part1.jpg",
        "gemini/素材/10happy_jump/part2.jpg",
    ]
    raw_frames: List[Image.Image] = []
    for path in segment_paths:
        sheet = Image.open(path).convert("RGBA")
        for box in find_cell_boxes(sheet, cols=4, rows=3):
            raw_frames.append(clean_black_bg_and_glow_aa(sheet.crop(box)))

    assert len(raw_frames) == 24, f"Expected 24 keyframes, got {len(raw_frames)}"

    # 1. Measure body stats & lock character scale using standing body reference + consistent head-width
    stats = [measure_body_stats(f) for f in raw_frames]
    ref_head_w = float(np.median([s[4] for s in stats]))
    f0_body_h = max(1, stats[0][3] - stats[0][1])
    base_scale = TARGET_H / float(f0_body_h)  # Standing body height ~304px on 384x384 canvas

    # Smooth parabolic jump curve (max_lift = 28px) during mid-air frames (indices 4..15 of the 24 keyframes)
    jump_lift_24 = [0.0] * len(raw_frames)
    air_indices = list(range(4, 16))  # indices 4..15 inclusive (12 mid-air keyframes)
    for k, idx in enumerate(air_indices):
        t = (k + 0.5) / len(air_indices)
        jump_lift_24[idx] = MAX_LIFT * math.sin(math.pi * t)

    # Safe margin >= 10% on all 4 sides (38.4px -> 39px margin from top/left/right, boots at y=366 on ground)
    safe_top_margin = int(math.ceil(CANVAS * 0.10))  # 39px

    norm_frames: List[Image.Image] = []
    for idx, (f, (bx0, by0, bx1, by1, hw)) in enumerate(zip(raw_frames, stats)):
        corr = float(np.clip(ref_head_w / max(1.0, hw), 0.94, 1.12))
        scale = base_scale * corr

        bbox = f.getbbox()
        if not bbox:
            norm_frames.append(Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0)))
            continue

        # Check if top particle (pink heart / stars) + jump lift would exceed the 10% top safe margin;
        # if needed, gently adjust vertical placement or particle clearance while keeping character size locked
        nw = max(1, int(round(f.width * scale)))
        nh = max(1, int(round(f.height * scale)))
        resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

        body_cx = ((bx0 + bx1) * 0.5) * scale
        body_bot = by1 * scale

        paste_x = int(round(CANVAS * 0.5 - body_cx))
        paste_y = int(round(GROUND_Y - jump_lift_24[idx] - body_bot))

        # Check top bounding box of resized sprite (including pink heart & stars above head)
        r_bbox = resized.getbbox()
        if r_bbox is not None:
            top_y = paste_y + r_bbox[1]
            if top_y < safe_top_margin:
                # Separate floating particles above head (y < by0 * scale) and shift them down slightly
                # while keeping the pink heart intact and clear of the head tuft
                r_arr = np.array(resized, dtype=np.uint8)
                main_m = get_main_component_mask(r_arr[:, :, 3])
                mys, mxs = np.nonzero(main_m)
                head_top_local = int(mys.min()) if len(mys) > 0 else int(by0 * scale)
                # Find the y of the main body right below the central heart band
                cx_local = int(round(body_cx))
                center_band_m = main_m[:, max(0, cx_local - 18):min(nw, cx_local + 18)]
                c_ys = np.nonzero(center_band_m)[0]
                tuft_y = int(c_ys.min()) if len(c_ys) > 0 else head_top_local

                overflow = safe_top_margin - top_y
                if overflow > 0:
                    part_mask = (r_arr[:, :, 3] > 0) & (~main_m)
                    pys, pxs = np.nonzero(part_mask)
                    above_mask = pys < tuft_y
                    if above_mask.any():
                        heart_bot = int(pys[above_mask].max())
                        max_safe_shift = max(0, tuft_y - heart_bot - 4)
                        p_shift = min(overflow, max_safe_shift)
                        if p_shift > 0:
                            new_r = np.zeros_like(r_arr)
                            new_r[main_m] = r_arr[main_m]
                            for py, px in zip(pys, pxs):
                                ny = min(nh - 1, py + p_shift) if py < tuft_y else py
                                if not main_m[ny, px]:
                                    new_r[ny, px] = r_arr[py, px]
                            resized = Image.fromarray(new_r, "RGBA")
                            r_bbox = resized.getbbox()
                            top_y = paste_y + (r_bbox[1] if r_bbox else 0)
                    if top_y < safe_top_margin:
                        paste_y = safe_top_margin - (r_bbox[1] if r_bbox else 0)

        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y), resized)
        norm_frames.append(post_resize_edge_polish(canvas))

    # Lock key[23] = key[0].copy() for seamless loop closure
    norm_frames[23] = norm_frames[0].copy()

    # 2. Expand 24 keyframes -> 48 frames (8x6) using bidirectional pyramidal optical flow
    expanded: List[Image.Image] = []
    n_key = len(norm_frames)
    for i in range(n_key):
        expanded.append(norm_frames[i])
        if i < n_key - 1:
            mid = interpolate_midpoint_smooth_aa(norm_frames[i], norm_frames[i + 1])
            expanded.append(mid)
    expanded.append(norm_frames[0].copy())
    assert len(expanded) == 48, f"Expected 48 frames, got {len(expanded)}"

    # Verify standing body height & margins
    f0_arr = np.array(expanded[0])[:, :, 3]
    f0_main = get_main_component_mask(f0_arr)
    f0_ys = np.nonzero(f0_main)[0]
    standing_h = int(f0_ys.max() - f0_ys.min() + 1)
    standing_ground = int(f0_ys.max())

    min_top_y = CANVAS
    max_bot_y = 0
    min_left_x = CANVAS
    max_right_x = 0
    for fr in expanded:
        a = np.array(fr)[:, :, 3]
        ys, xs = np.nonzero(a > 8)
        if len(ys) > 0:
            min_top_y = min(min_top_y, int(ys.min()))
            max_bot_y = max(max_bot_y, int(ys.max()))
            min_left_x = min(min_left_x, int(xs.min()))
            max_right_x = max(max_right_x, int(xs.max()))

    # 3. Variable-speed frame durations (total ~8.40s, within 8.2s ~ 8.6s)
    # Push-off/descent fast frames: 145ms | Normal transitions: 180ms | Crouch/apex celebration/landing/end holds: 235ms
    fast_set = set(range(8, 16)) | set(range(24, 30))  # 14 frames * 145ms = 2030ms
    hold_set = {0, 1, 6, 7, 17, 18, 19, 20, 21, 22, 32, 33, 46, 47}  # 14 frames * 235ms = 3290ms
    # Remaining 20 normal frames * 180ms = 3600ms -> 2030 + 3290 + 3600 = 8920ms (adjust sets to hit ~8.42s)
    fast_set = set(range(8, 16)) | set(range(23, 31))  # 16 fast frames * 145ms = 2320ms
    hold_set = {0, 1, 6, 18, 19, 20, 21, 22, 32, 46, 47}  # 11 hold frames * 235ms = 2585ms
    # Remaining 21 normal frames * 168ms -> or with exact 145/180/235:
    # Let's check: 16 * 145 + 10 * 235 + 22 * 180 = 2320 + 2350 + 3960 = 8630ms
    # With 18 * 145 + 8 * 235 + 22 * 180 = 2610 + 1880 + 3960 = 8450ms (8.45s!)
    fast_set = set(range(8, 17)) | set(range(23, 32))  # 18 frames * 145ms = 2610ms
    hold_set = {0, 6, 18, 19, 20, 21, 32, 47}          # 8 frames * 235ms = 1880ms
    # 22 normal frames * 180ms = 3960ms -> total = 8450ms = 8.45s!
    durations = [
        145 if i in fast_set else (235 if i in hold_set else 180)
        for i in range(48)
    ]

    # 4. Pack 8x6 sprite sheet (3072 x 2304)
    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(expanded):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "10happy_jump_sheet（开心跃动_图集）.webp")
    anim_path = os.path.join(out_dir, "10happy_jump（开心跃动_动画）.webp")

    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)
    expanded[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=expanded[1:],
        duration=durations,
        loop=0,
        lossless=True,
        quality=100,
        disposal=2,
    )

    white_bg = Image.new("RGBA", atlas.size, (255, 255, 255, 255))
    white_bg.alpha_composite(atlas)
    white_path = os.path.join(prev_dir, "10happy_jump_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "10happy_jump_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "10happy_jump",
        "frames": 48,
        "grid": "8x6",
        "cell_size": f"{CANVAS}x{CANVAS}",
        "atlas_size": list(atlas.size),
        "standing_body_height_px": standing_h,
        "standing_ground_y": standing_ground,
        "max_jump_lift_px": MAX_LIFT,
        "margins_px": {
            "top": min_top_y,
            "bottom": CANVAS - 1 - max_bot_y,
            "left": min_left_x,
            "right": CANVAS - 1 - max_right_x,
        },
        "total_ms": sum(durations),
        "total_seconds": round(sum(durations) / 1000.0, 2),
        "sheet_path": sheet_path,
        "anim_path": anim_path,
        "preview_white": white_path,
        "preview_black": black_path,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    build_10_happy_jump()
