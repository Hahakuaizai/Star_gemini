import json
import math
import os
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask, measure_body_stats, pyramidal_optical_flow, warp_rgba

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366


def clean_black_bg_rich_particles(
    img_rgba: Image.Image,
    bg_thresh: int = 22,
    hole_thresh: int = 16,
    glow_cutoff: int = 20,
) -> Image.Image:
    """
    Anti-aliased black-background removal that preserves and boosts rich glowing starlight particles
    while ensuring smooth, burr-free dark-purple character outlines on both white and black backgrounds.
    """
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    chroma = max_rgb - min_rgb

    # 1. 4-edge exterior flood-fill on near-black background
    is_dark = (max_rgb <= bg_thresh) & (chroma <= 14)
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

    # 2. Interior pure-black gaps (between legs/arms/broom handle)
    interior_black = (max_rgb <= hole_thresh) & (chroma <= 10) & (~bg_mask)
    bg_mask |= interior_black

    # 3. Identify main character component vs floating particles/glow
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
                    for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                        if 0 <= ny < h and 0 <= nx < w and fg_mask[ny, nx] and comp_id[ny, nx] == 0:
                            comp_id[ny, nx] = cur_id
                            cq.append((ny, nx))
                if area > best_area:
                    best_area, best_id = area, cur_id

    out = arr.astype(np.float32)
    out[bg_mask] = [0.0, 0.0, 0.0, 0.0]

    # Smooth anti-aliased edge decontamination on main component boundary (eliminates jagged burrs & black fringe)
    main_mask = comp_id == best_id
    bg_dil1 = np.zeros((h, w), dtype=bool)
    bg_dil1[1:, :] |= bg_mask[:-1, :]
    bg_dil1[:-1, :] |= bg_mask[1:, :]
    bg_dil1[:, 1:] |= bg_mask[:, :-1]
    bg_dil1[:, :-1] |= bg_mask[:, 1:]
    main_edge = main_mask & bg_dil1

    ey, ex = np.where(main_edge)
    for y, x in zip(ey, ex):
        m = int(max_rgb[y, x])
        c = int(chroma[y, x])
        if m < 75:
            # Softly ramp alpha and tint dark boundary toward clean cel-shaded purple or golden glow
            alpha = float(np.clip((m - bg_thresh) / max(1.0, 75.0 - bg_thresh) * 245.0, 0.0, 245.0))
            if c > 18 and r[y, x] > b[y, x]:
                # Golden broom / starlight edge attached to main component -> un-premultiply warm glow
                scale = 255.0 / max(1.0, float(m))
                out[y, x, 0] = min(255.0, out[y, x, 0] * scale)
                out[y, x, 1] = min(255.0, out[y, x, 1] * scale)
                out[y, x, 2] = min(255.0, out[y, x, 2] * scale)
            else:
                # Dark-purple character outline edge -> decontaminate black bleed
                out[y, x, 0] = max(out[y, x, 0], 68.0)
                out[y, x, 1] = max(out[y, x, 1], 48.0)
                out[y, x, 2] = max(out[y, x, 2], 98.0)
            out[y, x, 3] = alpha

    # Un-premultiply and boost all floating starlight particles & dust trails
    particle_mask = fg_mask & (comp_id != best_id)
    py_arr, px_arr = np.where(particle_mask)
    for y, x in zip(py_arr, px_arr):
        m = int(max_rgb[y, x])
        if m < glow_cutoff:
            out[y, x] = [0.0, 0.0, 0.0, 0.0]
        elif m < 245:
            # Boost particle opacity & color saturation so golden/pink/cyan starlight pops on white & black bg
            alpha = min(255.0, float((m - glow_cutoff) / max(1.0, 215.0 - glow_cutoff) * 255.0))
            scale = 255.0 / max(1.0, float(m))
            out[y, x, 0] = min(255.0, out[y, x, 0] * scale)
            out[y, x, 1] = min(255.0, out[y, x, 1] * scale)
            out[y, x, 2] = min(255.0, out[y, x, 2] * scale)
            out[y, x, 3] = alpha

    # Gentle 0.55px Gaussian alpha feather strictly on the 1px outer boundary for burr-free outlines
    alpha_ch = out[:, :, 3]
    smooth_alpha = gaussian_filter(alpha_ch, sigma=0.55)
    out[:, :, 3] = np.where(main_edge, smooth_alpha, alpha_ch)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def remove_nebula_blobs_aligned(img: Image.Image) -> Image.Image:
    """
    Remove bulky dark-brown and dark-purple nebula/smoke halos attached to the character or broom
    in a 384x384 aligned cell while 100% preserving thin 1-3px dark-purple outlines, sailor top, skirt, and boots.
    """
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]
    max_rgb = np.maximum(np.maximum(r, g), b)

    # Protect character's solid dark-purple clothing & boots core zones:
    # Sailor top & skirt core: x in [148, 236], y in [172, 282]
    # Dark purple boots core: x in [150, 234], y in [312, 366]
    cloth_protect = np.zeros((h, w), dtype=bool)
    cloth_protect[172:282, 148:236] = True
    cloth_protect[312:366, 150:234] = True

    # 1) Warm dark-brown/muddy-gold smoke bands (R >= B - 4 and max_rgb < 118):
    # Note: character clothing/outlines are cool purple (B > R + 8). Broom handle is thin; smoke bands are bulky.
    warm_dark = (a > 0) & (max_rgb < 118) & (r >= b - 4)
    # 2) Cool/neutral dark-purple nebula clouds outside the protected clothing/boots core (max_rgb < 110)
    cool_dark_neb = (a > 0) & (max_rgb < 110) & (~cloth_protect)

    dark_candidate = warm_dark | cool_dark_neb

    # Erode by k=2 so thin 1-3px character outlines and broom handle outlines vanish and stay 100% protected,
    # while bulky nebula/smoke masses survive!
    def erode_mask(m: np.ndarray, k: int) -> np.ndarray:
        out_m = m.copy()
        for _ in range(k):
            e = out_m.copy()
            e[1:, :] &= out_m[:-1, :]
            e[:-1, :] &= out_m[1:, :]
            e[:, 1:] &= out_m[:, :-1]
            e[:, :-1] &= out_m[:, 1:]
            out_m = e
        return out_m

    def dilate_mask(m: np.ndarray, k: int) -> np.ndarray:
        out_m = m.copy()
        for _ in range(k):
            d = out_m.copy()
            d[1:, :] |= out_m[:-1, :]
            d[:-1, :] |= out_m[1:, :]
            d[:, 1:] |= out_m[:, :-1]
            d[:, :-1] |= out_m[:, 1:]
            out_m = d
        return out_m

    core = erode_mask(dark_candidate, 2)
    if core.any():
        # Regrow core within dark_candidate (stops before thin character outlines attached to bright hair/skin)
        # Keep pixels that are immediately adjacent (1px) to bright non-nebula character body (max_rgb >= 145)
        bright_body = (a > 180) & (max_rgb >= 145)
        outline_shield = dilate_mask(bright_body, 1) & (~warm_dark)
        regrow_domain = dark_candidate & (~outline_shield)

        nebula = dilate_mask(core, 3) & regrow_domain
        arr[nebula] = [0, 0, 0, 0]

        # Softly un-premultiply the 1-2px fringe around the erased nebula so golden stars inside have zero dark rim
        neb_fringe = dilate_mask(nebula, 2) & (arr[:, :, 3] > 0) & (~cloth_protect)
        fy, fx = np.where(neb_fringe)
        for y, x in zip(fy, fx):
            m = int(max_rgb[y, x])
            if m < 75:
                arr[y, x] = [0, 0, 0, 0]
            elif m < 165 and r[y, x] >= b[y, x] - 5:
                alpha_new = int(np.clip((m - 75) / 90.0 * 220, 0, 220))
                scale = 255.0 / max(1, m)
                arr[y, x, 0] = min(255, int(arr[y, x, 0] * scale))
                arr[y, x, 1] = min(255, int(arr[y, x, 1] * scale))
                arr[y, x, 2] = min(255, int(arr[y, x, 2] * scale))
                arr[y, x, 3] = alpha_new

    # Also clean thin muddy-brown arc lines near floor (y > 315) where R > B + 12 and max_rgb < 165
    floor_mud = (a > 0) & (r > b + 12) & (max_rgb < 165)
    floor_mud[:315, :] = False
    arr[floor_mud] = [0, 0, 0, 0]

    return Image.fromarray(arr, "RGBA")


def measure_character_head_body(img: Image.Image) -> Tuple[int, int, float]:
    """
    Measure character top y0, boot bottom y1, and head center x, ignoring wide broom bristles or particles.
    Uses the central vertical column band around the character's head so extending broom handles don't shift scale.
    """
    arr = np.array(img, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, img.height, img.width * 0.5
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    # Head band: top 8%..32% of main component
    top_mask = mask[y0 + int(0.06 * h): y0 + int(0.32 * h), :]
    top_xs = np.nonzero(top_mask)[1]
    head_cx = float(np.median(top_xs)) if len(top_xs) > 0 else float(np.median(xs))
    # Measure boot bottom near head_cx (+/- 22% of image width) so low broom bristles on far left/right don't alter ground
    band_half = int(img.width * 0.18)
    x_lo = max(0, int(round(head_cx - band_half)))
    x_hi = min(img.width, int(round(head_cx + band_half)))
    center_ys = np.nonzero(mask[:, x_lo:x_hi])[0]
    boot_y1 = int(center_ys.max()) if len(center_ys) > 0 else y1
    head_y0 = int(center_ys.min()) if len(center_ys) > 0 else y0
    return head_y0, boot_y1, head_cx


def align_sweeping_frame(f: Image.Image, ref_scale: float) -> Image.Image:
    """Align sweeping frame using consistent character height TARGET_H and grounded boots at GROUND_Y."""
    hy0, by1, _ = measure_character_head_body(f)
    char_h = max(1, by1 - hy0)
    # Blend frame's direct character height scale with ref_scale (clamped gently) to lock character size
    scale = float(np.clip(TARGET_H / char_h, ref_scale * 0.96, ref_scale * 1.04))

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    rhy0, rby1, rhead_cx = measure_character_head_body(resized)
    paste_x = int(round(CANVAS * 0.5 - rhead_cx))
    paste_y = int(round(GROUND_Y - rby1))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def draw_extra_starlight_particles(img: Image.Image, frame_idx: int, total_frames: int = 48) -> Image.Image:
    """
    Add extra crisp, high-contrast discrete 4-point magic stars & shimmering dust dots during active sweeping
    and broom-dissolve frames (frames 3..43), fading in and out smoothly so first/last 3 frames are 100% clean.
    """
    if frame_idx < 3 or frame_idx >= total_frames - 3:
        return img

    out = img.copy()
    overlay = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Smooth envelope (0 at ends, 1 in middle)
    env = min(1.0, (frame_idx - 2) / 5.0, (total_frames - 3 - frame_idx) / 5.0)

    # Deterministic procedural starlight emitters along sweep arc & around broom
    rng_seeds = [
        (82, 298, 6.5, (255, 217, 102), 0.0),    # warm gold low left
        (298, 292, 6.0, (255, 217, 102), 1.1),   # warm gold low right
        (110, 240, 5.5, (244, 114, 182), 2.2),   # pastel pink mid-left
        (282, 232, 5.5, (192, 132, 252), 3.3),   # bright violet mid-right
        (145, 328, 4.5, (255, 255, 255), 4.1),   # clean white floor left
        (242, 326, 4.5, (92, 225, 230), 5.2),    # cyan-white floor right
        (88, 175, 5.0, (255, 217, 102), 1.8),    # upper-left gold star
        (296, 168, 5.5, (244, 114, 182), 2.9),   # upper-right pink star
        (192, 336, 4.0, (255, 240, 150), 0.7),   # center floor dust
        (266, 268, 5.0, (255, 217, 102), 3.7),   # broom side gold star
    ]

    t = frame_idx / float(total_frames)
    for idx, (bx, by, rad, col, phase) in enumerate(rng_seeds):
        twinkle = 0.5 + 0.5 * math.sin(2.0 * math.pi * (t * 3.0) + phase)
        alpha = int(round(235 * env * (0.45 + 0.55 * twinkle)))
        if alpha < 35:
            continue
        # Gentle orbital drift
        px = bx + 10.0 * math.sin(2.0 * math.pi * t * 2.0 + phase)
        py = by - 14.0 * ((t * 2.0 + phase / 6.28) % 1.0)
        r = rad * (0.75 + 0.35 * twinkle)

        # Crisp 4-point star polygon with dark-purple micro-outline for dual-background visibility
        pts_outer = [
            (px, py - r * 1.25),
            (px + r * 0.32, py - r * 0.32),
            (px + r * 1.25, py),
            (px + r * 0.32, py + r * 0.32),
            (px, py + r * 1.25),
            (px - r * 0.32, py + r * 0.32),
            (px - r * 1.25, py),
            (px - r * 0.32, py - r * 0.32),
        ]
        draw.polygon(pts_outer, fill=(col[0], col[1], col[2], alpha), outline=(85, 55, 125, min(220, alpha)))
        # Bright white core dot
        cr = max(1.2, r * 0.25)
        draw.ellipse((px - cr, py - cr, px + cr, py + cr), fill=(255, 255, 255, min(255, int(alpha * 1.1))))

    out.alpha_composite(overlay)
    return out


def interpolate_smooth_no_burr(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """Bidirectional optical-flow midpoint with anti-aliased alpha curve (zero double-image ghosting, zero burrs)."""
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

    agree_weight = np.clip(1.0 - disagree * 2.6, 0.0, 1.0)
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
    # Smooth anti-aliased ramp [28..228] prevents jagged staircase burrs while eliminating faint ghosts
    out[:, :, 3] = np.where(
        out_a_u8 < 28, 0.0, np.where(out_a_u8 > 228, 255.0, (out_a_u8 - 28.0) / 200.0 * 255.0)
    )
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def main() -> dict:
    s1 = Image.open("gemini/素材/02sweeping/part1.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/02sweeping/part2.jpg").convert("RGBA")

    p1 = [clean_black_bg_rich_particles(s1.crop(b)) for b in find_cell_boxes(s1, 4, 3)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b)) for b in find_cell_boxes(s2, 4, 3)]

    hy0_0, by1_0, _ = measure_character_head_body(p1[0])
    ref_scale = TARGET_H / max(1.0, float(by1_0 - hy0_0))

    a1 = [remove_nebula_blobs_aligned(align_sweeping_frame(f, ref_scale)) for f in p1]
    a2 = [remove_nebula_blobs_aligned(align_sweeping_frame(f, ref_scale)) for f in p2]

    # 24 physically continuous sweeping keyframes:
    # Summon (4) -> Sweep 1 Left-to-Right (4) -> Sweep 2 Right-to-Left (3) ->
    # Sweep 3 Left-to-Right & Vertical Hold (5) -> Brow Wipe & Starlight Broom Dissolve (8)
    keys_24 = [
        a1[0], a1[1], a1[2], a1[3],
        a1[7], a1[8], a1[5], a1[4],
        a1[9], a1[10], a1[6],
        a2[2], a2[1], a1[11], a2[0], a2[3],
        a2[4], a2[5], a2[6], a2[7], a2[8], a2[9], a2[10], a1[0].copy(),
    ]
    assert len(keys_24) == 24

    # Expand 24 keyframes -> 48 silky-smooth frames:
    # Skip optical-flow warp on the 4 cross-body broom flip transitions {5, 7, 10, 12} so broom/tail never ghost;
    # compensate by inserting 2 smooth optical-flow frames (t=1/3, 2/3) on 4 same-side transitions {1, 6, 17, 18}.
    skip_cross = {5, 7, 10, 12}
    double_same = {1, 6, 17, 18}
    frames_48: List[Image.Image] = []
    for i in range(24):
        frames_48.append(keys_24[i])
        if i < 23:
            if i in skip_cross:
                continue
            elif i in double_same:
                m_mid = interpolate_smooth_no_burr(keys_24[i], keys_24[i + 1])
                m_1 = interpolate_smooth_no_burr(keys_24[i], m_mid)
                m_2 = interpolate_smooth_no_burr(m_mid, keys_24[i + 1])
                frames_48.extend([m_1, m_2])
            else:
                frames_48.append(interpolate_smooth_no_burr(keys_24[i], keys_24[i + 1]))
    frames_48.append(a1[0].copy())
    assert len(frames_48) == 48, len(frames_48)

    frames_48 = [draw_extra_starlight_particles(fr, idx, 48) for idx, fr in enumerate(frames_48)]

    char_heights = []
    for fr in frames_48:
        hy0, by1, _ = measure_character_head_body(fr)
        char_heights.append(by1 - hy0)

    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "02sweeping_sheet（打扫_图集）.webp")
    anim_path = os.path.join(out_dir, "02sweeping（打扫_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Doc target: 8.5s total (sweep power frames 135ms, normal 180ms, holds/brow-wipe/dissolve 225ms -> ~8.48s)
    fast_set = {7, 8, 9, 10, 11, 15, 16, 17, 18, 21, 22, 23, 24}
    hold_set = {0, 1, 5, 6, 13, 14, 19, 20, 26, 27, 32, 33, 34, 35, 36, 37, 46, 47}
    durations = [135 if i in fast_set else (225 if i in hold_set else 180) for i in range(48)]

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
    white_path = os.path.join(prev_dir, "02sweeping_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "02sweeping_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "02sweeping",
        "frames": 48,
        "grid": "8x6",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "char_height_min": min(char_heights),
        "char_height_max": max(char_heights),
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
