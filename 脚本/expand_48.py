import json
import math
import os
from collections import deque
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from build_webp import clean_black_bg_and_glow, find_cell_boxes, save_qa_previews


def get_main_component_mask(alpha: np.ndarray) -> np.ndarray:
    """Return boolean mask of the largest connected foreground component (character body)."""
    h, w = alpha.shape
    fg = alpha > 40
    comp = np.zeros((h, w), dtype=np.int32)
    cur = 0
    best_id, best_area = 0, 0
    for y in range(h):
        for x in range(w):
            if fg[y, x] and comp[y, x] == 0:
                cur += 1
                area = 0
                q = deque([(y, x)])
                comp[y, x] = cur
                while q:
                    py, px = q.popleft()
                    area += 1
                    for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                        if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and comp[ny, nx] == 0:
                            comp[ny, nx] = cur
                            q.append((ny, nx))
                if area > best_area:
                    best_area, best_id = area, cur
    return comp == best_id


def measure_body_stats(img: Image.Image) -> Tuple[int, int, int, int, float]:
    """Return (x0, y0, x1, y1, head_width) of the character's main connected component."""
    arr = np.array(img, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, img.width, img.height, float(img.width)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    body_h = y1 - y0
    # Measure head+ear width in the upper 18%..42% band of the body (stable across standing/crouch/jump)
    band_top = y0 + int(body_h * 0.18)
    band_bot = y0 + int(body_h * 0.42)
    band = mask[band_top:band_bot, :]
    row_widths = []
    for r in range(band.shape[0]):
        rxs = np.where(band[r])[0]
        if len(rxs) > 0:
            row_widths.append(int(rxs.max() - rxs.min() + 1))
    head_w = float(np.median(row_widths)) if row_widths else float(x1 - x0)
    return x0, y0, x1, y1, head_w


def pyramidal_optical_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 4,
    win_sigma: float = 3.5,
    iters: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarse-to-fine pyramidal Lucas-Kanade dense optical flow in pure NumPy/SciPy.
    Returns (vx, vy) mapping pixels in img1 -> img2.
    """
    h, w = img1.shape

    def build_pyr(im: np.ndarray) -> List[np.ndarray]:
        pyr = [im.astype(np.float32)]
        for _ in range(1, levels):
            blurred = gaussian_filter(pyr[-1], sigma=1.0)
            pyr.append(blurred[::2, ::2])
        return pyr

    p1 = build_pyr(img1)
    p2 = build_pyr(img2)

    vx = np.zeros_like(p1[-1], dtype=np.float32)
    vy = np.zeros_like(p1[-1], dtype=np.float32)

    for lvl in range(levels - 1, -1, -1):
        i1 = p1[lvl]
        i2 = p2[lvl]
        lh, lw = i1.shape
        if lvl < levels - 1:
            sy = lh / vx.shape[0]
            sx = lw / vx.shape[1]
            vx = zoom(vx, (sy, sx), order=1) * sx
            vy = zoom(vy, (sy, sx), order=1) * sy

        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        for _ in range(iters):
            wx = np.clip(xx + vx, 0, lw - 1)
            wy = np.clip(yy + vy, 0, lh - 1)
            i2_w = map_coordinates(i2, [wy, wx], order=1, mode="nearest")

            # Spatial & temporal gradients
            ix = np.gradient(i2_w, axis=1)
            iy = np.gradient(i2_w, axis=0)
            it = i2_w - i1

            s_xx = gaussian_filter(ix * ix, sigma=win_sigma) + 1e-3
            s_yy = gaussian_filter(iy * iy, sigma=win_sigma) + 1e-3
            s_xy = gaussian_filter(ix * iy, sigma=win_sigma)
            s_xt = gaussian_filter(ix * it, sigma=win_sigma)
            s_yt = gaussian_filter(iy * it, sigma=win_sigma)

            det = s_xx * s_yy - s_xy * s_xy + 1e-4
            dvx = -(s_yy * s_xt - s_xy * s_yt) / det
            dvy = -(s_xx * s_yt - s_xy * s_xt) / det

            # Clamp per-iteration update for stability
            dvx = np.clip(dvx, -2.5, 2.5)
            dvy = np.clip(dvy, -2.5, 2.5)

            vx = gaussian_filter(vx + dvx, sigma=1.2)
            vy = gaussian_filter(vy + dvy, sigma=1.2)

    return vx, vy


def warp_rgba(arr: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Warp RGBA image using displacement field (dx, dy) with bilinear interpolation."""
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sy = np.clip(yy + dy, 0, h - 1)
    sx = np.clip(xx + dx, 0, w - 1)

    # Premultiply RGB by alpha before warping to prevent dark fringe bleeding
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    out = np.zeros((h, w, 4), dtype=np.float32)
    wa = map_coordinates(alpha, [sy, sx], order=1, mode="constant", cval=0.0)
    for c in range(3):
        premul = arr[:, :, c].astype(np.float32) * alpha
        wc = map_coordinates(premul, [sy, sx], order=1, mode="constant", cval=0.0)
        out[:, :, c] = np.where(wa > 1e-3, np.clip(wc / np.maximum(wa, 1e-3), 0, 255), 0.0)
    out[:, :, 3] = np.clip(wa * 255.0, 0, 255)
    return out


def interpolate_midpoint_no_ghost(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """
    Generate a crisp midpoint frame between img_a and img_b using bidirectional optical flow warping.
    Uses consistency-gated blending + crisp alpha sharpening to eliminate double-image ghosting.
    """
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    # Build luminance + silhouette signal for accurate cel-shaded flow tracking
    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa = feature_map(a)
    fb = feature_map(b)

    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=4, win_sigma=3.5, iters=6)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=4, win_sigma=3.5, iters=6)

    # Warp A forward by +0.5 and B backward by +0.5 into the midpoint frame
    wa = warp_rgba(a, -0.5 * vx_ab, -0.5 * vy_ab)
    wb = warp_rgba(b, -0.5 * vx_ba, -0.5 * vy_ba)

    # Measure pixel disagreement between forward and backward warped frames
    rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
    alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
    disagree = rgb_diff * 0.7 + alpha_diff * 0.9

    # Where warps agree -> smooth 50/50 blend; where they disagree -> snap toward sharper/higher-alpha frame
    # to prevent translucent double-image ghosting
    agree_weight = np.clip(1.0 - disagree * 2.8, 0.0, 1.0)
    pick_a = (wa[:, :, 3] >= wb[:, :, 3]).astype(np.float32)
    w_a = 0.5 * agree_weight + pick_a * (1.0 - agree_weight)
    w_b = 1.0 - w_a

    # Premultiplied blend
    aa = wa[:, :, 3] / 255.0
    ab = wb[:, :, 3] / 255.0
    out_a = w_a * aa + w_b * ab

    out = np.zeros_like(wa)
    for c in range(3):
        num = w_a * wa[:, :, c] * aa + w_b * wb[:, :, c] * ab
        out[:, :, c] = np.where(out_a > 1e-3, np.clip(num / np.maximum(out_a, 1e-3), 0, 255), 0.0)

    # Crisp alpha curve to keep cel-shaded edges sharp and avoid semi-transparent halos
    out_a_u8 = np.clip(out_a * 255.0, 0, 255)
    out_a_sharp = np.where(out_a_u8 < 55, 0.0, np.where(out_a_u8 > 205, 255.0, (out_a_u8 - 55) / 150.0 * 255.0))
    out[:, :, 3] = out_a_sharp

    return Image.fromarray(out.astype(np.uint8), "RGBA")


def build_smooth_48_happy_jump(
    segment_paths: Sequence[str],
    out_dir: str,
    canvas_size: int = 256,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    raw_frames: List[Image.Image] = []
    for path in segment_paths:
        sheet = Image.open(path).convert("RGBA")
        for box in find_cell_boxes(sheet, cols=4, rows=3):
            raw_frames.append(clean_black_bg_and_glow(sheet.crop(box)))

    # 1. Measure body stats & normalize head width so Row 3 isn't shrunk by heart particles
    stats = [measure_body_stats(f) for f in raw_frames]
    ref_head_w = float(np.median([s[4] for s in stats]))

    # Target body height so character + heart fits comfortably with >=10% margins
    norm_frames: List[Image.Image] = []
    body_bottoms = []
    body_centers_x = []

    # Compute base scale from Frame 0 body height -> 66% of canvas (leaves room for jump lift + heart!)
    f0_body_h = stats[0][3] - stats[0][1]
    base_scale = (canvas_size * 0.66) / max(1, f0_body_h)

    # Smooth parabolic jump lift for mid-air frames (indices 4..15 in 24-frame sequence)
    jump_lift_24 = [0.0] * len(raw_frames)
    air_indices = list(range(4, 15))
    max_lift_px = canvas_size * 0.085  # ~22px smooth jump height
    for k, idx in enumerate(air_indices):
        t = (k + 0.5) / len(air_indices)
        jump_lift_24[idx] = max_lift_px * math.sin(math.pi * t)

    base_bottom_y = int(canvas_size * 0.89)  # >=10% bottom margin

    for idx, (f, (bx0, by0, bx1, by1, hw)) in enumerate(zip(raw_frames, stats)):
        # Per-frame head-width correction clamped gently so proportions stay rock-solid
        corr = float(np.clip(ref_head_w / max(1.0, hw), 0.94, 1.14))
        scale = base_scale * corr

        bbox = f.getbbox()
        if not bbox:
            norm_frames.append(Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0)))
            continue

        # Resize entire frame (including particles) by the same scale
        nw = max(1, int(round(f.width * scale)))
        nh = max(1, int(round(f.height * scale)))
        resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

        # Align character's body horizontal center to canvas X=50%, and boots to base_bottom_y - jump_lift
        body_cx = ((bx0 + bx1) * 0.5) * scale
        body_bot = by1 * scale

        paste_x = int(round(canvas_size * 0.5 - body_cx))
        paste_y = int(round(base_bottom_y - jump_lift_24[idx] - body_bot))

        # Safety clamp so top particles never clip top 4% edge
        top_pixel = bbox[1] * scale
        if paste_y + top_pixel < canvas_size * 0.04:
            paste_y = int(round(canvas_size * 0.04 - top_pixel))

        canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y), resized)
        norm_frames.append(canvas)

    # Lock last keyframe to Frame 0 for clean loop closure
    norm_frames[-1] = norm_frames[0].copy()

    # 2. Expand 24 keyframes -> 48 frames (insert 1 optical-flow warped midpoint between each adjacent pair)
    expanded: List[Image.Image] = []
    n_key = len(norm_frames)
    for i in range(n_key):
        expanded.append(norm_frames[i])
        if i < n_key - 1:
            mid = interpolate_midpoint_no_ghost(norm_frames[i], norm_frames[i + 1])
            expanded.append(mid)
    # 24 keyframes + 23 midpoints = 47 frames; append final Frame 0 hold -> 48 frames (clean 8x6!)
    expanded.append(norm_frames[0].copy())

    assert len(expanded) == 48

    # 3. Export 8x6 lossless sprite sheet
    cols, rows = 8, 6
    sheet = Image.new("RGBA", (cols * canvas_size, rows * canvas_size), (0, 0, 0, 0))
    for i, f in enumerate(expanded):
        r, c = divmod(i, cols)
        sheet.paste(f, (c * canvas_size, r * canvas_size), f)

    action_name = "10_happy_jump_48f"
    sheet_path = os.path.join(out_dir, f"{action_name}_sheet_8x6.webp")
    sheet.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 4. Variable-speed durations for 48 frames (~6.8s total, silky smooth!)
    # Keyframe indices in 48-frame timeline are 2*k
    fast_set = set(range(8, 16)) | set(range(24, 32))       # push-off ascent & descent: crisp 110ms
    hold_set = {0, 1, 17, 18, 19, 20, 21, 22, 32, 33, 46, 47}  # start, apex hang, landing, end: 180ms
    durations = [
        110 if i in fast_set else (180 if i in hold_set else 140)
        for i in range(48)
    ]

    anim_path = os.path.join(out_dir, f"{action_name}_anim.webp")
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

    qa_dir = os.path.join(out_dir, "qa_previews")
    os.makedirs(qa_dir, exist_ok=True)
    white_path, black_path = save_qa_previews(sheet, qa_dir, action_name)

    res = {
        "action": action_name,
        "frames": 48,
        "grid": "8x6",
        "total_seconds": round(sum(durations) / 1000.0, 2),
        "sheet_path": sheet_path,
        "anim_path": anim_path,
        "preview_white": white_path,
        "preview_black": black_path,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    build_smooth_48_happy_jump(
        segment_paths=[
            "gemini/素材/happy_jump/part1.jpg",
            "gemini/素材/happy_jump/part2.jpg",
        ],
        out_dir="gemini/成品",
        canvas_size=256,
    )
