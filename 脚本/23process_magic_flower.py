import importlib.util
import json
import os
from collections import deque
from typing import List

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask

_spec = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned

_spec18 = importlib.util.spec_from_file_location("n18", "gemini/脚本/18process_note_taking.py")
_n18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(_n18)
align_cropped_with_ref_boots = _n18.align_cropped_with_ref_boots

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
OUTLINE_RGB = np.array([68.0, 48.0, 98.0], dtype=np.float32)


def clean_dark_gradient_bg(cell_img: Image.Image) -> Image.Image:
    """
    Remove dark greenish-grey gradient background (as seen in 23magic_flower/part1.jpg)
    using row-wise border background estimation + flood fill + anti-aliased purple outline decontamination.
    """
    arr = np.array(cell_img.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    rgb = arr[:, :, :3].astype(np.float32)

    # Estimate background RGB per row from left/right 10px margin strips
    row_bg = np.zeros((h, 3), dtype=np.float32)
    for y in range(h):
        strip = np.concatenate([rgb[y, :10, :], rgb[y, -10:, :]], axis=0)
        row_bg[y] = np.median(strip, axis=0)

    dist = np.linalg.norm(rgb - row_bg[:, None, :], axis=2)
    is_bg_cand = dist < 24.0

    visited = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if is_bg_cand[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if is_bg_cand[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        bg_mask[cy, cx] = True
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg_cand[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    # Also clear enclosed leg/hair gaps matching row background closely
    bg_mask |= (dist < 18.0) & (~bg_mask)
    fg_mask = ~bg_mask

    out = arr.astype(np.float32)
    out[bg_mask] = 0.0

    main_mask = get_main_component_mask((fg_mask.astype(np.uint8) * 255))
    boundary = binary_dilation(main_mask, iterations=1) & (~binary_erosion(main_mask, iterations=2))
    max_rgb = rgb.max(axis=2)
    dark_edge = boundary & (max_rgb < 115.0) & (out[:, :, 3] > 0)
    out[dark_edge, :3] = out[dark_edge, :3] * 0.45 + OUTLINE_RGB * 0.55

    smooth_a = gaussian_filter(out[:, :, 3], sigma=0.58)
    out[boundary, 3] = smooth_a[boundary]
    out[out[:, :, 3] < 10.0] = 0.0

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def clean_sheet_for_split(sheet_rgba: Image.Image) -> Image.Image:
    """Zero out dark greenish-grey gradient background across the full sheet so find_cell_boxes splits cleanly."""
    arr = np.array(sheet_rgba.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    row_bg = np.zeros((h, 3), dtype=np.float32)
    for y in range(h):
        strip = np.concatenate([arr[y, :12, :], arr[y, -12:, :]], axis=0)
        row_bg[y] = np.median(strip, axis=0)
    dist = np.linalg.norm(arr - row_bg[:, None, :], axis=2)
    cleaned = arr.copy()
    cleaned[dist < 26.0] = 0.0
    return Image.fromarray(cleaned.astype(np.uint8), "RGB").convert("RGBA")


def main() -> dict:
    s1 = Image.open("gemini/素材/23magic_flower/part1.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/23magic_flower/part2.jpg").convert("RGBA")

    b1 = find_cell_boxes(clean_sheet_for_split(s1), 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    # Part 1 has dark greenish-grey gradient bg; Part 2 is pure black bg
    p1 = [clean_dark_gradient_bg(s1.crop(b1[i])) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    ref_stand = remove_nebula_blobs_aligned(align_full_body_at(p1[0], GROUND_Y))

    a1: List[Image.Image] = []
    for i in range(12):
        if i == 0:
            a1.append(ref_stand)
        elif i == 9:
            # Skip p1[9] where rose accidentally vanished before tossing
            a1.append(None)  # type: ignore[arg-type]
        else:
            a1.append(remove_nebula_blobs_aligned(align_full_body_at(p1[i], GROUND_Y)))

    # Part 2 Row 1 (indices 0..3) has slightly cropped boot bottoms -> full-width head-locked composite with ref_stand
    a2: List[Image.Image] = []
    for i in range(9):
        if i <= 3:
            a2.append(remove_nebula_blobs_aligned(align_cropped_with_ref_boots(p2[i], ref_stand, body_ratio=0.96)))
        else:
            a2.append(remove_nebula_blobs_aligned(align_full_body_at(p2[i], GROUND_Y)))

    # 21 keyframes -> 17 intervals * 2 + 3 intervals * 3 = 64 frames (8x8)
    keys_21 = a1[0:9] + a1[10:12] + a2[0:9] + [ref_stand.copy()]
    assert len(keys_21) == 21

    # Triple-midpoint on: conjuring rose (idx 3), wink->toss prep (idx 8), petals bursting down (idx 11)
    triple_intervals = {3, 8, 11}
    frames_64: List[Image.Image] = []
    for i in range(21):
        frames_64.append(keys_21[i])
        if i < 20:
            if i in triple_intervals:
                frames_64.extend(interpolate_flow_multi(keys_21[i], keys_21[i + 1], [0.25, 0.5, 0.75]))
            else:
                frames_64.extend(interpolate_flow_multi(keys_21[i], keys_21[i + 1], [1.0 / 3.0, 2.0 / 3.0]))
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

    sheet_path = os.path.join(out_dir, "23magic_flower_sheet（凭空生花_图集）.webp")
    anim_path = os.path.join(out_dir, "23magic_flower（凭空生花_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 64 frames variable durations (~9.0s): magic flip & toss 115ms, sniff/wink/petal watch/blink/ends 185ms, normal 135ms
    fast_set = set(range(9, 16)) | set(range(28, 35))
    hold_set = {0, 21, 22, 24, 25, 42, 45, 57, 63}
    durations = [115 if i in fast_set else (185 if i in hold_set else 135) for i in range(64)]

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
    white_path = os.path.join(prev_dir, "23magic_flower_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "23magic_flower_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "23magic_flower",
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
