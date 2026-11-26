import json
import os
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image

from build_webp import save_qa_previews
from expand_48 import interpolate_midpoint_no_ghost, measure_body_stats


def find_white_grid_boxes(sheet: Image.Image, cols: int = 4, rows: int = 3, thresh: int = 242) -> List[Tuple[int, int, int, int]]:
    """Find per-column cell boxes on a pure white background sprite sheet."""
    arr = np.array(sheet.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    non_bg = (arr.min(axis=2) < thresh).astype(np.int32)

    col_energy = non_bg.sum(axis=0)
    col_cuts = [0]
    col_radius = max(12, int(w / cols * 0.22))
    for i in range(1, cols):
        nominal = int(round(w * i / cols))
        lo = max(col_cuts[-1] + 10, nominal - col_radius)
        hi = min(w - 10, nominal + col_radius)
        window = col_energy[lo:hi + 1]
        min_val = int(window.min())
        mins = np.where(window == min_val)[0] + lo
        col_cuts.append(int(mins[np.argmin(np.abs(mins - nominal))]))
    col_cuts.append(w)
    col_bounds = [(col_cuts[i], col_cuts[i + 1]) for i in range(cols)]

    row_radius = max(12, int(h / rows * 0.22))
    grid_boxes: List[List[Tuple[int, int, int, int]]] = [[None] * cols for _ in range(rows)]  # type: ignore

    for c_idx, (c0, c1) in enumerate(col_bounds):
        row_energy = non_bg[:, c0:c1].sum(axis=1)
        row_cuts = [0]
        for r_idx in range(1, rows):
            nominal = int(round(h * r_idx / rows))
            lo = max(row_cuts[-1] + 10, nominal - row_radius)
            hi = min(h - 10, nominal + row_radius)
            window = row_energy[lo:hi + 1]
            min_val = int(window.min())
            mins = np.where(window == min_val)[0] + lo
            row_cuts.append(int(mins[np.argmin(np.abs(mins - nominal))]))
        row_cuts.append(h)
        for r_idx in range(rows):
            grid_boxes[r_idx][c_idx] = (c0, row_cuts[r_idx], c1, row_cuts[r_idx + 1])

    return [grid_boxes[r][c] for r in range(rows) for c in range(cols)]


def clean_white_bg(img_rgba: Image.Image, bg_thresh: int = 236) -> Image.Image:
    """
    Remove pure white background via 4-edge flood-fill + anti-aliased white-fringe decontamination.
    Preserves interior white clothing (sailor collar, sleeves, lace) enclosed by dark purple outlines.
    """
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    min_rgb = np.minimum(np.minimum(r, g), b)
    chroma = np.maximum(np.maximum(r, g), b) - min_rgb

    # Exterior flood-fill on near-white low-chroma background
    is_white = (min_rgb >= bg_thresh) & (chroma <= 18)
    visited = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    q = deque()

    for x in range(w):
        if is_white[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if is_white[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if is_white[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if is_white[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        cy, cx = q.popleft()
        bg_mask[cy, cx] = True
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_white[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    # Also remove pure-white enclosed gaps between hair/tail/legs (very pure white >=247 in lower 65% of cell)
    pure_white_hole = (min_rgb >= 247) & (chroma <= 6) & (~bg_mask)
    # Only clear holes in the lower 50% (between legs/hair/tail) or touching outer hair gaps of area >= 25px
    hole_visited = np.zeros((h, w), dtype=bool)
    for y in range(int(h * 0.45), h):
        for x in range(w):
            if pure_white_hole[y, x] and not hole_visited[y, x]:
                pts = []
                hq = deque([(y, x)])
                hole_visited[y, x] = True
                while hq:
                    py, px = hq.popleft()
                    pts.append((py, px))
                    for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                        if 0 <= ny < h and 0 <= nx < w and pure_white_hole[ny, nx] and not hole_visited[ny, nx]:
                            hole_visited[ny, nx] = True
                            hq.append((ny, nx))
                # Clear medium/large pure-white pockets in lower half (gaps between legs/hair/tail)
                if len(pts) >= 40:
                    ys_p = [p[0] for p in pts]
                    # Skip upper-body white sleeves/lace (centered around 45%..68% height)
                    if max(ys_p) > int(h * 0.72):
                        for py, px in pts:
                            bg_mask[py, px] = True

    out = arr.copy()
    out[bg_mask] = [0, 0, 0, 0]

    # Decontaminate 1px white anti-aliased fringe touching bg
    fg_mask = ~bg_mask
    bg_dilated = np.zeros((h, w), dtype=bool)
    bg_dilated[1:, :] |= bg_mask[:-1, :]
    bg_dilated[:-1, :] |= bg_mask[1:, :]
    bg_dilated[:, 1:] |= bg_mask[:, :-1]
    bg_dilated[:, :-1] |= bg_mask[:, 1:]
    edge = fg_mask & bg_dilated
    ey, ex = np.where(edge)
    for y, x in zip(ey, ex):
        mn = int(min_rgb[y, x])
        if mn > 190 and int(chroma[y, x]) < 35:
            # Bright grey/white anti-aliased edge -> convert to soft alpha on dark-purple tint
            alpha = max(0, int((255 - mn) / 65.0 * 255))
            out[y, x, 0] = min(out[y, x, 0], 95)
            out[y, x, 1] = min(out[y, x, 1], 75)
            out[y, x, 2] = min(out[y, x, 2], 125)
            out[y, x, 3] = alpha

    return Image.fromarray(out, "RGBA")


def build_wave_hello_36(src_path: str, out_dir: str, preview_dir: str, canvas_size: int = 256) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    sheet = Image.open(src_path).convert("RGBA")
    boxes = find_white_grid_boxes(sheet, cols=4, rows=3)
    # Use Frames 0..7 (Rows 1 & 2: clean full-boot right-arm lift & energetic wave)
    raw_8 = [clean_white_bg(sheet.crop(boxes[i])) for i in range(8)]

    stats = [measure_body_stats(f) for f in raw_8]
    ref_head_w = float(np.median([s[4] for s in stats]))
    f0_body_h = stats[0][3] - stats[0][1]
    base_scale = (canvas_size * 0.76) / max(1, f0_body_h)
    base_bottom_y = int(canvas_size * 0.90)

    aligned_8: List[Image.Image] = []
    for f, (bx0, by0, bx1, by1, hw) in zip(raw_8, stats):
        corr = float(np.clip(ref_head_w / max(1.0, hw), 0.95, 1.06))
        scale = base_scale * corr
        nw = max(1, int(round(f.width * scale)))
        nh = max(1, int(round(f.height * scale)))
        resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

        body_cx = ((bx0 + bx1) * 0.5) * scale
        body_bot = by1 * scale
        paste_x = int(round(canvas_size * 0.5 - body_cx))
        paste_y = int(round(base_bottom_y - body_bot))

        canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y), resized)
        aligned_8.append(canvas)

    # Build natural 3-wave cycle (19 keyframes):
    # Lift: 0->1->2->3->4 | Wave 1: 5->6->7 | Wave 2: 6->5->4->5->6->7 | Wave 3 & lower: 6->4->3->2->0
    seq_indices = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 6, 4, 3, 2, 0]
    keyframes = [aligned_8[i] for i in seq_indices]

    # Cache optical-flow midpoints for each unique ordered pair (a, b) for speed & exact symmetry
    pair_cache = {}
    expanded: List[Image.Image] = []
    for k in range(len(seq_indices)):
        expanded.append(keyframes[k])
        if k < len(seq_indices) - 1:
            a_i, b_i = seq_indices[k], seq_indices[k + 1]
            if (a_i, b_i) not in pair_cache:
                pair_cache[(a_i, b_i)] = interpolate_midpoint_no_ghost(aligned_8[a_i], aligned_8[b_i])
            expanded.append(pair_cache[(a_i, b_i)])

    # 19 keyframes + 18 midpoints = 37 -> trim last redundant duplicate to 36 (clean 6x6 with Frame 0 == Frame 35)
    frames_36 = expanded[:35] + [aligned_8[0].copy()]
    assert len(frames_36) == 36

    cols, rows = 6, 6
    out_sheet = Image.new("RGBA", (cols * canvas_size, rows * canvas_size), (0, 0, 0, 0))
    for i, f in enumerate(frames_36):
        r, c = divmod(i, cols)
        out_sheet.paste(f, (c * canvas_size, r * canvas_size), f)

    sheet_path = os.path.join(out_dir, "wave_hello_sheet.webp")
    out_sheet.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Variable-speed durations (~6.48s total, crisp waves 145ms, normal 180ms, start/peaks/end 240ms)
    fast_set = set(range(10, 28)) - {14, 20, 26}
    hold_set = {0, 1, 14, 20, 26, 34, 35}
    durations = [145 if i in fast_set else (240 if i in hold_set else 185) for i in range(36)]

    anim_path = os.path.join(out_dir, "wave_hello.webp")
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

    white_bg = Image.new("RGBA", out_sheet.size, (255, 255, 255, 255))
    white_bg.alpha_composite(out_sheet)
    white_path = os.path.join(preview_dir, "wave_hello_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", out_sheet.size, (0, 0, 0, 255))
    black_bg.alpha_composite(out_sheet)
    black_path = os.path.join(preview_dir, "wave_hello_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "wave_hello",
        "frames": 36,
        "grid": "6x6",
        "total_seconds": round(sum(durations) / 1000.0, 2),
        "sheet_path": sheet_path,
        "anim_path": anim_path,
        "preview_white": white_path,
        "preview_black": black_path,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    build_wave_hello_36(
        src_path="gemini/素材/wave_hello/part1_white.jpg",
        out_dir="gemini/成品",
        preview_dir="gemini/预览",
        canvas_size=256,
    )
