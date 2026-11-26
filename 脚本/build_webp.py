import argparse
import json
import math
import os
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def clean_black_bg_and_glow(
    img_rgba: Image.Image,
    bg_thresh: int = 24,
    hole_thresh: int = 20,
    glow_cutoff: int = 34,
) -> Image.Image:
    """
    Three-stage black-background removal for cel-shaded anime sprites + bright particles:
    1) 4-edge flood fill removes exterior near-black background.
    2) Interior pure-black gaps (between legs/arms/tail) are removed while preserving dark purple clothing.
    3) Non-main connected components (floating glowing particles) are un-premultiplied to eliminate dark halos.
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
                    for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                        if 0 <= ny < h and 0 <= nx < w and fg_mask[ny, nx] and comp_id[ny, nx] == 0:
                            comp_id[ny, nx] = cur_id
                            cq.append((ny, nx))
                if area > best_area:
                    best_area, best_id = area, cur_id

    out = arr.copy()
    out[bg_mask] = [0, 0, 0, 0]

    # Softly clean dark fringe pixels immediately touching background on the main character
    main_mask = comp_id == best_id
    bg_dilated = np.zeros((h, w), dtype=bool)
    bg_dilated[1:, :] |= bg_mask[:-1, :]
    bg_dilated[:-1, :] |= bg_mask[1:, :]
    bg_dilated[:, 1:] |= bg_mask[:, :-1]
    bg_dilated[:, :-1] |= bg_mask[:, 1:]
    main_edge = main_mask & bg_dilated
    edge_ys, edge_xs = np.where(main_edge)
    for y, x in zip(edge_ys, edge_xs):
        m = int(max_rgb[y, x])
        if m < 42:
            # Dark anti-aliased edge against black bg -> slightly soften alpha
            out[y, x, 3] = max(0, int((m - bg_thresh) / max(1, 42 - bg_thresh) * 255))

    # Un-premultiply independent floating particles & remove dim dark noise.
    # NOTE: scaling RGB by 255/luma turns a dark, saturated pixel (a nebula
    # wisp, or the dark interior of a "< / >" glyph) into an opaque black
    # island. Only pixels that are genuinely luminous get un-premultiplied;
    # anything darker than `glow_floor` is either faint glow (low alpha, keep
    # colour) or background spill (drop it).
    particle_mask = fg_mask & (comp_id != best_id)
    glow_floor = 70
    ys, xs = np.where(particle_mask)
    for y, x in zip(ys, xs):
        m = int(max_rgb[y, x])
        if m < glow_cutoff:
            out[y, x] = [0, 0, 0, 0]
        elif m < glow_floor:
            # Dark spill next to a particle: keep it nearly invisible, and do
            # NOT brighten the colour.
            out[y, x, 3] = max(0, int((m - glow_cutoff) / max(1, glow_floor - glow_cutoff) * 90))
        elif m < 235:
            alpha = min(255, int((m - glow_cutoff) / max(1, 235 - glow_cutoff) * 255))
            scale = 255.0 / max(1, m)
            out[y, x, 0] = min(255, int(out[y, x, 0] * scale))
            out[y, x, 1] = min(255, int(out[y, x, 1] * scale))
            out[y, x, 2] = min(255, int(out[y, x, 2] * scale))
            out[y, x, 3] = alpha

    return Image.fromarray(out, "RGBA")


def find_cell_boxes(sheet: Image.Image, cols: int = 4, rows: int = 3, thresh: int = 22) -> List[Tuple[int, int, int, int]]:
    """
    Find exact (x0, y0, x1, y1) crop boxes for each cell in a cols x rows sprite sheet.
    1) Split columns first using vertical dark valleys.
    2) Within each column independently, locate the horizontal dark gap between adjacent rows,
       preferring the gap immediately below the upper character's boots (so floating head
       particles like hearts/stars always stay attached to the lower character).
    """
    arr = np.array(sheet.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    non_bg = (arr.max(axis=2) > thresh).astype(np.int32)

    # 1. Split columns globally
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
        best_cut = int(mins[np.argmin(np.abs(mins - nominal))])
        col_cuts.append(best_cut)
    col_cuts.append(w)
    col_bounds = [(col_cuts[i], col_cuts[i + 1]) for i in range(cols)]

    # 2. Split rows independently per column
    row_radius = max(12, int(h / rows * 0.24))
    grid_boxes: List[List[Tuple[int, int, int, int]]] = [[None] * cols for _ in range(rows)]  # type: ignore

    for c_idx, (c0, c1) in enumerate(col_bounds):
        col_slice = non_bg[:, c0:c1]
        row_energy = col_slice.sum(axis=1)
        row_cuts = [0]
        for r_idx in range(1, rows):
            nominal = int(round(h * r_idx / rows))
            lo = max(row_cuts[-1] + 10, nominal - row_radius)
            hi = min(h - 10, nominal + row_radius)
            window = row_energy[lo:hi + 1]
            # Find zero-energy (or near-zero <= 2px) contiguous horizontal gap bands
            min_val = int(window.min())
            tol = max(min_val, 2)
            gap_mask = window <= tol
            # Group contiguous gap bands
            bands = []
            in_band = False
            b_start = 0
            for idx, val in enumerate(gap_mask):
                if val and not in_band:
                    in_band = True
                    b_start = idx
                elif not val and in_band:
                    in_band = False
                    bands.append((lo + b_start, lo + idx - 1))
            if in_band:
                bands.append((lo + b_start, lo + len(gap_mask) - 1))

            # Prefer the FIRST (uppermost) gap band that has real density above it (boots)
            # and width >= 3px, so floating particles above the lower character's head stay below the cut.
            valid_bands = [b for b in bands if (b[1] - b[0] + 1) >= 3]
            if not valid_bands:
                valid_bands = bands
            # Pick the uppermost band whose midpoint is within reasonable range of nominal
            chosen = valid_bands[0]
            best_cut = (chosen[0] + chosen[1]) // 2
            row_cuts.append(best_cut)
        row_cuts.append(h)

        for r_idx in range(rows):
            grid_boxes[r_idx][c_idx] = (c0, row_cuts[r_idx], c1, row_cuts[r_idx + 1])

    # Flatten in row-major order (left-to-right, top-to-bottom)
    return [grid_boxes[r][c] for r in range(rows) for c in range(cols)]


def extract_relay_anchors(segment_path: str, out_dir: str, prefix: str) -> Tuple[str, str]:
    """Crop cell 0 (first frame) and cell 11 (last frame) from a 4x3 segment sheet using smart per-column bounds."""
    os.makedirs(out_dir, exist_ok=True)
    sheet = Image.open(segment_path).convert("RGB")
    boxes = find_cell_boxes(sheet, cols=4, rows=3)
    first_cell = sheet.crop(boxes[0])
    last_cell = sheet.crop(boxes[-1])

    # Add small pure black padding around anchor so it's never tight at edges
    def pad_black(im: Image.Image, pad_ratio: float = 0.12) -> Image.Image:
        pw = int(im.width * pad_ratio)
        ph = int(im.height * pad_ratio)
        bg = Image.new("RGB", (im.width + 2 * pw, im.height + 2 * ph), (0, 0, 0))
        bg.paste(im, (pw, ph))
        return bg

    first_path = os.path.join(out_dir, f"{prefix}_first.png")
    last_path = os.path.join(out_dir, f"{prefix}_last.png")
    pad_black(first_cell).save(first_path)
    pad_black(last_cell).save(last_path)
    return first_path, last_path


def slice_and_align_segments(
    segment_paths: Sequence[str],
    canvas_size: int = 256,
    is_drag: bool = False,
    is_sleep_loop: bool = False,
    lock_first_last: bool = True,
) -> List[Image.Image]:
    raw_frames: List[Image.Image] = []
    for path in segment_paths:
        sheet = Image.open(path).convert("RGBA")
        boxes = find_cell_boxes(sheet, cols=4, rows=3)
        for box in boxes:
            cell = sheet.crop(box)
            raw_frames.append(clean_black_bg_and_glow(cell))

    bboxes = [f.getbbox() for f in raw_frames if f.getbbox() is not None]
    max_w = max((b[2] - b[0]) for b in bboxes)
    max_h = max((b[3] - b[1]) for b in bboxes)

    safe_ratio = 0.74 if is_drag else 0.80
    target_box = int(canvas_size * safe_ratio)
    global_scale = min(target_box / max_w, target_box / max_h)
    bottom_margin = int(canvas_size * (0.16 if is_drag else 0.10))

    aligned: List[Image.Image] = []
    for f in raw_frames:
        bbox = f.getbbox()
        if not bbox:
            aligned.append(Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0)))
            continue
        cropped = f.crop(bbox)
        nw = max(1, int(round(cropped.width * global_scale)))
        nh = max(1, int(round(cropped.height * global_scale)))
        resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        paste_x = (canvas_size - nw) // 2
        paste_y = max(int(canvas_size * 0.10), canvas_size - bottom_margin - nh)
        canvas.paste(resized, (paste_x, paste_y), resized)
        aligned.append(canvas)

    if (is_sleep_loop or lock_first_last) and len(aligned) > 1:
        aligned[-1] = aligned[0].copy()

    return aligned


def save_qa_previews(sheet: Image.Image, out_dir: str, action_name: str) -> Tuple[str, str]:
    """Save white-background and black-background preview PNGs for visual inspection."""
    white_bg = Image.new("RGBA", sheet.size, (255, 255, 255, 255))
    white_bg.alpha_composite(sheet)
    white_path = os.path.join(out_dir, f"{action_name}_preview_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", sheet.size, (0, 0, 0, 255))
    black_bg.alpha_composite(sheet)
    black_path = os.path.join(out_dir, f"{action_name}_preview_black.png")
    black_bg.convert("RGB").save(black_path)
    return white_path, black_path


def build_webp_assets(
    action_name: str,
    segment_paths: Sequence[str],
    out_dir: str,
    fast_indices: Optional[Sequence[int]] = None,
    hold_indices: Optional[Sequence[int]] = None,
    base_ms: int = 180,
    fast_ms: int = 135,
    hold_ms: int = 250,
    canvas_size: int = 256,
    is_loop: bool = True,
    lock_first_last: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    frames = slice_and_align_segments(
        segment_paths,
        canvas_size=canvas_size,
        is_drag=("drag" in action_name),
        is_sleep_loop=("sleep_loop" in action_name),
        lock_first_last=lock_first_last,
    )

    n = len(frames)
    cols = 8 if n >= 40 else 6
    rows = math.ceil(n / cols)

    sheet = Image.new("RGBA", (cols * canvas_size, rows * canvas_size), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet.paste(f, (c * canvas_size, r * canvas_size), f)

    sheet_path = os.path.join(out_dir, f"{action_name}_sheet_{cols}x{rows}.webp")
    sheet.save(sheet_path, format="WEBP", lossless=True, quality=100)

    fast_set = set(fast_indices or [])
    hold_set = set(hold_indices or [])
    durations = [
        fast_ms if i in fast_set else (hold_ms if i in hold_set else base_ms)
        for i in range(n)
    ]

    anim_path = os.path.join(out_dir, f"{action_name}_anim.webp")
    frames[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0 if is_loop else 1,
        lossless=True,
        quality=100,
        disposal=2,
    )

    qa_dir = os.path.join(out_dir, "qa_previews")
    os.makedirs(qa_dir, exist_ok=True)
    white_path, black_path = save_qa_previews(sheet, qa_dir, action_name)

    total_sec = sum(durations) / 1000.0
    result = {
        "action": action_name,
        "frames": n,
        "grid": f"{cols}x{rows}",
        "total_seconds": round(total_sec, 2),
        "sheet_path": sheet_path,
        "anim_path": anim_path,
        "preview_white": white_path,
        "preview_black": black_path,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build desktop pet WebP sprite sheets & animations.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract-anchors")
    p_ext.add_argument("--segment", required=True)
    p_ext.add_argument("--out-dir", required=True)
    p_ext.add_argument("--prefix", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--action", required=True)
    p_build.add_argument("--segments", nargs="+", required=True)
    p_build.add_argument("--out-dir", required=True)
    p_build.add_argument("--fast", nargs="*", type=int, default=[])
    p_build.add_argument("--hold", nargs="*", type=int, default=[])
    p_build.add_argument("--base-ms", type=int, default=180)
    p_build.add_argument("--fast-ms", type=int, default=135)
    p_build.add_argument("--hold-ms", type=int, default=250)
    p_build.add_argument("--canvas", type=int, default=256)
    p_build.add_argument("--once", action="store_true")
    p_build.add_argument("--no-lock-ends", action="store_true")

    args = parser.parse_args()
    if args.cmd == "extract-anchors":
        f1, f2 = extract_relay_anchors(args.segment, args.out_dir, args.prefix)
        print(json.dumps({"first": f1, "last": f2}, ensure_ascii=False))
    elif args.cmd == "build":
        build_webp_assets(
            action_name=args.action,
            segment_paths=args.segments,
            out_dir=args.out_dir,
            fast_indices=args.fast,
            hold_indices=args.hold,
            base_ms=args.base_ms,
            fast_ms=args.fast_ms,
            hold_ms=args.hold_ms,
            canvas_size=args.canvas,
            is_loop=not args.once,
            lock_first_last=not args.no_lock_ends,
        )
