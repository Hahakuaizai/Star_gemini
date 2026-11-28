import importlib.util
import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask

_spec = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned
interpolate_smooth_no_burr = _swp.interpolate_smooth_no_burr

CANVAS = 384
TARGET_H = 304
GROUND_Y = 366


def measure_head_metrics(img: Image.Image, body_ratio: float = 1.0) -> tuple[float, float, int, int]:
    """
    Measure head width and head center X from the upper head band (safe even when lower legs are cropped).
    `body_ratio` estimates how much of the full body height is visible in this cell (1.0=full, ~0.72=skirt-crop).
    """
    arr = np.array(img, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    ch = y1 - y0 + 1
    # Full-body head band is ~18%..42% of full height -> scaled by 1/body_ratio for cropped cells
    lo_f = min(0.55, 0.18 / body_ratio)
    hi_f = min(0.75, 0.42 / body_ratio)
    band = mask[y0 + int(lo_f * ch): y0 + int(hi_f * ch), :]
    widths = []
    for r in range(band.shape[0]):
        rxs = np.nonzero(band[r])[0]
        if len(rxs) > 0:
            widths.append(int(rxs.max() - rxs.min() + 1))
    hw = float(np.median(widths)) if widths else float(xs.max() - xs.min() + 1)
    top_cut = y0 + max(4, int(ch * min(0.45, 0.28 / body_ratio)))
    head_cx = float(np.median(xs[ys <= top_cut]))
    return hw, head_cx, y0, y1


def align_full_body(f: Image.Image) -> Image.Image:
    """Align full-body frame strictly to TARGET_H (304px) grounded at GROUND_Y (366px)."""
    arr = np.array(f, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys = np.nonzero(mask)[0]
    body_h = max(1, int(ys.max() - ys.min() + 1))
    scale = TARGET_H / body_h

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    _, head_cx, my0, my1 = measure_head_metrics(resized, body_ratio=1.0)
    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(GROUND_Y - my1))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def align_cropped_with_ref_boots(f: Image.Image, ref_stand: Image.Image, body_ratio: float) -> Image.Image:
    """
    Align a lower-cropped frame to match `ref_stand` head width 1:1, and seamlessly composite
    `ref_stand` across the FULL WIDTH below `seam_y` (preserving both complete boots AND full cat tail!).
    """
    ref_arr = np.array(ref_stand, dtype=np.uint8)
    ref_mask = get_main_component_mask(ref_arr[:, :, 3])
    rys = np.nonzero(ref_mask)[0]
    ry0, ry1 = int(rys.min()), int(rys.max())
    ref_hw, _, _, _ = measure_head_metrics(ref_stand, body_ratio=1.0)

    hw, _, _, _ = measure_head_metrics(f, body_ratio=body_ratio)
    scale = ref_hw / max(1.0, hw)

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    _, head_cx, my0, my1 = measure_head_metrics(resized, body_ratio=body_ratio)
    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(ry0 - my0))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)

    # Full-width feather composite below seam_y (safe above cropped bottom so boots & lower cat tail stay 100% intact)
    c_arr = np.array(canvas, dtype=np.float32)
    r_flt = ref_arr.astype(np.float32)
    cropped_bot_in_canvas = paste_y + my1
    seam_y = min(int(ry0 + (ry1 - ry0 + 1) * 0.84), cropped_bot_in_canvas - 10)
    feather = 8

    out = c_arr.copy()
    for y in range(seam_y, CANVAS):
        t = min(1.0, (y - seam_y) / max(1.0, float(feather)))
        out[y, :, :] = (1.0 - t) * c_arr[y, :, :] + t * r_flt[y, :, :]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def main() -> dict:
    s1 = Image.open("gemini/素材/18note_taking/part1.jpg").convert("RGBA")
    s1_fix = Image.open("gemini/素材/18note_taking/part1_glasses_fix.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/18note_taking/part2.jpg").convert("RGBA")

    b1 = find_cell_boxes(s1, 4, 3)
    b1_fix = find_cell_boxes(s1_fix, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    # Part 1 Row 3 (indices 8..11) has a faint grey floor gradient -> use higher bg_thresh=58 for those 4 cells.
    # Replace missing-glasses keyframes p1[3..7] from part1_glasses_fix.jpg while keeping all other cells untouched.
    p1 = [
        clean_black_bg_rich_particles(
            (s1_fix.crop(b1_fix[i]) if 3 <= i <= 7 else s1.crop(b1[i])),
            bg_thresh=(58 if i >= 8 else 24),
            hole_thresh=(42 if i >= 8 else 18),
            glow_cutoff=(50 if i >= 8 else 22),
        )
        for i in range(12)
    ]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    ref_stand = remove_nebula_blobs_aligned(align_full_body(p1[0]))

    # Align Part 1:
    # - p1[0] is full-body clean reference
    # - p1[8..11] are full-body with a thin bottom grey floor line -> replace bottom 14px cleanly via full-width ref_stand blend
    # - p1[1..7] are lower-cropped (~0.85 body visible) -> head-locked + full-width boot/tail composite
    a1: List[Image.Image] = []
    for i in range(12):
        if i == 0:
            a1.append(ref_stand)
        elif i >= 8:
            full = align_full_body(p1[i])
            f_arr = np.array(full, dtype=np.float32)
            r_arr = np.array(ref_stand, dtype=np.float32)
            for y in range(GROUND_Y - 14, CANVAS):
                t = min(1.0, (y - (GROUND_Y - 14)) / 6.0)
                f_arr[y, :, :] = (1.0 - t) * f_arr[y, :, :] + t * r_arr[y, :, :]
            a1.append(remove_nebula_blobs_aligned(Image.fromarray(np.clip(f_arr, 0, 255).astype(np.uint8), "RGBA")))
        else:
            a1.append(remove_nebula_blobs_aligned(align_cropped_with_ref_boots(p1[i], ref_stand, body_ratio=0.85)))

    # Align Part 2[:11] (Row 1 ~0.72 visible, Row 2 ~0.82 visible, Row 3 ~0.92 visible)
    a2: List[Image.Image] = []
    for i in range(11):
        ratio = 0.72 if i < 4 else (0.82 if i < 8 else 0.92)
        a2.append(remove_nebula_blobs_aligned(align_cropped_with_ref_boots(p2[i], ref_stand, body_ratio=ratio)))

    keys_24 = a1 + a2 + [ref_stand.copy()]
    assert len(keys_24) == 24

    # Expand 24 keyframes -> 48 frames (8x6) with anti-aliased optical flow
    frames_48: List[Image.Image] = []
    for i in range(24):
        frames_48.append(keys_24[i])
        if i < 23:
            frames_48.append(interpolate_smooth_no_burr(keys_24[i], keys_24[i + 1]))
    frames_48.append(ref_stand.copy())
    assert len(frames_48) == 48

    heights = []
    for fr in frames_48:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "18note_taking_sheet（记笔记_图集）.webp")
    anim_path = os.path.join(out_dir, "18note_taking（记笔记_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # Doc target ~8.5s: writing strokes 145ms, holds (think/twirl/nod/close/take-off/blink/ends) 225ms, normal 175ms -> ~8.60s
    fast_set = set(range(6, 16)) | {20, 21}
    hold_set = {0, 1, 16, 17, 18, 19, 22, 23, 24, 25, 32, 33, 42, 43, 46, 47}
    durations = [145 if i in fast_set else (225 if i in hold_set else 175) for i in range(48)]

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
    white_path = os.path.join(prev_dir, "18note_taking_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "18note_taking_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "18note_taking",
        "frames": 48,
        "grid": "8x6",
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
