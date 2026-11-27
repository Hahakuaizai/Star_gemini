import json
import os
from typing import List

import numpy as np
from PIL import Image

from build_webp import clean_black_bg_and_glow, find_cell_boxes
from expand_48 import get_main_component_mask, measure_body_stats, pyramidal_optical_flow, warp_rgba

CANVAS = 384
TARGET_H = 308
GROUND_Y = 366


def extract_rows(path: str, keep_rows: int = 3) -> List[Image.Image]:
    sheet = Image.open(path).convert("RGBA")
    arr = np.array(sheet.convert("RGB"), dtype=np.uint8)
    bg = int(np.percentile(arr[:12, :, :].max(axis=2), 95))
    t = max(22, bg + 14)

    if keep_rows < 3:
        h, w = arr.shape[:2]
        cut_h = int(round(h * keep_rows / 3.0))
        left_edge = arr[:, :15, :].max(axis=2).mean(axis=1)
        right_edge = arr[:, -15:, :].max(axis=2).mean(axis=1)
        row_bg_edge = np.maximum(left_edge, right_edge)
        bright_rows = np.where(row_bg_edge > bg + 16)[0]
        if len(bright_rows) > 0 and bright_rows[0] > int(h * 0.55):
            cut_h = int(bright_rows[0]) - 12
        src = sheet.crop((0, 0, w, cut_h))
        boxes = find_cell_boxes(src, cols=4, rows=keep_rows, thresh=t)
    else:
        src = sheet
        boxes = find_cell_boxes(sheet, cols=4, rows=3, thresh=t)

    return [
        clean_black_bg_and_glow(
            src.crop(b),
            bg_thresh=max(24, bg + 12),
            hole_thresh=max(20, bg + 8),
            glow_cutoff=max(34, bg + 16),
        )
        for b in boxes
    ]


def align_full_body(f: Image.Image) -> Image.Image:
    """Strictly lock full-body standing frame to TARGET_H (308px) grounded at GROUND_Y (366px)."""
    bx0, by0, bx1, by1, _ = measure_body_stats(f)
    body_h = max(1, by1 - by0)
    scale = TARGET_H / body_h

    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    r_arr = np.array(resized, dtype=np.uint8)
    mask = get_main_component_mask(r_arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    my0, my1 = int(ys.min()), int(ys.max())
    top_cut = my0 + max(4, int((my1 - my0 + 1) * 0.30))
    head_cx = float(np.median(xs[ys <= top_cut]))

    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(GROUND_Y - my1))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def align_part2_with_ref_boots(f: Image.Image, ref_stand: Image.Image) -> Image.Image:
    """
    Align Part 2 upper-body frame (cropped at lower shins) to match `ref_stand` head-to-thigh scale,
    and seamlessly composite `ref_stand`'s lower shins & full dark purple boots below the shin seam.
    """
    ref_arr = np.array(ref_stand, dtype=np.uint8)
    ref_mask = get_main_component_mask(ref_arr[:, :, 3])
    rys, rxs = np.nonzero(ref_mask)
    ry0, ry1 = int(rys.min()), int(rys.max())
    ref_h = ry1 - ry0 + 1
    r_band = ref_mask[ry0 + int(0.18 * ref_h): ry0 + int(0.42 * ref_h), :]
    ref_hw = float(np.median([int(np.nonzero(r_band[r])[0].max() - np.nonzero(r_band[r])[0].min() + 1)
                              for r in range(r_band.shape[0]) if r_band[r].any()]))

    arr = np.array(f, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    ch = y1 - y0 + 1
    band = mask[y0 + int(0.21 * ch): y0 + int(0.49 * ch), :]
    p2_hw = float(np.median([int(np.nonzero(band[r])[0].max() - np.nonzero(band[r])[0].min() + 1)
                             for r in range(band.shape[0]) if band[r].any()]))

    scale = ref_hw / max(1.0, p2_hw)
    nw = max(1, int(round(f.width * scale)))
    nh = max(1, int(round(f.height * scale)))
    resized = f.resize((nw, nh), Image.Resampling.LANCZOS)

    r2_arr = np.array(resized, dtype=np.uint8)
    m2 = get_main_component_mask(r2_arr[:, :, 3])
    ys2, xs2 = np.nonzero(m2)
    my0, my1 = int(ys2.min()), int(ys2.max())
    top_cut = my0 + max(4, int((my1 - my0 + 1) * 0.34))
    head_cx = float(np.median(xs2[ys2 <= top_cut]))

    paste_x = int(round(CANVAS * 0.5 - head_cx))
    paste_y = int(round(ry0 - my0))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)

    c_arr = np.array(canvas, dtype=np.float32)
    r_flt = ref_arr.astype(np.float32)
    seam_y = min(int(ry0 + ref_h * 0.84), paste_y + my1 - 6)
    feather = 6

    out = c_arr.copy()
    for y in range(seam_y, CANVAS):
        t = min(1.0, (y - seam_y) / max(1.0, float(feather)))
        out[y, :, :] = (1.0 - t) * c_arr[y, :, :] + t * r_flt[y, :, :]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def interpolate_between(img_a: Image.Image, img_b: Image.Image, ts: List[float]) -> List[Image.Image]:
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa, fb = feature_map(a), feature_map(b)
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
    p1 = extract_rows("gemini/素材/15tickle_laugh/part1.jpg", keep_rows=3)
    p2 = extract_rows("gemini/素材/15tickle_laugh/part2.jpg", keep_rows=2)

    aligned_p1 = [align_full_body(f) for f in p1]
    ref_stand = aligned_p1[0]

    p2_selected = [align_part2_with_ref_boots(p2[i], ref_stand) for i in (0, 2, 4, 5, 6, 7)]

    keys_19 = aligned_p1 + p2_selected + [ref_stand.copy()]
    assert len(keys_19) == 19

    double_mid_pairs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 14}
    frames_48: List[Image.Image] = []
    for i in range(18):
        frames_48.append(keys_19[i])
        if i in double_mid_pairs:
            frames_48.extend(interpolate_between(keys_19[i], keys_19[i + 1], [1 / 3, 2 / 3]))
        else:
            frames_48.extend(interpolate_between(keys_19[i], keys_19[i + 1], [0.5]))
    frames_48.append(ref_stand.copy())
    assert len(frames_48) == 48, len(frames_48)

    heights = []
    for fr in frames_48:
        a = np.array(fr)[:, :, 3]
        ys = np.nonzero(a.any(axis=1))[0]
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

    sheet_path = os.path.join(out_dir, "15tickle_laugh_sheet（挠痒大笑_图集）.webp")
    anim_path = os.path.join(out_dir, "15tickle_laugh（挠痒大笑_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    fast_set = set(range(3, 28))
    hold_set = {0, 1, 34, 35, 39, 40, 41, 46, 47}
    durations = [120 if i in fast_set else (215 if i in hold_set else 155) for i in range(48)]

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
    white_path = os.path.join(prev_dir, "15tickle_laugh_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "15tickle_laugh_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "15tickle_laugh",
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
