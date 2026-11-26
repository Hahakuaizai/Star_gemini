#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build sprite assets with per-frame BODY-height normalisation.

The stock pipeline scales the whole action by one global factor taken from the
union bbox, which is fine for big gesture actions. For subtle loops (idle) the
generator sometimes returns a segment drawn ~5% smaller, and that reads as a
visible size pop. This variant measures the *character body* (largest alpha
connected component, so floating sparkles are ignored), scales every frame so
the body height matches the median, and then anchors the feet and body centre.
"""
import argparse
import json
import math
import os
import sys
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_webp import clean_black_bg_and_glow, find_cell_boxes, save_qa_previews  # noqa: E402


def largest_component_bbox(frame: Image.Image) -> Tuple[int, int, int, int]:
    """Bbox of the largest alpha connected component (the character body)."""
    alpha = np.array(frame.getchannel("A"), dtype=np.uint8)
    mask = alpha > 40
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best = (0, 0, 0, 0)
    best_area = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            area = 0
            x0 = x1 = sx
            y0 = y1 = sy
            while q:
                cy, cx = q.popleft()
                area += 1
                if cx < x0:
                    x0 = cx
                elif cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                elif cy > y1:
                    y1 = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if area > best_area:
                best_area = area
                best = (x0, y0, x1 + 1, y1 + 1)
    return best


def foot_center_x(frame: Image.Image, body: Tuple[int, int, int, int]) -> float:
    """Horizontal centre of the boots: the alpha centroid of the lowest band
    of the body. Far steadier than the whole-body bbox centre, which the tail
    and floating particles drag around."""
    alpha = np.array(frame.getchannel("A"), dtype=np.uint8)
    band_top = int(body[3] - (body[3] - body[1]) * 0.06)
    band = alpha[band_top:body[3], body[0]:body[2]] > 40
    cols = band.sum(axis=0).astype(np.float64)
    if cols.sum() <= 0:
        return (body[0] + body[2]) / 2.0
    xs = np.arange(len(cols), dtype=np.float64) + body[0]
    return float((cols * xs).sum() / cols.sum())


def slice_frames(segment_paths: Sequence[str]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for path in segment_paths:
        sheet = Image.open(path).convert("RGBA")
        for box in find_cell_boxes(sheet, cols=4, rows=3):
            frames.append(clean_black_bg_and_glow(sheet.crop(box)))
    return frames


def body_metrics(frame: Image.Image, thr: int = 40) -> Optional[dict]:
    """Body bbox, height, torso axis and boot centre of one frame."""
    alpha = np.array(frame.getchannel("A"))
    body = _largest_component(alpha > thr)
    if not body.any():
        return None
    ys = np.where(body.any(axis=1))[0]
    xs = np.where(body.any(axis=0))[0]
    top, bot = int(ys[0]), int(ys[-1])
    h = bot - top + 1

    t0, t1 = top + int(h * 0.40), top + int(h * 0.56)
    txs = np.where(body[t0:t1].any(axis=0))[0]
    axis = float((txs[0] + txs[-1]) / 2.0) if len(txs) else float(frame.width / 2)

    f0 = max(0, bot - int(h * 0.06))
    cols = body[f0:bot + 1].sum(axis=0).astype(np.float64)
    foot_x = float((cols * np.arange(len(cols))).sum() / cols.sum()) if cols.sum() else axis
    return dict(body=body, top=top, bot=bot, h=h, axis=axis, foot_x=foot_x)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best = np.zeros((h, w), dtype=bool)
    best_area = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            pts = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(pts) > best_area:
                best_area = len(pts)
                best = np.zeros((h, w), dtype=bool)
                for y, x in pts:
                    best[y, x] = True
    return best


def normalise(frames: List[Image.Image], canvas: int,
              bottom_ratio: float = 0.10) -> List[Image.Image]:
    """Uniform, resolution-independent normalisation.

    Segments can arrive at different cell sizes (320 vs 384 px). Rather than
    scale by cell size, measure the BODY of every frame and give each frame its
    own factor so every body lands on the same target height. Then pin the sole
    to a fixed row and the boots to the centre column. Because the factor comes
    from the body height and not from the cell size, the result is identical
    whether the source cell was 320 or 384 px.
    """
    metas = [body_metrics(f) for f in frames]
    heights = [m["h"] for m in metas if m]
    target_h = float(np.median(heights))

    side_margin = int(canvas * 0.125)
    bottom_margin = int(canvas * bottom_ratio)
    usable_h = canvas - bottom_margin - side_margin
    usable_w = canvas - 2 * side_margin

    # widest ink across the whole set, in BODY-relative units
    max_ink_w = 0.0
    for f, m in zip(frames, metas):
        if m is None:
            continue
        a = np.array(f.getchannel("A"))
        ink = a > 40
        if ink.any():
            xs = np.where(ink.any(axis=0))[0]
            # width relative to this frame's body height, so it is comparable
            rel = (xs[-1] - xs[0] + 1) / max(1.0, float(m["h"]))
            max_ink_w = max(max_ink_w, rel * target_h)

    out: List[Image.Image] = []
    for f, m in zip(frames, metas):
        if m is None:
            out.append(Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0)))
            continue

        scale = target_h / max(1.0, float(m["h"]))
        # Safety: the widest frame in the set (laptop, tail) must also fit the
        # 10% side margins, otherwise that frame alone blows the border.
        if max_ink_w > 0:
            scale = min(scale, usable_w / max_ink_w)
        nw = max(1, int(round(f.width * scale)))
        nh = max(1, int(round(f.height * scale)))
        scaled = f.resize((nw, nh), Image.Resampling.LANCZOS)

        # after scaling, verify the ink fits; shrink further only if needed
        a = np.array(scaled.getchannel("A"))
        ink = a > 40
        if ink.any():
            ys, xs = np.where(ink)
            h_ink = int(ys[-1] - ys[0] + 1)
            w_ink = int(xs[-1] - xs[0] + 1)
            shrink = min(1.0, usable_h / max(1.0, h_ink), usable_w / max(1.0, w_ink))
            if shrink < 1.0:
                scale *= shrink
                nw = max(1, int(round(f.width * scale)))
                nh = max(1, int(round(f.height * scale)))
                scaled = f.resize((nw, nh), Image.Resampling.LANCZOS)

        paste_x = int(round(canvas / 2.0 - m["foot_x"] * scale))
        paste_y = int(round(canvas - bottom_margin - m["bot"] * scale))

        c = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        c.paste(scaled, (paste_x, paste_y), scaled)

        # LANCZOS blurs the alpha edge: nail the sole to an exact row
        a = np.array(c.getchannel("A"))
        rows = np.where((a > 40).any(axis=1))[0]
        if len(rows):
            drift = int(rows[-1]) + 1 - (canvas - bottom_margin)
            if drift:
                sh = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
                sh.paste(c, (0, -drift), c)
                c = sh
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True)
    ap.add_argument("--segments", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--fast", nargs="*", type=int, default=[])
    ap.add_argument("--hold", nargs="*", type=int, default=[])
    ap.add_argument("--base-ms", type=int, default=175)
    ap.add_argument("--fast-ms", type=int, default=110)
    ap.add_argument("--hold-ms", type=int, default=220)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-lock-ends", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames = normalise(slice_frames(args.segments), canvas=args.canvas)
    n = len(frames)
    if not args.no_lock_ends:
        frames[-1] = frames[0].copy()

    cols = 8 if n >= 40 else 6
    rows = math.ceil(n / cols)
    sheet = Image.new("RGBA", (cols * args.canvas, rows * args.canvas), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet.paste(f, (c * args.canvas, r * args.canvas), f)

    sheet_path = os.path.join(args.out_dir, f"{args.action}_sheet_{cols}x{rows}.webp")
    sheet.save(sheet_path, format="WEBP", lossless=True, quality=100)

    fast, hold = set(args.fast), set(args.hold)
    durations = [args.fast_ms if i in fast else (args.hold_ms if i in hold else args.base_ms)
                 for i in range(n)]
    anim_path = os.path.join(args.out_dir, f"{args.action}_anim.webp")
    frames[0].save(anim_path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=durations, loop=1 if args.once else 0,
                   lossless=True, quality=100, disposal=2)

    qa_dir = os.path.join(args.out_dir, "qa_previews")
    os.makedirs(qa_dir, exist_ok=True)
    white, black = save_qa_previews(sheet, qa_dir, args.action)

    print(json.dumps({
        "action": args.action, "frames": n, "grid": f"{cols}x{rows}",
        "total_seconds": round(sum(durations) / 1000.0, 2),
        "sheet_path": sheet_path, "anim_path": anim_path,
        "preview_white": white, "preview_black": black,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
