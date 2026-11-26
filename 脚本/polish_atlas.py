#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polish a transparent 8x6 sprite atlas into a game-ready asset.

Deterministic steps, no generative model involved:
  1. split the atlas into cells
  2. recolour opaque near-black blobs (generator artefacts inside the tail)
  3. move a tail that ended up on the VIEWER'S LEFT over to the right side
  4. lock body height to the atlas median
  5. pin the sole row and the boot axis to fixed pixel rows/columns
  6. re-pack the atlas and write the animated WebP with variable timing
"""
import argparse
import json
import math
import os
import sys
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- mask utils
def largest_component(mask: np.ndarray) -> np.ndarray:
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


def body_metrics(frame: Image.Image, thr: int = 40) -> Optional[dict]:
    alpha = np.array(frame.getchannel("A"))
    body = largest_component(alpha > thr)
    if not body.any():
        return None
    ys = np.where(body.any(axis=1))[0]
    top, bot = int(ys[0]), int(ys[-1])
    h = bot - top + 1
    t0, t1 = top + int(h * 0.40), top + int(h * 0.56)
    txs = np.where(body[t0:t1].any(axis=0))[0]
    axis = float((txs[0] + txs[-1]) / 2.0) if len(txs) else float(frame.width / 2)
    f0 = max(0, bot - int(h * 0.06))
    cols = body[f0:bot + 1].sum(axis=0).astype(np.float64)
    foot_x = float((cols * np.arange(len(cols))).sum() / cols.sum()) if cols.sum() else axis
    return dict(body=body, top=top, bot=bot, h=h, axis=axis, foot_x=foot_x)


# ------------------------------------------------------- black blob repair
def repair_dark_blobs(frame: Image.Image, dark_max: int = 45,
                      opaque_min: int = 200, ring: int = 16) -> Tuple[Image.Image, int]:
    """Recolour opaque near-black pixels from the surrounding ring.

    The generator sometimes leaves a solid black patch inside the fluffy tail.
    It is a hole in the artwork, not a shadow: recolour it so the silhouette
    survives instead of punching a hole through the tail.
    """
    arr = np.array(frame.convert("RGBA"))
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    dark = (alpha >= opaque_min) & (rgb.max(axis=2) < dark_max)
    if not dark.any():
        return frame, 0

    ys, xs = np.where(dark)
    fixed = 0
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - ring), min(arr.shape[0], y + ring + 1)
        x0, x1 = max(0, x - ring), min(arr.shape[1], x + ring + 1)
        win_rgb = arr[y0:y1, x0:x1, :3].astype(np.int16)
        win_a = arr[y0:y1, x0:x1, 3]
        good = (win_a >= opaque_min) & (win_rgb.max(axis=2) >= dark_max)
        if good.sum() >= 8:
            arr[y, x, :3] = win_rgb[good].mean(axis=0).astype(np.uint8)
            fixed += 1
    return Image.fromarray(arr, "RGBA"), fixed


# ------------------------------------------------------------- tail side
def find_tail(frame: Image.Image, meta: dict, clearance: float = 42.0) -> Optional[np.ndarray]:
    """Grow the tail region from the outer tip on the viewer's left.

    The tail is fused to the torso, so connected components cannot separate
    them. Instead: seed at the leftmost alpha pixel of the lower body, grow
    within the region left of `axis - clearance` (which excludes the legs,
    they sit near the axis), and keep what the flood reaches.
    """
    body, top, bot, h, axis = meta["body"], meta["top"], meta["bot"], meta["h"], meta["axis"]
    y_floor = top + int(h * 0.50)
    limit_x = axis - clearance

    seed_mask = body & (np.arange(body.shape[1])[None, :] < limit_x)
    seed_mask[:y_floor] = False
    if seed_mask.sum() < 80:
        return None
    seed = largest_component(seed_mask)
    if seed.sum() < 80:
        return None
    return seed


def shift_tail_right(frame: Image.Image, meta: dict,
                     target_reach: float = 100.0) -> Tuple[Image.Image, bool]:
    """Mirror the left-hand tail lobe over to the viewer's right."""
    axis, h, top = meta["axis"], meta["h"], meta["top"]
    body = meta["body"]
    lower = body[top + int(h * 0.55):top + h]
    lxs = np.where(lower.any(axis=0))[0]
    if not len(lxs):
        return frame, False
    reach_l = axis - float(lxs[0])
    reach_r = float(lxs[-1]) - axis
    if reach_l <= reach_r + 8:            # tail already on the right
        return frame, False

    lobe = find_tail(frame, meta)
    if lobe is None:
        return frame, False

    arr = np.array(frame.convert("RGBA"))
    lobe_px = np.zeros_like(arr)
    lobe_px[lobe] = arr[lobe]
    kept = arr.copy()
    kept[lobe] = 0

    lobe_img = Image.fromarray(lobe_px, "RGBA").transpose(Image.FLIP_LEFT_RIGHT)
    ys, xs = np.where(lobe.any(axis=0)[None, :] & (lobe.any(axis=0)[None, :]))
    lx = np.where(lobe.any(axis=0))[0]
    lobe_w = int(lx[-1] - lx[0] + 1)

    # put the mirrored lobe so its outer edge lands at axis + target_reach
    dest_right = axis + target_reach
    shift = int(round(dest_right - lobe_w))

    out = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    base = Image.fromarray(kept, "RGBA")
    out.paste(base, (0, 0), base)
    out.paste(lobe_img, (shift, 0), lobe_img)

    # heal the vacated area: smear nearby tail/hair colour inward
    out = heal_hole(out, lobe, arr)
    return out, True


def heal_hole(frame: Image.Image, hole: np.ndarray, original: np.ndarray,
              ring: int = 10) -> Image.Image:
    """Fill the emptied lobe area with colours from just outside it."""
    arr = np.array(frame.convert("RGBA"))
    ys, xs = np.where(hole)
    if not len(ys):
        return frame
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - ring), min(arr.shape[0], y + ring + 1)
        x0, x1 = max(0, x - ring), min(arr.shape[1], x + ring + 1)
        win_a = arr[y0:y1, x0:x1, 3]
        win_rgb = arr[y0:y1, x0:x1, :3].astype(np.int16)
        good = win_a > 120
        if good.sum() >= 6:
            arr[y, x, :3] = win_rgb[good].mean(axis=0).astype(np.uint8)
            arr[y, x, 3] = 255
    return Image.fromarray(arr, "RGBA")


# -------------------------------------------------------------- normalise
def normalise(frames: List[Image.Image], canvas: int,
              bottom_ratio: float = 0.10) -> List[Image.Image]:
    metas = [body_metrics(f) for f in frames]
    heights = [m["h"] for m in metas if m]
    target_h = float(np.median(heights))
    max_h = int(canvas * (1.0 - bottom_ratio - 0.11))
    target_h = min(target_h, float(max_h))
    bottom_margin = int(canvas * bottom_ratio)

    out = []
    for f, m in zip(frames, metas):
        if m is None:
            out.append(Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0)))
            continue
        scale = target_h / m["h"]
        fw, fh = f.size
        scale = min(scale, canvas / fw, canvas / fh)
        nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        scaled = f.resize((nw, nh), Image.Resampling.LANCZOS)

        px = int(round(canvas / 2.0 - m["foot_x"] * scale))
        py = int(round(canvas - bottom_margin - m["bot"] * scale))
        c = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        c.paste(scaled, (px, py), scaled)

        # LANCZOS blurs the alpha edge: re-measure and nail the sole to a row
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
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--fast", nargs="*", type=int, default=[])
    ap.add_argument("--hold", nargs="*", type=int, default=[])
    ap.add_argument("--base-ms", type=int, default=175)
    ap.add_argument("--fast-ms", type=int, default=135)
    ap.add_argument("--hold-ms", type=int, default=250)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-tail-fix", action="store_true")
    ap.add_argument("--no-black-fix", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sheet = Image.open(args.atlas).convert("RGBA")
    W, H = sheet.size
    cw, ch = W // args.cols, H // args.rows
    cells = [sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
             for r in range(args.rows) for c in range(args.cols)]
    print(f"[split] {len(cells)} cells of {cw}x{ch}", file=sys.stderr)

    dark_total, flipped = 0, []
    fixed_cells = []
    for i, cell in enumerate(cells):
        if not args.no_black_fix:
            cell, n = repair_dark_blobs(cell)
            dark_total += n
        if not args.no_tail_fix:
            m = body_metrics(cell)
            if m is not None:
                cell, did = shift_tail_right(cell, m)
                if did:
                    flipped.append(i + 1)
        fixed_cells.append(cell)
    print(f"[clean] dark px recoloured: {dark_total}", file=sys.stderr)
    print(f"[tail] moved to the right on frames: {flipped or 'none'}", file=sys.stderr)

    frames = normalise(fixed_cells, args.canvas)
    n = len(frames)
    frames[-1] = frames[0].copy()

    cols = 8 if n >= 40 else 6
    rows = math.ceil(n / cols)
    out_sheet = Image.new("RGBA", (cols * args.canvas, rows * args.canvas), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        out_sheet.paste(f, (c * args.canvas, r * args.canvas), f)
    sheet_path = os.path.join(args.out_dir, f"{args.action}_sheet_{cols}x{rows}.webp")
    out_sheet.save(sheet_path, format="WEBP", lossless=True, quality=100)

    fast, hold = set(args.fast), set(args.hold)
    durations = [args.fast_ms if i in fast else (args.hold_ms if i in hold else args.base_ms)
                 for i in range(n)]
    anim_path = os.path.join(args.out_dir, f"{args.action}_anim.webp")
    frames[0].save(anim_path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=durations, loop=1 if args.once else 0,
                   lossless=True, quality=100, disposal=2)

    qa = os.path.join(args.out_dir, "qa_previews")
    os.makedirs(qa, exist_ok=True)
    for name, bg in (("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255))):
        canvas_img = Image.new("RGBA", out_sheet.size, bg)
        canvas_img.alpha_composite(out_sheet)
        canvas_img.convert("RGB").save(os.path.join(qa, f"{args.action}_preview_{name}.png"))

    print(json.dumps({
        "action": args.action, "frames": n, "grid": f"{cols}x{rows}",
        "total_seconds": round(sum(durations) / 1000.0, 2),
        "sheet": sheet_path, "anim": anim_path,
        "tail_moved_frames": flipped, "dark_px_recoloured": dark_total,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
