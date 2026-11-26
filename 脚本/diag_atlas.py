#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose an 8x6 transparent atlas before polishing it.

Reports, per cell: body bbox, foot row, body axis, and which side the tail
sticks out on (the spec requires the tail on the VIEWER'S RIGHT). Uses the
torso band as the axis so a whole-body horizontal shift does not fake a flip.
"""
import sys
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def largest_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
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
    return best, best_area


def analyse(cell: Image.Image, thr: int = 40) -> Optional[dict]:
    a = np.array(cell.getchannel("A"))
    body, area = largest_component(a > thr)
    if area == 0:
        return None
    ys = np.where(body.any(axis=1))[0]
    top, bot = int(ys[0]), int(ys[-1])
    h = bot - top + 1

    # Torso band (40%-56% of the body height) gives the body axis.
    t0, t1 = top + int(h * 0.40), top + int(h * 0.56)
    torso = body[t0:t1]
    txs = np.where(torso.any(axis=0))[0]
    axis = float((txs[0] + txs[-1]) / 2.0) if len(txs) else float("nan")

    # Lower body (58%-100%): how far does each side reach past the axis?
    l0 = top + int(h * 0.58)
    lower = body[l0:bot + 1]
    lxs = np.where(lower.any(axis=0))[0]
    reach_l = axis - float(lxs[0]) if len(lxs) else 0.0
    reach_r = float(lxs[-1]) - axis if len(lxs) else 0.0

    # Feet: alpha centroid of the bottom 6% band
    f0 = max(0, bot - int(h * 0.06))
    feet = body[f0:bot + 1]
    cols = feet.sum(axis=0).astype(np.float64)
    foot_x = float((cols * np.arange(len(cols))).sum() / cols.sum()) if cols.sum() else axis

    return dict(top=top, bot=bot, h=h, axis=axis, foot_x=foot_x,
                reach_l=reach_l, reach_r=reach_r, area=area,
                tail_side="right" if reach_r > reach_l else "LEFT")


def main() -> int:
    src = sys.argv[1]
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    im = Image.open(src).convert("RGBA")
    W, H = im.size
    cw, ch = W // cols, H // rows
    print(f"{src}  {W}x{H}  cell {cw}x{ch}  ({cols}x{rows} = {cols*rows} frames)")
    bad_tail, bad_foot = [], []
    rows_out: List[dict] = []
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        m = analyse(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
        if m is None:
            print(f"{i+1:2d} EMPTY")
            continue
        rows_out.append(m)
        flag = ""
        if m["tail_side"] != "right":
            flag += " TAIL-LEFT"
            bad_tail.append(i + 1)
        print(f"{i+1:2d} h={m['h']:3d} top={m['top']:3d} bot={m['bot']:3d} "
              f"axis={m['axis']:6.1f} foot={m['foot_x']:6.1f} "
              f"reachL={m['reach_l']:5.1f} reachR={m['reach_r']:5.1f} "
              f"tail={m['tail_side']:5s}{flag}")

    hs = [m["h"] for m in rows_out]
    fx = [m["foot_x"] for m in rows_out]
    ax = [m["axis"] for m in rows_out]
    print(f"\nheight   min={min(hs)} max={max(hs)} spread={max(hs)-min(hs)}")
    print(f"foot_x   min={min(fx):.1f} max={max(fx):.1f} spread={max(fx)-min(fx):.1f}")
    print(f"axis     min={min(ax):.1f} max={max(ax):.1f} spread={max(ax)-min(ax):.1f}")
    print(f"tail on LEFT: {bad_tail or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
