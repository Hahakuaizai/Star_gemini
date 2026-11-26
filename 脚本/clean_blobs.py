#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair solid black patches painted inside the artwork (e.g. inside a tail).

The image model sometimes renders a dark blob as if it were a shadow on the
tail. After matting it becomes an opaque near-black island that reads as a hole
in the sprite. Deleting it punches a gap through the tail, so instead this
fills it from the surrounding artwork:

  * find opaque near-black pixels that have NO bright artwork within a small
    radius (an outline pixel always touches bright colour, so outlines survive)
  * grow the fill inward from the blob border using the mean of the already
    known neighbours, so fur texture flows through instead of a flat patch
"""
import argparse
import os
import sys
from collections import deque
from typing import List

import numpy as np
from PIL import Image


def find_blobs(mask: np.ndarray, min_area: int = 12) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
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
            if len(pts) >= min_area:
                m = np.zeros((h, w), dtype=bool)
                for y, x in pts:
                    m[y, x] = True
                out.append(m)
    return out


def fill_blob(arr: np.ndarray, blob: np.ndarray) -> int:
    """Grow colours from the blob's outer border inward."""
    h, w = blob.shape
    rgb = arr[:, :, :3].astype(np.float32)
    # 只用"亮"像素做已知源：暗像素本身就是星云，拿它当参考只会把黑块
    # 一路染开。亮像素才是尾巴/头发/皮肤真正的颜色。
    lum_all = arr[:, :, :3].astype(np.int16).max(axis=2)
    known = (~blob) & (arr[:, :, 3] > 120) & (lum_all >= 140)
    todo = blob.copy()
    filled = 0
    while todo.any():
        grow = np.zeros_like(todo)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
            grow |= todo & src
        if not grow.any():
            break
        ys, xs = np.where(grow)
        for y, x in zip(ys, xs):
            acc = np.zeros(3, dtype=np.float32)
            n = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and known[ny, nx]:
                    acc += rgb[ny, nx]
                    n += 1
            if n:
                base = acc / n
                rgb[y, x] = base
                arr[y, x, 3] = 255
                filled += 1
        known |= grow
        todo &= ~grow
    arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return filled


def repair_cell(cell: Image.Image, dark_max: int = 62, opaque_min: int = 170,
                bright_min: int = 150, grow: int = 3, min_area: int = 12) -> tuple:
    arr = np.array(cell.convert("RGBA"))
    alpha = arr[:, :, 3]
    lum = arr[:, :, :3].astype(np.int16).max(axis=2)

    dark = (alpha >= opaque_min) & (lum < dark_max)
    if not dark.any():
        return cell, 0

    bright = (alpha >= opaque_min) & (lum >= bright_min)
    near = bright.copy()
    for _ in range(grow):
        g = near.copy()
        g[1:, :] |= near[:-1, :]
        g[:-1, :] |= near[1:, :]
        g[:, 1:] |= near[:, :-1]
        g[:, :-1] |= near[:, 1:]
        near = g
    isolated = dark & (~near)

    total = 0
    for blob in find_blobs(isolated, min_area=min_area):
        total += fill_blob(arr, blob)
    return Image.fromarray(arr, "RGBA"), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--dark-max", type=int, default=62)
    ap.add_argument("--min-area", type=int, default=12)
    args = ap.parse_args()

    sheet = Image.open(args.atlas).convert("RGBA")
    W, H = sheet.size
    cw, ch = W // args.cols, H // args.rows
    out = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    total, cells = 0, []
    for i in range(args.cols * args.rows):
        r, c = divmod(i, args.cols)
        cell = sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
        fixed, n = repair_cell(cell, dark_max=args.dark_max, min_area=args.min_area)
        if n:
            cells.append((i + 1, n))
            total += n
        out.paste(fixed, (c * cw, r * ch))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out, format="WEBP", lossless=True, quality=100)
    print(f"[repair] filled {total} px across {len(cells)} cells", file=sys.stderr)
    print(f"[repair] cells: {cells}", file=sys.stderr)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
