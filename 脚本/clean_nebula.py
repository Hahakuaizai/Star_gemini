#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove dark nebula patches painted behind the character.

The image model sometimes renders a soft dark-purple nebula behind the hair.
After matting it becomes a large opaque near-black region that reads as a hole
in the sprite.

Separating it from the character's outline cannot be done by brightness alone —
both are dark purple. But their *thickness* differs: an outline is a thin line
1-3 px wide, a nebula is a solid mass tens of pixels across. So erode the dark
mask with a small kernel: an outline disappears, a nebula survives. Everything
that survives the erosion is nebula and gets erased (alpha 0), while outline
pixels stay exactly as they were.
"""
import argparse
import os
import sys
from collections import deque
from typing import List

import numpy as np
from PIL import Image


def erode(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(k):
        e = out.copy()
        e[1:, :] &= out[:-1, :]
        e[:-1, :] &= out[1:, :]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        out = e
    return out


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(k):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def components(mask: np.ndarray, min_area: int) -> List[np.ndarray]:
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


def clean_cell(cell: Image.Image, dark_max: int = 95, erode_k: int = 2,
               min_area: int = 60) -> tuple:
    arr = np.array(cell.convert("RGBA"))
    alpha = arr[:, :, 3]
    lum = arr[:, :, :3].astype(np.int16).max(axis=2)

    dark = (alpha >= 150) & (lum < dark_max)
    if dark.sum() < min_area:
        return cell, 0

    core = erode(dark, erode_k)
    if core.sum() < 20:
        return cell, 0

    # a nebula's core is a solid mass; grow each core back into its dark body
    removed = 0
    for c in components(core, min_area=20):
        seed = c.copy()
        # restrict the regrow to dark pixels so outlines are never touched
        grown = seed.copy()
        while True:
            nxt = dilate(grown, 1) & dark
            if nxt.sum() == grown.sum():
                break
            grown = nxt
        # only erase when the surviving body is genuinely bulky
        if grown.sum() >= min_area:
            ys, xs = np.where(grown)
            hgt = ys.max() - ys.min() + 1
            wid = xs.max() - xs.min() + 1
            if min(hgt, wid) >= 7:
                arr[grown] = [0, 0, 0, 0]
                removed += int(grown.sum())
    return Image.fromarray(arr, "RGBA"), removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--dark-max", type=int, default=95)
    ap.add_argument("--erode", type=int, default=2)
    ap.add_argument("--min-area", type=int, default=60)
    args = ap.parse_args()

    sheet = Image.open(args.atlas).convert("RGBA")
    W, H = sheet.size
    cw, ch = W // args.cols, H // args.rows
    out = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    total, cells = 0, []
    for i in range(args.cols * args.rows):
        r, c = divmod(i, args.cols)
        cell = sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
        fixed, n = clean_cell(cell, dark_max=args.dark_max,
                              erode_k=args.erode, min_area=args.min_area)
        if n:
            cells.append((i + 1, n))
            total += n
        out.paste(fixed, (c * cw, r * ch))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out, format="WEBP", lossless=True, quality=100)
    print(f"[nebula] erased {total} px across {len(cells)} cells", file=sys.stderr)
    print(f"[nebula] cells: {cells}", file=sys.stderr)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
