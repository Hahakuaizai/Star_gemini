#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matte a pure-black-background sprite cell into RGBA without destroying glow.

The stock `clean_black_bg_and_glow` floods the background, then un-premultiplies
every floating component by scaling RGB to 255 and using brightness as alpha.
That is right for a soft halo, but for a *saturated mid-bright* sprite such as
a cyan "< / >" glyph the scale factor blows the colour out and the dark interior
of the glyph collapses to an opaque black patch.

This matte keeps the ratio instead: alpha is driven by how far the pixel sits
above the black background, and RGB is restored by dividing by that same alpha
(the standard un-premultiply for a sprite composited onto black):

    a = clamp(lum / K, 0, 1)      # K = the brightness of a fully opaque pixel
    rgb_out = rgb_in / a

With K below full white, bright glow keeps a high alpha instead of being
stretched, and dark pixels simply get low alpha — no black islands.
"""
import argparse
import os
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image


def flood_bg(lum: np.ndarray, thresh: int) -> np.ndarray:
    """Background = near-black pixels reachable from the cell border."""
    h, w = lum.shape
    dark = lum <= thresh
    visited = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if dark[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if dark[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        bg[cy, cx] = True
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return bg


def components(mask: np.ndarray, min_area: int = 1) -> List[np.ndarray]:
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


def matte_cell(cell: Image.Image, bg_thresh: int = 22, K: float = 208.0,
               min_alpha: int = 12, drop_below: int = 10) -> Image.Image:
    arr = np.array(cell.convert("RGB")).astype(np.float32)
    lum = arr.max(axis=2)

    bg = flood_bg(lum, bg_thresh)
    fg = ~bg

    # alpha from brightness above the background floor
    a = np.clip((lum - bg_thresh) / max(1.0, K - bg_thresh), 0.0, 1.0)
    a[bg] = 0.0

    # kill isolated specks that are too dim to be real artwork
    dim = fg & (a * 255 < min_alpha)
    for comp in components(dim, min_area=1):
        a[comp] = 0.0

    safe = np.maximum(a, 1e-3)
    rgb = np.clip(arr / safe[:, :, None], 0, 255)

    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    out[:, :, :3] = rgb.astype(np.uint8)
    out[:, :, 3] = (a * 255).astype(np.uint8)
    out[out[:, :, 3] < drop_below] = 0
    return Image.fromarray(out, "RGBA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="+", required=True,
                    help="4x3 black-background segments, in order")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--K", type=float, default=208.0)
    ap.add_argument("--bg-thresh", type=int, default=22)
    ap.add_argument("--prefix", default="matted")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for si, path in enumerate(args.segments, start=1):
        sheet = Image.open(path).convert("RGB")
        w, h = sheet.size
        cw, ch = w // 4, h // 3
        out = Image.new("RGBA", (cw * 4, ch * 3), (0, 0, 0, 0))
        for r in range(3):
            for c in range(4):
                cell = sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
                m = matte_cell(cell, bg_thresh=args.bg_thresh, K=args.K)
                out.paste(m, (c * cw, r * ch))
        p = os.path.join(args.out_dir, f"{args.prefix}_part{si}.png")
        out.save(p)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
