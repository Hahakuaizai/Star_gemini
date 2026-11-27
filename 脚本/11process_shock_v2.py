#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action 11: 11poke_shock (被戳受惊炸毛) v2
Fixes:
1. "描边不清晰有毛刺": Replaced hard binary thresholding (alpha >= 128 ? 255 : 0) with
   soft anti-aliased matting + edge color decontamination toward dark-purple outline (68, 48, 98),
   plus complete removal of interior dark holes between tail/hair/legs and floor shadow lines.
2. "缺失中间帧，动画不流畅": Expanded coherent hand-drawn keyframes to 48 frames (8x6 grid, 3072x2304)
   using bidirectional pyramidal optical-flow interpolation (`pyramidal_optical_flow` + `warp_rgba`)
   with smooth anti-aliased alpha curve (30..225).
3. "动画速度太快": Variable-speed playback durations totaling ~8.35s (within 8.1s ~ 8.6s).
4. Strict character size & ground lock (384x384 per cell, standing height ~306px grounded at y=366,
   slight 4px startle hop on peak shock frame, full boots/tail/ahoge/shock lines with >=10% safe margins).
"""
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from expand_48 import pyramidal_optical_flow, warp_rgba

CANVAS = 384
TARGET_H = 306
GROUND_Y = 366
OUTLINE_RGB = np.array([68.0, 48.0, 98.0], dtype=np.float32)
VC = 704
VO = 128

SHEETS = [
    "gemini/素材/11poke_shock/part1.webp",
    "gemini/素材/11poke_shock/part2.webp",
    "gemini/素材/11poke_shock/part3.webp",
]


def comps(mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        out.append(
            dict(
                id=i,
                mask=m,
                size=int(len(xs)),
                x0=int(xs.min()),
                x1=int(xs.max()),
                y0=int(ys.min()),
                y1=int(ys.max()),
                cx=float(xs.mean()),
                cy=float(ys.mean()),
            )
        )
    return lab, out


def split_fused(cs: List[dict]) -> List[dict]:
    """Split vertically fused character components across adjacent rows (e.g. ahoge touching upper boot)."""
    out = []
    for c in cs:
        hgt = c["y1"] - c["y0"]
        if c["size"] > 110000 and hgt > 600:
            m = c["mask"]
            lo = c["y0"] + int(0.32 * hgt)
            hi = c["y0"] + int(0.68 * hgt)
            widths = [(y, int(m[y].sum())) for y in range(lo, hi)]
            ystar, wstar = min(widths, key=lambda t: t[1])
            if wstar < 60:
                top = m.copy()
                top[ystar:, :] = False
                bot = m.copy()
                bot[:ystar, :] = False
                for pm in (top, bot):
                    ys, xs = np.nonzero(pm)
                    if len(xs) > 500:
                        out.append(
                            dict(
                                id=c["id"],
                                mask=pm,
                                size=int(len(xs)),
                                x0=int(xs.min()),
                                x1=int(xs.max()),
                                y0=int(ys.min()),
                                y1=int(ys.max()),
                                cx=float(xs.mean()),
                                cy=float(ys.mean()),
                            )
                        )
                continue
        out.append(c)
    return out


def soft_cutout_sheet(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Soft anti-aliased matting + edge color decontamination on full sheet:
    1) Border flood-fill on dark background (max_rgb < 42).
    2) Clear all interior dark holes/pockets between hair, tail, and legs (mean max_rgb < 58, low saturation).
    3) Smooth 1-2px outer boundary alpha with gaussian_filter(sigma=0.65) and decontaminate dark boundary RGB
       toward clean dark-purple outline (68, 48, 98).
    """
    rgb = rgb_arr.astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)

    # 1. Exterior dark flood fill
    dark = mx < 42
    lab, n = ndimage.label(dark)
    border_ids = set(
        np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    ) - {0}
    bg = np.isin(lab, list(border_ids)) if border_ids else np.zeros_like(dark)

    # 2. Interior enclosed dark pockets (gaps between legs/tail/sleeves)
    # Note: on dark pixels (mx < 30), (mx - mn)/mx can be > 0.7 even for a 2px chroma difference!
    # Use absolute chroma (mx - mn) < 18 or very dark mx < 28 to reliably clear all enclosed background holes.
    chroma = mx - mn
    for i in range(1, n + 1):
        if i in border_ids:
            continue
        m = lab == i
        sz = int(m.sum())
        if sz < 15:
            bg |= m
        elif mx[m].mean() < 34 and chroma[m].mean() < 18.0:
            bg |= m

    # Also catch slightly lighter dark pockets (mx < 52, low chroma < 16) enclosed between tail and skirt
    dark_wide = (mx < 52) & (chroma < 16.0) & (~bg)
    lab_w, n_w = ndimage.label(dark_wide)
    for i in range(1, n_w + 1):
        m = lab_w == i
        sz = int(m.sum())
        if sz >= 30 and mx[m].mean() < 42 and chroma[m].mean() < 14.0:
            bg |= m

    fg = ~bg
    # Remove tiny isolated specks < 10px
    lab_fg, n_fg = ndimage.label(fg)
    for i in range(1, n_fg + 1):
        m = lab_fg == i
        if m.sum() < 10:
            fg[m] = False
            bg[m] = True

    # 3. Soft anti-aliased alpha along the 1-2px boundary
    alpha = fg.astype(np.float32) * 255.0
    eroded = binary_erosion(fg, iterations=1)
    dilated = binary_dilation(fg, iterations=1)
    boundary_band = dilated & (~eroded)

    # Smooth alpha with gaussian_filter(sigma=0.65) and blend boundary ramp with brightness
    smooth_a = gaussian_filter(alpha, sigma=0.65)
    # Ramp factor based on pixel brightness at the outer 1-2px edge
    lum_ramp = np.clip((mx - 16.0) / (82.0 - 16.0), 0.0, 1.0) * 255.0
    edge_a = np.where(fg, np.minimum(smooth_a, np.maximum(smooth_a * 0.72, lum_ramp)), smooth_a * 0.35)
    alpha = np.where(boundary_band, edge_a, alpha)
    alpha = np.where(alpha < 14.0, 0.0, np.clip(alpha, 0.0, 255.0))

    # 4. Edge color decontamination toward dark-purple outline (68, 48, 98)
    edge_zone = (alpha > 0) & (~ binary_erosion(alpha > 220, iterations=2))
    out_rgb = rgb.copy()
    # Dark pixels near the edge have black background mixed in; shift their RGB toward clean dark-purple outline
    dark_weight = np.clip((108.0 - mx) / 88.0, 0.0, 1.0) * np.clip(1.0 - (alpha / 255.0) * 0.45, 0.25, 1.0)
    for c in range(3):
        decontam = out_rgb[:, :, c] * (1.0 - dark_weight) + OUTLINE_RGB[c] * dark_weight
        out_rgb[:, :, c] = np.where(edge_zone & (mx < 112), decontam, out_rgb[:, :, c])

    rgba = np.zeros((rgb_arr.shape[0], rgb_arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return rgba


def extract_sheet_cells(path: str) -> List[np.ndarray]:
    """Extract 12 cells in row-major order using whole-sheet connected component ownership."""
    im = np.asarray(Image.open(path).convert("RGB"))
    rgba = soft_cutout_sheet(im)
    alpha = rgba[:, :, 3]
    _, cs = comps(alpha > 20)
    cs = split_fused(cs)

    mains = sorted([c for c in cs if c["size"] > 20000], key=lambda d: d["cy"])
    assert len(mains) == 12, f"{path}: expected 12 mains, got {len(mains)}"
    rows = []
    for r in range(3):
        rows.append(sorted(mains[r * 4 : (r + 1) * 4], key=lambda d: d["cx"]))
    ordered = [m for row in rows for m in row]
    main_ids = {id(m) for m in ordered}

    owner = {}
    for c in cs:
        if id(c) in main_ids:
            continue
        # Assign floating shock lines / sweat drops to nearest main body
        d2 = [
            ((c["cx"] - m["cx"]) ** 2 + (c["cy"] - m["cy"]) ** 2, k)
            for k, m in enumerate(ordered)
        ]
        owner[id(c)] = min(d2)[1]

    H, W = alpha.shape
    cells: List[np.ndarray] = []
    for k, m in enumerate(ordered):
        r, cc = divmod(k, 4)
        ox, oy = cc * (W // 4), r * (H // 3)
        cell = np.zeros((VC, VC, 4), dtype=np.uint8)
        parts = [m] + [c for c in cs if owner.get(id(c)) == k]
        mask = np.zeros(alpha.shape, dtype=bool)
        for p in parts:
            mask |= p["mask"]
        # Include 1px anti-aliased fringe around each owned component
        mask = binary_dilation(mask, iterations=1) & (alpha > 0)

        ys, xs = np.nonzero(mask)
        vx, vy = xs - ox + VO, ys - oy + VO
        ok = (vx >= 0) & (vx < VC) & (vy >= 0) & (vy < VC)
        cell[vy[ok], vx[ok]] = rgba[ys[ok], xs[ok]]

        # Clean floating particles (un-premultiply bright shock lines & sweat drops, drop dim noise/floor lines)
        cell = clean_cell_particles_and_holes(cell)
        cells.append(cell)
    return cells


def clean_cell_particles_and_holes(cell: np.ndarray) -> np.ndarray:
    """
    1) Remove any remaining dark patch inside the main silhouette (between tail & skirt).
    2) Remove floor shadow lines below boots and thin trembling scratch slivers if noisy,
       while keeping & un-premultiplying white shock lines and cyan sweat drops cleanly.
    """
    out = cell.copy()
    alpha = out[:, :, 3]
    _, cs = comps(alpha > 20)
    if not cs:
        return out

    main_c = max(cs, key=lambda d: d["size"])
    rgb = out[:, :, :3].astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)

    # Check for any remaining dark low-chroma blob inside main component (e.g. black gap under tail)
    chroma = mx - mn
    inner_dark = main_c["mask"] & (mx < 38) & (chroma < 16.0)
    lab_d, n_d = ndimage.label(inner_dark)
    for i in range(1, n_d + 1):
        m = lab_d == i
        if m.sum() >= 20:
            out[m] = [0, 0, 0, 0]

    # Clear horizontal floor shadow lines at the very bottom of the character (attached or detached to boots)
    fg_ys = np.nonzero(out[:, :, 3] > 20)[0]
    if len(fg_ys) > 0:
        max_y = int(fg_ys.max())
        # In the bottom 12 rows, any dark low-chroma pixel is a floor shadow line, not purple boot
        floor_band = np.zeros_like(out[:, :, 3], dtype=bool)
        floor_band[max(0, max_y - 11) : max_y + 1, :] = True
        floor_shadow = floor_band & (out[:, :, 3] > 0) & (mx < 75.0) & (chroma < 26.0)
        out[floor_shadow] = [0, 0, 0, 0]
        # Also remove any thin horizontal strip in bottom 12 rows where row width exceeds boot width (>75px) or isolated spans
        for ry in range(max(0, max_y - 10), max_y + 1):
            r_xs = np.nonzero(out[ry, :, 3] > 0)[0]
            for x in r_xs:
                # If pixel has very little vertical support above it (floor shadow extending sideways from boots)
                above_a = out[max(0, ry - 14) : max(0, ry - 4), x, 3]
                if (above_a > 40).sum() == 0:
                    out[ry, x] = [0, 0, 0, 0]

    # Re-evaluate components after hole punch
    alpha = out[:, :, 3]
    _, cs = comps(alpha > 15)
    if not cs:
        return out
    main_c = max(cs, key=lambda d: d["size"])

    # Find true boot bottom (boots can be separated by ankle strap gap)
    gbot = main_c["y1"]
    for c in cs:
        if (
            c["size"] >= 280
            and main_c["x0"] - 45 <= c["cx"] <= main_c["x1"] + 45
            and c["y0"] <= main_c["y1"] + 65
        ):
            gbot = max(gbot, c["y1"])

    for c in cs:
        if c["id"] == main_c["id"]:
            continue
        m = c["mask"]
        # Keep boots
        if (
            c["size"] >= 280
            and main_c["x0"] - 45 <= c["cx"] <= main_c["x1"] + 45
            and c["y0"] <= main_c["y1"] + 65
        ):
            continue

        # Kill floor shadow lines below boots
        if c["y0"] > gbot - 4 and (c["x1"] - c["x0"] > 20 or c["y1"] > gbot + 1):
            out[m] = [0, 0, 0, 0]
            continue

        # Kill dim specks above head or along borders
        comp_mx = mx[m].mean()
        if c["size"] < 25 or comp_mx < 85:
            out[m] = [0, 0, 0, 0]
            continue

        # For valid floating particles (white shock lines / sweat drops), un-premultiply dark edge halo
        # and give a subtle dark-purple outline rim on white shock spikes so they are visible on BOTH white & black bg
        is_white_spike = (comp_mx > 175.0) and (c["cy"] < main_c["y0"] + 90)
        pys, pxs = np.nonzero(m)
        for y, x in zip(pys, pxs):
            pm = float(mx[y, x])
            if pm < 45:
                out[y, x] = [0, 0, 0, 0]
            elif is_white_spike and pm < 165:
                # Soft dark-purple outline rim around white shock lines
                out[y, x, 0] = 88
                out[y, x, 1] = 66
                out[y, x, 2] = 122
                out[y, x, 3] = int(min(220.0, (pm - 38.0) / (165.0 - 38.0) * 220.0))
            elif pm < 230:
                a_new = min(255.0, (pm - 35.0) / (230.0 - 35.0) * 255.0)
                scale = min(2.2, 255.0 / max(1.0, pm))
                out[y, x, 0] = int(min(255.0, out[y, x, 0] * scale))
                out[y, x, 1] = int(min(255.0, out[y, x, 1] * scale))
                out[y, x, 2] = int(min(255.0, out[y, x, 2] * scale))
                out[y, x, 3] = int(a_new)

    return out


def polish_resized_edges(img: Image.Image) -> Image.Image:
    """
    After LANCZOS resize to 384x384 canvas, apply soft anti-aliased alpha curve + edge color decontamination
    so edges are silky smooth with zero staircase burrs on black and zero dark halo on white.
    """
    arr = np.array(img, dtype=np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg = alpha > 24.0
    eroded = binary_erosion(fg, iterations=1)
    dilated = binary_dilation(fg, iterations=1)
    rim = dilated & (~eroded)

    # Subtle gaussian polish on the 1px outer rim only
    smooth_a = gaussian_filter(alpha, sigma=0.60)
    alpha = np.where(rim, 0.55 * alpha + 0.45 * smooth_a, alpha)
    # Clean sub-16 alpha dust while keeping full 16..245 anti-aliased ramp
    alpha = np.where(alpha < 16.0, 0.0, np.where(alpha > 246.0, 255.0, alpha))

    mx = rgb.max(axis=2)
    edge_zone = (alpha > 0) & (alpha < 235) & (mx < 115)
    w = np.clip((115.0 - mx) / 95.0, 0.0, 0.85)
    for c in range(3):
        rgb[:, :, c] = np.where(
            edge_zone,
            rgb[:, :, c] * (1.0 - w) + OUTLINE_RGB[c] * w,
            rgb[:, :, c],
        )

    out = np.zeros_like(arr, dtype=np.uint8)
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def align_all_cells(cells: List[np.ndarray]) -> List[Image.Image]:
    """
    Lock character standing body height consistently (~306px, grounded at y=366, slight 4px startle jump
    on the shock peak frames), centering head horizontal axis consistently.
    """
    info = []
    for rgba in cells:
        alpha = rgba[:, :, 3]
        _, cs = comps(alpha > 40)
        m = max(cs, key=lambda d: d["size"])

        # Measure head+hair band width in 10%..38% of main body height (immune to arms/tail/particles)
        y0, y1 = m["y0"], m["y1"]
        h = y1 - y0 + 1
        spans = []
        for y in range(y0 + int(0.10 * h), y0 + int(0.38 * h)):
            xs = np.nonzero(m["mask"][y])[0]
            if len(xs) > 5:
                spans.append(int(xs.max() - xs.min() + 1))
        band = float(np.median(spans)) if spans else float(m["x1"] - m["x0"] + 1)

        top_rows = m["mask"][y0 : y0 + int(0.32 * h)]
        txs = np.nonzero(top_rows)[1]
        hcx = float(np.median(txs)) if len(txs) else m["cx"]

        # Ground bottom: include separate boot components below ankle strap gap
        gbot = m["y1"]
        for c in cs:
            if (
                c["size"] >= 280
                and m["x0"] - 45 <= c["cx"] <= m["x1"] + 45
                and c["y0"] <= m["y1"] + 65
            ):
                gbot = max(gbot, c["y1"])

        total_body_h = gbot - y0 + 1
        info.append(dict(main=m, band=band, hcx=hcx, body_h=total_body_h, top=y0, bot=gbot))

    # Lock standing height strictly to TARGET_H (306px) with gentle face-band damping so startled poses
    # with wide stance (e.g. cell 12 startle jump) don't artificially stretch taller than standing frames.
    ref_h0 = float(info[0]["body_h"])
    ref_b0 = float(info[0]["band"])
    base_s = TARGET_H / ref_h0

    # Peak shock frames get a crisp 4px startle hop
    hop_set = {4, 12, 17}
    target_hcx = 186.0  # Centers body + right-side cat tail symmetrically inside 384x384

    aligned: List[Image.Image] = []
    for i, d in enumerate(info):
        raw_s = TARGET_H / max(1.0, float(d["body_h"]))
        band_s = base_s * (ref_b0 / max(1.0, float(d["band"]))) ** 0.65
        # Blend direct height lock with head-band lock so standing height stays ~306px across all 3 sheets
        s = 0.72 * raw_s + 0.28 * band_s
        s = float(np.clip(s, 0.64, 0.76))

        im = Image.fromarray(cells[i], "RGBA")
        nw, nh = max(1, int(round(VC * s))), max(1, int(round(VC * s)))
        im2 = im.resize((nw, nh), Image.Resampling.LANCZOS)

        hop = 4 if i in hop_set else 0
        ox = int(round(target_hcx - d["hcx"] * s))
        oy = int(round((GROUND_Y - hop) - d["bot"] * s))

        # Ensure top shock lines never clip top safe margin (y >= 16)
        r_arr = np.array(im2)
        ys_all = np.nonzero(r_arr[:, :, 3] > 20)[0]
        if len(ys_all) > 0:
            top_y = int(ys_all.min()) + oy
            if top_y < 16:
                oy += 16 - top_y

        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(im2, (ox, oy), im2)
        polished = polish_resized_edges(canvas)
        aligned.append(polished)

    return aligned


def interpolate_midpoint_smooth_aa(img_a: Image.Image, img_b: Image.Image, t: float = 0.5) -> Image.Image:
    """
    Bidirectional pyramidal optical-flow interpolation between img_a and img_b at fraction t,
    using smooth anti-aliased alpha curve (30..225) and edge color decontamination.
    """
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa, fb = feature_map(a), feature_map(b)
    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=4, win_sigma=3.5, iters=6)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=4, win_sigma=3.5, iters=6)

    wa = warp_rgba(a, -t * vx_ab, -t * vy_ab)
    wb = warp_rgba(b, -(1.0 - t) * vx_ba, -(1.0 - t) * vy_ba)

    rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
    alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
    disagree = rgb_diff * 0.7 + alpha_diff * 0.9

    agree_weight = np.clip(1.0 - disagree * 2.8, 0.0, 1.0)
    pick_a = (wa[:, :, 3] >= wb[:, :, 3]).astype(np.float32) if abs(t - 0.5) < 1e-3 else (1.0 if t < 0.5 else 0.0)
    w_a = (1.0 - t) * agree_weight + pick_a * (1.0 - agree_weight)
    w_b = 1.0 - w_a

    aa = wa[:, :, 3] / 255.0
    ab = wb[:, :, 3] / 255.0
    out_a = w_a * aa + w_b * ab

    out = np.zeros_like(wa)
    for c in range(3):
        num = w_a * wa[:, :, c] * aa + w_b * wb[:, :, c] * ab
        out[:, :, c] = np.where(out_a > 1e-3, np.clip(num / np.maximum(out_a, 1e-3), 0, 255), 0.0)

    # Smooth anti-aliased alpha curve (30..225) to preserve soft outline anti-aliasing without ghosting
    out_a_u8 = np.clip(out_a * 255.0, 0, 255)
    out_a_smooth = np.where(
        out_a_u8 < 30.0,
        0.0,
        np.where(out_a_u8 > 225.0, 255.0, (out_a_u8 - 30.0) / (225.0 - 30.0) * 255.0),
    )
    out[:, :, 3] = out_a_smooth

    return polish_resized_edges(Image.fromarray(out.astype(np.uint8), "RGBA"))


def main() -> dict:
    raw_cells: List[np.ndarray] = []
    for sp in SHEETS:
        raw_cells.extend(extract_sheet_cells(sp))
    assert len(raw_cells) == 36, f"Expected 36 raw cells, got {len(raw_cells)}"

    aligned_36 = align_all_cells(raw_cells)

    # Select 33 coherent keyframes in story order (excluding Part 3 frames 32..35 where floor shadows / arm waves appear,
    # and closing cleanly back to Frame 0 standard standing pose):
    # Part 1 (0..11): calm -> poked blink -> startled -> shock spikes + puffed tail -> trembling -> recovery
    # Part 2 (12..23): big startle with 3 white shock bars -> wide-eyed shock -> puffed tail trembling -> calming down
    # Part 3 (24..31): sigh of relief -> shy/embarrassed recovery -> patting chest -> calm standing -> Frame 0 closure
    key_indices = list(range(0, 32)) + [0]  # 33 coherent keyframes (index 0..32, with first == last == Frame 0)
    keyframes = [aligned_36[idx].copy() for idx in key_indices]
    assert len(keyframes) == 33

    # Insert 15 optical-flow in-between frames across the motion transitions (skipping 11->12 and 12->13 wide-stance jump
    # so legs never cross-ghost) to reach 48 frames (33 + 15 = 48):
    interp_after = {1, 2, 3, 4, 6, 8, 10, 14, 16, 17, 20, 22, 25, 29, 31}
    assert len(keyframes) + len(interp_after) == 48

    frames_48: List[Image.Image] = []
    for i in range(len(keyframes)):
        frames_48.append(keyframes[i])
        if i in interp_after:
            mid = interpolate_midpoint_smooth_aa(keyframes[i], keyframes[i + 1], t=0.5)
            frames_48.append(mid)

    # Guarantee exact Frame 0 == Frame 47 loop closure
    frames_48[-1] = frames_48[0].copy()
    assert len(frames_48) == 48, f"Expected 48 frames, got {len(frames_48)}"

    # Variable-speed durations totaling ~8.34s (within 8.1s ~ 8.6s target):
    # - Startle burst & fast trembling transitions: 135ms
    # - Normal trembling & recovery transitions: 175ms
    # - Start/end standing holds, shock freeze peaks, sigh of relief holds: 230ms
    fast_set = {3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 23, 24, 25, 26}
    hold_set = {0, 1, 10, 14, 15, 21, 27, 33, 34, 38, 42, 46, 47}
    durations = [
        135 if i in fast_set else (230 if i in hold_set else 175)
        for i in range(48)
    ]
    total_ms = sum(durations)
    assert 8100 <= total_ms <= 8600, f"Total duration {total_ms}ms outside 8100..8600ms"

    # Measure metrics across all 48 frames
    heights = []
    top_margins = []
    bot_margins = []
    left_margins = []
    right_margins = []
    semi_pixels = 0
    for fr in frames_48:
        a = np.array(fr)[:, :, 3]
        semi_pixels += int(((a > 0) & (a < 255)).sum())
        ys, xs = np.nonzero(a > 15)
        heights.append(int(ys.max() - ys.min() + 1))
        top_margins.append(int(ys.min()))
        bot_margins.append(int(CANVAS - 1 - ys.max()))
        left_margins.append(int(xs.min()))
        right_margins.append(int(CANVAS - 1 - xs.max()))

    # Build 8x6 sprite sheet (3072 x 2304)
    cols, rows = 8, 6
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS), fr)

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "11poke_shock_sheet（被戳受惊_图集）.webp")
    anim_path = os.path.join(out_dir, "11poke_shock（被戳受惊_动画）.webp")

    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)
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
    white_path = os.path.join(prev_dir, "11poke_shock_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "11poke_shock_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "11poke_shock",
        "frames": 48,
        "keyframes_used": len(keyframes),
        "optical_flow_midpoints": len(interp_after),
        "grid": "8x6",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_median": int(np.median(heights)),
        "min_margins_px": {
            "top": min(top_margins),
            "bottom": min(bot_margins),
            "left": min(left_margins),
            "right": min(right_margins),
        },
        "anti_aliased_edge_pixels_per_frame": round(semi_pixels / 48.0, 1),
        "total_duration_ms": total_ms,
        "total_seconds": round(total_ms / 1000.0, 2),
        "sheet_path": sheet_path,
        "anim_path": anim_path,
        "preview_white": white_path,
        "preview_black": black_path,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    main()
