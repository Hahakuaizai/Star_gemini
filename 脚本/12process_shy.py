#!/usr/bin/env python3
"""
12shy_praise（害羞捂脸扭捏）平滑48帧升级管线：
1. 全图连通域智能归属切分：由于原图第 3 行角色头顶呆毛与第 2 行角色双靴在垂直方向距离极近，
   采用「主连通域 + 靴子/粒子垂直带约束归属」将每个角色的双靴、悬浮粉红爱心、高光星星完整归属到本帧，
   彻底消除「上一行靴子被切到下一行头顶」或「靴尖被裁切」的问题。
2. 移除原本 alpha >= 128 硬二值切边导致的描边锯齿毛刺，改用四边黑底漫水填充 + 内部纯黑孔洞清理 + 悬浮粉红爱心去预乘，
   并在角色外轮廓 1~2px 边界处将残留暗黑像素向角色深紫描边色 (68, 48, 98) 净化融合 + 柔和 Alpha 高斯抗锯齿平滑（sigma=0.65）。
3. 严格锁定角色站姿尺寸与接地位置（画布 384x384，主体站高统一约 300px，脚底锚定 y=345，水平居中 x=192），
   确保靴子、猫尾、呆毛与悬浮爱心四周留白 >= 10%（>=38px），绝不裁切。
4. 在对齐关键帧之间的 12 个最大动作过渡区间（抬手捂脸、害羞左右扭捏、手指张开偷看、放下双手）插入金字塔光流中间帧
   （pyramidal_optical_flow + warp_rgba，平滑 Alpha 曲线 30..225），扩展为丝滑 48 帧（8x6，3072x2304）。
5. 放慢整体播放节奏，采用变速帧率使总时长达到 8.63s（8.4s~8.9s 区间）。
"""

import json
import os
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, map_coordinates, zoom

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHEETS = [
    os.path.join(BASE_DIR, "gemini", "素材", "12shy_praise", "part1.webp"),
    os.path.join(BASE_DIR, "gemini", "素材", "12shy_praise", "part2.webp"),
    os.path.join(BASE_DIR, "gemini", "素材", "12shy_praise", "part3.webp"),
]
OUT_DIR = os.path.join(BASE_DIR, "gemini", "成品")
PREVIEW_DIR = os.path.join(BASE_DIR, "gemini", "预览")

CANVAS = 384
TARGET_H = 300
GROUND = 345
HCX = 192
VC = 704
VO = 128
OUTLINE_RGB = np.array([68.0, 48.0, 98.0], dtype=np.float32)


def clean_sheet_black_bg_and_glow(
    img_rgba: Image.Image,
    bg_thresh: int = 28,
    hole_thresh: int = 20,
    glow_cutoff: int = 34,
) -> Image.Image:
    """
    对整张 4x3 素材图执行三阶段黑底清理 + 悬浮粉红爱心去预乘：
    1) 四边漫水填充去除外部近黑背景；
    2) 清理内部纯黑孔洞（双腿/手臂/猫尾之间，以及行与行之间的黑色空隙）；
    3) 对非角色主体的悬浮粉红爱心/星光进行去预乘，同时保护靴子连通域不被当成发光粒子。
    """
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    max_rgb = np.maximum(np.maximum(r, g), b)

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

    interior_black = (max_rgb <= hole_thresh) & (~bg_mask)
    bg_mask |= interior_black

    fg_mask = ~bg_mask
    lab, n_comp = ndimage.label(fg_mask)
    sizes = ndimage.sum(fg_mask, lab, index=range(1, n_comp + 1)) if n_comp > 0 else []

    out = arr.copy()
    out[bg_mask] = [0, 0, 0, 0]

    # 识别角色主体（面积 > 2000）与悬浮粒子/靴子连通域
    main_labels = [idx + 1 for idx, sz in enumerate(sizes) if sz > 2000]
    solid_mask = np.isin(lab, main_labels) if main_labels else np.zeros_like(fg_mask)

    bg_dilated = binary_dilation(bg_mask, iterations=1)
    solid_edge = solid_mask & bg_dilated
    edge_ys, edge_xs = np.where(solid_edge)
    for y, x in zip(edge_ys, edge_xs):
        m = int(max_rgb[y, x])
        if m < 48:
            out[y, x, 3] = max(0, int((m - bg_thresh) / max(1, 48 - bg_thresh) * 255))

    # 仅对偏粉红/高亮度的悬浮粒子做去预乘，不改变暗紫靴子的颜色
    particle_mask = fg_mask & (~solid_mask)
    glow_floor = 65
    ys, xs = np.where(particle_mask)
    for y, x in zip(ys, xs):
        m = int(max_rgb[y, x])
        if m < glow_cutoff:
            out[y, x] = [0, 0, 0, 0]
        elif m < glow_floor:
            out[y, x, 3] = max(0, int((m - glow_cutoff) / max(1, glow_floor - glow_cutoff) * 110))
        elif m < 235 and r[y, x] > b[y, x] - 10:
            alpha = min(255, int((m - glow_cutoff) / max(1, 235 - glow_cutoff) * 255))
            scale = 255.0 / max(1, m)
            out[y, x, 0] = min(255, int(out[y, x, 0] * scale))
            out[y, x, 1] = min(255, int(out[y, x, 1] * scale))
            out[y, x, 2] = min(255, int(out[y, x, 2] * scale))
            out[y, x, 3] = alpha

    return Image.fromarray(out, "RGBA")


def refine_outline_antialias(img_rgba: Image.Image) -> Image.Image:
    """
    平滑抗锯齿描边精修：
    - 在角色外轮廓 1~2px 边界处，将残留的暗黑背景杂色向角色干净的深紫描边色 (68, 48, 98) 净化收敛；
    - 对边界 Alpha 施加轻柔高斯平滑 (sigma=0.65)，彻底消除毛刺与锯齿，在白底和黑底下均保持清晰顺滑。
    """
    arr = np.array(img_rgba, dtype=np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg = alpha > 16.0
    if not fg.any():
        return img_rgba

    inner = binary_erosion(fg, iterations=2)
    outer = binary_dilation(fg, iterations=1)
    boundary = outer & (~inner)

    max_c = rgb.max(axis=2)
    dark_edge = boundary & (alpha > 8.0) & (max_c < 120.0)
    t = np.clip((120.0 - max_c) / 95.0, 0.0, 1.0) * 0.80
    for c in range(3):
        rgb[:, :, c] = np.where(
            dark_edge,
            rgb[:, :, c] * (1.0 - t) + OUTLINE_RGB[c] * t,
            rgb[:, :, c],
        )

    smooth_alpha = gaussian_filter(alpha, sigma=0.65)
    new_alpha = np.where(boundary, smooth_alpha, alpha)
    new_alpha = np.where(inner, np.maximum(alpha, 250.0), new_alpha)
    new_alpha = np.where(new_alpha < 10.0, 0.0, new_alpha)
    new_alpha = np.clip(new_alpha, 0.0, 255.0)

    arr[:, :, :3] = np.clip(rgb, 0.0, 255.0)
    arr[:, :, 3] = new_alpha
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def comps(alpha: np.ndarray):
    lab, n = ndimage.label(alpha > 15)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        out.append(
            dict(
                id=i,
                mask=m,
                size=len(xs),
                x0=int(xs.min()),
                x1=int(xs.max()),
                y0=int(ys.min()),
                y1=int(ys.max()),
                cx=float(xs.mean()),
                cy=float(ys.mean()),
            )
        )
    return lab, out


def split_fused(cs):
    out = []
    for c in cs:
        hgt = c["y1"] - c["y0"]
        if c["size"] > 45000 and hgt > 500:
            m = c["mask"]
            lo = c["y0"] + int(0.25 * hgt)
            hi = c["y0"] + int(0.60 * hgt)
            widths = [(y, int(m[y].sum())) for y in range(lo, hi)]
            ystar, wstar = min(widths, key=lambda t: t[1])
            if wstar < 60:
                # 粘连发生在上方角色靴底与下方角色呆毛尖之间：
                # 在 ystar 附近向下搜索呆毛尖（宽度 <= 18px 的细颈末端），在呆毛尖处断开，使整只靴子留在 top (上方角色)！
                y_cut = ystar
                for yy in range(ystar, min(c["y1"] - 10, ystar + 55)):
                    if int(m[yy].sum()) <= 18:
                        y_cut = yy
                        break
                top = m.copy()
                top[y_cut + 1 :, :] = False
                bot = m.copy()
                bot[: y_cut + 1, :] = False
                for pm in (top, bot):
                    ys, xs = np.nonzero(pm)
                    if len(xs) > 500:
                        out.append(
                            dict(
                                mask=pm,
                                size=len(xs),
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


def extract_sheet(path: str) -> List[np.ndarray]:
    """
    提取单张 4x3 图集的 12 个角色单元格：
    - 使用原脚本经验证 100% 准确的暗色分割掩码 (mx < 46) 提取连通域与 split_fused；
    - 使用 clean_sheet_black_bg_and_glow 生成柔和抗锯齿 + 发光粒子去预乘的 RGBA 像素；
    - 靴子归属给上方主身体，且在单格内清理任何位于头顶上方的暗紫残留块或脚底外的跨行碎屑。
    """
    im_pil = Image.open(path).convert("RGBA")
    raw_rgb = np.asarray(im_pil.convert("RGB"))
    cleaned = clean_sheet_black_bg_and_glow(im_pil, bg_thresh=28, hole_thresh=20, glow_cutoff=34)
    rgba = np.asarray(cleaned)

    mx = raw_rgb.max(axis=2)
    dark = mx < 46
    lab_d, n_d = ndimage.label(dark)
    border_labels = set(np.unique(np.concatenate([lab_d[0, :], lab_d[-1, :], lab_d[:, 0], lab_d[:, -1]])))
    border_labels.discard(0)
    bg = np.isin(lab_d, list(border_labels)) if border_labels else np.zeros_like(dark)
    sat = (mx - raw_rgb.min(axis=2)) / np.maximum(mx, 1)
    seg_alpha = np.ones(mx.shape, dtype=np.uint8) * 255
    seg_alpha[bg] = 0
    for i in range(1, n_d + 1):
        if i in border_labels:
            continue
        m = lab_d == i
        if m.sum() < 8 or (dark[m].all() and mx[m].mean() < 40 and sat[m].mean() < 0.16):
            seg_alpha[m] = 0

    _, cs = comps(seg_alpha)
    cs = split_fused(cs)
    mains = sorted([c for c in cs if c["size"] > 20000], key=lambda d: d["cy"])
    assert len(mains) == 12, f"{os.path.basename(path)}: expected 12 mains, got {len(mains)}"
    rows = []
    for r in range(3):
        rows.append(sorted(mains[r * 4 : (r + 1) * 4], key=lambda d: d["cx"]))
    ordered = [m for row in rows for m in row]
    main_ids = {id(m) for m in ordered}

    owner = {}
    for c in cs:
        if id(c) in main_ids:
            continue
        if c["size"] < 8:
            continue
        d2 = [((c["cx"] - m["cx"]) ** 2 + (c["cy"] - m["cy"]) ** 2, k) for k, m in enumerate(ordered)]
        owner[id(c)] = min(d2)[1]

    cells = []
    H, W = rgba.shape[:2]
    for k, m in enumerate(ordered):
        r, cc = divmod(k, 4)
        ox, oy = cc * (W // 4), r * (H // 3)
        cell = np.zeros((VC, VC, 4), dtype=np.uint8)
        parts = [m] + [c for c in cs if owner.get(id(c)) == k]
        mask = np.zeros((H, W), dtype=bool)
        for p in parts:
            mask |= p["mask"]
        ys_raw, _ = np.nonzero(mask)
        min_y_cell = max(0, int(ys_raw.min()) - 2)
        max_y_cell = min(H - 1, int(ys_raw.max()) + 2)
        dil = binary_dilation(mask, iterations=2) & (rgba[:, :, 3] > 0)
        dil[:min_y_cell, :] = False
        dil[max_y_cell + 1 :, :] = False
        ys, xs = np.nonzero(dil)
        vx, vy = xs - ox + VO, ys - oy + VO
        ok = (vx >= 0) & (vx < VC) & (vy >= 0) & (vy < VC)
        cell[vy[ok], vx[ok]] = rgba[ys[ok], xs[ok]]
        cells.append(cell)
    return cells


def face_metrics(alpha: np.ndarray, main: dict) -> Tuple[float, float]:
    m = main["mask"]
    y0, y1 = main["y0"], main["y1"]
    h = y1 - y0 + 1
    spans = []
    for y in range(y0 + int(0.08 * h), y0 + int(0.38 * h)):
        xs = np.nonzero(m[y])[0]
        if len(xs) > 3:
            spans.append(int(xs.max() - xs.min()))
    band = float(np.median(spans)) if spans else float(main["x1"] - main["x0"])
    top_rows = m[y0 : y0 + int(0.30 * h)]
    xs = np.nonzero(top_rows)[1]
    hcx = float(np.median(xs)) if len(xs) else main["cx"]
    return band, hcx


def pyramidal_optical_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 4,
    win_sigma: float = 3.5,
    iters: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """金字塔由粗到精 Lucas-Kanade 稠密光流。"""
    def build_pyr(im: np.ndarray) -> List[np.ndarray]:
        pyr = [im.astype(np.float32)]
        for _ in range(1, levels):
            blurred = gaussian_filter(pyr[-1], sigma=1.0)
            pyr.append(blurred[::2, ::2])
        return pyr

    p1 = build_pyr(img1)
    p2 = build_pyr(img2)

    vx = np.zeros_like(p1[-1], dtype=np.float32)
    vy = np.zeros_like(p1[-1], dtype=np.float32)

    for lvl in range(levels - 1, -1, -1):
        i1 = p1[lvl]
        i2 = p2[lvl]
        lh, lw = i1.shape
        if lvl < levels - 1:
            sy = lh / vx.shape[0]
            sx = lw / vx.shape[1]
            vx = zoom(vx, (sy, sx), order=1) * sx
            vy = zoom(vy, (sy, sx), order=1) * sy

        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        for _ in range(iters):
            wx = np.clip(xx + vx, 0, lw - 1)
            wy = np.clip(yy + vy, 0, lh - 1)
            i2_w = map_coordinates(i2, [wy, wx], order=1, mode="nearest")

            ix = np.gradient(i2_w, axis=1)
            iy = np.gradient(i2_w, axis=0)
            it = i2_w - i1

            s_xx = gaussian_filter(ix * ix, sigma=win_sigma) + 1e-3
            s_yy = gaussian_filter(iy * iy, sigma=win_sigma) + 1e-3
            s_xy = gaussian_filter(ix * iy, sigma=win_sigma)
            s_xt = gaussian_filter(ix * it, sigma=win_sigma)
            s_yt = gaussian_filter(iy * it, sigma=win_sigma)

            det = s_xx * s_yy - s_xy * s_xy + 1e-4
            dvx = -(s_yy * s_xt - s_xy * s_yt) / det
            dvy = -(s_xx * s_yt - s_xy * s_xt) / det

            dvx = np.clip(dvx, -2.5, 2.5)
            dvy = np.clip(dvy, -2.5, 2.5)

            vx = gaussian_filter(vx + dvx, sigma=1.2)
            vy = gaussian_filter(vy + dvy, sigma=1.2)

    return vx, vy


def warp_rgba(arr: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """使用预乘 Alpha 的双线性插值变形，防止黑边渗色。"""
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sy = np.clip(yy + dy, 0, h - 1)
    sx = np.clip(xx + dx, 0, w - 1)

    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    out = np.zeros((h, w, 4), dtype=np.float32)
    wa = map_coordinates(alpha, [sy, sx], order=1, mode="constant", cval=0.0)
    for c in range(3):
        premul = arr[:, :, c].astype(np.float32) * alpha
        wc = map_coordinates(premul, [sy, sx], order=1, mode="constant", cval=0.0)
        out[:, :, c] = np.where(wa > 1e-3, np.clip(wc / np.maximum(wa, 1e-3), 0, 255), 0.0)
    out[:, :, 3] = np.clip(wa * 255.0, 0, 255)
    return out


def interpolate_midpoint_smooth(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """
    双向金字塔光流插帧 + 一致性门控融合 + 平滑 Alpha 曲线 (30..225) + 描边抗锯齿精修。
    """
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.45 * al

    fa = feature_map(a)
    fb = feature_map(b)

    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=4, win_sigma=3.5, iters=6)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=4, win_sigma=3.5, iters=6)

    wa = warp_rgba(a, -0.5 * vx_ab, -0.5 * vy_ab)
    wb = warp_rgba(b, -0.5 * vx_ba, -0.5 * vy_ba)

    rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
    alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
    disagree = rgb_diff * 0.7 + alpha_diff * 0.9

    agree_weight = np.clip(1.0 - disagree * 2.8, 0.0, 1.0)
    pick_a = (wa[:, :, 3] >= wb[:, :, 3]).astype(np.float32)
    w_a = 0.5 * agree_weight + pick_a * (1.0 - agree_weight)
    w_b = 1.0 - w_a

    aa = wa[:, :, 3] / 255.0
    ab = wb[:, :, 3] / 255.0
    out_a = w_a * aa + w_b * ab

    out = np.zeros_like(wa)
    for c in range(3):
        num = w_a * wa[:, :, c] * aa + w_b * wb[:, :, c] * ab
        out[:, :, c] = np.where(out_a > 1e-3, np.clip(num / np.maximum(out_a, 1e-3), 0, 255), 0.0)

    # 平滑 Alpha 曲线 30..225，保留柔和抗锯齿过渡同时消除半透明重影
    out_a_u8 = np.clip(out_a * 255.0, 0, 255)
    out_a_smooth = np.where(
        out_a_u8 < 30.0,
        0.0,
        np.where(out_a_u8 > 225.0, 255.0, (out_a_u8 - 30.0) / 195.0 * 255.0),
    )
    out[:, :, 3] = out_a_smooth

    mid_img = Image.fromarray(out.astype(np.uint8), "RGBA")
    return refine_outline_antialias(mid_img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    cells = []
    for sp in SHEETS:
        cells.extend(extract_sheet(sp))
    assert len(cells) == 36, f"Expected 36 cells, got {len(cells)}"

    info = []
    for rgba in cells:
        alpha = rgba[:, :, 3]
        _, cs = comps(alpha)
        m = max(cs, key=lambda d: d["size"])
        band, hcx = face_metrics(alpha, m)
        gbot = m["y1"]
        for c in cs:
            if (
                c["size"] >= 200
                and m["x0"] - 40 <= c["cx"] <= m["x1"] + 40
                and c["y0"] <= m["y1"] + 60
            ):
                gbot = max(gbot, c["y1"])
        top_y = m["y0"]
        info.append(
            dict(
                main=m,
                band=band,
                hcx=hcx,
                h=gbot - top_y + 1,
                bot=gbot,
                top=top_y,
            )
        )

    bands = np.array([d["band"] for d in info])
    sm = np.array([np.median(bands[max(0, i - 2) : min(36, i + 3)]) for i in range(36)])
    B0 = sm[0]
    s1 = TARGET_H / info[0]["h"]
    scales = [float(np.clip(s1 * (B0 / sm[i]) ** 0.80, 0.58, 0.78)) for i in range(36)]

    # 生成 36 个对齐且抗锯齿精修的关键帧（384x384，脚底锁定 GROUND=345，确保 top >= 42px, bottom = 38px 即四周均 >=10%）
    keyframes: List[Image.Image] = []
    for i in range(36):
        d = info[i]
        s = scales[i]
        if d["h"] * s > TARGET_H:
            s = TARGET_H / d["h"]
        im = Image.fromarray(cells[i], "RGBA")
        nw, nh = max(1, round(VC * s)), max(1, round(VC * s))
        im2 = im.resize((nw, nh), Image.Resampling.LANCZOS)
        ox = int(round(HCX - d["hcx"] * s))
        oy = int(round(GROUND - d["bot"] * s))
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(im2, (ox, oy), im2)

        arr = np.array(canvas)
        alpha = arr[:, :, 3]
        _, cs = comps(alpha)
        if cs:
            main_c = max(cs, key=lambda c_item: c_item["size"])
            # 若头顶呆毛最顶端 18px 内粘连了上一行残留的暗紫靴底像素（R < B - 12 且 max_rgb < 140），直接抹除该残块
            top_band_y1 = min(CANVAS, main_c["y0"] + 16)
            for yy in range(main_c["y0"], top_band_y1):
                xs_row = np.nonzero(main_c["mask"][yy])[0]
                if len(xs_row) > 0:
                    row_r = arr[yy, xs_row, 0].astype(int)
                    row_b = arr[yy, xs_row, 2].astype(int)
                    row_mx = arr[yy, xs_row, :3].max(axis=1)
                    # 靴底为深暗蓝紫 (mx < 115)，而呆毛发丝较亮 (mx > 135)
                    boot_tip = xs_row[(row_mx < 120) & (row_b >= row_r)]
                    if len(boot_tip) > 0 and (yy - main_c["y0"]) < 14:
                        arr[yy, boot_tip] = [0, 0, 0, 0]
            alpha = arr[:, :, 3]
            _, cs = comps(alpha)
            main_c = max(cs, key=lambda c_item: c_item["size"])
            rgbmax = arr[:, :, :3].max(axis=2)
            gbot_f = main_c["y1"]
            for c in cs:
                if (
                    c["size"] >= 200
                    and main_c["x0"] - 40 <= c["cx"] <= main_c["x1"] + 40
                    and c["y0"] <= main_c["y1"] + 55
                ):
                    gbot_f = max(gbot_f, c["y1"])
            for c in cs:
                if c["id"] == main_c["id"]:
                    continue
                wdt = c["x1"] - c["x0"] + 1
                hgt = c["y1"] - c["y0"] + 1
                kill = False
                if c["y0"] > gbot_f + 2:
                    kill = True
                # 清除任何位于角色头顶上方 (y1 < main_c["y0"] + 15) 的非粉红暗紫块（上一行靴底粘连到呆毛尖的残块，其 R 通道 <= B 通道）
                if c["y1"] < main_c["y0"] + 15 and arr[c["mask"], 0].mean() <= arr[c["mask"], 2].mean() + 8:
                    kill = True
                if wdt <= 14 and hgt >= 20 and (c["x1"] < 75 or c["x0"] > 309):
                    kill = True
                if c["x0"] < 38 or c["x1"] > CANVAS - 39 or c["y0"] < 38 or c["y1"] > CANVAS - 39:
                    kill = True
                if kill:
                    arr[c["mask"]] = [0, 0, 0, 0]

        refined = refine_outline_antialias(Image.fromarray(arr, "RGBA"))
        keyframes.append(refined)

    # 首尾帧严格锁定为 Frame 0 标准站姿
    keyframes[35] = keyframes[0].copy()

    # 在动作幅度最大的 12 个关键过渡位置插入光流中间帧（36 + 12 = 48 帧）
    priority_pairs = {1, 2, 3, 5, 6, 14, 18, 19, 23, 25, 27, 28}
    assert len(priority_pairs) == 12

    frames_48: List[Image.Image] = []
    frame_roles: List[str] = []

    for i in range(36):
        frames_48.append(keyframes[i])
        if i in {0, 4, 12, 13, 16, 17, 22, 23, 33, 34, 35}:
            frame_roles.append("hold")
        elif 5 <= i <= 11 or 18 <= i <= 21:
            frame_roles.append("twist")
        else:
            frame_roles.append("transition")

        if i in priority_pairs:
            mid = interpolate_midpoint_smooth(keyframes[i], keyframes[i + 1])
            frames_48.append(mid)
            if 5 <= i <= 11 or 18 <= i <= 21:
                frame_roles.append("twist")
            else:
                frame_roles.append("transition")

    assert len(frames_48) == 48, f"Expected 48 frames, got {len(frames_48)}"

    role_ms = {
        "transition": 150,
        "twist": 180,
        "hold": 235,
    }
    durations = [role_ms[r] for r in frame_roles]
    durations[0] = 250
    durations[-1] = 260
    total_ms = sum(durations)
    assert 8400 <= total_ms <= 8900, f"Total duration {total_ms}ms out of 8.4s..8.9s range"

    min_top, max_bot, min_left, max_right = CANVAS, 0, CANVAS, 0
    semi_count = 0
    for fr in frames_48:
        a = np.array(fr)[:, :, 3]
        semi_count += int(((a > 0) & (a < 255)).sum())
        ys, xs = np.nonzero(a > 10)
        if len(ys) > 0:
            min_top = min(min_top, int(ys.min()))
            max_bot = max(max_bot, int(ys.max()))
            min_left = min(min_left, int(xs.min()))
            max_right = max(max_right, int(xs.max()))

    cols, rows = 8, 6
    sheet = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_48):
        r, c = divmod(i, cols)
        sheet.paste(fr, (c * CANVAS, r * CANVAS), fr)

    sheet_path = os.path.join(OUT_DIR, "12shy_praise_sheet（害羞捂脸_图集）.webp")
    anim_path = os.path.join(OUT_DIR, "12shy_praise（害羞捂脸_动画）.webp")
    white_preview = os.path.join(PREVIEW_DIR, "12shy_praise_white.png")
    black_preview = os.path.join(PREVIEW_DIR, "12shy_praise_black.png")

    sheet.save(sheet_path, format="WEBP", lossless=True, quality=100)
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

    white_bg = Image.new("RGBA", sheet.size, (255, 255, 255, 255))
    white_bg.alpha_composite(sheet)
    white_bg.convert("RGB").save(white_preview)

    black_bg = Image.new("RGBA", sheet.size, (0, 0, 0, 255))
    black_bg.alpha_composite(sheet)
    black_bg.convert("RGB").save(black_preview)

    summary = {
        "action": "12shy_praise（害羞捂脸扭捏）",
        "frames": len(frames_48),
        "grid": f"{cols}x{rows}",
        "cell_size": f"{CANVAS}x{CANVAS}",
        "sheet_size": f"{sheet.width}x{sheet.height}",
        "total_duration_ms": total_ms,
        "total_seconds": round(total_ms / 1000.0, 2),
        "interpolated_pairs": sorted(list(priority_pairs)),
        "antialiased_edge_pixels": semi_count,
        "bbox_margins": {
            "top_px": min_top,
            "bottom_px": CANVAS - 1 - max_bot,
            "left_px": min_left,
            "right_px": CANVAS - 1 - max_right,
            "ground_y": max_bot,
        },
        "first_last_diff": int(
            np.abs(np.array(frames_48[0], dtype=int) - np.array(frames_48[-1], dtype=int)).max()
        ),
        "outputs": {
            "sheet_webp": sheet_path,
            "anim_webp": anim_path,
            "preview_white": white_preview,
            "preview_black": black_preview,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
