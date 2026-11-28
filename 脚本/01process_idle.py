import importlib.util
import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask

_spec_swp = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec_swp)
_spec_swp.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned

_spec22 = importlib.util.spec_from_file_location("e22", "gemini/脚本/22process_eat_cookie.py")
_e22 = importlib.util.module_from_spec(_spec22)
_spec22.loader.exec_module(_e22)
interpolate_flow_multi = _e22.interpolate_flow_multi

CANVAS = 384
SS = 2  # 2x 超采样画布 (768x768)，彻底消除网格变形后的边框发糊与三次样条振铃
HR = CANVAS * SS
GROUND_Y = 366
HR_GROUND_Y = GROUND_Y * SS
HR_TARGET_H = 304 * SS
TOTAL_FRAMES = 96


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bell_window(t: float, t0: float, t1: float, ramp: float = 0.05) -> float:
    """Smooth raised-cosine window in [0, 1] active on [t0, t1]."""
    if t <= t0 or t >= t1:
        return 0.0
    if t < t0 + ramp:
        u = (t - t0) / ramp
        return 0.5 - 0.5 * math.cos(math.pi * u)
    if t > t1 - ramp:
        u = (t1 - t) / ramp
        return 0.5 - 0.5 * math.cos(math.pi * u)
    return 1.0


def clean_black_holes_and_defringe_hr(img_hr: Image.Image) -> Image.Image:
    """
    在 768x768 超采样层彻底消除黑底 JPG 抠图带来的所有黑边、发缝黑块与毛边，
    使外轮廓清透度完全对标原生透明 PNG（零黑边、柔和淡紫描边）：
    1. 清除发缝/衣缝所有内部黑洞与洞内孤立噪点
    2. 物理削薄最外 1.1px 黑底混合暗边（768 层收缩约 2.2px），保留丝滑抗锯齿
    3. 对发丝/猫耳/猫尾外轮廓及发缝残暗像素执行 EDT 内侧净色替换 + 柔紫线稿提亮（避开眼瞳与黑衣裙）
    """
    arr = np.array(img_hr, dtype=np.float32)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    uy, ux = yy / float(SS), xx / float(SS)

    # --- 1. 清除内部封闭黑洞（两侧波浪发缝隙里的纯黑/暗灰三角区） ---
    # 仅在头发/头侧/手外侧区域放宽黑洞判定，避免误伤深色衣裙与靴子
    in_outfit_core = (uy >= 198.0) & (uy <= 368.0) & (np.abs(ux - 192.0) <= 56.0)
    dark_bg_like = (a > 0) & (~in_outfit_core) & (mx < 62.0) & (chroma < 26.0)
    labeled, n_comp = label(dark_bg_like)
    hole_mask = np.zeros_like(dark_bg_like)
    if n_comp > 0:
        counts = np.bincount(labeled.ravel())
        for cid in range(1, n_comp + 1):
            if counts[cid] >= 12:
                hole_mask |= (labeled == cid)
    if hole_mask.any():
        hole_dilated = binary_dilation(hole_mask, iterations=4)
        solid_after_hole = (a > 20) & (~hole_mask)
        l_sol, n_sol = label(solid_after_hole)
        if n_sol > 1:
            sc = np.bincount(l_sol.ravel())
            main_id = int(np.argmax(sc[1:]) + 1)
            tiny_islands = (l_sol > 0) & (l_sol != main_id) & (sc[l_sol] < 45) & hole_dilated
            hole_mask |= tiny_islands
        a = np.where(hole_mask, 0.0, a)

    # --- 2. 外轮廓物理削去黑底暗皮 + 丝滑抗锯齿 ---
    # 先对原 Alpha 做轻微内缩（削去黑底 JPG 混入的最外圈 1px 死黑边），再做平滑抗锯齿
    a_smooth = gaussian_filter(a, sigma=1.4)
    a_clean = np.clip((a_smooth - 152.0) / (244.0 - 152.0), 0.0, 1.0) * 255.0

    # --- 3. 边缘色彩去黑污 + 清透淡紫描边润色（对标原生透底 PNG） ---
    dist_from_bg = distance_transform_edt(a_clean > 12.0)
    # 内侧干净彩色像素作为参考源（彻底排除死黑/暗灰边缘像素）
    interior_clean = (a_clean > 248.0) & ((mx >= 105.0) | (in_outfit_core & (mx >= 65.0)))
    edge_band = (a_clean > 0.0) & (dist_from_bg <= 8.0)

    if interior_clean.any() and edge_band.any():
        _, indices = distance_transform_edt(~interior_clean, return_indices=True)
        nr = r[indices[0], indices[1]]
        ng = g[indices[0], indices[1]]
        nb = b[indices[0], indices[1]]

        # 外缘 4px 内（768 层即 8px）任何暗于内侧的边缘像素大幅替换为内侧纯净色
        edge_prox = np.clip((8.0 - dist_from_bg) / 6.5, 0.0, 1.0)
        is_dark = np.clip((125.0 - mx) / 75.0, 0.0, 1.0)
        is_semi = np.clip((250.0 - a_clean) / 140.0, 0.0, 1.0)

        # 衣裙/靴子核心区只做轻度去黑边，头/发/猫耳/猫尾彻底去黑边
        strength = np.where(in_outfit_core, 0.45, 1.0)
        blend_w = np.where(edge_band, np.maximum(is_dark * edge_prox * strength, is_semi * 0.92), 0.0)
        blend_w = np.clip(blend_w, 0.0, 1.0)

        r = r * (1.0 - blend_w) + nr * blend_w
        g = g * (1.0 - blend_w) + ng * blend_w
        b = b * (1.0 - blend_w) + nb * blend_w

        # 非深色衣裙区（头/发/猫耳/猫尾）最外 6px 仍偏暗的描边柔和提亮至清透淡紫 (112, 92, 146)
        # 避开眼睛与眉毛内部区 (uy ∈ [134..186], |ux-192| < 46)
        in_face_features = (uy >= 134.0) & (uy <= 186.0) & (np.abs(ux - 192.0) <= 46.0)
        soft_outline = (
            (a_clean > 0.0)
            & (~in_outfit_core)
            & (~in_face_features)
            & (dist_from_bg <= 6.5)
            & (np.maximum(np.maximum(r, g), b) < 115.0)
        )
        if soft_outline.any():
            # 按原亮度比例柔和抬升，保留线稿层次感而非死板单色
            lift = np.clip((115.0 - np.maximum(np.maximum(r, g), b)) / 115.0, 0.0, 1.0) * 0.85
            r = np.where(soft_outline, r * (1.0 - lift) + 114.0 * lift, r)
            g = np.where(soft_outline, g * (1.0 - lift) + 94.0 * lift, g)
            b = np.where(soft_outline, b * (1.0 - lift) + 148.0 * lift, b)

    out = np.zeros((HR, HR, 4), dtype=np.uint8)
    out[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(a_clean, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def align_full_body_hr(img: Image.Image) -> Image.Image:
    """直接将高清原图单格对齐到 2x 超采样画布 (768x768)，并执行黑洞清理与边缘去黑污平滑。"""
    arr = np.array(img, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = max(1, y1 - y0 + 1)

    # 仅保留最大连通主体，清理外围星光
    clean = arr.copy()
    clean[~mask] = 0
    crop = Image.fromarray(clean[y0 : y1 + 1, x0 : x1 + 1], "RGBA")

    scale = HR_TARGET_H / float(h)
    nw = max(1, int(round(crop.width * scale)))
    nh = max(1, int(round(crop.height * scale)))
    resized = crop.resize((nw, nh), Image.LANCZOS)

    # 用重心 x 对齐画布中心
    r_arr = np.array(resized, dtype=np.uint8)
    r_mask = r_arr[:, :, 3] > 60
    r_xs = np.nonzero(r_mask)[1]
    cx = float(r_xs.mean()) if len(r_xs) > 0 else nw / 2.0

    paste_x = int(round(HR / 2.0 - cx))
    paste_y = HR_GROUND_Y - nh

    canvas = Image.new("RGBA", (HR, HR), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y))
    return clean_black_holes_and_defringe_hr(canvas)


def build_wink_frame_hr(base_open: Image.Image, base_blink: Image.Image) -> Image.Image:
    """在 768x768 超采样画布上无缝合成零色差、零残留的俏皮单眼 wink 帧（完整覆盖右眼及睫毛）。"""
    a = np.array(base_open, dtype=np.float32)
    b = np.array(base_blink, dtype=np.float32)

    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    mask_x = smoothstep(184.0 * SS, 194.0 * SS, xx) * (1.0 - smoothstep(236.0 * SS, 246.0 * SS, xx))
    mask_y = smoothstep(130.0 * SS, 140.0 * SS, yy) * (1.0 - smoothstep(174.0 * SS, 184.0 * SS, yy))
    w = (mask_x * mask_y)[:, :, None]

    out = a * (1.0 - w) + b * w
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def blend_face_only_hr(base_a: Image.Image, base_b: Image.Image, t: float) -> Image.Image:
    """
    仅对五官面部区域做高清平滑过渡（身体与外轮廓 100% 保持 base_a 刀锋锐利原画），
    彻底杜绝光流插值把人物外框拉糊！
    """
    if t <= 0.001:
        return base_a
    if t >= 0.999:
        return base_b
    a = np.array(base_a, dtype=np.float32)
    b = np.array(base_b, dtype=np.float32)

    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    # 仅覆盖双眼与嘴巴所在面部中心区域
    mask_x = smoothstep(145.0 * SS, 158.0 * SS, xx) * (1.0 - smoothstep(228.0 * SS, 241.0 * SS, xx))
    mask_y = smoothstep(134.0 * SS, 146.0 * SS, yy) * (1.0 - smoothstep(186.0 * SS, 198.0 * SS, yy))
    w = (mask_x * mask_y * t)[:, :, None]

    out = a * (1.0 - w) + b * w
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def warp_live2d_idle_hr(img_hr: Image.Image, phase: float) -> Image.Image:
    """
    在 768x768 (2x 超采样) 上执行零振铃双线性 (order=1) 网格变形，
    再通过 LANCZOS 一次性高质量降采样至 384x384：
    - 彻底消除三次样条振铃导致的边框发灰发糊
    - 保留刀锋般锐利的深紫描边 + 天然 1px 柔和抗锯齿
    """
    arr = np.array(img_hr, dtype=np.float32)
    alpha = arr[:, :, 3] / 255.0
    premul_rgb = arr[:, :, :3] * alpha[:, :, None]

    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    # 归一化到 384 坐标系方便参数一致
    ux = xx / float(SS)
    uy = yy / float(SS)

    # 1. 呼吸曲线：10 秒内 3 个柔和完整呼吸周期
    breath = 0.5 - 0.5 * math.cos(2.0 * math.pi * 3.0 * phase)
    breath_signed = math.sin(2.0 * math.pi * 3.0 * phase)

    # 2. 头部歪头与重心轻晃曲线
    left_tilt = bell_window(phase, 0.16, 0.45, ramp=0.09)
    right_tilt = bell_window(phase, 0.58, 0.87, ramp=0.09)
    head_angle_deg = -4.2 * left_tilt + 3.8 * right_tilt + 0.6 * math.sin(2.0 * math.pi * phase)
    head_angle = math.radians(head_angle_deg)
    sway_x = -2.2 * left_tilt + 2.0 * right_tilt + 0.6 * math.sin(2.0 * math.pi * phase)

    # 3. 猫耳轻抖微动
    ear_twitch = (
        bell_window(phase, 0.23, 0.30, ramp=0.025)
        + bell_window(phase, 0.68, 0.75, ramp=0.025)
    )
    ear_w = (1.0 - smoothstep(106.0, 126.0, uy)) * smoothstep(18.0, 40.0, np.abs(ux - 192.0))
    ear_dy = 2.6 * ear_twitch * ear_w
    ear_dx = -np.sign(ux - 192.0) * 1.4 * ear_twitch * ear_w

    # 部位平滑权重
    ground_free = 1.0 - smoothstep(295.0, 338.0, uy)
    head_w = 1.0 - smoothstep(184.0, 226.0, uy)
    torso_w = smoothstep(185.0, 215.0, uy) * (1.0 - smoothstep(295.0, 332.0, uy))
    hair_side = smoothstep(32.0, 68.0, np.abs(ux - 192.0)) * smoothstep(160.0, 210.0, uy) * (1.0 - smoothstep(295.0, 330.0, uy))
    tail_x = smoothstep(222.0, 250.0, ux)
    tail_y = smoothstep(242.0, 265.0, uy) * (1.0 - smoothstep(336.0, 355.0, uy))
    tail_w = tail_x * tail_y

    # 头部刚性旋转
    pivot_x, pivot_y = 192.0, 196.0
    dx_p = ux - pivot_x
    dy_p = uy - pivot_y
    cos_a = math.cos(-head_angle)
    sin_a = math.sin(-head_angle)
    rot_dx = (dx_p * cos_a - dy_p * sin_a) - dx_p
    rot_dy = (dx_p * sin_a + dy_p * cos_a) - dy_p

    breath_dy = 2.8 * breath * ground_free
    breath_dx = -(ux - 192.0) * (0.008 * breath) * torso_w
    sway_dx = -sway_x * ground_free

    hair_wave = 2.0 * math.sin(2.0 * math.pi * 2.0 * phase - 0.9) + 1.0 * breath_signed
    hair_dx = -hair_wave * hair_side * (1.0 - tail_w)
    hair_dy = -0.9 * math.cos(2.0 * math.pi * 2.0 * phase - 0.6) * hair_side * (1.0 - tail_w)

    tail_dist = np.clip((ux - 222.0) / 72.0, 0.0, 1.4)
    tail_phase = 2.0 * math.pi * 2.0 * phase - tail_dist * 1.1
    tail_dy = (6.2 * np.sin(tail_phase) + 1.8 * math.sin(2.0 * math.pi * 4.0 * phase)) * tail_w * tail_dist
    tail_dx = (2.2 * np.cos(tail_phase)) * tail_w * tail_dist

    src_x = (ux + rot_dx * head_w + breath_dx + sway_dx + hair_dx + tail_dx + ear_dx) * float(SS)
    src_y = (uy + rot_dy * head_w + breath_dy + hair_dy + tail_dy + ear_dy) * float(SS)

    coords = np.array([src_y, src_x])
    # 使用 order=1 双线性插值避免任何三次样条振铃，再靠 2x 超采样 LANCZOS 降采样获得极致清晰锐度
    out_alpha = np.clip(map_coordinates(alpha, coords, order=1, mode="constant", cval=0.0), 0.0, 1.0)
    out_rgb = np.zeros((HR, HR, 3), dtype=np.float32)
    for c in range(3):
        ch = map_coordinates(premul_rgb[:, :, c], coords, order=1, mode="constant", cval=0.0)
        out_rgb[:, :, c] = np.where(out_alpha > 1e-4, np.clip(ch / np.maximum(out_alpha, 1e-4), 0.0, 255.0), 0.0)

    # 在 2x 超采样层收紧 Alpha 边带 [75..215]，降采样到 384 时自动形成极锐利无发糊的 1px 抗锯齿描边
    out_a_u8 = out_alpha * 255.0
    clean_a = np.where(
        out_a_u8 < 75.0,
        0.0,
        np.where(out_a_u8 > 215.0, 255.0, (out_a_u8 - 75.0) / 140.0 * 255.0),
    )

    out_hr = np.zeros((HR, HR, 4), dtype=np.uint8)
    out_hr[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out_hr[:, :, 3] = np.clip(clean_a, 0, 255).astype(np.uint8)

    # 使用 Premultiplied 方式 LANCZOS 降采样到 384x384，保证半透明边缘色彩 100% 纯净不发灰
    hr_img = Image.fromarray(out_hr, "RGBA")
    lr_img = hr_img.resize((CANVAS, CANVAS), Image.LANCZOS)
    return lr_img


def draw_four_point_star(
    layer: Image.Image,
    cx: float,
    cy: float,
    r: float,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    opacity: float,
) -> None:
    """绘制华丽清晰的抗锯齿四芒星（带柔和外光晕 + 清晰深紫描边 + 亮白星核，白底黑底均醒目通透）。"""
    if opacity <= 0.03 or r < 2.0:
        return
    scale = 4
    pad = int(math.ceil(r * 2.2)) + 10
    sz = pad * 2 * scale
    tmp = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    c = pad * scale
    R = r * scale
    rin = R * 0.27

    # 柔和同色微光底晕（极轻，白底不发脏、黑底更璀璨）
    rg = R * 0.85
    draw.ellipse(
        (c - rg, c - rg, c + rg, c + rg),
        fill=fill_rgb + (int(55 * opacity),),
    )

    # 外描边稍粗一层确保白底清晰锐利
    R_out = R + 1.4 * scale
    rin_out = R_out * 0.29
    pts_out = [
        (c, c - R_out),
        (c + rin_out, c - rin_out),
        (c + R_out, c),
        (c + rin_out, c + rin_out),
        (c, c + R_out),
        (c - rin_out, c + rin_out),
        (c - R_out, c),
        (c - rin_out, c - rin_out),
    ]
    draw.polygon(pts_out, fill=outline_rgb + (int(230 * opacity),))

    pts = [
        (c, c - R),
        (c + rin, c - rin),
        (c + R, c),
        (c + rin, c + rin),
        (c, c + R),
        (c - rin, c + rin),
        (c - R, c),
        (c - rin, c - rin),
    ]
    draw.polygon(pts, fill=fill_rgb + (int(255 * opacity),))

    # 中心亮白星核
    rc = R * 0.34
    draw.ellipse((c - rc, c - rc, c + rc, c + rc), fill=(255, 255, 255, int(245 * opacity)))

    star_img = tmp.resize((pad * 2, pad * 2), Image.LANCZOS)
    ix, iy = int(round(cx - pad)), int(round(cy - pad))
    layer.alpha_composite(star_img, dest=(ix, iy))


def overlay_idle_sparkles(frame: Image.Image, phase: float) -> Image.Image:
    """
    在角色左右两侧安全区叠加贯穿 10 秒循环的华丽金粉四芒星与星尘特效：
    - 8 颗大中小错落星芒，按严格周期函数交错呼吸闪烁、轻柔上下悬浮
    - 第 0 帧与第 95 帧数学 100% 连续闭环，始终有精致星光环绕陪伴
    """
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    # (base_x, base_y, max_r, fill_rgb, freq_cycles, phase_offset, float_amp)
    specs = [
        # 左侧主星组（暖金大星 + 粉紫中星 + 亮金小星）
        (68.0, 148.0, 17.0, (255, 222, 110), 2.0, 0.00, 6.0),
        (54.0, 212.0, 13.5, (255, 168, 226), 2.0, 0.35, 5.0),
        (88.0, 102.0, 9.5, (255, 238, 160), 3.0, 0.60, 4.0),
        (66.0, 268.0, 8.5, (210, 172, 255), 3.0, 0.15, 4.5),
        # 右侧主星组（暖金大星 + 粉紫中星 + 冰紫小星）
        (314.0, 138.0, 18.0, (255, 222, 110), 2.0, 0.50, 6.5),
        (330.0, 204.0, 14.0, (255, 168, 226), 2.0, 0.85, 5.5),
        (294.0, 92.0, 9.5, (255, 238, 160), 3.0, 0.25, 4.0),
        (322.0, 264.0, 9.0, (206, 170, 255), 3.0, 0.70, 4.5),
    ]
    outline = (116, 82, 156)
    for bx, by, mr, col, freq, poff, famp in specs:
        # 严格周期性平滑呼吸闪烁（[0.08 .. 1.0] 交错明暗，永不生硬跳变）
        wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * (freq * phase + poff))
        op = 0.12 + 0.88 * (wave ** 1.35)
        r = mr * (0.48 + 0.52 * wave)
        cy = by + famp * math.sin(2.0 * math.pi * (phase + poff))
        draw_four_point_star(layer, bx, cy, r, col, outline, op)

    out = frame.copy()
    out.alpha_composite(layer)
    return out


def main() -> dict:
    s1 = Image.open("gemini/素材/01idle/part1.jpg").convert("RGBA")
    b1 = find_cell_boxes(s1, 4, 3)

    p1_0 = clean_black_bg_rich_particles(s1.crop(b1[0]), bg_thresh=24, hole_thresh=18, glow_cutoff=22)
    p1_8 = clean_black_bg_rich_particles(s1.crop(b1[8]), bg_thresh=24, hole_thresh=18, glow_cutoff=22)
    p1_10 = clean_black_bg_rich_particles(s1.crop(b1[10]), bg_thresh=24, hole_thresh=18, glow_cutoff=22)

    # 全部直接对齐到 2x 超采样画布 (768x768)，保留原画刀锋锐利外框
    base_open = align_full_body_hr(p1_0)
    base_blink = align_full_body_hr(p1_8)
    base_wink = build_wink_frame_hr(base_open, base_blink)
    base_yawn = align_full_body_hr(p1_10)

    # 面部局部高清平滑过渡（身体与外轮廓 100% 锁定 base_open 原始锐度，绝不拉糊外框）
    open_to_blink = [blend_face_only_hr(base_open, base_blink, 0.35), blend_face_only_hr(base_open, base_blink, 0.70)]
    open_to_wink = [blend_face_only_hr(base_open, base_wink, 0.35), blend_face_only_hr(base_open, base_wink, 0.70)]
    open_to_yawn = [blend_face_only_hr(base_open, base_yawn, 0.35), blend_face_only_hr(base_open, base_yawn, 0.70)]

    def pick_expression_base(idx: int) -> Image.Image:
        # 第 1 次自然眨眼 (12..17)
        if idx == 12:
            return open_to_blink[0]
        if idx == 13:
            return open_to_blink[1]
        if idx in (14, 15):
            return base_blink
        if idx == 16:
            return open_to_blink[1]
        if idx == 17:
            return open_to_blink[0]

        # 左歪头短暂俏皮单眼 wink (26..33)
        if idx == 26:
            return open_to_wink[0]
        if idx == 27:
            return open_to_wink[1]
        if 28 <= idx <= 31:
            return base_wink
        if idx == 32:
            return open_to_wink[1]
        if idx == 33:
            return open_to_wink[0]

        # 第 2 次自然眨眼 (48..53)
        if idx == 48:
            return open_to_blink[0]
        if idx == 49:
            return open_to_blink[1]
        if idx in (50, 51):
            return base_blink
        if idx == 52:
            return open_to_blink[1]
        if idx == 53:
            return open_to_blink[0]

        # 可爱轻哈欠 (55..62)
        if idx == 55:
            return open_to_yawn[0]
        if idx == 56:
            return open_to_yawn[1]
        if 57 <= idx <= 60:
            return base_yawn
        if idx == 61:
            return open_to_yawn[1]
        if idx == 62:
            return open_to_yawn[0]

        # 第 3 次自然眨眼 (80..85)
        if idx == 80:
            return open_to_blink[0]
        if idx == 81:
            return open_to_blink[1]
        if idx in (82, 83):
            return base_blink
        if idx == 84:
            return open_to_blink[1]
        if idx == 85:
            return open_to_blink[0]

        return base_open

    frames_96: List[Image.Image] = []
    for i in range(TOTAL_FRAMES):
        phase = i / float(TOTAL_FRAMES)
        expr_hr = pick_expression_base(i)
        deformed_lr = warp_live2d_idle_hr(expr_hr, phase)
        final_fr = overlay_idle_sparkles(deformed_lr, phase)
        frames_96.append(final_fr)

    assert len(frames_96) == TOTAL_FRAMES

    heights = []
    for fr in frames_96:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 12
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_96):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS))

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "01idle_sheet（待机_图集）.webp")
    anim_path = os.path.join(out_dir, "01idle（待机_动画).webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    durations: List[int] = []
    for i in range(TOTAL_FRAMES):
        if (12 <= i <= 17) or (48 <= i <= 53) or (80 <= i <= 85):
            durations.append(92)
        elif (28 <= i <= 31) or (57 <= i <= 60):
            durations.append(112)
        else:
            durations.append(104)

    diff = 10000 - sum(durations)
    adjust_indices = [i for i in range(TOTAL_FRAMES) if durations[i] == 104]
    step_sign = 1 if diff > 0 else -1
    for k in range(abs(diff)):
        durations[adjust_indices[k % len(adjust_indices)]] += step_sign

    assert sum(durations) == 10000

    frames_96[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_96[1:],
        duration=durations,
        loop=0,
        lossless=True,
        quality=100,
        disposal=2,
    )

    white_bg = Image.new("RGBA", atlas.size, (255, 255, 255, 255))
    white_bg.alpha_composite(atlas)
    white_path = os.path.join(prev_dir, "01idle_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "01idle_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "01idle",
        "frames": len(frames_96),
        "grid": f"{cols}x{rows}",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "height_min": min(heights),
        "height_max": max(heights),
        "total_ms": sum(durations),
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
