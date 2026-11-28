import importlib.util
import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expand_48 import get_main_component_mask, pyramidal_optical_flow

_spec17 = importlib.util.spec_from_file_location("w17", "gemini/脚本/17build_wave_hello.py")
_w17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(_w17)
find_white_grid_boxes = _w17.find_white_grid_boxes
clean_white_bg = _w17.clean_white_bg

_spec18 = importlib.util.spec_from_file_location("n18", "gemini/脚本/18process_note_taking.py")
_n18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(_n18)
measure_head_metrics = _n18.measure_head_metrics

CANVAS = 384
SS = 2  # 2x 超采样画布 (768x768)
HR = CANVAS * SS
GROUND_Y = 366
HR_GROUND_Y = GROUND_Y * SS
TARGET_H_0 = 308
TOTAL_FRAMES = 80


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def clean_white_crisp(img_rgba: Image.Image) -> Image.Image:
    """
    纯白底高精抠图：
    1. 四边泛洪去白底
    2. 精准清理外侧大发缝与双腿下方纯白孔洞（严密保护白色袖口与百褶裙蕾丝）
    3. 基于白底反预乘还原边界 RGB 并压制亮灰白边，双底色 100% 锐利零毛边
    """
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.float32)
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mn = np.minimum(np.minimum(r, g), b)
    mx = np.maximum(np.maximum(r, g), b)
    chroma = mx - mn

    base = clean_white_bg(img_rgba, bg_thresh=236)
    barr = np.array(base, dtype=np.float32)
    alpha = barr[:, :, 3]

    is_pure_w = (alpha > 0) & (mn >= 249.0) & (chroma <= 6.0)
    lbl, n = label(is_pure_w)
    if n > 0:
        fg_ys, fg_xs = np.nonzero(alpha > 40)
        cx = float(np.median(fg_xs))
        y_top, y_bot = int(fg_ys.min()), int(fg_ys.max())
        body_h = y_bot - y_top
        counts = np.bincount(lbl.ravel())
        for cid in range(1, n + 1):
            if counts[cid] < 10:
                continue
            ys_c, xs_c = np.nonzero(lbl == cid)
            ym, xm = float(ys_c.mean()), float(xs_c.mean())
            in_outer_hair_gap = (abs(xm - cx) > 0.27 * w) and (ym > y_top + 0.30 * body_h)
            in_leg_gap = ym > y_top + 0.74 * body_h
            if in_outer_hair_gap or in_leg_gap:
                alpha[lbl == cid] = 0.0

    dist = distance_transform_edt(alpha > 8.0)
    edge = (alpha > 0.0) & (dist <= 2.2)
    fg_solid = (alpha > 240.0) & (dist > 2.2)
    if fg_solid.any() and edge.any():
        _, idxs = distance_transform_edt(~fg_solid, return_indices=True)
        nr = barr[:, :, 0][idxs[0], idxs[1]]
        ng = barr[:, :, 1][idxs[0], idxs[1]]
        nb = barr[:, :, 2][idxs[0], idxs[1]]

        denom = np.maximum(255.0 - np.minimum(np.minimum(nr, ng), nb), 35.0)
        num = np.maximum(255.0 - mn, 0.0)
        a_est = np.clip(num / denom, 0.0, 1.0)
        is_bright_fringe = edge & (mn > 175.0) & (chroma < 48.0)
        alpha = np.where(is_bright_fringe, np.minimum(alpha, a_est * 255.0), alpha)

        decontam_w = np.where(edge & (mn > 160.0), np.clip((mn - 160.0) / 95.0, 0.0, 0.92), 0.0)
        barr[:, :, 0] = barr[:, :, 0] * (1.0 - decontam_w) + nr * decontam_w
        barr[:, :, 1] = barr[:, :, 1] * (1.0 - decontam_w) + ng * decontam_w
        barr[:, :, 2] = barr[:, :, 2] * (1.0 - decontam_w) + nb * decontam_w

    barr[:, :, 3] = np.clip(alpha, 0.0, 255.0)
    return Image.fromarray(barr.astype(np.uint8), "RGBA")


def align_by_head_hr(c: Image.Image, ref_hw_hr: float) -> Image.Image:
    """严格 1:1 头宽比例对齐至 768x768 超采样画布，双脚锚定 HR_GROUND_Y。"""
    hw_i, _, _, _ = measure_head_metrics(c, body_ratio=1.0)
    sc = ref_hw_hr / max(1.0, hw_i)
    nw, nh = max(1, int(round(c.width * sc))), max(1, int(round(c.height * sc)))
    rz = c.resize((nw, nh), Image.Resampling.LANCZOS)
    _, hcx, _, my1 = measure_head_metrics(rz, body_ratio=1.0)
    px = int(round(HR * 0.5 - hcx))
    py = int(round(HR_GROUND_Y - my1))
    cv = Image.new("RGBA", (HR, HR), (0, 0, 0, 0))
    cv.paste(rz, (px, py), rz)
    return cv


def warp_rgba_hr(img_arr: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """在 768x768 超采样画布执行 Premultiplied Alpha 双线性形变。"""
    h, w, _ = img_arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.array([yy + dy, xx + dx])

    alpha = img_arr[:, :, 3].astype(np.float32) / 255.0
    premul = img_arr[:, :, :3].astype(np.float32) * alpha[:, :, None]

    wa = np.clip(map_coordinates(alpha, coords, order=1, mode="constant", cval=0.0), 0.0, 1.0)
    out = np.zeros((h, w, 4), dtype=np.float32)
    for c in range(3):
        wc = map_coordinates(premul[:, :, c], coords, order=1, mode="constant", cval=0.0)
        out[:, :, c] = np.where(wa > 1e-4, np.clip(wc / np.maximum(wa, 1e-4), 0.0, 255.0), 0.0)
    out[:, :, 3] = np.clip(wa * 255.0, 0.0, 255.0)
    return out


def interpolate_flow_hr(img_a: Image.Image, img_b: Image.Image, ts: List[float]) -> List[Image.Image]:
    """
    在 768x768 (2x 超采样) 上执行 5 层金字塔高精双向光流插帧：
    - 5 层金字塔覆盖长达 ~40px 的大幅身体下沉位移
    - 不一致门控 strictly follows time-nearest frame (1.0 if t<=0.5 else 0.0)，
      彻底杜绝“并集鬼影”（绝不会同时画出高低两对猫耳！）
    """
    a = np.array(img_a, dtype=np.uint8)
    b = np.array(img_b, dtype=np.uint8)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        al = arr[:, :, 3].astype(np.float32) / 255.0
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        return lum * al + 0.48 * al

    fa, fb = feature_map(a), feature_map(b)
    vx_ab, vy_ab = pyramidal_optical_flow(fa, fb, levels=5, win_sigma=5.0, iters=7)
    vx_ba, vy_ba = pyramidal_optical_flow(fb, fa, levels=5, win_sigma=5.0, iters=7)

    out_frames: List[Image.Image] = []
    for t in ts:
        wa = warp_rgba_hr(a, -t * vx_ab, -t * vy_ab)
        wb = warp_rgba_hr(b, -(1.0 - t) * vx_ba, -(1.0 - t) * vy_ba)

        rgb_diff = np.abs(wa[:, :, :3] - wb[:, :, :3]).mean(axis=2) / 255.0
        alpha_diff = np.abs(wa[:, :, 3] - wb[:, :, 3]) / 255.0
        disagree = rgb_diff * 0.8 + alpha_diff * 1.1
        agree_w = np.clip(1.0 - disagree * 3.2, 0.0, 1.0)

        # 关键防双耳鬼影规则：光流未重合的不一致区域严格选择时间最近的单帧，绝不做并集！
        nearest_a = 1.0 if t <= 0.5 else 0.0
        w_a = (1.0 - t) * agree_w + nearest_a * (1.0 - agree_w)
        w_b = 1.0 - w_a

        out = np.zeros_like(wa, dtype=np.float32)
        out[:, :, :3] = wa[:, :, :3] * w_a[:, :, None] + wb[:, :, :3] * w_b[:, :, None]
        out_a = wa[:, :, 3] * w_a + wb[:, :, 3] * w_b

        out[:, :, 3] = np.where(
            out_a < 55.0,
            0.0,
            np.where(out_a > 215.0, 255.0, (out_a - 55.0) / 160.0 * 255.0),
        )
        out_frames.append(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA"))
    return out_frames


def get_body_top_lr(img_hr: Image.Image) -> float:
    arr = np.array(img_hr, dtype=np.uint8)
    mask = get_main_component_mask(arr[:, :, 3])
    ys = np.nonzero(mask)[0]
    return float(ys.min()) / float(SS)


def make_gradual_crouch_pre_hr(base_stand_hr: Image.Image, target_crouch_hr: Image.Image) -> Image.Image:
    """
    生成一个“半下沉过渡桥接帧”：把站姿或吸气帧的躯干平稳向下屈膝压低 18px，
    并轻度向外舒展双耳（纯刚性绕耳根旋转），从而把一次 37px 的巨幅跳跃拆成两步小跨度光流，
    让耳朵和双臂 100% 丝滑跟踪下沉，绝不分裂！
    """
    top_y = get_body_top_lr(base_stand_hr)
    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    ux, uy = xx / float(SS), yy / float(SS)

    # 躯干与头部平稳向下沉 18px，双脚牢牢抓地
    ground_free = 1.0 - smoothstep(295.0, 342.0, uy)
    sink_dy = -18.0 * ground_free

    # 绕左右耳根轻度外展旋转 (±12°)，衔接 K2 飞机耳
    ear_l_w = (1.0 - smoothstep(top_y + 42.0, top_y + 74.0, uy)) * (1.0 - smoothstep(168.0, 186.0, ux))
    ear_r_w = (1.0 - smoothstep(top_y + 42.0, top_y + 74.0, uy)) * smoothstep(200.0, 218.0, ux)

    plx, ply = 166.0, top_y + 56.0
    prx, pry = 220.0, top_y + 56.0
    al = math.radians(-14.0)
    ar = math.radians(14.0)

    dlx, dly = ux - plx, uy - ply
    rot_ldx = (dlx * math.cos(-al) - dly * math.sin(-al) - dlx) * ear_l_w
    rot_ldy = (dlx * math.sin(-al) + dly * math.cos(-al) - dly) * ear_l_w

    drx, dry = ux - prx, uy - pry
    rot_rdx = (drx * math.cos(-ar) - dry * math.sin(-ar) - drx) * ear_r_w
    rot_rdy = (drx * math.sin(-ar) + dry * math.cos(-ar) - dry) * ear_r_w

    warped = warp_rgba_hr(
        np.array(base_stand_hr, dtype=np.uint8),
        (rot_ldx + rot_rdx) * float(SS),
        (sink_dy + rot_ldy + rot_rdy) * float(SS),
    )
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")


def make_wink_pose_hr(base_open_hr: Image.Image, base_closed_hr: Image.Image) -> Image.Image:
    """
    合出零残影左金眼+右弯睫毛 Wink，同时施加以颈部为轴的左歪头 (-9.0°) + 右耳纯旋转轻压 (+8°)。
    """
    top_open = get_body_top_lr(base_open_hr)
    top_cls = get_body_top_lr(base_closed_hr)
    dy_align = (top_cls - top_open) * float(SS)

    a = np.array(base_open_hr, dtype=np.float32)
    b_arr = warp_rgba_hr(
        np.array(base_closed_hr, dtype=np.uint8),
        np.zeros((HR, HR), dtype=np.float32),
        np.full((HR, HR), dy_align, dtype=np.float32),
    )

    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    ux, uy = xx / float(SS), yy / float(SS)

    eye_cy = top_open + 96.0
    mx = smoothstep(184.0, 193.0, ux) * (1.0 - smoothstep(234.0, 243.0, ux))
    my = smoothstep(eye_cy - 22.0, eye_cy - 14.0, uy) * (1.0 - smoothstep(eye_cy + 15.0, eye_cy + 23.0, uy))
    w = (mx * my)[:, :, None]
    wink_raw = np.clip(a * (1.0 - w) + b_arr * w, 0, 255).astype(np.uint8)

    pivot_x, pivot_y = 192.0, top_open + 132.0
    head_w = 1.0 - smoothstep(pivot_y - 12.0, pivot_y + 30.0, uy)
    ang = math.radians(-9.0)
    dx_p, dy_p = ux - pivot_x, uy - pivot_y
    rot_dx = (dx_p * math.cos(-ang) - dy_p * math.sin(-ang) - dx_p) * head_w
    rot_dy = (dx_p * math.sin(-ang) + dy_p * math.cos(-ang) - dy_p) * head_w

    # 右猫耳绕耳根保形状纯旋转微折 (+8°)
    prx, pry = 220.0, top_open + 54.0
    ear_r_w = (1.0 - smoothstep(top_open + 42.0, top_open + 66.0, uy)) * smoothstep(200.0, 218.0, ux)
    ar = math.radians(8.0)
    drx, dry = ux - prx, uy - pry
    ear_rdx = (drx * math.cos(-ar) - dry * math.sin(-ar) - drx) * ear_r_w
    ear_rdy = (drx * math.sin(-ar) + dry * math.cos(-ar) - dry) * ear_r_w

    warped = warp_rgba_hr(wink_raw, (rot_dx + ear_rdx) * float(SS), (rot_dy + ear_rdy) * float(SS))
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")


def make_right_tilt_ear_pose_hr(base_open_hr: Image.Image) -> Image.Image:
    """
    向右歪头 (+8.5°) + 右耳纯刚性旋转压耳 (+16°，绝不压扁变形耳朵！) + 轻度身体弹性下沉 (14px)。
    """
    top_y = get_body_top_lr(base_open_hr)
    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    ux, uy = xx / float(SS), yy / float(SS)

    # 先绕颈部做保形右歪头旋转
    pivot_x, pivot_y = 192.0, top_y + 136.0
    head_w = 1.0 - smoothstep(pivot_y - 14.0, pivot_y + 30.0, uy)
    ang = math.radians(8.5)
    dx_p, dy_p = ux - pivot_x, uy - pivot_y
    rot_dx = (dx_p * math.cos(-ang) - dy_p * math.sin(-ang) - dx_p) * head_w
    rot_dy = (dx_p * math.sin(-ang) + dy_p * math.cos(-ang) - dy_p) * head_w

    # 再绕右耳根纯旋转压耳（保持整个耳廓饱满完整，丝毫不缺角、不发糊）
    prx, pry = 222.0, top_y + 52.0
    ear_r_w = (1.0 - smoothstep(top_y + 38.0, top_y + 62.0, uy)) * smoothstep(200.0, 222.0, ux)
    ar = math.radians(15.5)
    drx, dry = ux - prx, uy - pry
    ear_rdx = (drx * math.cos(-ar) - dry * math.sin(-ar) - drx) * ear_r_w
    ear_rdy = (drx * math.sin(-ar) + dry * math.cos(-ar) - dry) * ear_r_w

    # 身体轻度弹性下沉 14px
    ground_free = 1.0 - smoothstep(298.0, 342.0, uy)
    sink_dy = -14.0 * ground_free

    warped = warp_rgba_hr(
        np.array(base_open_hr, dtype=np.uint8),
        (rot_dx + ear_rdx) * float(SS),
        (rot_dy + ear_rdy + sink_dy) * float(SS),
    )
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")


def apply_continuous_live2d_motion_hr(img_hr: Image.Image, phase: float) -> Image.Image:
    """
    全程 80 帧连续施加大猫尾大弧度波浪摇摆 + 柔风长发微摆 + 耳尖微颤与轻量辅助呼吸，
    再一次性 LANCZOS 降采样至 384x384 锁定锋利线条。
    """
    top_y = get_body_top_lr(img_hr)
    tail_base_y = min(312.0, top_y + 228.0)

    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    ux, uy = xx / float(SS), yy / float(SS)

    tail_x_w = smoothstep(224.0, 246.0, ux)
    tail_y_w = smoothstep(tail_base_y - 34.0, tail_base_y - 12.0, uy) * (1.0 - smoothstep(tail_base_y + 42.0, tail_base_y + 56.0, uy))
    tail_w = tail_x_w * tail_y_w

    pivot_tx, pivot_ty = 224.0, tail_base_y + 8.0
    tail_dist = np.clip((ux - 224.0) / 68.0, 0.0, 1.35)
    t_wave = 2.0 * math.pi * 4.0 * phase - tail_dist * 0.75
    tail_ang = math.radians(16.0 * math.sin(2.0 * math.pi * 4.0 * phase) + 4.0 * math.sin(2.0 * math.pi * 2.0 * phase))
    tx_p, ty_p = ux - pivot_tx, uy - pivot_ty
    t_rot_dx = (tx_p * math.cos(-tail_ang) - ty_p * math.sin(-tail_ang) - tx_p) * tail_w
    t_rot_dy = (tx_p * math.sin(-tail_ang) + ty_p * math.cos(-tail_ang) - ty_p) * tail_w
    tail_tip_dy = (7.5 * np.sin(t_wave)) * tail_w * tail_dist
    tail_tip_dx = (3.5 * np.cos(t_wave)) * tail_w * tail_dist

    hair_side = (
        smoothstep(34.0, 72.0, np.abs(ux - 192.0))
        * smoothstep(top_y + 92.0, top_y + 145.0, uy)
        * (1.0 - smoothstep(292.0, 330.0, uy))
        * (1.0 - tail_w)
    )
    hair_wave = 2.8 * math.sin(2.0 * math.pi * 3.0 * phase - 0.8)
    hair_dx = -hair_wave * hair_side
    hair_dy = -1.1 * math.cos(2.0 * math.pi * 3.0 * phase - 0.5) * hair_side

    ear_micro = 1.6 * math.sin(2.0 * math.pi * 6.0 * phase)
    ear_w = (1.0 - smoothstep(top_y + 38.0, top_y + 60.0, uy)) * smoothstep(20.0, 44.0, np.abs(ux - 192.0))
    ear_dy = -ear_micro * ear_w * np.where(ux > 192.0, 1.0, 0.55)

    ground_free = 1.0 - smoothstep(298.0, 338.0, uy)
    micro_breath_dy = -1.6 * math.sin(2.0 * math.pi * 3.0 * phase) * ground_free

    dx_total = (t_rot_dx + tail_tip_dx + hair_dx) * float(SS)
    dy_total = (t_rot_dy + tail_tip_dy + hair_dy + ear_dy + micro_breath_dy) * float(SS)

    warped_hr = warp_rgba_hr(np.array(img_hr, dtype=np.uint8), dx_total, dy_total)
    wa = warped_hr[:, :, 3]
    warped_hr[:, :, 3] = np.where(wa < 22.0, 0.0, np.where(wa > 232.0, 255.0, (wa - 22.0) / 210.0 * 255.0))
    hr_img = Image.fromarray(np.clip(warped_hr, 0, 255).astype(np.uint8), "RGBA")
    return hr_img.resize((CANVAS, CANVAS), Image.Resampling.LANCZOS)


def draw_twinkle_star(
    layer: Image.Image,
    cx: float,
    cy: float,
    r: float,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    opacity: float,
    with_cross: bool = True,
) -> None:
    """绘制一闪一闪的高清抗锯齿四芒星（带细腻深紫描边、柔和亮晕与白芯闪光十字）。"""
    if opacity <= 0.04 or r < 1.8:
        return
    scale = 4
    pad = int(math.ceil(r * 2.4)) + 10
    sz = pad * 2 * scale
    tmp = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    c = pad * scale
    R = r * scale
    rin = R * 0.25

    rg = R * 0.78
    draw.ellipse((c - rg, c - rg, c + rg, c + rg), fill=fill_rgb + (int(56 * opacity),))

    if with_cross and opacity > 0.50:
        ray_len = R * (1.32 + 0.28 * (opacity - 0.50))
        lw = max(2, int(round(1.2 * scale)))
        ray_a = int(225 * ((opacity - 0.50) / 0.50))
        draw.line([(c - ray_len, c), (c + ray_len, c)], fill=(255, 255, 255, ray_a), width=lw)
        draw.line([(c, c - ray_len), (c, c + ray_len)], fill=(255, 255, 255, ray_a), width=lw)

    R_out = R + 1.35 * scale
    rin_out = R_out * 0.27
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
    draw.polygon(pts_out, fill=outline_rgb + (int(242 * opacity),))

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

    rc = R * 0.34
    draw.ellipse((c - rc, c - rc, c + rc, c + rc), fill=(255, 255, 255, int(252 * opacity)))

    star_img = tmp.resize((pad * 2, pad * 2), Image.Resampling.LANCZOS)
    ix, iy = int(round(cx - pad)), int(round(cy - pad))
    layer.alpha_composite(star_img, dest=(ix, iy))


def overlay_twinkling_stars(frame: Image.Image, phase: float) -> Image.Image:
    """12 颗错落分布的星星严格按多频率周期一闪一闪闪烁与轻盈浮动。"""
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    stars = [
        (58.0, 124.0, 15.0, (255, 222, 88), 4, 0.00, 5.0, True),
        (44.0, 198.0, 12.0, (255, 156, 220), 5, 0.38, 4.5, True),
        (86.0, 80.0, 9.5, (255, 236, 150), 6, 0.65, 3.5, False),
        (56.0, 266.0, 10.5, (196, 158, 255), 4, 0.52, 4.0, True),
        (36.0, 154.0, 7.0, (255, 242, 180), 7, 0.21, 3.0, False),
        (82.0, 314.0, 8.0, (255, 168, 228), 5, 0.82, 3.5, False),
        (324.0, 116.0, 15.5, (255, 222, 88), 4, 0.25, 5.5, True),
        (340.0, 192.0, 12.5, (255, 156, 220), 5, 0.72, 4.5, True),
        (296.0, 76.0, 9.8, (255, 236, 150), 6, 0.12, 3.5, False),
        (334.0, 260.0, 10.0, (196, 158, 255), 4, 0.88, 4.0, True),
        (350.0, 148.0, 7.2, (255, 242, 180), 7, 0.54, 3.0, False),
        (336.0, 316.0, 8.0, (255, 220, 96), 5, 0.44, 3.5, False),
    ]
    outline = (102, 68, 144)
    for bx, by, mr, col, freq, poff, famp, cross in stars:
        u = (freq * phase + poff) % 1.0
        flash = math.pow(max(0.0, math.sin(math.pi * u)), 2.3)
        op = 0.06 + 0.94 * flash
        r = mr * (0.30 + 0.70 * flash)
        cy = by + famp * math.sin(2.0 * math.pi * (phase + poff))
        draw_twinkle_star(layer, bx, cy, r, col, outline, op, with_cross=cross)

    out = frame.copy()
    out.alpha_composite(layer)
    return out


def main() -> dict:
    sheet = Image.open("gemini/素材/01idle_v4/part1_white.jpg").convert("RGBA")
    boxes = find_white_grid_boxes(sheet, cols=4, rows=2, thresh=242)
    cells = [clean_white_crisp(sheet.crop(b)) for b in boxes]

    arr0 = np.array(cells[0], dtype=np.uint8)
    m0 = get_main_component_mask(arr0[:, :, 3])
    ys0 = np.nonzero(m0)[0]
    h0 = int(ys0.max() - ys0.min() + 1)
    hw0, _, _, _ = measure_head_metrics(cells[0], body_ratio=1.0)
    ref_hw_hr = hw0 * ((TARGET_H_0 * float(SS)) / max(1, h0))

    aligned_hr = [align_by_head_hr(c, ref_hw_hr) for c in cells]

    k0 = aligned_hr[0]  # 标准站姿 :3
    k1 = aligned_hr[1]  # 吸气微抬+举拳虎牙笑
    k2 = aligned_hr[2]  # 大幅度屈膝身体往下沉 + 双耳外折飞机耳
    k3 = aligned_hr[3]  # 保持下沉 + 慵懒半眯眼 + 双手轻拢
    k4 = aligned_hr[4]  # 下沉抱胸 + 左歪头 + 左单耳压折 + 鼓脸无辜盯
    k5 = aligned_hr[5]  # 弹性回弹起身 + 开心虎牙笑
    k6 = aligned_hr[6]  # 俏皮回正 :3
    k7 = aligned_hr[7]  # 温柔闭眼呼吸休息

    # 两项纯保形高清派生：半下沉桥接帧(彻底消除K1->K2耳朵跳跃双影)、左歪头Wink、右歪头右折耳
    k_crouch_mid = make_gradual_crouch_pre_hr(k0, k2)
    k_wink = make_wink_pose_hr(k6, k7)
    k_rtilt = make_right_tilt_ear_pose_hr(k0)

    # 12 段高精过渡 -> 共计 80 帧闭环 (8x10)
    segments = [
        (k0, k1, 4),           # 0: 站姿 -> 吸气抬拳虎牙笑 (5)
        (k1, k0, 3),           # 1: 吸气微收准备下沉 (4)
        (k0, k_crouch_mid, 3), # 2: 屈膝开始下沉+外展耳朵 (4)
        (k_crouch_mid, k2, 4), # 3: 沉到底部+飞机耳舒展 (5)
        (k2, k3, 5),           # 4: 保持下沉 -> 慵懒半眯眼 (6)
        (k3, k4, 6),           # 5: 下沉 -> 抱胸左歪头+左单耳折耳 (7)
        (k4, k5, 6),           # 6: 蹲姿 -> Q弹起身回正虎牙笑 (7)
        (k5, k6, 4),           # 7: 虎牙笑 -> 回正:3微笑 (5)
        (k6, k_wink, 5),       # 8: 回正:3 -> 左歪头+右眼俏皮Wink+右耳微抖 (6)
        (k_wink, k6, 4),       # 9: Wink -> 回正:3 (5)
        (k6, k_rtilt, 6),      # 10: 回正 -> 右大歪头+右耳折耳+轻度二次下沉 (7)
        (k_rtilt, k7, 6),      # 11: 右歪头 -> 温柔闭眼深呼吸休息 (7)
        (k7, k0, 11),          # 12: 闭眼休息 -> 睁眼优雅回正闭环 (12 -> 总计 80 帧!)
    ]
    assert sum(1 + n for _, _, n in segments) == TOTAL_FRAMES

    frames_hr: List[Image.Image] = []
    key_indices: List[int] = []
    for fa, fb, n_mid in segments:
        key_indices.append(len(frames_hr))
        frames_hr.append(fa)
        ts = [(m + 1) / float(n_mid + 1) for m in range(n_mid)]
        frames_hr.extend(interpolate_flow_hr(fa, fb, ts))

    assert len(frames_hr) == TOTAL_FRAMES

    frames_80: List[Image.Image] = []
    for i, fr_hr in enumerate(frames_hr):
        phase = i / float(TOTAL_FRAMES)
        moved_lr = apply_continuous_live2d_motion_hr(fr_hr, phase)
        final_lr = overlay_twinkling_stars(moved_lr, phase)
        frames_80.append(final_lr)

    heights: List[int] = []
    top_ys: List[int] = []
    for fr in frames_80:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))
        top_ys.append(int(ys.min()))

    cols, rows = 8, 10
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_80):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS))

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "01idle_sheet（待机_图集）.webp")
    anim_path = os.path.join(out_dir, "01idle（待机_动画).webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 变速呼吸节奏 (严格 9200ms): 蹲沉停顿、左歪头、Wink、右歪头、闭眼各留足呼吸节奏
    hold_set = {0, 17, 18, 19, 23, 24, 25, 30, 31, 32, 47, 48, 49, 58, 59, 60, 66, 67, 68}
    fast_set = set(range(9, 17)) | set(range(33, 39)) | set(range(50, 54))
    durations = [135 if i in hold_set else (95 if i in fast_set else 112) for i in range(TOTAL_FRAMES)]

    target_ms = 9200
    diff = target_ms - sum(durations)
    adj_idxs = [i for i in range(TOTAL_FRAMES) if durations[i] == 112]
    sign = 1 if diff > 0 else -1
    for k in range(abs(diff)):
        durations[adj_idxs[k % len(adj_idxs)]] += sign
    assert sum(durations) == target_ms

    frames_80[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_80[1:],
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
        "frames": len(frames_80),
        "grid": f"{cols}x{rows}",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "height_min": min(heights),
        "height_max": max(heights),
        "sink_range_px": max(top_ys) - min(top_ys),
        "key_indices": key_indices,
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
