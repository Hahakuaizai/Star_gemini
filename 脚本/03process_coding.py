import importlib.util
import json
import os
import sys
from typing import List

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_webp import find_cell_boxes
from expand_48 import get_main_component_mask

_spec_swp = importlib.util.spec_from_file_location("swp", "gemini/脚本/02process_sweeping.py")
_swp = importlib.util.module_from_spec(_spec_swp)
_spec_swp.loader.exec_module(_swp)
clean_black_bg_rich_particles = _swp.clean_black_bg_rich_particles
remove_nebula_blobs_aligned = _swp.remove_nebula_blobs_aligned
interpolate_smooth_no_burr = _swp.interpolate_smooth_no_burr

_spec_18 = importlib.util.spec_from_file_location("n18", "gemini/脚本/18process_note_taking.py")
_n18 = importlib.util.module_from_spec(_spec_18)
_spec_18.loader.exec_module(_n18)
align_full_body = _n18.align_full_body
align_cropped_with_ref_boots = _n18.align_cropped_with_ref_boots

CANVAS = 384


def interp_3(a: Image.Image, b: Image.Image) -> List[Image.Image]:
    """使用双向一致光流递归中点插值，在两帧间平滑生成 3 个中间过渡帧。"""
    mid = interpolate_smooth_no_burr(a, b)
    q1 = interpolate_smooth_no_burr(a, mid)
    q3 = interpolate_smooth_no_burr(mid, b)
    return [q1, mid, q3]


def main() -> dict:
    s1 = Image.open("gemini/素材/03coding/part1_v2.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/03coding/part2_v2.jpg").convert("RGBA")

    b1 = find_cell_boxes(s1, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    p1 = [clean_black_bg_rich_particles(s1.crop(b1[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    # 以新 Part 1 首帧作为全局绝对站姿基准（身高 304px，脚底 GROUND_Y=366）
    ref_stand = remove_nebula_blobs_aligned(align_full_body(p1[0]))

    # Part 1 关键帧重排：交换 p1[3] 与 p1[4] 使“双手准备召唤 -> 笔记本出现”顺序正确，并跳过笔记本在右侧的 p1[5]
    p1_order = [0, 1, 2, 4, 3, 6, 7, 8, 9, 10, 11]
    a1: List[Image.Image] = []
    for idx in p1_order:
        if idx == 0:
            a1.append(ref_stand)
        else:
            a1.append(remove_nebula_blobs_aligned(align_full_body(p1[idx])))

    # Part 2 关键帧（跳过换手拿眼镜的 p2[8]，尾帧直接使用 ref_stand 保证 100% 首尾闭环）
    # Row 1/2 (p2[0..7]) 裁切至大腿中部，使用全宽羽化补靴保尾；Row 3 (p2[9..10]) 仅靴底微裁，同样用 0.94 比例补齐靴底
    a2: List[Image.Image] = []
    for idx in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]:
        ratio = 0.78 if idx <= 7 else 0.94
        aligned = align_cropped_with_ref_boots(p2[idx], ref_stand, body_ratio=ratio)
        a2.append(remove_nebula_blobs_aligned(aligned))
    a2.append(ref_stand.copy())

    keys_22 = a1 + a2
    assert len(keys_22) == 22

    # 21 个相邻区间：
    # - 符号/道具突变区间（3:召唤出电脑, 10:问号切换灯泡, 12:灯泡消失切换敲码）不光流拉糊，保持干净利落
    # - 16, 17: 笔记本淡出与摘下眼镜 仅插 1 帧，其余 16 个平滑连续动作区间各插 3 帧（4 倍加密）
    # 总帧数 = 22 + 16*3 + 2*1 + 0 = 72 帧（8×9）
    zero_intervals = {3, 10, 12}
    one_intervals = {16, 17}

    frames_72: List[Image.Image] = []
    for i in range(22):
        frames_72.append(keys_22[i])
        if i < 21:
            if i in zero_intervals:
                continue
            elif i in one_intervals:
                frames_72.append(interpolate_smooth_no_burr(keys_22[i], keys_22[i + 1]))
            else:
                frames_72.extend(interp_3(keys_22[i], keys_22[i + 1]))

    assert len(frames_72) == 72, f"Expected 72 frames, got {len(frames_72)}"

    heights = []
    for fr in frames_72:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 9
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_72):
        r, c = divmod(i, cols)
        # 严禁传 mask 参数，直接拷贝 RGBA 避免半透明星尘与黑底预乘发灰
        atlas.paste(fr, (c * CANVAS, r * CANVAS))

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "03coding_sheet（写代码_图集）.webp")
    anim_path = os.path.join(out_dir, "03coding（写代码_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 72 帧高帧率变速帧表（总时长约 8.4s）：
    # - 敲代码/按回车爆发区（16..37, 45..52）: 95ms
    # - 思考停顿 ?（38..41）、灵光 💡（42..44）、欢呼定格（53..56）: 165ms
    # - 首尾标准站姿（0, 71）: 180ms
    # - 其余戴镜/收电脑/摘镜过渡: 115ms
    durations: List[int] = []
    for i in range(72):
        if i in (0, 71):
            durations.append(180)
        elif (38 <= i <= 44) or (53 <= i <= 56):
            durations.append(165)
        elif (16 <= i <= 37) or (45 <= i <= 52):
            durations.append(95)
        else:
            durations.append(115)

    frames_72[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_72[1:],
        duration=durations,
        loop=0,
        lossless=True,
        quality=100,
        disposal=2,
    )

    white_bg = Image.new("RGBA", atlas.size, (255, 255, 255, 255))
    white_bg.alpha_composite(atlas)
    white_path = os.path.join(prev_dir, "03coding_white.png")
    white_bg.convert("RGB").save(white_path)

    black_bg = Image.new("RGBA", atlas.size, (0, 0, 0, 255))
    black_bg.alpha_composite(atlas)
    black_path = os.path.join(prev_dir, "03coding_black.png")
    black_bg.convert("RGB").save(black_path)

    res = {
        "action": "03coding",
        "frames": len(frames_72),
        "grid": f"{cols}x{rows}",
        "cell_size": CANVAS,
        "atlas_size": list(atlas.size),
        "height_min": min(heights),
        "height_max": max(heights),
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
