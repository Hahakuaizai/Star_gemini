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

_spec_18 = importlib.util.spec_from_file_location("n18", "gemini/脚本/18process_note_taking.py")
_n18 = importlib.util.module_from_spec(_spec_18)
_spec_18.loader.exec_module(_n18)
align_full_body = _n18.align_full_body
align_cropped_with_ref_boots = _n18.align_cropped_with_ref_boots

_spec22 = importlib.util.spec_from_file_location("e22", "gemini/脚本/22process_eat_cookie.py")
_e22 = importlib.util.module_from_spec(_spec22)
_spec22.loader.exec_module(_e22)
interpolate_flow_multi = _e22.interpolate_flow_multi

CANVAS = 384


def main() -> dict:
    s1 = Image.open("gemini/素材/03coding/part1_v2.jpg").convert("RGBA")
    s2 = Image.open("gemini/素材/03coding/part2_v2.jpg").convert("RGBA")

    b1 = find_cell_boxes(s1, 4, 3)
    b2 = find_cell_boxes(s2, 4, 3)

    p1 = [clean_black_bg_rich_particles(s1.crop(b1[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]
    p2 = [clean_black_bg_rich_particles(s2.crop(b2[i]), bg_thresh=24, hole_thresh=18, glow_cutoff=22) for i in range(12)]

    # 以 Part 1 首帧作为全局绝对站姿基准（身高 ~304px，脚底 GROUND_Y=366）
    ref_stand = remove_nebula_blobs_aligned(align_full_body(p1[0]))

    # Part 1 精简去重关键帧（9 帧）：
    # 剔除 p1[5]（电脑在右侧）、p1[7]（与 p1[6]/p1[8] 重复）、p1[10]（漏画眼镜的问号帧）
    p1_order = [0, 1, 2, 4, 3, 6, 8, 9, 11]
    a1: List[Image.Image] = []
    for idx in p1_order:
        if idx == 0:
            a1.append(ref_stand)
        else:
            a1.append(remove_nebula_blobs_aligned(align_full_body(p1[idx])))

    # Part 2 精简去重关键帧（8 帧）：
    # 剔除 p2[1]（重复灯泡帧）、p2[5]（相似欢呼帧）、p2[8]（换手拿眼镜）、p2[10]/p2[11]（重复站姿），尾帧复用 ref_stand 闭环
    a2: List[Image.Image] = []
    for idx in [0, 2, 3, 4, 6, 7, 9]:
        ratio = 0.78 if idx <= 7 else 0.94
        aligned = align_cropped_with_ref_boots(p2[idx], ref_stand, body_ratio=ratio)
        a2.append(remove_nebula_blobs_aligned(aligned))
    a2.append(ref_stand.copy())

    keys_17 = a1 + a2
    assert len(keys_17) == 17

    # 16 个相邻区间的插值帧数（共 47 帧中间帧，17 + 47 = 64 帧，8x8）：
    # - 区间 3 (合十->召唤电脑)、8 (?->灯泡)、9 (灯泡->敲码)、12 (欢呼收电脑->扶镜) 设为 0，杜绝半透明糊电脑/黑糊灯泡鬼影
    # - 其余肢体连续区间使用 interpolate_flow_multi 多步高精度双向光流补齐
    counts = [4, 3, 3, 0, 3, 5, 4, 2, 0, 0, 5, 5, 0, 4, 4, 5]
    assert len(counts) == 16 and sum(counts) == 47

    frames_64: List[Image.Image] = []
    for i, k in enumerate(keys_17):
        frames_64.append(k)
        if i < len(counts) and counts[i] > 0:
            n = counts[i]
            ts = [(j + 1) / (n + 1) for j in range(n)]
            frames_64.extend(interpolate_flow_multi(keys_17[i], keys_17[i + 1], ts))

    assert len(frames_64) == 64, f"Expected 64 frames, got {len(frames_64)}"

    heights = []
    for fr in frames_64:
        a = np.array(fr)[:, :, 3]
        mask = get_main_component_mask(a)
        ys = np.nonzero(mask)[0]
        heights.append(int(ys.max() - ys.min() + 1))

    cols, rows = 8, 8
    atlas = Image.new("RGBA", (cols * CANVAS, rows * CANVAS), (0, 0, 0, 0))
    for i, fr in enumerate(frames_64):
        r, c = divmod(i, cols)
        atlas.paste(fr, (c * CANVAS, r * CANVAS))

    out_dir = "gemini/成品"
    prev_dir = "gemini/预览"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    sheet_path = os.path.join(out_dir, "03coding_sheet（写代码_图集）.webp")
    anim_path = os.path.join(out_dir, "03coding（写代码_动画）.webp")
    atlas.save(sheet_path, format="WEBP", lossless=True, quality=100)

    # 64 帧变速帧表（总时长约 7.8s，节奏紧致明快）：
    # - 首尾标准站姿（0, 63）: 180ms
    # - 思考停顿 ?（31..32）、灵光 💡（33）与欢呼定格（45）: 380ms / 240ms
    # - 敲代码/按回车爆发区（14..30, 34..45）: 95ms
    # - 戴镜/摘镜收尾过渡: 115ms
    durations: List[int] = []
    for i in range(64):
        if i in (0, 63):
            durations.append(180)
        elif i == 32:  # ? 思考定格
            durations.append(420)
        elif i == 33:  # 💡 灵光一现定格
            durations.append(420)
        elif i == 45:  # 握拳欢呼定格
            durations.append(260)
        elif (14 <= i <= 31) or (34 <= i <= 44):
            durations.append(95)
        else:
            durations.append(115)

    frames_64[0].save(
        anim_path,
        format="WEBP",
        save_all=True,
        append_images=frames_64[1:],
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
        "frames": len(frames_64),
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
