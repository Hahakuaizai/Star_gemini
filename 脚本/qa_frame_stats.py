import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label


def load_anim(path):
    im = Image.open(path)
    frames = []
    durs = []
    i = 0
    while True:
        try:
            im.seek(i)
        except EOFError:
            break
        frame = np.array(im.convert("RGBA"), dtype=np.uint8).copy()
        frames.append(frame)
        durs.append(im.info.get("duration", 0))
        i += 1
    return frames, durs


def body_metrics(a):
    al = a[:, :, 3]
    lab, n = label(al > 60)
    if n == 0:
        return 0, 0, 0, 0
    s = np.bincount(lab.ravel())
    s[0] = 0
    m = lab == int(s.argmax())
    ys, xs = np.where(m)
    return int(ys.max() - ys.min() + 1), int(ys.max()), int(xs.max() - xs.min() + 1), int((xs.min() + xs.max()) / 2)


def semi_ratio(a):
    pos = int((a[:, :, 3] > 0).sum())
    if pos == 0:
        return 0.0
    return float(((a[:, :, 3] > 0) & (a[:, :, 3] < 128)).sum()) / pos


def gap(x, y):
    a = x.astype(np.float32)
    b = y.astype(np.float32)
    m = (a[:, :, 3] > 20) | (b[:, :, 3] > 20)
    if not m.any():
        return 0.0
    return float(np.abs(a[:, :, :3] - b[:, :, :3]).max(2)[m].mean())


def main():
    if len(sys.argv) < 2:
        print("usage: qa_frame_stats.py <anim.webp> [sheet.webp cols rows]")
        return
    anim = sys.argv[1]
    frames, durs = load_anim(anim)
    seq = frames + [frames[0]]
    ms = [gap(seq[i], seq[i + 1]) for i in range(len(frames))]
    hs = []
    ft = []
    cx = []
    sr = []
    for a in frames:
        h, f, w, c = body_metrics(a)
        hs.append(h)
        ft.append(f)
        cx.append(c)
        sr.append(semi_ratio(a) * 100)
    print("帧数 %d   总时长 %d ms   单帧 %d / %d / %d ms" % (len(frames), sum(durs), min(durs), int(sum(durs) / len(durs)), max(durs)))
    print("身高 %d - %d   脚底 %d - %d   主体框中心x %d - %d   半透明边界 %.1f%% - %.1f%%" % (min(hs), max(hs), min(ft), max(ft), min(cx), max(cx), min(sr), max(sr)))
    print("帧间动度 均值 %.1f  最小 %.1f  最大 %.1f   首尾差 %.1f" % (float(np.mean(ms[:-1])), float(np.min(ms[:-1])), float(np.max(ms[:-1])), float(ms[-1])))
    low = [i + 1 for i in range(len(frames) - 1) if ms[i] < 5.0]
    print("近重复间隙 %d 个 %s" % (len(low), low[:24]))
    if len(sys.argv) >= 5:
        sheet = Image.open(sys.argv[2]).convert("RGBA")
        cols = int(sys.argv[3])
        rows = int(sys.argv[4])
        cell = sheet.width // cols
        bad = []
        for i, a in enumerate(frames):
            r, c = divmod(i, cols)
            b = np.array(sheet.crop((c * cell, r * cell, c * cell + cell, r * cell + cell)), dtype=np.int16)
            m = (a[:, :, 3] >= 200) & (b[:, :, 3] >= 200)
            if m.sum() < 200:
                bad.append((i + 1, "空格"))
                continue
            md = float(np.median(np.abs(a[:, :, :3].astype(np.int16) - b[:, :, :3]).max(2)[m]))
            if md > 2.0 or cell != a.shape[0]:
                bad.append((i + 1, md))
        print("图集 %d × %d  单格 %d px  与动画帧不一致的格子 %d 个 %s" % (cols, rows, cell, len(bad), bad[:12]))


main()
