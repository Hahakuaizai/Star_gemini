import os, glob
import numpy as np
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
NEW = ["05thinking（思考_动画）.webp", "06working_busy（忙碌_动画）.webp"]
OLD = ["05thinking（思考_动画).webp", "06working_busy（忙碌_动画).webp"]


def stats(path, step=1):
    im = Image.open(path)
    fr = []
    dur = []
    for f in ImageSequence_iter(im):
        fr.append(np.asarray(f.convert("RGBA").resize((128, 128), Image.NEAREST)).astype(np.int16))
        dur.append(f.info.get("duration", 0))
    d = [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
    d.append(float(np.abs(fr[-1] - fr[0]).mean()))
    return len(fr), sum(dur), np.mean(d), max(d), d


def ImageSequence_iter(im):
    from PIL import ImageSequence
    return ImageSequence.Iterator(im)


for t in NEW:
    p = os.path.join(OUT, "0*", t)
    hits = [x for x in glob.glob(os.path.join(OUT, "0*", "*.webp")) if os.path.basename(x) == t]
    if not hits:
        print("缺:", t)
        continue
    n, ms, mean, mx, d = stats(hits[0])
    idx = sorted(range(len(d)), key=lambda i: -d[i])[:6]
    print("%-34s frames=%3d total=%5dms 帧差 mean=%.2f max=%.1f" % (t, n, ms, mean, mx))
    print("   最大跳变(帧i->i+1):", [(i + 1, round(d[i], 1)) for i in idx])
for t in OLD:
    hits = [x for x in glob.glob(os.path.join(G, "成品", "*.webp")) if os.path.basename(x) == t]
    if not hits:
        print("旧版缺:", t)
        continue
    n, ms, mean, mx, d = stats(hits[0])
    print("%-34s frames=%3d total=%5dms 帧差 mean=%.2f max=%.1f" % ("旧 " + t, n, ms, mean, mx))
