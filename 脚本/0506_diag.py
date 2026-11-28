import os, glob
import numpy as np
from PIL import Image, ImageSequence

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for key in ["05thinking（", "05thinking_sheet", "06working_busy（", "06working_busy_sheet"]:
    hits = [p for p in glob.glob(os.path.join(G, "成品", "*.webp")) if os.path.basename(p).startswith(key)]
    if not hits:
        print("未找到:", key); continue
    p = hits[0]
    im = Image.open(p)
    n = getattr(im, "n_frames", 1)
    durs, frames = [], []
    for f in ImageSequence.Iterator(im):
        durs.append(f.info.get("duration", 0))
        frames.append(np.asarray(f.convert("RGBA").resize((96, 96), Image.NEAREST)).astype(np.int16))
    print("=" * 60)
    print(os.path.basename(p), im.size, "frames=", n, "total_ms=", sum(durs))
    print("  dur 范围:", (min(durs), max(durs)))
    if n > 1:
        d = [float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, n)]
        d.append(float(np.abs(frames[-1] - frames[0]).mean()))
        print("  帧间差 min=%.2f mean=%.2f max=%.2f 首尾=%.2f" % (min(d), np.mean(d), max(d), d[-1]))
        idx = sorted(range(len(d)), key=lambda i: -d[i])[:8]
        print("  最大跳变(->):", [(i + 1, round(d[i], 1)) for i in idx])
    a = np.asarray(im.convert("RGBA"))[:, :, 3]
    semi = int(((a > 0) & (a < 255)).sum())
    print("  sheet 半透明占比: %.1f%%" % (100 * semi / a.size))
