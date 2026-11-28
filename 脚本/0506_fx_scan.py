import os
import numpy as np
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")

for act in ("05thinking", "06busy"):
    print("==", act)
    for i in range(1, 49):
        a = np.array(Image.open(os.path.join(OUT, act, "_norm", "norm_%02d.png" % i)).convert("RGBA"), dtype=np.int16)
        r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        h = a.shape[0]
        top = np.zeros_like(al, bool)
        top[: int(h * 0.45)] = True
        red = (r > 135) & (g < 125) & (b < 145) & (r - g > 45) & (al > 150) & top
        gold = (r > 210) & (g > 165) & (b < 120) & (al > 150)
        bright = (r > 240) & (g > 240) & (b > 235) & (al > 150) & top
        print("%02d red=%5d gold=%5d brightTop=%5d" % (i, int(red.sum()), int(gold.sum()), int(bright.sum())))
