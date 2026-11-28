import os
import numpy as np, cv2
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
ref = rd(os.path.join(F, "_cells", "c45.png"))
# 参考腿区（c45 完整）：y 330..447, x 150..300
tpl = ref[330:448, 150:300]
print("tpl", tpl.shape)
for i in [1,5,17,20,22,24,26,37,40,46,47,48]:
    im = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    res = cv2.matchTemplate(im[300:448], tpl, cv2.TM_SQDIFF_NORMED)
    mn, _, ml, _ = cv2.minMaxLoc(res)
    print("c%02d  best SQDIFF=%.4f at x=%d y=%d" % (i, mn, ml[0], ml[1] + 300))
