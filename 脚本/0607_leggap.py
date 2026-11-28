import os
import numpy as np, cv2
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
def rd(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
base = rd(os.path.join(F, "_cells", "c46.png"))
tpl = base[110:330, 95:335]
shift = {}
for i in range(1, 49):
    im = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    res = cv2.matchTemplate(im, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    shift[i] = (ml[0] - 95, ml[1] - 110, mx)
print("相对 c46 的躯干位移 / 相似度")
for i in [1,2,5,8,17,20,22,24,25,26,27,28,33,37,45,47,48]:
    print("c%02d dx=%3d dy=%3d score=%.3f" % (i, shift[i][0], shift[i][1], shift[i][2]))

def leg(im, sx, sy, h=118):
    y0 = 330 + sy
    return im[max(0,y0):max(0,y0)+h, 150+sx:320+sx]
print("\n腿区差异（与 c46 对齐后 MSE，仅取非全黑位置）")
for i in [1,2,5,8,17,20,22,24,45,47,48]:
    A = leg(base, 0, 0); B = leg(rd(os.path.join(F, "_cells","c%02d.png"%i)), shift[i][0], shift[i][1])
    hh = min(A.shape[0], B.shape[0]); mm = min(A.shape[1], B.shape[1])
    A2, B2 = A[:hh,:mm], B[:hh,:mm]
    m = (A2.max(axis=2) > 18) | (B2.max(axis=2) > 18)
    print("c%02d  mse=%.1f" % (i, float(((A2-B2)[m]**2).mean()) if m.any() else -1))
