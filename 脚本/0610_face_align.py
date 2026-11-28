import os, json
import numpy as np, cv2
from scipy.ndimage import label

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")

def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def face(im):
    r, g, b = [im[..., k].astype(np.int16) for k in range(3)]
    m = ((r > 225) & (g > 185) & (b > 170) & ((r - b) > 18) & ((r - b) < 90)).astype(np.uint8)
    m[int(im.shape[0] * 0.55):] = 0
    lab, n = label(m)
    if n == 0: return None
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    k = sizes.argmax()
    ys, xs = np.where(lab == k)
    if len(ys) < im.shape[0] * 0.003: return None
    return dict(cx=(xs.min()+xs.max())/2, cy=(ys.min()+ys.max())/2, w=xs.max()-xs.min()+1, h=ys.max()-ys.min()+1, n=len(ys))

for i in [1, 25, 29, 33, 40, 46]:
    o = rd(os.path.join(F, "_cells", "c%02d.png" % i))
    print("orig c%02d" % i, face(o))
for i in [25, 29, 33, 40]:
    a = rd(os.path.join(F, "_ai", "c%02d.png" % i))
    print("ai   c%02d" % i, face(a))
