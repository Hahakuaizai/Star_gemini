import os, json
import numpy as np, cv2
from PIL import Image
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
rep = {r[0]: r for r in json.load(open(os.path.join(F, "_align.json")))}
ids = [29, 33, 40]
tiles = []
for i in ids:
    score, s, x0, y0 = rep[i][1], rep[i][2], rep[i][3], rep[i][4]
    orig = cv2.cvtColor(cv2.imread(os.path.join(F, "_cells", "c%02d.png" % i)), cv2.COLOR_BGR2RGB)
    ai = cv2.cvtColor(cv2.imread(os.path.join(F, "_ai", "c%02d.png" % i)), cv2.COLOR_BGR2RGB)
    M = np.array([[1.0/s, 0, x0/s], [0, 1.0/s, y0/s]], np.float32)
    wai = cv2.warpAffine(ai, M, (448, 448))
    tiles.append(np.hstack([orig, wai]))
Image.fromarray(np.vstack(tiles)).save(os.path.join(F, "_dbg_warp.png"))
print("ok")
