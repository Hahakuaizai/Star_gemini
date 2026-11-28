import os
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(G, "待审", "06busy_fix", "_cells")
D = os.path.join(G, "待审", "06busy_fix")
idx = list(range(17, 49))
T = 220
cols = 8
rows = (len(idx) + cols - 1) // cols
sheet = Image.new("RGB", (T * cols, (T + 18) * rows), (30, 30, 30))
dr = ImageDraw.Draw(sheet)
for k, i in enumerate(idx):
    im = Image.open(os.path.join(C, "c%02d.png" % i))
    t = im.resize((T, T), Image.LANCZOS)
    x, y = (k % cols) * T, (k // cols) * (T + 18)
    sheet.paste(t, (x, y + 18))
    dr.text((x + 4, y + 3), "c%02d" % i, fill=(255, 220, 0))
sheet.save(os.path.join(D, "_diag_c17_48.png"))
print("ok", sheet.size)
