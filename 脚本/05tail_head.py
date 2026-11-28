import os
import numpy as np
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
D = os.path.join(OUT, "_diag0506")
idx = [44, 45, 46, 47, 48, 1, 2, 3]
W, H, cols = 200, 160, 4
rows = (len(idx) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, (H + 16) * rows), (245, 245, 245))
dr = ImageDraw.Draw(sheet)
for k, i in enumerate(idx):
    a = Image.open(os.path.join(OUT, "05thinking", "_norm", "norm_%02d.png" % i)).convert("RGBA")
    head = a.crop((60, 20, 324, 190)).resize((W, H), Image.LANCZOS)
    c = Image.new("RGBA", head.size, (245, 245, 245, 255))
    c.alpha_composite(head)
    x, y = (k % cols) * W, (k // cols) * (H + 16)
    sheet.paste(c.convert("RGB"), (x, y + 16))
    dr.text((x + 4, y + 2), "f%d" % i, fill=(180, 20, 20))
sheet.save(os.path.join(D, "05_tail_head.png"))
print("ok")
