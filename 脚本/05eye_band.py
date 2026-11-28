import os
import numpy as np
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
D = os.path.join(OUT, "_diag0506")
a = Image.open(os.path.join(OUT, "05thinking", "_norm", "norm_44.png")).convert("RGBA")
b = Image.open(os.path.join(OUT, "05thinking", "_norm", "norm_45.png")).convert("RGBA")
box = (120, 95, 290, 165)
sheet = Image.new("RGB", (170 * 4 * 2 + 20, 70 * 4 + 30), (240, 240, 240))
dr = ImageDraw.Draw(sheet)
for k, im in enumerate((a, b)):
    c = Image.new("RGBA", im.size, (240, 240, 240, 255))
    c.alpha_composite(im)
    sheet.paste(c.convert("RGB").crop(box).resize((680, 280), Image.LANCZOS), (k * 690, 20))
dr.text((6, 4), "f44 eye band 4x", fill=(160, 20, 20))
dr.text((696, 4), "f45 eye band 4x", fill=(160, 20, 20))
sheet.save(os.path.join(D, "05_eye_band.png"))
arr = np.array(a.convert("RGBA"))[95:165, 120:290]
print("sample colors:", arr[::12, ::24, :3].reshape(-1, 3)[:12].tolist())
