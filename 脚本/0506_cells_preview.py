import os, glob
import numpy as np
from PIL import Image, ImageDraw

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审", "_diag0506")
os.makedirs(OUT, exist_ok=True)

for act in ["05thinking", "06busy"]:
    cells = []
    for p in sorted(glob.glob(os.path.join(G, "素材", act, "part*.webp"))):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        cw, ch = w // 4, h // 3
        for r in range(3):
            for c in range(4):
                cells.append(im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
    print(act, "cells:", len(cells), "cell:", cells[0].size)
    TH = 180
    tw = round(cells[0].width * TH / cells[0].height)
    cols = 8
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * TH), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(cells):
        thumb = c.resize((tw, TH), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * TH
        sheet.paste(thumb, (x, y))
        dr.rectangle([x, y, x + tw - 1, y + TH - 1], outline=(60, 60, 60))
        dr.text((x + 4, y + 3), str(i + 1), fill=(255, 210, 0))
    sp = os.path.join(OUT, f"{act}_cells.png")
    sheet.save(sp)
    print("->", sp, sheet.size)
