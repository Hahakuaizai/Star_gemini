import os
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(G, "待审", "06busy_fix", "_cells")
D = os.path.join(G, "待审", "06busy_fix")
W, H = 300, 240
for tag, idx in (("a", range(17, 33)), ("b", range(33, 49))):
    cols = 4; rows = 4
    sheet = Image.new("RGB", (W * cols, (H + 16) * rows), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        im = Image.open(os.path.join(C, "c%02d.png" % i)).crop((100, 250, 348, 448)).resize((W, H), Image.LANCZOS)
        x, y = (k % cols) * W, (k // cols) * (H + 16)
        sheet.paste(im, (x, y + 16))
        dr.text((x + 4, y + 3), "c%02d" % i, fill=(255, 225, 0))
    sheet.save(os.path.join(D, "_legs_%s.jpg" % tag), quality=88)
    print(tag, sheet.size, os.path.getsize(os.path.join(D, "_legs_%s.jpg" % tag)) // 1024, "KB")
