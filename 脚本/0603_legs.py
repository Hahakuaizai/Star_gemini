import os
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(G, "待审", "06busy_fix", "_cells")
D = os.path.join(G, "待审", "06busy_fix")
for tag, idx in (("a", range(17, 33)), ("b", range(33, 49))):
    crops = []
    for i in idx:
        im = Image.open(os.path.join(C, "c%02d.png" % i))
        crops.append((i, im.crop((100, 250, 348, 448)).resize((496, 396), Image.LANCZOS)))
    cols = 4
    rows = (len(crops) + cols - 1) // cols
    sheet = Image.new("RGB", (496 * cols, (396 + 20) * rows), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    for k, (i, t) in enumerate(crops):
        x, y = (k % cols) * 496, (k // cols) * (396 + 20)
        sheet.paste(t, (x, y + 20))
        dr.text((x + 5, y + 4), "c%02d legs 2x" % i, fill=(255, 225, 0))
    sheet.save(os.path.join(D, "_legs_%s.png" % tag))
    print(tag, sheet.size)
