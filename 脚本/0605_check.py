import os
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
idx = list(range(25, 45))
W, H = 168, 230
for part, sub in ((1, idx[:10]), (2, idx[10:])):
    sheet = Image.new("RGB", ((W * 2 + 8) * 5, (H + 16) * 2), (45, 45, 45))
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(sub):
        a = Image.open(os.path.join(F, "_cells", "c%02d.png" % i)).crop((85, 218, 365, 448)).resize((W, H), Image.LANCZOS)
        b = Image.open(os.path.join(F, "_fixed", "c%02d.png" % i)).crop((85, 218, 365, 448)).resize((W, H), Image.LANCZOS)
        cx, cy = (k % 5) * (W * 2 + 8), (k // 5) * (H + 16)
        sheet.paste(a, (cx, cy + 16)); sheet.paste(b, (cx + W, cy + 16))
        dr.text((cx + 3, cy + 3), "c%02d 原" % i, fill=(255, 220, 0))
        dr.text((cx + W + 3, cy + 3), "c%02d 修" % i, fill=(120, 255, 160))
    sheet.save(os.path.join(F, "_cmp%d.jpg" % part), quality=90)
    print(part, os.path.getsize(os.path.join(F, "_cmp%d.jpg" % part)) // 1024, "KB")
