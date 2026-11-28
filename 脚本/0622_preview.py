import os
from PIL import Image, ImageDraw
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
S = 384

sheet = Image.open(os.path.join(F, "06working_busy_sheet（忙碌_图集）.webp")).convert("RGBA")
for name, bg in (("06busy_white.png", (255, 255, 255)), ("06busy_black.png", (0, 0, 0))):
    base = Image.new("RGB", sheet.size, bg)
    base.paste(sheet, (0, 0), sheet)
    base.resize((1536, int(sheet.height * 1536 / sheet.width)), Image.LANCZOS).save(os.path.join(F, name))

# 1:1 对照（左原右修）
idx = list(range(25, 45))
cols, rows = 5, 4
w = S
out = Image.new("RGB", (w * 2 * cols + 10 * (cols - 1), (w + 22) * rows), (28, 28, 28))
dr = ImageDraw.Draw(out)
for k, i in enumerate(idx):
    a = Image.open(os.path.join(F, "_cells", "c%02d.png" % i)).convert("RGB")
    b = Image.open(os.path.join(F, "_fixed", "c%02d.png" % i)).convert("RGB")
    x = (k % cols) * (w * 2 + 10)
    y = (k // cols) * (w + 22)
    out.paste(a, (x, y + 22)); out.paste(b, (x + w, y + 22))
    dr.text((x + 4, y + 5), "c%02d 补腿前" % i, fill=(255, 200, 80))
    dr.text((x + w + 4, y + 5), "c%02d 补腿后" % i, fill=(120, 255, 160))
out.save(os.path.join(F, "对照_补腿20格_1比1.png"))
print("cmp", out.size)
