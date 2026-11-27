import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

G = r"C:\Users\30894\Desktop\workspace\帧动画\gemini"
D = os.path.join(G, "待审", "04roam")
D48 = D
CANVAS = 384


def load(path):
    im = Image.open(path)
    fr = []
    du = []
    i = 0
    while True:
        try:
            im.seek(i)
        except EOFError:
            break
        fr.append(im.convert("RGBA").copy())
        du.append(im.info.get("duration", 0))
        i += 1
    return fr, du


def font(sz):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def motion(a, b):
    x = np.array(a, dtype=np.float32)
    y = np.array(b, dtype=np.float32)
    m = (x[:, :, 3] > 20) | (y[:, :, 3] > 20)
    if not m.any():
        return 0.0
    return float(np.abs(x[:, :, :3] - y[:, :, :3]).max(2)[m].mean())


f36, d36 = load(os.path.join(D, "36帧基准", "04roam（踏步_动画）.webp"))
f48, d48 = load(os.path.join(D48, "04roam（踏步_动画）.webp"))

t36 = np.cumsum([0] + d36)
t48 = np.cumsum([0] + d48)
seq36 = f36 + [f36[0]]
seq48 = f48 + [f48[0]]
m36 = [motion(seq36[i], seq36[i + 1]) for i in range(len(f36))]
m48 = [motion(seq48[i], seq48[i + 1]) for i in range(len(f48))]

combo = []
for i in range(len(f48)):
    tm = t48[i]
    k = int(np.argmin(np.abs(t36[:-1] - tm)))
    img = Image.new("RGBA", (CANVAS * 2 + 4, CANVAS), (255, 255, 255, 255))
    img.alpha_composite(f36[k], (0, 0))
    img.alpha_composite(f48[i], (CANVAS + 4, 0))
    combo.append(img)
combo[0].save(os.path.join(D48, "对照_动画_36vs48.webp"), format="WEBP", save_all=True,
              append_images=combo[1:], duration=d48, loop=0, lossless=True, quality=100,
              disposal=2, minimize_size=False)

s36 = Image.open(os.path.join(D, "36帧基准", "04roam_sheet（踏步_图集）.webp")).convert("RGBA")
s48 = Image.open(os.path.join(D48, "04roam_sheet（踏步_图集）.webp")).convert("RGBA")
scale = 968.0 / 2304.0
a = s36.resize((int(s36.width * scale), int(s36.height * scale)), Image.Resampling.LANCZOS)
b = s48.resize((int(s48.width * scale), int(s48.height * scale)), Image.Resampling.LANCZOS)
for tone, bg in (("白底", (255, 255, 255, 255)), ("黑底", (0, 0, 0, 255))):
    w = a.width + b.width + 96
    h = max(a.height, b.height) + 96
    c = Image.new("RGBA", (w, h), bg)
    c.alpha_composite(a, (32, 72))
    c.alpha_composite(b, (a.width + 64, 72))
    dc = ImageDraw.Draw(c)
    fc = (255, 255, 255) if tone == "黑底" else (20, 20, 20)
    dc.text((32, 28), "36 帧基准（6 × 6）", font=font(34), fill=fc)
    dc.text((a.width + 64, 28), "48 帧版（8 × 6 · 8 列 6 行）", font=font(34), fill=fc)
    c.convert("RGB").save(os.path.join(D48, "对照_图集_36vs48_%s.png" % tone))

W, H = 1500, 560
pad = (92, 46, 44, 84)
c = Image.new("RGB", (W, H), (255, 255, 255, 255))
dc = ImageDraw.Draw(c)
f14 = font(22)
f17 = font(24)
xg = (W - pad[0] - pad[2]) / 7000.0
yg = (H - pad[1] - pad[3]) / 88.0
for v in range(0, 90, 10):
    y = pad[1] + (88 - v) * yg
    dc.line((pad[0], y, W - pad[2], y), fill=(232, 232, 238), width=1)
    dc.text((pad[0] - 46, y - 12), str(v), font=f14, fill=(120, 120, 130))
for t in range(0, 7001, 1000):
    x = pad[0] + t * xg
    dc.line((x, pad[1], x, H - pad[3]), fill=(232, 232, 238), width=1)
    dc.text((x - 22, H - pad[3] + 10), "%d s" % (t // 1000), font=f14, fill=(120, 120, 130))
for m, tt, col in ((m36, t36, (190, 120, 60)), (m48, t48, (108, 70, 200))):
    pts = [(pad[0] + tt[i] * xg, pad[1] + (88 - m[i]) * yg) for i in range(len(m))]
    dc.line(pts, fill=col, width=3, joint="curve")
for m, tt, col in ((m36, t36, (190, 120, 60)), (m48, t48, (108, 70, 200))):
    for i in range(len(m)):
        x = pad[0] + tt[i] * xg
        y = pad[1] + (88 - m[i]) * yg
        dc.ellipse((x - 3, y - 3, x + 3, y + 3), fill=col)
dc.text((pad[0], 10), "帧间动度 · 36 帧基准 vs 48 帧版（横轴播放时间 0–7 s）", font=font(28), fill=(30, 30, 40))
dc.text((pad[0], H - 52), "— 36 帧均值 %.1f / 峰值 %.1f      — 48 帧均值 %.1f / 峰值 %.1f" % (np.mean(m36), np.max(m36), np.mean(m48), np.max(m48)), font=f17, fill=(60, 60, 72))
c.save(os.path.join(D48, "对照_动度曲线_36vs48.png"))
print("combo", len(combo), "36 mean/max %.1f/%.1f 48 mean/max %.1f/%.1f" % (np.mean(m36), np.max(m36), np.mean(m48), np.max(m48)))
