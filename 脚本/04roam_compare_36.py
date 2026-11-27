import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

G = r"C:\Users\30894\Desktop\workspace\帧动画\gemini"
D = os.path.join(G, "待审", "04roam", "36帧基准")
OLD_ANIM = os.path.join(G, "成品", "04roam（踏步_动画）.webp")
OLD_SHEET = os.path.join(G, "成品", "04roam_sheet（踏步_图集）.webp")
NEW_ANIM = os.path.join(D, "04roam（踏步_动画）.webp")
NEW_SHEET = os.path.join(D, "04roam_sheet（踏步_图集）.webp")
FONT = "C:/Windows/Fonts/msyh.ttc"


def font(size):
    for path in (FONT, "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def durations_from_webp(path):
    data = open(path, "rb").read()
    i = 12
    d = []
    while i < len(data) - 8:
        tag = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        if tag == b"ANMF":
            d.append(int.from_bytes(data[i + 20:i + 23], "little"))
        i += 8 + size + (size & 1)
    return d


def old_frames():
    im = Image.open(OLD_ANIM)
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(im.convert("RGBA").copy())
    return out


def new_frames():
    sheet = Image.open(NEW_SHEET).convert("RGBA")
    out = []
    for i in range(36):
        r, c = divmod(i, 6)
        out.append(sheet.crop((c * 384, r * 384, c * 384 + 384, r * 384 + 384)))
    return out


def build_anim_pair():
    old = old_frames()
    new = new_frames()
    durs = durations_from_webp(NEW_ANIM)
    pairs = []
    for o, n in zip(old, new):
        canvas = Image.new("RGBA", (772, 384), (0, 0, 0, 0))
        canvas.paste(o, (0, 0), o)
        canvas.paste(n, (388, 0), n)
        pairs.append(canvas)
    out = os.path.join(D, "对照_动画_旧vs36.webp")
    pairs[0].save(out, format="WEBP", save_all=True, append_images=pairs[1:],
                  duration=durs, loop=0, lossless=True, quality=100, disposal=2, minimize_size=False)
    return out, sum(durs)


def build_sheet_pair(tone):
    scale = 0.4
    w = int(2304 * scale)
    old = Image.open(OLD_SHEET).convert("RGBA")
    old.seek(0)
    new = Image.open(NEW_SHEET).convert("RGBA")
    old = old.convert("RGBA").resize((w, w), Image.Resampling.LANCZOS)
    new = new.resize((w, w), Image.Resampling.LANCZOS)
    bg = (255, 255, 255, 255) if tone == "white" else (0, 0, 0, 255)
    head = 46
    canvas = Image.new("RGBA", (w * 2 + 16, w + head), bg)
    canvas.alpha_composite(old, (0, head))
    canvas.alpha_composite(new, (w + 16, head))
    d = ImageDraw.Draw(canvas)
    fg = (40, 36, 60, 255) if tone == "white" else (230, 226, 244, 255)
    f = font(30)
    d.text((10, 8), "旧版 6×6 图集 · 6.24s", font=f, fill=fg)
    d.text((w + 26, 8), "精修版 6×6 图集 · 7.00s", font=f, fill=fg)
    if tone == "white":
        d.line([(w + 8, head), (w + 8, w + head)], fill=(120, 110, 150, 255), width=2)
    path = os.path.join(D, "对照_图集_旧vs36_%s.png" % ("白底" if tone == "white" else "黑底"))
    canvas.convert("RGB").save(path)
    return path


if __name__ == "__main__":
    a, t = build_anim_pair()
    print(a, t)
    print(build_sheet_pair("white"))
    print(build_sheet_pair("black"))
