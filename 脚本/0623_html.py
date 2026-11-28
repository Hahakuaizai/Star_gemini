import os, io, json, base64
from PIL import Image
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(G, "待审", "06busy_fix")
SRC = os.path.join(G, "src", "成品")
NEW = os.path.join(F, "06working_busy（忙碌_动画）.webp")
OLD = os.path.join(F, "旧版143帧_对照用.webp")

def parse_durs(path):
    data = open(path, "rb").read(); i = 12; d = []
    while i < len(data) - 8:
        tag = data[i:i+4]; size = int.from_bytes(data[i+4:i+8], "little")
        if tag == b"ANMF":
            d.append(int.from_bytes(data[i+8+12:i+8+15], "little"))
        i += 8 + size + (size & 1)
    return d

def anim_b64(path, size, quality=80):
    im = Image.open(path)
    fr = []
    n = getattr(im, "n_frames", 1)
    dur = parse_durs(path) or im.info.get("duration", 60)
    for k in range(n):
        im.seek(k)
        fr.append(im.convert("RGBA").resize((size, size), Image.LANCZOS))
    buf = io.BytesIO()
    fr[0].save(buf, format="WEBP", save_all=True, append_images=fr[1:], duration=(dur if isinstance(dur, list) else [dur]*n), loop=0, quality=quality, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()

def sheet_b64(path, width, quality=88):
    im = Image.open(path).convert("RGBA")
    w = width
    im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    bg = Image.new("RGB", im.size, (16, 16, 18))
    bg.paste(im, (0, 0), im)
    buf = io.BytesIO(); bg.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def pair_b64(i):
    a = Image.open(os.path.join(F, "_cells", "c%02d.png" % i)).convert("RGB")
    b = Image.open(os.path.join(F, "_fixed", "c%02d.png" % i)).convert("RGB")
    out = Image.new("RGB", (768, 384))
    out.paste(a, (0, 0)); out.paste(b, (384, 0))
    buf = io.BytesIO(); out.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

print("编码动画…")
nb = anim_b64(NEW, 240)
ob = anim_b64(OLD, 200)
sb = sheet_b64(os.path.join(F, "06working_busy_sheet（忙碌_图集）.webp"), 1000)
pairs = [(i, pair_b64(i)) for i in range(25, 45)]
meta = json.load(open(os.path.join(F, "06busy_meta.json")))

cards = "\n".join(
    '<figure><div class="duo"><img src="%s"><img src="%s"></div>'
    '<figcaption>c%02d　<span class="o">补腿前</span> / <span class="n">补腿后</span></figcaption></figure>' % (u, u, i)
    for i, u in pairs)

html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>06 忙碌 · 补腿修复对照</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0b0b0e;color:#e7e7ee;font:15px/1.7 -apple-system,"Segoe UI","PingFang SC",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:34px 0 10px;color:#c9c9d8}
.sub{color:#8d8d9e;font-size:13px;margin-bottom:22px}
.row{display:flex;gap:26px;flex-wrap:wrap}
.card{background:#15151b;border:1px solid #23232c;border-radius:12px;padding:14px}
img{display:block;border-radius:8px}
.duo{display:flex}
.duo img{width:50%%;border-radius:0}
figure{margin:0;background:#15151b;border:1px solid #23232c;border-radius:10px;overflow:hidden}
figcaption{padding:6px 8px;font-size:12px;color:#9a9aab}
.o{color:#ffb95e}.n{color:#6be3a0}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.note{background:#15151b;border-left:3px solid #6b5cff;padding:12px 14px;border-radius:8px;font-size:13.5px;color:#bcbccd}
</style></head><body><div class="wrap">
<h1>06 忙碌（void_press）· 补腿修复 ＋ 重出片</h1>
<div class="sub">素材 48 格（1792×1344，4×3） · 补腿 20 格 · 成片 %d 帧 / %.2fs / 384×384 · 图集 %d×%d（8×%d） · 节奏 关键140ms/过渡95ms</div>
<div class="row">
<figure class="card"><img src="%s" width="240" height="240"><figcaption>新版 %d 帧 · 关键 %dms / 过渡 %dms</figcaption></figure>
<figure class="card"><img src="%s" width="200" height="200"><figcaption>旧版 143 帧（腿脚缺失）</figcaption></figure>
</div>
<div class="note" style="margin-top:18px">缺腿位置：part3 #25–36 全缺、part4 #37–44 脚被格底切掉、part2 #20–23 脚小贴边。修复方式：以 c46（完整帧）腿部原画为基准贴图，按每帧脸部中线与脚底线定位，只贴下半身、上身像素不动。</div>
<h2>补腿逐格 1:1 对照（左＝原格，右＝补腿后）</h2>
<div class="grid">
%s
</div>
<h2>成片图集</h2>
<figure class="card"><img src="%s" style="width:100%%"></figure>
</div></body></html>""" % (
    meta["frames"], meta["total_ms"] / 1000, meta["sheet"][0], meta["sheet"][1], meta["rows"],
    nb, meta["frames"], meta["duration_ms"]["key"], meta["duration_ms"]["mid"], ob, cards, sb)

p = os.path.join(F, "06busy_fix.html")
open(p, "w", encoding="utf-8").write(html)
print("html", round(os.path.getsize(p) / 1024 / 1024, 2), "MB")
