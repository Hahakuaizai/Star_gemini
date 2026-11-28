import base64, io, json, os
from PIL import Image, ImageSequence

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
HTML = os.path.join(OUT, "05_06_思考忙碌.html")


def anim_b64(path, size, q):
    im = Image.open(path)
    fr, du = [], []
    for f in ImageSequence.Iterator(im):
        fr.append(f.convert("RGBA").resize((size, size), Image.LANCZOS))
        du.append(f.info.get("duration", 60))
    b = io.BytesIO()
    fr[0].save(b, "WEBP", save_all=True, append_images=fr[1:], duration=du, loop=0,
              quality=q, alpha_quality=90, disposal=2)
    return base64.b64encode(b.getvalue()).decode(), len(fr), sum(du), len(b.getvalue())


def sheet_b64(path, width, q):
    im = Image.open(path)
    h = round(im.height * width / im.width)
    r = im.resize((width, h), Image.LANCZOS)
    b = io.BytesIO()
    r.save(b, "WEBP", quality=q, method=5)
    return base64.b64encode(b.getvalue()).decode(), r.size, len(b.getvalue())


def meta(d):
    return json.load(open(os.path.join(OUT, d, d + "_meta.json")))


sets = [
    ("05thinking", "05thinking（思考_动画）.webp", "05thinking_sheet（思考_图集）.webp",
     os.path.join(G, "成品"), "05thinking（思考_动画）.webp", "05thinking_sheet（思考_图集）.webp",
     "思考｜戴上眼镜 → 抱胸沉思（问号气泡）→ 金瞳灵光爆发 → 摘下眼镜回位"),
    ("06busy", "06working_busy（忙碌_动画）.webp", "06working_busy_sheet（忙碌_图集）.webp",
     os.path.join(G, "成品"), "06working_busy（忙碌_动画）.webp", "06working_busy_sheet（忙碌_图集）.webp",
     "忙碌｜手持扳手修修补补 → 双手摊开 → 头顶齿轮/扳手/告示牌浮现 → 对勾完成收工"),
]

cards = []
tot = 0
for d, anim, sheet, olddir, oldanim, oldsheet, desc in sets:
    nb, nfr, nms, nsz = anim_b64(os.path.join(OUT, d, anim), 240, 84)
    ob, ofr, oms, osz = anim_b64(os.path.join(olddir, oldanim), 160, 70)
    sb, ssize, ssz = sheet_b64(os.path.join(OUT, d, sheet), 1000, 72)
    tot += nsz + osz + ssz
    m = meta(d)
    cards.append(dict(d=d, desc=desc, nb=nb, ob=ob, sb=sb, nfr=nfr, nms=nms, ofr=ofr, oms=oms,
                      cols=m["cols"], rows=m["rows"], grid=f'{m["cols"]*384}x{m["rows"]*384}',
                      ssize=f"{ssize[0]}x{ssize[1]}"))

tpl = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>05 思考 / 06 忙碌 精修对照</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#141418;color:#e9e9f0;font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:16px 14px 48px}
h1{font-size:19px;margin:6px 0 4px}h2{font-size:16px;margin:26px 0 4px;color:#c9b6ff}
p.desc{color:#a9a9bb;font-size:13px;margin:4px 0 10px}
.card{background:#1e1e26;border:1px solid #2c2c38;border-radius:14px;padding:12px;margin:12px 0}
.stage{display:flex;justify-content:center;background:linear-gradient(160deg,#2a2438,#1b1b23);border-radius:10px;padding:8px}
.stage img{width:min(72vw,300px);height:auto;image-rendering:auto}
.tabs{display:flex;gap:8px;margin:10px 0}
.tabs button{flex:1;padding:8px;border-radius:9px;border:1px solid #3a3a4a;background:#262632;color:#cfcfe0;font-size:13px}
.tabs button.on{background:#5b46a8;border-color:#7a63d0;color:#fff}
.info{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px;font-size:12px;color:#b9b9cc}
.info div{background:#25252f;border-radius:8px;padding:6px 8px}
.info span{display:block;color:#8f8fa5;font-size:11px}
.sheet{margin-top:10px}.sheet img{width:100%;border-radius:10px;background:#000}
.hide{display:none}
.note{font-size:12.5px;color:#9d9db0;line-height:1.7;margin-top:8px}
.note b{color:#cdb8ff}
</style></head><body><div class="wrap">
<h1>05 思考 / 06 忙碌 · 精修对照</h1>
<p class="desc">新版在待审区，未替换成品。点上方按钮切换新旧版本播放。</p>
__CARDS__
<div class="card"><h2 style="margin-top:0">本次改了什么</h2>
<p class="note">
<b>1. 抠图重做</b>：黑底按亮度软阈值抠，边缘反解去黑边，白底无脏边、无灰晕。<br>
<b>2. 底部杂物清除</b>：06 素材个别帧（28/32/39/43）脚下有橙块与灰条残渣，已按连通域规则清掉。<br>
<b>3. 归一化固定对齐</b>：全段同一缩放（05 0.6907 / 06 0.7125），躯干中线与脚底用中位数锁死，不再逐帧对齐，彻底消除左右/上下呼吸抖动。<br>
<b>4. 插帧</b>：每对关键帧补 2 帧，位移由下半身相位相关估计（限幅 3px）；帧差小的走全混合、中等的 0.7、大的逐像素权重（动作区不混），粒子层单侧取帧，消除双轮廓灰影。<br>
<b>5. 节奏</b>：05 戴镜前 138ms 慢入、沉思段 76ms、收尾 104ms；06 起手 84ms、忙碌段 62ms、收尾 96ms。<br>
<b>6. 首尾闭合</b>：末帧回到首帧，05 末段补 2 帧过渡（原 48 帧末首差 12.6）。
</p></div>
</div>
<script>
document.querySelectorAll('.tabs').forEach(function(t){
  t.addEventListener('click',function(e){
    var b=e.target.closest('button'); if(!b) return;
    var card=t.closest('.card');
    t.querySelectorAll('button').forEach(function(x){x.classList.remove('on')});
    b.classList.add('on');
    card.querySelectorAll('.stage img').forEach(function(im){
      im.classList.toggle('hide', im.dataset.k!==b.dataset.k);
    });
  });
});
</script></body></html>"""

out = []
for c in cards:
    out.append(f"""
<h2>{c['d'][:2]} · {c['d'][2:].replace('thinking','思考').replace('busy','忙碌')}</h2>
<p class="desc">{c['desc']}</p>
<div class="card">
<div class="stage">
<img data-k="new" src="data:image/webp;base64,{c['nb']}">
<img data-k="old" class="hide" src="data:image/webp;base64,{c['ob']}">
</div>
<div class="tabs">
<button data-k="new" class="on">新版 {c['nfr']} 帧</button>
<button data-k="old">旧版 {c['ofr']} 帧</button>
</div>
<div class="info">
<div><span>帧数</span>{c['nfr']}</div>
<div><span>时长</span>{c['nms']/1000:.2f}s</div>
<div><span>单帧</span>{c['nms']//c['nfr']}ms</div>
<div><span>图集</span>{c['grid']}</div>
</div>
<div class="sheet"><img src="data:image/webp;base64,{c['sb']}" alt="图集"></div>
</div>""")

html = tpl.replace("__CARDS__", "\n".join(out))
with open(HTML, "w", encoding="utf-8") as fh:
    fh.write(html)
print("html", round(os.path.getsize(HTML) / 1048576, 2), "MB  payload", round(tot / 1048576, 2), "MB")
