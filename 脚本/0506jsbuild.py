import json, os, io
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(G, "待审")
SPEC = [
    ("05thinking", "05thinking", "05. 深度思考冒泡 👓（145帧精修版）", "思考｜戴镜→沉思→灵光→摘镜", "嗯……让我好好想想 🤔", "../成品/05thinking（思考_动画）.webp", "../成品/05thinking_sheet（思考_图集）.webp"),
    ("06busy", "06working_busy", "06. 忙碌虚空点按调度（143帧精修版）", "忙碌｜修修补补到完成", "咔哒、咔哒——马上好！🔧", "../成品/06working_busy（忙碌_动画）.webp", "../成品/06working_busy_sheet（忙碌_图集）.webp"),
    ("07sleep_start", "07sleep_start", "07. 睡觉·开始（104帧精修版）", "入睡｜站姿→坐睡", "困了……趴一会儿……", "../成品/07sleep_start（入睡_动画）.webp", "../成品/07sleep_start_sheet（入睡_图集）.webp"),
    ("08sleep_loop", "08sleep_loop", "08. 睡觉·循环（137帧精修版）", "循环｜坐睡姿", "Zzz…… Zzz……", "../成品/08sleep_loop（睡眠循环_动画）.webp", "../成品/08sleep_loop_sheet（睡眠循环_图集）.webp"),
    ("09sleep_wake", "09sleep_wake", "09. 睡觉·起来（98帧精修版）", "醒来｜坐睡→站姿", "唔……睡醒啦～", "../成品/09sleep_wake（睡醒_动画）.webp", "../成品/09sleep_wake_sheet（睡醒_图集）.webp"),
]
meta_dir = {k: os.path.join(OUT, k) for k in ("05thinking", "06busy")}
meta_dir.update({k: os.path.join(OUT, k) for k in ("07sleep_start", "08sleep_loop", "09sleep_wake")})
blocks = []
for key, fid, name, quote_desc, quote, anim, sheet in SPEC:
    mp = os.path.join(meta_dir[key], key + "_meta.json")
    m = json.load(open(mp, encoding="utf-8"))
    dur = ", ".join(str(x) for x in m["durations"])
    blocks.append(f'''      {{
        id: "{fid}",
        name: "{name}",
        sheet: "{sheet}",
        anim: "{anim}",
        cols: {m["cols"]},
        rows: {m["rows"]},
        frames: {m["frames"]},
        cellSize: {m["cellSize"]},
        quote: "{quote}",
        durations: [{dur}]
      }},''')
    print(key, m["frames"], "frames", m["cols"], "x", m["rows"], "dur项", len(m["durations"]))
open(os.path.join(OUT, "_diag0506", "js_blocks.txt"), "w", encoding="utf-8").write("\n".join(blocks))
print("written")
