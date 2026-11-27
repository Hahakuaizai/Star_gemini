# 紫发猫耳少女桌宠 25 动作 WebP 制作交接文档（AI 执行规范版）

> **致接手 AI**：请**先完整阅读本文档**以及工程技术指南 [AI_ANIMATION_PIPELINE_GUIDE.md](AI_ANIMATION_PIPELINE_GUIDE.md)（含 48 帧光流加中间帧、去毛刺抗锯齿、剥离星云黑块、老图集精修与防裁尾铁律），再结合同目录的 [gemini_25_webp.md](gemini_25_webp.md) 动作分镜表继续推进剩余动作。本文档中的所有【铁律】均为前置实战踩坑总结，**必须严格执行，不得自行放宽标准或更改目录与命名规则**。

---

## 一、 当前工作区结构与资产清单

工作区根目录：`c:\Users\30894\Desktop\workspace\帧动画`

```text
gemini/
├── HANDOFF.md                     # 本交接文档
├── 成品/                           # 【只放最终透明 WebP，严禁放中间文件】
│   ├── idle.webp                  # 已完成：48帧 8.51s 透明动态 WebP（待机呼吸循环）
│   ├── coding.webp                # 已完成：48帧 9.3s 透明动态 WebP（专注写代码·戴摘眼镜）
│   ├── coding_sheet.webp          # 已完成：8×6 透明精灵图集 WebP
│   ├── idle_sheet.webp            # 已完成：8×6 透明精灵图集 WebP
│   ├── happy_jump.webp            # 已完成：48帧 6.72s 透明动态 WebP
│   ├── happy_jump_sheet.webp      # 已完成：8×6 透明精灵图集 WebP
│   ├── wave_hello.webp            # 已完成：36帧 6.45s 透明动态 WebP
│   └── wave_hello_sheet.webp      # 已完成：6×6 透明精灵图集 WebP
├── 预览/                           # 网页交互播放器 + 白/黑双底色质检图
│   ├── web_pet.html               # 零依赖本地网页播放器（已接入上述 2 个动作）
│   ├── happy_jump_white.png / happy_jump_black.png
│   └── wave_hello_white.png / wave_hello_black.png
├── 脚本/                           # 已验证通过的本地自动化管线（Python 3 + Pillow + NumPy + SciPy）
│   ├── build_webp.py              # 逐列智能黑隙切分 + 双层零黑边反预乘抠图 + 锚点提取
│   ├── expand_48.py               # 头宽比例锁死 + 双向金字塔光流插帧（无重影）+ 抛物线起跳
│   └── build_wave_hello.py        # 白底素材专用抠图与扩帧参考脚本
└── 素材/                           # 原始分段图与基准锚点（按动作短名分文件夹）
    ├── happy_jump/
    │   ├── anchor_stand.png       # 【全局唯一纯黑底标准站姿基准图，所有新动作必用！】
    │   ├── part1.jpg
    │   └── part2.jpg
    └── wave_hello/
        └── part1_white.jpg
```

---

## 二、 最高优先级纠偏与绝对铁律（严禁违背）

### 1. ⚠️ 异色瞳与配件

形象**唯一真值以 `gemini/素材/happy_jump/anchor_stand.png`及原立`C:\Users\30894\Desktop\workspace\帧动画\gemini\素材\10happy_jump\形象基准图.jpg`为准**：

- **画面左侧眼睛（角色自身右眼）**：**琥珀金黄瞳**
- **画面右侧眼睛（角色自身左眼）**：**宝石粉紫瞳**
- **彩虹五角星发夹**：**严格锁定在画面右侧鬓角**（绝不允许在中间帧镜像跑到左侧！）
- **蓬松紫粉猫尾**：**严格锁定从画面右侧伸出**（除非特定转身动作，绝不跳到左侧）
- **暗紫星形腿环**：**画面左侧大腿（角色自身右腿）**

### 2. 必须使用纯黑底锚点图生图（禁止直接喂白底原立绘）

- 生成任何新动作的第 1 段 `4×3` 图集时，**必须**将 `gemini/素材/happy_jump/anchor_stand.png` 作为主要参考图输入。
- 原因：该图已是纯黑底 `#000000`、完整靴底、四边留白、严格匹配人设的标准站姿，能保证 AI 稳定输出纯黑底与统一画风比例，避免出现白底或靴子被裁切。
- 接力生成第 2、3、4 段时，使用 `python gemini/脚本/build_webp.py extract-anchors` 提取上一段最后 1 格，并与 `anchor_stand.png` 一同作为参考图输入。

### 3. 生图省配额铁律

- 调用生图工具时，**每次只生成 1 张**，先用读图工具人工核查质量，**严禁设置自动循环重试 3 次**导致配额迅速耗尽。
- 对于纯动作或对称动作，可生成 **2 段 `4×3`（24 关键帧）**，再用 `expand_48.py` 的双向光流插帧平滑扩展到 **36 或 48 帧**，兼顾流畅度与额度成本。

### 4. 文件命名与目录整洁铁律（用户明确要求）

- **命名必须简短直白**，严禁带数字编号、帧数、网格尺寸后缀（如禁止 `10_happy_jump_48f_sheet_8x6.webp`）：
  - 最终动态 WebP：`gemini/成品/<动作短名>.webp`（例如 `sweeping.webp`）
  - 最终精灵图集：`gemini/成品/<动作短名>_sheet.webp`（例如 `sweeping_sheet.webp`）
  - 双底色质检图：`gemini/预览/<动作短名>_white.png` 与 `gemini/预览/<动作短名>_black.png`
  - 原始素材目录：`gemini/素材/<动作短名>/part1.jpg`、`part2.jpg`...
- **严禁**在 `gemini/` 根目录堆放临时文件、`output_webp/` 或 `__pycache__/`，每次做完必须保持四大子目录（`成品/`、`预览/`、`脚本/`、`素材/`）干净整洁。

### 5. 画面与动画质量红线

1. **时长**：每个动作总时长严格落在 **6.0s ～ 10.0s**，必须使用变速帧表（发力/敲击/挥动/眨眼 `110～145ms`，常态过渡 `150～200ms`，停顿/定格/首尾 `220～320ms`）。
2. **安全留白与完整度**：单格 `256×256` 无损透明 WebP，四边留白 `≥10%`，严禁裁切呆毛、靴底、手尖、猫尾或任何道具粒子。
3. **首尾闭环**：除睡觉三段式（`sleep_start` / `sleep_loop` / `sleep_wake`）外，第 1 帧与最后 1 帧必须 100% 锁死为干净标准站姿（无道具、无眼镜、无粒子）。
4. **严禁半透明重影（No Alpha Cross-fade）**：禁止用简单透明度叠化补帧；若使用光流扩展，必须沿用 `expand_48.py` 中的一致性门控 + 锐利 Alpha 曲线，并视觉检查无重影。
5. **眼镜规范（仅限 5 个标注 👓 的动作）**：细框圆金属镜，高透不遮挡异色瞳，且必须在动作内完成“无 → 戴上 → 摘下消散 → 无”的完整闭环。

---

## 三、 25 个动作总进度 Checklist（已出片 17 / 25，其中精修 10 个：02、04、10–17）

> 详细分镜、粒子描写与配速见同目录 [gemini_25_webp.md](gemini_25_webp.md) 第四节。

| 状态 |  编号  | 动作名称                | 目标文件短名         | 规格建议           | 备注 |
| :--: | :----: | :---------------------- | :------------------- | :----------------- | :--- |
| [x]  | **01** | **待机呼吸休闲**        | **`idle`**           | **48帧 / 8.51s**   | **已完成，未精修**（`build_norm.py` 逐帧身高归一 + 靴底钉死 + 四边留白≥10%） |
| [x]  | **02** | **打扫清理·利落站姿版** | **`02sweeping`**     | **48帧 / 8.87s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，24张关键帧+24张防拉丝光流补帧，丰富暖金/亮粉/青白四芒星与星尘，星云黑块剥离） |
| [x]  | **03** | **专注写代码 👓**       | **`coding`**         | **48帧 / 9.31s**   | **已完成，未精修**（四段接力锚点重画，星云残渣清除，眼镜全程连贯） |
| [x]  | **04** | **原地轻快踏步**        | **`roam`**           | **48帧 / 7.00s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，32张原生关键帧+16张中点补帧，去掉4张原地停顿帧；深紫描边抗锯齿、发缝黑点清除、身高锁死 `306px`、脚底锁 `366px`、绝对原地零水平位移、首尾完全重合） |
| [x]  |   05   | 深度思考冒泡 👓         | `thinking`           | 48帧 / 8.5s / 循环 | 已完成，未精修；👓 戴摘眼镜；亮紫白边 `?` + 思考泡 |
| [x]  |   06   | 忙碌虚空点按调度        | `working_busy`       | 48帧 / 8.0s / 循环 | 已完成，未精修；指尖青白涟漪 + 齿轮/扳手/文档小图标 |
| [x]  |   07   | 睡觉-开始               | `sleep_start`        | 36帧 / 7.5s / 单次 | 已完成，未精修；首帧标准站姿 → 尾帧定格【坐睡姿】，无 `Zzz` |
| [x]  |   08   | 睡觉-循环               | `sleep_loop`         | 48帧 / 8.0s / 循环 | 已完成，未精修；首尾 100% 重合【坐睡姿】，亮紫白边 `Zzz` |
| [x]  |   09   | 睡觉-起来               | `sleep_wake`         | 36帧 / 7.5s / 单次 | 已完成，未精修；首帧【坐睡姿】→ 伸懒腰 → 尾帧标准站姿 |
| [x]  | **10** | **开心跃动庆祝**        | **`10happy_jump`**   | **48帧 / 8.45s**   | **✅ 已完成并精修验收**（升级 `384×384` `8×6`，深紫描边去毛刺抗锯齿，48帧抛物线光流补帧，放慢节奏） |
| [x]  | **11** | **被戳受惊炸毛**        | **`11poke_shock`**   | **48帧 / 8.52s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，废弃硬二值切边消除毛刺，33关键帧+15光流补帧，放慢节奏） |
| [x]  | **12** | **害羞捂脸扭捏**        | **`12shy_praise`**   | **48帧 / 8.63s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，深紫描边去毛刺抗锯齿，36关键帧+12光流补帧，放慢节奏） |
| [x]  | **13** | **傲娇抱胸侧转**        | **`13tsundere`**     | **36帧 / 6.97s**   | **✅ 已完成并精修验收**（`384×384` `6×6`，29张手绘真过渡帧+7张精准补帧，含完整侧转/回正/放手过渡，身高锁死 `310px`） |
| [x]  | **14** | **鼠标拎起悬空晃动**    | **`14drag_float`**   | **48帧 / 7.37s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，双周期正弦钟摆 ±15° + 24张光流补帧 + 头发猫尾惯性滞后飘动 + 踢脚/眨眼，身高严格锁死 `262px`） |
| [x]  | **15** | **挠痒咯咯大笑**        | **`15tickle_laugh`** | **48帧 / 7.11s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，19张关键帧+29张光流补帧，含捂肚爆笑/蓝泪珠/金音符/袖口擦泪/吐舌眨眼，身高锁死 `307～310px`） |
| [x]  | **16** | **报错慌张冒汗**        | **`16error_panic`**  | **48帧 / 7.46s**   | **✅ 已完成并精修验收**（`384×384` `8×6`，24张关键帧+24张光流补帧，含`!`/`@_@`/蓝汗滴/飞机耳/小乌云擦汗/握拳振作，全靴全尾零裁切，身高锁死 `303～310px`） |
| [x]  | **17** | **元气挥手打招呼**      | **`wave_hello`**     | **36帧 / 6.45s**   | **✅ 已完成并精修验收**（在 `gemini/成品/`） |
| [ ]  |   18   | 轻快记笔记 👓           | `note_taking`        | 48帧 / 8.5s / 循环 | 👓 戴摘眼镜；紫皮手账 + 星空羽毛笔 |
| [ ]  |   19   | 下五子棋五连胜 👓       | `gomoku`             | 48帧 / 9.5s / 单次 | 👓 戴摘眼镜；悬浮小棋盘 + 金白五连星轨 |
| [ ]  |   20   | 穿针乞巧 👓             | `thread_needle`      | 48帧 / 9.5s / 单次 | 👓 戴摘眼镜；银针红线 + 右上淡入小圆月 |
| [ ]  |   21   | 喝星空珍珠奶茶          | `drink_milktea`      | 48帧 / 8.0s / 单次 | 紫粉奶茶液面下降 + 满足擦嘴 |
| [ ]  |   22   | 吃星星糖霜曲奇          | `eat_cookie`         | 48帧 / 8.0s / 单次 | 仓鼠鼓腮咀嚼 + 舔指尖 + 粉小花光效 |
| [ ]  |   23   | 凭空生花小魔术          | `magic_flower`       | 48帧 / 8.5s / 单次 | 翻手变玫瑰 → 抛起化作粉红花瓣雨消散 |
| [ ]  |   24   | 蓝鲸光灵环绕            | `spirit_whale`       | 48帧 / 9.0s / 单次 | 胸针游出半透明青蓝小蓝鲸绕身归位 |
| [ ]  |   25   | 优雅屈膝礼仪            | `curtsy_bow`         | 36帧 / 7.0s / 单次 | 提裙屈膝致意 + 双肩高光亮白四角星芒 |

---

## 四、 标准出图 Prompt 模板（已修正瞳色与黑底锚点）

生成任意动作第 X 段（`4×3` 共 12 格）时，按以下模板填写并**附上 `gemini/素材/happy_jump/anchor_stand.png`**（第 2 段起同时附上一段最后 1 格锚点）：

```text
Generate a high-resolution 4 columns × 3 rows (12-frame) sprite sheet on a PURE SOLID BLACK background (#000000), with NO grid lines, NO borders, NO text, NO numbers, NO watermarks, and NO ground shadows.

Reference images (STRICTLY match character design, scale, full-body boots, line thickness, and pure solid black background):
1) Standing baseline anchor: c:/Users/30894/Desktop/workspace/帧动画/gemini/素材/happy_jump/anchor_stand.png
2) [If Part >= 2] Previous segment's final frame to continue smoothly from: <path_to_prev_last.png>

Character lock (DO NOT horizontally mirror):
- Chibi 2.5-head-tall purple-haired cat-ear anime girl, crisp Japanese cel-shaded style with clean dark purple outlines.
- FULL BODY WITH COMPLETE DARK PURPLE BOOTS fully visible in ALL 12 cells with at least 10% pure black margin on all 4 sides.
- Rainbow 5-pointed star hairclip strictly on VIEWER'S RIGHT side; fluffy purple-pink cat tail strictly on VIEWER'S RIGHT side; dark purple star leg ring on viewer's left thigh.
- Heterochromia strictly matching reference: VIEWER'S LEFT eye is amber-gold, VIEWER'S RIGHT eye is gem pink-purple.

Composition rules:
- 4 columns × 3 rows evenly spaced invisible grid (read left-to-right, top-to-bottom).
- Identical character height and head-to-body ratio across all 12 cells; feet centered horizontally at X=50%.
- Zero semi-transparent ghosting, zero motion blur, zero dark grey/black smoke clouds.

Action: <action_name> — Part <X> of <N> (Frames <start> to <end>):
- Frame 1 (Row 1 Col 1): ...
- ...
- Frame 12 (Row 3 Col 4): ...

Particle & effect style: High-saturation, crisp-edged discrete sparkles/symbols only. NO blurry dark smoke halos.
```

---

## 五、 标准化制作与验收流程（每做一个动作必走）

1. **保存分段原图**：将生图结果存入 `gemini/素材/<短名>/part1.jpg`、`part2.jpg`...
2. **提取接力锚点**（生成下一段前执行）：
   ```powershell
   python gemini\脚本\build_webp.py extract-anchors --segment "gemini\素材\<短名>\part1.jpg" --out-dir "gemini\素材\<短名>\anchors" --prefix "part1"
   ```
3. **抠图、对齐与导出 WebP**：
   - 36/48 帧原生分段：直接用 `gemini/脚本/build_webp.py` 切分、双层去黑边抠图并导出；
   - 24 帧扩 48 帧丝滑版：参考 `gemini/脚本/expand_48.py`（头宽比例锁死 + 双向光流无重影补帧）；
   - 将最终成品严格命名为 `gemini/成品/<短名>.webp` 与 `gemini/成品/<短名>_sheet.webp`，双底色预览图输出到 `gemini/预览/<短名>_white.png` 与 `gemini/预览/<短名>_black.png`。
4. **接入网页播放器**：在 `gemini/预览/web_pet.html` 的 `ACTIONS` 数组中追加该动作配置（相对路径指向 `../成品/<短名>_sheet.webp` 与 `../成品/<短名>.webp`）。
5. **强制视觉验收（全部 PASS 才可勾选 Checklist）**：
   - [ ] 打开 `<短名>_white.png` 检查：粒子外圈、腿间缝隙、猫尾弯折处**零黑边/零灰块**。
   - [ ] 打开 `<短名>_black.png` 检查：深紫衣裙、短靴、轮廓线完整无缺口，粒子明亮清晰。
   - [ ] 核对每一格：无裁切靴底/呆毛/粒子，头顶粒子未串入上一行。
   - [ ] 核对人设：**左金右粉异色瞳**、**右侧彩虹星发夹**、**右侧猫尾**、**左大腿腿环**全程未镜像漂移。
   - [ ] 核对首尾闭环与总时长：严格落在 **6.0s ～ 10.0s**，首尾干净一致（睡觉三段式除外）。
   - [ ] 更新本交接文档第三节 Checklist 进度，并清理任何多余中间临时文件。
