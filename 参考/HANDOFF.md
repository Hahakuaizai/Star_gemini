# 紫发猫耳少女桌宠 25 动作 WebP 制作交接文档（AI 执行规范版）

> **致接手 AI**：请**先完整阅读本文档**以及同目录下的工程技术指南 [AI_ANIMATION_PIPELINE_GUIDE.md](AI_ANIMATION_PIPELINE_GUIDE.md)（含 48 帧光流加中间帧、去毛刺抗锯齿、剥离星云黑块、老图集精修与防裁尾铁律），再结合同目录下的 [gemini_25_webp.md](gemini_25_webp.md) 动作分镜表继续推进剩余动作。本文档中的所有【铁律】均为前置实战踩坑总结，**必须严格执行，不得自行放宽标准或更改目录与命名规则**。

---

## 一、 当前工作区结构与资产清单

工作区根目录：`c:\Users\30894\Desktop\workspace\帧动画`

```text
gemini/
├── 参考/                           # 【核心文档与人设真值目录】
│   ├── HANDOFF.md                  # 本交接文档（当前进度：25 / 25 全部完成 🎉）
│   ├── AI_ANIMATION_PIPELINE_GUIDE.md  # 高流畅动画制作与老图集精修工程指南
│   ├── gemini_25_webp.md           # 25 动作完整分镜、粒子描写、配速与 Python 管线教程
│   ├── 形象基准图.webp             # 官方角色设定形象基准图（核对服设/配饰/异色瞳）
│   └── 图文/                       # 历史参考图与旧版教程归档（只读勿改）
├── 成品/                           # 【只放最终透明 WebP，严禁放中间文件；25 组 / 50 文件全部归档】
│   ├── 01idle（待机_动画).webp / 01idle_sheet（待机_图集）.webp
│   ├── ...                         # 01 ～ 25 全部已归档（详见第三节总进度表）
│   ├── 24spirit_butterfly（灵蝶环绕_动画）.webp / 24spirit_butterfly_sheet（灵蝶环绕_图集）.webp
│   └── 25curtsy_bow（屈膝礼仪_动画）.webp / 25curtsy_bow_sheet（屈膝礼仪_图集）.webp
├── 预览/                           # 网页交互播放器 + 01～25 白/黑双底色质检图（50 张 PNG）
│   ├── web_pet.html                # 零依赖本地网页播放器（file:// 直接打开，已接入全部 01～25）
│   └── 01idle_white.png / 01idle_black.png ... 25curtsy_bow_white.png / _black.png
├── 脚本/                           # 本地自动化管线（Python 3 + Pillow + NumPy + SciPy，无 OpenCV）
│   ├── build_webp.py               # 逐列智能黑隙切分 + 双层零黑边反预乘抠图 + 锚点提取
│   ├── expand_48.py                # 头宽比例锁死 + 双向金字塔光流插帧（无重影）+ 抛物线起跳
│   ├── 02process_sweeping.py       # 抗锯齿去黑边 + 星云黑块剥离 + 柔和光流插帧（主力复用模块）
│   ├── 10process_happy_jump.py ~ 25process_curtsy_bow.py  # 各动作精修/合成脚本
│   └── 04~09 专项精修脚本          # 04踏步 / 05思考 / 06忙碌 / 07~09睡觉联动专项脚本
├── 素材/                           # 原始分段图与基准锚点（按 <编号><英文短名> 分文件夹）
│   ├── 10happy_jump/
│   │   ├── anchor_stand.png        # 【全局唯一纯黑底标准站姿基准图，所有新动作必用！】
│   │   ├── 形象基准图.jpg          # 历史原立绘备份
│   │   └── part1.jpg / part2.jpg
│   └── 01idle/ ~ 25curtsy_bow/     # 全部 25 个动作的分段原图、接力锚点与局部补图
├── 待审/                           # 高帧率精修/多段联动专用对比预览区（01/04/05/06/07~09 联动 HTML 与中间诊断）
└── backup/                         # 【不可逆覆写保护备份区】覆写旧版成品/脚本/文档前的历史备份
```

---

## 二、 最高优先级纠偏与绝对铁律（严禁违背）

### 1. ⚠️ 异色瞳与配件唯一真值

形象**唯一真值以 `gemini/素材/10happy_jump/anchor_stand.png` 及 `gemini/参考/形象基准图.webp` 为准**：

- **画面左侧眼睛（角色自身右眼）**：**琥珀金黄瞳**
- **画面右侧眼睛（角色自身左眼）**：**宝石粉紫瞳**
- **彩虹五角星发夹**：**严格锁定在画面右侧鬓角**（绝不允许在中间帧镜像跑到左侧！）
- **蓬松紫粉猫尾**：**严格锁定从画面右侧伸出**（除非特定转身动作，绝不跳到左侧）
- **暗紫星形腿环**：**画面左侧大腿（角色自身右腿）**

### 2. 必须使用纯黑底锚点图生图（禁止直接喂白底原立绘）

- 生成任何新动作的第 1 段 `4×3` 图集时，**必须**将 `gemini/素材/10happy_jump/anchor_stand.png` 作为主要参考图输入。
- 原因：该图已是纯黑底 `#000000`、完整靴底、四边留白、严格匹配人设的标准站姿，能保证 AI 稳定输出纯黑底与统一画风比例，避免出现白底或靴子被裁切。
- 接力生成第 2 段时，使用 `python gemini/脚本/build_webp.py extract-anchors` 提取上一段第 1 格与最后 1 格，并与 `anchor_stand.png` 一同作为参考图输入。

### 3. 生图省配额铁律

- 调用生图工具时，**每次严格只生成 1 次**（Prompt 开头声明 `Make ONLY 1 generation attempt. Do NOT retry`），先用读图工具人工核查质量，**严禁自动循环重试**导致配额迅速耗尽。
- 常规新动作推荐生成 **2 段 `4×3`（24 关键帧）**，再用双向光流插帧平滑扩展到 **48 帧（`8×6`）**，兼顾流畅度与额度成本。
- 若仅某几格漏画道具/眼镜（如动作 18 第 7～16 帧），优先单次补画局部替换图并仅替换对应索引，避免整套重画。

### 4. 文件命名、备份与目录整洁铁律

- **不可逆覆写保护（最高优先级）**：修改或覆盖 `gemini/成品/` 下任何已有 `.webp` 文件前，**必须先备份至 `gemini/backup/`**。
- **统一命名规范**（严格带两位编号 + 英文短名 + 中文全角括号说明）：
  - 最终动态 WebP：`gemini/成品/<编号><英文短名>（<中文名>_动画）.webp`（例如 `21drink_milktea（喝奶茶_动画）.webp`）
  - 最终精灵图集：`gemini/成品/<编号><英文短名>_sheet（<中文名>_图集）.webp`（例如 `21drink_milktea_sheet（喝奶茶_图集）.webp`）
  - 双底色质检图：`gemini/预览/<编号><英文短名>_white.png` 与 `gemini/预览/<编号><英文短名>_black.png`
  - 原始素材目录：`gemini/素材/<编号><英文短名>/part1.jpg`、`part2.jpg`...
  - 处理脚本：`gemini/脚本/<编号>process_<英文短名>.py`
- **目录整洁**：严禁在 `gemini/` 根目录堆放临时文件；每次跑完脚本后必须清理 `gemini/脚本/__pycache__/`。

### 5. 画面与动画质量红线

1. **画幅与规范**：单格标准尺寸 **`384×384 px`**（48 帧 `8×6` = `3072×2304`，36 帧 `6×6` = `2304×2304`；高帧率专项精修版如 01/05/06/07/08/09 维持实际高清网格），无损透明 WebP（`lossless=True, quality=100, disposal=2, loop=0`）。
2. **时长与节奏**：常规新动作总时长严格落在 **6.0s ～ 10.0s**（48 帧舒展版推荐 **8.0s ～ 9.2s**），必须使用变速帧表（爆发/发力 `120～145ms`，常态过渡 `160～185ms`，停顿/定格/首尾 `215～250ms`）。
3. **安全留白与完整度**：生图人物占格高 `68%～72%`，四边留白 `≥12%`，严禁裁切呆毛、靴底、手尖、右侧猫尾或任何道具粒子；局部补靴时必须全宽羽化拼接 `ref_stand`，严禁截断猫尾。
4. **首尾闭环**：除睡觉三段式（`07sleep_start` / `08sleep_loop` / `09sleep_wake`）外，第 1 帧与最后 1 帧必须 100% 锁死为干净标准站姿（无道具、无眼镜、无粒子）。
5. **严禁半透明重影与硬切毛刺**：禁止用简单透明度叠化补帧，也禁止 `alpha >= 128` 硬二值切边；必须沿用 `02process_sweeping.py` / `expand_48.py` 的一致性门控光流 + 深紫边界去污 + 高斯抗锯齿平滑。
6. **眼镜规范**：细框圆金属镜，高透不遮挡异色瞳，且在动作内完成“无 → 戴上 → 摘下消散 → 无”完整闭环（注：动作 20 实际成品已按无眼镜版精修交付）。

---

## 三、 25 个动作总进度 Checklist（已完成 25 / 25 🎉）

> 详细分镜、粒子描写与配速见同目录 [gemini_25_webp.md](gemini_25_webp.md) 第四节。
> 注：部分 48 格（`8×6`）图集在导出动态 WebP 时因最后两帧像素完全相同，编码器自动合并尾帧并累加时长，实测动画帧数显示为 `47` 帧、总时长与图集播放一致；22～25 均采用 `64` 帧（`8×8`）多步光流插帧以获得超高流畅度。

| 状态 | 编号 | 动作名称 | 成品文件前缀 | 实测规格（图集 / 动画 / 时长） | 备注 |
| :--: | :--: | :--- | :--- | :--- | :--- |
| [x] | **01** | **待机呼吸休闲** | `01idle` | `384×384` `8×10` / 80帧 / 9.20s | **✅ 已重做纯白底超清全动80帧版**（`01process_idle.py` 基于 `01idle_v4/part1_white.jpg` 纯白底原画头宽锁定对齐：大幅屈膝身体下沉 `38px` + 飞机耳/左右单折耳 + 左右大歪头 + 右眼俏皮Wink + 大猫尾4周期波浪摇摆 + 12颗一闪一闪闪烁四芒星） |
| [x] | **02** | **打扫清理·利落站姿版** | `02sweeping` | `384×384` `8×6` / 47帧 / 8.87s | **✅ 已完成验收**（24关键帧+防跨边拉丝光流，星云黑块剥离 + 四芒星增强） |
| [x] | **03** | **专注写代码 👓** | `03coding` | `384×384` `8×8` / 64帧 / 7.67s | **✅ 已去重64帧精修验收**（`03process_coding.py` 剔除漏眼镜问号/重复帧与黑糊灯泡鬼影，17关键帧多步光流扩至64帧） |
| [x] | **04** | **原地轻快踏步** | `04roam` | `384×384` `8×6` / 48帧 / 7.00s | **✅ 已精修入库**（绝对原地零水平位移，48帧变速版） |
| [x] | **05** | **深度思考冒泡 👓** | `05thinking` | `384×384` `8×19` / 145帧 / 10.50s | **✅ 已精修验收**（下半身相位相关限幅3px + 粒子单侧取帧，无双眼镜重影） |
| [x] | **06** | **忙碌虚空点按调度** | `06working_busy` | `384×384` `8×18` / 144帧 / 15.84s | **✅ 已精修入库**（c46 原画补腿 + 48姿态各插2帧光流，关键140ms/过渡95ms） |
| [x] | **07** | **睡觉-开始** | `07sleep_start` | `384×384` `8×13` / 104帧 / 11.00s | **✅ 已精修验收**（固定缩放 0.7116，脸中心对齐+脚底366，尾帧直连 08 首帧） |
| [x] | **08** | **睡觉-循环** | `08sleep_loop` | `384×384` `8×18` / 137帧 / 10.55s | **✅ 已精修验收**（固定缩放 0.7377，首尾 100% 重合，亮紫白边 `Zzz`） |
| [x] | **09** | **睡觉-起来** | `09sleep_wake` | `384×384` `8×13` / 98帧 / 9.50s | **✅ 已精修验收**（固定缩放 0.7234，头部衔接 08 首帧保证三段丝滑联动） |
| [x] | **10** | **开心跃动庆祝** | `10happy_jump` | `384×384` `8×6` / 47帧 / 8.45s | **✅ 已精修验收**（深紫描边去毛刺抗锯齿，48格抛物线起跳光流，舒展节奏） |
| [x] | **11** | **被戳受惊炸毛** | `11poke_shock` | `384×384` `8×6` / 48帧 / 8.52s | **✅ 已精修验收**（废除硬二值切边消除毛刺，33关键帧+15光流补帧） |
| [x] | **12** | **害羞捂脸扭捏** | `12shy_praise` | `384×384` `8×6` / 48帧 / 8.62s | **✅ 已精修验收**（深紫描边去毛刺抗锯齿，36关键帧+12光流补帧） |
| [x] | **13** | **傲娇抱胸侧转** | `13tsundere` | `384×384` `6×6` / 36帧 / 6.97s | **✅ 已完成验收**（29手绘真过渡帧含 `part_turns.jpg` 桥接转身 + 7补帧，身高310px） |
| [x] | **14** | **鼠标拎起悬空晃动** | `14drag_float` | `384×384` `8×6` / 48帧 / 7.37s | **✅ 已完成验收**（双周期 ±15° 正弦钟摆 + 90°相位滞后发尾猫尾 + 提拉线，身高262px） |
| [x] | **15** | **挠痒咯咯大笑** | `15tickle_laugh` | `384×384` `8×6` / 48帧 / 7.11s | **✅ 已完成验收**（19关键帧+29光流补帧，全宽羽化补靴保尾，身高307～310px） |
| [x] | **16** | **报错慌张冒汗** | `16error_panic` | `384×384` `8×6` / 47帧 / 7.46s | **✅ 已完成验收**（24关键帧+24光流补帧，自动识别黑白底，全靴全尾零裁切） |
| [x] | **17** | **元气挥手打招呼** | `17wave_hello` | `256×256` `6×6` / 36帧 / 6.45s | **✅ 已完成验收**（白底抠图保护白衣蕾丝） |
| [x] | **18** | **轻快记笔记 👓** | `18note_taking` | `384×384` `8×6` / 47帧 / 8.84s | **✅ 已修复验收**（已用 `part1_glasses_fix.jpg` 补齐 p1[3..7] 眼镜，全靴保尾+去底线） |
| [x] | **19** | **下五子棋五连胜 👓** | `19gomoku` | `384×384` `8×6` / 47帧 / 9.17s | **✅ 已完成验收**（戴摘眼镜 👓 + 悬浮棋盘 + 金白五连星轨 + 庆祝星芒） |
| [x] | **20** | **穿针乞巧** | `20thread_needle` | `384×384` `8×6` / 47帧 / 9.20s | **✅ 已完成验收**（采用无眼镜精修版，剔除镜像废帧 `p1[3]` + 模板缩放淡出圆月防灰晕） |
| [x] | **21** | **喝星空珍珠奶茶** | `21drink_milktea` | `384×384` `8×6` / 47帧 / 8.54s | **✅ 已完成验收**（22干净关键帧+26光流补帧，紫粉星空奶茶液面递减+闭眼吐息星泡粉爱心+袖口擦嘴+星尘消散，身高锁死 `303～305px`） |
| [x] | **22** | **吃星星糖霜曲奇** | `22eat_cookie` | `384×384` `8×8` / 64帧 / 8.64s | **✅ 已完成验收**（22关键帧+42双向光流补帧，身高 302～305px，零裁切零脏边） |
| [x] | **23** | **凭空生花小魔术** | `23magic_flower` | `384×384` `8×8` / 64帧 / 8.81s | **✅ 已完成验收**（暗灰绿渐变底置黑精准切格 + Part2 Row1全宽补靴保尾 + 64帧光流，身高 302～306px） |
| [x] | **24** | **星光灵蝶环绕** | `24spirit_butterfly` | `384×384` `8×8` / 64帧 / 8.76s | **✅ 已完成验收**（替换原鲸鱼方案为发光紫粉星光灵蝶；14关键帧+50步光流闭环，身高 303～307px） |
| [x] | **25** | **优雅屈膝礼仪** | `25curtsy_bow` | `384×384` `8×8` / 64帧 / 8.65s | **✅ 已完成验收**（白底Part1+黑底Part2混编 + `align_curtsy_by_head` 锁头宽下蹲 + 双肩亮白四角星芒） |

---

## 四、 标准出图 Prompt 模板（已修正瞳色、路径与防印字）

生成任意动作第 X 段（`4×3` 共 12 格）时，按以下模板填写并**附上 `gemini/素材/10happy_jump/anchor_stand.png`**（第 2 段起同时附上一段提取的接力锚点）：

```text
IMPORTANT QUOTA RULE: Make ONLY 1 generation attempt. Do NOT retry.

Generate a high-resolution 4 columns × 3 rows (12-frame) sprite sheet on a PURE SOLID BLACK background (#000000) across ALL 3 rows, with NO grid lines, NO borders, NO text, NO numbers, NO watermarks, and NO ground shadows.

Reference images (STRICTLY match character design, scale, full-body boots, line thickness, and pure solid black background):
1) Standing baseline anchor: c:/Users/30894/Desktop/workspace/帧动画/gemini/素材/10happy_jump/anchor_stand.png
2) [If Part >= 2] Previous segment's final frame to continue smoothly from: <path_to_prev_last.png>

Character lock (DO NOT horizontally mirror):
- Chibi 2.5-head-tall purple-haired cat-ear anime girl, crisp Japanese cel-shaded style with clean dark purple outlines.
- FULL BODY WITH COMPLETE DARK PURPLE BOOTS fully visible in ALL 12 cells; character occupies ~68%-72% of cell height with at least 12% pure black margin on all 4 sides.
- Rainbow 5-pointed star hairclip strictly on VIEWER'S RIGHT side; fluffy purple-pink cat tail strictly on VIEWER'S RIGHT side; dark purple star leg ring on viewer's left thigh.
- Heterochromia strictly matching reference: VIEWER'S LEFT eye is amber-gold, VIEWER'S RIGHT eye is gem pink-purple.

Composition rules:
- 4 columns × 3 rows evenly spaced invisible grid (read left-to-right, top-to-bottom).
- Identical character height and head-to-body ratio across all 12 cells; feet centered horizontally at X=50%.
- Zero semi-transparent ghosting, zero motion blur, zero dark grey/black smoke clouds.

Action: <action_name> — Part <X> of <N> (DO NOT print any frame numbers on the image):
- Row 1 Col 1: ...
- ...
- Row 3 Col 4: ...

Particle & effect style: High-saturation, crisp-edged discrete sparkles/symbols only. NO blurry dark smoke halos.
```

---

## 五、 标准化制作与验收流程（每做一个动作必走）

1. **保存分段原图**：将生图结果存入 `gemini/素材/<编号><英文短名>/part1.jpg`、`part2.jpg`...
2. **提取接力锚点**（生成第 2 段前执行）：
   ```powershell
   python gemini\脚本\build_webp.py extract-anchors --segment "gemini\素材\<编号><英文短名>\part1.jpg" --out-dir "gemini\素材\<编号><英文短名>\anchors" --prefix "part1"
   ```
3. **抠图、对齐、光流补帧与导出 WebP**：
   - 新动作推荐直接复制参考 `gemini/脚本/18process_note_taking.py`、`20process_thread_needle.py` 或 `02process_sweeping.py` 编写 `gemini/脚本/<编号>process_<英文短名>.py`；
   - 若覆盖 `gemini/成品/` 下已有文件，**务必先备份到 `gemini/backup/`**；
   - 严格按规范命名输出 `gemini/成品/<编号><英文短名>（<中文名>_动画）.webp`、`_sheet（<中文名>_图集）.webp` 以及 `gemini/预览/<编号><英文短名>_white.png` / `_black.png`。
4. **接入网页播放器**：在 `gemini/预览/web_pet.html` 的 `ACTIONS` 数组中追加该动作配置（相对路径指向 `../成品/<编号><英文短名>_sheet（<中文名>_图集）.webp` 与 `../成品/<编号><英文短名>（<中文名>_动画）.webp`）。
5. **强制视觉验收（全部 PASS 才可勾选 Checklist）**：
   - [ ] 打开 `<编号><英文短名>_white.png` 检查：粒子外圈、腿间缝隙、猫尾弯折处、淡出道具**零黑边/零灰晕/零暗棕星云块**。
   - [ ] 打开 `<编号><英文短名>_black.png` 检查：深紫衣裙、短靴、轮廓线完整无缺口、无锯齿毛刺，粒子明亮清晰。
   - [ ] 核对每一格：无裁切靴底/呆毛/右侧猫尾/粒子，头顶粒子未串入上一行。
   - [ ] 核对人设：**左金右粉异色瞳**、**右侧彩虹星发夹**、**右侧猫尾**、**左大腿腿环**全程未镜像漂移。
   - [ ] 核对首尾闭环与总时长：常规动作严格落在 **6.0s ～ 10.0s**，首尾干净一致（睡觉三段式除外）。
   - [ ] 清理 `gemini/脚本/__pycache__/`，并同步更新本交接文档第三节 Checklist 进度。
