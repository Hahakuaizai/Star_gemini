# 动作 21（喝星空珍珠奶茶）Part 2 · Gemini 网页端生图指导

> **用途**：动作 21 的第 1 段原图（`part1.jpg`）及接力锚点已准备就绪并验收通过。本指导专用于在 **Gemini 网页端** 生成衔接第 1 段的 **第 2 段 `4×3`（12 格）精灵图集**，生成后直接交给本地脚本合成 48 帧透明 WebP 成品。

---

## 一、 在 Gemini 网页端上传参考图（严格按此 2 张上传）

1. **图 1（全局标准站姿纯黑底基准图）**：
   `c:\Users\30894\Desktop\workspace\帧动画\gemini\素材\10happy_jump\anchor_stand.png`
2. **图 2（动作 21 第 1 段末格接力图）**：
   `c:\Users\30894\Desktop\workspace\帧动画\gemini\素材\21drink_milktea\anchors\part1_last.png`

> **补充参考**：如需核对角色服饰、发饰或左金右粉异色瞳细节，可对照人设形象基准图 `c:\Users\30894\Desktop\workspace\帧动画\gemini\参考\形象基准图.webp`（生图背景与全身比例仍以纯黑底 `anchor_stand.png` 和 `part1_last.png` 为主输入）。

---

## 二、 推荐英文 Prompt（可直接整段复制发送）

```text
CRITICAL LAYOUT & BACKGROUND REQUIREMENTS (TOP PRIORITY):
- Generate ONE 4 columns × 3 rows (12-cell) sprite sheet on a PURE SOLID FLAT BLACK (#000000) BACKGROUND EVERYWHERE across ALL 3 rows — ZERO white bands, ZERO grey backgrounds, ZERO circular frames behind the character, ZERO grid lines, ZERO text, ZERO numbers, ZERO watermarks, ZERO floor shadows.
- WIDE FULL-BODY ZOOMED-OUT VIEW in ALL 12 cells: the small chibi girl MUST occupy ONLY ~65% of each cell's height, leaving generous pure solid black (#000000) empty space above her cat ears/ahoge AND below BOTH complete dark purple boots. NEVER zoom in or crop legs/boots in ANY row.
- NO GLASSES in ANY cell! Her eyes must remain bare and unobstructed throughout all 12 cells.

Reference images (STRICTLY match character scale, full-body boots, proportions, line thickness, and pure solid black background):
1) First uploaded image: Standing baseline anchor (anchor_stand.png)
2) Second uploaded image: Previous segment's final frame to continue smoothly from (part1_last.png)

Character lock (DO NOT horizontally mirror):
- Chibi 2.5-head-tall purple-haired cat-ear anime girl, Japanese cel-shaded style with clean dark purple outlines RGB(68, 48, 98).
- FULL BODY WITH COMPLETE DARK PURPLE BOOTS, ahoge, and fluffy purple-pink tail fully visible in ALL 12 cells with at least 12% pure solid black margin on all 4 sides.
- Rainbow 5-pointed star hairclip strictly on VIEWER'S RIGHT temple; fluffy purple-pink cat tail strictly on VIEWER'S RIGHT side; dark purple star leg ring on viewer's left thigh.
- Heterochromia strictly matching reference: VIEWER'S LEFT eye is amber-gold, VIEWER'S RIGHT eye is gem pink-purple.

Action: Drinking Starry Boba Milk Tea ("喝星空珍珠奶茶" — NO glasses) — Part 2 of 2 (4 columns × 3 rows, read left-to-right, top-to-bottom):
- Row 1 Col 1: Smoothly continuing from the second reference image (part1_last.png): full-body standing, happy closed-eye smile, cradling the mostly-empty starry boba cup (~25% left) in left hand while raising right sleeve hand close to mouth corner, small star bubbles & mini pink heart floating cleanly.
- Row 1 Col 2: Full-body standing, playfully wiping the corner of her mouth with her right sleeve cuff while squinting/smiling sweetly, bubbles and mini pink heart floating upward.
- Row 1 Col 3: Full-body standing, pausing with a cute satisfied expression right after wiping mouth corner, bubbles fading away.
- Row 1 Col 4: Full-body standing, lowering right hand to hold the cup lightly with both hands, opening bare heterochromia eyes (viewer's left amber-gold, viewer's right pink-purple) with a satisfied smile.
- Row 2 Col 1: Full-body standing, the starry boba cup begins glowing with pale-purple and warm-gold sparkles ready to vanish.
- Row 2 Col 2: Full-body standing, the milk tea cup dissolves into shimmering pale-purple stardust sparkles as hands gently lower.
- Row 2 Col 3: Full-body standing, stardust mostly vanished, hands naturally lowering toward sides with gentle smile.
- Row 2 Col 4: Full-body standing, stardust completely gone, arms relaxed at sides, sweet warm smile.
- Row 3 Col 1: Clean standard front-facing standing pose, quick cute blink (both eyes gently closed, no cup, no particles).
- Row 3 Col 2: Clean standard front-facing standing pose, eyes open (viewer's left amber-gold, viewer's right pink-purple) with soft smile.
- Row 3 Col 3: Clean standard front-facing standing pose, calm steady breathing returning to neutral.
- Row 3 Col 4: Exact clean standard front-facing standing pose matching anchor_stand.png (arms down, neutral smile, complete boots and tail).

Particle style: High-saturation, crisp-edged discrete sparkles, bubbles, and mini heart only on pure #000000 black background.
```

---

## 三、 中文版备用 Prompt（效果不佳时可切换）

```text
请严格参考上传的两张图：第 1 张为全局纯黑底标准站姿基准图（anchor_stand.png），第 2 张为动作 21 第 1 段最后一格接力图（part1_last.png）。绘制一张【4列 × 3行（共12格）】高清无网格线精灵图集，全 3 行必须保持 100% 纯黑背景（#000000），严禁出现白条、灰底、背景圆圈、文字、数字编号、水印、边框或地面阴影。

【最高优先级防裁切与禁眼镜锁】
- 全部 12 格必须为全身远景：人物高度仅占单格高度约 65%，头顶呆毛上方与双脚暗紫短靴下方都必须留出充足纯黑留白，严禁放大成半身或裁切靴底！
- 全程 12 格绝对不戴眼镜！必须保持裸眼清晰展示异色瞳。
- 严格锁定人设（严禁镜像）：Q版2.5头身紫发猫耳少女，画面右侧鬓角彩虹五角星发夹、画面右侧紫粉蓬松猫尾、画面左侧大腿星形腿环；异色瞳【画面左眼琥珀金黄、画面右眼宝石粉紫】。

【本段 12 格分镜（从左到右、从上到下）】
- 第1行第1格：无缝承接第 2 张参考图，闭眼幸福微笑，左手轻抱剩余约25%液面的星空珍珠奶茶杯，右手抬起凑近嘴角准备擦嘴，旁飘小星星气泡与迷你粉爱心。
- 第1行第2格：抬起右手袖口俏皮轻擦嘴角，眯眼甜笑，小星星气泡与迷你粉爱心轻盈上浮。
- 第1行第3格：右手袖口擦完嘴角微顿，露出满足可爱表情，气泡渐淡。
- 第1行第4格：放下右手轻捧奶茶杯，睁开左金右粉异色瞳满足微笑。
- 第2行第1格：手中奶茶杯泛起淡紫与暖金星光准备消散。
- 第2行第2格：奶茶杯化作淡紫星尘开始消散，双手微放。
- 第2行第3格：淡紫星尘消散大半，双手自然垂落身体两侧，温和微笑。
- 第2行第4格：星尘完全散尽，双臂自然垂落，甜美微笑。
- 第3行第1格：干净正面站立，轻快闭眼眨眼（无杯子、无粒子）。
- 第3行第2格：干净正面站立，睁开左金右粉异色瞳温和微笑。
- 第3行第3格：干净正面站立，平稳呼吸归位。
- 第3行第4格：干净标准正面站立（与第 1 张参考图 anchor_stand.png 完全一致）。
```

---

## 四、 出图自检与保存路径

在网页端生成后，先目视检查以下 5 项：
1. **纯黑底**：全 3 行均为纯黑 `#000000`，无灰底、白条、格线或数字印字；
2. **全身无裁切**：第 3 行双脚暗紫短靴完整，呆毛与右侧猫尾零截断；
3. **无眼镜 + 人设不镜像**：全程无眼镜，左金右粉异色瞳、右侧彩虹星发夹、右侧猫尾、左大腿腿环全程正确；
4. **首尾衔接自然**：第 1 行第 1 格顺利承接 `part1_last.png` 擦嘴动作，第 2 行奶茶杯星尘消散完整，第 3 行干净收尾；
5. **保存位置**：确认无误后下载保存为：
   `c:\Users\30894\Desktop\workspace\帧动画\gemini\素材\21drink_milktea\part2.jpg`

> **本地脚本对接提示**：`part2.jpg` 就位后，接手 AI 可直接编写并运行 `gemini/脚本/21process_drink_milktea.py`（注意跳过 `part1.jpg` 中漏画奶茶杯的 `p1[8]`，并在平滑过渡段插双中点凑齐 `48` 帧），导出后验收白/黑底预览并更新 `web_pet.html` 与 `HANDOFF.md`。
