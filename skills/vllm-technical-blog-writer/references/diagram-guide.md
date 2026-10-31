# 技术图片规范

图片的目标是减少理解成本，而不是增加视觉效果。所有新图都要复刻作者过去用 draw.io 手绘的配图风格（样本见 `technical-blogs/assets/figures/` 下 `vllm-v1-overview/`、`vllm-platform-plugin-and-customop/`、`vllm-multimodal-vit-optimization/`、`vllm-multimodal-conv-acceleration/` 中的 `.svg`）。[`../assets/diagram-style-template.svg`](../assets/diagram-style-template.svg) 把这套风格整理成了可直接复用的 `<defs>` + `<style>` 和示例构图，画新图时先复制它的 `<defs>`/`<style>`，再按本文规则摆放元素。

## 作者画图风格（硬性要求）

### 画布与字体

- **背景**：纯白画布 `#ffffff`（第一个元素是铺满的白色 `<rect>`），不用圆角、渐变或浅灰底；四周留 20～50px 边距。
- **字体**：所有文字都用 Comic Sans MS，并保留中文回退（Comic Sans 不含中文字形）：

  ```css
  font-family: "Comic Sans MS", "Comic Sans", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", cursive;
  ```

- **字号层级**：
  - 节点文字和边标签用 12px 常规字重，文字颜色 `#000000`；深色填充节点内用白字 `#ffffff`；
  - 容器标题用 12px **粗体**，放在容器左上角（距左边 10～16px，距顶 16px），不居中；
  - 并排对比面板的标题用 16px 常规字重，居中放在面板正上方，可以用中文（如“文本理解 / 多模态理解”）；
  - 整个系统或项目的大标题（如 *vLLM*、*vLLM-Ascend*）用 20px 粗斜体，居中放在区域顶部；
  - 不使用其他字号，不加下划线、阴影或描边字。
- **文字内容**：节点里写源码中真实的类名、函数名、变量名（如 `add_request()`、`EngineCoreRequest`、`process_outputs_socket()`），函数名带 `()`；伪代码或配置写成一到三行，如 `1. path = get_xxx_cls()`。
- **行内强调**：节点或边标签里需要突出的标识符（参数、关键字段、注册名）用红色 `#cc0000` 的 `<tspan>`，其余文字保持黑色，例如 `register(<tspan class="hl">"Aaa"</tspan>)`。

### 配色

使用 draw.io 标准配色对，填充色和描边色必须成对出现：

| 角色 | 填充 / 描边 | 文字 | 典型用法 |
| --- | --- | --- | --- |
| 关键函数 / 算子 / 核心模块 | `#a20025` / `#6f0000` | 白 | `add_request()`、`RMSNorm`、`Attention`、`NPUPlatform`，作者最常用的“主角色” |
| 输出路径函数 | `#1ba1e2` / `#006eaf` | 白 | `get_output()`、`process_outputs()` |
| 大模型主干长条 | `#6a00ff` / `#3700cc` | 白 | `LLM AR-decoder`、`DiT`、`Codec` |
| 框架 / 系统节点 | `#76608a` / `#432d57` | 白 | `CustomOp`、`vLLM Plugin System`、`current_platform` |
| 绿色实色 | `#60a917` / `#2d7600` 或 `#008a00` / `#005700` | 白 | `CudaPlatform`、GPU |
| 浅蓝 | `#dae8fc` / `#6c8ebf` | 黑 | 输出侧容器、音频 token |
| 浅绿 | `#d5e8d4` / `#82b366` | 黑 | 视觉 token、Encoder |
| 浅黄 | `#fff2cc` / `#d6b656` | 黑 | 文本 token、Tokenizer |
| 浅橙 | `#ffe6cc` / `#d79b00` | 黑 | Tensor 分块 |
| 浅红 | `#f8cecc` / `#b85450` | 黑 | 输入侧容器（如 `Processor`） |
| 浅紫 | `#e1d5e7` / `#9673a6` | 黑 | 入口方法（`generate()`）、客户端容器 |
| 浅青 | `#b1ddf0` / `#10739e` | 黑 | 队列、output socket |
| 浅鲑红 | `#fad9d5` / `#ae4132` | 黑 | input socket |
| 米白 | `#f9f7ed` / `#36393d` | 黑 | 容器内的方法说明框、代码片段框 |
| 灰色容器 | 外层 `#eeeeee` / `#36393d`，内层 `#f5f5f5` / `#666666` | 黑 | 进程、类、子模块的嵌套容器 |

- **渐变**：表示数据对象或输入输出实体（`Prompt`、`RequestOutput`、`EngineCoreRequest`、`Text Tokenizer`、`Vision Encoder`）时，用从上到下的 draw.io 渐变：蓝 `#dae8fc→#7ea6e0`、红 `#f8cecc→#ea6b66`、黄 `#fff2cc→#ffd966`、绿 `#d5e8d4→#97d077`、灰 `#f5f5f5→#b3b3b3`；大面积背景容器可用橙 `#ffcd28→#ffa500`（如 `DecoderLayer` 外框）。
- 同一张图里，同一语义始终用同一颜色；输入侧偏红、输出侧偏蓝是作者的固定习惯。
- 不用 Tailwind 风格的灰蓝色、低饱和莫兰迪色、霓虹色或纯黑填充大面积区域。

### 线条与箭头

- 连线 1px，使用正交折线（只有水平和竖直线段，拐角为直角），箭头为 draw.io classic 实心箭头（模板中的 `marker`），箭头颜色与线同色。
- **按数据流着色**：
  - 请求 / 输入路径：红 `#cc0000`，通常在图左侧自上而下；
  - 输出 / 返回路径：蓝 `#0066cc`，通常在图右侧自下而上；
  - 一般调用、加载、创建关系：黑 `#000000`；
  - 两条互斥的分发路径：深绿 `#006600` 与深红 `#660000`。
- 边标签用 12px 黑字，写一个动作或数据名（`load`、`init`、`call`、`set`、`create`），放在线段旁边，不加背景框。
- **块箭头**：模型结构图中“输入进入模块”用短而粗的深红块箭头（`#a20025` / `#6f0000`，约 16×32px），下方写输入名称（`Text Prompt`、`Images`）。
- **步骤箭头**：数据变换流程用白色空心粗箭头（白填充、黑描边），旁边放带圆圈的步骤编号（①②③）和对应的代码，如 `unfold(2, K1, K1)`。

### 形状与构图

- **默认直角矩形**；圆角（`rx=6`）只用于方法说明框、Tokenizer / Encoder 等“组件卡片”；进程边界、对比面板用大圆角（`rx≈40`）+ 虚线。
- **嵌套容器**是作者最核心的构图方式：`Process`（虚线无填充）→ 类（灰色实线容器）→ 子模块（浅色容器）→ 函数（实色节点）。每层容器标题写在左上角，内层与外层之间留 20px 内边距。
- **尺寸网格**：所有坐标和尺寸对齐 10px 网格。普通节点高 40px（次要节点 30px），宽 120 / 140 / 160 / 180 / 200px；同一列节点等宽、中心对齐；同级节点间距 20～30px。
- **双通道布局**：调用链图采用“左列红色请求下行、右列蓝色结果上行”的两条竖直通道，跨进程的 socket / queue 放在进程边界外。
- **并排对比**：多个方案或模型对比时，横向排列多个等高的虚线圆角面板，标题在面板上方，面板内同类元素保持同一行高度，方便横向比较。
- **Token 序列**：用一排 20×20 的小方块表示 token，颜色与产生它的 Encoder / Tokenizer 一致（文本黄、视觉绿、音频蓝、生成紫），方块间距 10px，按模态分组。
- **Tensor / 数据布局**：用 40×40 的格子画矩阵并在格内写编号；用 2～3 层向右上错位的灰色渐变矩形表示多通道；用不同浅色区分 patch / 分块；形状说明写在图下方，如 `input_size: (1, 3, 6, 6)`；用花括号标注一组对象的共同属性。
- 设备类节点（CPU / GPU / NPU）可以用 3D 立方体，其余不用图标、emoji、阴影或拟物元素。
- 宽度控制在 660～1050px；纵向调用链图可以更高（作者常用 740px 宽的长图），但一张图只回答一个主要问题。

## 选择图形

- **架构图**：展示模块边界、层次和依赖关系，优先用嵌套容器。
- **流程图 / 调用链图**：展示请求从入口到底层的调用路径，优先用双通道布局。
- **时序图**：展示进程、线程或组件之间按时间发生的调用与返回。
- **状态图**：展示 Scheduler、FSM、Request 等对象的状态迁移。
- **数据布局图**：展示 Tensor Shape、Shard、Patch、KV Block 或内存排布，按上面的 Tensor 规则画。
- **表格**：只有二维对比关系时用表格，不要为表格内容画图。

复杂主题拆成“整体架构”和“局部流程”两张图，不要塞进一张超大图。

## 绘制方式

- **默认直接手写 SVG**，基于 `assets/diagram-style-template.svg`：复制它的 `<defs>`（箭头 marker、渐变）和 `<style>`（字体、节点类、连线类），再用 `<rect>` + `<text class="c">` 摆放节点、用 `<path d="M.. V.. H..">` 画正交连线。这是唯一能稳定复刻作者手绘风格的方式。
- 文字居中用 `text-anchor="middle"` + `dominant-baseline="central"`，y 坐标取节点的垂直中心；两行文字分别放在中心 ±9px。
- Comic Sans 英文字宽较大，12px 下按每个字符约 7px、每个汉字约 12px 估算宽度，节点宽度至少比文字宽 20px。
- 只有时序图等手写成本过高的场景才用 Mermaid。此时在 `.mmd` 第一行加入下面的初始化配置，让颜色和字体尽量贴近手绘风格，渲染后再把 SVG 背景改为纯白：

  ```text
  %%{init: {"theme": "base", "themeVariables": {"fontFamily": "Comic Sans MS, Comic Sans, Microsoft YaHei, PingFang SC, cursive", "fontSize": "12px", "background": "#ffffff", "primaryColor": "#a20025", "primaryTextColor": "#ffffff", "primaryBorderColor": "#6f0000", "secondaryColor": "#dae8fc", "tertiaryColor": "#f5f5f5", "lineColor": "#000000", "actorBkg": "#a20025", "actorTextColor": "#ffffff", "actorBorder": "#6f0000", "signalColor": "#000000", "noteBkgColor": "#fff2cc", "noteBorderColor": "#d6b656"}}}%%
  ```

## 文件组织

每篇新文章：

```text
outputs/<article-slug>/
├── <article-slug>.md
└── images/
    ├── <name>.svg
    └── <name>.mmd   # 仅 Mermaid 图需要
```

- 文件名使用简短的 lowercase `kebab-case` 英文。
- 手写 SVG 本身就是源文件；Mermaid 图保留 `.mmd` 源文件和 `.svg` 渲染结果。
- 只有发布平台不支持 SVG 或用户明确要求时再生成 PNG。
- Markdown 使用相对路径：`![图：简短说明](./images/<name>.svg)`。
- 场景二的 `images/` 位于被编辑文章旁边，不复制文章到 skill 的 `outputs/`。

## 内容约束

- 节点 ID / 文件名用 ASCII，展示文本可以用中文；类名、函数名保持源码原样。
- 节点文字控制在两行内；详细解释放到正文。
- 同层节点保持同一抽象级别。
- 边上只写动作、数据或条件，不写完整句子。
- 时序图参与者通常不超过 6 个；超过时拆图。
- 不依赖颜色作为唯一语义，同时要有文字、位置或线型（虚线容器、箭头方向）区分。
- 不使用 emoji。

## 渲染与检查

1. 将 SVG（以及 Mermaid 源）保存到目标文章的 `images/`。
2. 用浏览器（如 Playwright + Chromium）或 `rsvg-convert` 把 SVG 渲染成 PNG 查看；Mermaid 优先用已安装的 `mmdc`。不要只写源文件后假装图片已完成。
3. 打开渲染结果，对照 `assets/diagram-style-template.svg` 和作者原图检查：
   - 白底、Comic Sans 字体、12px 字号层级是否一致；
   - 配色是否使用上面的配色对，输入红 / 输出蓝的约定是否遵守；
   - 连线是否为 1px 正交折线，箭头方向是否正确，是否大量交叉；
   - 中文是否缺字，标签是否截断、溢出或重叠；
   - 坐标是否对齐 10px 网格，同列节点是否等宽对齐；
   - 图是否与当前代码版本一致。
4. 修正后再写入 Markdown 链接。

不要引用不存在的文件，也不要用 ASCII 图冒充最终图片。

## 内容检查

每张图都应满足：

- 从标题或图注能看出它解释的问题。
- 节点和边可从源码或可靠资料中核验。
- 正文明确告诉读者从哪里开始看、关键路径是什么。
- 删除任意装饰元素都不会损失技术含义。
- 删除整张图会明显增加正文的解释成本；否则不需要这张图。
