---
pr: 57379
repo: vllm-project/vllm
head_sha: 91b702900d2c95f850c75a822595ab6a252bfbd5
verdict: needs-work
reviewed_at: 2026-11-26T08:36:06Z
skill: vllm-vit-cg-pr-review
main_sha: 632ba4f
---

# PR #57379: [Model][Perf] Add LLaVA-OneVision encoder CUDA graph support

> **Author**: @trueoneplusone | **State**: OPEN | **Date**: 2026-09-17（最近一次 rebase 2026-11-17）
> **Branch**: `llava-onevision-encoder-cudagraph` → `main` | **Labels**: documentation, multi-modality, nvidia
> **Changes**: +720 -4 lines across 4 files | **ViT CG 相关性**: 完全相关 | **类型**: 模型集成
> 本报告分两部分：§1–6 为 PR 详细总结，§7–8 为 ViT CUDA graph review 意见（对照 vllm main `632ba4f`）。

---

## 1. 总结 (Summary)

本 PR 为 `LlavaOnevisionForConditionalGeneration` 实现 `SupportsEncoderCudaGraph`，图片和视频都支持。图里放的是 SigLIP vision tower（含设备侧归一化 `input_norm`）加上 multimodal projector；anyres 的 unpad / 合并 / newline 拼接和视频的 bilinear pooling 留在图外，由 `postprocess_encoder_output` 在 eager 侧完成。为了让每次 replay 的 tile 数与输入严格一致，PR 额外引入一个 "padding tiles" capture axis，并在模型里复刻了 manager 的 budget 生成逻辑来推导这个轴。实现思路正确，CPU 单测充分，但当前 head 没有 GPU 验证。capture axis 的设计在 main 上已有更简单的现成方案（path 的 `min_token_budget`）：默认配置下它会多 capture 约 39 个永远用不到的 graph。

## 2. 背景与动机 (Background & Motivation)

本 PR 属于 tracker [#38175](https://github.com/vllm-project/vllm/issues/38175) 的模型集成任务，作者在 tracker 评论中认领了 LLaVA-OneVision。计划是 capture vision tower + projector，any-resolution 合并保持 eager，同时覆盖图片和视频路径。LLaVA-OneVision 的视觉输入以固定 384×384 tile 为单位：图片为 1 个 base tile 加 anyres 网格 tile（`anyres_max_9`），视频的每一帧也是一个 tile。每个 tile 经 SigLIP（patch 14 → 27×27 = 729 token）编码，因此 encoder 侧 token 数总是 729 的整数倍。这与 Qwen 系的 varlen patch 序列不同，也是本 PR 不能直接套用 Qwen3-VL 模板的原因。

PR 描述给出的收益来自 9 月 RTX 5090 上的历史测量：1/2/4/8 tile 的 encoder 延迟加速 5.92×/3.77×/1.99×/1.59×，首 token 代理延迟从 134.03 ms 降到 78.48 ms，eager 与 graph 输出 `max_abs=0.0`。11 月的两次维护 rebase 只做了 CPU 验证（18 passed），作者明确写明 "CUDA graph replay and image/video E2E on this revision await accelerator validation"。

## 3. 代码修改分析 (Code Change Analysis)

### 3.1 修改的模块

| 文件 | 改动 |
|------|------|
| `vllm/model_executor/models/llava_onevision.py` (+394 -4) | 实现协议全部方法；新增 token 计数（`_get_image_output_tokens` / `_get_video_output_tokens`）、tile 计数、capture-axis 推导（`_encoder_cudagraph_token_budgets` / `_encoder_padding_axis_values` / `_encoder_padding_axis`）；`__init__` 中保存 `self.vllm_config` |
| `tests/models/multimodal/test_llava_onevision_cudagraph.py`（新增 306 行） | 18 个 CPU 单测，用 `object.__new__` + `SimpleNamespace` 构造假模型，验证 item spec、capture/replay buffer dtype、postprocess 与 eager merge 等价、budget 与 manager 一致 |
| `tests/models/multimodal/generation/test_vit_cudagraph.py` (+19) | 新增 `llava_onevision` case：`llava-onevision-qwen2-0.5b-ov-hf`，图片 + 4 帧视频，显式 budget `[8192]`，`core_model` |
| `docs/design/cuda_graphs_multimodal.md` (+1) | 支持表新增一行：Image ✅︎ / Video ✅︎ / Multi-Path ❌︎ |

### 3.2 架构 / 流程图

```mermaid
flowchart TD
    A[mm_kwargs: pixel_values 或 pixel_values_videos] --> B[get_encoder_cudagraph_item_specs<br/>input_size=tile 数<br/>path tokens = tile 数 × 729<br/>output_tokens = 最终 LLM token 数]
    B --> C{manager: 总 tile token<br/>≤ 最大 budget 且 item 数 ≤ max_batch_size?}
    C -- 否 --> E[encoder_eager_forward<br/>同一 SigLIP + projector]
    C -- 是 --> D[select_encoder_cudagraph_items<br/>裁掉多余 tile，计算 padding 轴 key]
    D --> F[(budget, padding) 对应的 graph<br/>恰好 bucket_tiles - padding 个 tile]
    F --> G[replay: copy pixel_values → SigLIP+input_norm → projector]
    E --> H[postprocess_encoder_output]
    G --> H
    H --> I{modality}
    I -- image --> J[_merge_image_patch_embeddings<br/>spatial_unpad + newline（eager）]
    I -- video --> K[apply_pooling bilinear + image_newline（eager）]
```

```mermaid
graph LR
    subgraph 模型内复刻的 manager 逻辑
        M1[_encoder_cudagraph_token_budgets<br/>调用 EncoderCudaGraphManager._generate_budgets] --> M2[_encoder_padding_axis_values<br/>capture_axes=(0..max_padding)]
        M1 --> M3[_encoder_padding_axis<br/>每个 batch 重新生成 budget 列表]
    end
    M2 --> CFG[get_encoder_cudagraph_config]
    M3 --> SEL[select_encoder_cudagraph_items]
```

### 3.3 关键实现细节

- **graph 区域**：`encoder_cudagraph_forward` = `_image_pixels_to_features`（`input_norm` + SigLIP，`feature_select_strategy`）+ `multi_modal_projector`，输出 `flatten(0, 1)` 成 `[tiles × 729, hidden]`。图片和视频共用同一 graph 集合；视频的 pooling 改为在 postprocess 中执行（eager 侧顺序本来就是 projector → pooling，二者等价）。
- **token 双口径**：`path_output_tokens["default"] = tile 数 × 729`（原始 ViT token，用于选 budget），`output_tokens` 是 unpad/pooling 后的 LLM token 数（用于 scatter 与校验）。postprocess 中若长度不符会 `raise ValueError`。
- **capture axis**：每个 budget 按 `bucket_tiles = ceil(budget/729)` capture，轴值 `p` 表示少放 p 个 tile，replay 时 `p = bucket_tiles - 实际 tile 数`。这样 replay 的 tile 数与 capture 完全一致、没有 padding tile。代价是模型必须预知 manager 会生成哪些 budget，于是复刻了 manager 的 auto-inference 分支并调用私有静态方法 `_generate_budgets`。
- **tile 裁剪**：`select_encoder_cudagraph_items` 用 `pixel_items[i][:tile_counts[i]]` 去掉 batch padding tile。eager 路径会先编码全部 tile 再在 merge 中 `[:num_patches]` 截断，两者输出等价，graph 路径还省掉了这部分计算。
- **设备侧归一化**：capture 时按 `self.input_norm.input_dtype or dtype` 分配 dummy（`uint8` 或模型 dtype），与 replay 的 `pixel_values` dtype 一致，满足共享前缀 buffer 的 dtype 约束。
- **其他**：声明 `supports_encoder_cudagraph = True`；实现 `get_max_frames_per_video()`（未被调用）；`get_encoder_cudagraph_budget_range` 返回 `(729, min(max_num_batched_tokens, max_model_len))`。

## 4. 涉及的技术原理 (Technical Principles)

**Budget 式 encoder CUDA graph**：manager 按 token budget 预先 capture 多个 graph，运行时每个 path 每个 batch 只 replay 一次"能装下整个 batch 的最小 budget"。超出最大 budget（默认 `encoder_cudagraph_max_token_budget=5120`）或超过 `max_batch_size` 的 batch 整体走 eager。main 上同一 (path, capture-axis) 组内的 budget 按从大到小 capture，并共享前缀 buffer，所以 buffer 显存约为 1× 最大 budget，但每个 graph 仍各自占用 graph pool 并增加启动 capture 时间。

**固定尺寸 item 的 path 预算**：`EncoderCudaGraphPathConfig(min_token_budget=N)` 声明"该 path 的 item 都是 N token 的整数倍"，manager 就改用 `_item_multiple_budgets(N, max_budget)`：1–16 个 item 逐个计数，之后按 1/8 倍频程，budget 都是 N 的倍数，16 个以内的 batch 能精确放下（#65084 / #65227 为 DeepSeek-OCR、Step3-VL 引入）。LLaVA-OneVision 的 tile 恰好是 729 token 的固定单位。

**Capture axes**：`EncoderCudaGraphConfig.capture_axes` 在 token budget 之外再加离散维度，graph 数 = budget 数 × ∏ 轴长。模型须在 `select_encoder_cudagraph_items` 的返回值中写入 `ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG`。这个机制面向"同样 token 数但形状不同"的输入（如 MiniCPM-V 的次级轴），超过 64 个 graph 时 manager 会 warn。

**LLaVA-OneVision anyres**：图片先按 `image_grid_pinpoints` 选出最佳网格，切成 base tile 加网格 tile 分别编码，再 unpad 回原始宽高比，必要时 bilinear 下采样（`anyres_max_9`），每行末尾追加 `image_newline`。视频每帧单独编码后做 2× bilinear pooling（27→14），末尾加一个 newline。这些步骤的输出形状与原图尺寸相关，因此 PR 把它们留在图外是合理选择。

## 5. 评论区讨论亮点 (Discussion Highlights)

- 目前没有任何 maintainer review；claude[bot] 因 fork PR 未执行 review，CodeRabbit 等自动检查为绿。
- 作者自评（09-18）修了两个边界：`image_sizes=None` 的多 tile 图片 token 计数与 eager 对齐；只指定 `encoder_cudagraph_max_vision_items_per_batch` 时 capture-axis 的 budget 推导与 manager 对齐。第二个修复本身说明，"在模型里复刻 manager budget 逻辑"已经出过一次不同步（11-17 的说明里还提到旧 grid `[2000, 4000, 8000, 10000]` 导致单测失败）。
- mergify 在 11-14 报冲突，作者已于 11-17 rebase 到 `f3f0a014`；之后 main 上 #66067（默认 budget range / mixin）、#68831（`eager_fallback_fast_path`）改过框架，当前 head 未与之冲突，但距 main 已有 755 个提交。
- PR 描述包含 AI 辅助声明（ChatGPT / Codex 参与实现、测试与维护）。

## 6. 风险与潜在问题 (Risk Analysis)

| 风险 | 严重程度 | 说明 |
|------|---------|------|
| 冗余 graph（capture 时间 / graph pool） | Medium | 默认配置 capture 46 个 graph，只有 7 个可达，见 §7 第 1 条 |
| 模型依赖 manager 私有实现 | Medium | 模型调用 `_generate_budgets` 并复刻分支，manager 一改就悄悄失配，见 §7 第 1 条 |
| 当前 head 无 GPU 验证 | Medium | 性能、parity 数字都是 9 月旧代码上的结果，见 §7 第 2 条 |
| 视频 / 大图默认几乎全走 eager | Medium | 默认上限 5120 token ≈ 7 个 tile，见 §7 第 4 条 |
| 新单测不在 CI 中运行 | Medium | 缺 `core_model` 标记，见 §7 第 3 条 |
| 数值正确性 | Low | graph/eager 共用 `_image_pixels_to_features` 与 projector，postprocess 复用 eager merge，等价性有保障 |
| 输出别名 | Low | postprocess 在 `clone=True` 时对 merged 结果 `clone()`；merge 自身也产生新张量 |

---

## 7. Review 意见 (Findings)

对照 vllm main `632ba4f`。我逐一读过 PR head 中的 capture / replay / eager / postprocess 实现，以及 main 上 `EncoderCudaGraphManager` 的 `_capture_budget_graph`、`_execute_local`、`_select_items`、`_item_multiple_budgets`。数值等价性没有发现问题：graph 与 eager 走同一个 `_image_pixels_to_features` 和 projector，postprocess 直接复用 eager 的 `_merge_image_patch_embeddings` / `apply_pooling`。主要问题集中在 capture-axis 设计、验证证据和覆盖范围。

**集成清单**

| 项 | 状态 | 说明 |
|---|---|---|
| 协议实现 | ⚠️ | 方法齐全且签名与 main 一致；capture-axis 方案可用 path `min_token_budget` 替代，另有冗余成员 |
| 文档支持表 | ⚠️ | 已加行，但未按字母序；Video ✅︎ 未说明默认配置下只覆盖 ≤7 帧 |
| example 注册 | ❌ | `MODELS_SUPPORT_VIT_CUDA_GRAPH` 未加入 `llava-onevision` |
| `test_vit_cudagraph.py` | ✅ | 0.5B 小模型，图片 + 视频，`core_model` |
| 单测 | ⚠️ | 18 个 CPU 单测，但不会在 CI 中运行 |
| benchmark | ⚠️ | 历史数字，未给脚本，不可复现 `[unverified]` |
| 精度 | ❌ | 无 MMMU 等精度结果，也没有 example 的 eager vs graph 输出对比 |

| 意见类型 | 数量 |
|---------|------|
| 【设计】 | 1 |
| 【测试】 | 2 |
| 【兼容性】 | 1 |
| 【接口同步】 | 1 |
| 【注释/文档】 | 1 |
| 【性能】 | 1 |

### ⚠️ 建议修复

**⚠️【设计】capture axis + 复刻 manager budget 逻辑可以用 `EncoderCudaGraphPathConfig(min_token_budget=729)` 替代；现方案默认会 capture 约 39 个永远不会被 replay 的 graph** `[已验证]`
- **问题**：`llava_onevision.py:582-633` 的 `_encoder_cudagraph_token_budgets` 调用 manager 私有静态方法 `EncoderCudaGraphManager._generate_budgets`（:604），并复刻 manager `__init__` 中"用户 budget / 用户 max items / 全自动"三个分支，以此推导 `capture_axes`（:643）。我用 main 的 `_generate_budgets` 模拟了默认配置（tile = 729 token，`max_model_len ≥ 8192`，上限 5120）：manager 生成 23 个 budget，padding 轴长 2，共 capture **46 个 graph**。但 replay 只可能命中 tile 数 1–7 对应的 **7 个** (budget, padding) 组合。其余 graph 都不会被选中：例如 budget 768–1408 的 axis=0 graph 都装 2 个 tile（1458 token > budget），而 2 个 tile 永远会选 budget 1536。LLaVA-OneVision 的 encoder token 恒为 729 的整数倍，正是 main 为 DeepSeek-OCR / Step3-VL 引入 `min_token_budget` 的场景：设置 `paths={"default": EncoderCudaGraphPathConfig(min_token_budget=tokens_per_tile)}` 后，manager 自己生成 `[729, 1458, …, 7×729]`，每个 batch 精确命中、不需要 padding tile，也不需要 capture axis。
- **影响**：46 次 SigLIP-400M + projector capture 拖慢启动，graph pool 也更大，而这些 graph 对性能没有贡献。更大的风险在维护面：模型依赖 manager 私有方法和分支结构，manager 的 budget 规则一变（#65084、#66067 都改过），模型推导出的轴就悄悄失配。作者在本 PR 生命周期内已经遇到两次（旧 grid 单测失败、`max_vision_items` 分支不一致）。另外 `select_encoder_cudagraph_items` → `_encoder_padding_axis`（:713/:728）每个 batch 都会重新生成一遍 budget 列表。
- **行动**：建议作者删除 `_encoder_cudagraph_token_budgets` / `_encoder_padding_axis_values` / `_encoder_padding_axis` 和 `capture_axes`，改为在 config 中声明 path 的 `min_token_budget`；capture 时直接用 `token_budget // tokens_per_tile` 个 tile。完成后可以顺带删掉保存的 `self.vllm_config` 和各处 `hasattr(self, "vllm_config")` 守卫。

**⚠️【测试】当前 head 没有任何 GPU 验证，PR 中的加速比与 `max_abs=0.0` parity 无法追溯** `[已验证]`
- **问题**：PR 描述明确说明，9 月之后的两次维护（包括 11-17 对 budget / capture-axis 逻辑和 capture dtype 的改动）只做了 CPU 单测，"CUDA graph replay and image/video E2E on this revision await accelerator validation"。5.92×/3.77×/1.99×/1.59× 的 encoder 延迟和 134.03→78.48 ms 的首 token 代理都来自 9 月旧代码上一个未公开的脚本（"7 interleaved rounds, median"），PR 里没有命令；`max_abs=0.0` 的对比也不在仓库的任何测试中，`test_vit_cudagraph.py` 只检查输出非空。此外 eager 下 1 tile 32.01 ms、2 tile 34.64 ms 几乎持平，说明测的是严重 host-bound 的场景，代表性存疑；首 token 代理关闭了 decoder CUDA graph 和 prefix cache，与默认 serving 配置不同。表中 8 tile（5832 token）一行在默认上限 5120 下根本不会走 graph。
- **影响**：当前代码在 GPU 上是否能 capture/replay、是否仍与 eager 一致，目前没有证据；性能卖点无法被 reviewer 复现。
- **行动**：作者应当在当前 head 上重新给出：① `vllm bench mm-processor --dataset-name random-mm`（例如 `(384,384,1)`、`(768,768,1)`、一个 ≥8 tile 的大图 bucket，graph on/off）的 encoder 延迟；② `examples/generate/multimodal/vision_language_offline.py -m llava-onevision --modality image/video` 加与不加 `--enable-vit-cuda-graph` 的输出对比；③ 一个多模态精度结果（如 MMMU / MMStar）。同时把历史表格标注为"旧代码、非默认配置"，或者删除。

**⚠️【测试】新增的 18 个单测在 CI 中不会被执行，且部分断言是对实现的同义反复** `[已验证]`
- **问题**：`tests/models/multimodal/test_llava_onevision_cudagraph.py` 没有任何 `core_model` 标记，而覆盖 `tests/models/multimodal` 根目录的 CI step（`.buildkite/test_areas/models_multimodal.yaml` 的 "Multimodal Models (Standard) 4"）只运行 `pytest models/multimodal -m core_model`，这个文件会被整体过滤掉。另外，`test_llava_onevision_auto_budgets_match_encoder_manager` 断言模型的 budget 等于 `EncoderCudaGraphManager._generate_budgets(...)`，相当于用实现去验证实现。这些测试都用 `object.__new__` + `SimpleNamespace` 构造假模型，覆盖不到真实 SigLIP 和 capture。
- **影响**：PR 描述中的 "18 passed" 在 upstream CI 中不会出现；回归只能依赖 `test_vit_cudagraph.py` 的冒烟测试。
- **行动**：建议作者给有价值的用例（postprocess 与 eager merge 等价、token 计数与 eager 一致、capture/replay dtype）加上 `pytest.mark.core_model`，或者并入已在 CI 中运行的文件。改用 `min_token_budget` 之后，budget 和 capture-axis 相关的用例可以删除。

**⚠️【兼容性】"CG for Video ✅︎" 在默认配置下几乎不生效：超过 7 帧的视频、超过 7 个 tile 的图片都会整体走 eager** `[已验证]`
- **问题**：每帧或每个 tile 为 729 token，main 的默认 `encoder_cudagraph_max_token_budget=5120` 最多容纳 7 个 tile。vLLM 默认视频加载 32 帧（`vllm/multimodal/media/video.py:109`，`num_frames=32`），即 23328 token，因此默认视频请求不会命中任何 graph。anyres 图片（base tile + 最多 3×3 网格）超过 7 个 tile 时同样走 eager。PR 的 E2E 用显式 budget `[8192]` 加 4 帧视频，所以没有暴露这一点（`test_vit_cudagraph.py:311`）。
- **影响**：用户看到支持表的 Video ✅︎ 并开启 `cudagraph_mm_encoder`，却拿不到预期收益，而且日志里只会看到 miss 统计。
- **行动**：建议作者在文档或 PR 描述中写明默认覆盖范围（≤7 帧 / ≤7 tile），并给出推荐配置（如 `encoder_cudagraph_max_token_budget` 或显式 budget），附上对应帧数下 graph on/off 的数据；如果实测大 batch 没有收益，可以如实标注只推荐小图和短视频。

**⚠️【接口同步】冗余成员：`supports_encoder_cudagraph = True` 与 `get_max_frames_per_video()`** `[已验证]`
- **问题**：`llava_onevision.py:476` 重复声明协议自带的 `supports_encoder_cudagraph: ClassVar = True`（#41759 review 中被要求删除过）。`:732` 的 `get_max_frames_per_video()` 在 main 的 manager 中没有调用者，PR 自己的 config 也直接传 `_MAX_FRAMES_PER_VIDEO`，属于死代码；main 上的约定是只通过 `EncoderCudaGraphConfig.max_frames_per_video` 给出。另外，本模型的 buffer 只有 `pixel_values`，没有与帧数相关的 buffer，`max_frames_per_video` 实际不影响 capture。
- **影响**：增加维护面，也会误导后来者以为这些成员是必需的。
- **行动**：建议作者删除这两处，`max_frames_per_video` 保留在 config 中即可，或说明为何需要非默认值。

**⚠️【性能】热路径上反复计算常量** `[已验证]`
- **问题**：`_encoder_tokens_per_tile()`（:536）每次都调用 `get_vision_encoder_info(self.config)` 构造 encoder info 并重新计算。它在 `get_encoder_cudagraph_item_specs`、`select_encoder_cudagraph_items`、`prepare_encoder_cudagraph_capture_inputs`、`postprocess_encoder_output` 中被调用，每个 encoder batch 要执行多次；`_encoder_padding_axis` 还会每个 batch 重新生成整张 budget 表。
- **影响**：tile 数少时恰好是 launch-bound、图收益最大的场景，这类 host 开销会直接抵消一部分收益。
- **行动**：建议作者在 `__init__` 中缓存 tokens-per-tile（或用 `cached_property`）；改用 `min_token_budget` 后 budget 表的重复生成会自然消失。

### 📝 建议/备注

**📝【注释/文档】支持表未按字母序，example 未注册** `[已验证]`
- **问题**：`docs/design/cuda_graphs_multimodal.md:22` 把 `LlavaOnevisionForConditionalGeneration` 插在 `InternVLChatModel` 与 `KimiVLForConditionalGeneration` 之间，按 Architecture 字母序应在 `Llama4ForConditionalGeneration` 之后、`Mistral3ForConditionalGeneration` 之前。`examples/generate/multimodal/vision_language_offline.py` 的 `MODELS_SUPPORT_VIT_CUDA_GRAPH` 没有加入 `llava-onevision`，因此 `--enable-vit-cuda-graph` 对这个模型不生效，无法用 example 做 eager vs graph 对比。
- **影响**：影响文档一致性；也缺少 tracker 集成工作流中的 example 一步。
- **行动**：建议作者调整表格位置，并在 example 列表中加入 `llava-onevision`。

## 8. 结论 (Verdict)

⚠️ **NEEDS WORK**：graph 与 eager 共用同一套 encoder / projector 和 merge 逻辑，没有发现数值或别名问题。但 capture-axis 方案默认会多 capture 约 39 个无用 graph，并依赖 manager 私有实现，应改为 main 现成的 `min_token_budget` 方案；当前 head 也缺少 GPU 验证、可复现的 benchmark 和精度证据，视频支持在默认配置下基本不生效。

## 9. 英文 Review 评论 (Copy-Paste English Comments)

**C1** `vllm/model_executor/models/llava_onevision.py:582-646` — ⚠️ comment

```text
I suppose we don't need the padding capture axis (and the copy of the manager's budget logic) at all. Every LLaVA-OneVision tile is exactly 729 encoder tokens, which is the case `EncoderCudaGraphPathConfig.min_token_budget` was added for (#65084/#65227): with `paths={"default": EncoderCudaGraphPathConfig(min_token_budget=tokens_per_tile)}` the manager generates budgets that are exact tile multiples, so every batch replays a graph with exactly its tile count. With the current design and default settings the manager captures 46 graphs but only 7 (budget, padding) pairs are reachable, and the model depends on the private `EncoderCudaGraphManager._generate_budgets`, which already drifted twice in this PR. Could you switch to `min_token_budget` and drop `_encoder_cudagraph_token_budgets` / `_encoder_padding_axis*` / `capture_axes`?
```

**C2** `tests/models/multimodal/generation/test_vit_cudagraph.py:297-315` — ⚠️ comment

```text
The PR description says the current head has only CPU validation, and the speedup table and the `max_abs=0.0` parity come from the September code with a script that isn't in the PR. Could you re-validate this revision on GPU? Please share `vllm bench mm-processor --dataset-name random-mm` results (graph on/off, e.g. 384x384, 768x768 and a large image bucket), compare `vision_language_offline.py -m llava-onevision` outputs with and without `--enable-vit-cuda-graph`, and add a multimodal accuracy result such as MMMU. Note that the 8-tile row (5832 tokens) runs eagerly with the default `encoder_cudagraph_max_token_budget=5120`.
```

**C3** `docs/design/cuda_graphs_multimodal.md:22` — ⚠️ comment

```text
With the default `encoder_cudagraph_max_token_budget=5120` at most 7 tiles fit in a graph, so videos with more than 7 frames (the default video loader samples 32) and images with more than 7 tiles always run eagerly. The E2E test only passes because it sets an explicit [8192] budget and uses 4 frames. Could you document the effective coverage and a recommended config for video in the doc or PR description? Also, please keep the table in alphabetical order (after `Llama4ForConditionalGeneration`), and add `llava-onevision` to `MODELS_SUPPORT_VIT_CUDA_GRAPH` in the offline example.
```

**C4** `tests/models/multimodal/test_llava_onevision_cudagraph.py:1-20` — ⚠️ comment（锚定在新文件开头）

```text
These unit tests won't run in CI: the step that covers the root of `tests/models/multimodal` runs `pytest models/multimodal -m core_model`, and none of these tests has the `core_model` marker. Could you mark the valuable ones (postprocess vs eager merge, token counts, capture/replay dtype) or move them into a file that CI already runs? The auto-budget test just compares the model against `EncoderCudaGraphManager._generate_budgets`, so it can go away once the capture axis is replaced by `min_token_budget`.
```

**C5** `vllm/model_executor/models/llava_onevision.py:476` — ⚠️ comment

```text
`supports_encoder_cudagraph = True` is already provided by the `SupportsEncoderCudaGraph` protocol, and `get_max_frames_per_video()` (line 732) is never called — the manager only reads `EncoderCudaGraphConfig.max_frames_per_video`. Could we remove both? Also, `_encoder_tokens_per_tile()` rebuilds the encoder info on every batch (item specs, select, postprocess); it is a constant, so we can cache it in `__init__`.
```

## 10. 英文总结评论 (Overall Review Summary)

```text
Thanks for the LLaVA-OneVision integration. Capturing SigLIP plus the projector while reusing the eager anyres merge and video pooling in postprocessing keeps graph and eager numerically aligned. Main asks: replace the padding capture axis and the copied manager budget logic with `min_token_budget`, since tiles are fixed 729-token units and most captured graphs are unreachable; re-validate this head on GPU with reproducible encoder benchmarks, an eager-vs-graph comparison and an accuracy check; and document that video and large images mostly run eagerly under the default token cap. Needs changes before merge.
```
