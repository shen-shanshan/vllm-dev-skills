# vLLM ViT CUDA Graph PR Review 规则目录

> 来源：从 [#38175](https://github.com/vllm-project/vllm/issues/38175) tracker 中全部 PR 上 @shen-shanshan 与 @Isotr0py 的 review 意见提炼（原始语料：`references/upstream/reviewer-comments.md`，由 `scripts/update_skill.py` 采集），并结合 vllm main（`632ba4f`，2026-11-26）的 encoder CUDA graph 框架代码校准。
> 通用审查纪律（🔴 gate、数字溯源、CI 归因、AI 代码诊断）沿用 `vllm-rocm-pr-review` 的方法，去掉了 aiter/mori/ROCm 专项规则。
> 提炼日期：2026-11-26。规则编码（I/M/B/N/V/C/T/D/P/A/H/HK）是内部标签，**绝不出现在报告中**。

**何时读全文**：模型集成 PR（新模型实现 `SupportsEncoderCudaGraph`）、改 manager/defs/interfaces 的框架 PR → 全文。只改文档/测试/example 的 PR → 读「核心纪律」「框架地图」+ D/T/HK。

---

## 1. 核心纪律

1. **以 main 为准，而非 PR 的 base**：这个子系统接口迭代极快（#41234 合并 3 个方法为 `get_encoder_cudagraph_item_specs`、#42288 改为单一 `values` dict、#44484 删掉 image-only 模型的 `get_input_modality`、#49934 统一多路径、#65084 共享前缀 buffer + 新 budget 网格）。review 前必须 `git fetch upstream main`，用 `git show upstream/main:<path>` 读框架当前形态；PR 落后于 main 时，"与 main 接口不一致"本身就是 finding。注意本地 `origin` 可能是个人 fork，比 upstream 旧。
2. **跨文件验证**：声称"capture 与 replay 不一致"、"eager 与 graph 数值不等价"、"buffer 没被 replay 覆盖"之前，必须把模型文件里 `prepare_encoder_cudagraph_capture_inputs` / `prepare_encoder_cudagraph_replay_buffers` / `encoder_cudagraph_forward` / `encoder_eager_forward` / 原 eager `forward` 五处对照读完，并读 manager 的 `_run_budget_graph` / `_execute_local` 确认调用约定。只看 hunk 下结论是误报主因。
3. **🔴 gate**：任何"必须修复"必须写出**具体触发输入**（图片分辨率 / tile 数 / 视频帧数 / budget 列表 / `max_vision_items_per_batch` / TP+`mm_encoder_tp_mode=data` 组合）。写不出 → 降级 ⚠️ 或放弃。
4. **数字溯源**：PR 描述里的 TTFT/encoder 加速数字必须能追溯到真正送了图片/视频的 workload。`vllm bench latency` 只生成随机 token 的 `TokensPrompt`，不送任何图片——用它测出的 "ViT CG 加速" 是噪声（@shen-shanshan 在 #44394 指出）。追溯不到 = `[unverified]`。
5. **CI 归因**：红叉先看 step 名与 main 同窗口是否同挂；`pre-run-check` / readthedocs 的 "needs ready label" 门禁与代码无关；多模态 CI 失败若命中 `test_vit_cudagraph.py` 的本模型 case 才算本 PR 的。

## 2. 框架地图（main 快照，review 时以 `upstream/main` 为准）

| 文件 | 作用 | review 关注 |
|------|------|------------|
| `vllm/model_executor/models/interfaces.py` → `SupportsEncoderCudaGraph` | 模型协议：`get_encoder_cudagraph_config` / `get_encoder_cudagraph_budget_range`（默认 `(64, min(max_num_batched_tokens, max_model_len))`）/ `get_encoder_cudagraph_item_specs` / `select_encoder_cudagraph_items` / `postprocess_encoder_output`（默认 `scatter_output_slices(outputs["default"], …)`）/ `prepare_encoder_cudagraph_capture_inputs(…, path, axis_keys)` / `prepare_encoder_cudagraph_replay_buffers(…, path)` / `encoder_cudagraph_forward(values, path)` / `encoder_eager_forward(mm_kwargs, path)` | 签名是否与 main 一致；有默认实现的方法不要重复实现 |
| `vllm/v1/worker/encoder_cudagraph_defs.py` | `EncoderItemSpec(input_size, output_tokens, path_output_tokens)`、`EncoderCudaGraphPathConfig(min_token_budget, allow_zero_tokens)`、`EncoderCudaGraphConfig(modalities, buffer_keys, out_hidden_size, padding_logics, max_frames_per_video=1, paths, capture_axes, eager_fallback_fast_path)`、`EncoderCudaGraphCaptureInputs(values)`、`EncoderCudaGraphReplayBuffers(values)`、`ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG` | 新字段是否必要；是否滥用 None 值 |
| `vllm/v1/worker/encoder_cudagraph.py` → `EncoderCudaGraphManager` | budget 生成（四分之一倍频程到 1024、128 步长到 2560、再 1/8 倍频程，上限 `encoder_cudagraph_max_token_budget`=5120）；largest-first 捕获 + **共享前缀 buffer**；每 path 每 batch 只 replay 一次最小可容纳 budget，超出整体 eager；DP 分片 `get_load_balance_assignment` + all_gather；默认 padding：cumulative key（`cu_seqlens`/`cu_window_seqlens`）重复末值，其余补零 | 框架 PR 必须保持 model-agnostic |
| `vllm/model_executor/models/encoder_cudagraph_utils.py` → `Qwen2VLEncoderCudaGraphMixin` | Qwen2-VL 风格 vision tower（`pixel_values[_videos]` + `*_grid_thw`，`visual.prepare_encoder_metadata()`）的复用实现；宿主只需实现 `get_encoder_cudagraph_config()` | 复用 Qwen 系 tower 的模型（Qwen2.5-Omni、Kanana-V 等）应继承它而不是复制 |
| `vllm/model_executor/models/utils.py` → `scatter_output_slices` | 默认输出切分 | `clone=True` 时结果不得别名 graph 输出 buffer |
| `vllm/config/compilation.py` | `cudagraph_mm_encoder`、`encoder_cudagraph_token_budgets`、`encoder_cudagraph_max_token_budget`、`encoder_cudagraph_max_vision_items_per_batch`、`encoder_cudagraph_max_frames_per_batch` | 新增配置项需必要性论证 |
| `docs/design/cuda_graphs_multimodal.md` | Supported Models 表（Architecture / Models / CG for Image / CG for Video / Multi-Path Graph） | 模型集成 PR 必须更新 |
| `tests/models/multimodal/generation/test_vit_cudagraph.py` | `MODEL_CONFIGS: dict[str, VitCudagraphTestConfig]`（`modalities`、`image_prompt`/`video_prompt`、`vllm_runner_kwargs` 可放 `hf_overrides=dummy_hf_overrides`、`compilation_config_overrides`、`marks`）；`test_eonly_vit_cudagraph_outputs`（多尺寸 graph vs eager 数值对比 + hit/miss 断言） | 模型集成 PR 必须加 case，能进 eonly 对比测试的优先进 |
| `vllm/v1/worker/gpu/model_states/interface.py` | V2 runner 创建 `EncoderCudaGraphManager` 的门控：`not enforce_eager and cudagraph_mm_encoder and supports_encoder_cudagraph(model)` | `enforce_eager=True` 会关掉 encoder graph |
| `tests/v1/cudagraph/test_encoder_cudagraph.py` | manager 单测 | 框架 PR 必须覆盖 |
| `examples/generate/multimodal/vision_language_offline.py` → `MODELS_SUPPORT_VIT_CUDA_GRAPH` + `--enable-vit-cuda-graph` | example 注册（字母序） | 模型集成 PR 应更新 |

**Tracker 集成工作流（#38175 NOTE）**：① 参照 Qwen3-VL 实现接口 → ② ut/e2e/benchmark → ③ 更新 design doc 支持表 → ④ 加入 CI 测试。缺任何一步都是 finding。

## 3. 规则目录

严重程度：🔴 必须修复 / ⚠️ 建议修复 / 📝 备注。所有 🔴 都要过 🔴 gate。

### I — 接口与 main 同步（最高频意见）

- **I1 使用已废弃/冗余接口** ⚠️：image-only 模型实现 `get_input_modality`（#44484 后默认返回 `"image"`）；实现 `get_max_frames_per_video`（改由 `EncoderCudaGraphConfig.max_frames_per_video`，默认 1）；`supports_encoder_cudagraph: ClassVar = True`（协议已带）；旧三件套 `get_encoder_cudagraph_num_items / per_item_output_tokens / per_item_input_sizes`（#41234 合并为 `get_encoder_cudagraph_item_specs`）；`input_key_by_modality`（#42288 删除，像素值放进 `buffer_keys`）；`mm_kwargs` 形参的旧 forward 签名。出处：#45254、#41992、#43591、#41759、#46005、#40576。
- **I2 重复实现已有默认/Mixin** ⚠️：Qwen2-VL 风格 tower 却逐个复制 mixin 方法；`postprocess_encoder_output` 与默认 `scatter_output_slices` 等价却重写；自写 budget range 与默认 `(64, min(max_num_batched_tokens, max_model_len))` 相同。@shen-shanshan："Since these codes are totally the same as that of Qwen3-VL, maybe we should find out a way to eliminate redundancy"（#40830）。
- **I3 与 sibling PR 的修复不同步** ⚠️：同类问题已在别的 PR 修过（ceil→floor 容量、`max_budget` cap、EVS/VidCom2 pruning 禁用视频 CG），新 PR 仍用旧写法 → 指出并给链接（#40830 → #40580，#40576 → #38040）。

### M — Manager 通用性（框架 PR / 改 manager 的模型 PR）

- **M1 manager 出现模型特定逻辑** 🔴/⚠️：manager 里按 key 名判断（`"pixel_values" in mm_kwargs and "image_grid_thw" in …`）、`getattr(model, "some_model_hook", None)` 包装、为一个模型加专用分支。@Isotr0py："I feel this is too model-specific, and it's difficult to use for other models with different mm_kwargs naming"（#35963）。正确做法：把需求抽成协议方法（带默认实现）或 `EncoderCudaGraphConfig` 字段。@shen-shanshan："We can make this method as an ViT CG interface (default: return `None`)" / "there is no need to add additional wrappers in `EncoderCudaGraphManager`"（#42785）。
- **M2 新配置/接口不够通用** ⚠️：为单模型新增"次级 capture 轴"、"dual-path"之类，应问能否泛化（@Isotr0py："Is it possible to make arbitrary capture axis support?" #42785；"represent `budget_graphs` as `dict[str, dict[int, ...]]` if there're multiple paths?" #43586；padding 逻辑应放 `EncoderCudaGraphConfig.padding_logics` #42796）。
- **M3 用户体验** ⚠️：要求用户手填 budget/新参数而不能自动推断（@Isotr0py："asking user to pass `encoder_cudagraph_token_budgets` manually will cause poor user experience" #35963）。
- **M4 改共享框架未同步文档/单测** ⚠️：改 budget 网格、padding、共享 buffer 语义而不改 `docs/design/cuda_graphs_multimodal.md` 与 `tests/v1/cudagraph/test_encoder_cudagraph.py`（#65084 @shen-shanshan："The doc should also be updated here"）。

### B — Budget / 容量数学

- **B1 capture 容量计算方向错** 🔴：`per_item_output = token_budget // max_batch_size` 与 `ceil` 混用；"每 batch 能放几张图" 应 floor（`token_budget // tokens_per_image`），"总容量不小于 budget" 才 ceil。错用 → 某些 budget 下 capture 的 batch 容量 > budget（越界）或 < 声称容量（replay 溢出 → 不该 eager 的走 eager，或索引越界）。出处：#43591（`ceil`→`floor`）、#40830/#38040（"Use ceil here (not floor) so total captured capacity is never smaller"）。
- **B2 budget range 不合理** ⚠️：`min_budget` 应等于该模型"最小合法输入"的输出 token 数，并有推导注释（224×224/patch14/merge2 → 64；固定 tile 模型 = 每 tile token 数；Gemma 系 = `mm_tokens_per_image`）；`max_budget` 不应超过 `min(max_num_batched_tokens, max_model_len)`（超出曾触发 `IndexKernel.cu` 越界，#40580）。固定尺寸 item 的 path 应设 `EncoderCudaGraphPathConfig(min_token_budget=...)` 让 budget 落在 item 整数倍上，避免冗余 graph（#65084 → #65227）。
- **B3 padding 浪费吞掉收益** ⚠️：固定大 tile（OCR 文档页 ~1000 token）被 pad 到下一档 budget，实测比 eager 慢（#63074 作者自己报告 0.56–0.61×）；多图 batch 的 padding 让 attention 做 N 倍无用功（#56277）。要求给出 "graph 变慢" 的区间与默认 budget 的取舍依据，不要默默改全局默认。**不要在模型的 `get_encoder_cudagraph_budget_range()` 里硬编码一个小 `max_budget` 来规避变慢**：manager 取 `min(model_max, encoder_cudagraph_max_token_budget)`，模型写死的上限会静默压住用户调高的全局旋钮。先在当前 main 的细粒度网格上重测（main 的 1024–2560 是 128 步长，旧 power-of-2 网格的 padding 数据已过时），仍变慢则在文档中说明并让用户通过 `encoder_cudagraph_max_token_budget` 调整。
- **B4 共享前缀 buffer 约束** 🔴：main 上 budgets largest-first 捕获并共享前缀 buffer——replay 必须提供每个共享 buffer（manager 缺失即 raise）；自定义 `padding_logics` 必须写满整个目标 buffer；0-dim scalar 不共享。模型在 `prepare_encoder_cudagraph_replay_buffers` 里对某 key 返回 `None`（"保持不变"）而该 key 被共享 → 读到别的 budget 的残值。

### N — 数值等价（eager vs graph）

- **N1 graph 路径重写了 eager 计算但不等价** 🔴：为去掉 `.tolist()` 重写位置编码插值、RoPE、patch merge 时改了算法（GLM-4.1V 是 **bicubic**，照抄 Qwen3-VL 的 bilinear 即错，@Isotr0py #40576）。要求：graph 路径与 eager 共用同一 `prepare_encoder_metadata()`（@Isotr0py 在 #35963 提出的模式），或提供 eager vs graph 同输入逐元素对比（`max_abs=0` / bit-identical）。
- **N2 `prepare_encoder_metadata` 只给 graph 用** ⚠️：docstring 说 "shared by eager and graph" 但 eager 没走它（#41992 @shen-shanshan）——两套实现会漂移。
- **N3 padding 区域泄漏到有效输出** 🔴：padded 序列必须在 attention 中为空（cumulative key 重复末值）；非 varlen 的 attention（batched CLIP/SigLIP）里 padded 图要么独立 batch 维要么被 mask；后处理只取有效 token（`output[:num_valid]`）。
- **N4 精度证据** ⚠️：模型集成 PR 需提供多模态精度（MMMU / MMStar 等）或至少 eager vs CG 文本输出对比（@Isotr0py："Can you provide some multimodal accuracy benchmark results like MMMU etc?" #40576、#45254；@shen-shanshan 给的命令：`python examples/generate/multimodal/vision_language_offline.py -m <model> --modality image [--enable-vit-cuda-graph]` #41992）。

### V — 模态与输入形态覆盖

- **V1 声称支持的模态与实际不符** ⚠️：模型本身支持视频但 PR 只做图片，文档表格必须标 ❌︎ 并说明（#45254）；反之 config 里加了 `"video"` 却没测。
- **V2 动态 token 数特性** 🔴：EVS / VidCom2 视频 token pruning 开启时视频 token 数数据依赖 → 必须禁用视频 CG（参照现有模型 `is_multimodal_pruning_enabled` 判断写法）；tower LoRA（`enable_tower_connector_lora`）与 encoder graph 不兼容——main 已在 `vllm/config/vllm.py` 统一拦截（#52376 讨论），模型 PR 不需要自己再判断，重复判断属冗余。
- **V3 多 crop / tile / 全局+局部** ⚠️：模型有 pan-and-scan / 动态 tile / global+local 结构时，问清 PR 是否覆盖（@shen-shanshan "if the original image ratio > min_ratio, the mm inputs may contain N crops, does this PR support this?" #43591）；global+local 同构 tile 可作为 tile batch，异构应考虑 multi-path（`paths` + `postprocess_encoder_output`，#43586/#49934/#46005）。
- **V4 空路径/零 token** ⚠️：batch 只有 global 无 local 时 `outputs["local"]` 不存在或为 `None`；`len(indices)==0` 分支要提前返回（#49934、#43591）。
- **V5 混合模态 / 每 prompt 多 item** 📝：image+video 混合、`--limit-mm-per-prompt` 对 `max_frames_per_batch` 的影响（video 限制为 0 → 帧 budget 置 0）。

### C — Capture 安全（CUDA graph 兼容性）

- **C1 captured 区域内的 host 同步 / 动态形状** 🔴：`encoder_cudagraph_forward` 里出现 `.tolist()`、`.item()`、`.cpu()`、`torch.nonzero`、依赖数据的 Python 分支/循环、`torch.empty` 形状依赖输入、`np` 计算 → capture 失败或 replay 用 capture 时的值（静默错）。这些必须挪到 `prepare_encoder_cudagraph_replay_buffers`（host 侧）预计算成 buffer（Kimi-VL #41992 的 moonvit 重构是范例）。
- **C2 buffer 身份不变式** 🔴：capture 返回的 `values` 张量对象就是 graph 读写的地址；forward 里对 buffer 重新赋值（`x = x.to(...)` 产生新张量后被 graph 外引用）、replay 时替换张量对象而不是 `copy_` → replay 读旧数据。
- **C3 输出别名** 🔴：`clone=True` 时写入 `dest` 的张量不能是 graph 输出 buffer 的 view（下一次 replay 覆盖；#65351 修过 Step3-VL 的同类 bug）。自定义 `postprocess_encoder_output` 必须 clone 或拷贝。
- **C4 attention backend 依赖** ⚠️：graph 依赖 varlen attention 读 device 侧 `cu_seqlens`/`max_seqlen`；某些 backend 需要 host 侧 `seq_lens`（如 FLASHINFER/TORCH_SDPA 的处理不同）。PR 应写明测过的 `--mm-encoder-attn-backend`，未测的 backend 是否会静默走 eager 或报错。
- **C5 DP 模式** ⚠️：`mm_encoder_tp_mode="data"` 下 manager 自己分片；模型若在 eager 路径另有 `run_dp_sharded_*` 逻辑，确认 graph 路径不会二次分片。

### T — 测试与 CI

- **T1 未加入 `test_vit_cudagraph.py`** ⚠️：模型集成 PR 应在 `MODEL_CONFIGS` 增加 case，而不是另写一个模型专属 e2e（@shen-shanshan："Could we just reuse this test… We only need to add new configs" #40830）。只检查“输出非空”的 e2e case 在 graph 根本没跑时也会通过；main 上的 `test_eonly_vit_cudagraph_outputs`（同文件，按模型参数化，多尺寸 graph vs eager `assert_close` + hit/miss 断言）是更强的覆盖，结构合适的模型（尤其 Qwen2-VL 风格处理器）应加进它的模型列表。不要在共享测试函数里加 `if model_id == "xxx"` 分支。
- **T2 CI 模型过大** ⚠️：用尽可能小的 checkpoint（InternVL3-1B 而非 2B #41759；gemma-4-E2B #46837）；MoE / 大模型用 `hf_overrides=dummy_hf_overrides` + 小 budget（#43082、#41992、#44720）。**不要**用 `large_gpu_mark` 跳过 L4 —— 等于没有覆盖（@Isotr0py #43082）。
- **T3 冗余 CI** 📝：与已测模型共用同一 ViT 的变体（Qwen3.6 vs Qwen3.5、dense vs MoE）不应再加 CI（@Isotr0py："just wasting the valuable CI resource" #44720）。
- **T4 单测** 📝：新增 manager 行为 / 新 path / capture 轴应在 `tests/v1/cudagraph/test_encoder_cudagraph.py` 有 GPU 单测；模型级单测尽量复用已有 helper。

### D — 文档与 example 注册

- **D1 支持表未更新** ⚠️：`docs/design/cuda_graphs_multimodal.md` 的 Supported Models 表需新增一行；同一 arch 的多个模型合并到一行（#42785 MiniCPMV2.6/4.0）；符号统一用 `✅︎` / `❌︎`（#41992）；**按 Architecture 字母序**插入（@Isotr0py #42224、#43586）；列数与当前表头一致（main 已有 `Multi-Path Graph` 列）。
- **D2 example 未注册** 📝：`MODELS_SUPPORT_VIT_CUDA_GRAPH` 加入 model type（字母序，不重复）；同 `model_type` 复用已有 runner 不新增（#44720）。
- **D3 无关改动** ⚠️：与 ViT CG 无关的 example / 配置改动应拆出去（#40576）。**例外**：main 的 V2 runner 只在 `not enforce_eager and cudagraph_mm_encoder and supports_encoder_cudagraph(model)` 时创建 `EncoderCudaGraphManager`（`vllm/v1/worker/gpu/model_states/interface.py`），所以 example runner 里写死的 `enforce_eager=True` 会让 `--enable-vit-cuda-graph` 静默失效——为此删除它是必要改动，不要当成无关改动（#40576 时的意见早于这一门控，已过时）；反过来，模型已注册进 `MODELS_SUPPORT_VIT_CUDA_GRAPH` 但 runner 仍写死 `enforce_eager=True` 是 finding。`assert model in MODELS_SUPPORT_VIT_CUDA_GRAPH` 放在开关判断外面会让不支持的模型直接报错（#42224）。
- **D4 注释/docstring 失真** 📝：推导注释写错（"16x16 patches" 是 patch 数还是 patch size，#38061）；方法行为改了 docstring 没改（#49934）；看不懂的新方法需加说明（#40576 `pos_embeds_interpolate`）。

### P — 性能证据

- **P1 benchmark 没测到 encoder** 🔴（若 PR 以性能为卖点）/ ⚠️：见核心纪律 4。推荐命令（@shen-shanshan #44394）：
  ```bash
  vllm bench mm-processor --model <model> --max-model-len 8192 \
    --dataset-name random-mm --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --random-mm-bucket-config '{(896, 896, 1): 1.0}' \
    --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --num-prompts 100 \
    --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_max_vision_items_per_batch": 8}'
  ```
  或 `vllm bench serve --dataset-name random-mm` 报 TTFT + encoder 时间，graph on/off 对照。
- **P2 只测单一尺寸** ⚠️：至少覆盖 小图（launch-bound，收益最大）/ 典型图 / 大图或多图（padding 可能变慢）三档；报告 "何时比 eager 慢"。设计文档原则：一个 replay 只节省 host launch 时间，GPU-bound 时 graph 不会更快。
- **P3 可复现性** ⚠️：缺 GPU 型号、attention backend、budget 配置、命令。
- **P4 显存** 📝：graph 数量 × buffer（main 已共享前缀 buffer，主要是 graph 本身的 pool）；新增 capture 轴使 graph 数量乘法膨胀（manager 超 64 会 warn）。

### H — 热路径开销

- **H1 每次推理重复计算常量** ⚠️：`_proj_hw()`、`(image_size // patch_size) ** 2`、`_get_num_input_output_tokens()` 等固定值应 `@property` / `functools.cached_property` / `__init__` 缓存（#46005、#43591、#43586）。
- **H2 host 侧低效** 📝：`select_encoder_cudagraph_items` 里逐 item Python 循环 + 小张量 `torch.tensor(...)` H2D、可以先判空提前返回（#43591）。

### HK — 卫生检查

| 检查 | 标注 |
|---|---|
| 函数内 import 可放顶层（无循环依赖时）（#35963） | 📝 |
| `ClassName._method(...)` 应写 `self._method(...)`（#42224） | 📝 |
| 通用 helper 放 `vllm/model_executor/models/utils.py`（#42224） | 📝 |
| 把日志消息做成模块级常量并跨文件 import（#52376） | 📝 |
| 变量命名含糊（`normalized` #41992） | 📝 |
| 笔误（`input_key = input_key = …` #41234） | ⚠️ |
| 与 PR 目的无关的文件 / 临时脚本 | ⚠️ |
| PR 描述的 Test Plan/Result 是模板占位或 AI 生成未验证 | ⚠️ |
| 未 rebase、有冲突 / 落后 main 的接口（"please rebase on the latest main and resolve the conflicts"） | ⚠️ |

## 4. AI 代码诊断（沿用 rocm skill，ViT CG 化）

预过滤（3+ 命中 → "AI 代码风险升高"，提高 capture/replay 一致性与测试的审查优先级）：描述只讲 WHAT 不讲 WHY；加速比可疑地整或大（5.92× 之类需要看测法）；只有单尺寸 benchmark；`Test Result` 占位；无关文件；描述 footer 带 AI 生成署名；评论区 "claiming via" 式机器人认领。

结构验证（diff 触及代码时强制）：
1. **幻觉符号扫描**：diff 中引用的协议方法名/`EncoderCudaGraphConfig` 字段/defs 类是否在 main 存在、签名一致（`grep` `upstream/main`）。
2. **孪生分歧**：image 与 video 分支、capture 与 replay、graph forward 与 eager forward 逐字段对比（dtype、device、buffer key 集合、padding 方式）。
3. **声明 ↔ 代码**：描述说 "bit-identical" → 有没有对应测试；说 "image + video" → config 的 `modalities` 与测试 `modalities` 是否一致。
4. **安全戏法**：`raise AssertionError("unreachable")`、`except: pass`、永不触发的 guard。
5. **为通过而校准的测试**：graph vs eager 对比的两边是否走了同一实现（同 bug 两边都有）；`max_tokens` 太短、只比 1 个 token。
6. **魔法常量**：min budget、tile token 数无推导注释。

## 5. 判定校准

- 每条 finding 三要素：**问题**（file:line）/ **影响**（错输出 / capture 崩溃 / 静默 eager 回退 / 性能倒退 / 维护负担）/ **行动**（以"作者应当…"/"建议作者…"/"建议 review 时追问…"结尾）。
- 标 `[已验证]`（读过 main 与 PR 代码、有证据链）或 `[推测]`。
- 🔴 仅用于：数值错误、capture/replay 崩溃或越界、输出别名、在声称支持的配置下静默错结果、性能卖点的数字完全没测到 encoder。接口过时、缺文档/CI、命名等一般是 ⚠️/📝。
- 好 finding 示例：
  - `🔴【正确性】vllm/model_executor/models/foo.py:812 graph 路径的位置编码用 bilinear 插值，而 eager forward（同文件 :640）用 bicubic；在任何非原生分辨率（如 896×672）输入下 graph 与 eager 输出不一致。作者应当让两条路径共用 prepare_encoder_metadata()，并补一个非原生分辨率的 eager-vs-graph 对比。` `[已验证]`
  - `⚠️【设计】foo.py:1020 重新实现了 get_input_modality 且恒返回 "image"；main 上该方法已有默认实现（#44484）。建议作者删除。` `[已验证]`
- 坏 finding：`⚠️ 缺测试`（无影响无行动）；`🔴 I1 violation`（规则码）。

## 6. 更新日志

| 日期 | 语料（PR 数 / 评论块） | 变更 |
|------|----------------------|------|
| 2026-11-26 | 58 / 387 | 首批 8 个 PR review 后校准：D3 增加 `enforce_eager` 门控例外、T1 补充 `test_eonly_vit_cudagraph_outputs`、B3 明确不在模型 budget range 中硬编码上限 |
| 2026-11-26 | 58 / 387 | 初始提炼：自 #38175 tracker 全部 PR 的 @shen-shanshan、@Isotr0py review 意见建立规则目录；框架地图对齐 vllm main 632ba4f（共享前缀 buffer、`encoder_cudagraph_max_token_budget`、`Qwen2VLEncoderCudaGraphMixin`、multi-path + capture_axes） |
