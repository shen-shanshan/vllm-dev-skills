# Reviewer Playbook：@shen-shanshan 与 @Isotr0py 的 ViT CG review 思路

> 从 `references/upstream/reviewer-comments.md`（#38175 tracker 全部 PR 的 review 语料）提炼。用途：让 §7 的关注点、§9/§10 的英文评论在思路与语气上贴近这两位 maintainer。规则本身见 `vit-cg-review-rules.md`。

## 1. 分工与关注点

| | @shen-shanshan（tracker owner） | @Isotr0py（多模态 maintainer） |
|---|---|---|
| 角色 | 逐个 review 模型集成 PR，保证每个 PR 跟上最新接口，推动集成工作流 4 步走完 | 把关框架设计与通用性，决定能否合入，触发 CI |
| 首轮关注 | 接口是否过时（贴出删除该方法的 PR 链接）、文档表/example/CI 是否更新、测试复用、模型特性覆盖（视频、多 crop、global+local tile）、budget 数学（ceil/floor、min budget 推导） | manager 是否被模型特定逻辑污染、能否泛化成接口/配置、数值等价（插值方式等实现差异）、CI 资源（小模型、dummy 权重、不要跳过 L4） |
| 证据要求 | benchmark 必须真正测到 encoder（给出 `vllm bench mm-processor` random-mm 命令）；让作者跑 example eager vs `--enable-vit-cuda-graph` 对比输出 | 多模态精度（MMMU 等）；eager vs graph 同输入对比 |
| 典型收尾 | "I left some comments and questions." → 修完后 "LGTM. Also CC @Isotr0py" / "Done with my pass." | "LGTM, thanks!" / "Overall LGTM, leave a nit." / 合入前 "/ci run" |

## 2. 思考顺序（照这个顺序看 diff）

1. **这个 PR 跟得上 main 吗？** 先列出它实现的协议方法，和 main 上 `SupportsEncoderCudaGraph` 对比：多了（已删除的方法）、少了（必需方法）、签名不同（`path`、`axis_keys`、`values`）。
2. **它有没有往 manager 里塞模型特定逻辑？** 有就要求抽成接口方法（带默认实现）或 config 字段，而不是 `getattr` 包装。
3. **capture / replay / eager 三条路径一致吗？** 读 `prepare_encoder_cudagraph_capture_inputs`、`prepare_encoder_cudagraph_replay_buffers`、`encoder_cudagraph_forward` 与原 `forward`：同一个 `prepare_encoder_metadata()`？插值/RoPE/merge 与 eager 等价？captured forward 里有没有 `.tolist()` / 数据依赖分支？
4. **budget 数学对吗？** min budget 推导注释；ceil vs floor；`max_batch_size` 与最小 budget 关系；固定尺寸 item 是否用 `min_token_budget`。
5. **模型的输入形态都覆盖了吗？** 视频、多 crop、global+local、空 local、pruning（EVS/VidCom2）开启时禁用视频 CG。
6. **集成 4 步走完了吗？** 测试（`test_vit_cudagraph.py` 小模型/dummy）、文档表（字母序、✅︎/❌︎、合并同 arch 行）、example 注册、benchmark + 精度。
7. **数字可信吗？** 测法、尺寸分档、何时比 eager 慢。
8. **小问题**：常量缓存成 property、import 位置、`self.` vs 类名调用、命名、docstring。

## 3. 语气与措辞模板（§9/§10 英文评论用）

两位 reviewer 的英文评论都**短、直接、礼貌**，常用 "I suppose …"、"Maybe it's better to …"、"Can we …?"、"Could you …?"、"FYI, <link>"、"Ditto."，提出问题时常用 "QQ:"。需要作者跟进的接口变化总是附上对应 PR 链接。

| 场景 | 模板（改写自真实评论） |
|---|---|
| 接口已删除 | "This method is not needed anymore, since #44484 makes `get_input_modality` default to `"image"` for image-only models. FYI, …" |
| 接口已合并/改名 | "These three interfaces have been merged into `get_encoder_cudagraph_item_specs()` in #41234. Could you update this model accordingly?" |
| 跟进 sibling 修复 | "This needs to sync the fix in #40580 (the `max_budget` cap), otherwise …" |
| manager 太模型特定 | "I feel this is too model-specific for `EncoderCudaGraphManager`. Can we make it a `SupportsEncoderCudaGraph` method with a default implementation instead of a `getattr` wrapper?" |
| 能否泛化 | "Is it possible to make this generic (e.g. arbitrary capture axes) rather than a single secondary axis?" |
| 数值等价 | "Can you double check that the graph path and the eager `forward` are numerically equivalent? I think the eager path uses bicubic interpolation while this one uses bilinear." |
| 共用 metadata | "I think we can reuse `prepare_encoder_metadata()` in both the eager forward and the capture/replay path to avoid two implementations drifting apart." |
| 热路径常量 | "This value is fixed, but it is recomputed on every inference step. We can make it a property or cache it in `__init__`." |
| ceil/floor | "I suppose we should use floor instead of ceil here? Otherwise the captured batch can exceed the token budget." |
| 覆盖问题 | "QQ: Does this PR support <video / pan-and-scan crops / images without local tiles>? If not, please mark it ❌︎ in the doc table." |
| 测试复用 | "Could we just reuse `tests/models/multimodal/generation/test_vit_cudagraph.py`? We only need to add a new config for this model." |
| CI 资源 | "For CI we should use a model as small as possible; this one is too big for L4, please use dummy weights (`hf_overrides=dummy_hf_overrides`) and a small budget, see #43082." |
| 文档表 | "Please keep the table in alphabetical order, and use ✅︎ / ❌︎ like the other rows." |
| 集成清单 | "Please also update this model in the doc (supported-models table), the example (`MODELS_SUPPORT_VIT_CUDA_GRAPH`) and the CI test (`test_vit_cudagraph.py`)." |
| benchmark 无效 | "I don't think these TTFT numbers measure the encoder at all: `vllm bench latency` only sends random-token prompts, so no image reaches the ViT. Could you re-measure with `vllm bench mm-processor --dataset-name random-mm` (graph on/off) and report encoder latency?" |
| 精度 | "Can you provide some multimodal accuracy results (e.g. MMMU) with ViT CG on and off?" / "Can you run `vision_language_offline.py -m <model> --modality image` with and without `--enable-vit-cuda-graph` and compare the outputs?" |
| 无关改动 | "I suppose this change is irrelevant to ViT CG? We don't need to touch it in this PR." |
| rebase | "Could you please rebase on the latest main and resolve the conflicts? The ViT CG interfaces changed in #65084." |
| 分 PR | "Let's get this initial support in first and optimize it in a follow-up PR." |

总结评论（§10）参考两位的收尾风格：先肯定做对的地方（具体到实现），再列 1–3 个要改的点，最后给立场（"LGTM once …" / "needs changes before merge"）。不使用 emoji，不提内部规则码。

## 4. 对作者的取舍习惯（决定严重程度时参考）

- **初版支持优先合入，优化放 follow-up**：global+local 同构 tile 先用 tile batch（#46005）、视频/resampler 入图放后续 PR（#45254）都被接受 —— 只要文档表如实标注、eager 路径不受影响。这类问题标 📝 而非 ⚠️。
- **实验性功能可接受不完美的 UX，但不能接受错误结果**：数值不等价、capture 越界、输出别名一律阻塞。
- **CI 资源很宝贵**：大模型/重复 arch 的 CI 会被要求改；用 `large_gpu_mark` 跳过等于没测，也会被要求改。
- **不要求作者做超出范围的重构**，但会要求跟上已合入的接口简化。
