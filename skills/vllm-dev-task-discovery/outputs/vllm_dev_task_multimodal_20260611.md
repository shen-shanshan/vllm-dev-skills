# vLLM 社区开发需求分析报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 🎯 目标模块 | 多模态（Multimodal）— ViT 编码器、多模态 Pipeline、视觉/音频/视频模型 |
| 📅 分析日期 | 2026-06-11 |
| 📊 Task 总数 | 30 |
| 🔍 数据来源 | Open Issues (100+) / Merged PRs (30+) / Code TODOs (15+) / Roadmap / Multi-modality Labeled Issues (13) / Discussions |
| 💡 分析范围 | 近 3 个月活跃内容为主，辅以长期 keep-open issue 和 roadmap 追踪项 |

---

## 📋 需求总览（Top 30 Tasks）

| # | 类型 | 标题 | 难度 | 紧急度 | 重要度 | 来源 |
|---|------|------|------|--------|--------|------|
| 1 | 🧩 | DeepSeek-OCR-2 ViT CUDA Graph 适配 | ⭐⭐⭐ | 🔴 | ⭐⭐⭐ | #38175 |
| 2 | 🧩 | Ernie-4.5-VL ViT CUDA Graph 适配 | ⭐⭐⭐ | 🔴 | ⭐⭐⭐ | #38175 |
| 3 | 🚀 | 多模态 Speculative Decoding Draft Model 支持 | ⭐⭐⭐ | 🟡 | ⭐⭐⭐ | #33458 |
| 4 | 🐛 | Prefix Caching 忽略视觉输入导致并发下输出错误 | ⭐⭐⭐ | 🔴 | ⭐⭐⭐ | #20261 |
| 5 | 🐛 | V1 引擎 Native Weight Update 后 Encoder Cache 脏数据 | ⭐⭐ | 🔴 | ⭐⭐ | #44910 |
| 6 | 🐛 | LoRA 同名 reload 复用 Stale Encoder Cache | ⭐⭐ | 🔴 | ⭐⭐ | #44939 |
| 7 | 🧩 | Gemma 4 接入 /v1/audio/transcriptions 端点 | ⭐⭐ | 🟡 | ⭐⭐ | #40994 |
| 8 | 🚀 | Multimodal Encoder 异步 Embedding Cache 加载 | ⭐⭐⭐ | 🟡 | ⭐⭐⭐ | #45242 |
| 9 | 🐛 | prompt_embeds 模式在 VL 模型中不工作 | ⭐⭐ | 🟡 | ⭐⭐ | #44842 |
| 10 | 🐛 | TurboQuant + Gemma 4 Multimodal 5 层 Gate Blocker | ⭐⭐⭐ | 🟡 | ⭐⭐⭐ | #41403 |
| 11 | 🚀 | EPD 解耦下 mm_hash 和 mm_transfer_params 优化 | ⭐⭐ | 🟡 | ⭐⭐ | #36328 |
| 12 | 🐛 | Qwen3-VL EVS 视频剪枝 CPU/CUDA Device Mismatch | ⭐⭐ | 🟡 | ⭐⭐ | #44200 |
| 13 | 🐛 | Multimodal Processor Cache 丢失 Audio（Qwen3-Omni） | ⭐⭐ | 🟡 | ⭐⭐ | #44538 |
| 14 | 🐛 | Qwen3.5-122B-A10B 并发图像请求 EngineCore Crash | ⭐⭐⭐ | 🟡 | ⭐⭐ | #37602 |
| 15 | 🚀 | 多模态模型 LoRA Tower/Connector 扩展支持 | ⭐⭐ | 🟢 | ⭐⭐ | #31479 |
| 16 | 🚀 | 多模态模型 CI 性能 Benchmark | ⭐⭐ | 🟡 | ⭐⭐⭐ | #16353 |
| 17 | 🔧 | Transformers v5 升级 — InternVL2 适配 | ⭐ | 🟡 | ⭐⭐ | #38425 |
| 18 | 🔧 | Transformers v5 升级 — IsaacForConditionalGeneration 适配 | ⭐ | 🟡 | ⭐⭐ | #38389 |
| 19 | 🔧 | Transformers v5 升级 — Tarsier2 适配 | ⭐ | 🟡 | ⭐⭐ | #38736 |
| 20 | 🚀 | Image Token Pruning for Multimodal Models 实现 | ⭐⭐⭐ | 🟢 | ⭐⭐ | #45098 |
| 21 | 🚀 | DeepStream GPU 视频解码后端接入 | ⭐⭐⭐ | 🟢 | ⭐⭐ | #41843 |
| 22 | 🚀 | 直接二进制/Multipart 文件上传 API | ⭐⭐ | 🟢 | ⭐⭐ | #38531 |
| 23 | 🐛 | AsyncLLM 在多模态 Pooling 请求下静默 Hang | ⭐⭐ | 🟡 | ⭐⭐ | #41969 |
| 24 | 🐛 | Voxtral-Realtime 多 Session 下 Transcription 停止 | ⭐⭐ | 🟡 | ⭐⭐ | #35863 |
| 25 | 🧩 | Whisper 功能完善（language detection / response_format） | ⭐ | 🟢 | ⭐ | #25750 |
| 26 | 🚀 | Semantic KV Cache Reuse Interface 实现 | ⭐⭐⭐ | 🟢 | ⭐⭐ | #44223 |
| 27 | 🧩 | VLA (Vision-Language-Action) 模型支持 | ⭐⭐⭐ | 🟢 | ⭐ | #42100 |
| 28 | 🐛 | Gemma-4 + DFlash 在 Ampere 上无兼容 Attention Backend | ⭐⭐ | 🟡 | ⭐⭐ | #40382 |
| 29 | 🚀 | Explicit Cache Breakpoints & Cache Usage Accounting API | ⭐⭐ | 🟢 | ⭐⭐ | #44775 |
| 30 | 📖 | NIXL KV Connector Metrics 聚合语义文档 | ⭐ | 🟢 | ⭐ | #41230 |

---

## 🚀 特性开发

### Task 3: 多模态 Speculative Decoding Draft Model 支持

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • vLLM Speculative Decoding 机制<br>• Multimodal Pipeline（MRoPE 位置编码、多模态状态管理）<br>• vLLM Scheduler / Model Runner 流程 |
| **社区联系人** | @adityakamat24 (主动认领) |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐⭐ 核心功能 — 解锁 VL 模型的投机解码加速 |
| **关联来源** | • Issue: [#33458](https://github.com/vllm-project/vllm/issues/33458) |
| **建议起点** | `vllm/spec_decode/` 目录，EAGLE drafter 的 multimodal 扩展路径 |

**背景与描述：**

目前 vLLM 已支持 EAGLE drafter 的多模态推理，但仅限于不使用并行 drafting 的场景。对于外部 drafter 和并行 drafting 模式，由于 MRoPe 位置编码和多模态状态管理的复杂性，尚未实现支持。完成此任务将显著加速 VL 模型的推理吞吐。

**预期工作项：**

1. 研究 MRoPe 位置编码在 draft model 中的传递机制
2. 修改 `vllm/spec_decode/` 中的多模态状态管理，使其兼容并行 drafting
3. 为外部 drafter 实现多模态输入传递
4. 编写 e2e 测试，覆盖图像/视频输入场景

---

### Task 8: Multimodal Encoder 异步 Embedding Cache 加载

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • vLLM V1 引擎架构<br>• 多模态 Encoder Cache 机制<br>• EC Connector / EPD 解耦架构<br>• Python async/await |
| **社区联系人** | 关注 EC-Connector 相关 PR 作者 |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐⭐ 对 EPD 解耦场景影响大 |
| **关联来源** | • Issue: [#45242](https://github.com/vllm-project/vllm/issues/45242) |
| **建议起点** | `vllm/v1/engine/` 中的 scheduler 和 encoder cache 相关代码 |

**背景与描述：**

在 EPD 解耦架构中，encoder 的 embedding cache 加载是同步阻塞的。当 cache miss 时，scheduler 必须等待 embedding 从 remote encoder 拉取完毕才能继续调度。该 task 要实现异步加载模式，让 scheduler 在等待 embedding 的同时可以调度其他就绪请求。

**预期工作项：**

1. 设计异步 cache loading 状态机
2. 在 scheduler 中实现 PENDING 状态 hold-back 逻辑
3. 确保与现有 EC Connector 的兼容性
4. 添加 benchmark 验证吞吐改善

---

### Task 11: EPD 解耦下 mm_hash 和 mm_transfer_params 优化

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM EPD 解耦架构<br>• 多模态数据处理流程<br>• HTTP/gRPC 通信 |
| **社区联系人** | — |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 减少冗余网络 I/O |
| **关联来源** | • Issue: [#36328](https://github.com/vllm-project/vllm/issues/36328) |
| **建议起点** | Disaggregated Encoder 的 response 构建代码 |

**背景与描述：**

当前 EPD 模式下，当用户通过 URL 提交多模态请求时，Proxy 会向 Encoder 发送子请求获取 embedding。但 Encoder 的 response 中没有携带 `mm_hash` 和 `mm_transfer_params`，导致 Prefill 节点收到 embedding 后仍需重新获取，造成冗余网络 I/O 和 CPU 开销。

**预期工作项：**

1. 修改 Encoder 的 response 结构，附加 `mm_hash` 和 `mm_transfer_params`
2. 修改 Proxy/Prefill 节点逻辑，利用这些信息避免重复数据拉取
3. 编写集成测试验证端到端流程

---

### Task 15: 多模态模型 LoRA Tower/Connector 扩展支持

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM LoRA 架构<br>• 多模态模型 ViT / Connector 结构<br>• HuggingFace PEFT |
| **社区联系人** | @Zyyeric, @linitra24 (主动认领) |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐⭐ 提升 LoRA 生态覆盖 |
| **关联来源** | • Issue: [#31479](https://github.com/vllm-project/vllm/issues/31479) |
| **建议起点** | `vllm/lora/` 目录和已支持模型的 tower/connector LoRA 实现 |

**背景与描述：**

当前仅 Qwen VL 系列和 idefics3 支持 tower encoder 和 connector 的 LoRA 注入。需要在更多多模态模型（InternVL、LLaVA、Pixtral 等）中实现此功能。

**预期工作项：**

1. 选择一个目标模型（如 InternVL2 或 Pixtral）
2. 参考 Qwen VL 系列的 LoRA 实现
3. 为目标模型的 vision tower 和 connector 添加 LoRA 层支持
4. 添加单元测试和 e2e 测试

---

### Task 16: 多模态模型 CI 性能 Benchmark

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM Benchmark 工具<br>• CI/CD（GitHub Actions）<br>• GPU 性能分析（memory、throughput、latency） |
| **社区联系人** | @p88h（已实现 fake-vqa 工具）、@ywang96 |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐⭐ 防止多模态性能回退 |
| **关联来源** | • Issue: [#16353](https://github.com/vllm-project/vllm/issues/16353) |
| **建议起点** | `vllm/benchmarks/` 目录，@p88h 的 [fake-vqa](https://github.com/p88h/fake-vqa) 工具 |

**背景与描述：**

当前 vLLM CI 只有文本模型 benchmark，缺少多模态模型的性能回归测试。已有社区开发者 @p88h 实现了 fake-vqa 工具可用于诊断性能问题，需要将其集成到 vLLM CI 流水线中，定期测量峰值显存使用、batch 吞吐和编码延迟。

**预期工作项：**

1. 基于 fake-vqa 或类似方法设计多模态 benchmark 脚本
2. 选择 2-3 个代表性 VL 模型（如 Qwen3-VL-4B、InternVL2-2B）
3. 测量峰值显存、编码延迟、batch 吞吐三个核心指标
4. 配置 GitHub Actions workflow
5. 设置性能回归阈值告警

---

### Task 20: Image Token Pruning for Multimodal Models 实现

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • ViT 编码器和多模态融合机制<br>• Token pruning 算法（如 EVS、FastV 等）<br>• vLLM 多模态 Pipeline |
| **社区联系人** | RFC 作者 |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐⭐ 优化 VL 模型吞吐 |
| **关联来源** | • RFC: [#45098](https://github.com/vllm-project/vllm/issues/45098) |
| **建议起点** | `vllm/multimodal/` 中的图像预处理和编码流程 |

**背景与描述：**

该 RFC 提出在 LLM fusion 之前对 image tokens 进行剪枝，通过 `--image-pruning-rate` 配置启用。RFC 有意不描述具体剪枝方法，留给实现者基于 embedding 矩阵设计剪枝策略。此功能可减少传递给 LLM 的 token 数量，提升多模态模型吞吐。

**预期工作项：**

1. 研究现有 image token pruning 算法（如 FastV、TokenPacker 等）
2. 在 vLLM 多模态 Pipeline 中添加 pruning hook
3. 实现一种高效的剪枝方法（基于 embedding 范数或 attention score）
4. 添加 CLI 参数 `--image-pruning-rate`
5. 编写 benchmark 验证吞吐改善

---

### Task 21: DeepStream GPU 视频解码后端接入

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • NVIDIA DeepStream SDK<br>• GStreamer Pipeline<br>• vLLM 视频处理流程<br>• CUDA/NVMM |
| **社区联系人** | @wangshangsam, @ywang96, @Isotr0py |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐⭐ GPU 加速视频解码 |
| **关联来源** | • RFC: [#41843](https://github.com/vllm-project/vllm/issues/41843) |
| **建议起点** | `vllm/multimodal/video.py` |

**背景与描述：**

当前 vLLM 使用 CPU-based 解码器（如 PyAV）处理视频输入，对高分辨率、多路视频场景存在性能瓶颈。DeepStream 是 NVIDIA 的 GStreamer 工具包，支持硬件加速的 H.264/H.265 解码，覆盖常见封装格式和 RTSP 流。此任务将其作为视频加载后端集成到 vLLM 中。

**预期工作项：**

1. 研究 DeepStream Python bindings 和 GStreamer pipeline 构建
2. 实现 `DeepStreamVideoLoader` 类
3. 集成到 vLLM 的视频预处理 pipeline
4. 添加文档说明 DeepStream 依赖安装和配置
5. Benchmark 对比 CPU vs GPU 解码性能

---

### Task 22: 直接二进制/Multipart 文件上传 API

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • FastAPI multipart/form-data<br>• OpenAI-compatible API 协议<br>• 文件生命周期管理和安全 |
| **社区联系人** | @Alberto-Codes (已 draft PR #39003), @DarkLight1337 |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐⭐ 提升 API 易用性 |
| **关联来源** | • Issue: [#38531](https://github.com/vllm-project/vllm/issues/38531)<br>• Draft PR: [#39003](https://github.com/vllm-project/vllm/pull/39003) |
| **建议起点** | `vllm/entrypoints/openai/` 和 PR #39003 |

**背景与描述：**

当前 `/v1/chat/completions` 仅支持通过 URL 或 base64 编码提交图像/视频。此功能允许客户端直接通过 multipart/form-data 上传二进制文件，与 SGLang 和 LMDeploy 的功能对齐。已有 @Alberto-Codes 的 draft PR #39003 包含 atime-based TTL 和 LRU quota 机制。

**预期工作项：**

1. Review PR #39003 的设计和实现
2. 完善文件上传的 FastAPI endpoint
3. 实现安全的文件清理（TTL + LRU quota）
4. 添加 OSS 安全审查（防止恶意大文件上传）
5. 编写文档和 usage example

---

### Task 26: Semantic KV Cache Reuse Interface 实现

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • vLLM KV Cache 管理<br>• Prefix Caching 机制<br>• Embedding 相似度计算<br>• KV Cache Transfer |
| **社区联系人** | — |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐⭐ 创新性缓存复用机制 |
| **关联来源** | • RFC: [#44223](https://github.com/vllm-project/vllm/issues/44223) |
| **建议起点** | `vllm/v1/core/kv_cache_utils.py` |

**背景与描述：**

当前 Prefix Caching 基于精确 token 匹配，无法识别语义等价前缀。此 RFC 提出基于语义相似度的 KV Cache 复用接口，允许在语义相似的 prompt 之间共享 KV Cache，进一步提升缓存命中率。

**预期工作项：**

1. 设计语义哈希计算方法（基于 embedding clustering 或 LSH）
2. 实现 KV Cache lookup 的语义匹配层
3. 修改 scheduler 中的 cache 查询逻辑
4. 评估 cache hit rate 改善幅度
5. 编写设计文档

---

### Task 29: Explicit Cache Breakpoints & Cache Usage Accounting API

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • OpenAI-compatible API 协议<br>• vLLM Prefix Caching<br>• Tokenizer 和 prompt 处理<br>• REST API 设计 |
| **社区联系人** | @ajayr4j (已有 POC), @DarkLight1337 |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐⭐ 与 Anthropic API 对齐 |
| **关联来源** | • Issue: [#44775](https://github.com/vllm-project/vllm/issues/44775) |
| **建议起点** | `vllm/entrypoints/openai/` 和 @ajayr4j 的 POC 分支 |

**背景与描述：**

用户在从 Anthropic API 迁移到自部署 vLLM 时，失去了两个关键能力：(1) 通过 `cache_control` 显式标记缓存断点，(2) 查看每个请求的缓存性能账单。@ajayr4j 已实现 tokenizer-skip POC，需要进一步实现完整的 cache usage accounting。

**预期工作项：**

1. 实现 `/v1/chat/completions` 中对 `cache_control` 参数的解析
2. 在 response 中添加 cache usage 统计（hit tokens / miss tokens）
3. 与 Anthropic API 行为对齐
4. 编写 API 文档

---

## ⚡ 性能优化

### Task 10: TurboQuant + Gemma 4 Multimodal 5 层 Gate Blocker

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • KV Cache 量化（TurboQuant）<br>• Gemma 4 非因果 Attention（SWA + Global）<br>• CUDA kernel 适配<br>• FlashAttention 参数兼容 |
| **社区联系人** | @baoskee, @noonghunna, @mgoin |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐⭐ 解锁 Gemma 4 的内存优化 |
| **关联来源** | • Tracking: [#41403](https://github.com/vllm-project/vllm/issues/41403)<br>• Code TODO: `gemma4_mm.py:1024` |
| **建议起点** | `vllm/model_executor/models/gemma4_mm.py`, TurboQuant kernel |

**背景与描述：**

尝试在 Gemma 4 31B（多模态变体）上启用 TurboQuant KV Cache 量化时，需要穿过 5 个独立的 blocker gate（kernel padding、non-causal attention + head_dim=256 的 FA 兼容性、FP8 KV cache 路径等）。社区开发者 @noonghunna 在 Ampere 架构上又发现了第 6 个硬件物理限制。此任务是系统性解耦和逐一修复这些 blocker。

**预期工作项：**

1. 验证 TurboQuant kernel 对 Gemma 4 non-causal attention mask 的兼容性
2. 修复 kernel padding（ref: `gemma4_mm.py:1024` TODO）
3. 实现或适配适用于 head_dim=256 + non-causal 的 attention backend
4. 测试 TurboQuant 4-bit 量化精度
5. Benchmark 显存减少比例

---

### Task 28: Gemma-4 + DFlash 在 Ampere 上无兼容 Attention Backend

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • FlashAttention / DFlash 机制<br>• CUDA SM 架构差异（Ampere vs Hopper）<br>• Gemma 4 Attention 结构 |
| **社区联系人** | — |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 解决 Ampere 用户无法使用 DFlash |
| **关联来源** | • Issue: [#40382](https://github.com/vllm-project/vllm/issues/40382) |
| **建议起点** | `vllm/attention/backends/` |

**背景与描述：**

Gemma 4 使用 non-causal attention + head_dim=256，DFlash 当前缺少对这种参数组合的 kernel 支持，导致 Ampere 架构 GPU 无法使用 DFlash 后端。需要为 Ampere 找到可用的 attention backend 方案，或修改 DFlash 以支持此参数组合。

**预期工作项：**

1. 排查 DFlash 对 non-causal + head_dim=256 的限制根因
2. 实现 DFlash 对该参数组合的支持，或为 Ampere 推荐替代 backend
3. 在 Ampere GPU 上验证功能正确性
4. Benchmark 对比 eager mode vs 新方案

---

## 🐛 Bug 修复

### Task 4: Prefix Caching 忽略视觉输入导致并发下输出错误

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • vLLM Prefix Caching 机制（APC）<br>• 多模态输入的 token 化和 hash 计算<br>• KV Cache manager |
| **社区联系人** | @Godricly (提供了 `--disable-mm-preprocessor-cache` 的 workaround) |
| **紧急程度** | 🔴 High |
| **重要程度** | ⭐⭐⭐ 影响并发服务正确性 |
| **关联来源** | • Issue: [#20261](https://github.com/vllm-project/vllm/issues/20261) |
| **建议起点** | `vllm/v1/core/kv_cache_utils.py` 中的 cache key 计算逻辑 |

**背景与描述：**

当启用 prefix caching 时，APC 的 hash key 计算仅基于 text tokens，未包含视觉输入的信息。当两个请求共享相同的文本前缀但携带不同的图像时，cache 会错误地复用之前的 KV cache，导致第二个请求生成错误的输出。此问题是并发场景下的严重正确性 bug。

**预期工作项：**

1. 修改 cache key 计算逻辑，将多模态输入 hash（mm_hash）纳入 key
2. 确保不同图像的相同文本前缀产生不同的 cache entry
3. 编写多请求并发测试，覆盖相同文本+不同图像的场景
4. 验证不影响纯文本请求的 cache 行为
5. 测试对 cache hit rate 的影响

---

### Task 5: V1 引擎 Native Weight Update 后 Encoder Cache 脏数据

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM V1 引擎架构<br>• Encoder Cache 生命周期<br>• Weight Update / Hot Reload 机制 |
| **社区联系人** | @littlecircle0730 (已认领), @glaziermag (报告者) |
| **紧急程度** | 🔴 High |
| **重要程度** | ⭐⭐ 影响在线 weight 更新场景 |
| **关联来源** | • Issue: [#44910](https://github.com/vllm-project/vllm/issues/44910) |
| **建议起点** | `vllm/v1/engine/` 中的 weight update 和 encoder cache reset 逻辑 |

**背景与描述：**

当通过 `reset_encoder_cache` 进行 native weight update 时，encoder cache 中的旧 embedding 未正确失效，导致后续请求复用 stale 的多模态编码结果。直到手动调用 `reset_encoder_cache` 才会刷新。@littlecircle0730 已认领修复，但可以作为参考了解类似的 cache invalidation 问题。

**预期工作项：**

1. 在 weight update 流程中添加 encoder cache 自动失效逻辑
2. 确保 mm_hash 在 weight 变化时更新
3. 编写测试覆盖 weight update + 多模态请求场景

---

### Task 6: LoRA 同名 Reload 复用 Stale Encoder Cache

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM LoRA 管理<br>• Encoder Cache 生命周期<br>• Multimodal Tower/Connector LoRA |
| **社区联系人** | — |
| **紧急程度** | 🔴 High |
| **重要程度** | ⭐⭐ 影响 LoRA 热切换 |
| **关联来源** | • Issue: [#44939](https://github.com/vllm-project/vllm/issues/44939) |
| **建议起点** | `vllm/lora/` 中的 adapter 管理代码 + encoder cache |

**背景与描述：**

在启用 `--enable-tower-connector-lora` 时，用不同的 LoRA adapter 替换同名的 tower/connector LoRA 后，encoder cache 仍保留旧 adapter 的 embedding。Scheduler 将相同图像视为 cache hit，跳过重新编码，导致请求使用错误的 vision encoder embedding。

**预期工作项：**

1. 在 LoRA adapter 替换时，使相关的 encoder cache entries 失效
2. 在 cache key 中纳入 LoRA adapter 的标识
3. 编写测试覆盖 LoRA swap + 同图像请求场景

---

### Task 9: prompt_embeds 模式在 VL 模型中不工作

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM V1 引擎<br>• MRoPE 位置编码<br>• Prompt embedding 处理流程 |
| **社区联系人** | @zmylol (已分析根因), @xijia-tao (提供了 fix) |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 影响 VL embedding 推理 |
| **关联来源** | • Issue: [#44842](https://github.com/vllm-project/vllm/issues/44842) |
| **建议起点** | `vllm/v1/engine/` 中 MRoPE 初始化代码 |

**背景与描述：**

当使用 `prompt_embeds` 参数（而非 `prompt_token_ids`）调用 VL 模型时，V1 引擎的 MRoPE 路径在 `_init_mrope_positions` 中崩溃。@zmylol 已指出根因：纯 `prompt_embeds` 请求缺少 `prompt_token_ids`。@xijia-tao 提供了一个临时的 workaround。需要正式的修复方案。

**预期工作项：**

1. 分析 `prompt_embeds` 路径在 VL 模型中的完整处理流程
2. 修复 MRoPE 初始化对 prompt_embeds 请求的支持
3. 确保多模态输入（图像/视频）在 prompt_embeds 模式下正确处理
4. 编写测试

---

### Task 12: Qwen3-VL EVS 视频剪枝 CPU/CUDA Device Mismatch

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • Qwen3-VL EVS (Efficient Video Sampling) 机制<br>• PyTorch device placement |
| **社区联系人** | — |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 影响 Qwen3-VL 视频推理 |
| **关联来源** | • Issue: [#44200](https://github.com/vllm-project/vllm/issues/44200) |
| **建议起点** | `vllm/model_executor/models/qwen3_vl.py:1828` (`_create_final_video_embeddings`) |

**背景与描述：**

当使用 `video_pruning_rate > 0`（EVS）处理视频输入时，`_create_final_video_embeddings` 中发生 CPU/CUDA 设备不匹配错误。核心原因是剪枝后的索引在 CPU 上而 embedding tensor 在 GPU 上。此问题在 v0.22.0 版本中报告。

**预期工作项：**

1. 在 `_create_final_video_embeddings` 中添加设备一致性检查
2. 确保 EVS 剪枝索引被正确移到 GPU
3. 编写视频输入 + EVS 的测试用例

---

### Task 13: Multimodal Processor Cache 丢失 Audio（Qwen3-Omni）

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • Multimodal Processor Cache 机制<br>• Qwen3-Omni 的 audio/video 处理<br>• `use_audio_in_video` 参数 |
| **社区联系人** | — |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 影响 Qwen3-Omni 音频功能 |
| **关联来源** | • Issue: [#44538](https://github.com/vllm-project/vllm/issues/44538) |
| **建议起点** | `vllm/multimodal/processing/processor.py` 中的 cache 逻辑 |

**背景与描述：**

当使用 `use_audio_in_video=True` 参数时，multimodal processor cache 在处理过程中丢失了音频数据，导致 `StopIteration` 异常和 HTTP 400 错误。此问题在 v0.21.0 + Qwen3-Omni-30B-A3B 上出现。

**预期工作项：**

1. 追踪 processor cache 中 audio 数据的生命周期
2. 修复 `use_audio_in_video` 场景下的 cache key 计算和 data 保留
3. 编写测试覆盖 audio + video 混合输入场景

---

### Task 14: Qwen3.5-122B-A10B 并发图像请求 EngineCore Crash

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • vLLM EngineCore 架构<br>• MoE 模型 + 多模态的并发调度<br>• PyTorch 分布式 / NCCL<br>• 内存管理 |
| **社区联系人** | @ehofm (提供了新的复现信息) |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 影响大模型多模态服务稳定性 |
| **关联来源** | • Issue: [#37602](https://github.com/vllm-project/vllm/issues/37602) |
| **建议起点** | `vllm/v1/engine/core.py` 和 MoE 调度的并发路径 |

**背景与描述：**

当并发发送图像请求到 Qwen3.5-122B-A10B-FP8 (TP=4, 4xH200) 时，`EngineCore_DP0` 崩溃，错误为 `RuntimeError: Already borrowed`。社区反馈表明此问题可能与特定 workload 形状有关，最近仍有人遇到类似错误。

**预期工作项：**

1. 复现并发图像请求场景下的 crash
2. 分析 `Already borrowed` 错误的根因（可能的 tokenizer 并发问题或 memory pool 竞争）
3. 添加适当的锁或资源保护
4. 编写压力测试

---

### Task 23: AsyncLLM 在多模态 Pooling 请求下静默 Hang

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM AsyncLLM 引擎<br>• Encoder Cache 调度<br>• `max_num_batched_tokens` 配置 |
| **社区联系人** | @SoluMilken (提出 fail-fast fix 方案), @Sunt-ing (验证当前 main 已修复) |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 影响 embedding 服务的可靠性 |
| **关联来源** | • Issue: [#41969](https://github.com/vllm-project/vllm/issues/41969) |
| **建议起点** | `vllm/v1/engine/` 中的 `_try_schedule_encoder_inputs` |

**背景与描述：**

在 NVIDIA L4 + 多模态 pooling 模型 + 视频请求场景下，`AsyncLLM.encode()` 静默 hang（无日志、无错误、无返回），而同步版本 `LLM.embed()` 正常工作。根因是当 encoder input 需要的 embeds 超过 `max_num_batched_tokens` 的配置预算时，`_try_schedule_encoder_inputs` 无限期返回 `num_new_tokens=0`。

**预期工作项：**

1. 添加 fail-fast 检测：当单个 encoder input 超出配置预算时立即报错
2. 改进日志输出，明确提示配置调整建议
3. 编写测试覆盖边界条件

---

### Task 24: Voxtral-Realtime 多 Session 下 Transcription 停止

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • vLLM Realtime/Streaming 引擎<br>• Voxtral 模型架构<br>• ASR 多 session 管理 |
| **社区联系人** | — |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 影响实时 ASR 服务 |
| **关联来源** | • Issue: [#35863](https://github.com/vllm-project/vllm/issues/35863) |
| **建议起点** | `vllm/model_executor/models/voxtral_realtime.py` |

**背景与描述：**

Voxtral-Realtime 模型在建立第 3 个并发 session 后停止返回 transcription 文本，但模型仍在运行（无 crash）。推测是 encoder cache 或 session 状态管理在多 session 场景下存在资源泄漏或竞争。

**预期工作项：**

1. 复现 3+ 并发 Voxtral session 的问题
2. 排查 encoder cache 在多 session 下的分配策略
3. 修复 session 状态隔离或 encoder cache 共享逻辑
4. 编写多 session 并发测试

---

## 🧩 模型支持

### Task 1 & 2: ViT CUDA Graph 适配（DeepSeek-OCR-2 / Ernie-4.5-VL）

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • ViT 编码器架构<br>• CUDA Graph 捕获机制<br>• vLLM `SupportsEncoderCUDAGraph` 接口<br>• NCCL / CUDA 编程 |
| **社区联系人** | @shen-shanshan (tracking), @ywang96, @Isotr0py |
| **紧急程度** | 🔴 High |
| **重要程度** | ⭐⭐⭐ ViT CUDA Graph 全覆盖的重要里程碑 |
| **关联来源** | • Tracker: [#38175](https://github.com/vllm-project/vllm/issues/38175)<br>• Reference PR: [#38061](https://github.com/vllm-project/vllm/pull/38061) (Qwen3-VL) |
| **建议起点** | `vllm/model_executor/models/qwen3_vl.py` 中 Qwen3-VL 的 CUDA Graph 实现 |

**背景与描述：**

ViT Full CUDA Graph 是当前 vLLM 多模态优化的核心项目。已有 Qwen3-VL、Qwen3.5、GLM-4.1V、Kimi K2.5 等多个模型完成适配，但 DeepSeek-OCR-2 和 Ernie-4.5-VL 仍待支持。适配流程已标准化：实现 `SupportsEncoderCUDAGraph` 接口 → 测试 → 更新文档 → 加入 CI。

**预期工作项：**

1. 研究目标模型的 ViT 架构，参照 Qwen3-VL 的实现
2. 实现 `get_encoder_cudagraph_outputs()` 和相关接口
3. 在对应 GPU 上进行 e2e 测试和 benchmark
4. 更新文档中的支持模型列表

---

### Task 7: Gemma 4 接入 /v1/audio/transcriptions 端点

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • Gemma 4 模型架构<br>• vLLM `SupportsTranscription` 接口<br>• OpenAI-compatible audio API |
| **社区联系人** | @SoluMilken (已认领), @montvid (提供了实现方案) |
| **紧急程度** | 🟡 Medium |
| **重要程度** | ⭐⭐ 完善 Gemma 4 音频能力 |
| **关联来源** | • Issue: [#40994](https://github.com/vllm-project/vllm/issues/40994) |
| **建议起点** | `vllm/model_executor/models/gemma4_mm.py`，参考 #23735 (Gemma3n 的 transcription 实现) |

**背景与描述：**

Gemma 4 已有 audio tower 路径，但未实现 `SupportsTranscription` 接口，导致 `/v1/audio/transcriptions` 端点不可用。需要添加 `get_generation_prompt()` + `get_speech_to_text_config()` 等接口，参考 Gemma3n 的实现。@SoluMilken 已认领此 task。

**预期工作项：**

1. 在 `Gemma4ForConditionalGeneration` 中实现 `SupportsTranscription` 接口
2. 实现 `get_generation_prompt()` 和 `get_speech_to_text_config()`
3. 在 Gemma 4 E4B 变体上进行 e2e 测试
4. 更新文档和测试

---

### Task 25: Whisper 功能完善（language detection / response_format）

| 维度 | 详情 |
|------|------|
| **难度** | ⭐ Beginner |
| **前置知识** | • Whisper 模型架构<br>• OpenAI-compatible `/v1/audio/transcriptions` API<br>• vLLM ASR 处理流程 |
| **社区联系人** | @xsank (已在 tracker 下讨论) |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐ 完善性工作 |
| **关联来源** | • Tracker: [#25750](https://github.com/vllm-project/vllm/issues/25750) |
| **建议起点** | `vllm/model_executor/models/whisper.py` |

**背景与描述：**

Whisper 的 feature request tracker 中列出了多个待实现的功能：
- 支持不同的 `response_format`（如 SRT、VTT）
- 自动语言检测（`language` 参数为空时）
- 改进多语言处理能力
这些 task 相对简单，适合首次贡献者。

**预期工作项：**

1. 选择一个 sub-feature（如 `response_format` 支持）
2. 参考 OpenAI Whisper API 规范
3. 在 vLLM 的 Whisper 实现中添加对应处理
4. 编写测试和文档

---

### Task 27: VLA (Vision-Language-Action) 模型支持

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐⭐ Advanced |
| **前置知识** | • VLA 模型架构（如 RT-2, Octo 等）<br>• Action token 编码<br>• vLLM 多模态模型注册机制 |
| **社区联系人** | — |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐ 新兴方向 |
| **关联来源** | • Issue: [#42100](https://github.com/vllm-project/vllm/issues/42100) |
| **建议起点** | `vllm/model_executor/models/registry.py` |

**背景与描述：**

VLA（Vision-Language-Action）模型是具身智能领域的核心，将视觉/语言理解与机器人动作生成结合。vLLM 当前不支持此类模型。此任务需要研究 VLA 模型架构，在 vLLM 的多模态框架下设计通用的 VLA 支持方案。

**预期工作项：**

1. 研究 1-2 种代表性 VLA 模型架构（如 OpenVLA、RT-2-X）
2. 设计 VLA 模型的注册和接口方案
3. 实现一个简化的 VLA 模型作为 prototype
4. 编写 RFC 文档

---

## 🔧 代码重构

### Task 17~19: Transformers v5 升级适配

| 维度 | 详情 |
|------|------|
| **难度** | ⭐ Beginner |
| **前置知识** | • HuggingFace Transformers 库<br>• vLLM 模型注册机制<br>• 目标模型架构 |


**关联来源**：
• InternVL2: [#38425](https://github.com/vllm-project/vllm/issues/38425) — `good first issue`
• IsaacForConditionalGeneration: [#38389](https://github.com/vllm-project/vllm/issues/38389) — `good first issue`
• Tarsier2: [#38736](https://github.com/vllm-project/vllm/issues/38736) — `good first issue`
• Meta issue: [#38379](https://github.com/vllm-project/vllm/issues/38379)

**背景与描述：**

vLLM 正在从 Transformers v4 迁移到 v5，需要社区开发者帮助适配各个模型。这些 task 被标记为 `good first issue` + `help wanted`，适合首次贡献者。工作内容包括修复模型注册、tokenizer 配置、meta device 初始化等问题。

**预期工作项：**

1. 选择一个目标模型（建议从 InternVL2 开始，因为有明确的错误信息）
2. 在 Transformers v5 环境下复现问题
3. 修复兼容性问题（如 API 变更、参数重命名）
4. 运行现有测试确保无回归
5. 提交 PR

**社区联系人**：@njhill, @DarkLight1337（Transformers v5 升级相关）

---

## 📖 文档补充

### Task 30: NIXL KV Connector Metrics 聚合语义文档

| 维度 | 详情 |
|------|------|
| **难度** | ⭐ Beginner |
| **前置知识** | • vLLM KV Connector 机制<br>• NIXL 通信框架<br>• Prometheus metrics |
| **社区联系人** | — |
| **紧急程度** | 🟢 Low |
| **重要程度** | ⭐ 文档完善 |
| **关联来源** | • Issue: [#41230](https://github.com/vllm-project/vllm/issues/41230) |
| **建议起点** | `docs/source/features/` 和 NIXL connector 代码 |

**背景与描述：**

NIXL connector 暴露的 metrics 在多节点场景下有复杂的聚合语义（per-node vs global），当前缺少文档说明，用户容易误解 metric 含义。需要添加文档解释各个 metric 的聚合方式。

**预期工作项：**

1. 阅读 NIXL connector 代码，理解每个 metric 的计算方式
2. 编写文档：说明哪些 metric 是 per-node，哪些是 global aggregate
3. 提供 Prometheus 查询示例

---

## 📊 附录

### A. 关键数据来源汇总

| 来源类型 | 数量 | 备注 |
|----------|------|------|
| Open Issues（多模态相关） | 50+ | 其中 good first issue: 4 个 (Transformer v5 相关) |
| Multi-modality Labeled Issues | 13 | 含 3 个 RFC、4 个 feature request、5 个 new-model |
| Related Merged PRs | 30+ | 近 3 个月 |
| Discussions | 2 | 相关度较低 |
| Code TODOs | 15+ | 分布在 Qwen3-VL、Gemma4、InternVL、Pixtral 等模型 |
| ViT CUDA Graph 未完成子任务 | 10+ | 含 2 个新模型支持 + 1 个 bugfix |
| Whisper Feature Tracker 子项 | 5+ | language detection / response_format 等 |

### B. 活跃社区联系人

| GitHub ID | 角色 | 相关模块 | 活跃度 |
|-----------|------|---------|--------|
| @ywang96 | Maintainer | Multimodal Core | 高 |
| @Isotr0py | Maintainer | Multimodal Core | 高 |
| @DarkLight1337 | Maintainer | V1 Engine / Multimodal | 高 |
| @shen-shanshan | Contributor | ViT CUDA Graph / Qwen3-VL | 高 |
| @njhill | Maintainer | Transformers v5 迁移 | 高 |
| @wangshangsam | Contributor | Video / DeepStream | 中 |
| @BugenZhao | Contributor | Multimodal / EPD | 中 |
| @mgoin | Contributor | Gemma 4 / TurboQuant | 中 |
| @johncalesp | Contributor | ViT CUDA Graph | 中 |
| @grYe99 | Contributor | GLM-V CUDA Graph | 中 |
| @littlecircle0730 | Contributor | Encoder Cache Bugfix | 新 |
| @SoluMilken | Contributor | Gemma 4 Audio | 新 |
| @ajayr4j | Contributor | Cache Control API | 新 |
| @zmylol | Contributor | VL prompt_embeds Bugfix | 新 |

### C. 参考链接

- [Multi-modality Labeled Issues](https://github.com/vllm-project/vllm/issues?q=is%3Aopen+label%3Amulti-modality)
- [Good First Issues](https://github.com/vllm-project/vllm/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
- [ViT CUDA Graph Tracker (#38175)](https://github.com/vllm-project/vllm/issues/38175)
- [Whisper Feature Tracker (#25750)](https://github.com/vllm-project/vllm/issues/25750)
- [Transformers v5 Migration (#38379)](https://github.com/vllm-project/vllm/issues/38379)
- [Multi-modality RFC (#4194)](https://github.com/vllm-project/vllm/issues/4194)
