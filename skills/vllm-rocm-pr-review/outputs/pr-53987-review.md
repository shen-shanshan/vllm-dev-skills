# PR #53987: [Spec Decode][ROCm] Add FLy: Entropy-Gated Deferred Verification for Draft-Model Speculative Decoding

> **Author**: @eecspan (AMD) | **State**: OPEN | **Date**: 2026-08-27（最新活动 2026-09-05）
> **Branch**: `AMD-AGI:feat/fly-verifier` → `vllm-project:main` | **Labels**: `documentation`, `rocm`, `speculative-decoding`, `mrv2`, `verified`
> **Changes**: +846 -23 行，13 个文件 | **ROCm 相关性**: 部分相关（代码全部在 backend-agnostic 的 spec-decode 共享路径，无 aiter/mori/rocm 专属文件；AMD 出品、`rocm` label、MI355X/MI300X 主评测）
> **CI**: 状态未知（fetch 脚本未取到 check 数据）

## 1. 动机 (Motivation)

标准投机解码的拒绝采样在第一个被拒位置截断整段 draft，丢弃后续已被 target 验证过的 token。FLy（arXiv:2511.22972，ICLR 2026）提出熵门控延迟验证：当某位置被拒但 target 熵高（自身不确定）且其后 W 个位置都被原生接受时，把该次拒绝翻转成接受。本 PR 将 FLy 作为 opt-in 验证策略（`rejection_sample_method="fly"`）落地到 vLLM V1/V2 的 rejection sampler，AMD MI355X 上 40 个模型-数据集配置全部快于标准 SD（1.067x–1.938x），质量保留率 ≥97.9%。设计上不改动 drafting/KV/attention，仅在验证阶段改写原生 kernel 的输入（V1 pre-pass）或 kernel 内延迟提交（V2 状态机）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 文件 | 改动 |
|------|------|------|
| 核心实现 | `vllm/v1/spec_decode/fly.py`（新增 191 行） | `compute_fly_entropy()` + 两个 Triton pre-pass kernel（greedy/random） |
| V1 采样 | `vllm/v1/sample/rejection_sampler.py` (+65) | 接入 pre-pass、复用单次 softmax、CPU 显式报错 |
| V2 采样 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` (+79) | `_rejection_kernel` 内 pending-rejection 状态机（原地延迟验证） |
| V1 worker | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` (+7) | 解析 `"fly"` 配置并传参 |
| 配置 | `vllm/config/speculative.py` (+39) | `fly_window_size`/`fly_entropy_threshold` 字段 + 校验（含 lossy 警告、异构词表支持、与 local_argmax_reduction 互斥） |
| 环境变量 | `vllm/envs.py` (+3) | `VLLM_FLY_ENTROPY_TOP_K`（已正确注册到 envs.py ✓） |
| 其他 | `llm_base_proposer.py`、`gpu_model_runner.py` | draft_probs 收集条件加入 `"fly"` |
| 文档/测试 | docs（fly.md 新增 + README drive-by fix）、3 个测试文件 (+402) | 单测覆盖 greedy/random 延迟、熵门控、窗口边界、配置校验 |

## 3. Review 意见 (Findings)

**意见类型 × 数量**：⚠️ 建议修复 ×4，📝 备注 ×4，🔴 0

**⚠️【一致性】MRV1 与 MRV2 的熵门控输入分布不一致** `[已验证]`
- **问题**：PR 声称"熵从 temperature、top-k、top-p 处理后的目标分数计算"。验证 V1 路径（`vllm/v1/sample/rejection_sampler.py`）：`apply_sampling_constraints` 对非 greedy 请求先做温度缩放 + top-k/top-p 再进 `rejection_sample`，熵算自处理后分布；但 `all_greedy` 时直接返回原始 logits。而 V2 路径（`rejection_sampler_utils.py`）的 `compute_fly_entropy(target_logits[:, :vocab_size], from_logits=True)` 永远算自**未除温度的原始 logits**——温度是在 kernel 内部（`/ temp`）才应用的。即：temperature=0 时两边一致（主评测不受影响），但 **temperature>0 时同一个 0.3 阈值在 V1/V2 门控的是不同分布**，延迟验证行为随 runner 分叉。
- **影响**：PR 明确声称支持 temp>0（补充评测含 temperature 1），但那些结果只反映 V1 行为；用户在 V2 + temp>0 下会得到不同的验证决策，跨 runner 的评测结论不可推广。
- **行动**：作者应当在 V2 侧对熵输入应用与 V1 相同的 temperature 处理（或明确文档化差异），并给出 V2 + temp>0 的行为验证。

**⚠️【性能】V2 路径全词表 fp32 物化 + 多个额外 kernel，开销未在 V2 语境下测量** `[已验证代码 / 影响为推测]`
- **问题**：`compute_fly_entropy` 对完整 `[num_logits, vocab]` logits 做 `to(torch.float32)` 全量拷贝 + `topk` + 全词表 `logsumexp`。V2 rejection kernel 的设计初衷正是按 block 处理避免全词表物化（max/sumexp 均 block 化），此改动在解码关键路径上每轮验证引入约 3 次全词表扫描（BS=32×K=15 时约 480×128K×4B ≈ 235MB 临时内存 + 多个 kernel 启动）。作者"softmax 复用比 logits 直算快 1.17–1.23x"的基准是在 V1 流程下测的，V2 语境（大 batch、block 化设计、HIP graph 环境）无对应测量。
- **影响**：V2 + FLy 每轮验证的额外延迟与临时内存峰值未量化；在 MI355X 大 batch 下可能侵蚀收益。
- **行动**：作者应当提供 V2 路径熵计算开销的独立测量（或改用 block-wise top-k 近似），并将结论从 V1 基准中区分开。

**⚠️【文档】PR body 与当前代码状态不一致（V1-only 描述已过时）** `[已验证]`
- **问题**：PR body 的 Usage 与 Limitations 章节仍写"FLy 是 V1-only、V2 配置需回退 V1 或直接报错、V2 适配是 planned follow-up work"，并描述了代码中并不存在的"V2 回退警告"行为；而代码已实现 V2 支持（diff 中的状态机 + 最新评论确认），`fly.md` 也写明"supported by both model runners"。
- **影响**：reviewer 和用户按 body 理解会低估功能范围、误解配置行为。
- **行动**：作者应当同步更新 PR body 的 Usage/Limitations 章节，删除已不存在的回退/报错描述。

**⚠️【评测】maintainer 要求的 V2/MTP 系统评测与前沿基准尚未补齐** `[已验证（来自讨论区）]`
- **问题**：@benchislett 明确要求：至少一个新模型（DSV4/GLM5.3/Kimi K3/Qwen3.8）+ 前沿基准（TerminalBench/SWEBench-Pro/有区分度的 HLE）+ 至少一次 MTP 主评测。作者目前仅在评论中给出一个 Qwen3.5-9B+MTP(K=4) 的 smoke test（~1.15x，GSM8K 128 条），PR body 的评测章节（40 配置表）全部是 V1 `draft_model` 路径。此外主评测目标模型均为 2 年以上的 Llama 3.1/Qwen3-235B 系列。
- **影响**：合并审查可能因此被卡；V2 路径（本 PR 最新且最复杂的新增部分）只有 smoke test 级别的端到端验证。
- **行动**：作者应当补充 V2/MTP 的系统评测或将 smoke test 结果及约束正式写入 body。

**📝【兼容性】新增的异构词表 × local_argmax_reduction 互斥硬错误影响非 FLy 用户** `[已验证]`
- **问题**：`_verify_args` 新增 `use_heterogeneous_vocab and use_local_argmax_reduction → ValueError`，该检查**不依赖 FLy**，对所有用户的既有配置组合生效。当前 main 中无此检查，即此前该组合是允许的（token-level intersection 确实需要完整 draft logits，组合本身大概率是坏的）。
- **影响**：此前"能跑"（哪怕结果不对）的配置升级后直接报错——若该组合此前实际可用，则是行为回归。
- **行动**：建议作者在 PR 描述中说明该组合此前确实损坏/不支持，并在 CHANGELOG 类文档中标注这一新限制。

**📝【可维护性】同一算法在 V1 pre-pass 与 V2 in-kernel 双实现，门控不变量已出现漂移** `[已验证]`
- **问题**：四个门控条件（熵阈值、窗口完整性、p>0、后续 W 位置原生接受）在 `fly.py` 两个 kernel 和 `_rejection_kernel` 状态机中重复实现，且已产生语义漂移（见 ⚠️ 一致性 finding）。未来任一改动需同步三处。
- **影响**：长期维护成本与行为分叉风险。
- **行动**：建议作者至少在 `fly.md` 或 `fly.py` docstring 中固化门控定义，注明两 runner 的输入差异；长期考虑抽公共抽象。

**📝【数字溯源】评测数字无脚本/命令可复现** `[unverified]`
- **问题**：40 配置 + NVIDIA 交叉验证表格详尽（GPU 型号、TP、BS、torch.compile/HIP graph 均有说明），但未附评测脚本、命令或日志，ROCm 版本、vllm commit 亦未给出。
- **行动**：建议作者附上评测脚本或日志链接，便于 maintainer 抽查。

**📝 其他**：mergify 已提示存在 merge conflict，合并前需 rebase。

**正面确认**：FLy 关闭时零开销保证成立（`FLY_WINDOW_SIZE=0` constexpr 分支，不计算熵、不启动 kernel）；`VLLM_FLY_ENTROPY_TOP_K` 已正确注册进 `vllm/envs.py` 并写入文档；NVIDIA B300 交叉验证覆盖了 A2（共享路径跨后端回归）风险；新参数均带默认值，旧调用点不受影响；对 `_rejection_kernel` 的 pending-window 状态机做了逐路径推演（窗口完成提交、窗口失败回退到 rejected_argmax、greedy/random/placeholder 边界），未发现 🔴 级正确性问题。

## 4. 现有讨论 (Existing Discussion)

- **@benchislett（核心评审，两轮）**：总体正面（"integration is clean, gains significant, degradation minimal, algorithm elegant"），但提出两点硬要求：① MRV1 支持不够，**必须支持 MRV2**（→ 作者已实现并给出 MTP smoke test）；② 评测需新模型 + 前沿基准 + MTP（→ 待补）。其余意见（熵 top-k 环境变量化、lossy warning_once、窗口默认值动态推导、放宽异构词表限制）均已落实于 commit `355ff75`。
- **CodeRabbit 静态审查**：两个 Minor 问题（固定默认 6 导致 `num_speculative_tokens≤6` 时必失败；异构词表被过度拒绝）均已修复。
- **作者回复**：解释 softmax 复用方案（benchmark 1.17–1.23x 更快，熵误差 1.86e-8）；确认 FLy 现已支持 MRV2。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 算法与集成质量高（门控设计、pre-pass 改写输入的思路、零开销回退均干净），但存在 MRV1/MRV2 熵门控分布不一致这一实际行为分叉、V2 熵开销未量化、以及 maintainer 要求的 V2/MTP 系统评测未补齐；另有 merge conflict 与 PR body 过时问题。建议作者优先处理熵门控一致性 + 补充 V2 评测后推进合并。

---
*报告生成时间: 2026-09-09 | 规则目录版本: 2026-08-20*
