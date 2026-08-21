# PR #52882: [ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER

> **Author**: @Fangzhou-Ai | **State**: OPEN | **Date**: 2026-08-19
> **Branch**: `Fangzhou-Ai:rocm-dsv4-c4a-topk` → `vllm-project:main` | **Labels**: performance, rocm, deepseek, DSv4
> **Changes**: +464 -27 across 5 files | **ROCm 相关性**: 完全相关

## 1. 动机 (Motivation)

DeepSeek V4 C4A（压缩上下文稀疏注意力）的 indexer top-k（FP32 logits、固定宽度 262,144 列、k=1024）是 MI355 上的瓶颈。此前简单把 decode top-k 无条件切到 AITER（#52878 的思路）在长上下文任务（agentX benchmark）上有回归——AITER v0.1.19 的 decode kernel 是 one-block，长行不如 vLLM 原生 split/merge 路径。本 PR 提出混合方案：短/中上下文走 AITER 公开 `aiter.ops.topk` API，长上下文（gfx950、≤256 行）走调优后的原生两-launch split/merge 路径（新增 capture-stable 的 device-length-aware kernel），>256 行选 AITER。同时修复了 AITER 集成的两个语义坑：MTP 的 2-D 精确 per-row 长度需 flatten + `next_n=1` 传入；prefill 的 AITER 全局索引需本地化。跟踪 issue #41820，是 #46172/#50470/#52878(decode 部分) 的完整替代。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|---|---|
| `csrc/libtorch_stable/sampler.cu` (+122/-11) | 新增 `topKPerRowDecodeDeviceLengthAware` split/merge kernel 对（行内 64K 列阈值在 3/2 blocks 间切换，capture-stable）；gfx950 专属 dispatch：≤64 行用 8/4-block 调优配置（原为固定 10），65-256 行走新 kernel；CUDA 路径零变化 |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` (+158/-16) | 新增 `_aiter_top_k_per_row_prefill`/`_decode` 包装（ImportError 守卫回退 native）、prefill 索引本地化 Triton kernel（保留 -1 sentinel）、`_use_aiter_c4a_decode_topk_auto` 混合策略（≤65,536 压缩列或 >256 行 → AITER）；FULL 图用 `max_model_len` 保守上界选 kernel |
| `vllm/model_executor/layers/sparse_attn_indexer.py` (+3) | `compress_ratio` 参数贯穿到 custom op 调用 |
| `vllm/models/deepseek_v4/attention.py` (+1) | 向 `SparseAttnIndexer` 传 `self.compress_ratio` |
| `tests/kernels/test_top_k_per_row.py` (+180) | 5 个新测试：gfx950 长上下文 2-D/1-D seq_lens、AITER prefill 本地化、AITER decode 精确 MTP 长度、auto 策略纯函数 |

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---|---|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 5 |
| 📝 建议/备注 | 3 |

> 本次 review 做了逐项语义验证：AITER v0.1.19 两个 API 签名与调用完全匹配；`decode_metadata.seq_lens` 在所有形状下（2-D 精确 per-row / flatten 1-D per-token / plain 1-D per-request）flatten + `next_n=1` 均语义正确（`indexer.py` 的 docstring 与 builder 代码证实）；prefill 本地化与原生 kernel 的 local 索引语义一致；新 split/merge kernel 的 activeBlocks 计算在 split 与 merge 间一致、skipped block 不会被 merge 读取；`direct_register_custom_op` 按函数签名推断 schema，real/fake 签名同步更新，注册无需额外改动；`max_model_len`、`forward_context.cudagraph_runtime_mode`、`CUDAGraphMode`、`get_device_prop()->gcnArchName` 等符号均确认存在。未发现 🔴 级正确性问题。

**⚠️【测试】gfx950 专属测试在 CI 硬件上全部跳过，新路径零持续覆盖** `[已验证]`
- **问题**: 新增测试均 `skipif(not is_rocm())` + 运行时 `gcnArchName.startswith("gfx950")` 才执行（`tests/kernels/test_top_k_per_row.py:390-392` 等 5 处），而 vLLM 的 AMD CI 队列是 gfx942（MI300）。head commit 的 check-runs 中无任何 AMD 硬件 CI 结果，仅 pre-commit 失败（作者已说明是 main 上 #49688 引入的 `vllm/platforms/cpu.py` 格式问题，与本 PR 5 个文件无关，可信）。
- **影响**: 新 kernel 与 AITER 路径合入后没有任何 CI 防线，未来任何改动破坏 gfx950 top-k 正确性都会静默落到 MI355 用户头上（这是静默数值错误风险最高的路径）。
- **行动**: 建议作者确认 AMD 是否有 gfx950 CI 队列；若没有，应在 PR 描述中显式声明覆盖缺口，并请求 maintainer 在 gfx950 机器上跑 `tests/kernels/test_top_k_per_row.py` 后再合入。

**⚠️【测试/方法学】84-cell kernel 矩阵与 serving 数字不可从仓库复现，标 [unverified]** `[已验证]`
- **问题**: (a) 84-cell 矩阵由"intentionally not part of this PR"的本地 HIP-event harness 生成，无脚本可查；(b) 主要 serving 对比（+4.95%）两轮运行除代码外还改了 `max_model_len`(1M→9472)、`max_num_seqs`、`max_num_batched_tokens`，作者自述"observed deployment comparison, rather than an isolated commit or single-flag A/B"；(c) same-flags parent 对比（+26.76%）中 parent 进程复用了 PR head 构建的 extension 未重建（作者已承认），且 +26.76% 数字与 primary 对比的 +2~9% 差距本身说明了方法学问题。
- **影响**: 核心性能声明（10K 上下文 2.85x、500K 1.27x、E2E +4.95% 等）无法被 reviewer 独立验证；其中 +26.76% 的 parent 对比结论不可信。
- **行动**: 建议作者在 PR 描述附 harness 脚本（或 gist 链接），并在标 ready 前完成 extension 重建后的交替 A/B。

**⚠️【性能】PIECEWISE 图下 dispatch 按 trace 时 `max_seq_len` 烘焙进图** `[已验证]`
- **问题**: `rocm_aiter_mla_sparse.py:1014-1017` 只在 FULL 模式用保守上界 `max_model_len`；PIECEWISE 模式（按 batch 键控、不按 seq len）用 `layer_attn_metadata.max_seq_len // compress_ratio`，该值在 trace 时求值并烘焙进图。replay 时实际 seq len 变化不会改变已捕获的 kernel 选择。
- **影响**: 无正确性影响（AITER 与 native 两条路径均产出正确 top-k），但可能"烘焙了慢路径"：例如 trace 时 max_seq_len 短→烘焙 AITER，replay 时行已变长→本应走 native 却仍在跑 AITER one-block。FULL 图已正确处理，PIECEWISE 是遗漏。
- **行动**: 建议作者确认 PIECEWISE 下的烘焙偏差是否可接受；若不可接受，对 PIECEWISE 也使用 `max_model_len` 保守上界（代价是 PIECEWISE 长行用 native，与 FULL 行为一致）。

**⚠️【兼容性】sampler.cu 的 gfx950 分支激活面宽于验证面** `[已验证]`
- **问题**: `csrc/libtorch_stable/sampler.cu:744-746`（device-length-aware）与 `:782-784`（8/4-block 调优）的门控只有 `topK == 1024 && numColumns >= 200,000 && gfx950` 与行数区间——任何满足条件的调用者（不限 C4A，`_C.top_k_per_row_decode` 是通用采样算子）都会命中新路径。调优参数（8/4 blocks、64K 阈值、1024 threads）全部针对 C4A 的 FP32、262,144 列、k=1024 shape。
- **影响**: 正确性上我已验证 job 对任意 config 成立（per-block topK + merge 语义与原始 10-block 路径一致），风险主要是非 C4A 大 vocab 模型在 gfx950 上可能性能回退；当前此类调用者极少（需 ≥200K 列 vocab），但门控未表达"C4A 专属"意图。
- **行动**: 建议作者在 PR 描述中说明该分支对所有 ≥200K 列 + topK=1024 调用者生效，或考虑收紧门控（如限定 stride1==1 / 列数区间），至少留注释。

**⚠️【可维护性】65,536 阈值与 block 数在 C++/Python 双处硬编码，无单一来源** `[已验证]`
- **问题**: `sampler.cu:630` 的 `kLongRowThreshold = 64 * 1024` 与 `rocm_aiter_mla_sparse.py:136` 的 `max_valid_seq_len <= 65_536` 是同一分界线的两份拷贝（当前数值一致，边界语义也对齐：=65536 → AITER，>65536 → native 2-block）；另有 `kShortRowNumBlocks = 3`（kernel 内）与 host 端 `multipleBlocksPerRowConfig = 3`（workspace 形状）的硬编码耦合。
- **影响**: 将来调参只改一处会导致 dispatch 边界与 kernel 行为错位（例如 Python 放行 65537 行给 native、C++ 却因阈值不同选了不同 block 数——正确性仍由 job 保证，但性能结论失效）；`kShortRowNumBlocks` 若与 host config 失配则是越界写。
- **行动**: 建议作者在两处互相加注释引用（"must match ..."），或在 PR 描述中记录该耦合。

**📝【设计】`deviceLengthAware` 模板参数在 job 体内未使用** `[已验证]`
- **问题**: `sampler.cu:334-337` 的 `topKPerRowJob` 新增 `deviceLengthAware` 参数，但 job 体（histogram/shortcut/final store）无任何 `if constexpr (deviceLengthAware)` 分支，仅有 `static_assert` 用它约束实例化组合。
- **影响**: 无运行时影响，属于死参数/意图声明。读者会期待它改变 job 行为。
- **行动**: 建议作者在注释中说明该参数仅用于实例化区分（或移除，保留 static_assert 语义即可）。

**📝【依赖】优化路径依赖 aiter ≥ 0.1.19 的 `aiter.ops.topk` 公开 API，requirements 无版本体现** `[已验证]`
- **问题**: `aiter.ops.topk.top_k_per_row_decode/prefill` 在 v0.1.19 中存在且签名匹配（已核实），但 vLLM 的 `requirements/rocm.txt` 不 pin aiter（由 docker/环境管理）。ImportError 守卫（`_get_aiter_topk_ops` 返回 None → 回退 native）保证旧版本下正确性无虞。
- **影响**: 若部署环境 aiter < 0.1.19，新优化路径静默不激活，性能收益为 0 且无任何日志提示。
- **行动**: 建议作者在 PR 描述中注明最低 aiter 版本要求（0.1.19），便于部署方核对。

**📝【流程】作者自述验证未完成，prefill multi-block 路径的图捕获安全性建议补证** `[已验证]`
- **问题**: PR 描述明确"final-head serving smoke 与完整 1,319 题 GSM8K 留待提交者"；另外 aiter prefill 在 `topk_use_mulblocks(numRows, stride0)` 为真时走 multi-block 路径（跨 block atomic counter + 自复位 workspace），84-cell harness 是 eager 的、serving smoke 的 prompt 仅 5K token，长 prefill chunk 在 PIECEWISE 捕获下走 mb 路径的 replay 正确性未被直接验证（aiter 的 workspace 设计目标是 graph-safe，但无本 PR 的直接证据）。
- **行动**: 建议 review 时追问：长 prefill（如 64K+ token chunk）在 FULL_AND_PIECEWISE 图下是否实测过 mb 路径的 capture/replay。

## 4. 现有讨论 (Existing Discussion)

- **作者**（2026-08-19）：回应 #52878 的简单 AITER 替换在长上下文（agentX benchmark）引入回归，说明本 PR 的混合方案动机。
- **作者**（2026-08-19）：pre-commit 失败系 main 上 #49688 引入的 `vllm/platforms/cpu.py:184,190` 格式问题，与本 PR 文件无关，待 main 修复后重跑。
- **@shen-shanshan**（2026-08-20）：评论 `@claude review` 触发 bot 审查；**claude[bot]** 结论：未发现 bug，但鉴于新 kernel 模板特化、gfx950 dispatch、cudagraph 捕获安全与混合选择策略，建议人工 review 后再合入——与本报告的独立分析一致。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK**

核心代码经逐项语义验证未发现正确性缺陷（AITER API 签名、seq_lens 形状语义、prefill 本地化、split/merge capture 稳定性、custom op 注册均核实无误），作者披露的验证矩阵（84/84 正确性、57+14 pytest、GSM8K 0.970）也相当扎实。但存在 5 项 ⚠️：新路径在 gfx950 之外的 CI 零覆盖、核心性能数字不可复现（含作者承认的 parent A/B 方法学缺陷）、PIECEWISE 图 dispatch 烘焙偏差、C++ 分支激活面宽于验证面、跨语言魔法常量耦合——建议作者补齐可复现的 benchmark 材料与最终 serving/GSM8K 验证后再标 ready。
