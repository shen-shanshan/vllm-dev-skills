# PR #51794: [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4

> **Author**: @shen-shanshan | **State**: OPEN | **Date**: 2026-08-11
> **Branch**: `shen-shanshan:rocm-dsv4-csa-multi-stream` → `vllm-project:main`
> **Labels**: rocm, deepseek, nvidia, DSv4
> **Changes**: +230 -44 across 4 files | **ROCm 相关性**: 完全相关

## 1. 动机 (Motivation)

DeepSeek-V4 CSA（C4A，compress_ratio=4）层在 ROCm 上目前完全串行执行：此前 `amd/model.py` 因 "hang issues" 直接禁用了 aux streams。PR 通过新环境变量 `VLLM_ROCM_DSV4_CSA_MULTI_STREAM`（默认关闭）重新启用 kernel 级多流重叠：在输入 GEMM 之前 fork 三条 HIP 流——默认流跑 fused wqa+wkv GEMM → norm → wq_b → qnorm/RoPE → SWA KV 写入链，两条 aux 流分别跑主 compressor 和 indexer compressor 的 wkv_gate GEMM + 压缩 KV cache 写入链；join 之后 indexer weights GEMM、indexer q 侧、sparse indexer op 和 MLA 在默认流上串行。目的是把两条 compressor GEMM 移出串行关键路径。基准显示各并发度吞吐 +0.82%~+2.09%、TPOT 全降，TTFT 在 conc=16/64 有小幅回归（PR 正文已诚实说明）。

## 2. 代码改动总结 (Change Summary)

| 文件 | 改动 |
|------|------|
| `vllm/envs.py` | 注册 `VLLM_ROCM_DSV4_CSA_MULTI_STREAM`（默认 false，含 env 解析 lambda） |
| `vllm/models/deepseek_v4/amd/model.py` | `aux_stream_list` 由"ROCm 恒 None"改为 env var 门控的 3 条 `torch.cuda.Stream()` |
| `vllm/models/deepseek_v4/amd/rocm.py` | `DeepseekV4ROCMAiterMLAAttention` 新增：`__init__` 中非 CSA 层强制 `aux_stream_list=None`、CSA 层禁用 indexer 嵌套重叠（`indexer.aux_stream=None`）；新增 `_attn_pipeline` / `_prepare_and_attn` override 实现外层 fork/join；新增 `_run_sequential_pipeline` 串行回退 |
| `vllm/models/deepseek_v4/attention.py` | 共享重构：`forward` 主体抽为可 override 的 `_attn_pipeline`；`DeepseekV4Indexer.forward` 拆为 `forward_compressor` / `forward_q`（CUDA 路径语义不变，ROCm 可分别调度）；注释更新 |

关键验证结论（基于 base SHA `9521c60` 全文比对）：
- fork 中两条 aux 链的 `torch.mm(hidden_states, ...fused_wkv_wgate.weight.T, out_dtype=torch.float32)` 与 base `_run_parallel_input_projections` 的 `compressor_kv_score` / `indexer_compressor_kv_score` **公式完全一致**（`fused_wkv_wgate` 为 `quant_config=None` 的普通 bf16 权重），fork 只是把同一计算移到侧流，数值应 bitwise 一致。
- join 语义由 `execute_in_parallel`（`vllm/utils/multi_stream_utils.py`）保证：fan-out event 在默认流上 record、aux 流 wait，aux done events 由默认流 wait 后才返回；后续 `indexer_op`（`skip_k_cache_insert=True`）读 aux1 写的 K cache、MLA 读 aux0 写的压缩 cache，均有 join 兜底。无缺失同步。
- 串行回退路径（`_run_sequential_pipeline` → base `_attn_pipeline`）在 `aux_stream_list` 临时置 None 下运行，且 ROCm 侧 `_run_parallel_input_projections` 的融合 compressor GEMM（`prepare_compressor_gemm_fusion`）在回退路径仍生效；无递归、无 UnboundLocal。

## 3. Review 意见 (Findings)

**意见类型 × 数量**：⚠️ 建议修复 × 4，📝 建议/备注 × 3

---

**⚠️【测试】精度测试未完成，且 PR 无任何 CI 覆盖** `[已验证]`
- **问题**: PR 正文 "Acc Test: To be completed..."；仓库侧 CI 从未运行（fork PR，仅 readthedocs skipped status，无 Buildkite check）。PR 无新增测试文件。
- **影响**: 多流重排虽然数值上等价（同一 GEMM 公式、同一 join 顺序），但 HIP 多流 + hipGraph capture 组合历史上曾在 ROCm 出过问题（见下一条），无精度回归证据 + 无 CI 验证的 PR 合入后 AMD 用户首当其冲。
- **行动**: 作者应当完成 gsm8k 精度测试并把结果补进 PR 正文；合并前需 maintainer 触发 AMD CI（buildkite 的 rocm 构建 / DeepSeek 队列）。

**⚠️【设计】重新启用曾被"hang issues"禁用的 ROCm 多流，但未解释旧 hang 的根因机制** `[推测]`
- **问题**: `amd/model.py` 原注释"Disable them on ROCm because of hang issues"被反转，新代码只说明 WHAT（只做外层重叠、禁用 indexer 嵌套 stream wait），未说明 WHY 旧的实现会 hang、为何 outer-only 就安全（例如：是 hipGraph 多流捕获问题、事件 wait 嵌套问题、还是某个 aiter kernel 的 stream 假设被违反）。
- **影响**: 根因不明就无法界定安全边界——旧 hang 可能在特定 ROCm 版本 / 特定配置（如 gfx942 vs gfx950、特定 hipBLASLt 版本）复现。默认 off 的 env var 降低了风险，但无法排除。
- **行动**: 建议作者在 PR 描述或回复中补充旧 hang 的根因分析和已验证的 ROCm 版本范围（如 ROCm 6.x + MI300X/MI350）。

**⚠️【性能】fork 路径无条件 `enable=True`，没有 CUDA 路径的 token 阈值门控** `[已验证]`
- **问题**: `amd/rocm.py` fork 调用 `execute_in_parallel(..., enable=True)`，而 base CUDA 路径用 `hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`（默认 1024）门控——大 token 数下多流被关闭（大 GEMM 单独即可打满 GPU）。ROCm fork 对任意 token 数都分叉。
- **影响**: 在 FULL cudagraph 捕获 prefill 的场景下，大 token 数的 fused_wqa_wkv GEMM 与两条 aux GEMM 竞争算力/L2，可能退化；PR 的 benchmark（8k1k）只覆盖 decode，未测 prefill。
- **行动**: 建议作者补充 prefill 基准，或对齐 CUDA 路径加上同样的 token 阈值门控（`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`）。

**⚠️【文档/设计】env var 仅在 CUDA graph capture 下生效，PR 描述未说明** `[已验证]`
- **问题**: fork 条件是 `aux_stream_list is not None` 且非（metadata 为 dict 且不在 capture 中）。DSv4 的 `attn_metadata` 是 per-prefix dict，因此：`enforce_eager`（无 cudagraph）与 MRV1 piecewise 模式下所有 decode 都走串行回退；只有 V2 runner + full cudagraph 的 capture 区域才会 fork（replay 阶段由图中捕获的多流执行）。
- **影响**: 用户开启 env var 后若无 cudagraph（如调试场景 `--enforce-eager`），性能无任何变化，且无任何日志提示——"开了没生效"的静默行为。
- **行动**: 建议作者在 PR 描述/env var 说明中注明该限制（依赖 V2 runner + cudagraph capture）。

**📝【可维护性】aux 链的 mm 公式与 base 逐字复制，存在孪生分叉风险** `[已验证]`
- **问题**: `main_compressor_chain` / `indexer_compressor_chain` 复制了 base `_run_parallel_input_projections` 中 `torch.mm(hidden_states, ...weight.T, out_dtype=torch.float32)` 的公式。若未来 base 公式演进（如换成 aiter 融合 GEMM、改 accumulation dtype），fork 路径会悄悄分叉且无任何告警。
- **行动**: 建议抽取公共 helper（如 `_compressor_kv_score(hidden_states, compressor)`）供两处调用。

**📝【性能】benchmark 数字与运行环境不可溯源** `[unverified]`
- **问题**: PR 正文的性能表（tok/s、TTFT、TPOT）无原始 benchmark 输出/日志链接，未给出 GPU 型号（仅 "SA InferenceX"）、TP 配置、ROCm/aiter 版本、benchmark 脚本。
- **行动**: 建议作者附上 benchmark 脚本与完整输出（或 gist 链接），注明 ROCm 版本与 TP 配置。

**📝【性能】开启多流后 ROCm 的融合 compressor GEMM 在该路径失效** `[已验证]`
- **问题**: `prepare_compressor_gemm_fusion` 把主/indexer 两条 compressor 权重融合成单次 `torch.mm`（串行路径的优化）；fork 路径为并行化拆分回两条独立 mm。两者互斥，env var 开启时融合优化对 CSA 层不再生效（回退路径仍生效）。
- **行动**: 无需修改，但建议在代码注释或 PR 描述中记录该取舍，便于后续调优时理解两个优化的关系。

## 4. 现有讨论 (Existing Discussion)

- 无人类 reviewer 的实质性讨论。仅有 3 条 mergify[bot] 的 merge conflict 提示（8/13、8/16、8/19；当前 API 显示 `mergeable: true`，推测已 rebase 解决，合并前建议再确认）。
- @claude[bot] 提示 fork PR 自动 review 未启用，需 maintainer 手动触发。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK** — 核心调度逻辑经全文比对验证是自洽的（fork/join 事件语义正确、数值与串行路径等价、所有回退路径可达），无必须修复的正确性缺陷；但精度测试尚未完成、无 CI 覆盖、旧 hang 根因未解释、env var 生效条件与 benchmark 环境未说明，建议补齐这些证据后再合入。

---

*Review 基于 vllm main `9521c60`（PR base）的全文交叉验证：`vllm/utils/multi_stream_utils.py`、`vllm/models/deepseek_v4/{attention,compressor}.py`、`vllm/models/deepseek_v4/amd/{rocm,model}.py`。*
