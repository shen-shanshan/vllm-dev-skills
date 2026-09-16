# PR #56812: [ROCm][Perf] Route untuned fp8 block-scale GEMMs to AITER Triton at large M

> **Author**: @ZhengGong-amd | **State**: OPEN | **Date**: 2026-09-14 | **更新**: 2026-09-16（重构 +44 -0）
> **Branch**: `ZhengGong-amd:aiter-blockscale-triton-route` → `main` | **Labels**: `rocm`, `verified`
> **Changes**: +44 -0 lines across 1 file | **ROCm 相关性**: 完全相关

## 1. 动机 (Motivation)

aiter 的 `gemm_a8w8_blockscale` 按 `(gfx, cu_num, M, N, K)` 精确查 tuned CSV，shape 缺失时不搜索、不回退，直接跑默认 CK 实例——该实例在 prefill 尺寸 M 下无法填满 MFMA 流水线，gfx950 上平坦在 418 TFLOP/s，而 aiter 自带 Triton block-scale kernel 可达 870–1040 TFLOP/s。作者统计 27 个本地 fp8 checkpoint 在 TP∈{1,2,4,8} 下产生 115 个 distinct (N,K)，其中 88 个无 tuned 行——「补 config」需 ~3000 行且每卡一跑，解决不了普遍问题，因此需要**通用回退**。实测 Qwen3-14B-FP8 端到端 +74%（重构前实现）。

**自 09-15 初版 review 以来的重构**（回应 simondanielsson / tjtanaa / Fangzhou-Ai 意见）：谓词从「(N,K) 级 CSV 判断」改为**复用 aiter 内部 `get_CKGEMM_config(M,N,K)`**（共享 lru_cache、精确 M 匹配），阈值改为 **per-arch 表**（gfx942: 2048 / gfx950: 384，缺席架构不路由），并已把同一方案移植到 aiter 侧（**ROCm/aiter#5586**）。

## 2. 代码改动总结 (Change Summary)

| 位置 | 改动 |
|------|------|
| `vllm/_aiter_ops.py:864-869` | 新增 per-arch 阈值表 `_AITER_BLOCKSCALE_TRITON_MIN_M = {"gfx942": 2048, "gfx950": 384}` |
| `vllm/_aiter_ops.py:872-877` | 新增 `_rocm_aiter_blockscale_triton_min_m()`：`@functools.cache`；非 ROCm 或 arch 不在表中返回 `None`（fail-safe 不路由） |
| `vllm/_aiter_ops.py:880-895` | 路由谓词 `_rocm_aiter_use_triton_blockscale(m, n, k)`：M 低于阈值短路；否则调用 `aiter_gemm_a8w8_ops.get_CKGEMM_config(m, n, k, csv)`，返回 `None`（untuned）即路由 Triton |
| `vllm/_aiter_ops.py:898-914` | `_rocm_aiter_gemm_a8w8_blockscale_impl` 内按谓词委托给既有的 Triton impl |

跨文件验证（对照 aiter main 源码 `gemm_op_a8w8.py` / `jit/core.py`）：
- `gemm_a8w8_blockscale` 内部用**完全相同参数**调用 `get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)`——缓存键一致，vllm 侧调用为 cache hit，且与 aiter 实际行为**不可能分歧**；
- `get_CKGEMM_config` 对 CSV 读取失败是**抛异常**而非静默空集（相比初版 `_load_gemm_tuned_configs` 的 `except: return set()` 是改进——失败是响亮的）；
- `B.shape[0]/B.shape[1]` 作 n/k 与 aiter 侧 `n = WQ.shape[0], k = XQ.shape[1]` 一致；
- 初版的 `rocm_aiter_ops.is_blockscale_tuned()` 已整体移除，diff 不再触碰 `_load_gemm_tuned_configs`，与 #55001 重构零重叠。

## 3. Review 意见 (Findings)

| 类型 | 数量 | 状态 |
|------|------|------|
| 🔴 必须修复 | 0 | — |
| ⚠️ 建议修复 | 3 | 2 个新问题 + 1 个遗留 |
| 📝 建议/备注 | 3 | — |
| ✅ 已解决 | 3 | 初版 review 的核心问题均被重构解决 |

### 初版 findings 的处置（重构验证）

✅ **【性能】M 维度覆盖缺口 — 已解决**。初版 `is_blockscale_tuned(n,k)` 只按 (N,K) 判断，aiter 按 (M,N,K) 精确查行，导致部分 tuned 的 shape 漏路由（我曾用 ds_v3 CSV 验证 (1024,4096) 仅 M=128 有行）。重构后谓词直接调 `get_CKGEMM_config(M,N,K)`，粒度完全对齐 aiter 自身；作者独立确认 gfx950 上 69 个 tuned (N,K) 中 2 个在任意大 M 无行、13 个部分覆盖——旧代理确实漏掉了这些。

✅ **【兼容性】gfx942 阈值错配 — 已解决**。阈值改为 per-arch 表，gfx942 用实测选出的 2048（长 prefill +14.1%/+7.9%，优于 384 的 +11.7%/+4.9%；消除了 384 在 MoE 短 prefill 上可复现的 −1.1% 回归）。缺席架构一律不路由——「必须先测量才能路由」由构造保证。

✅ **【可维护性】CSV 静默空集 — 已解决**。vllm 侧不再有任何 pandas/CSV 读取；aiter 侧 `get_CKGEMM_config` 读取失败会抛异常（响亮失败），且该函数与 aiter 自身查询同生共死，不存在版本漂移的数据文件语义问题。

### 新增/遗留 findings

**⚠️【注释/文档】PR 描述与代码脱节** `[已验证]`

- **问题**: 代码已重构为 per-arch 阈值 + `get_CKGEMM_config`，但 PR body 仍是初版内容：Purpose 仍写「delegates to the Triton kernel when `M >= 384` and `(N, K)` has no tuned CK config」与「`is_blockscale_tuned()` is the non-preshuffled sibling…」，且 gfx942 章节仍标「untested and not equivalent」——实际已双阈值实测并选定 2048。
- **影响**: 若按当前 body 合入，描述会误导后续读者：`is_blockscale_tuned()` 已不存在、(N,K) 粒度已变为 (M,N,K)、gfx942 已测量。Test Result 中的 +74% 等数字来自重构前实现（作者已声明正在重跑 e2e）。
- **行动**: 作者应当同步 PR body 到重构后实现，并在 e2e 重跑完成后更新 Test Result（gfx942 部分可直接引用重构后测量的 +14.1%/+7.9% 数据）。

**⚠️【可维护性】同一谓词双仓库并存（本 PR vs ROCm/aiter#5586）** `[已验证]`

- **问题**: 作者按 Fangzhou-Ai/simondanielsson 的意向将相同谓词与相同阈值移植进了 aiter 的 `gemm_a8w8_blockscale` 内部（ROCm/aiter#5586，aiter 侧重测 +78.5%）。若两者都合入：vllm 谓词先行触发时 aiter 内部回退对同一 shape 不可达（两处都路由到同一 Triton kernel，结果一致、无正确性问题），但逻辑冗余；且 vllm 的 aiter 版本 pin 更新节奏意味着 aiter 侧回退可能已随新版到达、而 vllm 侧谓词仍存在——两处阈值未来分叉（如 gfx942 再调 1024）将造成行为差异。
- **影响**: 维护负担翻倍 + 潜在阈值分叉。当前 review 共识（simondanielsson：「keep just the aiter PR，let's hear from the others」；作者已表态可随时关闭本 PR）倾向于只留 aiter 侧。
- **行动**: 建议作者等待 @tjtanaa / @AndreasKaratzas 表态后二选一；若保留本 PR 作为 aiter 发版前的 fallback，应在 PR body 中明确标注「aiter#5586 合并后本 PR 应关闭」。

**⚠️【测试】AMD CI 仍未运行；微基准脚本仍未附** `[已验证]`

- **问题**: 新 head commit（bac179c）的 6 个 check-runs 全部为通用队列（pre-commit / pre-run-check / DCO 均 success），无 AMD/ROCm CI（vLLM AMD CI 走 Buildkite，需 `amd` label，PR 只有 `rocm`/`verified`）。旧 commit 上 pre-run-check 的 failure 是 label-gate 陈旧 run，非代码问题。微基准脚本仍标记 "authored" 未附，ROCm 版本未标注。
- **影响**: 纯 ROCm 路径的 backbone 文件（Tier-2）零 CI 回归防护。作者在 MI355X/MI325X 上的手动验证扎实（含零臂、交替 arm、噪声地板标定），但不能替代 CI。若 PR 最终被 aiter#5586 取代，此条自动消解。
- **行动**: 若 PR 保留，合入前建议加 `amd` label 触发 AMD CI；建议作者附上微基准脚本或标注 ROCm 版本。

**📝【性能】非单调交叉点：任何单一阈值都无法保证全 shape 安全** `[已验证]`

作者自己的 gfx942 数据显示 Triton/CK 比值随 M 非单调（如 5120×17408：1.73→1.36→0.98→1.27→1.02→0.90→1.14），归因于 aiter Triton 的 BLOCK_SIZE_M 分桶。当前 2048 阈值下，M=2048 的 5120×17408 实测 Triton 慢 14% 仍会被路由——长 prefill 工作负载因 token-mass 集中在 M≥8192 而净赢（+14.1%/+7.9%），短 prefill 持平，但以该 shape 为主的中 M 负载存在理论回归窗口。作者已明确承认「threshold above which every shape is safe has no solution」，并把 1024 列为下一候选。建议：在 PR body 中记录已知的亏损点（而非仅存在于评论中），供后续调阈值参考。

**📝【设计】无显式 kill-switch（遗留）** `[已验证]`

vllm 侧仍无 env var 开关。间接逃生口在重构后更直接：`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` env var 指向自定义 CSV 时，vllm 谓词与 aiter 读同一 property——把 shape 写进该文件即可让谓词返回 False（不路由）。但此机制过于隐晦，建议作者至少在 PR body 中记录这一逃生路径。

**📝【性能】首次大 M 调用的一次性开销（遗留，金额已测）** `[已验证]`

`get_CKGEMM_config` 首次调用填充其内部缓存（pandas 读合并 CSV、set_index），落在第一个 M≥阈值的 GEMM 调用处。作者剖析：旧版自读 CSV 为 39 ms 一次性 / 59 ns per hit / 谓词占所守护 GEMM（1467 µs）的 0.014%。重构后该一次性成本由 vllm 谓词或 aiter 自身调用两者中先到者支付（同一进程内总成本不变），可摊销——维持备注级别。

## 4. 现有讨论 (Existing Discussion)

- **@Fangzhou-Ai**：能否把改动放进 aiter config 而非 vLLM（补 config 即可）？作者以数据回应：115 个 (N,K) 中 88 个 untuned、~3000 行、每卡一跑——tuning 管已知 shape，本 PR 管「到达时无行的状态」，两者互补。
- **@simondanielsson** 澄清意图：把启发式回退放进 aiter 的 gemm 内部，vLLM 继续盲调。
- **作者移植到 aiter（ROCm/aiter#5586）**：同样谓词与阈值放进 `gemm_a8w8_blockscale`；发现**非 preshuffle 入口的 `libtype` dispatch 只接受 ck/cktile（否则 assert）**——「补 config 指向 Triton」在该路径从来不可用，回退必须是代码路径。aiter 侧重测 +78.5%（vs 本 PR 的 +74.0%），dispatch 边界 M=384 精确、kernel 一致到 ≤1 bf16 ULP。
- **@tjtanaa**：认可 simondanielsson 方案；要求剖析 CSV 读取性能——作者给出 39 ms/59 ns/0.014% 的完整数字，重构后 vllm 侧已无 CSV 读取。
- **共识方向**：simondanielsson 倾向只留 aiter PR；作者保留本 PR 作 fallback、随时可关，等其余 reviewer 表态。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 代码本身经重构后已解决初版 review 的全部实质问题（M 维度对齐、gfx942 阈值、CSV 静默失败），实现方式（复用 aiter 内部缓存查询）干净且与 aiter 行为零分歧，测量方法学（零臂、交替 arm、噪声地板）在同类 PR 中属上乘。剩余工作项是过程性的：① PR body 与代码同步 + 重构后 e2e 数字更新；② maintainer 就「vLLm 侧（本 PR）vs aiter 侧（#5586）二选一」拍板（当前共识倾向仅保留 aiter PR）；③ 若保留则需 AMD CI 覆盖。若归属决策落定为 aiter 侧，本 PR 应直接关闭，其 review 结论即「superseded by ROCm/aiter#5586」。
