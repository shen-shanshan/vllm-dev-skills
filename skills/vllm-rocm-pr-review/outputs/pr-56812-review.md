# PR #56812: [ROCm][Perf] Route untuned fp8 block-scale GEMMs to AITER Triton at large M

> **Author**: @ZhengGong-amd | **State**: OPEN | **Date**: 2026-09-14
> **Branch**: `ZhengGong-amd:aiter-blockscale-triton-route` → `main` | **Labels**: `rocm`, `verified`
> **Changes**: +32 -0 lines across 1 file | **ROCm 相关性**: 完全相关

## 1. 动机 (Motivation)

aiter 的 `gemm_a8w8_blockscale` 按 `(gfx, cu_num, M, N, K)` 精确查 tuned CSV，shape 缺失时不搜索、不回退，直接跑默认 CK 实例——该实例在 prefill 尺寸 M 下无法填满 MFMA 流水线，gfx950 上平坦在 418 TFLOP/s，而 aiter 自带 Triton block-scale kernel 可达 870–1040 TFLOP/s。Qwen3-14B-FP8 的全部四个线性层 shape 在 gfx950/256 上均未 tuned，每层都在付这个代价。本 PR 在 custom op 内部加路由谓词：`M >= 384` 且 `(N,K)` 无 tuned 配置时改走 Triton kernel；分支放 op 内是因为 Dynamo 会把调用方分支冻结在 trace 时的 shape。实测 Qwen3-14B-FP8 端到端 +74%。

## 2. 代码改动总结 (Change Summary)

| 位置 | 改动 |
|------|------|
| `vllm/_aiter_ops.py:861-867` | 新增 `_AITER_BLOCKSCALE_TRITON_MIN_M = 384` 与 `_rocm_aiter_use_triton_blockscale(m, n, k)` 谓词（`m<384` 或非 ROCm 短路；否则 `not is_blockscale_tuned(n, k)`） |
| `vllm/_aiter_ops.py:879-884` | `_rocm_aiter_gemm_a8w8_blockscale_impl` 内按谓词委托给既有的 `_rocm_aiter_triton_gemm_a8w8_blockscale_impl` |
| `vllm/_aiter_ops.py:3312-3326` | `rocm_aiter_ops` 类新增静态方法 `is_blockscale_tuned(n, k)`：复用 `_load_gemm_tuned_configs`，按 `(gfx, cu_num)` 过滤、`(N, K)` 为键查 aiter CSV |

跨文件验证结论：谓词与 aiter 侧约定一致——aiter `gemm_a8w8_blockscale` 中 `n = WQ.shape[0], k = XQ.shape[1]`（B 为 (N,K) 布局），PR 用 `B.shape[0]/B.shape[1]` 作 n/k 正确；两个 impl 签名（A, B, As, Bs, output_dtype）完全匹配，无参数丢弃；`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE` 在 aiter `jit/core.py` 中存在（含 model_configs 合并逻辑），非幻觉符号。

## 3. Review 意见 (Findings)

| 类型 | 数量 |
|------|------|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 4 |
| 📝 建议/备注 | 2 |

**⚠️【性能】tuned 判断忽略 M 维度，部分 tuned 的 shape 在大 M 下仍走 flatline 的默认 CK 实例** `[已验证]`

- **问题**: `is_blockscale_tuned(n, k)` 以 `key_cols=("N","K")` 判断，而 aiter 的 `get_CKGEMM_config(M, N, K)` 按 `(gfx, cu_num, M, N, K)` **精确查行**（仅带 gl∈{0,1} 的 M padding 回退，无法跨数量级命中）。CSV 中 `(N,K)` 有行 ≠ 当前 M 有行。
- **验证数据**: aiter 仓库 `configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv`（gfx950/256）中 `(1024,4096)` 仅在 M=128 有行、`(4096,1280)` 仅在 M=128、`(7168,9216)` 仅在 M∈{65536,131072}。这些 shape 在 prefill M（如 2048）下 aiter 仍命中默认实例（flatline），而谓词返回 True 阻止路由——本 PR 要修复的性能问题在这些 shape 上原样保留。
- **影响**: 漏优化而非回归。DSV3 系模型的这类部分 tuned 层在大 M 下仍付 418 TFLOP/s 的代价。
- **行动**: 建议作者将当前 M 传入谓词（`key_cols=("N","K","M")`，注意与 aiter `get_padded_m` 的 padding 语义对齐），或在 PR 描述中明确这一保守取舍（"任何 M 有行即不路由"）及其代价。

**⚠️【兼容性】gfx942 上 384 阈值错配：短 prefill 区间可能净回归** `[已验证]`

- **问题**: 384 由 gfx950 实测校准，但作者在 MI325X 补测显示 gfx942 上 CK 默认实例并不 flatline（250–300 TFLOP/s），Triton 上限 ~330，且实测 10 个 shape 中 6 个在 M∈[384,1024) 区间 Triton 输给 CK（小 N/大 K shape 要到 M=2048–4096 才反超）。该区间在短 prefill 负载下占比高。此外这是 gfx94x 的 e4m3fnuz 首次进入 Triton block-scale 路径（现有 `is_triton_gemm_w8a8_tuned()` 白名单不含 gfx942）——作者已实测精度无影响（GSM8K strict 0.8650→0.8628，噪声内；op 级 max|Δ| 3.1e-5），故非正确性问题。
- **影响**: gfx942 短 prefill 工作负载性能净回退；e2e 长 prefill 仍净赢（+11.7%/+4.3%），但阈值未按 arch 分化。
- **行动**: 作者应当与 reviewer（@simondanielsson 已倾向 gfx 特定阈值或仅 gfx950 生效）就三个选项定夺后再合入：① 保持现状（e2e 净赢）；② 加 `not current_platform.is_fp8_fnuz()` 锁回 gfx950（放弃 gfx942 收益）；③ gfx942 用 ~2048 的 per-arch 阈值（保留大 M 收益、避开亏损区间）。作者已为三个选项给出完整数据。

**⚠️【测试】AMD CI 未运行；微基准脚本未附** `[已验证]`

- **问题**: head commit 的 12 个 check-runs 全部为通用/CUDA 队列（pre-commit、DCO、format 等），无 AMD/ROCm CI；PR 未带 `amd` label。其中 `pre-run-check` 的一个 failure 是 label-gate 的陈旧 run（PR 现已有 `verified` label，同名另一 run 成功），非代码问题。此外 PR 描述中的微基准脚本标记为 "authored" 但未附脚本/路径，ROCm 版本未标注——微基准数字（+54.2% 等）无法复现；e2e 命令、模型、GPU、TP 配置均完整。
- **影响**: 改动位于 Tier-2 backbone 文件 `_aiter_ops.py` 且为纯 ROCm 路径，CI 零回归防护。作者在 MI355X/MI325X 上的手动验证扎实，但不能替代 CI。
- **行动**: 合入前建议加 `amd` label 触发 AMD CI；建议作者附上微基准脚本或至少标注 ROCm 版本。

**⚠️【可维护性】依赖 aiter 内部 API；CSV 读取失败时静默全量切换** `[已验证]`

- **问题**: `is_blockscale_tuned` 直接 `import aiter.ops.gemm_op_a8w8` 访问内部 `AITER_CONFIGS`、`get_gfx()`、`get_cu_num()`（沿用既有 bpreshuffle 兄弟方法的模式，但仍是内部 API）。更关键的是 `_load_gemm_tuned_configs`（`_aiter_ops.py:191-193`）对任何 CSV 读取/解析异常 `except Exception: return set()`——空集被本 PR 解释为「全部 untuned」→ 所有 M≥384 的 GEMM 静默切到 Triton，无任何日志。此外首次调用会触发 aiter `get_config_file` 的 lazy merge（读全部 per-model CSV、pandas 去重、写 `/tmp/aiter_configs/`），该路径失败同样落入静默空集。
- **影响**: aiter 升级破坏内部符号 → ImportError 崩溃；CSV 缺失/损坏（含 /tmp 合并文件异常）→ dispatch 静默全量切换，无诊断手段。
- **行动**: @simondanielsson 已建议等 #55001（`[ROCm] Refactor tuned gemms`，open，@afriedri）合并后改用 aiter 官方 utils——作者应当确认跟随该重构；至少为空集路径加一条 warning 日志。

**📝【设计】无显式 kill-switch** `[已验证]`

新默认行为改变了 gfx942/gfx950 上所有 untuned 大 M shape 的 kernel 选择，vllm 侧无 env var 可关。间接逃生口存在：`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` env var 可指向自定义 CSV（vllm 谓词与 aiter 读同一 property，行为联动），但过于隐晦。若 reviewer 采纳「锁 gfx950」方案此条自动消解；否则建议作者加临时 kill-switch（验证充分后移除）。

**📝【性能】首次大 M 调用的一次性开销** `[已验证]`

`is_blockscale_tuned` 首次调用触发 aiter 侧 `get_config_file` 的 lazy merge：pandas 读 main CSV（6630 行）+ 全部 per-model CSV、去重、写 `/tmp/aiter_configs/`。该开销落在第一个 M≥384 的 GEMM 调用处（eager 路径），是一次性 latency 尖峰。作者 e2e 中丢弃 leading pair 吸收了此成本；生产环境每次进程启动付一次，aiter 侧有 `lru_cache` + 文件锁，可摊销——仅备注。

## 4. 现有讨论 (Existing Discussion)

- **作者自证对 NVIDIA 零影响**（4 条独立防线）：block-scale 注册表仅 ROCm 条目、`is_supported()` 非 ROCm 拒绝、`register_ops_once()` 非 ROCm 不注册该 op、谓词自身短路。逻辑链完整。
- **作者 MI325X (gfx942) 补测**：精度无影响；e2e +11.7%/+4.3% 净赢；但 384 阈值是 gfx950 校准值，实测 6/10 shape 在 M∈[384,1024) 输给 CK——这是本 review finding 2 的数据来源，作者主动公开了反例数据并给出三选项。
- **@simondanielsson**：整体 "Nice work!"，三条意见——① 等 #55001 合并后改用 aiter utils；② 倾向 gfx 特定阈值或仅 gfx950 生效；③ 建议进一步微基准确定哪些 shape 走 Triton 有利。
- Claude Code bot：fork PR 自动 review 未启用，需 maintainer 手动触发。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 改动小、证据链扎实（微基准对照组 + e2e 交叉配对 + 可证明 no-op 的 control 模型 + 噪声地板标定）、方向正确；但合入前需解决三个开放项：gfx942 阈值定夺（作者已给三选项与数据）、`is_blockscale_tuned` 的 M 维度覆盖缺口（已用 aiter 出厂 CSV 验证存在部分 tuned shape 漏路由）、AMD CI 覆盖（当前零 CI 回归防护）。
