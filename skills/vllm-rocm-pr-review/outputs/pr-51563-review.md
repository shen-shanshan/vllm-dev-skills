# PR #51563: [ROCm] Migrate W4A16 FlyDSL MoE to AITER API

> **Author**: @coderfeli | **State**: OPEN | **Date**: 2026-08-09（更新 2026-09-03）
> **Branch**: `coderfeli:rocm/aiter-fused-moe-api` → `main` | **Labels**: `performance`, `rocm`, `quantization`, `verified`
> **Changes**: +128 -1658 lines across 13 files | **ROCm 相关性**: 完全相关（gfx950 W4A16 FlyDSL MoE → AITER `fused_moe`）
> **Tracking**: [#51541](https://github.com/vllm-project/vllm/issues/51541)

## 1. 动机 (Motivation)

当前 `fused_flydsl_moe.py` 直接调用 AITER 内部接口（`moe_sorting`、`compile_moe_gemm1/2`），由 vLLM 自己管 routing buffer、指针 ABI、JIT cache、两阶段 launch，以及一份重复的 tile JSON。这些 codegen API 不稳定，也会绕开 AITER 端到端的量化/内核选择与 `AITER_CONFIG_FMOE` 调优表。本 PR 把已经 preshuffle 的 INT4 W4A16 权重改走 `rocm_aiter_fused_experts` → `aiter.fused_moe`，让 sorting、kernel dispatch 和 tuned config 留在 AITER 侧。目标模型族是 Kimi K2.5 INT4（E=384、hidden=7168、inter_dim=256/512、topk=8、`--moe-backend flydsl`、gfx950）。PR 正文 Purpose / Test Plan / Test Result 仍是空模板；机制说明主要在关联 issue 里。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `fused_moe/fused_flydsl_moe.py`（删除 ~407 行） | 去掉 vLLM 自管的 FlyDSL 两阶段 launch / JIT cache / 默认 tile 表 |
| `fused_moe/experts/rocm_aiter_moe.py` | `use_int4_w4a16` 与 MXFP4 一样走 `QuantMethod.BLOCK_1X32`（AITER `per_1x32`） |
| `compressed_tensors_moe_w4a16_flydsl.py` | `apply()` 改调 `rocm_aiter_fused_experts`；packed 权重保持 `[E, N, K//2]` 并 `.view(torch.int4 或 uint8)`，不再 flatten 成 1D |
| `fused_moe/configs/*flydsl.json`（8 个删除） | 删除 MI350/MI355 上 E=384、N=256/512 的 vLLM 侧 tile 表 |
| `benchmarks/kernels/benchmark_flydsl_moe_w4a16.py`（删除） | 原 tile 搜索脚本一并移除 |
| `tests/kernels/moe/test_flydsl_moe.py` | 测 `rocm_aiter_fused_experts`；无 `_moe_C.moe_align_block_size` 时用 Python einsum 参考 |

AITER 侧靠 `w1.dtype == dtypes.i4x2` 把 `per_1x32` 判成 a16wi4（bf16 激活 + packed int4），再用 `is_shuffled` 跳过运行时 shuffle。vLLM 的 `_AITER_INT4_DTYPE = getattr(torch, "int4", torch.uint8)` 与 AITER `dtypes.i4x2` 定义一致。权重仍用原 FlyDSL 交错 packing 与 scale permute；`get_inter_dim` 需要 3D packed shape，不再 flatten 是正确配套。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 正确性 | — | 1 | — |
| 性能 | — | 2 | — |
| 兼容性 | — | 1 | — |
| 测试 | — | 1 | — |
| 注释/文档 | — | — | 1 |

未发现可写出具体触发输入的静默数值 🔴。合入前主要缺口是：AITER 版本契约、旧 tile 表被删后的性能对照、以及 gfx950 上的实测。

---

**⚠️【性能】删掉 vLLM 调优 JSON 后，AITER 默认 heuristic 与旧表不一致，且 PR 无任何可溯源数字** `[已验证]`（回归幅度 `[推测]`）

- **问题**: 旧路径对 Kimi K2.5 形状（E=384、N=256/512）在 MI350/MI355 上有按 token 数（1…8192）搜过的 `tile_m/n/k/n2/k2`。例如 N=256、M=1 是 `{16, 64, 512, 256, 256}`。AITER `fused_moe` 在 `q_dtype_w == i4x2` 时走 heuristic：`tile_m` 按 token 分档，**`tile_n = tile_k = 128` 固定**（见 AITER `fused_moe.py` a16wi4 分支）。本 PR 同时删掉 8 份 JSON 和 `benchmark_flydsl_moe_w4a16.py`，正文却没有脚本、ROCm 版本、GPU、TP、tokens/s。`performance` 标签无法核对。
- **影响**: 触发输入：**gfx950 + `--moe-backend flydsl` + Kimi K2.5 INT4，decode M=1 或 prefill M≥2048**。若 `AITER_CONFIG_FMOE` 没有对应 E/N/dtype 行，serving 会落到 heuristic，decode 的 `tile_k` 从 512 变成 128，延迟可能变差。Issue #51541 的「representative prefill/decode benchmarks」未完成。
- **行动**: 作者应当在 PR 里贴 MI355 TP=4/8 的 decode（ISL 短）和 prefill（ISL≥4096）对照（旧 `fused_flydsl_moe` vs 本 PR），并说明 AITER CSV 是否覆盖 E=384、N=256/512、a16wi4。追溯不到的数字不要当作事实。建议 review 时追问是否还需要保留一份 vLLM 侧 override。

**⚠️【兼容性】没有 AITER 版本/能力守卫；旧 AITER 上 `BLOCK_1X32` 可能走错 kernel** `[已验证]`（旧版本是否仍被 vLLM 镜像使用 `[推测]`）

- **问题**: Issue 验收标准要求「older AITER 的 capability guard/fallback」。diff 没有 `aiter` 版本检查，`requirements/rocm.txt` 也不 pin aiter。新路径一律把 INT4 标成 `QuantMethod.BLOCK_1X32`，正确性依赖 AITER `fused_moe` 里 `quant_type == per_1x32 and w1.dtype == dtypes.i4x2` 才选 a16wi4 FlyDSL。没有该分支的旧 AITER 会把同一 `per_1x32` 当成 MXFP4/fp4x2 路径。
- **影响**: 触发输入：**未含 a16wi4 fused_moe 分支的 AITER + gfx950 + 本 PR 的 W4A16 FlyDSL 权重（dtype 为 `torch.int4` 或 fallback `uint8`）**。可能 JIT/运行失败，或静默用错误 GEMM 读 packed int4。#44400 时期的内核就是为了避开当时不完整的 `aiter.fused_moe`。
- **行动**: 作者应当在 `apply()` / `rocm_aiter_fused_experts` 入口检查 AITER 是否支持 a16wi4（例如探测 `dtypes.i4x2` 路径或最小版本），不支持则明确报错并回退 Triton/emulation，不要静默 dispatch。建议 review 时钉死 Docker/nightly 的 aiter tag，并链接对应 AITER PR。

**⚠️【正确性】`apply_router_weight_on_input=True` 且 topk>1 时，新路径会 assert，旧 FlyDSL 路径不会** `[已验证]`

- **问题**: 旧 `fused_flydsl_moe` 把 `doweight_stage1=layer.apply_router_weight_on_input` 直接传给 kernel，没有 topk 限制。`rocm_aiter_fused_experts` 在 `apply_router_weight_on_input` 时要求 `topk_weights.shape[-1] == 1`。Kimi K2.5 是 topk=8。
- **影响**: 触发输入：**W4A16 FlyDSL MoE + `apply_router_weight_on_input=True` + topk=8**。运行时 `AssertionError`，不再出结果。若生产默认该 flag 为 False，则只是契约收紧；diff 和测试都没有写明。
- **行动**: 作者应当在 `CompressedTensorsW4A16FlydslMoEMethod.apply` 里对非法组合给出明确错误（或证明该 method 永远不会开这个 flag）。建议 review 时确认 Kimi INT4 serving 命令是否会打开它。

**⚠️【测试】单测仍只覆盖 gfx950 单卡 kernel，且 PR 未跑 AMD CI；参考容差很松** `[已验证]`

- **问题**: `test_flydsl_moe.py` 仍 `skip` 非 ROCm/非 gfx950；token 扫到 16384，但没有 TP/EP、`expert_map`、`apply_router_weight_on_input`。`allclose(..., atol=0.5, rtol=0.1)` 与旧测试相同。参考实现在没有 `_moe_C.moe_align_block_size` 时改走 Python einsum，仍是未 shuffle 的 GPTQ unpack，不是 AITER 内核孪生。Fork PR 默认不跑全量 CI（`verified` 只跑 pre-commit）；GitHub `mergeable_state=unstable`，本 review 未能拉到 AMD queue 的绿/红记录。
- **影响**: 触发输入：**合入后首次在 MI355 上跑 Kimi K2.5 INT4 serving**。dtype view、3D layout、`is_shuffled`、AITER heuristic tile 组合都不会被 CUDA CI 碰到。容差 0.5/0.1 很难抓住 scale 排布或错误 kernel 选择。
- **行动**: 作者应当在描述中写明必须在 gfx950 上跑 `tests/kernels/moe/test_flydsl_moe.py` 和一条 e2e（#44400 的 gsm8k 命令即可）；请 maintainer `/ci run` 后再合。建议补 TP=4/8 的 inter_dim 切分（256 vs 512）以及 `is_shuffled=True` 的断言。

**📝【注释/文档】`BLOCK_1X32` 注释仍写「fp4x2」，且 `AiterExperts` 的量化白名单没有 INT4** `[已验证]`

- **问题**: `QuantMethod.BLOCK_1X32` 注释仍是 `fp4x2`；`AiterExperts._supports_quant_scheme` 只有 FP8 / MXFP4。本 PR 从 quant method 直调 `rocm_aiter_fused_experts`，不经过 modular oracle，所以功能上能通，但后续若有人把 W4A16 改走 `AiterExperts` 会被静默判不支持。
- **行动**: 建议作者更新注释，说明 INT4 a16wi4 也走 `per_1x32`、靠 weight dtype 区分；若 INT4 仍走旁路，在 `CompressedTensorsW4A16FlydslMoEMethod` 上留一句，避免以后「统一走 AiterExperts」时漏掉。

## 4. 现有讨论 (Existing Discussion)

没有实质代码 review。`claude[bot]` 因 fork 关闭自动审查。#51541 列出的验收项（端到端 fused_moe、int4/bf16 仍正确、AITER 调优表生效、prefill/decode 数字、旧 AITER fallback）在 PR 勾选区全部未勾。github-actions 说明 fork PR 默认不跑全量 CI，需 write 权限 `/ci run`。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 方向正确：去掉 vLLM 对 FlyDSL codegen 的重复封装、把 packed INT4 保持成 AITER 需要的 3D `i4x2` 布局。但这是 Tier-1 `rocm_aiter_moe.py` 上的量化 dispatch 变更，PR 没有测试计划/结果、没有性能对照、没有 AITER 版本守卫，还删掉了唯一一份 gfx950 tile 表。建议补齐 #51541 验收项并在 MI355 上跑过 kernel + e2e 后再标 ready。
