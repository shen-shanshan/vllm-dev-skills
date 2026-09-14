# PR #54894: [ROCm][DSV4][Perf] Use FP8 WO_A output projection

> **Author**: @LiuYinfeng01 | **State**: OPEN | **Date**: 2026-09-02（最近更新 2026-09-12）
> **Branch**: `LiuYinfeng01:rocm-dsv4-fp8-woa-mxscale` → `main` | **Labels**: rocm, deepseek, verified, DSv4
> **Changes**: +163 -11 lines across 2 files | **ROCm 相关性**: 完全相关（DSV4 ROCm 专属路径 + aiter 算子）
> **CI**: pre-commit ✅ / DCO ✅ / pre-run-check ✅+❌（各一条，疑为旧 run）/ 无 AMD 硬件队列
> **依赖**: AITER >= 0.1.20（#52826，已于 2026-09-03 合并）

## 1. 动机 (Motivation)

DeepSeek V4 的注意力输出 `o` 保存在 RoPE 旋转后的空间中，输出投影为两级低秩结构 `wo_a` → `wo_b`，做 `wo_a` 前必须先逆 RoPE。ROCm 现有路径（#45103）将逆 RoPE 与 **BF16** 分组 einsum 融合为 `rocm_inv_rope_einsum`，但 checkpoint 原生以 FP8 E4M3 存储 `wo_a` 权重（配合 E8M0 block scale）——BF16 路径需要先转换权重、GEMM 内存流量翻倍，且逆 RoPE 后的 BF16 激活必须物化一次再量化。本 PR 用 AITER >= 0.1.20 新增的两个算子把这条链换成全 FP8：`inverse_rope_group_quant`（逆 RoPE + per-token 128 分组 FP8 量化融合）→ `batched_gemm_a8w8_mxscale`（MX-scale 批量 A8W8 GEMM，直接消费原生 FP8 权重）。实测 100K prefill TTFT -7.35%（TP1/PP8）、TP1/PP8 并发吞吐 +6.1~6.8%、TP8/PP1 +0.4~3.2%，GSM8K 全量 0.9651 不低于 BF16 基线 0.9621。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/models/deepseek_v4/amd/rocm.py` (+119 -11) | 新增 `_wo_a_block_scale_to_e8m0()`：scale 张量无损转 E8M0 指数字节（E8M0/uint8 透传；浮点仅接受正 2 的幂，log2+127 bias；否则 None）；新增 `_prepare_fp8_wo_a()`：gfx950 + `VLLM_ROCM_USE_AITER_FP8BMM` 门控下做 import 探测、权重/scale 形状与 dtype 校验、E8M0 转换、cos/sin cache 缓存；`_o_proj()` 增加 FP8/BF16 双分支，BF16 原路径完整保留为回退 |
| `tests/models/test_deepseek_v4_rocm_wo_a.py`（新增 +44） | `_wo_a_block_scale_to_e8m0` 的 CPU 单测：0.5/1/2/4 → 126/127/128/129 编码、E8M0 字节透传、非法输入（0、负数、非 2 的幂、inf、int32）拒绝 |

门控逻辑：`_ON_GFX950 and envs.VLLM_ROCM_USE_AITER_FP8BMM`（复用既有 gate，**默认 True**）→ import 探测（失败则 `warning_once` + 回退）→ 权重 dtype 必须 fp8_e4m3fn/fp8_e4m3fnuz、2D、`out = groups×o_lora_rank`、o_lora_rank 与 in_features 均 128 整除、scale 形状 `(out//128, in//128)` → scale 转 E8M0（失败回退）→ 全部通过才启用。任何一步失败都静默保留 BF16 路径。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 测试 | — | 1 | 2 |
| 可维护性 | — | 1 | — |
| 正确性 | — | — | 1 |
| 兼容性 | — | — | 1 |

---

**⚠️【测试】ROCm 硬件 CI 零覆盖，合入后 AMD 路径无回归兜底** `[已验证]`

- **问题**: PR 带 `verified` label（仅跑 pre-commit，不触发其他测试），head commit 的 CI 只有 pre-commit / DCO / pre-run-check，无任何 AMD 硬件队列 run；而 diff 全部位于 ROCm 专属路径（`deepseek_v4/amd/rocm.py` + 两个 aiter 算子调用）。新增单测只覆盖 scale 转换纯函数，FP8 快速路径本身的组合正确性无自动化测试。
- **影响**: 作者手工验证非常充分（8×MI355X、两个 checkpoint、A/B 同镜像同 commit、1,110 请求×2 arm 无错误），但合入后任何回归（aiter 升级、上游 shape 变化）都会先由 AMD 用户在生产环境发现。CUDA CI 全绿不等于 ROCm 路径正确。
- **行动**: 建议作者在 PR 上请求 AMD CI（rocm 相关 label/队列）至少跑一次 `tests/models/test_deepseek_v4_rocm_wo_a.py` + DSV4 smoke test，或由 maintainer 在 MI355X 上确认后合入。

**⚠️【可维护性】校验失败静默回退，无任何诊断信息** `[已验证]`

- **问题**: `_prepare_fp8_wo_a()` 中除 ImportError 有 `warning_once` 外，权重 dtype 不符、形状校验失败、scale 转 E8M0 失败（如 FP32 scale 非 2 的幂）等所有 return 均无日志。`VLLM_ROCM_USE_AITER_FP8BMM` 默认 True，用户开箱即期望 FP8 路径生效，但无从得知实际是否命中。
- **影响**: 用户/支持工程师无法区分「gate 未开」「checkpoint scale 布局不满足」「AITER 版本不足」三种情况，性能排查只能靠猜；静默的后端切换（FP8 → BF16）与「dispatch 无诊断」的反模式一致。
- **行动**: 建议作者在每个校验失败 return 处加一条 `logger.debug`（或一次性 `warning_once`）说明回退原因，例如 `"DSV4 FP8 WO_A disabled: weight_scale_inv is not power-of-two FP32"`。

**📝【测试】CUDA graph 捕获/回放兼容性未说明** `[推测]`

- **问题**: V1 decode 默认走 CUDA graph 捕获，`_o_proj` 的 FP8 分支在图捕获区域内调用两个 aiter 算子；若算子内部按调用分配 workspace，捕获会失败或回放读到旧地址。另外 `positions.to(torch.int64)` 在 positions 为 int32 时会每次分配新 tensor（同样破坏图回放）。PR 未说明测试环境的 `enforce_eager` 设置与 graph 模式验证情况。
- **影响**: 作者实测 1,110 请求/arm + GSM8K 全量生成均无错误，说明其环境下图捕获通过或未启用；但换 aiter 版本/拓扑后可能突然捕获失败，属于潜伏风险而非当前 bug。
- **行动**: 建议 review 时追问作者：benchmark 是否启用 CUDA graph？两个 aiter op 在图捕获下是否验证过？并在 PR 描述中记录结论。

**📝【测试】CI 有一条 pre-run-check 失败记录 + 需 rebase 以拿到 AITER pin** `[已验证]`

- **问题**: head commit 上 pre-run-check 同时存在 success 与 failure 两条记录；PR 基于 09-02 之前的 main（`rebaseable: false`），而依赖 #52826（AITER bump 至 0.1.21.post1）于 09-03 才合并——当前分支树尚未包含该 pin。
- **影响**: 不 rebase 直接合入时，requirements 中的 aiter pin 与快速路径所需版本可能错位；CI 失败记录需确认为 flake。
- **行动**: 建议作者 rebase 到最新 main（自然带入 #52826 的 pin）并重跑 CI，确认 pre-run-check 失败是否复现。

**📝【正确性】`weight_scale_inv` 的 E8M0 字节透传依赖 checkpoint 约定，无注释说明** `[已验证]`（风险为 `[推测]`）

- **问题**: `_wo_a_block_scale_to_e8m0` 对 `float8_e8m0fnu`/`uint8` 输入直接透传原始字节，隐含假设 checkpoint 的 `weight_scale_inv` 已是 MX 规范的 E8M0 inverse scale，且与 `batched_gemm_a8w8_mxscale` 期望的 scale 语义一致（inverse vs 直接 scale）。这一关键假设在代码和 PR 描述中均未明确写出。
- **影响**: 当前两个 checkpoint（0813 与 V4-Pro 旧快照）GSM8K 均 ≥ BF16 基线，说明约定成立；但若未来 checkpoint/转换工具改变 scale 语义，路径会静默产生精度下降而非报错——静默降精度比崩溃更难发现。
- **行动**: 建议作者在 `_prepare_fp8_wo_a` 处加一行注释说明该假设（"checkpoint stores MX-format inverse block scales as E8M0; verified on 0813 and V4-Pro checkpoints via GSM8K"）。

**📝【兼容性】能力探测仅验证 import，不校验算子签名/行为** `[已验证]`（影响 `[推测]`）

- **问题**: `_prepare_fp8_wo_a` 的探测只确认两个 aiter 模块可 import；算子签名变化（kwarg 改名/位置调整）要到运行时 `_o_proj` 才暴露 TypeError。当前由 #52826 pin 住 AITER 0.1.21.post1，风险可控。
- **影响**: 未来 aiter 升级若改变 `batched_gemm_a8w8_mxscale`/`inverse_rope_group_quant` 接口，用户会在推理中段崩溃而非启动时回退。
- **行动**: 建议作者在探测处顺带记录 aiter 版本（`logger.info`），或接受现状但在 PR 描述中注明"接口变更需同步更新本路径"。

---

**正面确认**（对照性能证据规范逐项核过）：

- 性能数字全部带单位、可复现元数据完整（镜像 digest、vLLM/AITER/FlyDSL commit、拓扑、A/B gate 设置），两个 checkpoint 交叉验证方向一致；
- 测试覆盖真实生产 shape（100K prefill、8K chunk、并发 1–48、TP8/PP1 与 TP1/PP8 双拓扑、GSM8K 全量 1319），无 toy shape；
- 回退设计保守：scale 转换严格无损（非 2 的幂直接拒绝），gate 有 kill-switch（`=0`），成本摊销正确（探测/转换/cache 全部在权重加载期一次性完成，不进 hot path）；
- 无新 env var、无无关文件、无 copy-paste 孪生代码、fnuz 判断基于 dtype 而非 arch（C1 达标）、`positions` 显式 int64（D9 达标）。

## 4. 现有讨论 (Existing Discussion)

- **@Fangzhou-Ai**（Collaborator，09-02）：要求补跑全量 GSM8K 1319 题（当时 100 题 gate 93/100 看似低于基线），期望全量接近 0.96。作者 09-11 回应：在旧 V4-Pro checkpoint 上全量结果 FP8 0.9651 vs BF16 0.9621（flexible-extract），达标且高于基线，并将 PR 描述扩展出完整重测章节（含两个 checkpoint 的并排数据）。回应质量高。
- 其余：claude[bot] 提示 fork PR 需 maintainer 手动触发 review；5 位请求 reviewer 暂无结论。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**（无阻断性问题）

本 PR 是一个设计保守、验证极其充分的高质量 ROCm 性能优化：无损 scale 转换 + 多层校验 + 完整 BF16 回退保证正确性，双 checkpoint/双拓扑/同镜像 A/B 数据链完整，依赖 #52826 已合并。未发现任何 🔴 级正确性缺陷；待改进项集中在**可观测性**（校验失败回退无日志）与 **CI 覆盖**（AMD 硬件队列零覆盖、需 rebase 重跑）——前者建议补日志后合入，后者建议由 maintainer 在 MI355X 上确认或触发 AMD CI 后合入。
