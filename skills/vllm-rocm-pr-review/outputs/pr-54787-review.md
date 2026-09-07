# PR #54787: [ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path

> **Author**: @benenzhu | **State**: OPEN (2026-09-01) | **Labels**: `rocm`, `verified`
> **Branch**: `benenzhu:m3/01-fused-ar-gemma-rmsnorm-aiter` → `main` | **Changes**: +71 -23 lines across 3 files
> **ROCm 相关性**: 完全相关（AITER fused allreduce+RMSNorm dispatch，ROCm 专属路径）

## 1. 动机 (Motivation)

`fused_allreduce_gemma_rms_norm` 辅助层（当前唯一消费者是 `vllm/models/minimax_m3/{amd,nvidia}/model.py`）此前只有 flashinfer 快路径；ROCm 上永远退化为 `all_reduce` + `GemmaRMSNorm` 两个 kernel，每层 2 次 launch + 一次 all-reduce 结果的显存往返。本 PR 把 AITER 已有的 `rocm_aiter_fused_allreduce_rmsnorm` custom op 接入该辅助层（镜像 flashinfer 分支，`gemma_norm=True`），并把快路径严格 gate 在 AITER **单阶段（1-stage）** kernel 适用范围内——reviewer 实测发现 1-stage 边界之外融合 kernel 反而慢于显式路径。MI355X x4 TP4 实测 decode 每层 ~12.5us → ~7.5us，conc=1 TPOT -1.7%。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py` (+33) | 新增 `_can_use_aiter_fused_ar_rms()` 判定链（custom AR 使能 → 2D 连续 + dtype → 未禁用 → ≤ effective_max_size → should_custom_ar → 1-stage 适用），在 flashinfer 分支后插入 AITER 快路径 |
| `vllm/distributed/device_communicators/aiter_custom_all_reduce.py` (+34) | 新增 `use_1stage_fused_ar_rms()`：镜像 aiter launcher 契约（≤80 tokens、16B pack 对齐且 ≤1024 packs/行、ws==2 无字节上限、全 NVLink 下 ws≤4 <256KiB / ws≤8 <128KiB），capture-static |
| `vllm/_aiter_ops.py` (+4 -23) | `_rocm_aiter_fused_allreduce_rmsnorm_{impl,fake}` 新增 `gemma_norm: bool = False` 参数；impl 内 inline 的 1-stage 判定删除，改调共享方法 |

已做跨文件验证：op 由 `direct_register_custom_op` 按 impl 签名推断 schema，新增默认参数会进入 schema，`gemma_norm=True` 调用合法；aiter main 的 `custom_fused_ar_rms(input, residual_inp, weight, eps, use_1stage, out_hidden_dim=0, gemma_norm=False)` 与调用形式匹配；新方法与旧 inline 谓词逐条件语义等价（旧调用方行为不变）；谓词只依赖 shape/dtype/TP/拓扑，CUDA/HIP Graph 安全。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 兼容性 | — | 1 | — |
| 可维护性 | — | 1 | — |
| 测试 | — | 1 | 2 |
| 正确性 | — | — | 1 |

---

**⚠️【兼容性】新 gate 缺少 aiter 能力探测，旧 aiter 构建下无防护（与既有 fusion pass 做法不一致）** `[已验证]`

- **问题**: `_can_use_aiter_fused_ar_rms()` 未探测 aiter 版本能力，而兄弟路径 `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1598` 对同一 op 明确 gate 了 `supports_dynamic_hidden_dim`（aiter < 0.1.12 时 fused kernel 对 hidden_dim 非 {512,1024,2048,4096} **静默 no-op**，M3 的 H=6144 正属此类）。同理，`gemma_norm` kwarg 在 aiter 中于 2026-06-18 才合入（ROCm/aiter#3792），更早构建的 `custom_fused_ar_rms` 无此参数 → TypeError。vLLM 不 pin aiter（`requirements/rocm.txt` 无 aiter，docker 按 `AITER_BRANCH` 源码构建），用户侧版本自由。
- **影响**: 触发输入 = aiter 早于 2026-06-18 的构建 + `VLLM_ROCM_USE_AITER_CUSTOM_AR=1`（**默认开启**）+ 调用该 helper 的模型 → decode 第一步 TypeError 崩溃；更早（<0.1.12）则静默数值错误。实际暴露度被缓解：MiniMax-M3-MXFP4 的 MXFP4 算子要求更新的 aiter，旧构建在模型加载期就会失败，未必走到这一步。但该 helper 是通用层，未来任何 Gemma 系模型接入即继承此风险。
- **行动**: 作者应当镜像 `supports_dynamic_hidden_dim` / `build_supports_per_group_quant` 的 hasattr 探测风格，在 `AiterCustomAllreduce` 上加 gemma_norm 能力探测并在 `_can_use_aiter_fused_ar_rms` 中要求（不支持时静默走 fallback），或在 PR 描述中注明最低 aiter 版本。

**⚠️【可维护性】1-stage 谓词三个副本只重构了一个，留下孪生分歧风险** `[已验证]`

- **问题**: `_aiter_ops.py` 中同一判定块现存三处：本 PR 重构的 `_rocm_aiter_fused_allreduce_rmsnorm_impl`（改调共享方法），以及未动的 `_rocm_aiter_fused_allreduce_rmsnorm_quant_per_group_impl`（~line 983）与 `_rocm_aiter_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm_impl`（~line 1056）。后两者保留逐字相同的 inline 副本（256K/128K 上限等）。
- **影响**: 未来 aiter launcher 契约变化（字节上限、token 上限）时三处需同步修改；只改一处 → 同一 op 家族的 gemma 路径与 per-group-quant 路径 1-stage 判定分歧 → quant 路径静默走慢的 2-stage 或反之，无报错。
- **行动**: 作者应当把另两处 inline 副本一并切换到 `use_1stage_fused_ar_rms()`（每处约 -20 行），或注释说明保留原因。

**⚠️【测试】ROCm CI 尚未运行，新 dispatch 无 CI 兜底** `[已验证]`

- **问题**: head commit 的 check-runs 仅有 pre-commit / DCO / format / meta 检查，commit statuses 无任何 AMD buildkite 队列记录；PR 带 `verified` label（该 label 只触发 pre-commit，见 `.github/workflows/pre-commit.yml` 的 `pre-run-check` 逻辑）。全部正确性/性能证据来自作者与 reviewer 在 MI355X 上的手动测试。
- **影响**: 该快路径为 ROCm 专属 dispatch（Tier-2 文件 `_aiter_ops.py` + 通用 layer helper），CUDA CI 全绿无法覆盖；若合入后 gfx942（MI300X）或不同 TP 拓扑上有回归，无任何自动化信号。
- **行动**: 建议作者请 maintainer 加 `ready` label 触发 AMD CI，并确认 PR 描述所列的 kernel 测试（`test_minimax_m3_amd_ops.py` 等）在 CI 结果中实际执行。

**📝【测试】PR 描述中 conc=16 的性能数据与最终 gating 不一致** `[已验证]`

- **问题**: 最终代码的 gate 使 H=6144 bf16 时 1-stage 字节上限（256KiB）对应 M≤21；EAGLE3 x3 下 conc=16 每步 M≈64（~786KB），快路径不会生效。作者在评论区亦承认该数据"maybe some unstable measurement error"。conc=1（M=4，~48KB）的 -1.7% TPOT 可归因于快路径。
- **影响**: 描述中的 "conc=16 TPOT -2.9%" 会被读者误解为快路径收益；且快路径的实际生效窗口是低并发（该配置下 conc≲5），PR 描述未说明。
- **行动**: 建议作者更新 PR 描述：标注 conc=16 数据为 fallback 下的波动、明确快路径生效窗口（token≤80 且 bytes<256KiB），并补上已承诺的 conc=32/64 数据。

**📝【测试】TP2 / TP8 拓扑未验证** `[已验证]`

- **问题**: 谓词中 `world_size == 2` 分支无字节上限（镜像旧逻辑），TP8 分支字节上限收紧为 128KiB（M≤10）。全部 parity 与 A/B 测试只在 TP4（MI355X x4）上完成。
- **影响**: TP2 下大 token 数也恒走 1-stage（grid-stride 支持、无正确性问题，但该 size 下 1-stage vs 显式路径的交叉点从未测量）；TP8 的行为未经任何验证。
- **行动**: 建议作者至少补一组 TP2 parity（无字节上限分支）验证，或注明 TP4 之外的拓扑不在本 PR 验证范围内。

**📝【正确性】gsm8k 精度微降未解释归因** `[已验证]`

- **问题**: 描述中 gsm8k 从 0.9704/0.9697（flexible/strict）降至 0.9644/0.9644，降幅 0.5~0.6pp，略超 stderr（±0.005）。而 parity 测试显示 decode 尺寸（1、4 tokens）bit-exact，prefill 尺寸 relerr ≤ 2.7e-3。
- **影响**: 若降幅来自快路径的 bf16 重结合顺序，则"decode 尺寸 bit-exact"的结论与 e2e 精度变化之间存在待解释的张力；若来自 EAGLE3 真实 rejection sampling 的噪声，则应在描述中说明。
- **行动**: 建议作者补充一次对照实验（关闭该快路径、其余配置不变）以确认 gsm8k 差异的归因，或注明为采样波动。

## 4. 现有讨论 (Existing Discussion)

- **@Fangzhou-Ai**（collaborator）：TP4 rank-max A/B 实测发现 1-stage/2-stage 边界处融合路径反慢 4.6%~8.2%，建议把 1-stage 谓词上移至 `AiterCustomAllreduce` 并在调度层 gate——作者采纳并实现（即本 PR 的 `use_1stage_fused_ar_rms`）。
- **@benenzhu**（作者）：在 MI355X TP4 复现交叉点（H=6144 下恰在 256KiB 边界，M=21/22），承认 conc=16 数据可疑并同意 gating；进一步在 `benenzhu/aiter#1` 定位 2-stage 慢的根因（stage-1 push vs pull 语义 + `local_device_load_rmsnorm` 慢 ~1.1us），从根因层面佐证 gating 的必要性。
- **@Fangzhou-Ai** 要求补 conc=64 测试；作者承诺补 conc=32/64（截至抓取时 PR 描述尚未更新）。
- claude[bot] review 因 fork PR 自动禁用，需 maintainer 手动触发；当前无 approve/change-request 记录。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**

设计收敛良好：gate 位置正确（恰在实测交叉点）、判定 capture-static、与 reviewer 的 A/B 驱动迭代质量高，数值 parity 证据充分（decode 尺寸 bit-exact、residual 全尺寸 bit-exact）。合入前建议补齐三件事：aiter 能力探测（对齐 fusion pass 既有做法）、谓词三副本统一到共享方法、加 `ready` label 跑通 AMD CI；并更新 PR 描述中与最终 gating 矛盾的 conc=16 数据。
