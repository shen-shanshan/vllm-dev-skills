# PR #52033: [Perf][ROCm] Dual-stream decode with hipgraphs

> **Author**: @simondanielsson | **State**: OPEN | **Date**: 2026-08-12（最近更新 2026-08-24）
> **Branch**: `simondanielsson:feat/dual-stream-decode-rocm` → `vllm-project:main` | **Labels**: rocm
> **Changes**: +70 -57 lines across 2 files | **ROCm 相关性**: 完全相关（ROCm 专属优化，但改动落在 CUDA/ROCm 共享的 fused_moe runner 层）

## 1. 动机 (Motivation)

原 PR #48223 为 CUDA-like 平台实现了 shared experts 的 dual-stream decode overlap（Fixes #48111），但因 Qwen3.5 在 DP2EP 配置下 gsm8k 精度显著下降被 PR #52024 revert。本 PR 重新开启该功能并修复根因：Qwen3.5 不量化 routed experts 输入，hidden states 与 shared experts 输入是同一 tensor（别名），多流并发下产生竞态；而 DeepSeek-V3 的 fp8 量化会"隐式创建拷贝"打破别名。修复策略是仅当 routed 输入被量化（`moe_quant_config.quant_dtype is not None`）时才允许 multi-stream overlap（仅 ROCm 生效），并将 shared experts 的启动从"routed experts 之后"前移到"dispatch 之前"，同时用 event-based 同步替代 `wait_stream` 以兼容 hipgraph。收益约 3–4% TPOT（DP 场景，8xMI300）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `fused_moe/runner/moe_runner.py` (+29 -18) | 新增 `routed_input_is_quantized` / `is_multistream_safe` 判定并传入 `SharedExperts`；`_forward_impl` 在 `_maybe_dispatch` 之前调用 `maybe_forward_async()`；`_apply_quant_method` 新增 `shared_experts_overlapping` 参数，重叠时仅 `wait()`；删除 `_maybe_sync_shared_experts_stream()` |
| `fused_moe/runner/shared_experts.py` (+41 -39) | 新增每 DBO ubatch 一对的 event（`_input_ready_event`/`_output_ready_event`）；新增 `maybe_forward_async()`（record input event → aux stream wait → 执行 → record output event）与 `wait()`（main stream 等 output event）；删除 `maybe_sync_shared_experts_stream()`/`_run_in_aux_stream()`；`_determine_shared_experts_order` 增加 `is_cuda_alike`、`overlap_is_beneficial`（ROCm 需 dp_size>1）、`_is_multistream_safe` 三重门控；`forward()` 删除 aux 分支只保留串行执行 |

新时序：main stream record `input_ready_event` → aux stream 先跑 shared experts（只等输入就绪）→ main 执行 gate/router/dispatch/routed GEMMs（与 aux 并行）→ main 等 `output_ready_event` → 合并输出。event 在 `__init__` 预分配（图捕获期间不可创建），按 ubatch id 轮转避免 DBO 双 ubatch 互踩。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 正确性 | — | 1 | — |
| 兼容性 | — | 1 | — |
| 测试 | — | 1 | — |
| 设计 | — | — | 2 |
| 可维护性 | — | — | 1 |

---

**⚠️【正确性】aux 读在 `_maybe_dispatch` 之前启动，`is_multistream_safe` 的量化启发式不覆盖 dispatch 阶段的潜在 in-place 写** `[已验证]`（触发场景 `[推测]`）

- **问题**: `moe_runner.py:883` 在 `_maybe_dispatch`（`:901`，DP/EP 的 SP scatter/all-to-all 重分布）**之前**将 shared experts 压入 aux stream；aux 只等 launch 时记录的 input event，此后 main stream 上 dispatch 及 routed 路径对 `shared_experts_input`（仍是 hidden_states 本身，DSv3 亦然）如有任何 in-place 写，与 aux 读之间**没有任何同步**。而 `is_multistream_safe` 只检查 routed 输入是否量化——该检查保护的是 routed GEMM 的输入（量化拷贝），并不能为 shared experts 自己的输入排除 dispatch 阶段的写。作者对根因的解释（"量化隐式创建拷贝 → 两个 buffer 不同"）没有说明 dispatch 阶段对 hidden_states 的写是如何被排除的。
- **影响**: 若某模型/配置的 dispatch 对 hidden_states 有 in-place 写（SP scatter、gate fusion 等），aux 读与之竞态 → 静默错误输出，与上次 revert 同型。DSv3/Qwen3.5 经 gsm8k 验证通过，说明当前主流配置安全，但安全不变式未在代码或注释中成立。
- **行动**: 建议作者在 PR 描述/代码注释中补全因果链——明确 dispatch 路径对 `shared_experts_input` 只读（或写的是量化后的新 buffer），或说明为何量化能间接保护 aux 读；若无法保证，应考虑将安全判定提升为 quant method 的显式能力声明（如 `routed_input_is_copied` 属性）而非用 `quant_dtype` 间接推断。建议 review 时追问这一点。

**⚠️【兼容性】CUDA 侧启动时序与同步机制同样被改变，PR 未提供 CUDA 验证** `[已验证]`

- **问题**: 三重门控中 `not is_rocm()` 使 CUDA 路径行为与 ROCm 相同地切换到新机制：shared experts 由"routed experts 之后启动 + `wait_stream` 同步"变为"dispatch 之前启动 + event 同步"。PR 描述中所有验证（trace、gsm8k、bench）均为 8xMI300 ROCm 环境，无 CUDA 侧数据。
- **影响**: CUDA 上 shared experts 现在与 gate/router/dispatch 并行，启动时序显著提前；CUDA 图捕获下 event 同步虽同为标准可捕获原语，但该共享路径的跨后端回归未被显式验证（vllm main CI 的 CUDA 单测有覆盖，但无针对性确认）。
- **行动**: 建议作者说明 CUDA CI 的覆盖范围（该改动位于 fused_moe runner，CUDA MoE 单测应会触达），或补一组 CUDA 冒烟/精度数据点。

**⚠️【测试】head 上最新重构（event 化）无新的 TPOT 数据，性能表来自 revert 前版本** `[已验证]`

- **问题**: PR 描述中的 TPOT 表（1k/1k、8k/1k 全并发档位）来自原 #48223；本次重新提交的验证只覆盖 Qwen3.5 gsm8k（0.8453/0.8332，与基线相当）+ DSv3/Qwen3.5 trace 截图，DSv3 的性能收益未在当前 head（event 化 + 新门控后）上复测。
- **影响**: event record/wait 开销本身极小，但启动时序变化与新增的 lambda 求值可能影响 TPOT；若存在轻微回退，合并前无法察觉。
- **行动**: 建议作者在最终 head 上重跑一轮 1k/1k 与 8k/1k 的 `vllm bench serve` 数据点（命令与硬件已在描述中给出，可直接复用）。

**📝【设计】`_determine_shared_experts_order` 在同一 forward 内被求值两次，依赖其纯函数性**

- **问题**: `maybe_forward_async`（`_forward_impl:883`）与 `forward()`（经 `_maybe_apply_shared_experts(NO_OVERLAP)`，`_apply_quant_method:610`）各自求值一次判定；两次结果若不一致（未来 `_mk_can_overlap_shared_experts` 或 `_disable_shared_experts_overlap` 变为状态化），会导致 `assert self._output[idx] is None` 崩溃（loud）或共享专家完全未执行（静默丢输出）。当前所有判定均为静态配置 + shape，风险为零，但属脆弱结构。
- **行动**: 建议作者把判定结果随 `shared_experts_overlapping` 一起从 `_forward_impl` 传下，或加注释声明纯函数性假设。

**📝【设计】`quant_dtype is not None` ⇒ "输入被拷贝" 的启发式对 weight-only 量化不成立**

- **问题**: weight-only 类量化（如 w8a16 MoE）同样可能设置 `quant_dtype` 但 kernel 直接读原输入、不创建拷贝——此时判定为"安全"但别名仍在。ROCm 上当前实际量化面（fp8、mxfp4/mxfp8、unquantized）恰好匹配假设（输入量化 ⇒ 拷贝），且 `quant_dtype` 在 MoE 框架中语义上确实指 activation 量化 dtype，故当前无实际触发。
- **行动**: 建议作者在注释中写明该启发式成立的依据（ROCm 上所有设置 quant_dtype 的量化方法均拷贝输入），或改为 quant method 显式声明能力。

**📝【可维护性】移除了 `shared_experts_input is not None` 断言与 `record_stream`**

- **问题**: 旧 `maybe_sync_shared_experts_stream` 中的 `assert shared_experts_input is not None` 被删除；现在若 shared experts 存在而输入为 None，将在 `hidden_states.shape[0]` 处抛 AttributeError，诊断信息变差（仍是 loud failure，非静默）。`record_stream` 的移除本身安全：输入 tensor 由调用方持有至 forward 返回（晚于 `wait()`），且 event 同步保证 aux 读完成后才会释放。
- **行动**: 建议作者在 `maybe_forward_async` 入口保留一个明确的 assert 以便诊断。

---

## 4. 现有讨论 (Existing Discussion)

- **@simondanielsson**（08-20，关键设计说明）：找到安全判定启发式——"routed experts 是否量化其输入"；总结为：DP 模式下量化 routed 输入时默认开启 multi-stream、DSv3 获得真实重叠且无额外拷贝、Qwen3.5 保精度、无新增 env var、hipgraph 安全。此前一天（08-20）曾表示"merge 前正在深入调查 Qwen3.5 的问题"，说明该结论是调查后的产物，非初始设计。
- **@shen-shanshan** `/ci run`（08-24）→ Buildkite #85320；此前 @tjtanaa 也触发过 #84809；作者自己的 `/ci run` 被 CI bot 拒绝（fork PR 作者无 write 权限）。
- **mergify bot** 两次提示 merge conflict（08-12、08-21），当前 `mergeable_state: unstable`，合并前需 rebase。
- **claude[bot]** 自动 review 因 fork PR 被禁用。
- 无 inline review comments；请求 reviewer：@mgoin、@pavanimajety、@zyongye；描述中另 at 了 @dllehr-amd、@AndreasKaratzas 请求 re-review。

**CI 状态**：head commit `b3ed778` 的 Buildkite #85320 正在运行，AMD 队列（mi300/mi355/mi250 的 MoE kernels、basic correctness、language models 等 job）均已启动、非 skipped，CI 覆盖面良好；截至撰写时均 pending，无 AMD 结果可参考。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 核心机制（event 化双流重叠 + 量化启发式门控）设计合理、经验证覆盖 DSv3 与 Qwen3.5 两个关键场景，无必须修复的阻断性问题；但安全不变式的因果链未在代码/注释中完整说明（aux 读与 dispatch 阶段写的关系），且 CUDA 侧行为变化与 head 上最新重构的性能数据存在验证缺口，建议补齐后再合并。
