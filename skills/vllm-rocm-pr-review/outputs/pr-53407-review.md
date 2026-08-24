# PR #53407: [Bugfix][MRV2][ROCm] Dispatch uniform decode to a padded FULL cudagraph

> **Author**: @xiaohuguo2023 | **State**: OPEN | **Date**: 2026-08-22
> **Branch**: `xiaohuguo2023:xiaohuguo/cudagraph-dispatch-pad-uniform-decode-fix` → `vllm-project:main` | **Labels**: `bug`, `rocm`, `nvidia`, `mrv2`, `verified`
> **Changes**: +272 -4 行，涉及 2 个文件 | **ROCm 相关性**: 部分相关（共享 GPU worker 文件 `cudagraph_utils.py`，行为变更由 `is_rocm()` 门控，E2E 数据来自 MI355X）

---

## 1. 动机 (Motivation)

推测解码场景下（EAGLE/MTP，`decode_query_len = 1 + num_speculative_tokens`），`FULL_AND_PIECEWISE` 模式的 FULL decode 图按 `round_up(capture_size, decode_query_len)` stage，而 PIECEWISE 图留在原始 capture 尺寸。当 query length 不能整除 capture ladder（如 qlen=3 vs `[1,2,4,8,16,24]`），两类图的尺寸交错产生空洞：落在空洞中的 uniform-decode batch 找不到精确尺寸的 FULL 图，而 PIECEWISE descriptor（`uniform_token_count=None`）对任何 batch 都兼容，于是 batch 静默降级为 eager attention 执行，每个 decode step 都在 host 关键路径上付出 metadata 构建 + kernel 启动开销。默认配置下 128 个 batch 尺寸中有一半受影响（64 个），问题最初在 MI355X profiling trace 中发现。修复思路：在 `_init_candidates` 中把「可 padding 后容纳 batch 的更大 FULL decode 图」排到 PIECEWISE fallback 之前，`dispatch()` 与 `_is_compatible()` 零改动，不新增任何捕获的图。

## 2. 代码改动总结 (Change Summary)

| 文件 | 改动 |
|------|------|
| `vllm/v1/worker/gpu/cudagraph_utils.py` (+65 -4) | `_init_candidates`：新增 `pad_up_uniform_decode` 门控（separate decode routine + `decode_mode==FULL` + `is_rocm()`）；收集 `decode_full_descs`（`uniform_token_count is not None` 的 FULL decode desc，按 num_tokens 升序）；对每个 `(i, num_active_loras)` staging key，把「每 query length 一个、num_tokens ≥ i、LoRA 匹配」的 FULL desc 前置到候选列表，PIECEWISE 兜底 |
| `tests/v1/cudagraph/test_cudagraph_manager.py` (+207 -0) | 6 个新单测 + 2 个 helper：gap batch 上移回归测试、ROCm 门控钉死（monkeypatch `is_rocm`）、精确匹配不过度 padding、mixed batch 安全性质、超出 ladder 回退 eager、qlen ∈ {1,2,8} 整除 ladder 时 dispatch 不变 |

关键事实（已对 head sha 代码逐行验证）：

- 门控在 `FULL_DECODE_ONLY` 模式下也成立但无害（该模式 fallback 本身就是 FULL 图，pad_up 与其重合被去重）；`varlen_decode=True` 时 decode desc 不设 `uniform_token_count`，被 filter 排除，严格 no-op——与 PR 描述一致。
- CUDA 上零差异**由构造保证**：门控关闭 → `decode_full_descs=[]` → `pad_up=[]` → 候选列表等于原 fallback 列表（同对象同顺序），并有 `test_pad_up_is_rocm_gated` 钉死。
- 去重 `d not in pad_up` 依赖 dataclass 值相等：desc 注册处已有 `if desc not in descs_by_mode[...]` 去重，decode 与 mixed desc 的 `cg_mode` 字段不同，不存在「值相等但非同一对象」的误去重风险。
- 「每 query length 一个候选」的最优性论证成立：uniform batch 恒有 `num_tokens ≥ num_reqs * query_len`（DP token-sync 时取大于号），最小可容纳 FULL 图的请求槽位 `num_tokens // query_len` 自动满足 `num_reqs` 检查。
- `dispatch()`（线性扫描 + `_is_compatible`）与 `_is_compatible()` 在 diff 中零改动，`BatchExecutionDescriptor` 语义未变。

## 3. Review 意见 (Findings)

| 意见类型 | 🔴 | ⚠️ | 📝 |
|---------|----|----|----|
| 设计 | 0 | 1 | 0 |
| 测试 | 0 | 2 | 1 |
| 可维护性 | 0 | 0 | 1 |

**⚠️【设计】ROCm 门控方向与问题范围不匹配** `[已验证]`

- **问题**: `cudagraph_utils.py:304-308` 用 `current_platform.is_rocm()` 门控一个 PR 自己承认「并非平台特有」的 bug——空洞来自 `round_up(capture_size, decode_query_len)`，任何后端都不影响该 staging 逻辑。CUDA 上的 EAGLE/MTP 部署（推测解码最主流的场景恰恰在 NVIDIA）在门控下继续为一半的 decode batch 尺寸付出 eager attention 代价。门控方向也反常：通常只有「后端特定限制」才按平台 gate，这里却是「通用 bug 因只在 MI355X 验证过而 gate」。
- **影响**: 修复的覆盖范围与 bug 的实际范围不匹配；且该分支位于 CUDA/ROCm 共享文件，将来通用化时大概率被重构，产生返工。maintainer LucasWilkinson 已明确要求通用化（见第 4 节），此 PR 以当前形态难以合入。
- **行动**: 建议作者采纳 LucasWilkinson 的通用化方向（去掉 ROCm gate、让 `_is_compatible` 统一处理），或给出 gate 在 CI 数据上的必要性论证。

**⚠️【测试】ROCm 路径行为变更，但无任何 GPU CI 覆盖** `[已验证]`

- **问题**: head commit 的 check-runs 仅有 pre-commit / DCO / pre-run-check 等轻量检查通过，`approved` 检查为 skipped——Buildkite GPU CI（含 AMD 队列）尚未触发（fork PR 待 maintainer 批准）。且 PR 描述自述新增单测跑在 `cpu-small` runner 上（靠 monkeypatch `is_rocm`），即使 CI 触发，committed 测试也不会在 AMD 硬件上执行。diff 中的 ROCm 专属行为分支（pad-up dispatch）在 CI 层面零硬件验证。
- **影响**: 合入后 AMD 推测解码用户首当其冲的行为变更没有 CI 兜底；唯一硬件验证是作者本地的 MI355X 跑测（见下一条，且是 stand-in 配置）。
- **行动**: 合入前必须触发 AMD CI（`rocm` label 队列）并确认 `tests/v1/cudagraph/` 相关 job 结果；建议 review 时追问作者是否已在 MI355X 上直接跑过本 PR 代码（而非仅 stand-in 配置）。

**⚠️【测试】E2E 性能数字来自 stand-in 配置，非本 PR 实际机制** `[已验证]`

- **问题**: PR 描述 E2E 表格的 −82% / −71% ITL 改善是在 capture sizes `{12,36}` 的**配置替身**下测得的（在 ladder 中直接补洞），而非本 PR 的 padding 机制（pad 到 size-18 / size-42）。两者不等价：stand-in 精确匹配、零 padding；本 PR 在 conc-4 场景下 12→18 tokens 意味着 attention 槽位放大 1.5×（4 实 + 2 dummy 请求）。PR 对等价性只有文字论证，没有直接测量。
- **影响**: 实际收益可能略低于 headline 数字（padding 开销未量化）；reviewer 可能高估修复效果。数字本身按 skill 纪律标注 `[unverified]`（作者自报，无脚本输出/日志附件，但配置描述完整）。
- **行动**: 建议作者在 MI355X 上直接跑本 PR 代码补一组 E2E 对比（pre vs post，同 conc 点），或附上 stand-in 与 padding 机制等价性的量化论证。

**📝【测试】differential harness 是一次性脚本，未作为测试提交** `[已验证]`

- **问题**: 覆盖最强的验证——4440 个可达 batch 的 dispatch 差分、105576 个 batch 的候选收窄差分——仅存在于 PR 描述中，未进入 `tests/`。committed 的 6 个单测里没有覆盖 DP 同步 case（`num_tokens > num_reqs * query_len`，即「最小 FULL 图必有足够请求槽位」论证中最需要验证的分支）。
- **影响**: 未来重构 `_init_candidates` 时（maintainer 反提案正是要重构它），这些不变量没有回归保护。
- **行动**: 建议作者把 harness 精简为参数化单测提交（至少补一个 DP 同步 case），或将其作为后续 PR 的测试基础。

**📝【可维护性】AI 辅助开发评估（低风险）** `[已验证]`

- **问题**: PR 带 Claude Code 协助披露，commit 有 `Co-authored-by: Claude` trailer。按 AI 代码诊断流程核查：描述解释机制（WHY）而非只讲动作；性能数字非可疑整数倍；测试非 toy shape；新符号（`decode_full_descs`、`pad_up_uniform_decode` 等）全部在 head sha 代码中存在；无孪生代码分歧；无「为通过而校准」的测试（断言具体 descriptor 而非自身输出）；门控两侧都被 monkeypatch 测试钉死。
- **影响**: 无。作者声称独立定位 bug 并 review/验证了改动，证据链（profiling 定位 + 测试设计深度）与其一致。
- **行动**: 无需行动，仅记录评估结论。

## 4. 现有讨论 (Existing Discussion)

- **LucasWilkinson（collaborator，CHANGES_REQUESTED）**: "I think we should just do this generally (and simplify it)"——并向作者 fork 提交了反提案 PR *「Simplify cudagraph candidate construction」*：去掉 ROCm 专用路径与中间 token/LoRA 候选 map，统一按「FULL 候选优先、PIECEWISE 兜底」构建 dispatch 区间，让 `_is_compatible` 处理 uniform decode / mixed batch / 请求容量 / query length / LoRA 匹配，并把原回归 case 钉为 `[FULL-18, PIECEWISE-16]`。截至报告时作者尚未回应。
- **claude[bot]**: fork PR 自动 review 被禁用，提示 maintainer 可评论 `@claude review` 触发一次性 review。
- 无 issue comments / inline review comments。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK**。核心机制经代码级验证是正确的、验证方法论（差分 + 安全性质）堪称范例，且未发现任何带具体触发输入的正确性缺陷；但 ROCm 门控方向与 maintainer 的通用化要求相悖（已有 CHANGES_REQUESTED），ROCm 路径在 CI 层面零硬件验证、E2E 数字来自 stand-in 配置。在门控设计分歧解决、AMD CI 跑绿、并补上本 PR 机制的直接 E2E 数据之前不建议合入。
