# PR #53821: [Bugfix][ROCm] Preserve AITER unified-attention metadata during graph replay

> **Author**: @andyluo7 | **State**: OPEN | **Date**: 2026-08-26
> **Branch**: `andyluo7:fix/rocm-aiter-unified-cg-metadata` → `main` | **Labels**: bug, rocm, nvidia, verified
> **Changes**: +59 -3 lines across 2 files（`vllm/v1/attention/backends/rocm_aiter_unified_attn.py` + 测试文件） | **ROCm 相关性**: 完全相关（Tier-1 文件）

## 1. 动机 (Motivation)

`ROCM_AITER_UNIFIED_ATTN` 后端在启用 cudagraph 后输出损坏（MI355X 上 GSM8K exact match = 0，输出 NUL 字符）。根因：通用 `RocmAttentionMetadataBuilder.build_for_cudagraph_capture` 在 capture 时执行 `common_attn_metadata.query_start_loc.zero_()`（#25985 为保护 legacy prefix-prefill kernel 在 capture 期间的非法内存访问而引入），而 aiter unified attention 的 block-first kernel 在 replay 时以 `query_start_loc` 作为查询边界（`cu_seqlens_q`），该 tensor 被清零后查询边界被破坏。本 PR 为该后端单独派生一个 metadata builder：保留 capture 时 `seq_lens = 1` 的廉价行为，但不再清零 `query_start_loc`。PR 描述给出了 MI355X 上修复前后的 GSM8K 对比（0 → 0.9697）与 MiniMax-M3 AgentX 长稳验证，并做了重复工作排查（与 #40003/#53695/#52849/#52628/#51171 互补而非重叠）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `rocm_aiter_unified_attn.py` (+19 -3) | 新增 `RocmAiterUnifiedAttentionMetadataBuilder(RocmAttentionMetadataBuilder)`：`build_for_cudagraph_capture` = `build(0, common)` + `seq_lens.fill_(1)`，**省略** `query_start_loc.zero_()`；`RocmAiterUnifiedAttentionBackend.get_builder_cls()` 改返回新 builder；import `CommonAttentionMetadata` |
| `tests/v1/attention/test_rocm_attention_backends_selection.py` (+40) | 两个单测：① 断言通用 ROCm 后端仍选原 builder、unified 后端选新 builder；② 断言新 builder capture 路径调用 `build(0, common)`、`seq_lens` 被填 1、`query_start_loc` 保持原值 |

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 测试 | — | 2 | — |
| 可维护性 | — | — | 1 |
| 注释/文档 | — | — | 1 |

---

**⚠️【测试】修复针对 replay 期损坏，但两个新增单测均为 mock 级，没有任何自动化测试执行真实的 capture/replay 往返** `[已验证]`

- **问题**: `test_aiter_unified_attention_capture_preserves_query_start_locations` 用 `object.__new__` + `MagicMock` 绕过 `__init__`，并 mock 掉 `build()` 后断言 `query_start_loc is common.query_start_loc`——该恒等关系由测试自己构造（`metadata.query_start_loc = common.query_start_loc`），实现即使完全删除该属性测试也照样通过；真正被测到的只有 `build(0, common)` 的调用签名与 `seq_lens.fill_(1)`（`seq_lens` 是真实 tensor，取值 1<<20/1<<19 刻意放大）。也就是说，**这个 bug 的原始失败模式（graph replay 读出被清零的查询边界）不在任何自动化覆盖内**，全靠 PR 描述的 MI355X 手工评测兜底。
- **影响**: 未来若有人"好心"把 `query_start_loc.zero_()` 加回新 builder（或重构 capture 元数据路径），CI 全绿但 unified attention 的 graph replay 输出重新损坏，且只有 AMD 真机部署才能发现。
- **行动**: 建议作者在 AMD CI 队列中补充一个最小真机用例：以 `ROCM_AITER_UNIFIED_ATTN` + `cudagraph` 跑一个短序列 decode（或对 capture 后的 metadata buffer 断言 `query_start_loc` 内容），替代/补充纯 mock 断言；至少应在 PR 描述中给出 MI355X 评测的脚本与配置（模型 checkpoint、TP、block size），使评测可复现。

**⚠️【测试】AMD CI 已触发但截至 review 时无结果；GitHub check-runs 中 pre-run-check 失败为贡献者门槛而非代码问题** `[已验证]`

- **问题**: head commit（`251d3d1`）的 9 个 check-run 中，`pre-commit`/`DCO` 通过，`pre-run-check` 一个实例成功、另一个在 "Check PR label and author merge count" 步骤失败（作者为社区贡献者，合并数未达标——该 PR 已带 `verified` label，属流程性失败）；AMD CI（Buildkite #85906，tjtanaa 于 08-28 `/ci run` 触发）在 GitHub check-runs 中不可见，PR 描述亦称 focused pytest "pending upstream AMD CI"。
- **影响**: 该修复触及 Tier-1 文件（`rocm_aiter_unified_attn.py`）且其正确性只能由 ROCm 真机验证（CUDA CI 不覆盖此后端）；AMD 队列结果缺失意味着"静默数值错误是否已消除"尚无上游 CI 证据。`.buildkite/test-amd.yaml` 覆盖 `vllm/v1/attention/` 路径，新测试文件会进入 AMD 队列——但需确认执行而非 skip。
- **行动**: 建议 review 时追问 Buildkite #85906 中 AMD 队列是否实际执行了 `tests/v1/attention/test_rocm_attention_backends_selection.py`（该文件 `pytestmark` 为 `is_rocm()` 门控，非 ROCm 队列会被静默 skip）；确认通过后再合入。

**📝【可维护性】新 builder 与基类 capture 逻辑构成孪生实现，存在漂移风险** `[已验证]`

- **问题**: 新 builder 逐行复制了 `RocmAttentionMetadataBuilder.build_for_cudagraph_capture` 的 `build(0, common)` + `seq_lens.fill_(1)` 骨架，仅省略一行 `zero_()`。`rocm_attn.py` 的这段逻辑近期刚被 #51585 改过一次（移除 CPU 侧清零），基类再演化时两处不会同步。
- **影响**: 若基类未来在 capture 路径新增必要处理（如新的 padding/保护逻辑），unified 后端会静默缺课。
- **行动**: 建议作者在新 builder 内以注释显式引用基类实现与 #25985，声明"此处故意省略 zero_() 且省略原因"，并在基类 capture 逻辑变更时由 reviewer 对照此派生类。

**📝【注释/文档】PR 描述的"during replay"损坏机制与代码注释的粒度不匹配，未解释 capture 期为何安全** `[推测]`

- **问题**: 新 builder 注释称 "preserving `query_start_loc`, which unified attention consumes during replay"，但未说明两个关键事实：① 省略 `zero_()` 不会重新引入 #25985 的 capture 期非法内存访问，是因为 unified 后端从不走 legacy prefix-prefill kernel（`use_cascade_attention()` 恒 False，forward 开头 `assert attn_metadata.use_cascade is False`），而 #25985 的保护对象正是那个 kernel；② capture 期 `_dummy_run` 会先 stage 好 dummy batch 的 cumsum 值（`gpu_model_runner.py` dummy-run 路径），unified kernel 在 capture 期读到的是合法边界而非垃圾值。这两个事实是本次省略 zeroing 的安全前提，目前只存在于 PR 描述中。
- **影响**: 未来维护者（尤其不熟悉 #25985 历史的）可能误判该省略为回归，或误以为 capture 期存在 OOB 风险而把 zeroing 加回。
- **行动**: 建议作者把上述两点以注释形式写进新 builder（或 PR 描述中已有的说明迁移为代码注释），并保留对 #25985 的引用。

## 4. 现有讨论 (Existing Discussion)

- tjtanaa 于 08-28 触发 `/ci run`（Buildkite #85906），暂无 reviewer 的实质性 code review 或设计辩论。
- claude[bot] 自动 review 因 PR 来自 fork 被禁用。
- 该 PR 已由作者声明为 Codex 辅助编写（root-cause 定位 + 测试 + 描述起草），提交含 co-author 署名与 DCO。

## 5. 结论 (Verdict)

⚠️ NEEDS WORK

修复本身方向正确、范围克制、与现有 capture 语义兼容：通用 builder 的 zeroing 是 #25985 为 legacy prefix-prefill kernel 引入的保护，unified 后端不经该路径（`use_cascade_attention()` 恒 False），因此为它单独免除 zeroing 是安全且必要的；`rocm_aiter_fa`/MLA builder 均不继承该 zeroing，无 A1 兄弟路径遗漏。主要缺口在验证侧：真实 capture/replay 往返无自动化覆盖（单测为 mock 级）、AMD CI 结果待确认——建议补齐真机回归用例并确认 AMD 队列执行后再合入。
