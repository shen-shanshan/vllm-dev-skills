# PR #56429: Revert "[Rocm][Kimi-k3] Add pipeline_parallel support for the kimik3 DCP mode"

> **Author**: @shen-shanshan | **State**: OPEN (DRAFT) | **Date**: 2026-09-11
> **Branch**: `shen-shanshan:revert` → `main` | **Labels**: kimi, k3
> **Changes**: +0 -12 lines across 2 files | **ROCm 相关性**: 完全相关（Kimi-K3 ROCm DCP + PP 路径）
> **Revert 目标**: PR #53664（合并于 2026-09-09，squash commit `d8d53f17`，+12/-0）

## 1. 动机 (Motivation)

PR 描述仅有一句 "Revert #53664"。根据作者口述，revert 的原因是：**有反馈指出 #53664 的改动是冗余的——在 vLLM 中 `MLA` decode 路径的 `dcp_local_seq_lens` 不可能为 None**。本报告对该说法做了逐路径验证（见第 3 节）：**结论成立**。vLLM 的两条 runner 路径（monolithic 与 modular/MRV2）在 DCP 启用时都会在 attention 元数据到达 MLA 层之前填充 `dcp_local_seq_lens`，因此 #53664 添加的 fallback 推导与 assert 在 vLLM 自身代码路径中是不可达的死代码，revert 在技术上是安全且合理的。revert 与 #53664 逐行镜像（已对照 `d8d53f17` 验证：同 2 个文件、+12/-0 ↔ -12/+0）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/model_executor/layers/attention/mla_attention.py` (-10) | 删除 decode 路径（`build()`，L2547-2556）中的 `assert seq_lens is not None` 与 `dcp_local_seq_lens is None` 时的 `get_dcp_local_seq_lens()` 回退推导；恢复为直接 `seq_lens = dcp_local_seq_lens` |
| `tests/models/test_registry.py` (-2) | 从 `test_registry_is_pp` 参数列表删除 `("KimiLinearForCausalLM", True, False)` 与 `("KimiK3ForConditionalGeneration", True, False)` |

无新增代码、无 API 变更、无性能声明。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 正确性 | — | — | 1 |
| 文档 | — | 1 | — |
| 测试 | — | 1 | — |
| 流程 | — | — | 1 |

---

### 冗余性验证（本报告核心，非 finding）

对 "`dcp_local_seq_lens` 在 MLA decode 路径不可能为 None" 的逐路径验证（基于 main 分支稀疏 clone 全量 grep + 代码阅读）：

1. **字段默认值**：`CommonAttentionMetadata.dcp_local_seq_lens: torch.Tensor | None = None`（`vllm/v1/attention/backend.py:417`）——默认确实为 None，关键在于到达 MLA 层之前是否被填充。
2. **Monolithic runner**（默认 V1 路径）：`gpu_model_runner.py:2456-2466` 在 `self.dcp_world_size > 1`（= `parallel_config.decode_context_parallel_size`，L548）时无条件计算 `get_dcp_local_seq_lens(...)` 并赋值 `cm_base.dcp_local_seq_lens`。`git log -S` 显示该填充块**至少自 2026-07-31 就存在**——早于 #53664 的创建时间（08-25）。
3. **Modular runner**（MRV2）：`model_runner.py:1728-1740` 在 attention 元数据准备前调用 `maybe_prepare_dcp_local_seq_lens(...)`（`cp_utils.py:8-35`），该函数**仅当 `dcp_size == 1` 时返回 None**；且调用点在 PCP 批分区**之后**（#55212 于 09-09 专门修复了"分区后才初始化 DCP 元数据"的顺序问题）。
4. **一致性**：MLA 层的 `self.dcp_world_size = get_dcp_group().world_size`（`mla_attention.py:2329`）与 runner 的 `decode_context_parallel_size` 同源，实际运行中两者同 >1 或同 =1，不存在"层认为 DCP 开、runner 认为 DCP 关"的错位。
5. **其他 `=None` 构造点**逐一排除：`pcp_manager.py:551`（PCP 本地批，随后被 L1728 覆盖）、`input_batch.py:193`（cudagraph dummy 批工厂，真实 attention 路径前被覆盖或跳过 attention）、`model_runner.py:1404`（InputBatch 初始构造，后被覆盖）。spec-decode 路径（`speculator.py`、`llm_base_proposer.py`）均为透传已填充的元数据。
6. **`seq_lens` 同理**：两条 runner 都无条件设置 `cm_base.seq_lens`，被删除的 assert 同样是不可达的。

**验证结论**：在 vLLM 自身代码路径中，#53664 的 fallback 与 assert 均为死代码，revert 不改变任何可达路径的行为。

---

**⚠️【文档】revert 动机与验证信息缺失——PR 描述必须写入冗余性论证** `[已验证]`

- **问题**: PR body 仅一句 "Revert #53664"；Test Plan / Test Result 两节为空但模板 checklist 勾选 `[x]`。上述冗余性验证（或等价论证）完全没有出现在 PR 描述中。
- **影响**: 对刚合入两天的崩溃修复做无解释 revert，maintainer 无法判断动机；没有证据链的 revert 描述很可能被要求补充后重提。尤其 #53664 的描述自述其修复了 PP×DCP decode 的元数据缺失，直接 revert 会被质疑"重新引入崩溃"。
- **行动**: 作者应当把冗余性论证写进 PR 描述：核心论点是两条 runner 路径在 `dcp_world_size > 1` 时均无条件填充 `dcp_local_seq_lens`（`gpu_model_runner.py:2456-2466` 自 07-31 已存在；`model_runner.py:1728` + `cp_utils.py:8-35` 于 09-09 由 #55212 引入），并说明 Test Plan（如：revert 后跑 registry 测试 + 说明行为等价性）。

**⚠️【测试】删除 registry 条目使 Kimi 模型 `supports_pp` 失去测试覆盖** `[已验证]`

- **问题**: `test_registry.py` 删除两个 Kimi 条目后，`supports_pp=True` 的声明不再有测试覆盖；kimi_k3 模型包（外部 `vllm.models.kimi_k3`，本 PR 无法改动）的 `supports_pp` 声明不受影响。
- **影响**: 测试条目的删除并非"冗余代码清理"的一部分——它们断言的是包侧声明的能力。若包侧仍宣称支持 PP，移除覆盖会留下一个无回归保护的声明。
- **行动**: 作者应当考虑只 revert `mla_attention.py` 的 10 行、保留两个测试条目（测试仍会通过，且保住覆盖）；若坚持完整 revert，在描述中说明包侧 `supports_pp` 的现状与计划。

**📝【正确性】残余风险仅存在于外部模型包的自定义元数据构造路径** `[推测]`

- **问题**: 本次验证覆盖 vLLM 主仓库全部路径；`vllm.models.kimi_k3` 是外部模型包（不在主仓库、无公开源码），若包侧存在自定义 runner/元数据构造（如 DSpark 模式，registry 中有 `vllm.models.kimi_k3.nvidia.dspark_mla`）不填充该字段，被删除的 fallback 曾是唯一兜底，删除后将从"静默兜底"变为 `TypeError: 'NoneType' object is not subscriptable` 崩溃。
- **影响**: 对 vLLM 自有路径无影响；仅当外部包走非标准元数据路径时才暴露，且崩溃是显式的（易于定位），好过静默错值。
- **行动**: 建议作者在 revert 后于 PP×DCP 配置（如 PP2×TP4×DCP4）跑一次 Kimi-K3 decode 冒烟验证，确认包侧路径同样不依赖该 fallback，并在 PR 描述中记录结果。

**📝【流程】revert 标题引用旧标题；draft 无 CI 记录** `[已验证]`

- **问题**: 标题引用 "#53664 Add pipeline_parallel support"，而 #53664 合并版标题为 "Fix pipeline_parallel support for the kimik3 DCP mode"；PR 为 draft，`mergeable_state: unstable`，无任何 CI 记录。
- **行动**: 建议合并时统一为 `Revert "<合并版标题>" (#53664)` 格式；/ci run 跑 registry 测试（CUDA 队列即可覆盖 `test_registry` 变更）。

## 4. 现有讨论 (Existing Discussion)

无（0 comments / 0 reviews，draft 状态）。旁证：作者是原 PR 的 co-author 且在 09-08 APPROVE 过原 PR；#53664 的最终合并版本（scope cleanup rebase 后仅剩这 12 行）未重新做硬件验证（其描述自述 "The hardware model evaluation was not rerun for this scope-only rebase"），与"改动实际冗余、未经严格推敲"的说法相符。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**（代码改动本身验证安全）

revert 的代码改动经逐路径验证是安全且合理的：被删除的 fallback 与 assert 在 vLLM 自身代码路径中是不可达的死代码（两条 runner 均在 `dcp_world_size > 1` 时无条件填充 `dcp_local_seq_lens`，且填充逻辑在 #53664 合并前已存在）。剩余工作全部在文档与测试层面：PR 描述必须写入冗余性论证与测试计划（当前完全缺失，maintainer 无法判断动机）；建议考虑保留 `test_registry` 的两个条目以维持 `supports_pp` 的覆盖；若坚持完整 revert，补充外部模型包路径的冒烟验证说明。补齐这些后即可 ✅ 合入。
