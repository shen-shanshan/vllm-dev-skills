# PR #53598: [ROCm][DSpark][DCP] Serve prefix cache hits under DCP for Kimi-K3

> **Author**: @YukioZzz | **State**: OPEN | **Date**: 2026-08-24
> **Branch**: `YukioZzz:yichaozhu/k3-draft-cache-fix` → `main` | **Labels**: bug, rocm, speculative-decoding, deepseek, kv-connector, nvidia, mrv2, dflash, kimi, k3, scheduler, kv-cache-manager
> **Changes**: +476 -70 lines across 7 files
> **ROCm 相关性**: 部分相关（ROCm/DSpark/Kimi-K3 动机 + rocm label，但改动全部位于 `vllm/v1/core/*` 双后端共享路径，未触及任何 ROCm 专属文件——aiter/mori/rocm_attn/envs/requirements 规则不适用，按共享路径跨后端回归规则审查）

## 1. 动机 (Motivation)

Kimi-K3 的 KV-cache 布局是混合的：full-attention/MLA 层组在 DCP 下被切分到多个 rank，而 Mamba 层组保持复制（`dcp_world_size = 1`）。此前的实现用单一全局 block-size 假设构造所有 cache manager 并做命中查找，导致调度器与缓存管理器在 DCP 下对前缀命中边界产生分歧：Mamba manager 的 block 被错误放大 `dcp` 倍、细粒度本地命中无法跨组对齐、EAGLE 修正后的 Mamba replay checkpoint 未被保留、stale partial-hit 元数据可能让 CoW 指向请求已不拥有的块。本 PR 通过把"每个 group 的有效 DCP 大小"下沉到各 `SingleTypeKVCacheManager`（切分层用进程 DCP size、复制层用 1），统一调度器/管理器对命中边界的认知。该 PR 是四段拆分（#51705 runtime / 本 PR cache 几何 / #53917 offload / #53730 Mooncake）的一环，刻意不含 runtime attention 与外部缓存改动。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/core/kv_cache_utils.py` (+22) | 新增 `dcp_world_size_for_kv_cache_spec()`：仅 `FullAttentionSpec`（含 MLA/RSWA/Sink 子类）返回进程 DCP size，其余 spec 返回 1 |
| `vllm/v1/core/kv_cache_coordinator.py` (+59 -15) | 按 per-group DCP 几何构造 manager；coordinator 的 `block_size`/`dcp_world_size` 改从 manager 读取；`enable_partial_hash_hits` 扩展至 DCP full-attention 组；新增 `_eagle_replay_boundaries()` 并把 `extra_reachable_boundaries` 传给 Mamba 保留逻辑；`find_longest_cache_hit(_per_group)` 改传 per-manager 的 dcp/pcp size；新增 manager 级 `cache_hit_alignment_tokens` 下发 |
| `vllm/v1/core/single_type_kv_cache_manager.py` (+44 -8) | 新增 `cache_hit_alignment_tokens` 属性（协调器可下调至 hash 粒度）；`allocate_new_blocks()` 的 CoW 源从请求当前 block table 刷新，stale 记录越界时跳过并打 debug 日志；`cache_blocks()` 增加 `extra_reachable_boundaries` 参数并在 reachable mask 中使用新对齐 |
| `vllm/v1/core/sched/scheduler.py` (+8) | `_mamba_block_aligned_split()` 新增 `eagle_replay_boundary` 停点：EAGLE 细粒度命中时把 Mamba replay 边界物化为 chunk 结束位置 |
| `tests/v1/core/*`（3 个测试文件，+343 -47） | 新增 per-group DCP 几何、DCP 细粒度命中、EAGLE/Mamba replay 保留（DCP8 参数化）、stale CoW 刷新/丢弃等测试 |

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 正确性 | — | 1 | — |
| 兼容性 | — | 2 | — |
| 测试 | — | 1 | 1 |
| 可维护性 | — | — | 1 |

---

**⚠️【兼容性】`dcp_world_size_for_kv_cache_spec` 使 ChunkedLocal/SWA/EncoderOnly 等 spec 的几何从进程 DCP 翻转为 1，行为变更无任何验证** `[已验证行为变更，影响推测]`

- **问题**: 旧代码给**所有** manager 传进程级 `dcp_world_size`（`SingleTypeKVCacheManager.__init__` 中 `block_size *= dcp_world_size` 对每种 spec 都生效）。新 helper 只对 `FullAttentionSpec` 及其子类（MLA、RSWA、Sink、HiddenStateCache）返回进程 DCP；`ChunkedLocalAttentionSpec`、`SlidingWindowSpec`、`SlidingWindowMLASpec`、`EncoderOnlyAttentionSpec` 等与 Mamba 一起翻转为 1。helper docstring 断言这些 spec "keep replicated per-rank state"，但 PR 中没有任何 DCP>1 下使用这些 spec 的模型/配置验证（测试只覆盖 FullAttention/MLA/Mamba 三种）。
- **影响**: 若存在使用 ChunkedLocal/SWA 且开启 DCP>1 的部署（如 MiniMax M3 类混合模型），其 cache manager 的 `block_size` 会从 `spec.block_size × dcp` 静默缩回 `spec.block_size`，命中边界、块分配量、与调度器 `scheduler_block_size`（LCM 公式仍按 AttentionSpec × dcp 计算）之间的整除关系全部改变——轻则命中率下降，重则 block 几何不一致导致断言失败或错位写入。Mamba 的翻转是本 PR 的核心修复（正确），但同一条规则把其他 spec 也带走了。
- **行动**: 建议作者在 PR 描述中列明哪些 spec 在 DCP>1 下实际被使用及依据（或给出上游关于这些 spec 复制的文档/代码引用），并补一条 SWA/ChunkedLocal + DCP>1 的几何单测；建议 review 时向 nvidia 侧 reviewer 确认 ChunkedLocal 模型（如 MiniMax M3）在 DCP 下的部署形态。

**⚠️【兼容性】`enable_partial_hash_hits` 激活范围扩大至所有 DCP full-attention 组，影响面超出 Kimi-K3/ROCm** `[已验证]`

- **问题**: `kv_cache_coordinator.py:635-646` 新增 `has_dcp_partial_full_attention_group` 条件——任何 `dcp_world_size > 1` 且 `manager.block_size > hash_block_size` 的 FullAttention 组都会开启细粒度命中，随之 `_cache_hit_alignment_tokens` 从 `scheduler_block_size` 下调到 `hash_block_size`，且 `find_longest_cache_hit` 全量改走 per-manager dcp/pcp。这影响所有 DCP + prefix caching 部署（NVIDIA 侧 DeepSeek-V3 类 DCP 部署同样走此路径，PR 也带了 nvidia label）。
- **影响**: 细粒度命中改变了所有受影响模型的命中长度语义（hash 粒度而非 block 粒度）与保留策略；PR 的运行时验证仅覆盖 Kimi-K3 + ROCm（DCP8 + DSpark）单一组合，非 EAGLE、非 Mamba 的纯 DCP full-attention 模型（如 NVIDIA DCP DeepSeek）行为变化未经任何验证。
- **行动**: 建议作者在 PR 描述中明确受影响面（哪些平台/模型在 DCP>1 下会因该条件改变命中行为），并请 nvidia 侧 reviewer（@LucasWilkinson 等）确认 NVIDIA DCP 路径的回归风险；至少补一条无 Mamba 的纯 DCP full-attention 命中测试。

**⚠️【正确性】CoW 跳过分支静默绕过 `_apply_cow` 的安全断言，支撑不变式未声明** `[推测]`

- **问题**: `single_type_kv_cache_manager.py:355-373` 中，当 stale partial-hit 记录的 `block_idx >= len(req_blocks)` 时直接跳过 CoW，只打 debug 日志。`_apply_cow` 内部的共享块写穿保护断言在该分支完全不执行。代码隐含的不变式是"记录越界 ⇒ 请求当前 block table 中不存在需要保护的共享尾块"，但该不变式未以注释或断言形式声明，也无测试证明"越界跳过"与"请求仍持有共享块"两种状态不可能共存。
- **影响**: 若未来某条路径（如 connector 刷新 block table 后重建 `_partial_hit_reqs`）使越界记录与共享块并存，CoW 被静默跳过 → 请求写穿共享 prefix-cache 块 → 污染其他请求的缓存前缀，静默输出错误且无崩溃信号。
- **行动**: 建议作者在跳过分支补一条注释声明不变式（或断言请求当前尾块非共享），并确认 `take_pending_cow_copies` 消费方对跳过场景（无 pending CoW）的行为。

**⚠️【测试】CI 完全未跑：pre-commit skipped、pre-run-check 失败（Mergify 与 #51705 的 head sha 冲突），无 AMD CI，验证全部来自作者本地** `[已验证]`

- **问题**: 该 PR 的 7 个 check 中 pre-commit 为 skipped、pre-run-check 在 "Check PR label and author merge count" 步骤失败（Mergify 因 head commit 与 #51705 冲突无法评估，与代码无关）、无任何 upstream CUDA CI 或 AMD CI 记录。PR 描述的运行时验证（GSM8K 96.66%、hit rate 50.6%、acceptance 2.80）均来自集成分支（#51705 + 本 PR）的本地 DCP8 + DSpark 运行。
- **影响**: 共享核心路径（DCP、前缀缓存、CoW）改动未经任何自动化回归即可能进入 main；尤其 `find_longest_cache_hit` 的 per-manager dcp/pcp 改动会影响所有 hybrid 模型（含 NVIDIA），现有 CUDA 前缀缓存测试套件是否覆盖 DCP 组合未知。
- **行动**: 作者应当在 #51705 落地后 rebase 并触发 `/ci run` 与 `/amd-ci run`，确认 DCP 相关测试在双平台通过；建议 review 时追问 NVIDIA DCP 是否有现成 CI 用例覆盖 hybrid 前缀缓存。

**📝【可维护性】Mooncake 镜像命中路径未同步 per-group dcp/pcp，依赖 #53730 补课** `[已验证]`

- **问题**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` 的 `find_longest_cache_hit` 镜像调用 `manager_cls.find_longest_cache_hit` 时完全不传 `dcp_world_size`/`pcp_world_size`（默认 1），而本 PR 已把本地协调器改为 per-manager 几何。
- **影响**: 本地命中与外部缓存（Mooncake）命中在 DCP full-attention 组上存在几何分歧（该分歧部分早于本 PR 存在）；PR 描述已明确 Mooncake 加固留给 #53730，但 #53730 需要同步这一镜像路径。
- **行动**: 建议作者在 #53730 的描述/实现中显式列出该镜像路径的同步点，避免遗漏。

**📝【测试】验证数字无复现脚本/配置** `[已验证]`

- **问题**: PR 描述的验证数字（GSM8K 96.66%、prefix hit rate 50.6%、mean acceptance 2.80）可追溯到集成分支的服务器日志，但未给出 benchmark/lm-eval 命令、模型 checkpoint、DCP8 配置清单。
- **行动**: 建议作者附上复现用的启动命令与 lm-eval 配置（不必完整日志），便于 reviewer 与后续回归对照。

## 4. 现有讨论 (Existing Discussion)

- **@GirasoleY**（2026-08-26）：要求把 scheduler 中 failed-blocks 处理类改动与 `simple_kv_offload/manager.py` 的 CPU offload 改动拆到独立 PR，并质疑 offload 改动发生在 coordinator 初始化之后是否真能产生预期行为。作者（2026-08-27）已移除两处正交改动，转入 #53917；stale partial-hit CoW 刷新予以保留，理由是它与 `_partial_hit_reqs`/`_apply_cow` 属同一条本地 partial-hit 正确性路径。
- **Mergify bot**（2026-08-25）：head commit sha 与 #51705 冲突，合并规则评估暂停——本 PR 的合并存在对 #51705 的硬性顺序依赖。
- 尚无正式 review 记录（reviews 为空），主线审查待 CI 解除阻塞后进行。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**

PR 的动机清晰、代码组织聚焦，核心几何计算经交叉验证与测试编码一致（调度器停点、协调器保留边界、Mamba retention mask 三处 floor 约定互相咬合），未发现幻觉符号或硬性正确性缺陷。但 `dcp_world_size_for_kv_cache_spec` 对 ChunkedLocal/SWA 等 spec 的几何翻转与 `enable_partial_hash_hits` 的扩大激活均属影响面超出 Kimi-K3 的行为变更，且无任何 CI 或跨平台验证；建议在 #51705 解除 Mergify 阻塞后补齐双平台 CI 与上述 spec 的几何确认再合入。
