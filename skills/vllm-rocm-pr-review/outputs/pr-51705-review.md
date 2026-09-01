# PR #51705: [ROCm][DSpark][DCP] Support decode context parallelism for Kimi-K3 DSpark

> **Author**: @YukioZzz | **State**: OPEN | **Date**: 2026-08-10（最后更新 2026-08-30）
> **Branch**: `YukioZzz/vllm:yichaozhu/k3-dspark-dcp-runtime` → `main` | **Labels**: rocm, speculative-decoding, deepseek, kv-connector, nvidia, mrv2, dflash, kimi, k3, scheduler, kv-cache-manager
> **Changes**: +1062 −154，7 个文件 | **ROCm 相关性**: 完全相关（Tier-1 backbone `rocm_aiter_mla.py` +543 −128）

## 1. 动机 (Motivation)

Kimi-K3 DSpark（因果多 token 目标验证）此前无法在 ROCm 上与 decode context parallelism 组合使用：ROCm 平台强制把 DCP 配置降级为 PIECEWISE CUDA graphs，且多 token 验证路径没有按行因果边界的 KV 视图。本 PR 引入分段（segmented）AITER MLA decode（`skip_reduce=True` 返回每段 partial + max + expsum），由新增的 Triton reduce kernel 归并出 BF16 输出和自然对数 LSE，交给既有的 DCP LSE 合并逻辑；同时把 Triton MLA 的非因果多 token 路径广告为通用 DCP 能力，供 DSpark draft 与目标路径配对。混合 prefix-cache 几何与外部 connector 集成明确留在 #53598 / #53917。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | 核心：`_build_dcp_verify_row_view`（按行因果边界 `committed + arange(1, qlen+1)` 计算每 rank 本地 KV 长度）、`_fill_dcp_verify_page_table`（物理 block → 128-token 子页展开）、FULL graph 持久缓冲（`_verify_row_lens`/`_dcp_verify_block_table`/`_graph_seq_lens`）、`_forward_segmented_dcp_verify`（aiter 分段 kernel + 归并）、单 token DCP decode 走 aiter 原生 `mla_decode_fwd(return_lse=True)`；Gluon/ASM 路径对 DCP 多 token 关闭；`use_gluon_verify` 增加 `dcp_world_size` 门控 |
| `vllm/v1/attention/ops/rocm_aiter_mla_reduce.py`（新） | `reduce_mla_segment_partials`：Triton kernel 把 base-2 分段统计归并为 BF16 输出 + 自然对数 LSE；空行输出 0 / LSE=−inf |
| `vllm/v1/attention/backends/mla/triton_mla.py` | 声明 `supports_non_causal_multi_token_dcp=True`；非因果多 token 组 `supports_dcp_with_varlen=True`（提升 reorder 阈值） |
| `vllm/platforms/rocm.py` | 移除 ROCm 平台级的 DCP→PIECEWISE 强制降级（PCP 降级保留）。经查 DCP 仅限 MLA 模型（`config/model.py` 对非 MLA DCP 有限制），且所有 ROCm MLA 路径都经过本 PR 改造的 builder/impl |
| `tests/v1/attention/*`（3 个文件） | FP8 真实分段 kernel 因果对照测试（DCP=2、qlen=3、96 heads、rank=512、block=128）、按行长度单测、空行/自然对数 LSE 单测、静态 graph 边界单测、Triton 能力声明测试；负对照（旧 committed-only 公式 → 2671/147456 元素超差） |

核心正确性设计（已验证）：aiter v0.1.19 `_mla_decode_fwd_kernel` 的分段公式 `tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS*TILE_SIZE)`、`active = segm_idx*tps*TILE < seq_len` 与 vLLM reduce kernel 的 `active_segments` 掩码**逐项一致**；非活跃段的 max/expsum 槽位是 `torch.empty` 垃圾值，vLLM 侧 masked load 处理；`segm_max`/`segm_expsum` 为 base-2 约定，归并公式 `exp2(segm_max − overall_max)` 与 LSE 自然对数换算正确；`TILE_SIZE = block_size`（即 vLLM 传入的 128-token 子页 view），页表展开 `block_id * pages_per_block + sub` 与物理布局一致。

## 3. Review 意见 (Findings)

| 意见类型 | 🔴 必须修复 | ⚠️ 建议修复 | 📝 建议/备注 |
|---|---|---|---|
| 数量 | 1 | 4 | 4 |

**🔴【正确性/兼容性】DCP8 + DSpark + prefix caching 组合在本 PR 内未修复，存在引擎崩溃复现** `[已验证（近期 head 复现，最终 head 待确认）]`
- **问题**: @jamesETsmith 于 2026-08-26 在 head `cb4922d` 复现：TP8/DCP8、FP8 KV、prefix unit 128、DSpark 双 draft token、AgentX warmup 期间 partial-hit metadata 指向已变更的 KV block，触发 `single_type_kv_cache_manager.py:416-420` 的 `_apply_cow` 安全断言 → 引擎被杀。CoW 代码早于本 PR，但该 PR 新启用的 DSpark+DCP 组合暴露了它；作者回复"retest + cherry-pick #53598"，PR 描述亦写明 prefix-cache 几何留在 #53598，**本 PR 内无修复**。作者最终 head 的 lm-eval（prefix caching 开启、0 server errors）未覆盖 AgentX warmup 的 partial-hit 模式，与复现不冲突。
- **影响**: prefix caching 默认开启；用户合并本 PR 后以 DCP8+DSpark 部署，特定 warmup 模式必现崩溃（非静默）。
- **行动**: 作者应当在合并前用上述配置在最终 head 复测并给出结论；若仍崩溃，应与 #53598 建立显式合并依赖，并在 PR 描述中标注该组合在 #53598 落地前不可用。

**⚠️【兼容性】aiter 版本依赖未 pin、未文档化，能力门控只做 import 探测** `[已验证]`
- **问题**: 验证所用的 nightly 镜像带 aiter v0.1.20 overlay（@andyluo7 明确说明 stock 镜像的 v0.1.19 早于 ROCm/aiter#4412）；而 `_segmented_mla_decode_supported()` 仅探测 `from aiter.ops.triton.attention.mla import mla_decode_fwd` 是否可导入——该模块在 v0.1.19 已存在（我核对了 v0.1.19 源码），因此 stock 镜像上能力门控照样通过。`docker/Dockerfile.rocm_base` 仍 pin `AITER_BRANCH="v0.1.19"`，requirements 无 aiter 版本约束，PR 描述未提及最低 aiter 版本。
- **影响**: 用 stock 镜像部署 DCP8+DSpark 时，96-head 分段路径在未经验证的 aiter 构建上激活（#4412 即 96-head 相关修复），存在静默错误或崩溃风险。
- **行动**: 作者应当与 #52826（aiter v0.1.20 bump）协调合并顺序，并在 PR 描述中写明最低 aiter 版本要求；建议能力探测升级为版本检测而非仅 import 探测。

**⚠️【设计/健壮性】`q_mla` 复用为分段 kernel 的 `out` 指针，依赖 aiter 内部不变式** `[已验证]`
- **问题**: `_forward_segmented_dcp_verify` 将 `q_mla` 同时作为 query 与 `out` 传入 aiter wrapper（`skip_reduce=True` 时不分配输出缓冲）。安全性完全依赖 `NUM_SEGMENTS > 1` 时 kernel 不写 `out` 这一 aiter 内部行为：v0.1.19 的 `select_3d_config` 在 gfx942/gfx950 上 `MIN_SEGMENTS=8` 保证成立，但 **gfx1250 分支无此下限**（短序列时 `MAX_SEGMENTS=1` → `NUM_SEGMENTS=1`），此时 kernel 会把归约结果写回 `q_mla` 再返回张量，vLLM 的 tuple assert 随后才触发。
- **影响**: 当前支持的 MI300/MI355 上安全（fail-closed）；若 aiter 调优参数变更或未来启用 gfx1250，表现为 assert 崩溃（非静默，但崩溃点在数据被覆写之后）。
- **行动**: 建议作者传入显式 dummy 输出缓冲（或调用前断言分段数 > 1），消除对 aiter 内部实现细节的别名依赖。

**⚠️【兼容性】平台级移除 DCP→PIECEWISE 降级覆盖所有 ROCm MLA 后端，graph-safety 工作只覆盖其中两个** `[已验证]`
- **问题**: `platforms/rocm.py` 的降级移除是全局的（GirasoleY 提议、作者同意退役该 flag），但 FULL-graph 静态缓冲改造只落在 `rocm_aiter_mla.py`（本 PR）与 `triton_mla.py`（能力声明）。注册表中还有 `ROCMAiterMLASparseBackend`（DeepSeek V4/Kimi 稀疏 MLA），其 builder 无任何 DCP 相关属性或 graph-safety 处理，也无 DCP 验证；非 MLA 的 GQA-DCP（`model.py` 允许 TP > 总 KV heads 时开启）同样未覆盖。
- **影响**: 此前被强制 PIECEWISE 的未验证组合，现在会直接进入 FULL graph capture，行为未知。
- **行动**: 建议作者确认 sparse/GQA-DCP 组合在 FULL graphs 下的行为（若不支持应在平台层按 backend 收窄放行范围），或至少在 PR 描述中明确限定已验证的 backend 集合。

**⚠️【测试】分段 verify 路径对所有 spec 方法通用激活，仅 DSpark 验证过** `[推测]`
- **问题**: `use_segmented_dcp_verify` 仅按 `max_qo_len > 1 and dcp_world_size > 1` 门控，与方法无关——DeepSeek MTP、Eagle 等 + DCP 的因果多 token 验证会走同一路径；验证只覆盖 Kimi-K3 DSpark（DCP8、interleave 1）。`_validate_dspark_dcp_support` 也只对 DSpark 做配置期校验。
- **影响**: 其他 MLA 模型 + DCP + 多 token 验证的组合未经验证（按行边界数学本身是模型无关的，风险偏低，但不能排除 head 数/dtype 差异触发新形状问题）。
- **行动**: 建议作者说明该路径模型通用性的依据，或补一个其他 MTP 模型的 smoke test。

**📝【性能】分段 partials 缓冲每层每步瞬时分配 ~151–250 MB fp32，不缓存** `[已验证]`
- **问题**: `NUM_SEGMENTS` 由 aiter 按 batch/max_seqlen 调优（batch=32、96 heads 时 8 段：96×96×8×512×4B ≈ 151 MB；FULL-graph 捕获 padding 下更大），wrapper 每次调用 `torch.empty` 新建三个 fp32 缓冲，layer 间复用靠缓存分配器，但每层每 step 均有 allocator churn。
- **影响**: 不影响正确性；Kimi-K3 61 层的瞬时峰值与分配开销值得量化。
- **行动**: 建议作者给出 DCP verify 路径相对无 DCP 的吞吐开销数据，或考虑缓存 partials 缓冲（形状在 FULL 模式下静态）。

**📝【性能】PIECEWISE 模式下 `NUM_SEGMENTS` 随最大行长度变化，引发 Triton 重编译** `[已验证]`
- **问题**: 非 FULL 模式 `max_kv_seq_len = int(row_lens.max().item())`，`NUM_SEGMENTS` 作为 constexpr 随长度桶变化 → 每个新桶触发一次 JIT 编译。
- **影响**: PCP+DCP 等落在 PIECEWISE 的组合存在逐步编译抖动（PCP 仍被强制 PIECEWISE）。
- **行动**: 建议作者说明 PIECEWISE 下 DCP verify 是否属于目标配置；若是，考虑固定段数或量化长度桶。

**📝【测试】因果对照测试使用 `torch.float8_e4m3fn` 而非生产环境的 e4m3fnuz** `[已验证]`
- **问题**: `test_segmented_dcp_verify_matches_causal_attention` 用 OCP e4m3fn 量化参考 KV（max 448），而 ROCm 生产 fp8 KV 为 fnuz（max 240）。分段 kernel 对两者同路径（K_WIDTH=16），归并数学不受影响，故不影响测试结论。
- **行动**: 建议作者补充或切换为 fnuz dtype 用例，贴近生产数值行为。

**📝【杂项】合并冲突未解决；PR 级 AMD CI 状态不可见** `[已验证]`
- mergify 截至 2026-08-30 仍报 merge conflicts；fork PR 未触发 `/ci run`，check 数据不可用。硬件验证依赖作者（MI355X 506 测试、GSM8K 三次跑分）与 @andyluo7（MI355X 独立复验：60 targeted + GSM8K-200 163/200、acceptance 2.365）的手工报告。建议合并前 rebase 并跑一次 AMD 队列。

## 4. 现有讨论 (Existing Discussion)

- **GirasoleY（CHANGES_REQUESTED ×2）** 曾指出"ROCm K3 DCP 验证路径遗漏当前验证块"——即按行因果边界问题；最终 head 以 `committed + arange(1, qlen+1)` 逐行边界修复，并附负对照（旧公式确定性失败 2671/147456 元素、max abs diff 0.1311）。此外要求拆分 PR（Q 复制等无关特性已移除）、质疑 `_DCPA2ABufferPool`（已删除）、建议退役 DCP 降级 flag（已采纳）。
- **jamesETsmith** 报告 DCP8+DSpark+prefix-cache 的 CoW 断言崩溃（见 🔴 意见），作者承诺复测，修复推迟至 #53598。
- **andyluo7** 详细分析了 aiter 依赖链：ROCm/aiter#4412（96-head）首个含修复 tag 为 v0.1.19.post1，stock 镜像 pin v0.1.19 早于它，上游 bump 为 #52826（v0.1.20），建议依赖官方 bump 而非另开 PR。
- **billishyahao** 要求移除 aiter 源码 regex 探测（已改为显式能力门控）、质疑 qo_indptr 覆写与 Triton 文件改动（ASM DCP 路径已删除、Triton 改动收窄为能力声明并保留）。

## 5. 结论 (Verdict)

🔴 **BLOCK**（在 prefix-cache 崩溃问题确认前）。核心注意力机制本身经受了高质量的审查与验证——分段公式与 aiter kernel 的逐项一致性、FULL-graph 静态缓冲设计、负对照测试都扎实；但 DCP8+DSpark+prefix-cache（默认开启配置）的引擎崩溃在本 PR 内未修复且被推迟到 #53598，合并顺序必须在 PR 描述或 reviewer 流程中显式化。若作者在最终 head 复测确认崩溃已消除（或与 #53598 建立合并依赖），其余 ⚠️ 项修复后可降级为 NEEDS WORK → LGTM。

---

*报告生成：vllm-rocm-pr-review skill（2026-08-30）。aiter v0.1.19 内核契约逐项核对：`aiter/ops/triton/attention/mla.py`、`aiter/ops/triton/_triton_kernels/attention/mla.py`、`aiter/mla.py`。*
