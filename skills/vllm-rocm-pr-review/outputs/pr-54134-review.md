# PR #54134: [ROCm][DeepSeek V4] Enable FHMoE with DP8 over RCCL

> **Author**: @LiuYinfeng01 (AMD) | **State**: OPEN | **创建**: 2026-08-28（最近更新 2026-09-11）
> **Branch**: `LiuYinfeng01:rocm-dsv4-dp8-fhmoe` → `vllm-project:main` | **Labels**: `rocm`, `deepseek`, `nvidia`, `DSv4`
> **Changes**: +515 -26 lines across 3 files | **ROCm 相关性**: 完全相关
> **CI**: 仅 7 个 checks；`pre-run-check` **failure**（与 `mergeable_state: unstable`、`rebaseable: false` 的 merge conflicts 一致）；无 AMD 硬件 CI job

---

## 1. 动机 (Motivation)

DeepSeek V4 在 ROCm 上的 MoE 是异构量化结构：shared expert 为 native FP8（E4M3 + E8M0 block-128 scale），routed experts 为 MXFP4。#53161（已合并）首次把 AITER 的 `fhmoe_` 异构融合 kernel 接入 vLLM，但仅支持 TP8/DP1 且走直连路径。本 PR 将同一融合路径推广到 **TP1/DP8**：shared expert 权重/scale 按展平 DP rank 切分，融合 kernel 挂进 modular expert 边界（新增 `DeepseekV4HeterogeneousAiterExperts`），外层保持标准 RCCL all-gather prepare / reduce-scatter finalize 流水线。动机很直接：TP1/DP8 是 8×MI355X 上大 MoE 模型的部署首选（每卡只持 1/8 专家权重），作者实测 decode 中位 TPOT -8.4%、输出吞吐 +8.1%；早先绕过 modular dispatch/combine 的直连 DP 原型因 GSM8K 失败被否决，本 PR 改走 modular 路径是经过正确性教训后的架构决策。PR 描述含完整设计、正确性（GSM8K parity + 方差分析）、性能（3+3 run 中位数）、trace 证据与重测章节，是少见的 WHY 驱动描述。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/models/deepseek_v4/amd/model.py` (+127/-26) | 门控放宽为 `tp*dp==8` 且拒绝组合并行；`_prepare_native_fp8_shared_expert` 增加 `shard_rank/shard_size` 分片；新增 `_shared_expert_shard_rank_and_size`；`DeepseekV4HeterogeneousMxfp4MoEMethod` 在 DP 下切换 `experts_cls` 并注入 shared 权重；`forward_modular` 增加 DP 分支（跨 rank 总 token 数做 CSV 探测，融合与回退均走 `quant_method.apply`） |
| `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` (+134) | 新增 `DeepseekV4HeterogeneousAiterExperts(AiterExperts)`：`configure_shared_expert()` 注入权重；`apply()` 按 route 列数（routed / routed+1）分流为共享融合路径或纯 routed 回退（切 `w1/w2[:shared_expert_id]` 并置 `is_shuffled`） |
| `tests/model_executor/layers/test_fused_shared_expert.py` (+254) | 分片选择、DP 分片权重重构（scale 字节级校验）、modular 路径选择（monkeypatch）、experts 权重选择、门控兼容性（新增 TP1 与 DP8 用例） |

**交叉验证结论**（对 vllm main 实际代码核对，非仅 diff）：
- `rocm_aiter_fused_experts` 已含 `shared_w1/w2/scale/shared_expert_id` 参数（#53161 引入）✓
- `FusedMoEParallelConfig.make` 在 DP 下把 tp 展平为 `dp_size×pcp_size×tp_size`（`flatten_tp_across_dp_and_pcp`），`moe_parallel_config.tp_rank/tp_size` 即展平值 → `_shared_expert_shard_rank_and_size` 在 DP8 下正确返回 `(dp_rank, 8)`，TP8/DP1 返回 `(0,1)` 避免二次切分 ✓
- `Mxfp4MoEMethod.__init__` 先设 `experts_cls`、`process_weights_after_loading` 后经 `_build_moe_kernel` 用它实例化 kernel → 子类覆写时序正确 ✓；`apply()` 签名接受 `shared_experts/shared_experts_input` ✓
- MoE runner 在 `sp_local_sizes` 上下文中调用 `forward_modular` → DP 路径的 `get_chunk_sizes_across_dp_rank()` 在标准执行路径下非 None ✓

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---------|------|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 4 |
| 📝 建议/备注 | 4 |

**⚠️【正确性】CUDA graph 下融合/回退分支在捕获期烘焙，DP 元数据用均匀分布假设** `[已验证捕获机制，后果为推测]`
- **问题**: `forward_modular` 的 DP 分支用 `sum(dp_metadata.get_chunk_sizes_across_dp_rank())` 作为 CSV 探测的 M。该 Python 分支在 CUDA graph 捕获期求值并烘焙进图。已核对 vllm main：捕获路径（`gpu_ubatch_wrapper.py:387-400`）构造的 ubatch DP 元数据是 `[ubatch_slice.num_tokens] * dp_size` 的**均匀分布**——烘焙值 = `dp_size × 本地捕获 token 数`。回放时真实跨 rank 总和 = 各 rank chunk 之和，与本 rank 的图 key（本地 token 数）解耦：同一张图回放的步，其他 7 个 rank 的负载分布可以不同。当烘焙值 ≤ 2048（选融合）而真实总和 ≥ 4096（应回退）时，回放会用超出 AITER 契约（M ≤ 2048）的 M 执行 `fhmoe_`，行为未定义。触发输入示例：TP1/DP8 + graph capture，本 rank 捕获 256 tokens（烘焙 2048 → 融合），回放步其他 rank 各 ~550 tokens（真实 4106 → 应回退）。
- **影响**: 即使不触发上述极端失衡，作者自己的 decode 配置（`max-num-batched-tokens=384`）烘焙/真实总和可达 3072，落在 aiter 探测的**未定义区（2048–4096）**——部署本身就依赖契约外的行为。失衡窗口内可能静默错值或崩溃。
- **行动**: 作者应当（a）在 PR 中明确并断言部署约束（如 FULL_DECODE_ONLY + max-num-batched-tokens 上限保证跨 rank 总和 ≤ 2048），或（b）要求 aiter 侧明确 2048–4096 区间的探测行为，并给出 `fhmoe_` 在超契约 M 下的行为（拒绝/回退/未定义）；建议 review 时追问作者在 graph 捕获下 DP 元数据的取值假设。

**⚠️【测试】ROCm 路径无任何 CI 覆盖** `[已验证]`
- **问题**: 该 PR 触及 Tier-1 文件 `rocm_aiter_moe.py` 与 AMD 专属模型代码，但 checks 列表中无任何 AMD 硬件 CI job；新增测试全部是 CPU monkeypatch 测试（dispatch 逻辑），kernel 正确性完全依赖作者手工 8×MI355X 跑测。
- **影响**: 合入后该路径在 CI 上零覆盖，后续任何回归（如 aiter 版本升级、DP 调度改动）只能靠人工发现，静默数值错误可进入 main。
- **行动**: 建议作者至少把 71 个通过的 `test_fused_shared_expert.py` 挂到 ROCm CI 队列，并在 PR 描述中说明手工验证的复现步骤（当前只有配置摘要，无脚本/日志）。

**⚠️【性能】+8.06% 吞吐声明的稳健性存疑** `[已验证（基于 PR 描述自述）]`
- **问题**: 主表格为单 checkpoint 3+3 run 中位数；PR 描述后半的重测章节自述：在旧 `DeepSeek-V4-Pro` checkpoint 上**未能复现** decode 吞吐表——单臂内出现 67–70% 运行间波动（共享前缀缓存预热效应），效应量被淹没；GSM8K ON 臂两次运行本身有 0.37pp 波动。
- **影响**: 主声明的效应量（-8.39% TPOT / +8.06% TPS）未通过第二个 checkpoint 的稳健性检验；作为 merge 依据的收益数字可能高估。
- **行动**: 建议作者附上原始 run 日志/复现脚本，或在当前 checkpoint 上重跑主 A/B（含 warm-up 丢弃规则）后再定稿性能章节。

**⚠️【可维护性】DP 分支依赖 Python assert 而非显式校验** `[已验证]`
- **问题**: `forward_modular` 中 `assert dp_metadata is not None`、`assert sizes is not None`，且 `get_chunk_sizes_across_dp_rank()` 内部还有 `assert self.local_sizes is not None`。vLLM 生产 Docker 镜像以 `python -O` 运行，assert 被剥离后若执行路径未处于 runner 的 `sp_local_sizes` 上下文（如未来非 runner 调用路径），会从 AssertionError 退化为 `sum(None)` 的 TypeError 或静默错值。
- **影响**: 标准路径下已由 runner 上下文保证（硬件验证通过），但防御层形同虚设，未来重构易踩。
- **行动**: 建议作者将关键断言改为显式 `raise RuntimeError` 并给出可操作的错误信息。

**📝【兼容性】PP>1 未门控，TP1/DP8+PP2 在加载期以晦涩报错失败** `[已验证]`
- **问题**: `_heterogeneous_shared_expert_enabled` 检查了 EP/EPLB/TP×DP/CP/量化等，但未检查 `pipeline_parallel_size`。已核对 `FusedMoEParallelConfig.make`：展平 tp 含 pcp 因子，TP1/DP8+PP2 时展平 size=16，`_prepare_native_fp8_shared_expert` 的 shape 校验（期望 `2×384×16`）对实际权重（`2×3072`）失败 → 加载期 `Unexpected shared W13 shape` ValueError。
- **影响**: 失败是响亮的（不静默），但用户无法从报错定位到「不支持该并行组合」。
- **行动**: 建议作者在门控中显式拒绝 `pipeline_parallel_size > 1`（或同时修正分片逻辑），让不支持组合在启动日志中以明确原因被拒。

**📝【可维护性】output 绑定块与基类 `AiterExperts.apply` 逐字重复** `[已验证]`
- **问题**: 新子类 `apply()` 末尾的 `output.set_(result)`/`copy_` 判断块与基类（`rocm_aiter_moe.py:585-595`）完全相同。CodeRabbit 已提过此 nitpick（建议抽 `_bind_output` 辅助），作者未处理。
- **影响**: 两处漂移风险；当前无功能影响。
- **行动**: 建议作者抽成基类静态方法复用，或至少在重复块加一行注释说明与基类保持一致。

**📝【设计】回退路径用全局 ID 切本地权重，DP8 下切片为 no-op** `[已验证]`
- **问题**: `DeepseekV4HeterogeneousAiterExperts.apply` 回退分支执行 `w1[: self.shared_expert_id]`，其中 `shared_expert_id = 384`（全局 ID）；DP8 下本地权重行数 ≈ 48+1，切片 `[:384]` 实为 no-op——「排除 shared expert」的意图靠「shared 行永远不会被 topk_ids 选中」这一隐式不变量兜底，而非切片本身。
- **影响**: 当前正确（GSM8K 回退臂 2257 次调用精度通过佐证），但耦合脆弱：一旦未来专家 map 语义变化（如 EPLB 重排后 shared 行位置改变），切片意图与实现会静默分歧。
- **行动**: 建议作者在切片处加注释说明 DP 下该切片依赖本地行数 < shared_expert_id 的不变量，或改用本地 slot 索引显式排除。

**📝【注释/文档】docstring 覆盖率 4.55%（阈值 80%）** `[已验证]`
- **问题**: CodeRabbit pre-merge 检查给出 warning：diff 涉及 22 个函数几乎无 docstring。
- **影响**: 不阻塞功能，但可能被 maintainer 要求补齐后再 approve。
- **行动**: 建议作者至少为新类与新辅助函数补一行 docstring（现有 `DeepseekV4HeterogeneousAiterExperts` 已有类级 docstring，其余缺失）。

## 4. 现有讨论 (Existing Discussion)

- **CodeRabbit Critical（已修复）**: 初版无条件用 `moe_parallel_config.tp_rank/tp_size` 做 shared expert 二次切分，会破坏 TP8/DP1（线性层已 TP 切分，再切触发 shape 报错）。作者在 f7f060d…97f0fbe 中以 `_shared_expert_shard_rank_and_size`（dp=1 时返回 `(0,1)`）+ 参数化回归测试修复——修复方向与本文交叉验证结论一致，✅ 该问题已闭环。
- **CodeRabbit nitpick**: 输出绑定块重复（未处理，见 3 节对应条目）。
- **Mergify ×2**（08-28、09-07）: merge conflicts，`rebaseable: false`，需 rebase。
- **Claude bot**: fork PR 自动 review 被禁用；**5 位请求 reviewer 均无人工 review/approval**，截至抓取时 PR 处于无人审状态（assignee @shen-shanshan）。
- **重测章节**（PR 描述后半）: 诚实报告了旧 checkpoint 上 GSM8K parity（0.08pp 差距 vs 0.37pp 噪声）与 decode 吞吐复现失败——透明度值得肯定，但也直接支撑了上文性能稳健性 finding。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK**

架构方向正确（modular 边界 + 展平分片均经交叉验证成立，TP8/DP1 行为保持不变且有回归测试锁定），但存在一个值得作者正面回答的正确性窗口问题（CUDA graph 捕获期分支烘焙 + aiter 契约未定义区 2048–4096），叠加 merge conflicts、aiter#4891 依赖未消费与性能声明稳健性存疑，不建议在 rebase 与上述澄清完成前 approve。建议 review 时重点追问：graph 捕获下 DP 元数据的取值假设、`max-num-batched-tokens=384` 部署下跨 rank 总和 3072 落于契约未定义区的事实是否被作者知悉并接受。
