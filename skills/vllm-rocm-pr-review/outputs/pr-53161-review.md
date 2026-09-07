# PR #53161: [ROCm][Perf][DeepSeek V4] Fuse native FP8 shared expert with MXFP4 routed experts

> **Author**: @Fangzhou-Ai | **State**: OPEN | **Date**: 2026-08-20（最近更新 2026-09-05）
> **Branch**: `Fangzhou-Ai:rocm-dsv4-fhmoe-i384` → `vllm-project:main` | **Labels**: `rocm`, `ready`, `deepseek`, `DSv4`
> **Changes**: +843 -6 行，4 个文件 | **ROCm 相关性**: 完全相关

## 1. 动机 (Motivation)

DeepSeek V4 在 ROCm gfx950 上的 MoE 是"异构"的：routed experts 用 MXFP4（384 专家、top-6），shared expert 用原生 FP8 E4M3（TP8 下每 rank I=384）。现有实现中 shared 与 routed 是**两次独立算子调用**，多付一次 kernel 启动和中间结果往返。本 PR 启用 AITER 的 FHMoE（异构融合 MoE）：在满足 ~20 项 fail-closed 门控（gfx950 / TP8 / DP1 / PCP1 / BF16 / noaux_tc / 无 EP、EPLB、offload / DSV4 量化契约 / AITER 能力探测）时，把两者合并进**单次 kernel 调用**，并保持原生 I384（不恢复旧方案 #48728 的 384→512 填充）。覆盖范围由 AITER 的 CSV 配置表决定（当前 1≤M≤2048），未覆盖的 M 走原分离路径。特性 opt-in（`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`），依赖 ROCm/aiter#4891 合入。实测 C8 吞吐 +8.14%、decode step -10.73%、MoE 耗时 -27.34%、kernel 启动 -15.57%；GSM8K 30-shot strict 0.96740。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/models/deepseek_v4/amd/model.py` (+381/-5) | 核心：`_heterogeneous_shared_expert_enabled` 门控（收集全部不满足原因后 debug 输出）；`_prepare_native_fp8_shared_expert` 把 block-128 E8M0 scale 扩展为 FHMoE 1×32 布局（W2 补齐到 16 列，0x7F 中性填充）；`DeepseekV4HeterogeneousSharedRoutedExperts(RoutedExperts)` 子类持 shared 模块 weakref，`forward_modular` 按 M 分派 fused / fallback 双路径；fallback 用 `dataclasses.replace` 剔除 append 槽位 scale 后跑独立 kernel 再相加；模型/层/FFN 三级构造器透传开关 |
| `vllm/_aiter_ops.py` (+107) | shared 参数全有或全无校验；`_probe_dsv4_i384_fhmoe_capability`（`aiter.fhmoe.supports_dsv4_i384_fhmoe` 返回字面量 True + fused_moe 签名含 5 个 shared-* 参数，任何异常→False，`@functools.cache` 缓存）；impl/fake/包装函数签名同步扩展；新增 `shuffle_scale` 包装 |
| `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` (+15) | 透传 5 个 shared-* 参数；**顺带**把 `moe_config.swiglu_limit` 显式传入 aiter kernel（此前从未传入） |
| `tests/model_executor/layers/test_fused_shared_expert.py` (+340/-1) | 8 组测试：原生 I384 保持、token 策略（0/1/1536/2048/2049/4096/4097）、能力探测异常路径（导入失败/签名异常/非布尔返回）、兼容性门控、路由形状校验、shared 参数校验 |

**跨文件验证结论**（对本 PR diff 引用的全部新符号逐一溯源至 base SHA `2a336d8`）：`FusedMoEFactory(routed_experts_cls/routed_experts_args)`、`RoutedExperts` 的 `w13_weight/w2_weight/expert_map/expert_map_manager/global_num_experts/_ensure_moe_quant_config_init/moe_config/quant_method`、`ExpertMapManager.num_fused_shared_experts`、`FusedMoEConfig.swiglu_limit/experts_per_token/intermediate_size_per_partition`、`FusedMoEQuantConfig._w1/_w2(FusedMoEQuantDesc.scale)/w1_scale/w2_scale`、`Mxfp4MoEMethod.moe_quant_config`、`rocm_aiter_fused_moe` 注册方式（schema 由 impl 签名推断，impl/fake 已同步扩展）、`aiter.ops.shuffle.shuffle_scale`（aiter main 中存在）——**未发现幻觉符号**。能力探测对 #4891 未发布 API 的 fail-closed 设计是正确做法。

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---------|------|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 4 |
| 📝 建议/备注 | 5 |

---

**⚠️【正确性】`swiglu_limit` 被静默接入 aiter 路径，行为变更未披露，影响所有既有 ROCm aiter-MoE 用户** `[已验证]`
- **问题**: `rocm_aiter_moe.py` 在透传 shared 参数的同时，新增了 `swiglu_limit=(0.0 if moe_config.swiglu_limit is None else float(...))`。改动前该值**从未传入** aiter kernel（impl 默认 0.0 = 不 clamp）；改动后，任何配置了 `swiglu_limit` 的模型（DeepSeek V4 本身即配置了该值）在 **aiter 路径下（含未开启融合的既有用户）** 的数值行为都会改变——SiLU clamp 从"静默不生效"变为"生效"。fused 与 fallback 两条路径都继承了该变更。
- **影响**: 这很可能是一个正确的 bug fix（对齐 `moe_config` 语义与 CUDA 路径），但它是**未披露的范围外改动**：既有部署升级后输出数值会漂移，且 PR 的精度验证（GSM8K 30-shot）只覆盖了"最终栈"整体结果，没有 clamp-on vs clamp-off 的隔离对比，无法把精度影响归因。
- **行动**: 作者应当在 PR 描述中明确列出该行为变更及其动机，并给出对既有 aiter 路径（feature-off）的精度对比；若无法归因，建议将该行拆出为独立小 PR 单独 review。

**⚠️【正确性】backend 回退导致 384→512 round 时，模型加载硬失败而非优雅回退** `[推测]`（CodeRabbit 已提出，未解决）
- **问题**: `prepare_heterogeneous_shared_expert` 用 `self.moe_config.intermediate_size_per_partition` 作为期望宽度，对原生 `(768, 7168)` 共享权重做严格形状校验并抛 ValueError。但 `select_deepseek_v4_mxfp4_moe_backend` 可能从 `AITER_MXFP4_BF16` 回退到其他 AITER backend，后者会把该宽度 round 到 512——门控检查的是配置契约，检查不到 backend 内部的 round 决策。
- **影响**: 门控全部通过后，在加载期以 ValueError 崩溃，违背了 PR "fail-closed → 回退分离路径" 的承诺（fail-closed 变成了 fail-hard）。
- **行动**: 建议作者采纳 CodeRabbit 方案——宽度改为从 `shared_expert.gate_up_proj.weight.shape[0] // 2` 推导，或对该场景增加显式回退分支。

**⚠️【兼容性】能力探测未覆盖 `shuffle_scale`，探测通过后仍可能在加载期崩溃** `[推测]`
- **问题**: `_probe_dsv4_i384_fhmoe_capability` 只校验 `supports_dsv4_i384_fhmoe` 与 `fused_moe` 签名，不校验 `aiter.ops.shuffle.shuffle_scale` / `shuffle_weight_a16w4` 的可用性；而 `prepare_heterogeneous_shared_expert` 在加载期无条件调用它们。
- **影响**: 若 aiter#4891 所在 revision 因某种原因缺失 `shuffle_scale`（当前 aiter main 中存在，风险低），探测通过 → 加载期 ImportError/AttributeError，而非承诺的分离路径回退。
- **行动**: 建议作者在探测中一并校验 shuffle 系列符号，或在 PR 描述中确认 #4891 目标版本包含 `shuffle_scale` 并说明依赖关系。

**⚠️【可维护性】M 超出 CSV 覆盖时静默回退，无任何运行时诊断** `[已验证]`
- **问题**: `forward_modular` 中 `_use_heterogeneous_fhmoe(x.shape[0])` 为 False 时直接走分离路径，不产生任何日志。初始化期的 `debug_once` 只覆盖配置级不可用；MTP4 满配（M=2560）这类**逐次调用级**的覆盖缺失对运维完全不可见。
- **影响**: 性能承诺对部分负载（M>2048，如 512 序列 MTP4）不生效，但线上无法观测到"当前走的是哪条路径"，性能排查困难。PR 的 M=2049 诊断数据（fused 比分离慢 37.98%）说明这类边界真实存在。
- **行动**: 建议作者在 fallback 分支加一条 rate-limited 的 debug 日志（可按 M 值 debug_once），成本极低。

---

**📝【测试】CI 状态：上游 Buildkite #87166 已触发，但 AMD CI 未见于记录** `[已验证]`
- fetch 时 checks 数据为空（API 未返回结论），评论中只见 `/ci run`。该 PR 触及的 gfx950 FHMoE 路径不在上游 CUDA 队列覆盖内。作者有充分的真实硬件验证（8×gfx950、C1-C64 全量），但建议 merge 前显式跑一次 `/amd-ci run`。

**📝【性能】数字为作者自报，无脚本/日志 artifact** `[已验证]`
- PR 给出了完整配置与方法学（8x gfx950、TP8、80 请求 8K/1K、`max_model_len=9472`、`max_num_seqs=512`、`max_num_batched_tokens=16384`），且"isolated flag A/B 归因"与"combined-stack 集成验证"的区分方法学正确；但未附 benchmark 脚本或原始日志，精确复现不可行。按纪律标记为作者自报，建议附上复现脚本。

**📝【资源】shared 权重双份常驻显存，已披露且量化** `[已验证]`
- fallback 路径需要原生 shared 模块，因此其权重与 shuffle 副本同时常驻（+0.84 GiB，+0.81%；KV 容量 -0.60%）。PR 已明确披露，`persistent=False` buffer 保证 checkpoint 干净，成本可接受，仅记录在案。

**📝【设计】0x7F 中性填充为魔法常量** `[已验证]`
- W2 scale 补齐列用 0x7F（E8M0 编码下恰为 1.0，中性乘数），测试断言了该值但源码无推导注释。建议加一行注释或命名常量，防止后人误改。

**📝【正确性】appended shared 路由权重恒为 1.0 是隐含不变量，未断言** `[推测]`
- fused 路径由 AITER kernel 按自身契约消费第 7 列路由；fallback 路径直接 `routed + shared_out`，**不应用**该列权重。两者仅在"appended 权重恒为 1.0"（DSV4 noaux_tc 契约）下等价。`_validate_heterogeneous_routes` 只校验形状不校验数值。若未来路由实现变化（如对 shared 加权），两条路径会静默发散。建议在校验函数中对 appended 列权重增加断言或注释说明依赖。

## 4. 现有讨论 (Existing Discussion)

- **@tjtanaa（AMD 维护者）**：追问 EP 模式是否可用——作者回复只测过 TP，EP 情况需 @LiuYinfeng01 / @jiacao-amd 补充；门控已显式排除 EP，但 EP 场景零验证记录。
- **@tjtanaa**：要求补全 30-shot GSM8K——作者已贴出 strict 1276/1319 = 0.96740（超过 0.94 验收线）。
- **CodeRabbit**：1 条 actionable 意见（即上文 ⚠️ 第 2 条，宽度应取自权重形状），综合风险 🟡 Moderate；另提示 docstring 覆盖率 16.98% 低于 80% 阈值。
- 合并状态：`mergeable_state: unstable`，需与 main 同步；**硬性合并阻塞为 ROCm/aiter#4891**（PR #52826 仅为 v0.1.20 基线 bump）。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**

实现质量高：fail-closed 门控、全链路符号验证无幻觉、真实硬件证据链完整（性能/精度/显存/图捕获），单测覆盖了探测异常路径与边界 M。当前不可合并的硬性理由是 aiter#4891 依赖；合入前应处理：① swiglu_limit 未披露行为变更的说明与归因（或拆分）、② CodeRabbit 宽度派生问题、③ merge 前跑 AMD CI 并解决 unstable 状态。以上均为 ⚠️ 级，无 🔴。
