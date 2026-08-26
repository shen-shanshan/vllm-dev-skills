# PR #53838: [ROCm][DSV4][Perf] Fuse DeepSeek V4 C4 compressor GEMMs

> **Author**: @Fangzhou-Ai | **State**: OPEN | **Date**: 2026-08-26
> **Branch**: `Fangzhou-Ai:afz/rocm-dsv4-compressor-gemm-fusion` → `vllm-project:main` | **Labels**: performance, rocm, deepseek, DSv4
> **Changes**: +221 -0 行，3 个文件（2 commits） | **ROCm 相关性**: 完全相关
> **Review 基线**: base `4663885`，head `41cee28`

## 1. 动机 (Motivation)

DeepSeek V4 C4 目标层的 attention 输入投影阶段需要执行两个共享输入与 K 维（7168）的压缩器 GEMM：主压缩投影（N=2048，产出 `kv_score`）与索引压缩投影（N=512，产出 `indexer_kv_score`）。CUDA 路径通过辅助 HIP stream 与主 GEMM 并行执行这些轻量 GEMM，而 ROCm 路径 `aux_stream_list is None`，`execute_in_parallel` 退化为严格串行——decode 阶段（M=4）两次 `torch.mm` 完全由 kernel 启动开销主导，其中 N=512 的 GEMM 利用率极低。

本 PR 不改动 stream/图调度，而是把两个权重在加载完成后沿 N 维 `torch.cat` 为 `[2560, 7168]`，用 `set_()` 将原 Parameter 零拷贝重绑定到新存储切片，把两次 FP32 输出 `torch.mm` 合并为一次，`split()` 零拷贝切分后返回原消费方。作者声称与 #51794（辅助 stream 方案）正交、不与 #53182 重复，并披露使用 Codex 辅助开发。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 | 说明 |
|------|------|------|
| `vllm/models/deepseek_v4/amd/rocm.py` | +73 | `prepare_compressor_gemm_fusion()`：offloader 守卫 + 形状校验 + `cat`/`set_` 重绑定 + 非持久 buffer；覆写 `_run_parallel_input_projections()`：单次 `torch.mm` + `split` 视图，未融合时回退 `super()` |
| `vllm/models/deepseek_v4/amd/model.py` | +7 | `process_weights_after_loading()` 中逐层调用融合准备，统计并打印融合层数 |
| `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py` | +141 | 3 个 CPU-only 单测：存储别名/state_dict 键名/幂等、offloading 跳过、单次 mm 行为 |

**已验证的关键事实**（跨文件核对，非仅 diff hunk）：
- 融合只作用于 C4 目标层（source 层无 indexer 自动跳过）；`process_weights_after_loading` 在权重加载完成后、图捕获前执行，buffer 地址在图捕获时已稳定。
- `NoopOffloader`/`get_offloader` 存在于 `vllm/model_executor/offloader`（re-export）✓；`logger` 在 amd/model.py 已定义 ✓；`torch.mm(out_dtype=)` 与基类 `attention.py:534/550` 用法一致（ROCm 上依赖 aiter 环境补丁）✓；`indexer.compressor.fused_wkv_wgate` 为既有结构（基类 `attention.py:555` 同用）✓。
- 无 Tier-1/Tier-2 backbone 文件被触及；改动落在 `deepseek_v4/amd/` 模型层（规则分级为 Tier-3 模型专属目录）。

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---------|------|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 3 |
| 📝 建议/备注 | 4 |

### ⚠️【可维护性】`set_` 存储别名是缺乏运行时保障的隐式不变量 `[已验证]`

- **问题**: `prepare_compressor_gemm_fusion()` 中 `main_weight.set_(fused_weight[:main_size])` 与 `indexer_weight.set_(fused_weight[main_size:])`（rocm.py，diff 行 552–554）使两个 Parameter 与 `_fused_compressor_weight` buffer 三方共享同一 storage；计算路径只读 buffer，Parameter 仅是别名镜像。任何后续对参数数据的**整体替换**（`param.data = new_tensor`，如未来 LoRA merge、权重热重载、对已融合模块调用 `.to(device/dtype)`）都会静默断开别名——参数与 buffer 分叉成两份数据，推理继续读 buffer，呈现"权重改了但输出没变"。作者已用 `NoopOffloader` 守卫挡住已知替换源，说明作者清楚这一风险，但该约束只存在于实现者的心智中。
- **影响**: 当前服务生命周期（GPU 加载 → 融合 → 图捕获）内无触发路径，故不构成 BLOCK；风险在未来维护者不知情时引入参数替换。
- **行动**: 作者应当在 `set_` 处加注释说明"此参数数据此后不得被整体替换"，并在 PR 描述中显式写明该不变量；review 时建议确认 maintainer 对此接受度。

### ⚠️【兼容性】形状校验 fail-fast 导致变体模型加载硬失败 `[已验证]`

- **问题**: K 维 / dtype / device 不一致时直接 `raise ValueError`（rocm.py，diff 行 543–550），而非降级回基类路径。对官方 DSV4 权重这是死路径，但 DeepSeek V4 生态正在扩展（社区变体、蒸馏模型），一旦某层两个压缩器权重 K 不一致，模型加载直接崩溃，用户没有任何绕过手段。同函数中其他跳过条件（offloading、无 indexer）都采用"返回 False 静默降级"风格，唯独形状校验例外。
- **影响**: 加载失败（可诊断但粗暴），错误发生在启动期，影响面为用户全部请求而非单次推理。
- **行动**: 建议作者改为 `logger.warning` + 返回 `False` 降级，与其他跳过条件风格统一；若作者有意 fail-fast 以尽早暴露异常 checkpoint，请在 PR 描述中说明理由。

### ⚠️【测试】ROCm CI 未跑，gfx942 未验证 `[已验证]`

- **问题**: head commit 的 `checks` 仅 5 个轻量检查全绿（pre-commit / pre-run-check / Summary / Meta Internal-Only / DCO），**无任何 AMD 队列 CI**（fork PR 需 maintainer 批准后才触发 Buildkite）。作者本地验证仅在 gfx950（MI350）+ ROCm/HIP 7.2.53211 完成；gfx942（MI300X/MI325X）上 N=2560 的 hipBLASLt/aiter kernel 选择、以及不同 aiter 版本下的行为均未验证。TP8 服务与 GSM8K 均为作者本地结果，无 CI 回归保障。
- **影响**: 若 N=2560 在 gfx942 或旧 aiter 组合触发未测试的 kernel 路径，可能出现性能劣化甚至数值异常，AMD 用户首当其冲；合入后无 CI 兜底。
- **行动**: 建议 assignee 批准触发 AMD CI（gfx942 + gfx950 队列）并在合入前确认全绿；作者应当补充 gfx942 上的正确性/性能抽查结果，或说明不适用理由。

### 📝【设计】新默认路径无临时 kill-switch

- **问题**: 融合在条件满足时默认开启，无环境变量可禁用；vLLM 惯例是新实现未经充分验证时提供临时开关（验证后移除）。
- **影响**: 若线上出现未预料的数值/性能问题，只能回退整个版本。
- **行动**: 建议作者评估是否加入临时开关（如 `VLLM_ROCM_DISABLE_DSV4_COMPRESSOR_FUSION`）；review 时可与 maintainer 确认是否需要。

### 📝【性能】microbench 数字无法溯源 `[已验证]`

- **问题**: "eager 42.04→29.16 us (1.442x) / graph-replay 37.16→24.44 us (1.520x)" 无脚本、无原始输出；serving 表格给了完整命令但未附原始 JSON。
- **影响**: 按溯源纪律，这些数字只能标注 [unverified]，不转述为事实；端到端 +2.2~2.9% 吞吐的结论方向可信（方法论完整），具体数值待佐证。
- **行动**: 建议作者附 microbench 脚本与 `benchmark-results/*.json`（或原始输出摘要），便于 reviewer 复现。

### 📝【可维护性】非连续视图传给下游——当前安全，未来脆弱 `[已验证]`

- **问题**: 融合后 `kv_score` / `indexer_kv_score` 是 stride (2560, 1) 的非连续视图（原路径为各自 mm 的连续输出）。我追踪了全部消费方：`compressor(kv_score)` → `save_partial_states` triton kernel 显式传 `kv.stride(0)`/`score.stride(0)`，stride-aware ✓；`indexer(indexer_kv_score)` → `rocm_aiter_sparse_attn_indexer` 因 `skip_k_cache_insert=True` 且调用点（attention.py:590）传 `k=None`，不消费该视图 ✓；`_indexer_k_quant_and_cache_kernel` 虽按行主序读 `k_ptr + tid * head_dim`（无 stride 参数），但此路径对 DSV4 不可达 ✓。**故当前无正确性问题**——这是经完整调用链核对后排除的疑点，而非一眼可见的结论。
- **影响**: 未来若新增直接消费这些视图的 kernel（或有人把 `skip_k_cache_insert` 改回 False），将静默读错内存。
- **行动**: 建议在 `_run_parallel_input_projections` 覆写处加一行注释："输出为非连续视图（stride 由 fused N 决定），下游消费必须 stride-aware"。

### 📝【兼容性】融合 mm 依赖平台级 `torch.mm(out_dtype=)` 补丁

- **问题**: 新调用点沿用基类的 `torch.mm(..., out_dtype=torch.float32)`（ROCm 上由 aiter 环境补丁提供）；N=2560 满足 16/128 对齐。作者 gfx950 实测正常，说明当前环境组合可行。
- **影响**: 若未来 aiter 收紧 N 对齐约束或移除补丁，需回查此处。
- **行动**: 无需行动，review 时知晓即可。

## 4. 现有讨论 (Existing Discussion)

- PR 创建数小时，**无 issue comments、无 inline review comments**，评审尚未实质展开。
- claude[bot] 唯一一条 review：fork PR 自动 review 被禁用，需 maintainer 评论 `@claude review` 触发一次性审查。
- 已请求 5 位 reviewer（tjtanaa、zyongye、AndreasKaratzas、hongxiayang、dllehr-amd，多为 AMD/ROCm 背景）；assignee 为 @shen-shanshan。
- `mergeable_state: unstable`，`rebaseable: false`——当前 5 个轻量检查绿但完整 CI（尤其 AMD 队列）尚未运行。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK** —— 优化思路扎实、收益数据可信（数字待补溯源），核心正确性经跨文件核对未发现缺陷（非连续视图疑点已排除），但合入前需处理：`set_` 别名不变量的文档化、形状校验降级策略的确认，以及 AMD CI（gfx942 + gfx950）的触发与通过。
