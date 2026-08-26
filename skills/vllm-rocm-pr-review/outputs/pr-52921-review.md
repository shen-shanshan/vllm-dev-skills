# PR #52921: fix(simple_kv_offload): align CPU offload pool size across PP/spec-decode workers

> **Author**: @charxwu | **State**: OPEN | **Date**: 2026-08-19(更新 08-25)
> **Branch**: `charxwu:fix/simple-cpu-offload-pp-alignment` → `vllm-project:main` | **Labels**: `kv-connector`, `verified`
> **Changes**: +410 -14 行,11 个文件 | **ROCm 相关性**: 部分相关(AMD 动机 + MI355X 实测,但改的是后端无关的共享 plumbing)

## 1. 动机 (Motivation)

`SimpleCPUOffloadConnector` 的调度器与 worker 各自独立估算 `num_cpu_blocks`:调度器按 worker 0 的 KVCacheConfig 估算,worker 按自己注册的真实 KV tensor 计算。在 PP 层数不均或投机解码(draft KV 落在最后一个 PP stage)下,较重 rank 每 block 字节数更大、能容纳的 block 更少,调度器却按轻 rank 的估算分配 `cpu_block_id`,导致越界 DMA——实测 MI355X TP8×PP2 上 `hipMemcpyBatchAsync` SIGSEGV(PP0=809 / PP1=746 / scheduler=809)。修复分三层:worker 按真实 stride 计算后 **all-reduce MIN** 统一各 rank 池大小;engine 在 `initialize_from_config()` 后经 collective RPC 取对齐值;调度器以对齐值为权威(带 config-min 兜底)。此修复取代 InferenceX 手工设置 `KV_OFFLOAD_MAX_CPU_BLOCKS` 的 monkey-patch,动机与机制(WHY)解释充分,实测数据有集群路径可溯源。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/simple_kv_offload/sizing.py`(新增) | stride 计算、本地 block 数、all-reduce MIN 同步、config min 估算 |
| `vllm/v1/simple_kv_offload/worker.py` | CPU/磁盘模式改走 `local_num_offload_blocks` + 集体同步后再分配缓冲区 |
| `vllm/v1/simple_kv_offload/manager.py`(调度器) | 新增 `worker_kv_cache_configs` / `aligned_num_cpu_blocks` 参数,三级 sizing 决策 |
| `vllm/v1/engine/core.py` + `executor/abstract.py` + `worker/gpu_worker.py` | 初始化链路:保存每 worker config → RPC 取对齐值(取 min + 告警)→ 写入 `cache_config` |
| `vllm/config/cache.py` | 两个 `init=False` 运行时字段作为数据通道 |
| `vllm/distributed/kv_transfer/.../simple_cpu_offload_connector.py` | 递归查找 connector(穿透 MultiConnector)+ `get_aligned_num_cpu_blocks()` |
| 测试 ×3 文件 | 3 个调度器测试 + 2 个 mock all-reduce 测试 + 1 个 GPU sizing 测试 |

已跨文件核实:`MultiConnector` 的子连接器属性确为 `_connectors`(递归查找有效);调度器 `offload_capacity = disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes`,与 worker 按模式的本地计算一致;`world_group.cpu_group`(gloo)支持 MIN all-reduce。未触及 aiter/mori/rocm 专属文件、无新 env var、无新依赖,相应规则不适用。

## 3. Review 意见 (Findings)

**意见类型分布**:⚠️ 建议修复 ×6 | 📝 备注 ×2 | 🔴 无

### ⚠️【兼容性】engine/core.py 无条件新增 collective RPC,目标方法只存在于 GPU worker 类 `[已验证 + 推测]`

- **问题**: `vllm/v1/engine/core.py:_initialize_kv_caches` 无条件调用 `self.model_executor.get_simple_cpu_offload_num_blocks()`,其 RPC 目标方法 `get_simple_cpu_offload_num_cpu_blocks` 只定义在 `gpu_worker.py` 的 `Worker` 类上。已核实 PR base(d9fbe52)下 V1 worker 全集为 Worker / CPUWorker / XPUWorker,且后两者均继承该方法,**当前树内无崩溃触发**;但已核实 multiproc 的 `collective_rpc` 是逐 worker 直接 `getattr(self.worker, method)`、无 hasattr/NotImplementedError 兜底,失败会被包装成 `RuntimeError`。
- **影响**: 任何不继承 `Worker` 的 worker(第三方 executor、外部插件、未来的新后端)会在引擎启动时崩溃;该方法也绕过了「基类默认实现」的既有模式(如 `determine_available_memory` 在 WorkerBase)。
- **行动**: 建议作者在 `WorkerBase` 定义返回 `None` 的默认实现(或在 executor 端做 `hasattr` 检查),并在 PR 描述中说明对非 GPU worker 的考虑。

### ⚠️【正确性】`gpu_total_bytes` 的「size 全相等 → 单一 backing allocation」启发式可能误判 `[推测]`

- **问题**: `sizing.py:gpu_total_bytes` 以「所有 `kv_cache_tensors` 的 size 相同」推断它们同属一个 backing allocation 并只取第一个 size;否则求和。若某配置下存在两个**互不相交、大小恰好相同**的分配,会被误判为单一分配 → 低估 GPU 总字节 → 高估 `num_cpu_blocks` → 兜底路径下重新引入越界 block id,原 SIGSEGV 复现。
- **影响**: config 兜底路径(RPC 不可用时)在特定 KV 布局下的静默越界风险。
- **行动**: 建议作者改用显式 offset/size 布局信息判断(测试 fixture 已引入 `offset` 字段,可直接作为判定依据),或在注释中写明「size 相等 = 单一分配」由哪个不变式保证。

### ⚠️【可维护性】sizing.py 三个 helper 与 worker 内联逻辑孪生,且未被生产路径使用

- **问题**: `build_unique_gpu_block_views` / `total_bytes_per_block_from_views` / `compute_total_bytes_per_block_from_kv_caches` 只被 `test_sizing.py` 引用;`worker.register_kv_caches` 仍使用自身内联的**相同算法**(同样的 `storage.data_ptr()` 去重、`stride(0) * element_size()`、`raw.view(-1, num_blocks, block_bytes)`)。PR 描述声称「workers compute num_cpu_blocks from live GPU KV tensor strides」——该计算是既有内联代码,新 helper 并未接入生产。
- **影响**: 测试验证的是生产不运行的代码(为通过而校准);两套实现漂移后,真实路径的正确性回归不可见。
- **行动**: 建议作者让 `worker.register_kv_caches` 复用 sizing.py 的 helper(单一真相源),或删除未使用的 helper。

### ⚠️【测试】多 rank 同步路径在 CI 中只有 mock 覆盖

- **问题**: `test_worker.py` 的两个测试 mock 了 `get_world_group` / `dist.all_reduce`;调度器测试为单进程;`test_sizing.py` 需 GPU。**没有任何自动化测试跑真实多进程的 all-reduce MIN 与 collective RPC 链路**——而这恰是本 PR 修复的核心(跨 rank 对齐)。
- **影响**: 后续对 ReduceOp、group、RPC 时序的任何重构不会被 CI 拦截。
- **行动**: 建议作者补一个基于 `torch.multiprocessing.spawn` 的双 worker 同步测试(或说明 CI 成本考量),把「all-reduce MIN 后所有 rank 一致、engine 侧取到 min」这条链路固化下来。

### ⚠️【CI/覆盖】ROCm CI 未跑、CUDA 侧未验证(改动在双后端共享路径)

- **问题**: checks 状态 pending 且无 check-run 明细(仅 mergify 报告的 pre-commit 失败);作者在 MI355X TP4×PP2 / TP8×PP2 手动验证并给出结果路径,但 `SimpleCPUOffloadConnector` 同样服务 CUDA 用户,engine/executor/worker 的改动是共享 plumbing,无任何 CUDA 验证记录,也未见 AMD CI 运行。
- **影响**: 合入后 CUDA 用户首当其冲踩回归;AMD 侧缺少自动 CI 防线(手动验证不可重跑)。
- **行动**: 建议作者在 CI 解锁后触发 `/ci run` 与 `/amd-ci run`,并补充 CUDA 单卡 smoke 或说明不需要的理由。

### ⚠️【流程】merge conflicts + pre-commit 失败 + DCO 待补

- **问题**: mergify 08-22 报 merge conflicts(`mergeable_state: unstable`);08-25 两次报 pre-commit 失败;作者 08-21 自述 DCO `Signed-off-by` 未补齐。
- **影响**: 短期无法进入 merge queue,review 通过后仍需返工。
- **行动**: 作者应当 rebase main、运行 `pre-commit run --all-files`、补 Signed-off-by 后重新推送。

### 📝【可维护性】`CacheConfig.worker_kv_cache_configs` 类型为 `Any`

建议改为 `list[KVCacheConfig] | None`(TYPE_CHECKING 导入),与同文件 `simple_cpu_offload_num_blocks: int | None` 的严谨度对齐。

### 📝【行为变化】对齐后 offload 池容量可能下降(实测 809→746,约 -8%)

正确性优先的有意取舍,但用户可感知(显存紧张场景下可 offload 的 block 数变少)。建议在 PR 描述或文档中明示容量变化,便于用户对比升级前后行为。

## 4. 现有讨论 (Existing Discussion)

- **@charxwu (08-21)**: 请求 @ivanium/@orozery 添加 `verified` label 解锁 CI(合并 PR 数 <4 导致 `pre-run-check` 卡住),并主动声明 DCO 待补。label 已加上。
- **mergify[bot] (08-22 / 08-25×2)**: 报 merge conflicts 与两次 pre-commit 失败。
- **claude[bot]**: fork PR 自动 review 禁用,需 maintainer 手动触发。
- 尚无人工 inline review 或 approval。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK** — 修复思路正确(三层对齐 + 兜底),根因与机制解释清晰,MI355X 双拓扑实测数据可溯源;但需先解决流程阻塞(rebase / pre-commit / DCO),并处理两处设计疑虑(非 GPU worker 的 RPC 健壮性、`gpu_total_bytes` 启发式),同时为跨 rank 同步核心链路补自动化测试。
