# 计划：在 vLLM 中重新实现跨 DP 的 prefill delayer（接手 #61834）

## Context

#61834（`--dp-prefill-delayer`）的作者已决定关闭该 PR，由我们开新 PR 接手。社区反馈（#61834 评论区，以及 @njhill 在同类 PR #51199 上的意见）可归纳为四点：
- 不要新增一堆调参开关和 scheduler 复杂度；
- 必须证明比现有的 `--prefill-schedule-interval` 更好；
- 必须有 TTFT 的**时间**上界；
- 新增的每步 collective 有 hang 风险，需要从结构上保证所有 rank 都会执行。

另外，#61834 的 stall 判定存在已知缺陷：pending 被截断到 budget 后，数值不变并不代表没有新工作。参考实现后来已修复这一点，修法是引入"新到达请求 / prefillable rank 数变化"信号，以及 blocked / unreachable 两类 rank。

用户已确认的决策：
1. **通用特性，默认关闭**，不限于 ROCm。这一点偏离 skill 的 ROCm-only 规则；其余规则照常遵守：代码、注释、提交信息、PR 描述中不出现参考引擎的名字，计划用中文，PR 描述持续同步。
2. 首版功能范围是"**核心规则 + stall/blocked/unreachable 修正 + 按毫秒的时间上界**"。不做 cost gate、adaptive、DP 代价均衡，这些留给后续 PR。
3. 配置接口为**复用 `--prefill-schedule-interval` 作为保护步数**：新增 `--prefill-schedule-policy {cadence,delayer}` 和 `--prefill-max-wait-ms`，不再有互斥报错。

目标：一个默认关闭、opt-in 的 DP prefill 调度策略。开启后，MoE DP 的各 rank 每步交换 prefill 状态，统一决定 FIRE（放行）或 HOLD（暂缓）。默认配置下，所有平台的行为逐字节不变。

## 第 0 步：保存计划副本

计划批准后，第一步先把本计划原样复制到 `/home/shashen/vllm/vllm-dev-skills/docs/tasks/prefill_delayer/dp_prefill_delayer_plan.md`。plan 模式下除计划文件外不允许写其他文件，所以放在批准后执行。

## 分支

```bash
cd /home/shashen/vllm/vllm
git fetch upstream main
git checkout -b dp-prefill-delayer upstream/main
```
（`dev/` 是未跟踪目录，不影响切分支。）

## 设计要点

### 1. 配置（`vllm/config/scheduler.py`）
- 新增 `prefill_schedule_policy: Literal["cadence", "delayer"] = "cadence"`。
  - `cadence`：现有行为，即固定节拍。
  - `delayer`：只对 MoE DP（DP>1）生效。每步对 DP 组做一次聚合，各 rank 一起决定是否放行 prefill；此时 `prefill_schedule_interval=N` 的含义变为"两次 prefill 步之间至少相隔 N 步"，即 prefill 后保护 N−1 个纯 decode 步，N=1 表示不保护。
- 新增 `prefill_max_wait_ms: float | None = Field(default=None, gt=0)`：只在 `delayer` 下生效，等待 prefill 的请求中最老的一个，从到达开始排队超过该时长就强制放行。如果在 `cadence` 下设置，报 `ValueError`。
- 更新 `prefill_schedule_interval` 的 docstring，说明它在两种策略下的含义。
- `vllm/engine/arg_utils.py`：两个新字段都走现有的 `scheduler_kwargs` 模式，加入 `EngineArgs`、`add_cli_args` 和 `create_engine_config`。

### 2. 决策核心（新文件 `vllm/v1/core/sched/prefill_delayer.py`，约 170 行）
- `PrefillDelayerState`（NamedTuple）表示单个 rank 上报的状态，共 10 个 int32 槽位，SUM 聚合：

  | 槽位 | 含义 |
  |---|---|
  | `prefillable` | 有可执行的 prefill（partial chunk，或可接纳的 waiting） |
  | `blocked` | 有 waiting，但 head 当前接纳不了：seq 槽满、KV 使用率 ≥ 0.9，或只剩远端 KV 加载中的请求。这类 rank 视为已对齐，不计 pending |
  | `unreachable` | 既没有 prefill 也没有 waiting，且最近一次 decode 完成落在剩余的 hold 预算之外，或根本没有 decode |
  | `pending` | 待 prefill 的 token 数，截断到 budget |
  | `decodes` | decode 请求数 |
  | `kv_high` | KV 使用率高（达到 0.9） |
  | `partial` | 有处于 chunked prefill 中途的请求 |
  | `queue_hot` | 最老的 waiting 超过 `prefill_max_wait_ms` |
  | `prefill_ran` | 上一步执行过 prefill |
  | `arrived` | 新到达的请求数 |

- `PrefillDelayer.decide(reduced) -> bool` 是确定性状态机，所有计数都按 tick 计，因此各 rank 结果一致。规则按顺序如下：
  1. `prefill_ran` 时启动 protect = N−1；
  2. 首次调用直接 FIRE；
  3. protect 期间：如果有 prefill 且有 decode，则 HOLD（不计入 hold_ticks，但工作量有变化时重置 stall）；
  4. 没有 prefillable 的 rank → FIRE，并 reset；
  5. 没有 decode → FIRE；
  6. `kv_high` → FIRE；
  7. `queue_hot` → FIRE；
  8. hold ≥ 200 → FIRE；
  9. 有 partial 且 hold ≥ 100 → FIRE；
  10. 更新 stall：pending 增长、prefillable rank 数变化或有新到达时清零，否则加一；
  11. 如果 `prefillable + blocked < dp_size`（未对齐）：只有在 `unreachable > 0` 且 stall ≥ 10 时 FIRE，否则 HOLD；
  12. fill ≥ 0.9 → FIRE；stall ≥ 10 → FIRE；其余情况 HOLD。
- 常量（0.9 / 200 / 100 / 10 / KV 阈值 0.9）作为模块常量，不对外暴露成开关，以回应社区"knob 太多"的意见。
- 记录每种 FIRE 原因的计数，每 1000 次决策输出一次 `logger.debug`，便于调参诊断，不新增开关。

### 3. 采集本地状态（`vllm/v1/core/sched/scheduler.py`）
- 新增 `Scheduler.prefill_delayer_state(hold_budget: int) -> PrefillDelayerState`，读取 `running`、`waiting`、`deferred_waiting`、`kv_holding_waiting`、`async_kv_waiting`、`kv_cache_manager.usage`、`max_num_running_reqs`。
  - pending 的扫描在达到 budget 时提前停止。
  - 计算 unreachable 时，取 decode 请求中 `max_tokens - num_output_tokens` 的最小值，再除以每步可接受的 token 数（1 + spec）。
  - 只有在设置了 `prefill_max_wait_ms` 时，才扫描 waiting 的 `arrival_time`。
  - 返回后清零 `prefill_ran` 和 `arrived` 两个累计量。
- `add_request` 中 `self._delayer_arrived += 1`（只在 delayer 开启时累计）。
- `schedule()` 末尾复用已有的 `prefill_scheduled or prefill_admitted` 标记设置 `prefill_ran`，不另写启发式判定（#61834 用的 `n > 1 + spec` 判定被替换掉）。
- `defer_prefills`：在 delayer 下忽略 `prefill_capacity_bound`，由 delayer 自己的上界兜底；**保留"本 rank 有 decode 才 HOLD"的条件**，避免出现空 batch 导致 hang（#61834 中 benenzhu/vllm#6 的修复）。
- `local_prefill_interval`（DP=1 时的本地节拍）不变：在非 DP 引擎上，`delayer` 退化为本地 cadence，并打一次 `info_once`。

### 4. 引擎集成（`vllm/v1/engine/core.py`，`DPEngineCoreProc`）
- `__init__`：当 `policy == "delayer"` 时：
  - 如果开启了 `enable_elastic_ep` 或 `enable_fault_tolerance`，报 `ValueError`（这两者会在运行时改变 DP size）；
  - 如果 scheduler 不是内置的 `Scheduler`，也报错；
  - 构造 `PrefillDelayer(dp_size, max_num_batched_tokens, prefill_schedule_interval)`。
- **把 collective 合并进现有的 DP 状态同步**（回应 hang 风险，同时省掉一次通信）：
  - `_has_global_unfinished_reqs` 在 delayer 开启时**每步**都同步，不受 `dp_sync_interval` 限制，并把 delayer 的 10 个槽位拼进同一次 `sync_dp_state`；
  - 这样 delayer 和已有的"所有 rank 每步必经"的同步点从结构上绑定在一起，不可能被单个 rank 跳过；
  - 同步结束后调用 `decide()`，把结果写入 `self.hold_prefills`；
  - wave 结束时 reset。
- `_should_throttle_prefills`：delayer 开启时返回 `self.hold_prefills`。
- `vllm/config/parallel.py` 的 `sync_dp_state` 增加可选参数 `extra: torch.Tensor | None = None`：
  - 传入时把它拼接到同一个 int32 tensor 后面，在同一次 `cpu_all_reduce_` 中完成，并原地写回聚合结果；
  - 默认 `None` 时，行为与现在逐字节一致。

### 5. 文档
- `docs/serving/data_parallel_deployment.md` 新增一个小节"Prefill scheduling across DP ranks"，内容包括：
  - 两种策略分别是什么、怎么开启、默认值；
  - `interval` 在两种策略下的含义；
  - `prefill_max_wait_ms`；
  - TTFT 代价，以及适合高并发 agentic 负载的场景；
  - 不支持的组合（elastic EP、fault tolerance、自定义 scheduler）。
- CLI 参数的参考页由 config docstring 自动生成。

## 对其他后端 / 默认行为的影响

- 默认 `prefill_schedule_policy="cadence"`，`prefill_max_wait_ms=None`：
  - `DPEngineCoreProc` 不构造 delayer；
  - `_has_global_unfinished_reqs` 走原来的分支，`sync_dp_state` 不传 `extra`；
  - scheduler 新增的累计逻辑只在 delayer 开启时执行。
- 因此 CUDA / ROCm / XPU / CPU 等所有平台在默认配置下的行为逐字节不变。实现完成后，我会逐一检查每个非新增文件的 diff 来确认这一点。
- 不改任何现有默认值、签名或语义。`sync_dp_state` 只新增一个带默认值的可选参数。

## 测试（只跑 CPU 快速测试，不跑耗时的端到端测试）

- `tests/v1/core/test_prefill_delayer.py`：
  - 逐条覆盖 `decide()` 的规则：保护步、未对齐时 HOLD、blocked 视为已对齐、unreachable + stall 放行、新到达重置 stall、fill、stall、各上界、`queue_hot`、reset；
  - 用真实的 `create_scheduler()` 测 `prefill_delayer_state()` 和 HOLD 门控（不使用 `SimpleNamespace` 模拟）。
- `tests/distributed/test_cpu_all_reduce.py` 的同类 gloo 多进程测试（新增于 `tests/v1/core/test_prefill_delayer.py` 或 `tests/distributed/`）：2–4 个 rank 上报不同的本地状态，经过 `sync_dp_state(extra=...)` 后，断言各 rank 的 `decide()` 序列完全一致，并且 `has_unfinished`、`pause` 语义不变。
- 配置校验：`cadence` 下设置 `max_wait_ms` 时报错；`delayer` 与 elastic EP / fault tolerance 同时开启时报错。
- 回归：`pytest tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py tests/engine/test_arg_utils.py tests/distributed/test_cpu_all_reduce.py`。
- `pre-commit run --files <改动文件>`（包括 ruff 和 mypy）。
- 自检：`git diff | grep -in atom` 必须无输出。

端到端的性能和精度测试暂时不跑，在 PR 描述中用 `<TODO>` 占位，列出后续要补的对比矩阵：
- interval ∈ {1, 8, 16, 32} 各自配 cadence，与 delayer 对比；
- 并发 c16 / c64 / c128 / c256；
- 指标：throughput vs E2E-normalized interactivity，以及 TTFT；
- 每组多次运行；
- 加一次 gsm8k smoke test。

## 关键文件

| 文件 | 改动 |
|---|---|
| `vllm/v1/core/sched/prefill_delayer.py` | 新增：状态定义和决策状态机 |
| `vllm/v1/core/sched/scheduler.py` | 新增 `prefill_delayer_state()`，累计 `prefill_ran` / `arrived`，`defer_prefills` 条件 |
| `vllm/v1/engine/core.py` | `DPEngineCoreProc`：构造 delayer、合并同步、`_should_throttle_prefills`、wave reset |
| `vllm/config/parallel.py` | `sync_dp_state(extra=None)` |
| `vllm/config/scheduler.py`、`vllm/engine/arg_utils.py` | 两个新字段和 CLI 参数，校验逻辑 |
| `docs/serving/data_parallel_deployment.md` | 新增小节 |
| `tests/v1/core/test_prefill_delayer.py` | 新增测试 |

可复用的现有代码：
- `cpu_all_reduce_`（`vllm/distributed/utils.py`）
- `ParallelConfig.sync_dp_state` 和 `cpu_distributed_timeout`
- scheduler 中已有的 `prefill_scheduled` / `prefill_admitted` / `deferred_waiting` / `prefill_capacity_bound`
- `tests/v1/core/utils.py` 中的 `create_scheduler` 和 `create_requests`
- `tests/distributed/test_cpu_all_reduce.py` 的 gloo 多进程测试模式

## PR 描述

- 路径：`/home/shashen/vllm/vllm-dev-skills/docs/tasks/prefill_delayer/dp_prefill_delayer_pr.md`（英文）。
- 用 `vllm-pr-desc-generator` 基于本地 diff 生成，此后每次代码更新都同步修改。
- 拟定标题：`[Core][DP] Add a cross-DP prefill delayer scheduling policy (--prefill-schedule-policy delayer)`。
- Details 中说明以下内容：
  - 接手 #61834，并说明与 #51199 的区别；
  - 为什么 cadence 不够用；
  - 为什么采用合并同步的方式；
  - 默认行为为什么不变；
  - CI 覆盖不到的部分及其原因。
- 正文中声明使用了 AI 辅助，并运行 `/pr-checklist`。
