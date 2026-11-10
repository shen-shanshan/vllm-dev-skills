# 调优计划：DSv4.1-Flash MI355X vLLM DP2 + prefill delayer 在 AgentX 上的最优配置

> 状态：v3，已确认。第 0～3 阶段已完成，2026-11-10 16:55 暂停；下次从第 4 阶段继续，进度见 `perf/tuning/README.md`
> 代码：vLLM 分支 `dp-prefill-delayer` `2921e6e5c2`。基于 main `feb4cb019f`，已包含"max-wait 可以打断保护期"的修复；`vllm-rs` 用同一份代码重新编译。
> 起点：leader 在 InferenceX PR [#4873](https://github.com/SemiAnalysisAI/InferenceX/pull/4873) 中提交的配置，以及他在 vllm#65655 review 中贴出的实测数据。
> 本计划替代 `perf/perf_plan.md` 的 4 点计划。

## 1. 已有结论（leader 的数据）

**#4873 提交的配置：**
- DP2（TP1 × DP2，MoE 在两个 rank 上做 TP 切分），c64 / c128 / c256；
- `--max-num-batched-tokens 16384`；
- `--prefill-schedule-policy delayer --prefill-schedule-interval 11 --prefill-max-wait-ms 5000`；
- 不开 monokernel，前面接 consistent-hash router。

**review 中的数据**（tok/s/GPU；interactivity = 1000 / ITL p90；括号内为默认调度器）：

| 配置 | 吞吐 | p90 interactivity | TTFT p90 / p99 | TTFT > 10 s |
|---|---:|---:|---|---:|
| DP2 c128，N11 W5000 16K，修复前 | 228K（218K） | — | 16.4 / 41.9 s | 20% |
| DP2 c128，N11 W2500 16K，修复后 | 227K（218K） | 33.0（28.8） | 2.9 / 7.6 s | 0.3% |
| DP2 c128，N11 W5000 16K，修复后 | 229K（218K） | 35.6（28.8） | 4.2 / 9.1 s | 0.6% |
| DP2 c256，N11 W2500 16K，修复后 | 283K（277K） | 14.5（14.5） | 6.1 / 26.8 s | 4.5% |

**他已经扫过的维度：**
- **token budget：** 8K 最差，32K 把 ITL 收益吐回去大半，16K 最好。
- **max-wait：** 只测了 2.5 s 和 5 s，W 越大 interactivity 越高，TTFT 越差。
- **interval：** 只用过 11。

**还没扫过的维度：**
1. interval 只测过 11 这一个值；
2. W 没测过 5 s 以上；
3. 16K 和 32K 之间的 budget 没测过；
4. c64 和 c256 直接套用了 c128 的参数；
5. 所有数据都是单次运行。

## 2. 缩短单次测试：使用 InferenceX 的快速模式

InferenceX 自带 `agentx-fast` 模式，CI 的 `e2e-tests.yml` 有对应选项。脚本里通过 `AIPERF_EXPERIMENTAL_FAST=1` 打开（`infx/bench/agentic/replay.py`），效果是：
- profile 阶段固定 1200 s；
- warmup 从默认的约 20 min 降到每条 lane 只发 1 个请求。

这样单个点从约 1.5 h 降到约 40 min，包括约 10 min 的引擎启动和关闭。

**为什么不用更短的时长（比如 600 s）：**
- AIPerf 规定的最短时长是 900 s，低于它要加 `--unsafe-override`；
- 快速模式是 InferenceX 官方支持的口径，不是我们自己定的。

**快速模式可能带来的偏差：**
1. warmup 很短，profile 刚开始时 prefix cache 还没到稳态，前一段的 prefill 会偏多；
2. 样本量只有 3600 s 时的 1/3，噪声更大。

所有点都在同样条件下测，所以排序仍然有意义，但需要对照 3600 s 的数据，确认能信到什么程度。

**做法：** 除最终确认外，全部用快速模式。不单独跑 3600 s 的校准点，改为前后各做一次对照：

1. **开头：** 用快速模式跑默认调度器和 leader 配置（F0、F1），把两者的差值和 leader 在 review 里给出的 3600 s 数据对照：吞吐 +4%，p90 interactivity 28.8 → 35.6（+24%），TTFT p90 1.2 → 4.2 s。方向和量级一致才继续筛选。机器和代码版本不完全相同，所以这一步只用来发现明显的偏差。
2. **结尾（第 4 阶段）：** 冠军配置和默认调度器各跑一次 3600 s，最终结论以这两个点为准；同时检查它们和快速模式下的排序是否一致。

代价是：如果快速模式的排序本身有偏差，要到最后才会发现。但最终推荐的配置一定经过 3600 s 验证，所以不会给出错误结论，最多是错过一个本来可能更好的配置。

## 3. 判断规则

| 类别 | 指标 |
|---|---|
| 主指标 | 每 GPU 总吞吐（tok/s/GPU）、p90 interactivity（1000 / ITL p90） |
| 辅助 | E2E-normalized interactivity、ITL p99、TTFT p50 / p90 / p99、TTFT > 10 s 的占比 |
| 健康 | 失败请求、preemption、GPU prefix 命中、两个 rank 的 delayer 决策统计 |

- **TTFT 门槛：** TTFT p90 ≤ 5 s，且 TTFT > 10 s 的请求 ≤ 1%。不满足的配置直接淘汰。c256 本身已经过载，门槛放宽到 p90 ≤ 2 × 默认调度器。
  - 快速模式下 TTFT 整体偏高，按第 0 阶段测得的偏移量折算（见 `perf/tuning/README.md`）：TTFT p90 ≤ 6 s，且 TTFT > 10 s 的请求 ≤ 2.5%。最终确认时仍用 3600 s 的门槛。
- **噪声：** 3600 s 下吞吐按 5%、interactivity 按 3% 算；快速模式下的样本只有 1/3，按 1.5 倍算，即吞吐 7.5%、interactivity 4.5%。
- **择优：** 在满足门槛的配置里，吞吐比当前最优低不超过噪声的，取 p90 interactivity 最高者。interactivity 也相差在噪声以内时，取吞吐高者。
- **进入第 4 阶段：** 快速模式下，冠军和第二名的差距如果在噪声以内，两个都进入 3600 s 的最终确认。

## 4. 固定条件

| 项 | 取值 |
|---|---|
| 硬件 | 2× MI355X，跑之前检查哪两张卡空闲（显存占用 > 2% 视为忙） |
| 引擎 | recipe `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml` 的 base 参数，加上 #4873 的 DP2 override：`--tensor-parallel-size 1 --data-parallel-size 2`、engram `cpu_offload`、`max-cudagraph-capture-size 1024`、不开 monokernel、DSpark 5 token（synthetic AL 3.51）、rust 前端 |
| 默认 budget | `--max-num-batched-tokens 16384` |
| 前端 | `vllm-router` 0.1.14，`consistent_hash` |
| 负载 | InferenceX `infx.bench agentic`，c128 是主调优点，**串行**跑 |
| 脚本 | 复用 `perf/` 下的 `run_vllm_dp2.sh`、`run_config.sh` 和 `compare.py`。`run_config.sh` 增加一个 `FAST=1` 开关，用来设置 `AIPERF_EXPERIMENTAL_FAST`。新增 `run_sweep.sh`，按列表逐点跑，失败即停 |

## 5. 测试点

**第 0 阶段：环境和对照（c128，快速，约 1.3 h）**

1. 用当前分支重新编译 `vllm-rs`，记录对应的 commit。
2. **卡死风险：** 上次在 `fd77011424` 上测时，开着 extensible KV 卡死过。如果 F0 或 F1 卡死，后续所有点都加 `--no-enable-extensible-kv-cache`，然后重跑。卡死问题本身另行处理。

| 点 | 配置 |
|---|---|
| F0 | 默认调度器 |
| F1 | leader 配置 N11 W5000 |

**第 1 阶段：interval N（c128，W=5000，快速，约 1.5 h）**

| 点 | N | 理由 |
|---|---:|---|
| F2 | 6 | 保护更少，看 TTFT 能好多少、interactivity 掉多少 |
| F3 | 16 | 保护更多，看 interactivity 能不能继续涨，TTFT 门槛会不会被打破 |
| （F1） | 11 | 已经测过 |

如果 16 仍然明显好于 11，再加 N=24（F3b）。

**第 2 阶段：max-wait W（c128，用最优的 N\*，快速，约 1.5 h）**

| 点 | W | 理由 |
|---|---:|---|
| F4 | 3000 | 看 TTFT 能不能明显变好，同时 interactivity 只掉一点 |
| F5 | 8000 | 5 s 以上从没测过，看 TTFT 门槛在哪里被打破 |

**第 3 阶段：budget（c128，(N\*, W\*)，快速，约 0.7 h）**

| 点 | 配置 | 理由 |
|---|---|---|
| F6 | 24K budget | 16K 和 32K 之间没测过。修复后 max-wait 已经能兜住 TTFT，更大的 budget 可能更划算 |

**第 4 阶段：最终确认（c128，3600 s，约 3～4.5 h）**

| 点 | 配置 |
|---|---|
| S0 | 默认调度器 |
| S1 | 冠军配置。如果冠军就是 leader 的配置，就跑 leader 配置 |
| S2（视情况） | 第二名，仅在它和冠军的差距在噪声以内时加跑 |

**第 5 阶段：按并发验证（快速，约 2.7 h）**

| 点 | 配置 | 并发 |
|---|---|---|
| F7 / F8 | 默认调度器 / c128 冠军 | c64 |
| F9 / F10 | 默认调度器 / c128 冠军 | c256 |
| F11（可选） | c256 不满足 TTFT 门槛时，试更保守的 W=2500 | c256 |

c256 已经过载，目标是"不比默认差"。最终可能会建议 c256 单独用更保守的参数，或者不开 delayer。

## 6. 总量与产出

| | 必跑 | 可选 |
|---|---|---|
| 3600 s 的点 | 2 个（S0、S1），约 3 h | S2 |
| 快速模式的点 | 11 个（F0～F10），约 7.5 h | F3b、F11 |
| **合计** | **约 10.5 h** | 全部可选点都跑约 13 h |

**执行方式：** 每跑完一个阶段，我先汇报结果以及下一阶段要用的 N\* / W\*，然后继续，不需要你逐点确认。如果出现卡死、异常，或者开头的对照偏差明显，立即停下来汇报。

**产出：**
- `perf/tuning/README.md`：每个点的汇总表、两次对照的结果、各阶段结论，以及每个并发点的推荐配置，可以直接用来更新 #4873 的 override；
- 原始结果备份到 `perf/tuning/results/`；
- 把最优结果和 baseline 的对比写进 vLLM PR 描述的 Performance 部分，并回复 Fangzhou-Ai。
