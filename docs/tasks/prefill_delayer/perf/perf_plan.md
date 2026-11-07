# 性能测试方案：prefill delayer 在 DSv4.1-Flash DP2 + expert TP2、AgentX c128 下的端到端表现

> 状态：已确认（2026-11-06），执行中。
> 代码：分支 `dp-prefill-delayer` 的 `b25a7dad68`（base 为 main `fd77011424`；HOLD 原因已改名为 `hold_*`）
> 执行方式：只用 GPU 0,1，串行运行，避免互相干扰。

## 1. 目标

1. **固定一个 interval N，比较两种调度方式：** cadence 和 delayer 用同一个 N，相对 baseline 各自在吞吐、TTFT、ITL 上得失多少。
2. **验证时间上界的作用：** 给 delayer 加上 `--prefill-max-wait-ms` 后，能否把 TTFT 的代价压到可以接受的范围，同时保住 decode 侧的收益。同事反馈当前配置拿 prefill 换 decode 换得太狠。
3. 只测默认的 16384 chunk。8K chunk 的交叉验证和重复测量这次都不做。

### 为什么 delayer 会让 TTFT 明显变差（推测，用测试结果来检验）

在当前实现里，每个 prefill chunk 之后都会进入保护期，要等 N−1 个纯 decode 步才能跑下一个 chunk。所以一个长 prompt 的 TTFT 大约会增加：

```
额外 TTFT ≈ chunk 数 × (N − 1) × TPOT  +  每个 chunk 等待对齐和填满的时间
```

AgentX 的 GPU prefix 命中约 93%，大部分轮次只需要算 1～2 个 chunk，但缓存没命中的长 prompt（约 120K token）需要 8 个 chunk。chunk 越小，chunk 数越多，额外的 TTFT 也越多。这也是同事说"chunk 调得太狠会更糟"的原因。

## 2. 固定条件

| 项 | 取值 |
|---|---|
| 模型 | DeepSeek-V4.1-Flash |
| 部署 | 2× MI355X（GPU 0,1）。`--tensor-parallel-size 1 --data-parallel-size 2`，不开 EP，MoE 在两个 rank 上做 TP 切分。不加 `--monokernel`，因为 mono decode 不支持 DP |
| 引擎参数 | recipe `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml` 的 base：block 128、mxfp4 indexer、DSpark 5 token（synthetic AL 3.51）、`gpu-memory-utilization 0.9`、engram `cpu_offload=true`、`max-cudagraph-capture-size 1024`、**`max-num-batched-tokens 16384`**、rust 前端 |
| 前端 | `vllm-router` 0.1.14，`consistent_hash`，`--intra-node-data-parallel-size 2`；client 带 `X-Session-ID` |
| 负载 | InferenceX `infx.bench agentic`，CONC=128，DURATION=3600，warmup 约 20 分钟。每个点约 1.5 h |
| 日志 | 不开全局 DEBUG。用 `VLLM_LOGGING_CONFIG_PATH` 只把 `vllm.v1.core.sched.prefill_delayer` 设为 DEBUG，用来拿决策统计 |

## 3. 测试点（4 个）

**N = 4。** 选择理由：
- 参考引擎在 TP2 c64 上调出来的值就是 4，而 TP2 c64 和这次的 DP2 c128 一样，每张卡或每个 rank 上大约 64 个会话；
- 同事已经觉得 TTFT 的代价太大，所以取偏保守的小值；
- #61834 用的 10 和 32 都偏激进。

| # | ID | 参数 | 作用 |
|---|---|---|---|
| P0 | `base` | 默认，相当于 `--prefill-schedule-interval 1` | **baseline**：既不开 cadence 也不开 delayer |
| P1 | `cad4` | `--prefill-schedule-interval 4` | cadence：每 4 步才允许一次 prefill |
| P2 | `del4` | `--prefill-schedule-policy delayer --prefill-schedule-interval 4` | delayer，用同一个 N |
| P3 | `del4_wait` | P2 再加 `--prefill-max-wait-ms W` | delayer 加上时间上界 |

**W 的取值规则：** 等于 P0 的 TTFT p90，四舍五入到 500 ms。含义是不让排队时间超过 baseline 的 p90。P0 跑完后定下这个值，并记录在结果里。

运行顺序：P0 → P1 → P2 → P3。P3 依赖 P0 的结果。

## 4. 冒烟（约 1 h）

1. 在当前 base 上重新编译 `vllm-rs`，现在的二进制是 11/05 编译的，并记录它对应的 commit。
2. 用 P2 的配置起引擎和 router，检查以下几项：
   - 两个 rank 都就绪，没有 Traceback；
   - 参数被正确解析；
   - 局部 DEBUG 生效：能看到 `Prefill delayer decisions`，而全局没有 DEBUG 刷屏。
3. 用 DURATION=300 跑一次短的 agentic，确认：
   - 结果 JSON 能生成；
   - 逐请求记录能用来计算 E2E-normalized interactivity。

## 5. 指标与判断规则

| 类别 | 指标 |
|---|---|
| 吞吐 | 每卡输出 tok/s、每卡总 tok/s |
| prefill 侧 | TTFT mean / p50 / p90 / p99 |
| decode 侧 | TPOT / ITL p50 / p90，interactivity p50 / p90（只看 ITL） |
| 端到端 | E2E latency mean / p90，E2E-normalized interactivity p50 / p90，即 输出 token 数 / E2E latency，按请求计算 |
| delayer | 各原因的 FIRE / HOLD 次数（`hold_*` 和其余 FIRE 原因分开统计），两个 rank 的统计是否一致 |
| 健康 | GPU prefix 命中、preemption、失败请求、两个 rank 的负载占比 |

**判断规则：**
- 单次运行的吞吐噪声约 3%，差异小于 3% 视为持平。
- 一个配置要算"**可用**"，需要同时满足两点：
  - 吞吐或 E2E-normalized interactivity 至少有一项优于 P0，且超过噪声；
  - TTFT p90 不超过 P0 的 1.5 倍。

## 6. 脚本（`docs/tasks/prefill_delayer/perf/`）

复用 `docs/mori/umbp-dsv41-flash/` 中已经跑通的 DP2 流程：

| 脚本 | 作用 |
|---|---|
| `run_vllm_dp2.sh` | 启动引擎，可配置 `POLICY` / `INTERVAL` / `MAX_WAIT_MS` / `SYNTHETIC_AL`，并加载局部 DEBUG 的 logging config |
| `run_config.sh` | 一个点的完整流程：引擎 → router → AgentX → 抓取 metrics → 关闭。关闭时按进程组或 PID 结束进程 |
| `compare.py` | 生成汇总表，包括 TTFT p99、E2E-normalized interactivity 和 delayer 统计 |
| `logging_delayer_debug.json` | 只给 delayer 模块开 DEBUG 的 logging 配置 |

## 7. 产出

- `perf/README.md`：汇总表和结论，推荐参数以及适用的负载。
- 更新 PR 描述，替换 Claims 和 Validation 里的性能 TODO。
- 原始结果备份到 `perf/results/`。

总耗时：冒烟约 1 h，加 4 个点约 6 h，**合计约 7 h**。
