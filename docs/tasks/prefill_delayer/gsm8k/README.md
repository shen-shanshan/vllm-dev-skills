# prefill delayer GSM8K smoke test

> 日期：2026-11-06 | 分支：`dp-prefill-delayer`（`5cc5e5abf5`，基于 main `fd77011424`）
>
> 包含两部分：Model Runner V2（带 DSpark MTP）和 Model Runner V1（不开投机解码）。

## 配置

- 模型：DeepSeek-V4.1-Flash，2× MI355X（GPU 0,1）。
- 并行方式：attention DP2，MoE 在两个 rank 上做 TP 切分（`--data-parallel-size 2 --tensor-parallel-size 1`）。
- 投机解码：DSpark MTP，5 个 draft token。
- 默认开启 async scheduling 和 Model Runner V2，这正是 #61834 当初卡死的组合。
- 评测：lm_eval `local-completions`，GSM8K 5-shot，共 1319 题，并发 64。用的是本地 task `../../../mori/umbp-dsv41-flash/gsm8k_task`。
- 两组只在 prefill 调度参数上不同，其余参数完全一致，并且都开了 `VLLM_LOGGING_LEVEL=DEBUG`。

```bash
TAG=cadence POLICY=cadence bash run_gsm8k_config.sh
TAG=delayer POLICY=delayer INTERVAL=10 MAX_WAIT_MS=5000 bash run_gsm8k_config.sh
```

## 结果：Model Runner V2（DSpark MTP，GPU 0,1）

| 策略 | 参数 | strict-match | flexible-extract | 两个 rank 合计的平均 waiting / running |
|---|---|---:|---:|---:|
| cadence（默认） | — | 0.9303 ± 0.0070 | 0.9303 | 2.5 / 60.5 |
| delayer | `--prefill-schedule-interval 10 --prefill-max-wait-ms 5000` | 0.9242 ± 0.0073 | 0.9242 | 15.9 / 45.9 |

- **精度：** 两组相差 0.006，在一个标准误以内。同一模型、同一节点此前的 GSM8K 基线是 0.9204～0.9212。结论是没有精度损失。
- **稳定性：** 两组都没有出现 Traceback、ERROR 或卡死。delayer 组在"async scheduling + V2 runner + MTP + DP"组合下完整跑完。
- **重复验证：** 卡死出现在 V1 之后，又在 V2 上用 GPU 6,7 把 delayer 重复跑了 3 次，配置不变。结果分别是 0.9249 ± 0.0073、0.9409 ± 0.0065、0.9303 ± 0.0070，全部正常完成，推理阶段没有出现等待超时。V2 上的 delayer 共 4 次全部通过，平均 0.9301，和 cadence 的 0.9303 一致。结果在 `results/delayer_v2_run{2,3,4}/`。
- **delayer 确实在工作：** waiting / running 取自 DP coordinator 收到的 `[waiting, running, kv_usage]`，即 ApiServer_0 的 DEBUG 日志 `Received counts`。delayer 组的平均 waiting 是 cadence 组的 6 倍多，说明 prefill 确实被暂缓了。
- **日志里为什么没有 `Prefill delayer decisions`：** 这条统计每 1000 次决策才输出一次，而整个评测只持续了约 50 秒，引擎总步数不到 1000，所以一次也没触发。

## 结果：Model Runner V1（不开投机解码，GPU 6,7）

V1 的配置与 V2 部分相同，只有三点不同：
- 设置了 `VLLM_USE_V2_MODEL_RUNNER=0`；
- 设置了 `SPEC_CONFIG=none`，即不开投机解码；
- 改用 GPU 6,7，因为 GPU 1–4 被其他容器占用。

V1 上没法开投机解码：DSpark 只在 V2 上实现（配置校验报 `Model Runner V1 does not support: dspark speculative decoding`），而 V4.1 的 checkpoint 里也没有经典 MTP 的权重（`method: "mtp"` 被拒绝，`mtp.*` 下存的是 DSpark 的各个阶段）。所以"投机解码 + delayer"这个组合只在 V2 上验证过。

```bash
GPUS=6,7 VLLM_USE_V2_MODEL_RUNNER=0 SPEC_CONFIG=none TAG=delayer_v1 POLICY=delayer INTERVAL=10 MAX_WAIT_MS=5000 bash run_gsm8k_config.sh
GPUS=6,7 VLLM_USE_V2_MODEL_RUNNER=0 SPEC_CONFIG=none TAG=cadence_v1 POLICY=cadence bash run_gsm8k_config.sh
```

| 运行 | 策略 | strict-match | flexible-extract | 两 rank 合计的平均 waiting / running | 结果 |
|---|---|---:|---:|---:|---|
| delayer_v1 | delayer（interval 10，max_wait 5000 ms） | 0.9242 ± 0.0073 | 0.9242 | 15.5 / 46.8 | 正常完成 |
| delayer_v1_run2 | 同上 | — | — | — | **卡死**，详见下文 |
| cadence_v1 | cadence（默认） | 0.8984 ± 0.0083 | 0.8976 | 1.2 / 61.6 | 正常完成 |
| cadence_v1_run2 | 同上 | 0.9030 ± 0.0082 | 0.9030 | 1.2 / 61.9 | 正常完成 |

- **精度：** delayer 那次正常完成的结果比两次 cadence 高约 0.02～0.026。V1 cadence 两次都在 0.898～0.903，低于 V2 的 0.9303 和此前的基线 0.9204～0.9212。cadence 路径的代码与 main 逐字节相同，所以这个偏低跟本 PR 无关。一个推测：delayer 减少了 prefill 和 decode 混合的步，因此在 V1 上反而更高，但这还没有验证。结论是 delayer 没有带来精度损失。
- **决策统计：** V1 没开投机解码，步数更多，这次日志里出现了 `Prefill delayer decisions`。最后一次统计为 `{'fill': 1669, 'protected': 257, 'no_prefill': 54, 'align': 14, 'stall': 4, 'max_wait': 2}`。**DP0 和 DP1 输出的统计完全相同**，说明两个 rank 的决策始终保持一致。需要注意，`fill` 同时用于 FIRE（填满后放行）和 HOLD（没填满继续等），这份统计里分不开两者各占多少，见末尾的待办。

### 已知问题（仅 V1，不处理）：delayer_v1_run2 卡死

- **现象：** 引擎在 09:29:50 启动完成，lm_eval 只完成了 3 个请求就卡住。之后两个 rank 上报的计数一直没变，DP0 为 2 个 waiting、30 个 running，DP1 为 1 个 waiting、31 个 running。两个 EngineCore 每 60 秒报一次 `No available shared memory broadcast block found`，09:34:55 因 `execute_model` 超时（`TimeoutError`）崩溃退出。
- **py-spy 调用栈：** 两个 worker 的主线程都空闲，在 `worker_busy_loop` 里等新任务；`WorkerAsyncOutputCopy` 线程都卡在 `Event.synchronize`，也就是 GPU 上已经下发的工作一直没有完成。这符合 DP rank 之间 GPU 集合通信不匹配导致死锁的特征。EngineCore 已经退出，没来得及抓到它的栈。
- **崩溃前转储的 SchedulerOutput：** 两个 rank 都在执行纯 decode 步，DP0 调度了 30 个 token，DP1 调度了 31 个，阶段一致。死锁更可能源于此前两个 rank 累计执行的前向次数已经不一致，比如其中一个 rank 多跑或少跑了一次 dummy batch。
- **目前能确定的：** 同样配置的 delayer V1 第一次正常跑完，cadence V1 两次正常，V2 上的 delayer 和 cadence 也都正常。只出现 1 次，还无法判断是 delayer 引入的问题，还是 V1 + DP + async scheduling 本身偶发的问题。
- **证据：** `results/delayer_v1_run2_hang/engine.log.gz`、`eval.log.gz`。
- **结论：** 同样配置在 V2 上的 delayer 连续 4 次都正常（见上文），卡死只在 V1 上出现过。vLLM 正在逐步迁移到 Model Runner V2，所以按决定（2026-11-06）不再排查这个仅 V1 出现的问题，只记录在案。

## 文件

- `run_vllm_dp2.sh`：启动服务，`POLICY` / `INTERVAL` / `MAX_WAIT_MS` / `SPEC_CONFIG`（设为 `none` 表示不开投机解码）/ `GPUS` 都可以配置。
- `run_gsm8k_config.sh`：依次完成启动引擎、跑 lm_eval、关闭引擎。
- `results/<tag>/`：每次运行的 lm_eval 输出（`eval.log`、`results_*.json`）。tag 包括 `cadence`、`delayer`、`delayer_v2_run{2,3,4}`（V2），以及 `cadence_v1`、`cadence_v1_run2`、`delayer_v1`（V1）。`delayer_v1_run2_hang/` 是卡死那次的引擎日志和 eval 日志。引擎日志在 `/tmp/prefill_delayer_runs/gsm8k/<tag>/engine.log`，容器重启后会丢失。

## 待办

- ~~让决策统计能分清 FIRE 和 HOLD~~：已在 `b25a7dad68` 完成，HOLD 的原因改为 `hold_protected` / `hold_align` / `hold_fill`。上文的统计是改名之前的，所以其中 `fill` 混合了 FIRE 和 HOLD。
