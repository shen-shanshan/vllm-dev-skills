# 性能测试方案：DeepSeek-V4.1-Flash 不开 KV offloading vs UMBP standalone 各配置（AgentX TP2 c128）

> 状态：**B0 和 S1 已跑完（2026-11-02），结果见 §7.6**；是否继续由用户决定。制定日期 2026-11-02，同日按用户意见修订。
> - 已确认：synthetic AL 3.51、开启 rust 前端。
> - 用户决定：只测 c128，不测 c64 和 c192；只测 UMBP standalone，不测 embedded。
> 参照：DeepSeek-V4-Pro 的同类方案，见 [`umbp-perf-tuning.md`](umbp-perf-tuning.md) §8。
> 进度文档：[`PROGRESS-umbp-dsv41-flash-standalone.md`](PROGRESS-umbp-dsv41-flash-standalone.md)。
> 范围：只比较「不开 offloading」和「UMBP standalone 不同配置」，找出最优配置。Mooncake 的对比放到后面做，记为待办，见 §8。

## 1. 目标

在 MI355X 上，单机 TP2，GPU KV 不够用的高并发 agentic 场景（c128），回答两个问题：

1. 开 UMBP 比不开快多少？
2. UMBP standalone 用哪个配置最好？

## 2. 为什么选 c128

- recipe 的官方点是 c64，但 c64 下 GPU KV 足够用，不需要 offloading。
- 在 c64 下，同时在线的会话大约占 8–15M token，而 GPU KV 有 27.6M token。
- c128 下同时在线的会话大约占 15–29M token，正好处在 GPU 容量的临界点附近。
- AgentX 会保留每轮对话之间的间隔，单个间隔最长 300 s（`trace_idle_gap_cap_seconds=300`）。间隔期间，会话的前缀会闲置在 GPU 上，这些前缀就是 offloading 的收益来源。
- 代价：c128 不在 InferenceX 的测试矩阵里。
  - matrix 的注释说，TP2 c128 的分数没有超过 c64：每卡 150,683 vs 151,896 tok/s。也就是说，c64 时算力已经饱和，c128 下请求会排队，TTFT 会变长。
  - 因此本轮结果的含义是「过载时 UMBP 能带来多少收益」，不能直接和官方排行榜对比。
- c128 不会超出 cudagraph 的覆盖范围：capture 上限是 1024 token，按每个请求 decode 6 个 token 算，大约能覆盖 170 个请求。

## 3. 固定条件

除被测变量外，各个点的下列条件完全一样：

| 项 | 取值 |
|---|---|
| 代码 | `rocm-dsv41-flash-umbp` HEAD `3c14ef42b6`，base 是 main `51c6dfd8c8`。B0 用同一份代码，只是不加 `--kv-transfer-config` |
| 引擎参数 | `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml` 的 base 加 `override_tp2_c64` 的引擎参数，与 `umbp-dsv41-flash/run_vllm_dsv41_flash.sh` 现有参数一致。recipe 本身没有 TP2 c128 的点，所以只改 client 的 `CONC`，引擎参数不动。`max_num_seqs` 按 MI355X 的默认值 1024，不会限流 |
| GPU | 0、1，开跑前确认是空闲的 |
| DSpark | 使用 InferenceX 跑吞吐时的口径：`rejection_sample_method=synthetic`，`synthetic_acceptance_length=3.51`。这个值取自 `golden_al_distribution/dsv41flash_dspark.yaml`，对应 thinking_on、5 个 token。InferenceX 的 `infx/srt_slurm/synthetic_acceptance.py` 也是这样替换的。这样也能去掉真实接受率波动带来的噪声 |
| 前端 | `VLLM_USE_RUST_FRONTEND=1`，这是 recipe 里有、但现有脚本漏掉的。recipe 只起单个 vllm 实例，不用 router |
| AgentX client | `infx.bench agentic`，**`CONC=128`**，**`DURATION=3600`**。trace 用 uncapped 的 `semianalysis_cc_traces_weka_062126`，已缓存在 `HF_HOME=/home/shashen/vllm/hf_cache` |
| matrix 标记 | `TP=2 EP_SIZE=1 DP_ATTENTION=false SPEC_DECODING=mtp MODEL_PREFIX=dsv41flash`。UMBP 的点另外加 `KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=umbp TOTAL_CPU_DRAM_GB=<池大小>` |
| 部署方式 | 每个点都全新部署：UMBP server（如有）→ 引擎 → client → 快照引擎指标 → 清理 |
| UMBP | standalone 模式，MORI v1.2.3.post1 的 `umbp_standalone_server`，只用 DRAM，默认 128GiB。不开 hugepages，因为容器里 `HugePages_Total: 0`，`/proc/sys` 也是只读的。`UMBP_SSD_ENABLED=0` |
| 输出 | `RUNS_DIR=/tmp/umbp_runs/agentic_dsv41/<tag>`，放在本地盘上。跑完后把汇总结果拷到 `umbp-dsv41-flash/results/` |

## 4. 被测变量（相对 v4-Pro 方案的调整）

| 变量 | 处理 |
|---|---|
| `save_decode_cache` | **测**。agentic 的下一轮 prompt 里含上一轮的回答和 tool call |
| `lazy_offload` | **测**。块从 GPU 淘汰时才写入（write-back）。和 `save_decode_cache` 不叠加测，因为 lazy 路径本身就会存 decode 块，`save_decode_cache` 在 lazy 模式下不起作用 |
| 容量 | 默认 128GiB，和精度测试、以后的 Mooncake 对比保持一致。按每个 token 约 3KB 估算，standalone 去重后能存约 45M token，是 GPU 容量的 1.6 倍。是否加大要看 S1 有没有发生淘汰 |
| `dedup_tp_replicated_kv` | 保持默认开启（TP2 下只存一份），不做对照。只有 pool 写满时才考虑 |
| `rebuild_sliding_window` | 不适用：v4.1 在 ROCm 上没有 SWA replay |
| embedded 模式 | 用户决定不测 |
| `lookup_async`、`num_workers` | 不测。只有 S1 出现 scheduler 停顿、load 排队或超时，才补测 |

## 5. 测试点（共 5 次完整 AgentX 运行，每次约 1.3–1.5 h）

### 第 0 步：冒烟（不计入 5 次，约 1 h）

脚本改好后，用 `smoke_umbp.sh` 的流程验证：两轮对话 → `/reset_prefix_cache` → 恢复。只起一个 standalone 引擎，不同组合之间只需要重启引擎：

1. standalone，加上新增的 synthetic AL 和 rust 前端，其余为默认配置；
2. `save_decode_cache=true`；
3. `lazy_offload=true`。

后两项还没在 v4.1 上跑过。

通过标准与 PROGRESS 5.1 相同：
- external hit 大于 0；
- 没有 Traceback、NaN，日志里没有 "is ignored"；
- rust 前端下 `/metrics` 和 `/reset_prefix_cache` 都能用。

另外，用 `CONC=128` 跑一次很短的 agentic，几分钟即可，只用来验证脚本能跑通，不作为结果。

### 第 1 步：正式运行

| # | ID | 配置 | 理由 |
|---|---|---|---|
| 1 | **B0** | 不开 KV offloading | baseline |
| 2 | **S1** | UMBP standalone 128GiB，默认开关：eager store，`dedup_tp_replicated_kv=true`，`save_decode_cache=false`，`num_workers=4` | 默认配置，也作为后续各点的对照 |
| 3 | **S2** | S1 加 `save_decode_cache=true` | 预期收益最大：agentic 的下一轮会复用上一轮的回答 |
| 4 | **S3** | S1 加 `lazy_offload=true` | write-back，store 流量更少，和 prefill、decode 争用的带宽也更少 |
| 5 | 动态 | 按下面的规则选一个 | |

第 5 个点的选择规则，在第 1–4 个点跑完后判断：
1. 如果 S1 的 server 日志里出现淘汰，说明池子写满了，就测 **S1 + 512GiB**。server 启动需要几分钟。
2. 否则重跑 1–4 中最好的那个点，用来估计波动。

### B0 之后的检查点

B0 跑完后，我先汇报 GPU KV 占用的峰值、GPU prefix 命中率和 preemption 次数，然后：
- **如果 GPU KV 打满了**，或者发生了淘汰、preemption，就继续跑 S1–S3。
- **如果 c128 下仍然明显没满**，UMBP 预计和 B0 持平。这时暂停，由你决定下一步：
  - 照原计划跑完；
  - 或者改用 `--kv-cache-memory-bytes` 把 GPU KV 缩小后再测。

总耗时：冒烟约 1 h，加上 5 个点约 7–7.5 h，合计约 8–8.5 h。各点串行运行。

## 6. 指标与选择规则

- **主指标**：
  - 输出吞吐，以及每 GPU 的总吞吐（tok/s/GPU，InferenceX 打分的口径）；
  - TTFT mean / p50 / p90 / p99。
- **辅助指标**：
  - TPOT、interactivity p50；
  - 每个请求实际计算的 prefill token 数，即 `prompt_tokens − prefix 命中 − external 命中`；
  - `external_prefix_cache_hits / queries`，以及 GPU prefix 命中率；
  - `kv_cache_usage_perc` 随时间的变化，以及 preemption 次数；
  - UMBP server 日志中的淘汰情况，以及 connector 的 failed / recompute 计数。
- **选择规则**（与 v4-Pro 相同）：
  - 差异小于约 3% 视为持平。如果第 5 个点做了重跑，就以实测到的波动为准。
  - 吞吐和 TTFT p90 冲突时，优先看吞吐。
  - 性能相当时，优先选默认配置。

## 7. 需要新增或修改的脚本（都在 `umbp-dsv41-flash/` 下）

| 脚本 | 内容 |
|---|---|
| `run_vllm_dsv41_flash.sh` | 加 `VLLM_USE_RUST_FRONTEND=1`；默认加 `--no-enable-extensible-kv-cache`（见 7.5）；新增 `SYNTHETIC_AL` 开关，默认关闭，GSM8K 脚本的行为不变，性能测试时设为 3.51 |
| `run_agentic.sh`（新） | 改自 `umbp-dsv4-mtp/run_agentic.sh`：matrix 标记改成 TP2，`CONC` 默认 128，模型名改为 dsv41flash；client 直连引擎的 8333 端口，不经过 router |
| `run_config.sh`（新） | 改自 `umbp-dsv4-mtp/run_config.sh`：去掉 router；清理时只杀自己起的进程（用 `pkill -x` 或 PID，避免误杀当前 shell） |
| `compare_agentic.py` | 复用 `umbp-dsv4-mtp/compare_agentic.py`，按需要支持多列输出 |

命令（确认后执行）：

```bash
cd /home/shashen/vllm/vllm-dev-skills/docs/mori/umbp-dsv41-flash
export RUNS_DIR=/tmp/umbp_runs/agentic_dsv41 GPUS=0,1 SYNTHETIC_AL=3.51 CONC=128
export UMBP_DRAM_CAPACITY=$((128 * 1024**3))
TAG=b0_c128        UMBP=0 bash run_config.sh
TAG=s1_c128        UMBP=1 bash run_config.sh
TAG=s2_decode_c128 UMBP=1 UMBP_EXTRA=',"save_decode_cache":true' bash run_config.sh
TAG=s3_lazy_c128   UMBP=1 UMBP_EXTRA=',"lazy_offload":true' bash run_config.sh
```

## 7.5 冒烟结果（2026-11-02，等用户决定后再跑正式测试）

日志在 `/tmp/umbp_runs/agentic_dsv41/smoke/`，`/tmp` 下不长期保留。

| 项 | 结果 |
|---|---|
| 新 base 上 standalone 起不来 | ❌→✅ main 的 extensible KV cache（#56492，ROCm 默认开启）用 VMM 分配 KV，`hipIpcGetMemHandle` 报 invalid argument，UMBP server 无法映射。已在 `run_vllm_dsv41_flash.sh` 中对所有臂统一加 `--no-enable-extensible-kv-cache`（`EXTENSIBLE_KV_CACHE=1` 可恢复）。之后 GPU KV 为每 rank 67.59 GiB，29.4M token |
| S1 默认 | ✅ 恢复 24,320 token，无 Traceback、NaN、"is ignored" |
| S2 `save_decode_cache` | ✅ 恢复 24,320 token |
| S3 `lazy_offload` | ❌ 恢复 0 token。即使设 `lazy_offload_max_blocks=1000000` 扫描整个空闲队列，DEBUG 日志也显示每次扫描都是 `unhashed=free`、`candidates=0`，即空闲队列里看不到带 hash 的块，什么都没存。同样的配置在 Qwen3-0.6B 上正常（扫到 953 个块，恢复 15,248 token），所以是 lazy offload 与 DSv4.1 的 KV 管理之间的兼容问题，根因未查清 |
| rust 前端 | ✅ `/metrics`、`/reset_prefix_cache` 都可用 |
| synthetic AL | ✅ 平均接受长度 3.51。恢复前后的回答不再逐字一致，这是预期的 |
| c128 agentic（S1，DURATION=300） | ✅ 脚本全流程跑通，0/1016 失败，结果 JSON 和校验正常。warmup 约 20 分钟，所以每个 3600 s 的点约 1.4 h |

**c128 负载偏轻，UMBP 很可能测不出收益**：
- 这次 c128 跑了约 25 分钟，UMBP 的 external hit 为 **0**。GPU prefix 命中率 91.9%，TTFT mean 0.38 s，没有 preemption，稳态下 GPU KV 实际占用只有约 6%。
- 实测平均 ISL 为 68K（p90 117K），128 个会话的工作集约 9–13M token，GPU 有 29.4M token，装得下。
- 被淘汰的主要是已经结束的会话，这些前缀之后不会再被复用。

用户决定（2026-11-02）：
1. S3（lazy offload）先不测，记为待办。
2. 不用 `--kv-cache-memory-bytes` 限制 GPU KV，因为那样测出来的结果对真实场景没有意义。
3. 直接用 c128 跑完整的 AgentX workload（`DURATION=3600`），先只测 B0 和 S1。测完后记录结果，由用户决定是否继续。

## 7.6 正式结果：B0 vs S1（TP2，c128，DURATION=3600，2026-11-02）

- 代码：`rocm-dsv41-flash-umbp` `3c14ef42b6`。
- 两个点的参数完全相同：`override_tp2_c64` 的引擎参数，`CONC=128`，synthetic AL 3.51，rust 前端，`--no-enable-extensible-kv-cache`，GPU 0/1，不限制 GPU KV。
- S1 只多了 UMBP standalone 128GiB，开关全部用默认值。
- 每个点只跑了一次。

| 指标 | B0 | S1 | Δ |
|---|---|---|---|
| 请求数（计入统计的） | 14,657 | 14,745 | |
| 每 GPU 总吞吐（tok/s） | 245,339 | 246,816 | +0.6% |
| 输出吞吐（tok/s） | 4,097 | 4,120 | +0.6% |
| TTFT mean / p50 / p90 / p95（s） | 0.50 / 0.23 / 0.96 / 1.99 | 0.47 / 0.23 / 0.87 / 1.69 | −6% / 0 / −9% / −15% |
| TPOT mean / p90（ms） | 19.5 / 26.9 | 19.2 / 26.5 | −1.5% / −1.5% |
| E2E mean / p90（s） | 19.9 / 44.1 | 19.6 / 43.1 | −1.5% / −2.3% |
| interactivity p50（tok/s/user） | 55.14 | 55.38 | +0.4% |
| GPU prefix 命中（整个运行） | 94.7% | 94.8% | |
| UMBP 恢复（整个运行） | 0 | 2.84M token，占 prompt 的 0.14%，占实际需要计算部分的 2.7% | |
| 实际计算的 prompt 比例 | 5.3% | 5.1% | |
| 平均接受长度 | 3.51 | 3.51 | |
| preemption | 0 | 0 | |
| 失败请求 | 0 | 1（client 侧 `ClientOSError`，写请求体失败，与 UMBP 无关） | |

- 负载：完整运行中 ISL mean 为 120K、median 86K、p90 252K，OSL mean 为 1,014。
- UMBP server 没有发生淘汰，128GiB 的池没有写满。S1 的引擎日志里没有 Traceback，也没有 "is ignored"。

**结论**：
1. **在这个场景下，开 UMBP 基本没有收益，也没有额外开销。**
   - 吞吐 +0.6%，在单次测量的噪声范围内（约 3%）。
   - TTFT 的 p90 和 p95 有 9–15% 的下降。但 TTFT 本身已经很低（mean 0.5 s），又只跑了一次，还不能认定是真实收益。
2. **原因是 GPU prefix cache 本身已经接住了 94.7% 的 prompt token。**
   - GPU KV 有 29.4M token，c128 下同时在线的会话装得下。
   - 真正需要从 UMBP 取回的只有 0.14%。
3. 这种情况下再比较 UMBP 的各种配置（S2 等）意义不大，因为能恢复的量太小，配置之间的差异会被噪声淹没。
4. 想让 offloading 发挥作用，需要 GPU KV 真正装不下的场景，例如更高并发、更长的会话，或者显存更小的部署（TP 更小、单卡同时跑多个模型）。是否继续由用户决定。

原始结果在 `umbp-dsv41-flash/results/`：
- `agentic_c128_compare.md`
- `dsv41flash_vllm_tp2_c128_{b0,s1}_c128.json`
- `agentic_c128_engine_metrics_*.txt`
- `agentic_c128_workload_b0.txt`

完整产物在 `/tmp/umbp_runs/agentic_dsv41/{b0,s1}_c128/`，不长期保留。

## 8. 不在本轮范围内（待办）

- **UMBP vs Mooncake 的性能对比**：用户要求放到后面测。开始之前要先做两件事：
  - 给容器挂上 RDMA，或者在报告里写明 Mooncake 走的是 tcp；
  - **把两个分支对齐到同一个 base**。目前 UMBP 分支在 `51c6dfd8c8`，Mooncake 分支（`fe28d309f3`）在 `410ab341a8`。
- 以下各项视本轮观察再决定是否补测：hugepages、`lookup_async`、`num_workers`、关闭去重的对照。
- 用户决定不测：c64、c192、UMBP embedded 模式、TP4、用 `--kv-cache-memory-bytes` 压缩 GPU KV。
- lazy offload（S3）在 DSv4.1 上不生效：扫描时找不到带 hash 的空闲块。根因待查。
