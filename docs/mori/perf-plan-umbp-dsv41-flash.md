# 性能测试方案：DeepSeek-V4.1-Flash 不开 KV offloading vs UMBP standalone 各配置（AgentX TP2 c128）

> 状态：TP2 c128（§7.6）、TP2 c192（§7.7）、DP2 c256（§9.6）的 B0 和 S1 都已完成；之后是否继续，由用户决定。
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

## 7.7 c192：B0 vs S1（TP2，c192，DURATION=3600）

- 条件与 §7.6 相同，只把 `CONC` 改成 192。
- 两个臂不是同一天跑的：B0 是 2026-11-02 跑的，S1 是 2026-11-03 跑的，中间容器重启过。不过两边的代码（`3c14ef42b6`）和 `vllm-rs`（`76c6f3f2a`）都相同，机器也是同一台。
- 和 c128 的结果不能严格对比，因为 c128 当时用的 `vllm-rs` 是 `e0d00e4b8`。
- c192 超出了 cudagraph 的覆盖范围（约 170 个请求），超出的步骤走 eager 模式。两个臂受到的影响相同。
- 每个臂只跑了一次。

| 指标 | B0 | S1（UMBP 128GiB，默认配置） | Δ |
|---|---|---|---|
| 请求数（计入统计的） | 11,300 | 15,289 | +35% |
| 每 GPU 总吞吐（tok/s） | 179,305 | 262,518 | **+46%** |
| 输出吞吐（tok/s） | 2,778 | 3,954 | **+42%** |
| TTFT mean / p50 / p90 / p95（s） | 7.12 / 2.45 / 19.46 / 25.75 | 1.59 / 0.42 / 4.18 / 8.19 | **−78% / −83% / −79% / −68%** |
| TPOT mean / p90（ms） | 58.0 / 88.5 | 37.0 / 54.2 | −36% / −39% |
| E2E mean / p90（s） | 59.9 / 135.0 | 36.3 / 81.5 | −39% / −40% |
| interactivity p50（tok/s/user） | 16.98 | 28.39 | +67% |
| GPU prefix 命中（整个运行） | 87.8% | 88.1% | |
| UMBP 恢复的 prompt（整个运行） | 0 | 4.8%（109.4M token） | |
| 实际计算的 prompt 比例 | 12.2% | 7.1% | −42% |
| 平均接受长度 | 3.51 | 3.51 | |
| preemption | 0 | 0 | |
| 失败请求 | 3 / 11,303 | 5 / 15,294 | 都是 client 侧的 `ClientOSError` |

- 负载（S1）：ISL mean 124K，median 88K，p90 258K；OSL mean 939。B0 为 114K / 78K / 252K / 892，两者接近。
- S1：
  - UMBP server 没有发生淘汰（日志里没有 eviction 记录），128GiB 池没有写满；
  - 引擎日志里没有 Traceback，也没有 "is ignored"；
  - S1 的引擎冷启动用了约 38 min，因为权重不在 page cache 里。这不影响测量。
- 第一次尝试（2026-11-02）在 warmup 阶段被 client 侧的 `ClientOSError` 终止。这次重跑时同样的错误只出现在 profiling 阶段，共 5 次，在允许范围内。

**结论**：
1. **c192 下 UMBP 收益显著。** 吞吐提升 46%，TTFT 的 mean 和 p90 都下降了约 80%，TPOT 下降 36%。
2. **原因：**
   - c192 时 GPU KV 装不下所有会话的前缀，GPU prefix 命中率只有 88%，需要实际计算的 prompt 达到 12.2%。这个负载下引擎是 prefill-bound 并且过载。
   - UMBP 恢复了 4.8% 的 prompt token，使实际计算的 prefill 减少了 42%。
   - 请求排队随之大幅缓解。AgentX 是闭环负载，前一轮完成得越快，下一轮就发得越早，所以吞吐也跟着上升。
3. **和 c128 对比：** c128 时 GPU KV 足够用，UMBP 只恢复了 0.14%，收益接近 0。所以 offloading 的价值取决于工作集是否超过 GPU KV 容量。
4. **注意事项：**
   - 每个臂只跑了一次，而且两个臂跨了容器重启。但 +46% 远大于约 3% 的噪声阈值，这个结论是可靠的。
   - B0 在 c192 下的吞吐比 c128 还低，说明 c192 对不开 offloading 的部署已经过载；开 UMBP 后，c192 的吞吐反而超过了 c128。

结果文件：`umbp-dsv41-flash/results/` 下的
- `agentic_c192_compare.md`
- `dsv41flash_vllm_tp2_c192_{b0,s1}_c192.json`
- `agentic_c192_engine_metrics_{b0,s1}_c192.txt`
- `agentic_c192_workload_{b0,s1}.txt`

## 8. 不在本轮范围内（待办）

- **UMBP vs Mooncake 的性能对比**：用户要求放到后面测。开始之前要先做两件事：
  - 给容器挂上 RDMA，或者在报告里写明 Mooncake 走的是 tcp；
  - **把两个分支对齐到同一个 base**。目前 UMBP 分支在 `51c6dfd8c8`，Mooncake 分支（`fe28d309f3`）在 `410ab341a8`。
- 以下各项视本轮观察再决定是否补测：hugepages、`lookup_async`、`num_workers`、关闭去重的对照。
- 用户决定不测：c64、c192、UMBP embedded 模式、TP4、用 `--kv-cache-memory-bytes` 压缩 GPU KV。
- lazy offload（S3）在 DSv4.1 上不生效：扫描时找不到带 hash 的空闲块。根因待查。

## 9. 2 卡 attention DP2 + expert TP2，AgentX c256 的 B0 vs S1（2026-11-03，已完成，结果见 9.6）

### 9.1 部署形态

| 项 | 取值 | 说明 |
|---|---|---|
| 并行 | `--tensor-parallel-size 1 --data-parallel-size 2`，**不加** `--enable-expert-parallel` | attention 按 DP2 切：每张卡一个 DP rank，各自持有完整的 attention 和 KV。vLLM 在未开 EP 时，MoE 用 DP×TP=2 做 TP 切分，也就是 expert TP2 |
| 代码 | `rocm-dsv41-flash-umbp` `3be8e20c17`，base 是 main `77d2790e78` | 这个 base 还没跑过 E2E，所以冒烟要多验一些项（见 9.4） |
| 引擎参数 | 与 TP2 的点相同：block 128、mxfp4 indexer、DSpark 5（synthetic AL 3.51）、`max-num-batched-tokens 16384`（每个 rank）、`max-cudagraph-capture-size 1024`、`gpu-memory-utilization 0.9`、engram `cpu_offload=true`、`--no-enable-extensible-kv-cache`、rust 前端 | 例外见下一行 |
| **去掉 `--monokernel`** | 两个臂都不加 | `mono_decode.py` 要求 TP 为 2/4/8，且不能有 DP 或 EP，显式打开但跑不了时会直接报错。所以 DP2 下 decode 走常规路径，这和 TP2 recipe 不同 |
| engram | `{"cpu_offload":true}`，`embedding_across_dp` 保持默认 false | 每个 DP rank 各有一份 TP 切分的 embedding；`dp_shared_memory` 默认开启，同一节点上的 host 表共用一份 |
| 前端 | `vllm-router==0.1.14`，`--policy consistent_hash --intra-node-data-parallel-size 2`，client 带 `X-Session-ID` | 写法与 InferenceX 的 DEP recipe（B200 的 `zip_override_dep2`）以及 v4-Pro 的 DEP8 测试一致。同一个会话固定落在同一个 DP rank 上。rust 前端支持 router 发来的 `X-data-parallel-rank` header（`rust/src/server/src/utils.rs`），冒烟时会验证 |
| client | `CONC=256`，`DURATION=3600`，`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1`；matrix 标记为 `TP=2 EP_SIZE=1 DP_ATTENTION=true` | 平均每个 DP rank 128 个会话 |
| UMBP（S1） | standalone 128GiB，默认开关，两个 DP rank 共用一个 server | TP1 下没有 TP 复制，去重不起作用。key 里不含 DP 编号，所以一个 rank 存进去的前缀，另一个 rank 也能读到；但 router 让会话固定在同一个 rank 上，这种跨 rank 复用会很少 |
| GPU | 0、1 | |

### 9.2 预期与风险

- **KV 压力可能不够**：
  - DP2 下每个 rank 都要存完整的 MLA latent。TP2 下 latent 本来也是每个 rank 复制一份，所以每个 rank 每个 token 的 KV 占用和 TP2 差不多，而两个 rank 存的是不同会话，总容量大约翻倍。
  - c256 平均到每个 rank 是 128 个会话，KV 压力和 TP2 c128 差不多，而 TP2 c128 时 GPU KV 是够用的。
  - 另一方面，attention 不再切分，每个 rank 的 prefill 算力更紧张，而且 MoE TP2 要求两个 rank 同步 step，一个 rank 做 prefill 会拖慢另一个 rank 的 decode。
  - 所以 UMBP 的收益可能和 TP2 c128 一样接近 0，也可能因为过载而像 c192 那样明显，要靠实测确定。按用户要求，B0 和 S1 都会跑，不设中途检查点。B0 跑完后我会先汇报 GPU prefix 命中率。
- **DP2 路径在 ROCm 上没跑过**：aiter MoE 在 DP2 不开 EP 的情况、DSpark 加 DP、mxfp4 indexer 加 DP 都没验证过，新 base 也还没跑过 E2E。这些都放在冒烟里检查。
- **cudagraph**：每个 rank 128 个会话 × 6 token = 768，在 capture 上限 1024 以内。
- 结果**不能和 TP2 的点直接比**：部署形态、有没有 monokernel、base 都不同。只比较 DP2 下的 B0 和 S1。

### 9.3 测试点（2 次完整运行，每次约 1.5 h）

| # | ID | 配置 |
|---|---|---|
| 1 | **B0-dp2** | DP2 + expert TP2，不开 KV offloading |
| 2 | **S1-dp2** | B0-dp2 加 UMBP standalone 128GiB，默认开关 |

### 9.4 冒烟（不计入 2 次，约 1 h）

1. **起 B0-dp2 引擎和 router**，检查以下几项：
   - 两个 DP rank 都就绪；
   - 日志里没有 monokernel 报错，也没有 "is ignored"；
   - 通过 router 发几条带不同 `X-Session-ID` 的请求，确认两个 rank 都分到了请求，同一个 session 始终落在同一个 rank 上（看各 rank 的 `vllm:prompt_tokens_total`）。
2. **S1-dp2 恢复检查**：
   - 先通过 router 发一段约 20K token 的两轮对话，然后 `/reset_prefix_cache`（直接打到引擎端口，所有 DP rank 都会被重置），再发一次第二轮；
   - 期望 external hit > 0，没有 Traceback 或 NaN；
   - 同时确认新 base 上 standalone 仍然需要 `--no-enable-extensible-kv-cache`。如果 main 已经修好了，两个臂都恢复默认值，并在结果里注明。
3. **c256 的短 agentic**（S1，DURATION=300）：确认 router、session header、结果 JSON 整个流程能跑通。

### 9.5 需要修改的脚本（`umbp-dsv41-flash/`）

| 脚本 | 改动 |
|---|---|
| `run_vllm_dsv41_flash.sh` | 新增 `DP` 变量，默认 1。`DP=2` 时用 `--tensor-parallel-size 1 --data-parallel-size 2`，并去掉 `--monokernel`。`DP=1` 时行为不变 |
| `run_config.sh` | `DP>1` 时，引擎就绪后启动 router（复用 `../umbp-dsv4-mtp/run_router.sh`，新增参数控制 `--intra-node-data-parallel-size`），client 改为连 router 的 30000 端口；清理时按 PID 关掉 router |
| `run_agentic.sh` | `DP>1` 时设置 `DP_ATTENTION=true`、`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1`、`AIPERF_SERVER_URL=http://localhost:30000`；engine metrics 仍从 8333 端口抓取 |

命令（确认后执行）：

```bash
cd /home/shashen/vllm/vllm-dev-skills/docs/mori/umbp-dsv41-flash
export GPUS=0,1 DP=2 SYNTHETIC_AL=3.51 CONC=256 DURATION=3600 RUNS_DIR=/tmp/umbp_runs/agentic_dsv41
export UMBP_DRAM_CAPACITY=$((128 * 1024**3))
TAG=b0_dp2_c256 UMBP=0 bash run_config.sh
TAG=s1_dp2_c256 UMBP=1 bash run_config.sh
```

总耗时：冒烟约 1 h，2 个点约 3 h，合计约 4 h。

指标和判断规则与 §6 相同。另外分别统计两个 DP rank 的 prefix 命中和 UMBP 命中，看负载是否均衡。

### 9.6 结果（2026-11-03）

**冒烟**全部通过：
- S1 在 DP2 下从 UMBP 恢复了 24,320 token，日志里没有 monokernel 报错，也没有 Traceback 或 NaN。每个 rank 的 GPU KV 为 26.8M token。
- c256 跑了 300 s：1,424 个请求全部成功，两个 rank 的 prompt token 为 54.2M / 53.0M，prefix 命中 93%。这说明 router 的会话亲和生效，rust 前端也认 `X-data-parallel-rank`。
- extensible KV cache：新 main 在 ROCm 上会对需要注册 KV 内存的 connector（包括 UMBP）**自动关闭** extensible KV（`vllm/config/vllm.py` 的 `_resolve_extensible_kv_cache`），所以 S1 默认就能起来。但不带 connector 的 B0 仍会开启 extensible KV，导致 KV 容量和 S1 不一样，所以两个臂照旧都显式加了 `--no-enable-extensible-kv-cache`。
- 注意：AIPerf 同时抓了 router（30000）和引擎（8333）的 `/metrics`，而 router 只是把引擎的指标转发过来，所以服务端指标每项都会出现两份。比例类指标不受影响；吞吐和延迟是 client 侧测的，也不受影响。

**正式运行**的条件：
- 代码 `3be8e20c17`，base 是 main `77d2790e78`，`vllm-rs` 为 `40941f2c4`；
- GPU 0/1，两个臂在同一天连续跑完；
- 每个臂只跑了一次。

| 指标 | B0-dp2 | S1-dp2（UMBP 128GiB，默认开关） | Δ |
|---|---|---|---|
| 请求数（计入统计的） | 16,344 | 16,846 | +3.1% |
| 每 GPU 总吞吐（tok/s） | 270,323 | 280,607 | **+3.8%** |
| 输出吞吐（tok/s） | 4,053 | 4,194 | +3.5% |
| TTFT mean / p50 / p90 / p95（s） | 1.69 / 0.62 / 2.66 / 5.36 | 1.40 / 0.59 / 2.47 / 4.83 | −17% / −5% / −7% / −10% |
| TPOT mean / p90（ms） | 54.8 / 77.0 | 52.4 / 72.2 | −4.4% / −6.2% |
| E2E mean / p90（s） | 50.2 / 115.6 | 48.0 / 109.7 | −4.4% / −5.1% |
| interactivity p50（tok/s/user） | 19.36 | 19.75 | +2.0% |
| GPU prefix 命中（整个运行） | 92.9% | 93.1% | |
| UMBP 恢复的 prompt token（整个运行） | 0 | 0.2%（5.5M token） | |
| 实际计算的 prompt 比例 | 7.1% | 6.7% | −6% |
| 两个 rank 的 prompt token 占比（profiling 阶段） | 48.7% / 51.3% | 51.5% / 48.5% | 负载均衡 |
| preemption / 失败请求 | 0 / 0 | 0 / 0 | |

- 负载：ISL mean 119K / 120K，p90 249K / 250K；OSL mean 900 / 904。两个臂的负载基本一样。
- S1：UMBP server 没有发生淘汰，128GiB 池没有写满；引擎日志里没有 Traceback，也没有 "is ignored"。
- 两个 DP rank 都从 UMBP 恢复了 KV：rank 0 为 2.24M token，rank 1 为 1.95M token（profiling 阶段）。

**结论**：
1. **DP2 c256 下 UMBP 收益不大，但方向一致。**
   - 吞吐 +3.8%，刚好超过约 3% 的单次噪声阈值。
   - TTFT mean −17%，p90 −7%；TPOT 和 E2E 各降约 4–6%。
   - 每个臂只跑了一次，吞吐上的差异还需要重复测量才能确认。
2. **GPU KV 压力介于 TP2 c128 和 TP2 c192 之间**：GPU prefix 命中 92.9%，TP2 c128 是 94.7%，c192 是 87.8%。
   - DP2 下两个 rank 各存不同的会话，总 KV 约 53.7M token，大约是 TP2 的 1.8 倍。c256 平均到每个 rank 只有 128 个会话，所以前缀大多还在 GPU 上。
   - UMBP 只恢复了 0.2% 的 prompt，把实际计算的 prefill 减少了 6%，收益和这个比例相符。
3. 这个部署形态和 TP2 不能直接比较：没有 monokernel，base 和 `vllm-rs` 也不同。只看形态内部的对比：DP2 c256 的 B0 吞吐（270K）和 TP2 c128 的 B0（245K，旧 base）在同一个量级，TPOT 高很多（54.8 ms vs 19.5 ms）。
4. 想让 DP2 下的 offloading 发挥更大作用，需要更高的并发，或者更长的会话，让每个 rank 的工作集超过 26.8M token。是否继续由用户决定。

结果文件在 `umbp-dsv41-flash/results/`：
- `agentic_dp2_c256_compare.md`
- `dsv41flash_vllm_dp2_c256_{b0,s1}_dp2_c256.json`
- `agentic_dp2_c256_engine_metrics_*.txt`
- `agentic_dp2_c256_workload_{b0,s1}.txt`

完整产物在 `/tmp/umbp_runs/agentic_dsv41/{b0,s1}_dp2_c256/`，不长期保留。
