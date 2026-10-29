# UMBP 性能调优：配置与方法

> 记录 MORI-UMBP 在 DeepSeek-V4-Pro / MI355X 上的官方推荐配置、可调参数与调优方法。
> 与 vLLM PR #63002 的进度见 [PROGRESS-umbp-dsv4-standalone.md](PROGRESS-umbp-dsv4-standalone.md)，
> 测试结果见 [pr-umbp-dsv4-standalone.md](pr-umbp-dsv4-standalone.md)。
>
> 最后更新：2026-10-29

## 0. 信息来源与版本

| 来源 | 版本 / 位置 | 说明 |
|---|---|---|
| SGLang DeepSeek-V4 cookbook | sglang `ef6678fae5`（2026-10-28），`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` §3.7、§3.8 | **DSv4 + UMBP 的官方部署文档** |
| SGLang cookbook 生成器 | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`（MI355X PD role 函数、`umbp` card） | 官方命令的实际生成逻辑 |
| InferenceX recipe（多机 PD + UMBP） | InferenceX `6cbbb96e`，`benchmarks/multi_node/srt-slurm-recipes/dsv4/sglang/mi355x-fp4/agentx/disagg-umbp-dspark.yaml`；config key `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark` | 官方跑分配置（镜像 `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026`） |
| InferenceX recipe（单机） | `benchmarks/single_node/srt-slurm-recipes/dsv4/sglang/mi355x-fp4-mtp/agentic.yaml`；config key `dsv4-fp4-mi355x-sglang-agentic-mtp` | 单机官方跑分，**用 HiCache，不用 UMBP** |
| MORI 文档 | mori `09aa23b`（2026-10-28），`src/umbp/doc/runtime-env-vars.md`、`docs/UMBP-TIER-MANAGEMENT-BENCHMARK.md`、`src/umbp/doc/design-standalone-process-mode.md` | env var 全集、tier policy JSON |
| AMD blog | [Rebuilding Agentic AI for AMD GPU](https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html) | 设计理念；实验是 **Kimi K2.6 1P2D**，不是 DSv4 |

## 1. 结论速览

1. **SGLang 没有官方的「DSv4-Pro + MI355X + 单机 8 卡聚合（非 PD）+ UMBP」配置。**
   - 官方验证过的 UMBP 形态只有一种：**1P1D PD 分离**，每个节点 8×MI355X。
   - 在这种形态下，**只有 prefill 节点**挂 standalone UMBP tier（`umbp_standalone_server`，DRAM 1.5 TB，hugepages，关 SSD）。
   - decode 节点不挂 offload tier。
2. 单机官方配置（InferenceX `dsv4-fp4-mi355x-sglang-agentic-mtp`）的 offload 都走 **HiCache**（`hicache-ratio 1.5`、`write_through`、`page_first_direct`），没有 UMBP 臂。
3. cookbook 明确说 linker（UMBP）**不依赖 PD**，也**不依赖 DP attention**，聚合模式同样可用。因此单机 8 卡的推荐配置只能从官方 prefill role 推导，见 §4。这一点必须在文档和 PR 中标明是「推导」，不是「官方」。
4. blog 中「可插拔 cache-management policy」的落地情况：
   - put-routing、get-routing、eviction 这三个 strategy 接口都在 **master** 侧。**单机 standalone（无 master）用不到它们**。
   - 单机下实际生效的只有本地 LRU 加 DRAM 水位线。
   - eviction 目前只有 LRU 一个实现；replication 还在规划中。
   - 详见 §5。

## 2. 官方 UMBP tier server 配置（SGLang cookbook §3.8 / InferenceX）

```bash
mkdir -p /tmp/umbp
UMBP_DRAM_CAPACITY=1500000000000 \
UMBP_DRAM_USE_HUGEPAGES=1 \
UMBP_SSD_ENABLED=0 \
MORI_UMBP_LOG_LEVEL=info \
<mori>/umbp_standalone_server unix:///tmp/umbp/standalone.grpc.sock
# worker 侧：
UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp/standalone.grpc.sock sglang serve ...
```

要点（均出自 cookbook 或 InferenceX recipe）：

- **容量**：DRAM 1.5 TB（`1500000000000`，十进制字节）。
  - cookbook 要求主机至少 3 TB DRAM。
  - InferenceX 的 preamble 会拒绝 `UMBP_DRAM_CAPACITY > host_mem / 2` 的配置。
  - 本机 `free -g` 总计 2751 GB，按这条规则上限约 **1.37 TB**。
- **hugepages**：
  - 预留页数为 `(1.5e12 / Hugepagesize + 1) × 1.02`（2 MiB 页，多留 2%）。
  - InferenceX 的 `hugepage_sanity.sh` 发现预留不足会直接失败。
  - `server_sglang.sh` 发现退回 4 KiB 页时拒绝启动。
  - 不用 hugepages 时，大容量池启动极慢：我们实测 1 TiB 需要 >10 min 才完成 fault-in。
- **就绪判据**：
  - 等日志出现 `host memory registered for GPU access` 再启动 worker，仅 socket 出现不够。
  - InferenceX 的就绪超时为 5400 s。
- **SSD**：`UMBP_SSD_ENABLED=0`，因为 config 默认开启 SSD。
- **默认值对照**：默认 DRAM 4 GiB、hugepages 关、startup wait 30 s、log level error。
- **客户端启动超时**：InferenceX 额外设了 `--hicache-storage-backend-extra-config '{"standalone_startup_timeout_ms":120000}'`，把客户端 ready probe 提到 120 s。
- **tier 进程的 env**（InferenceX）：`HSA_NO_SCRATCH_RECLAIM=0`、`GPU_MAX_HW_QUEUES=5`。
- **Docker 部署**：tier 和 worker 两个容器都要挂 `-v /tmp/umbp:/tmp/umbp`。

## 3. 官方 SGLang worker 配置（1P1D，UMBP 只在 prefill）

### 3.1 三档策略（Pro Official = DeepSeek-V4-Pro-0813，DSpark）

| 策略 | 并发 | Prefill → Decode | `--max-running-requests` | DSpark block size | mem-fraction / SWA ratio |
|---|---|---|---|---|---|
| Low-Latency | 16 | TP4 → TP8 | 32 | 6 | 0.86 / 0.1 |
| Balanced | 48 | TP4 → TP8 | 96 | 3 | 0.86 / 0.1 |
| High-Throughput | 256 | TP8+DP8+EP8（MegaMoE）→ TP8+DP8 | 512 | 3 | 0.92 / 0.15 |

- **规则**：
  - `--max-running-requests` 设为 2 × 目标并发，两个 role 一致。
  - decode 的 `--cuda-graph-bs-decode` 覆盖 `1 … 2N / attn_dp_size`。
  - decode 的 `MORI_MAX_DISPATCH_TOKENS_DECODE` 设为 64 × (block size + 1)，block size 为 6 时取 448，为 3 时取 256。
- InferenceX 实际跑分点更多（c1/c16/c32/c48/c192/c256/c384/c512），全部是 UMBP-linker 臂，原先的 c128 点已删除。

### 3.2 prefill role 中与 UMBP 相关的 flag

```bash
UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp/standalone.grpc.sock
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend mori
--optimistic-prefill-attempts 2        # 依赖 linker
--enable-cache-report
--disable-cuda-graph                   # FP4 indexer 的 prefill 路径不能 capture
--chunked-prefill-size 16384           # TP-only；DP8 档为 65536
```

prefill 的通用部分：

```bash
--tp 8 [--attn-dp-size 8 --enable-dp-lm-head --enable-dp-attention-local-control-broadcast]
--enable-deepseek-v4-fp4-indexer --attention-backend dsv4
--kv-cache-dtype fp8_e4m3 --page-size 256
--swa-full-tokens-ratio 0.1|0.15 --mem-fraction-static 0.86|0.92
--enforce-shared-experts-fusion --load-balance-method round_robin
--tokenizer-worker-num 8 --stream-interval 20 --context-length 1048576
--speculative-algorithm DSPARK --speculative-dspark-block-size <3|6>
```

env：

- 通用：
  - `HSA_NO_SCRATCH_RECLAIM=0`
  - `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`
  - `SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=1`
  - `SGLANG_OPT_USE_AITER_BATCHED_GEMM=1`
  - `AITER_BF16_FP8_MOE_BOUND=0`
  - `TORCH_BLAS_PREFER_HIPBLASLT=1`
  - `GPU_MAX_HW_QUEUES`：TP 为 2，DP 为 5
  - `SGLANG_DSV4_UNIFIED_KV_FP8=1`
- InferenceX recipe 在此之上还开了：
  - `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`
  - `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1`
  - `SGLANG_ENABLE_SPEC_V2=1`
- Pro Official 的 TP 档另设 `SGLANG_DSV4_DENSE_MXSCALE=1`。

### 3.3 关键约束

- **linker 与 HiCache 互斥**：SGLang 会拒绝同时开启两者。linker 直接把 unified radix tree 接到外部 store，中间没有 host tier。
- **decode 不要继承 prefill 的 linker flag**（InferenceX #3664 修过这个 bug）。decode 没有 UMBP tier，开着 linker 就是指向空地址。
- **router**：
  - 前端为 `sglang-router`，`--policy consistent_hashing`、`--decode-policy round_robin`、`cache-threshold 0.3`。
  - c384/c512 需要 `--request-timeout-secs 14400`。

## 4. 单机 8 卡（聚合，非 PD）的推导配置 —— 非官方

依据：

- cookbook §3.8 说明 linker 可以用于聚合模式，也可以用于纯 TP。
- 官方 TP8 prefill role 的 UMBP 部分可以原样搬过来。

聚合模式下需要改的地方：

- 去掉 `--disaggregation-*`。
- 由于同一个 worker 要做 decode，**不能再用 `--disable-cuda-graph`**，改用 decode graph（`--cuda-graph-bs-decode 1..2N/dp`）。
- 其余参数沿用单机 HiCache recipe 的同档设置（并发、DP、chunked prefill、mem-fraction）。

需要注意的容量陷阱（cookbook §3.8 原文要点）：

- **纯 TP 下 key 按 rank 区分**：
  - MLA KV 在 TP 间是复制的，TP8 会产生 8 个 keyspace，存 8 份相同的 token。
  - 因此 tier 实际能存的不同 token 数只有字节预算的 **1/8**，TP4 下为 1/4。
  - DP attention 会把所有 key 收敛到一个共享 keyspace，整个预算都有效。
- vLLM PR #63002 中，TP 下由 rank 0 去重（`tp*` key），不受这个 1/8 限制。和 SGLang 纯 TP 对比时，要按去重后的有效容量来算。
- **池是每节点一个进程**：跨节点的 worker 会按节点把 keyspace 分片。单机不受影响。

## 5. 可插拔 cache-management policy（blog「Pillar 2」）在代码中的落地

| blog 说法 | MORI 代码 | 单机 standalone（无 master）是否生效 |
|---|---|---|
| Offload / put-routing | `RoutePutStrategy`（`routing/route_put_strategy.h`）。env：`UMBP_ROUTE_PUT_SELECT_ALGO`（`most_available` / `random`）、`UMBP_ROUTE_PUT_NODE_AFFINITY`（`none` / `same` / `local`） | ❌ master 侧路由，单机只有一个节点 |
| Load / get-routing | `RouteGetStrategy`（`Random`、`LocalPreferring`） | ❌ master 侧 |
| Eviction | master 侧 `MasterEvictStrategy`，只有 `LruMasterEvictStrategy`（注释："only LRU exists today"）。节点侧 `UMBP_EVICTION_MODE=local` / `master_ranked` / `master`；`UMBP_EVICTION_POLICY` 默认 `lru` | ⚠️ 只有本地 LRU 和水位线 |
| Replication | — | 规划中，尚未实现 |
| 丰富的 metrics | standalone 无 master 时在 `UMBP_STANDALONE_METRICS_PORT`（默认 9092）提供 `/metrics`；`UMBP_SUMMARY_INTERVAL_MS`（默认 60 s）定期打印一行活动摘要 | ✅ |
| 多 tier policy（HBM → DRAM → SSD、`offload_to`、`promote_trigger`） | `UMBP_BACKEND_POLICY`（JSON，示例 `examples/umbp/multi_backend_policy.json`） | ✅ 可用于 distributed-backed standalone server，但设置后进入 distributed 配置路径；纯 DRAM 单机没有必要 |
| TTL / session pinning / priority hint | serving API → HiCache → UMBP | SGLang 侧尚未完整打通；vLLM 侧没有 |

结论：在单机 DRAM-only standalone 场景下，能调的「policy」实质上只有**容量**、**水位线**（`UMBP_DRAM_HIGH_WM` / `UMBP_DRAM_LOW_WM`；`UMBPConfig` 默认 0.95 / 0.9，但 MORI v1.2.3.post1 的 standalone server 实际日志为 `[PageBackend] local eviction ON tier=2 high=0.9 low=0.7`），以及**写入/读取侧做什么**（这部分由引擎 connector 决定，见 §6）。

blog 实验（Kimi K2.6，1P2D，c128，每节点 DRAM 预算 1 TB）的可借鉴结论：

- 在**总 DRAM 不变**的前提下，把 L2（每 rank 的 host cache）的一部分切给 L3（UMBP pool）：
  - 累计命中率基本不变；
  - 但**首轮/恢复会话**（first-call）的命中率提升 21 个百分点（58.2% → 79.3%）；
  - TTFT p99 从 17.3 s 降到 9.9 s。
  - 原因：MLA+TP 下 L2 按 rank 复制，L3 不付这份复制税。
- 再开启 loadback prefetch 优化（让 restore 完全藏在 forward 后面）：
  - p50 / p90 / p99 全部改善；
  - 吞吐 +7.7%。
- **调优时应拆开看 first-call 和 later-call**。累计命中率会掩盖效果。

## 6. 可调参数速查

### 6.1 tier server（`umbp_standalone_server`）

| env | 默认 | 建议 / 说明 |
|---|---|---|
| `UMBP_DRAM_CAPACITY` | 4 GiB | 官方 1.5 TB；本机上限约 1.37 TB（不超过主机内存的一半） |
| `UMBP_DRAM_USE_HUGEPAGES` | 0 | 官方 1；需要先预留 hugepages，否则启动慢，且 TLB 压力大 |
| `UMBP_SSD_ENABLED` | 1 | DRAM-only 时设为 0 |
| `UMBP_DRAM_HIGH_WM` / `UMBP_DRAM_LOW_WM` | 0.95 / 0.9（`UMBPConfig`）；v1.2.3.post1 standalone 实测 0.9 / 0.7 | 本地 LRU 水位线，以 server 启动日志 `[PageBackend] local eviction ON ...` 为准 |
| `UMBP_EVICTION_POLICY` | `lru` | 目前只有 lru |
| `UMBP_STANDALONE_METRICS_PORT` | 9092 | 设为 0 关闭 |
| `UMBP_SUMMARY_INTERVAL_MS` | 60000 | 活动摘要日志的周期 |
| `MORI_UMBP_LOG_LEVEL` | error | 设为 info 才能看到 ready 日志 |
| `UMBP_WORKLOAD_TRACE_PATH` | 空 | 录制 PUT/GET trace，可用 `umbp_tier_bench replay --config policy.json` 离线比较不同 policy |

### 6.2 SGLang linker 客户端（参考）

| env / 参数 | 默认 | 说明 |
|---|---|---|
| `UMBP_LAYER_GROUP` | 8 | 把多层合成一个对象，摊薄 RPC 开销；8 是实测默认值 |
| `UMBP_OFFLOAD_COALESCE_PAGES` | 1024 | 合并排队 offload 任务的页数 |
| `UMBP_RANGES_PER_CALL` | 8192 | 单次 RPC 的 range 预算，受 gRPC 消息上限约束 |
| `standalone_startup_timeout_ms`（extra-config） | — | InferenceX 取 120000 |
| `--optimistic-prefill-attempts` | — | 官方取 2 |

### 6.3 vLLM `UMBPStoreConnector`（PR #63002）

- `kv_connector_extra_config`：
  - `mode=standalone`、`endpoint=unix:///tmp/umbp/standalone.grpc.sock`；
  - `rebuild_sliding_window`：DSv4 SWA bounded replay，默认开启；
  - `lazy_offload` / `lazy_offload_max_blocks`；
  - `save_decode_cache`。
- `kv_load_failure_policy=recompute`。
- 我们的测试脚本：
  - 启动见 `umbp-dsv4-mtp/run_vllm_dsv4_mtp.sh`（`UMBP_EXTRA` 可追加 extra config）；
  - tier server 见 `umbp-dsv4/run_umbp_server.sh`。

## 7. 与我们 vLLM 测试设置的差异 / 调优待办

| 项 | 官方 SGLang | 我们的 vLLM agentic 测试（override_dep8_c128） |
|---|---|---|
| 形态 | 1P1D，UMBP 在 prefill | 单机聚合 DEP8 |
| tier 容量 | 1.5 TB | 512 GiB |
| hugepages | 开 | 关（`HugePages_Total: 0`） |
| GPU KV 占用 | — | 约 33–36%，GPU prefix cache 本身没满 |

待办（按预期收益排序）：

1. **把 workload 调到 GPU KV 打满的区间**（加长 duration 或提高并发），否则 UMBP 恢复量只占 prompt token 的约 0.8%。
2. **预留 hugepages，把容量提到约 1.3 TB**：

   ```bash
   echo $(( (1300000000000 / 2097152 + 1) * 102 / 100 )) > /proc/sys/vm/nr_hugepages
   ```

   需要在宿主机或特权容器中执行，并确认不影响其他用户。
3. 补一个 `rebuild_sliding_window:false` 对照点，并对 baseline 和 UMBP 各做方差检查（多跑几次）。
4. 统计时把指标拆成 first-call 和 later-call（参考 blog），看命中率和 TTFT 的重新分布，不只看累计值。
5. 详细对比方案见 §8。
6. 打开 `UMBP_WORKLOAD_TRACE_PATH` 录制真实 trace，用 `umbp_tier_bench replay` 离线比较水位线和容量的效果。

## 8. vLLM 单机 8 卡 UMBP 配置对比测试方案（2026-10-29 制定，待测）

目标：在当前 vLLM 代码（PR #63002）下，找出 DSv4-Pro / MI355X / standalone / InferenceX agentic 的最佳 UMBP 配置。

### 8.1 固定条件

除被测变量外，所有条件与 §7 的已有结果保持一致，结果才可比：

- recipe 点：`override_dep8_c128`（DP8×TP1，DSpark 6 token，`--max-num-seqs 128`，`--gpu-memory-utilization 0.86`）。
- 每个点都是全新部署：UMBP server → engine → router。
- `DURATION` 用 InferenceX 默认值 **3600 s**（`infx/matrix/validation.py` 中的 `DEFAULT_AGENTIC_DURATION_SECONDS`，也是 `run_agentic.sh` 的默认值，不要改成 600；≥ 900 s 时 AIPerf 也不需要 `--unsafe-override`）。§7 中已有的 600 s 结果不能与之直接比较，因此 B0 和 U1 也要按 3600 s 重测。
- `RUNS_DIR=/tmp/umbp_runs`（避免 NFS 配额写满）。
- 启动入口：`umbp-dsv4-mtp/run_config.sh`。
  - `UMBP_DRAM_CAPACITY` 和 `UMBP_DRAM_USE_HUGEPAGES` 会透传给 `run_umbp_server.sh`。
  - `UMBP_EXTRA` 会追加到 `kv_connector_extra_config`。
- DEP8 下每个 DP rank 都是 TP1，不存在 TP 复制，因此 `dedup_tp_replicated_kv` 在这个点不起作用，不纳入测试。

### 8.2 参照 SGLang 的变量映射

| SGLang 官方做法 | vLLM 对应开关 | 说明 |
|---|---|---|
| tier 1.5 TB + hugepages | server：`UMBP_DRAM_CAPACITY`、`UMBP_DRAM_USE_HUGEPAGES` | 本容器的 `/proc/sys` 只读、`HugePages_Total: 0`，**无法自行预留 hugepages**，需要宿主机管理员操作 |
| linker 挂在 unified radix tree 上，请求结束时连同 decode 输出一起 offload | `save_decode_cache`（默认 false，只存 prefill 产生的块） | agentic 的下一轮 prompt 包含上一轮的输出；但 DSv4 模板会丢弃历史 reasoning，实际可复用的只有回答和 tool call 部分 |
| 单机 HiCache 用 `write_through`（立即写） | 默认 eager store ≈ write-through；`lazy_offload=true` ≈ write-back（块从 GPU free queue 被淘汰时才写） | |
| `--optimistic-prefill-attempts 2`，不让 lookup/load 阻塞调度 | `lookup_async`（默认 false，lookup 在 scheduler 线程上做） | DSv4 有 indexer，`load_async` 被强制为 true，不需要测 |
| 客户端 IO 参数（`UMBP_LAYER_GROUP` 等） | `num_workers`（默认 4） | 每个 worker 的传输线程数 |
| SGLang 也只存压缩 KV，命中后重算最后一个窗口 | `rebuild_sliding_window`（默认 true） | false 作为对照点 |

### 8.3 精简后的测试项（最多 5 次完整 AgentX 运行）

每次完整运行约 1.5 h，所以最多测 5 个点：

| # | ID | 配置 | 理由 |
|---|---|---|---|
| 1 | B0 | 不开 UMBP | 3600 s 下没有 baseline，必须测 |
| 2 | U1 | 512 GiB，不开 hugepages，默认开关（eager store、`rebuild_sliding_window=true`、`save_decode_cache=false`、`num_workers=4`） | 默认配置，也是后续各项的对照 |
| 3 | U3 | U1 + `save_decode_cache=true` | 预期收益最大：agentic 下一轮 prompt 包含上一轮的回答和 tool call（SGLang linker 也会存 decode 输出） |
| 4 | U4 | U1 + `lazy_offload=true` | 对应 write-through 与 write-back 的取舍：减少 store 流量，与 decode 和 prefill 争用的带宽也更少 |
| 5 | 动态 | 按下面的规则决定 | |

第 5 个点的决定规则（在 1–4 跑完后判断）：

1. U1 的 UMBP server 日志出现淘汰，说明池子写满了：测 **U1 + 1 TiB**（`UMBP_DRAM_CAPACITY=$((1024*1024**3))`，启动 >10 min）。
2. 否则，测 **U1 重跑**，用来估计波动，判断 U3/U4 的差异是否可信。

不测 U3+U4 组合：`lazy_offload=true` 时，存储走 `_prepare_lazy_store_plans()`，从 GPU free queue 中挑所有带 hash 的块，不区分 prefill 和 decode。`save_decode_cache` 只作用于 eager 路径（`scheduler.py` 中 `not self.lazy_offload` 分支），所以 lazy 模式下它不起作用，组合与 U4 等价。

因此 U4 与 U1 相比其实改了两件事：存储时机（eager → lazy），以及 decode 块也会被存。解读 U4 结果时要结合 U3 一起看。

不再单独测的项，以及原因：

| 项 | 不测的原因 |
|---|---|
| 容量 256 GiB / 1 TiB | 改为看 U1 的淘汰情况来判断，只有写满时才在第 5 点测 1 TiB |
| `lookup_async` | lookup 只查 key，c128 下预期影响小；若 U1 的 scheduler 有明显停顿再补测 |
| `num_workers=8` | 只有 U1 出现 load/store 超时或排队时才有意义 |
| `rebuild_sliding_window=false` | 对照点，设计上预期更差，对选最优配置没有帮助 |
| hugepages | 容器内 `/proc/sys` 只读，需要宿主机管理员预留；作为后续单独项 |
| 每项重复 3 次 | 3600 s 的单次测量比 600 s 稳定；只在第 5 点的规则 3 下重跑一次 |

> ⚠️ U3（`save_decode_cache`）和 U4（`lazy_offload`）还没有在 DSv4 + DSpark + SWA replay 上跑过。正式测试前，先用 `umbp-dsv4-mtp/smoke_umbp.sh` 做冒烟测试（只需几分钟，不计入 5 次）：
> - 发一个约 20K token 的两轮对话，清空 GPU prefix cache 后重发第二轮；
> - 检查 external hit 是否大于 0、有无报错或 NaN，并对比恢复前后的输出。
>
> 用法：`TAG=u3_smoke UMBP_EXTRA=',"save_decode_cache":true' bash umbp-dsv4-mtp/smoke_umbp.sh`

命令（按顺序）：

```bash
cd /home/shashen/vllm/vllm-dev-skills/docs/mori
export RUNS_DIR=/tmp/umbp_runs UMBP_DRAM_CAPACITY=$((512 * 1024**3))
TAG=b0 UMBP=0 bash umbp-dsv4-mtp/run_config.sh
TAG=u1 UMBP=1 bash umbp-dsv4-mtp/run_config.sh
TAG=u3_save_decode UMBP=1 UMBP_EXTRA=',"save_decode_cache":true' bash umbp-dsv4-mtp/run_config.sh
TAG=u4_lazy UMBP=1 UMBP_EXTRA=',"lazy_offload":true' bash umbp-dsv4-mtp/run_config.sh
# 第 5 点按规则选一个：
TAG=u1_1tib UMBP=1 UMBP_DRAM_CAPACITY=$((1024 * 1024**3)) bash umbp-dsv4-mtp/run_config.sh
TAG=u1_rerun UMBP=1 bash umbp-dsv4-mtp/run_config.sh
```

全部默认 `DURATION=3600`，总计约 7.5 h。

### 8.4 后续（不在本轮 5 次之内）

- 如果 1 h 测量期间 GPU KV 占用仍明显不满，再考虑 `override_dep8_c192` 加压点。
- hugepages（需要管理员）、`lookup_async`、`num_workers` 视本轮观察结果再补测。

### 8.5 比较指标与选择标准

- **主指标**：输出吞吐（tok/s）、TTFT mean / p90 / p99。
- **辅助指标**：
  - TPOT；
  - 每请求实际计算的 prefill token 数；
  - `vllm:external_prefix_cache_hits_total / queries_total`（U1 实测为 3.58M / 48.1M ≈ 7.4%）；
  - GPU prefix 命中率；
  - UMBP server 日志中的淘汰与驻留情况（MORI v1.2.3.post1 没有周期性 summary，只能看 eviction 日志）。
- **选择规则**：
  - 差异小于约 3% 时视为持平（每项只有一次测量）；若第 5 点做了 U1 重跑，以实测波动为准。
  - 吞吐和 TTFT p90 冲突时，以吞吐为先。
  - 不做 GPQA：数据集已删除，`/home/shashen` 空间不足。这些开关只改变存取策略、不改变计算，正确性由 `smoke_umbp.sh` 的恢复前后输出对比来把关。
