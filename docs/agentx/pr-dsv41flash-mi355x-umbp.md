# [AMD][AgentX] DeepSeek-V4.1-Flash MI355X vLLM: add DP2 concurrency 256 and 384 with UMBP KV offloading / [AMD][AgentX] DeepSeek-V4.1-Flash MI355X vLLM：新增 DP2 并发 256、384 点并启用 UMBP KV offloading

## Summary

The MI355X DeepSeek-V4.1-Flash vLLM AgentX curve stops at concurrency 64. Above that, the GPU KV cache stops holding every live session's prefix. Evicted prefixes are then recomputed, and prefill crowds out decode. vllm-project/vllm#64715 (merged 2026-11-10) adds UMBP standalone support for DeepSeek-V4.1-Flash on ROCm. With it, the engine offloads KV blocks to a node-local host-DRAM pool owned by MORI's `umbp_standalone_server` and restores them on a prefix hit instead of recomputing.

This PR adds a new config key, `dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`, with two points: concurrency 256 and 384. Evals run at c384. Concurrency 128 is left out: at c128 UMBP did not help in local runs (see below), and #4992 already covers it without offloading.

- **Image:** `vllm/vllm-openai-rocm:nightly-be47189f37f6162991366ff6b91539f6e9d1f432` (2026-11-12, digest `sha256:a6263de1…`). It carries #64715 (UMBP standalone for DeepSeek-V4.1-Flash) and #66036 (per-tensor fp8 KV for sparse MLA on ROCm), and its MORI v1.2.3.post1 is built with UMBP. The existing TP4/TP2 key and the TP8 c1 key are untouched and are not rerun.
- **KV cache dtype:** `fp8_e4m3` (#66036): plain per-tensor fp8 KV rows for the AITER sparse MLA instead of the default block-scaled `fp8_ds_mla` record. This differs from #4992, which keeps the default.
- **Topology:** two GPUs per server with attention DP2. The MoE stays tensor-parallel across both ranks (`tensor-parallel-size 1`, `data-parallel-size 2`, no expert parallelism). vllm-router 0.1.14 fronts the ranks with `consistent_hash`, so each session keeps its rank and its prefix. Engine and router settings match the no-offload DP2 points in #4992, the current best MI355X DP2 baseline: prefill cadence (`prefill-schedule-interval 11`) and no `--monokernel`. The only difference is UMBP, so comparing the two sweeps isolates its effect.
- **Recipe:** the overrides are named `override_dp2_umbp_c{256,384}` and each row selects its own. #4992 adds `override_dp2_c{128,256,384}` to the same recipe file without `KV_OFFLOADING`, so automatic variant matching would select both sets for a UMBP point.
- **KV offloading:** the recipe launches `umbp_standalone_server` as a critical srt-slurm `generic` service that starts before the workers and becomes ready once it logs `[StandaloneServer] listening`.
  - The server owns one 449 GB pool: `dram-utilization` 0.60 of the two-GPU node share, on 4 KiB pages. Both DP ranks share it.
  - srtctl gives every step of a job its own mount namespace, so the server's `/tmp` is not the workers'. The recipe mounts the node-local `/dev/shm` into every step at `/umbp-host`, and the gRPC and fd sockets live in `/umbp-host/umbp/`. A new server rebinds over socket files an earlier job left behind.
  - The engine attaches through `UMBPStoreConnector` (`mode: standalone`, `kv_load_failure_policy: recompute`).
- **Draft:** weights unchanged. It is the shipped DSpark head (`mtp.0-2`), loaded with the checkpoint's FP8 config. Its attention layers share the target's KV cache dtype, so target and draft both use the per-tensor `fp8_e4m3` cache.

### Why: local no-offload (B0) vs UMBP (S1) runs

Local AgentX replays on one MI355X node (GPUs 0-1) with this recipe's engine settings before the fp8 KV change (default `fp8_ds_mla` KV): the `nightly-96710245` container with vLLM main `595b1514e8` Python code. B0 is #4992's DP2 configuration; S1 adds UMBP with the 449 GB pool. Each point is a single InferenceX agentx-fast run (1200 s profile, prefix cache cold at the start), not a CI run. As in CI, UMBP turns off the extensible KV cache, so S1 has 4.2% less GPU KV than B0 (26.71M vs 27.89M tokens per rank).

| Metric | c128 B0 → S1 | c256 B0 → S1 | c384 B0 → S1 | c544 B0 → S1 (not in this PR) |
|---|---|---|---|---|
| Total throughput per GPU (tok/s) | 193,099 → 183,984 (−4.7%) | 250,435 → 275,836 (**+10.1%**) | 143,888 → 291,619 (**+103%**) | 46,641 → 287,025 (**+515%**) |
| TTFT p50 / p90 (s) | 0.87 / 2.32 → 0.88 / 2.62 | 1.60 / 5.12 → 1.45 / 4.10 | 7.36 / 264.9 → 2.41 / 7.92 | 66.1 / 387.3 → 21.9 / 58.7 |
| TPOT mean (ms) | 24.8 → 26.6 | 50.0 → 43.6 | 102.7 → 69.1 | 144.9 → 87.5 |
| Interactivity p90 (tok/s/user) | 29.2 → 27.4 | 12.9 → 16.4 | 6.6 → 10.4 | 6.8 → 8.5 |
| Prompt tokens from GPU prefix cache (B0 / S1) | 90.0% / 89.4% | 87.4% / 88.1% | 73.0% / 64.6% | 27.8% / 21.7% |
| Prompt tokens restored from UMBP | 0.46% | 0.67% | 21.5% | 62.0% |
| Prompt tokens computed | 10.0% → 10.1% | 12.6% → 11.3% | 27.0% → 13.9% | 72.2% → 16.3% |

- The gain tracks how far the working set exceeds GPU KV. At c384 the no-offload server is overloaded: its GPU prefix hit rate falls to 73%, TTFT p90 reaches 265 s, and throughput drops below c256. UMBP restores 21.5% of prompt tokens, halves the prefill computed, doubles throughput and brings TTFT p90 down to 7.9 s.
- At c256 UMBP restores 0.67% of prompt tokens, cuts prefill computed by 11%, and raises throughput by 10% and p90 interactivity by 27%.
- At c128 GPU KV holds the working set (90% prefix hits). UMBP restores 0.46% and does not pay for the smaller GPU KV and its store traffic: throughput is 4.7% lower, near the ~4.5% noise of a single fast run, with TTFT and TPOT moving the same way. That point is therefore not part of this PR.
- c544 is beyond the knee and is not part of this PR. Without offloading the server collapses: 27.8% GPU prefix hits, 72% of prompt tokens recomputed, 46,641 tok/s/GPU. UMBP restores 62% of prompt tokens and holds 287,025 tok/s/GPU, level with c384, but TTFT p50/p90 rise to 21.9/58.7 s, 3.6% of steps exceed the cudagraph capture limit, and there are 20 preemptions. Throughput with UMBP therefore peaks around c384.
- No UMBP eviction (the pool never reached its watermark) at any point, and no preemption up to c384. Steps of 1,024-2,048 tokens, which include decode steps above the 1,024-token cudagraph capture limit, are 0.7% at c384, as at c256.
- The first S1 run at c544 failed during warmup, and a rerun succeeded. On the first requests, DP0's worker JIT-compiled Triton kernels that warmup does not cover (indexer prefill metadata among them) while holding the GIL. DP1 waited in the cross-DP all-reduce of a dummy batch until `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (300 s) expired and the engine died. In the rerun, with a partly warm Triton cache, the same stall lasted about 75 s. Every run, including B0, logs runtime JIT warnings, so this is not specific to UMBP.
- #4992 reports 310,739 tok/s/GPU at c256 from a one-hour local run on `nightly-083fbac02`. The B0 numbers above come from shorter, cold-cache fast runs, so compare UMBP against B0 here rather than against that absolute figure.

### Risks

- The first CI run (https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43151259956) failed on every job because the sockets were under `/tmp`: the workers could not attach to the server (`cannot attach to the UMBP standalone server at unix:///tmp/umbp/standalone.grpc.sock`). The host-directory mount above fixes the path. The server and the workers also exchange GPU memory handles over HIP IPC; enroot shares the host IPC and PID namespaces, so this should work, but the next sweep is the first confirmation.
- The local runs below predate the move to `nightly-be47189` and `fp8_e4m3`: they used the `nightly-96710245` image's native binaries with newer vLLM Python code, the default `fp8_ds_mla` KV cache, and fast mode. The fp8 KV layout changes the GPU KV capacity and UMBP block size and was not measured locally; the sweep is its first end-to-end run.
- A cold Triton cache on a CI node can make the first-request JIT stall exceed the 300 s execute-model timeout, as in the first local c544 run, and fail the point. The sweep shows whether this happens at c256 or c384.
- The upstream vLLM recipe for this deployment (start `umbp_standalone_server`, then `vllm serve --kv-transfer-config …`) must be merged in vllm-project/recipes before this PR merges.

<details>
<summary>AI model disclosure</summary>

- Model/version: `claude-opus-5-5[1m]` (Claude Opus 5.5, 1M context), via Claude Code.
- Role: read the existing config, recipe and srt-slurm service code; drafted the master-config key, recipe overrides, `perf-changelog.yaml` entry and this description; ran matrix generation, recipe rendering with the pinned srtctl, `infx.matrix.changed` and the sweep planner; ran and summarized the local B0/S1 AgentX benchmarks. No delegated agents. A human reviewed the plan and chose the topology (DP2), the concurrencies and the UMBP mode.

</details>

<details>
<summary>PR checklist and change type</summary>

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [x] Configuration change
- [ ] Documentation update
- [ ] Other (please describe)

## Checklist

- [x] I have completed the AI model disclosure and kept it current
- [x] I have tested my changes locally
- [ ] I have updated documentation if necessary
- [x] **If `python -m infx.matrix.changed --base origin/main` lists affected config keys, I have appended one new entry covering them to the physical end of `inferencex-e2e/perf-changelog.yaml`; if it lists none, I have added none. I have not edited historical entries**
- [ ] **If this PR appends an `inferencex-e2e/perf-changelog.yaml` entry, it carries exactly one primary sweep label** (a maintainer applies it on fork PRs): `full-sweep-fail-fast` (recommended), `full-sweep-enabled`, or `non-canary-full-sweep-enabled`. Optional modifiers `all-evals`, `evals-only`, and `agentx-fast` require a primary label; the last two block reuse while applied.
- [ ] **Before merging via reuse, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) has commented `/use <run_id>` (or the legacy `/reuse-sweep-run`) on this PR**. Do this **only once there is a final full sweep that is all green with evals passing**, since after this comment the primary sweep label will no longer automatically kick off new sweeps. Remove and re-add the primary sweep label to force a new sweep.

</details>

<details>
<summary>中文</summary>

MI355X 上 DeepSeek-V4.1-Flash 的 vLLM AgentX 曲线目前只到并发 64。并发更高时，GPU KV cache 放不下所有活跃会话的前缀，被淘汰的前缀只能重新计算，prefill 会挤占 decode。vllm-project/vllm#64715（2026-11-10 合入）在 ROCm 上为 DeepSeek-V4.1-Flash 支持了 UMBP standalone 模式：引擎把 KV block 卸载到节点本地的主机内存池（由 MORI 的 `umbp_standalone_server` 管理），前缀命中时直接恢复，不再重算。

本 PR 新增配置键 `dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`，包含并发 256、384 两个点，eval 在 c384 上跑。没有放 c128：本地测试中 UMBP 在 c128 没有收益（见下文），而且 #4992 已经有不开 offloading 的 c128 点。

- **镜像**：`vllm/vllm-openai-rocm:nightly-be47189f37f6162991366ff6b91539f6e9d1f432`（2026-11-12，digest `sha256:a6263de1…`），包含 #64715（DSv4.1-Flash 的 UMBP standalone 支持）和 #66036（ROCm 上 sparse MLA 的 per-tensor fp8 KV），其中的 MORI v1.2.3.post1 编译时开启了 UMBP。已有的 TP4/TP2 键和 TP8 c1 键不变，也不会重跑。
- **KV cache 类型**：`fp8_e4m3`（#66036），即给 AITER sparse MLA 用的 per-tensor fp8 KV，而不是默认的块缩放 `fp8_ds_mla`。这一点和 #4992 不同，#4992 用的是默认格式。
- **拓扑**：每个服务 2 张卡，attention 用 DP2，MoE 在两个 rank 间做张量并行（`tensor-parallel-size 1`、`data-parallel-size 2`，不开 expert parallelism）。前端是 vllm-router 0.1.14，策略为 `consistent_hash`，同一会话固定落在同一个 rank 上，前缀也留在那里。引擎和 router 参数与 #4992 中不开 offloading 的 DP2 点完全一致：prefill cadence 相同（`prefill-schedule-interval 11`），也都不开 `--monokernel`。#4992 是目前 MI355X DP2 最好的基线，两边唯一的区别是 UMBP，对比两次 sweep 就能单独看出 UMBP 的效果。
- **Recipe**：override 命名为 `override_dp2_umbp_c{256,384}`，每一行显式指定自己的 override。#4992 在同一个 recipe 文件里加了 `override_dp2_c{128,256,384}`，且没有设置 `KV_OFFLOADING`，如果靠自动匹配，UMBP 的点会同时匹配到两组 override。
- **KV offloading**：recipe 把 `umbp_standalone_server` 作为 srt-slurm 的 `generic` 服务启动，标记为 critical，在 worker 之前启动，日志出现 `[StandaloneServer] listening` 即视为就绪。
  - server 管理一个 449 GB 的池子，大小是两卡份额的 `dram-utilization` 0.60，使用 4 KiB 页，两个 DP rank 共用。
  - srtctl 会给同一个作业里的每个 step 分配独立的 mount namespace，server 看到的 `/tmp` 和 worker 看到的不是同一个。所以 recipe 把节点本地的 `/dev/shm` 挂载到每个 step 的 `/umbp-host`，gRPC socket 和 fd socket 都放在 `/umbp-host/umbp/` 下。之前作业留下的 socket 文件不影响新 server 重新绑定。
  - 引擎通过 `UMBPStoreConnector` 接入（`mode: standalone`，`kv_load_failure_policy: recompute`）。
- **Draft**：权重不变，仍是随 checkpoint 发布的 DSpark head（`mtp.0-2`），按 checkpoint 的 FP8 配置加载。它的 attention 层和 target 使用同一种 KV cache 类型，所以 target 和 draft 都用 per-tensor `fp8_e4m3` KV。

**依据：本地「不开 offloading（B0）」与「UMBP（S1）」的对比。**
- 条件：在一台 MI355X（GPU 0、1）上，用本 recipe 在启用 fp8 KV 之前的引擎参数（默认 `fp8_ds_mla` KV）跑 AgentX 回放。容器是 `nightly-96710245`，Python 代码是 vLLM main `595b1514e8`。B0 就是 #4992 的 DP2 配置，S1 在 B0 基础上加 UMBP，池子 449 GB。
- 每个点只跑一次 agentx-fast（计时 1200 s，开始计时时 prefix cache 是冷的），不是 CI 结果。
- 和 CI 一样，开 UMBP 后 extensible KV 会被自动关闭，因此 S1 的 GPU KV 比 B0 少 4.2%（每个 rank 26.71M 对 27.89M token）。

结果（每 GPU 总吞吐，B0 → S1）：

| 并发 | 吞吐 | 从 UMBP 恢复的 prompt | 说明 |
|---|---|---|---|
| c128 | 193,099 → 183,984（−4.7%） | 0.46% | TTFT p90 2.32 → 2.62 s，TPOT 均值 24.8 → 26.6 ms |
| c256 | 250,435 → 275,836（**+10.1%**） | 0.67% | TTFT p90 5.12 → 4.10 s，p90 interactivity +27% |
| c384 | 143,888 → 291,619（**+103%**） | 21.5% | B0 的 GPU prefix 命中降到 73%，TTFT p90 从 265 s 降到 7.9 s |
| c544（不在本 PR 中） | 46,641 → 287,025（**+515%**） | 62.0% | B0 的 GPU prefix 命中只剩 27.8%；S1 的吞吐与 c384 持平，但 TTFT p50 / p90 为 21.9 / 58.7 s，有 20 次 preemption |

- **收益取决于 GPU KV 放不下的程度。**
  - c384 下 B0 已经过载：吞吐比 c256 还低。UMBP 让需要计算的 prefill 减半，吞吐翻倍。
  - c256 下 UMBP 让需要计算的 prefill 减少 11%，吞吐提升 10%。
- **c128 没有收益。** GPU KV 基本够用，GPU prefix 命中 90%，UMBP 只恢复了 0.46% 的 prompt，抵不过 GPU KV 变少和 store 的开销。吞吐 −4.7%，接近单次 fast 运行约 4.5% 的噪声，但 TTFT 和 TPOT 的变化方向一致，所以本 PR 不包含 c128。
- **c544 已经过了拐点，不放进本 PR。** 不开 offloading 时服务崩溃：GPU prefix 命中只剩 27.8%，72% 的 prompt 要重算，吞吐只有 46,641。UMBP 恢复了 62% 的 prompt，吞吐保持在 287,025，与 c384 持平，但 TTFT p50 / p90 升到 21.9 / 58.7 s，3.6% 的 step 超过 cudagraph 上限，并出现 20 次 preemption。因此开 UMBP 时吞吐在 c384 附近达到上限。
- **c544 的 S1 第一次运行挂死，重跑成功。** 第一批请求到达时，DP0 的 worker 在运行时 JIT 编译 warmup 没覆盖到的 Triton kernel（包括 indexer 的 prefill 元数据），并一直持有 GIL。DP1 在 dummy batch 的跨 DP all_reduce 里等待，超过 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`（300 s）后引擎崩溃。重跑时 Triton cache 已部分预热，同样的停顿约 75 s 后恢复。所有运行（包括 B0）都有运行时 JIT 的警告，所以这不是 UMBP 特有的问题。
- **其他检查都正常。**
  - 所有点都没有发生 UMBP 淘汰；c384 及以下没有 preemption。
  - c384 下 1,024–2,048 token 的 step 只占 0.7%，和 c256 相当，cudagraph 上限 1,024 不需要调高。
- **和 #4992 的绝对值不能直接比。** #4992 的 c256 是 1 小时本地运行，镜像 `nightly-083fbac02`，结果为 310,739 tok/s/GPU。上表 B0 是较短、开始时 cache 为冷的 fast 运行，所以 UMBP 的收益应以表中的 B0 为基准，不要和这个绝对值比。

**风险**：
- 第一次 CI 运行（https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43151259956）所有 job 都失败了，原因是 socket 放在 `/tmp` 下，worker 连不上 server（`cannot attach to the UMBP standalone server at unix:///tmp/umbp/standalone.grpc.sock`）。上面的宿主机目录挂载已修复了这个路径问题。server 和 worker 之间还要通过 HIP IPC 交换 GPU 显存句柄；enroot 共享宿主机的 IPC 和 PID namespace，这一步应该没问题，但要等下一次 sweep 才能确认。
- 下面的本地测试是在换成 `nightly-be47189` 和 `fp8_e4m3` 之前做的：用的是 `nightly-96710245` 镜像的 native 部分加上更新的 vLLM Python 代码，KV 是默认的 `fp8_ds_mla`，而且是 fast 模式。fp8 KV 会改变 GPU KV 容量和 UMBP 每个 block 的大小，本地没有测过，这次 sweep 是它第一次端到端运行。
- CI 节点上的 Triton cache 如果是冷的，第一批请求触发的 JIT 停顿可能超过 300 s 的 execute-model 超时，就像本地 c544 第一次运行那样，导致该点失败。c256 和 c384 会不会遇到，要看 sweep 结果。
- 本部署方式（先启动 `umbp_standalone_server`，再用 `vllm serve --kv-transfer-config …` 起服务）对应的 vLLM recipe 必须先在 vllm-project/recipes 合入，本 PR 才能合入。

**AI 模型使用说明**：使用 `claude-opus-5-5[1m]`（Claude Opus 5.5，1M context），通过 Claude Code。工作内容：阅读现有配置、recipe 和 srt-slurm 服务代码；起草 master config 键、recipe override、`perf-changelog.yaml` 记录和本描述；运行 matrix 生成、用固定版本的 srtctl 渲染 recipe、`infx.matrix.changed` 和 sweep planner；运行本地 B0 / S1 的 AgentX 测试并汇总结果。没有委派给其他 agent。方案经人工审阅，拓扑（DP2）、并发点和 UMBP 模式由人工决定。

**改动类型**：配置变更。

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
