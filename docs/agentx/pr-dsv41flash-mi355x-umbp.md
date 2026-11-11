# [AMD][AgentX] DeepSeek-V4.1-Flash MI355X vLLM: add DP2 concurrency 256 and 384 with UMBP KV offloading / [AMD][AgentX] DeepSeek-V4.1-Flash MI355X vLLM：新增 DP2 并发 256、384 点并启用 UMBP KV offloading

## Summary

The MI355X DeepSeek-V4.1-Flash vLLM AgentX curve stops at concurrency 64. Above that, the GPU KV cache stops holding every live session's prefix. Evicted prefixes are then recomputed, and prefill crowds out decode. vllm-project/vllm#64715 (merged 2026-11-10) adds UMBP standalone support for DeepSeek-V4.1-Flash on ROCm. With it, the engine offloads KV blocks to a node-local host-DRAM pool owned by MORI's `umbp_standalone_server` and restores them on a prefix hit instead of recomputing.

This PR adds a new config key, `dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`, with two points: concurrency 256 and 384. Evals run at c384. Concurrency 128 is left out: at c128 UMBP did not help in local runs (see below), and #4992 already covers it without offloading.

- **Image:** `vllm/vllm-openai-rocm:nightly-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf` (digest `sha256:6eeac5c6…`). It is the first nightly with #64715, and its MORI v1.2.3.post1 is built with UMBP. The existing TP4/TP2 key and the TP8 c1 key are untouched and are not rerun.
- **Topology:** two GPUs per server with attention DP2. The MoE stays tensor-parallel across both ranks (`tensor-parallel-size 1`, `data-parallel-size 2`, no expert parallelism). vllm-router 0.1.14 fronts the ranks with `consistent_hash`, so each session keeps its rank and its prefix. Engine and router settings match the no-offload DP2 points in #4992, the current best MI355X DP2 baseline: prefill cadence (`prefill-schedule-interval 11`) and no `--monokernel`. The only difference is UMBP, so comparing the two sweeps isolates its effect.
- **Recipe:** the overrides are named `override_dp2_umbp_c{256,384}` and each row selects its own. #4992 adds `override_dp2_c{128,256,384}` to the same recipe file without `KV_OFFLOADING`, so automatic variant matching would select both sets for a UMBP point.
- **KV offloading:** the recipe launches `umbp_standalone_server` as a critical srt-slurm `generic` service that starts before the workers and becomes ready once it logs `[StandaloneServer] listening`.
  - The server owns one 449 GB pool: `dram-utilization` 0.60 of the two-GPU node share, on 4 KiB pages. Both DP ranks share it.
  - The engine attaches through `UMBPStoreConnector` (`mode: standalone`, `kv_load_failure_policy: recompute`).
- **Draft:** unchanged. It is the shipped DSpark head (`mtp.0-2`), loaded with the checkpoint's FP8 config.

### Why: local no-offload (B0) vs UMBP (S1) runs

Local AgentX replays on one MI355X node (GPUs 0-1) with this recipe's engine settings: the `nightly-96710245` container with vLLM main `595b1514e8` Python code. B0 is #4992's DP2 configuration; S1 adds UMBP with the 449 GB pool. Each point is a single InferenceX agentx-fast run (1200 s profile, prefix cache cold at the start), not a CI run. As in CI, UMBP turns off the extensible KV cache, so S1 has 4.2% less GPU KV than B0 (26.71M vs 27.89M tokens per rank).

| Metric | c128 B0 → S1 | c256 B0 → S1 | c384 B0 → S1 |
|---|---|---|---|
| Total throughput per GPU (tok/s) | 193,099 → 183,984 (−4.7%) | 250,435 → 275,836 (**+10.1%**) | 143,888 → 291,619 (**+103%**) |
| TTFT p50 / p90 (s) | 0.87 / 2.32 → 0.88 / 2.62 | 1.60 / 5.12 → 1.45 / 4.10 | 7.36 / 264.9 → 2.41 / 7.92 |
| TPOT mean (ms) | 24.8 → 26.6 | 50.0 → 43.6 | 102.7 → 69.1 |
| Interactivity p90 (tok/s/user) | 29.2 → 27.4 | 12.9 → 16.4 | 6.6 → 10.4 |
| Prompt tokens from GPU prefix cache (B0 / S1) | 90.0% / 89.4% | 87.4% / 88.1% | 73.0% / 64.6% |
| Prompt tokens restored from UMBP | 0.46% | 0.67% | 21.5% |
| Prompt tokens computed | 10.0% → 10.1% | 12.6% → 11.3% | 27.0% → 13.9% |

- The gain tracks how far the working set exceeds GPU KV. At c384 the no-offload server is overloaded: its GPU prefix hit rate falls to 73%, TTFT p90 reaches 265 s, and throughput drops below c256. UMBP restores 21.5% of prompt tokens, halves the prefill computed, doubles throughput and brings TTFT p90 down to 7.9 s.
- At c256 UMBP restores 0.67% of prompt tokens, cuts prefill computed by 11%, and raises throughput by 10% and p90 interactivity by 27%.
- At c128 GPU KV holds the working set (90% prefix hits). UMBP restores 0.46% and does not pay for the smaller GPU KV and its store traffic: throughput is 4.7% lower, near the ~4.5% noise of a single fast run, with TTFT and TPOT moving the same way. That point is therefore not part of this PR.
- No UMBP eviction (the pool never reached its watermark) and no preemption at any point. Steps of 1,024-2,048 tokens, which include decode steps above the 1,024-token cudagraph capture limit, are 0.7% at c384, as at c256.
- #4992 reports 310,739 tok/s/GPU at c256 from a one-hour local run on `nightly-083fbac02`. The B0 numbers above come from shorter, cold-cache fast runs, so compare UMBP against B0 here rather than against that absolute figure.

### Risks

- First CI run of a standalone UMBP sidecar. The server and the engine run as separate srun steps of one job. They must see the same GPUs, share the host IPC namespace, and share the `/tmp/umbp` socket path inside the job container. A failure here shows up as an engine startup failure ("server is not ready") on the canary.
- The local runs used this image's native binaries with vLLM Python code 59 commits newer, in fast mode. The sweep's one-hour runs are the first end-to-end DP2 + UMBP results on the pinned image.
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

- **镜像**：`vllm/vllm-openai-rocm:nightly-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf`（digest `sha256:6eeac5c6…`）。这是第一个包含 #64715 的 nightly，其中的 MORI v1.2.3.post1 编译时开启了 UMBP。已有的 TP4/TP2 键和 TP8 c1 键不变，也不会重跑。
- **拓扑**：每个服务 2 张卡，attention 用 DP2，MoE 在两个 rank 间做张量并行（`tensor-parallel-size 1`、`data-parallel-size 2`，不开 expert parallelism）。前端是 vllm-router 0.1.14，策略为 `consistent_hash`，同一会话固定落在同一个 rank 上，前缀也留在那里。引擎和 router 参数与 #4992 中不开 offloading 的 DP2 点完全一致：prefill cadence 相同（`prefill-schedule-interval 11`），也都不开 `--monokernel`。#4992 是目前 MI355X DP2 最好的基线，两边唯一的区别是 UMBP，对比两次 sweep 就能单独看出 UMBP 的效果。
- **Recipe**：override 命名为 `override_dp2_umbp_c{256,384}`，每一行显式指定自己的 override。#4992 在同一个 recipe 文件里加了 `override_dp2_c{128,256,384}`，且没有设置 `KV_OFFLOADING`，如果靠自动匹配，UMBP 的点会同时匹配到两组 override。
- **KV offloading**：recipe 把 `umbp_standalone_server` 作为 srt-slurm 的 `generic` 服务启动，标记为 critical，在 worker 之前启动，日志出现 `[StandaloneServer] listening` 即视为就绪。
  - server 管理一个 449 GB 的池子，大小是两卡份额的 `dram-utilization` 0.60，使用 4 KiB 页，两个 DP rank 共用。
  - 引擎通过 `UMBPStoreConnector` 接入（`mode: standalone`，`kv_load_failure_policy: recompute`）。
- **Draft**：不变，仍是随 checkpoint 发布的 DSpark head（`mtp.0-2`），按 checkpoint 的 FP8 配置加载。

**依据：本地「不开 offloading（B0）」与「UMBP（S1）」的对比。**
- 条件：在一台 MI355X（GPU 0、1）上，用本 recipe 的引擎参数跑 AgentX 回放。容器是 `nightly-96710245`，Python 代码是 vLLM main `595b1514e8`。B0 就是 #4992 的 DP2 配置，S1 在 B0 基础上加 UMBP，池子 449 GB。
- 每个点只跑一次 agentx-fast（计时 1200 s，开始计时时 prefix cache 是冷的），不是 CI 结果。
- 和 CI 一样，开 UMBP 后 extensible KV 会被自动关闭，因此 S1 的 GPU KV 比 B0 少 4.2%（每个 rank 26.71M 对 27.89M token）。

结果（每 GPU 总吞吐，B0 → S1）：

| 并发 | 吞吐 | 从 UMBP 恢复的 prompt | 说明 |
|---|---|---|---|
| c128 | 193,099 → 183,984（−4.7%） | 0.46% | TTFT p90 2.32 → 2.62 s，TPOT 均值 24.8 → 26.6 ms |
| c256 | 250,435 → 275,836（**+10.1%**） | 0.67% | TTFT p90 5.12 → 4.10 s，p90 interactivity +27% |
| c384 | 143,888 → 291,619（**+103%**） | 21.5% | B0 的 GPU prefix 命中降到 73%，TTFT p90 从 265 s 降到 7.9 s |

- **收益取决于 GPU KV 放不下的程度。**
  - c384 下 B0 已经过载：吞吐比 c256 还低。UMBP 让需要计算的 prefill 减半，吞吐翻倍。
  - c256 下 UMBP 让需要计算的 prefill 减少 11%，吞吐提升 10%。
- **c128 没有收益。** GPU KV 基本够用，GPU prefix 命中 90%，UMBP 只恢复了 0.46% 的 prompt，抵不过 GPU KV 变少和 store 的开销。吞吐 −4.7%，接近单次 fast 运行约 4.5% 的噪声，但 TTFT 和 TPOT 的变化方向一致，所以本 PR 不包含 c128。
- **其他检查都正常。**
  - 三个点都没有发生 UMBP 淘汰，也没有 preemption。
  - c384 下 1,024–2,048 token 的 step 只占 0.7%，和 c256 相当，cudagraph 上限 1,024 不需要调高。
- **和 #4992 的绝对值不能直接比。** #4992 的 c256 是 1 小时本地运行，镜像 `nightly-083fbac02`，结果为 310,739 tok/s/GPU。上表 B0 是较短、开始时 cache 为冷的 fast 运行，所以 UMBP 的收益应以表中的 B0 为基准，不要和这个绝对值比。

**风险**：
- 这是 standalone UMBP sidecar 第一次在 CI 上运行。server 和引擎是同一个作业里的不同 srun step，需要看到相同的 GPU、共享主机 IPC 命名空间，并共用作业容器内 `/tmp/umbp` 下的 socket。如果这一条不满足，canary 会在引擎启动阶段失败（"server is not ready"）。
- 本地测试用的是该镜像的 native 部分，Python 代码比镜像新 59 个 commit，而且是 fast 模式。sweep 中的 1 小时运行，才是固定镜像上第一次 DP2 + UMBP 的端到端结果。
- 本部署方式（先启动 `umbp_standalone_server`，再用 `vllm serve --kv-transfer-config …` 起服务）对应的 vLLM recipe 必须先在 vllm-project/recipes 合入，本 PR 才能合入。

**AI 模型使用说明**：使用 `claude-opus-5-5[1m]`（Claude Opus 5.5，1M context），通过 Claude Code。工作内容：阅读现有配置、recipe 和 srt-slurm 服务代码；起草 master config 键、recipe override、`perf-changelog.yaml` 记录和本描述；运行 matrix 生成、用固定版本的 srtctl 渲染 recipe、`infx.matrix.changed` 和 sweep planner；运行本地 B0 / S1 的 AgentX 测试并汇总结果。没有委派给其他 agent。方案经人工审阅，拓扑（DP2）、并发点和 UMBP 模式由人工决定。

**改动类型**：配置变更。

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
