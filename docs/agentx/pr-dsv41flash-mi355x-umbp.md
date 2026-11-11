# [AMD][AgentX] DeepSeek-V4.1-Flash MI355X vLLM: add DP2 concurrency 128-384 with UMBP KV offloading / [AMD][AgentX] DeepSeek-V4.1-Flash MI355X vLLM：新增 DP2 并发 128-384 点并启用 UMBP KV offloading

## Summary

The MI355X DeepSeek-V4.1-Flash vLLM AgentX curve stops at concurrency 64. Above that, the GPU KV cache stops holding every live session's prefix. Evicted prefixes are then recomputed, and prefill crowds out decode. vllm-project/vllm#64715 (merged 2026-11-10) adds UMBP standalone support for DeepSeek-V4.1-Flash on ROCm. With it, the engine offloads KV blocks to a node-local host-DRAM pool owned by MORI's `umbp_standalone_server` and restores them on a prefix hit instead of recomputing.

This PR adds a new config key, `dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`, with three points: concurrency 128, 256 and 384. Evals run at c384.

- **Image:** `vllm/vllm-openai-rocm:nightly-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf` (digest `sha256:6eeac5c6…`). It is the first nightly with #64715, and its MORI v1.2.3.post1 is built with UMBP. The existing TP4/TP2 key and the TP8 c1 key are untouched and are not rerun.
- **Topology:** two GPUs per server with attention DP2. The MoE stays tensor-parallel across both ranks (`tensor-parallel-size 1`, `data-parallel-size 2`, no expert parallelism). vllm-router 0.1.14 fronts the ranks with `consistent_hash`, so each session keeps its rank and its prefix. Engine and router settings match the no-offload DP2 points in #4992, the current best MI355X DP2 baseline: same concurrencies, prefill cadence (`prefill-schedule-interval 11`) and no `--monokernel`. The only difference is UMBP, so comparing the two sweeps isolates its effect.
- **Recipe:** the overrides are named `override_dp2_umbp_c{128,256,384}` and each row selects its own. #4992 adds `override_dp2_c{128,256,384}` to the same recipe file without `KV_OFFLOADING`, so automatic variant matching would select both sets for a UMBP point.
- **KV offloading:** the recipe launches `umbp_standalone_server` as a critical srt-slurm `generic` service that starts before the workers and becomes ready once it logs `[StandaloneServer] listening`.
  - The server owns one 449 GB pool: `dram-utilization` 0.60 of the two-GPU node share, on 4 KiB pages. Both DP ranks share it.
  - The engine attaches through `UMBPStoreConnector` (`mode: standalone`, `kv_load_failure_policy: recompute`).
- **Draft:** unchanged. It is the shipped DSpark head (`mtp.0-2`), loaded with the checkpoint's FP8 config.

### Why: local B0 vs UMBP runs

These come from local AgentX replays with this recipe's engine settings, on an MI355X dev container built from vLLM main `77d2790e78` plus #64715. Each is a single 3600 s run with synthetic AL 3.51. They are not CI runs.

| DP2 + expert TP2, c256 | No offload | UMBP standalone | Δ |
|---|---|---|---|
| Total throughput per GPU (tok/s) | 270,323 | 280,607 | +3.8% |
| TTFT mean / p90 (s) | 1.69 / 2.66 | 1.40 / 2.47 | −17% / −7% |
| TPOT mean (ms) | 54.8 | 52.4 | −4.4% |
| Prompt tokens from GPU prefix cache / UMBP | 92.9% / 0 | 93.1% / 0.2% | |

The benefit grows with KV pressure. At TP2 c192, where the GPU KV cache overflows, UMBP raised throughput per GPU by 46% and cut TTFT p90 by 79%. DP2 c128 and c384 were not measured locally; at c384 each rank serves about 192 sessions, so KV pressure and the expected UMBP benefit are higher than at c256.

### Risks

- First CI run of a standalone UMBP sidecar. The server and the engine run as separate srun steps of one job. They must see the same GPUs, share the host IPC namespace, and share the `/tmp/umbp` socket path inside the job container. A failure here shows up as an engine startup failure ("server is not ready") on the canary.
- The nightly is 278 commits ahead of the local test base. The sweep is the first end-to-end run of DP2 + UMBP on it.
- At c384 each rank can reach about 192 decoding sessions × 6 tokens = 1,152 tokens per step, above the 1,024-token cudagraph capture limit shared with the other points. Larger steps run eagerly, which can slow c384 decode. Agentic sessions rarely all decode at once, so the sweep shows whether the limit needs raising for this point.
- The upstream vLLM recipe for this deployment (start `umbp_standalone_server`, then `vllm serve --kv-transfer-config …`) must be merged in vllm-project/recipes before this PR merges.

<details>
<summary>AI model disclosure</summary>

- Model/version: `claude-opus-5-5[1m]` (Claude Opus 5.5, 1M context), via Claude Code.
- Role: read the existing config, recipe and srt-slurm service code; drafted the master-config key, recipe overrides, `perf-changelog.yaml` entry and this description; ran matrix generation, recipe rendering with the pinned srtctl, `infx.matrix.changed` and the sweep planner. No delegated agents. A human reviewed the plan and chose the topology (DP2), the concurrencies and the UMBP mode.

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
- [ ] I have tested my changes locally
- [ ] I have updated documentation if necessary
- [x] **If `python -m infx.matrix.changed --base origin/main` lists affected config keys, I have appended one new entry covering them to the physical end of `inferencex-e2e/perf-changelog.yaml`; if it lists none, I have added none. I have not edited historical entries**
- [ ] **If this PR appends an `inferencex-e2e/perf-changelog.yaml` entry, it carries exactly one primary sweep label** (a maintainer applies it on fork PRs): `full-sweep-fail-fast` (recommended), `full-sweep-enabled`, or `non-canary-full-sweep-enabled`. Optional modifiers `all-evals`, `evals-only`, and `agentx-fast` require a primary label; the last two block reuse while applied.
- [ ] **Before merging via reuse, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) has commented `/use <run_id>` (or the legacy `/reuse-sweep-run`) on this PR**. Do this **only once there is a final full sweep that is all green with evals passing**, since after this comment the primary sweep label will no longer automatically kick off new sweeps. Remove and re-add the primary sweep label to force a new sweep.

</details>

<details>
<summary>中文</summary>

MI355X 上 DeepSeek-V4.1-Flash 的 vLLM AgentX 曲线目前只到并发 64。并发更高时，GPU KV cache 放不下所有活跃会话的前缀，被淘汰的前缀只能重新计算，prefill 会挤占 decode。vllm-project/vllm#64715（2026-11-10 合入）在 ROCm 上为 DeepSeek-V4.1-Flash 支持了 UMBP standalone 模式：引擎把 KV block 卸载到节点本地的主机内存池（由 MORI 的 `umbp_standalone_server` 管理），前缀命中时直接恢复，不再重算。

本 PR 新增配置键 `dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`，包含并发 128、256、384 三个点，eval 在 c384 上跑。

- **镜像**：`vllm/vllm-openai-rocm:nightly-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf`（digest `sha256:6eeac5c6…`）。这是第一个包含 #64715 的 nightly，其中的 MORI v1.2.3.post1 编译时开启了 UMBP。已有的 TP4/TP2 键和 TP8 c1 键不变，也不会重跑。
- **拓扑**：每个服务 2 张卡，attention 用 DP2，MoE 在两个 rank 间做张量并行（`tensor-parallel-size 1`、`data-parallel-size 2`，不开 expert parallelism）。前端是 vllm-router 0.1.14，策略为 `consistent_hash`，同一会话固定落在同一个 rank 上，前缀也留在那里。引擎和 router 参数与 #4992 中不开 offloading 的 DP2 点完全一致：并发点相同，prefill cadence 相同（`prefill-schedule-interval 11`），也都不开 `--monokernel`。#4992 是目前 MI355X DP2 最好的基线，两边唯一的区别是 UMBP，对比两次 sweep 就能单独看出 UMBP 的效果。
- **Recipe**：override 命名为 `override_dp2_umbp_c{128,256,384}`，每一行显式指定自己的 override。#4992 在同一个 recipe 文件里加了 `override_dp2_c{128,256,384}`，且没有设置 `KV_OFFLOADING`，如果靠自动匹配，UMBP 的点会同时匹配到两组 override。
- **KV offloading**：recipe 把 `umbp_standalone_server` 作为 srt-slurm 的 `generic` 服务启动，标记为 critical，在 worker 之前启动，日志出现 `[StandaloneServer] listening` 即视为就绪。
  - server 管理一个 449 GB 的池子，大小是两卡份额的 `dram-utilization` 0.60，使用 4 KiB 页，两个 DP rank 共用。
  - 引擎通过 `UMBPStoreConnector` 接入（`mode: standalone`，`kv_load_failure_policy: recompute`）。
- **Draft**：不变，仍是随 checkpoint 发布的 DSpark head（`mtp.0-2`），按 checkpoint 的 FP8 配置加载。

**依据：本地「不开 offloading」与 UMBP 的对比。** 数据来自本地 MI355X 开发容器（vLLM main `77d2790e78` + #64715），用本 recipe 的引擎参数跑 AgentX 回放，每个臂单次 3600 s，synthetic AL 3.51，不是 CI 结果。DP2 + expert TP2 c256 下，每 GPU 总吞吐 +3.8%（270,323 → 280,607 tok/s），TTFT 均值 / p90 下降 17% / 7%，TPOT 均值下降 4.4%；有 0.2% 的 prompt token 从 UMBP 恢复。KV 压力越大收益越明显：在 GPU KV 装不下的 TP2 c192 下，UMBP 让每 GPU 吞吐提升 46%，TTFT p90 下降 79%。DP2 c128 和 c384 本地没有测过；c384 时每个 rank 约 192 个会话，KV 压力和预期的 UMBP 收益都比 c256 更大。

**风险**：
- 这是 standalone UMBP sidecar 第一次在 CI 上运行。server 和引擎是同一个作业里的不同 srun step，需要看到相同的 GPU、共享主机 IPC 命名空间，并共用作业容器内 `/tmp/umbp` 下的 socket。如果这一条不满足，canary 会在引擎启动阶段失败（"server is not ready"）。
- 该 nightly 比本地测试的 base 新 278 个 commit，DP2 + UMBP 在它上面的第一次端到端运行就是这次 sweep。
- c384 时每个 rank 最多约 192 个会话同时 decode，每个 step 约 192 × 6 = 1,152 个 token，超过了与其他点共用的 cudagraph capture 上限 1,024，超出的 step 会走 eager 模式，可能拖慢 c384 的 decode。agentic 负载中会话很少同时 decode，是否需要为这个点调高上限，要看 sweep 结果。
- 本部署方式（先启动 `umbp_standalone_server`，再用 `vllm serve --kv-transfer-config …` 起服务）对应的 vLLM recipe 必须先在 vllm-project/recipes 合入，本 PR 才能合入。

**AI 模型使用说明**：使用 `claude-opus-5-5[1m]`（Claude Opus 5.5，1M context），通过 Claude Code。工作内容：阅读现有配置、recipe 和 srt-slurm 服务代码；起草 master config 键、recipe override、`perf-changelog.yaml` 记录和本描述；运行 matrix 生成、用固定版本的 srtctl 渲染 recipe、`infx.matrix.changed` 和 sweep planner。没有委派给其他 agent。方案经人工审阅，拓扑（DP2）、并发点和 UMBP 模式由人工决定。

**改动类型**：配置变更。

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
