# [Docs][ROCm] Add DeepSeek-V4.1-Flash MI355X DP2 AgentX preset with UMBP KV offload

Repository: `vllm-project/vllm` · branch `recipes-dsv41flash-mi355x-umbp` · commit `45b3afe4a5`

<!-- markdownlint-disable -->

## Overview

Adds the `mi355x_agentx_dp2_umbp` preset to `recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml` for InferenceX [#5036](https://github.com/SemiAnalysisAI/InferenceX/pull/5036) (`dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`), and records UMBP as verified for the model.

Depends on vllm-project/recipes#XXXX, which adds the `umbp` KV Offload option this preset selects. The recipe check renders against recipes `main`, so it passes once that PR lands.

## Claims

- The recipes site renders the configuration InferenceX #5036 benchmarks on MI355X:
  - two GPUs per server (`--data-parallel-size 2`), attention DP2 with the MoE tensor-parallel across both ranks (`--enable-expert-parallel` removed), behind the consistent-hash vLLM Router;
  - UMBP KV offload (`UMBPStoreConnector`, standalone server tab);
  - `--kv-cache-dtype fp8_e4m3` (#66036), `--prefill-schedule-interval 11`, Engram tables in host memory, 16384-token prefill budget, cudagraph capture up to 1024;
  - concurrency points 256 and 384, on the nightly carrying #64715 and #66036.
- `kv_offload_support: { umbp: verified }` makes the opt-in UMBP option available for this model.

## Validation

- From a vllm-project/recipes checkout with the `umbp` option, `node scripts/validate-source.mjs --source vllm --dir <this branch>/recipes` passes: `✓ vllm recipes … are valid against the renderer`.
- The rendered command matches the InferenceX recipe flag for flag: the per-point `vllm serve` / `docker run`, the UMBP Server companion (socket address equal to the connector `endpoint`) and the vLLM Router (`--intra-node-data-parallel-size 2`). `pnpm test:presets` and `pnpm test:kv-offload` pass with this branch as the vLLM source.
- Local MI355X AgentX replays (agentx-fast, 1200 s, single runs) of this configuration on the default `fp8_ds_mla` KV and an earlier nightly, against the same configuration without offloading:

  | Concurrency | Total tok/s/GPU, no offload → UMBP | Prompt tokens restored from UMBP |
  |---|---|---|
  | 128 (not a point) | 193,099 → 183,984 (−4.7%) | 0.46% |
  | 256 | 250,435 → 275,836 (+10.1%) | 0.67% |
  | 384 | 143,888 → 291,619 (+103%) | 21.5% |

  The `fp8_e4m3` KV cache and the pinned nightly are qualified by the InferenceX sweep of #5036.

## Details

- `single_node_dep` does not inherit the MI355X kernels the TP rows take from `strategy_overrides.single_node_tp` (MXFP4 indexer, block size 128, AITER sparse MLA), so the preset restates them. `--monokernel` stays off, since mono decode rejects DP.
- The preset is opt-in, not `default: true`: it is a benchmark lane for high concurrency, where the working set outgrows GPU KV. At concurrency 128 UMBP did not help.
- The runbook records the pool sizing InferenceX uses (449 GB per two-GPU server) and the evidence above, with no launch commands, as SCHEMA.md requires.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity. (Recipe YAML only; reuses the DEP strategy, its router companion and the preset delta fields.)
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

AI assistance: drafted with `claude-opus-5-5[1m]` (Claude Opus 5.5, 1M context) via Claude Code, which read the recipe and renderer, wrote the preset, ran the validation and wrote this description; reviewed by a human.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
