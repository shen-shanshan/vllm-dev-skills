**Title:** `[ROCm][Perf][DSv4.1] Support dynamic EPLB with AITER MegaMoE` (vllm-project/vllm#70258)

## Overview

Enable `--enable-eplb` for DeepSeek-V4.1 on ROCm with `--moe-backend aiter_mega_moe`, which previously refused it, matching NVIDIA's DeepGEMM MegaMoE.

## Claims

- AITER MegaMoE serves with `--enable-eplb` and any `num_redundant_experts`; live rearrangements keep outputs correct (GSM8K unchanged).
- AgentX DEP8 at concurrency 512: p90 interactivity +5.2% (R=0) / +6.4% (R=8), per-GPU throughput +4.6% / +2.0% over EPLB off.
- With EPLB off, the MegaMoE path is unchanged. Only ROCm DeepSeek-V4/V4.1 model code changes.

## Validation

**Unit tests** (commit `5489b27b4c`, MI355X): `pytest tests/models/test_deepseek_v4_aiter_mega_moe.py tests/distributed/test_eplb_utils.py` → `45 passed in 33.19s` (25 in the MegaMoE file). New cases cover redundant-slot weight loading, `get_expert_weights` views, and the placement kernel against a torch reference.

**E2E** on 8×MI355X, DeepSeek-V4.1-Flash DEP8 with the stock MegaMoE config of SemiAnalysisAI/InferenceX#5056, sync `torch_nccl` EPLB:

| Config | GSM8K strict (8-shot, 1319) | AgentX c512: per-GPU tput (tok/s) | p90 interactivity (tok/s/user) | TPOT p50 (ms) | Rearrangements moved / skipped |
|---|---|---|---|---|---|
| EPLB off | 0.9280 | 213,793 | 29.44 | 16.80 | — |
| EPLB, R=0 | 0.9363 | 223,572 (+4.6%) | 30.97 (+5.2%) | 15.63 | 4 / 6 |
| EPLB, R=8 | 0.9257 | 218,127 (+2.0%) | 31.32 (+6.4%) | 16.88 | 4 / 6 |

- GSM8K: stderr ±0.007; the first live rearrangement lands mid-eval (`window_size=100`, `step_interval=200`).
- AgentX: InferenceX agentx-fast (1200 s), one run per config, vLLM's default schedule (`window_size=1000`, `step_interval=3000`).
- p90 interactivity = 1 / p90 ITL. Each live rearrangement takes 0.3–1 s.
- Idle EPLB (nothing ever moves), `vllm bench serve` random 2048/1024 at concurrency 512: throughput and TPOT within run-to-run spread, median ITL +1.2%. KV cache capacity drops 8% for the EPLB buffers.
- These runs used this patch on an earlier main (the rebase had no conflicts), with the ROCm rearrange gate fix of #70343 applied: without it, ROCm's layer-summed gate skips most rearrangements after the first.

<details>
<summary>Server command</summary>

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash --language-model-only --tokenizer-mode deepseek_v41 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend mori --moe-backend aiter_mega_moe --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 128 --max-num-batched-tokens 32768 \
  --enable-eplb --eplb-config '{"num_redundant_experts":8,"use_async":false,"communicator":"torch_nccl"}'
```
Remaining flags as in InferenceX#5056 `override_dep8_c512`.
</details>

## Details

In DEP8 every MegaMoE launch waits for the most-loaded EP rank; EPLB moves and replicates hot experts to even this out. `DeepseekV4AiterMegaMoEExperts` now plugs into the shared EPLB framework the same way `DeepGemmMegaMoEExperts` does:

- `num_experts` counts physical slots, and `weight_loader` fills each slot from EPLB's initial placement.
- `get_expert_weights` returns expert-major views (AITER's a16w4 shuffle keeps each expert contiguous), so migration copies in place and the shared `MegaMoEV2` weight pointers and CUDA graphs stay valid.
- A Triton kernel `_eplb_place_experts_kernel` (reusing `eplb_map_and_record`) maps routes to replicas, records load, and handles the fused shared expert in one launch.
- `deepseek_v41/amd/model.py` sets `supports_eplb = True`. V4 still refuses EPLB.

Limitations:
- DSpark draft layers are not rebalanced (draft EPLB is off on ROCm).
- Without NIXL the default communicator is `torch_gloo`; validation used `torch_nccl`.

#69097 (amd_kernels DEP8 routes) replaces the MegaMoE forward and ships a static placement (`expert_perm`). Whichever lands second should make `expert_perm` and `--enable-eplb` mutually exclusive.

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
