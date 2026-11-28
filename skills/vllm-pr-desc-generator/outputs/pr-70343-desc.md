**Title:** `[ROCm][Bugfix] Gate EPLB rearrangement on per-layer rank load imbalance` (vllm-project/vllm#70343)

## Overview

The ROCm rearrange gate (#47206) measures rank load imbalance after summing over all MoE layers. Per-layer hot spots on different ranks cancel out in that sum, so after the first rearrangement EPLB skips nearly every later one. This PR measures imbalance per layer instead.

## Claims

- The gate now lets through rearrangements that fix per-layer imbalance, and still skips those with less than 5% gain.
- Qwen3-30B-A3B DP8+EP: measured balancedness is 0.80 vs 0.76 where the two gates decide differently. GSM8K is unchanged.
- DeepSeek-V4.1-Flash DEP8 AgentX c512 with MegaMoE EPLB: 4 of 10 rearrangements move, and p90 interactivity is 5.2% (R=0) / 6.4% (R=8) above EPLB off.
- Only the ROCm sync rearrange path changes.

## Validation

**Unit tests** (commit `6cc29a7545`, MI355X): `pytest tests/distributed/test_eplb_utils.py tests/distributed/test_eplb_algo.py tests/distributed/test_eplb_events.py` → `59 passed, 2 skipped in 102.42s`. The 5 new cases take ≤0.09 s each. `test_rocm_rearrange_gate_uses_per_layer_imbalance` drives `EplbState.rearrange()` with two layers whose hot rank alternates; it fails on main and passes here.

**E2E** (8×MI355X, Qwen3-30B-A3B, DP8+EP, sync EPLB, `window_size=step_interval=200`, GSM8K 5-shot at 256 concurrency as the load). Two runs per gate, in opposite order; both runs gave the same decisions.

| Gate | Decisions (5 per run) | Balancedness after decision #4 | GSM8K strict (run 1 / 2) |
|---|---|---|---|
| main | move, skip ×4 | 0.765 / 0.761 | 0.894 / 0.896 |
| this PR | move, skip ×2, move ×2 | 0.801 / 0.795 | 0.895 / 0.895 |

**AgentX** (8×MI355X, DeepSeek-V4.1-Flash DEP8 with AITER MegaMoE EPLB from #70258 plus this PR; InferenceX#5056 `override_dep8_c512`, agentx-fast 1200 s at concurrency 512; EPLB default schedule, sync `torch_nccl`; one run per config):

<!-- l3-table -->
| Config | Per-GPU throughput (tok/s) | Interactivity p90 (tok/s/user) | Interactivity p50 | TTFT mean / p95 (s) | TPOT mean / p50 (ms) | E2E latency mean (s) | Requests OK | Rearrangements moved / skipped |
|---|---|---|---|---|---|---|---|---|
| EPLB off | 213,793 | **29.44** | 59.53 | 5.19 / 12.62 | 23.71 / 16.80 | 21.13 | 17,471 | — |
| EPLB on, R=0 | 223,572 (+4.6%) | **30.97** (+5.2%) | 63.98 | 4.58 / 10.15 | 21.36 / 15.63 | 19.64 | 18,201 | 4 / 6 |
| EPLB on, R=8 | 218,127 (+2.0%) | **31.32** (+6.4%) | 59.24 | 4.73 / 10.60 | 22.29 / 16.88 | 20.33 | 17,886 | 4 / 6 |
<!-- /l3-table -->

Interactivity p90 = 1 / p90 ITL (slow-tail tokens/s per user, higher is better); percentages are vs EPLB off.

With EPLB, p90 interactivity improves 5.2% (R=0) / 6.4% (R=8) and per-GPU throughput 4.6% / 2.0% over EPLB off; each config ran once, so treat small differences as noise-level. Each config made 10 rearrangement decisions: 4 moved (0.3–0.9 s each), 6 were skipped by the gate. The current gate was not measured on this tree.

<details>
<summary>Commands</summary>

```bash
VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3-30B-A3B --max-model-len 8192 \
  --data-parallel-size 8 --enable-expert-parallel --enable-eplb \
  --eplb-config '{"window_size":200,"step_interval":200,"num_redundant_experts":0,"use_async":false,"communicator":"torch_nccl","log_balancedness":true,"log_balancedness_interval":10}'

lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 \
  --model_args "model=Qwen/Qwen3-30B-A3B,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None,num_concurrent=256,tokenized_requests=False"
```
</details>

## Details

**Root cause:** each MoE layer waits for its busiest rank. Summing rank loads over layers before taking max / mean hides per-layer imbalance whenever different layers are hot on different ranks.

**The fix:** `_rank_load_imbalance` (now a module-level function in `eplb_state.py`) returns Σ_layers max_rank / Σ_layers mean_rank. This tracks MoE time and is the inverse of the balancedness EPLB already logs (`_compute_eplb_load_stats`). The 5% threshold, replica load splitting, and the conditions under which the gate runs are unchanged.

**Limitation:** async EPLB (the default `use_async=True`) has no gate; this PR does not change that.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
