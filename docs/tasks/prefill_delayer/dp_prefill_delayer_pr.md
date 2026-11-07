**Title:** `[Core][DP] Add a cross-DP prefill delayer scheduling policy (--prefill-schedule-policy delayer)`

## Overview

Adds an opt-in `--prefill-schedule-policy delayer` for MoE data parallelism. The DP ranks exchange their prefill state on every step and take the same decision: hold prefill while they decode, and release it on all ranks in the same step once it is aligned and well filled, within bounded waits. This supersedes #61834, whose author closed it and handed it over, and addresses the feedback on it and on #51199.

## Claims

- New scheduler option `--prefill-schedule-policy {cadence,delayer}`. The default `cadence` is the existing `--prefill-schedule-interval` behavior, unchanged on every platform.
- With `delayer`, `--prefill-schedule-interval N` keeps its meaning as the minimum spacing of prefill steps: the `N - 1` steps after a prefill decode only. The only new knob is `--prefill-max-wait-ms`, a wall-clock bound on how long a waiting request is held.
- The delayer adds no collective. Its state (10 int32s) rides on the existing DP state all-reduce (`ParallelConfig.sync_dp_state`), which then runs every step. Every rank already reaches that sync on every step, so no rank can skip the delayer's exchange.
- Fixes two problems of #61834's rules:
  - **Spurious stall releases.** #61834 released after 10 steps of flat pending tokens, but pending tokens are capped per rank, so the sum can stay flat while requests keep arriving. Here, arrivals anywhere in the group and ranks turning (non-)prefillable also restart the stall count.
  - **Waiting on ranks that alignment cannot help.** A rank whose waiting work cannot be admitted now counts as aligned. A group whose missing rank will finish a decode in time waits for it; only a rank that cannot get work within the hold bound lets a stall release an unaligned group.
- Performance: <TODO: end-to-end results, see Validation>.

## Validation

**Unit tests (CPU)**, branch based on main `fd77011424`:

```bash
pytest tests/v1/core/test_prefill_delayer.py   # 21 passed
pytest tests/v1/core/test_prefill_delayer.py tests/v1/core/test_scheduler.py \
       tests/v1/core/test_async_scheduler.py tests/engine/test_arg_utils.py \
       tests/distributed/test_cpu_all_reduce.py   # 477 passed
pre-commit run --files <changed files>          # all hooks pass, mypy included
```

`tests/v1/core/test_prefill_delayer.py` covers:
- each FIRE/HOLD rule and bound;
- the arrival-driven stall restart;
- blocked and unreachable ranks;
- the scheduler's state report and hold gating, on a real `Scheduler`;
- the engine core syncing every step with the delayer's state;
- a multi-process gloo test (2 and 3 ranks, 300 steps of per-rank random states) that checks `sync_dp_state(extra=...)` returns the exact per-field sums and that every rank takes the decision a single reference delayer takes from those sums.

**GSM8K smoke test** (5-shot, lm_eval `local-completions`, 1319 questions, concurrency 64), DeepSeek-V4.1-Flash on 2x MI355X:
- DP2 with the MoE TP-sharded across both ranks.
- DSpark MTP (5 draft tokens).
- Async scheduling and Model Runner V2, the combination in which #61834 once hung.

| Policy | Flags | GSM8K (strict / flexible) | Mean waiting / running over both ranks |
|---|---|---:|---:|
| `cadence` (default) | — | 0.9303 ± 0.0070 / 0.9303 | 2.5 / 60.5 |
| `delayer` | `--prefill-schedule-interval 10 --prefill-max-wait-ms 5000` | 0.9242 ± 0.0073 / 0.9242 | 15.9 / 45.9 |
| `delayer`, 3 repeats | same | 0.9249 ± 0.0073, 0.9409 ± 0.0065, 0.9303 ± 0.0070 | — |

Across the 4 delayer runs the mean is 0.9301, the same as `cadence`. The difference of any single run is within one standard error, and earlier GSM8K baselines of this model on this node were 0.9204–0.9212. No errors or hangs with the delayer. The higher waiting count shows it holding prefill. Reproduce:

<details>
<summary>Serve and eval commands</summary>

```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
       VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
       VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
       OMP_NUM_THREADS=1 HIP_VISIBLE_DEVICES=0,1
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --port 8333 \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --moe-backend aiter \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --max-model-len 1048576 \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 1024 \
  --max-num-batched-tokens 16384 \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --prefill-schedule-policy delayer --prefill-schedule-interval 10 \
  --prefill-max-wait-ms 5000
# The cadence arm omits the three --prefill-* flags.

lm_eval --model local-completions \
  --model_args "model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://0.0.0.0:8333/v1/completions,num_concurrent=64,max_retries=10,max_gen_toks=2048,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --batch_size auto
```
</details>


**GSM8K on Model Runner V1** (`VLLM_USE_V2_MODEL_RUNNER=0`, otherwise the same command). Speculative decoding is off: DSpark is V2-only, and the V4.1 checkpoint has no classic-MTP draft.

| Policy | Run | GSM8K (strict) | Mean waiting / running | Outcome |
|---|---|---:|---:|---|
| `cadence` | 1 | 0.8984 ± 0.0083 | 1.2 / 61.6 | completed |
| `cadence` | 2 | 0.9030 ± 0.0082 | 1.2 / 61.9 | completed |
| `delayer` | 1 | 0.9242 ± 0.0073 | 15.5 / 46.8 | completed |
| `delayer` | 2 | — | — | **hung** after 3 requests (see below) |

The completed delayer run shows no accuracy loss. Both DP ranks logged identical decision counters (`fill` 1669, `protected` 257, `no_prefill` 54, `align` 14, `stall` 4, `max_wait` 2), which shows the ranks stayed in lockstep. Those counters predate the `hold_` prefix: `fill` covered both releases and holds, and is now split into `fill` and `hold_fill`.

**Known issue, Model Runner V1 only:** delayer run 2 on V1 hung after 3 requests. The same delayer config on V2 completed 4 times out of 4. Since V1 is deprecated, this is recorded but not pursued. What was observed:
- Both EngineCores timed out in `execute_model`.
- On both workers, the async output-copy thread was blocked in `Event.synchronize`, which points to a collective mismatch across the DP ranks.
- The last dumped `SchedulerOutput` was decode-only on both ranks.

**End-to-end performance (not run yet):**
- <TODO: GPU serving benchmark on DeepSeek-V3/V4-class MoE, DP8 + EP8, multi-turn agentic trace, same node, ≥2 runs per arm with variance.>
  - Arms: `cadence` with interval ∈ {1, 8, 16, 32} vs `delayer` with interval ∈ {1, 10, 20}, with and without `--prefill-max-wait-ms`.
  - Concurrency: 16 / 64 / 128 / 256.
  - Metrics: output throughput vs E2E-normalized interactivity, TTFT p50/p90, ITL p50/p90.

**CI coverage:** the new tests are CPU tests (`pytest.mark.cpu_test`). The GPU serving path is not exercised in CI; validation outside CI is the GSM8K smoke test above; the performance runs are still TODO.

## Details

**Problem.** The DP ranks of a MoE model step in lockstep, so a step in which any rank prefills is a long (mixed, often non-graph) step for every rank. If each rank prefills in a fraction `p` of steps independently, `1 - (1 - p)^DP` of steps are long: 57% at `p = 10%` and DP8. `--prefill-schedule-interval` coordinates *when* ranks may prefill (a shared step counter), but not *whether* the step is worth it:
- a cadence step runs whatever prefill each rank has, even one small chunk on one rank;
- a rank whose queue a release cannot drain (`prefill_capacity_bound`) stops throttling and prefills every step, which makes every step long for all ranks.

**Decision** (`vllm/v1/core/sched/prefill_delayer.py`). Every rank reports `prefillable`, `blocked`, `unreachable`, `pending_tokens` (capped at the budget), `decodes`, `kv_high`, `partial`, `queue_hot`, `prefill_ran` and `arrived`. The reports are summed, and every rank runs the same state machine on the sums. All counters advance per step and are identical on all ranks; the only wall-clock check (`queue_hot`) compares each rank's own requests and is summed as a flag. The rules, in order:

1. Protect `interval - 1` steps after a step that ran prefill.
2. FIRE if no rank has prefill or no rank decodes.
3. FIRE on KV pressure or `--prefill-max-wait-ms`.
4. FIRE after 200 held steps (100 with a prompt chunk in progress).
5. HOLD while `prefillable + blocked < dp_size`. While unaligned, a stall (10 steps with no change in the group's work) releases only if some rank cannot get work within the hold bound.
6. FIRE once pending tokens reach 90% of the prefillable ranks' budgets, or on a stall. Otherwise HOLD.

The constants are module-level rather than options, to keep the knob count at two.

**Scheduler** (`Scheduler.prefill_delayer_state`, `defer_prefills`):
- The state report reuses the existing `prefill_scheduled` / `prefill_admitted` flags for `prefill_ran` instead of a token-count heuristic.
- A hold uses the existing `defer_prefills` path and, as before, defers only on a rank with a decode to run. A rank left with an empty batch while its peers step can hang the engine (seen with #61834 on the V2 model runner with spec decode).
- Under the delayer, `prefill_capacity_bound` does not lift a hold: the delayer's own bounds replace it.

**Engine core.** With the delayer, `_has_global_unfinished_reqs` syncs on every step instead of every `dp_sync_interval` steps and passes the delayer's buffer as `extra`. The decision sets `hold_prefills`, which `_should_throttle_prefills` returns. The delayer resets at each DP wave boundary.

**Unaffected by default.** With `prefill_schedule_policy="cadence"` (default):
- `DPEngineCoreProc` builds no delayer, and `_has_global_unfinished_reqs` takes the original branch with the original `sync_dp_state` call;
- `sync_dp_state`'s new `extra` argument defaults to `None` (same tensor, same all-reduce);
- the scheduler's additions are guarded by `prefill_delayer_enabled`, which is `False`.

Behavior is identical on all platforms. Engines without MoE DP (including dense DP ranks, which run with `data_parallel_size=1`) fall back to the local cadence and log once.

**Limitations:**
- TTFT is the tradeoff. At low concurrency the ranks rarely align and prompts wait up to the hold bound.
- `--prefill-max-wait-ms` is a queue-age release, not a hard TTFT limit.
- Rejected combinations: elastic EP and `--enable-fault-tolerance` (both change the DP size at runtime) and custom scheduler classes.
- Main-code diff is about 385 lines including docstrings and comments, and the core lives in a separate module. Left for follow-ups: cost-based release (rent-or-buy on measured step times) and per-rank prefill cost balancing on the same exchange.

**Related.** #61834 (predecessor, closed by its author in favor of this PR). #51199 coordinates prefill through the DP coordinator asynchronously with acks; this PR keeps a synchronous per-step exchange in the engine loop and adds decode protection and a fill target. Happy to consolidate.

AI assistance (Claude Code) was used for this PR; the submitter reviewed every changed line and ran the tests above.

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [ ] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
