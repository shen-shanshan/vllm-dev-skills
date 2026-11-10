**Title:** `[Core][DP] Add a cross-DP prefill delayer scheduling policy (--prefill-schedule-policy delayer)`

## Overview

Adds an opt-in `--prefill-schedule-policy delayer` for MoE data parallelism. On every step the DP ranks exchange their prefill state and take the same decision. They hold prefill while they decode, then release it on all ranks in the same step once it is aligned and well filled, within bounded waits. Supersedes #61834.

## Claims

- New option `--prefill-schedule-policy {cadence,delayer}`. The default `cadence` is the existing `--prefill-schedule-interval` behavior, unchanged on every platform.
- With `delayer`, `--prefill-schedule-interval N` sets the minimum spacing of prefill steps: the `N - 1` steps after a prefill decode only. The only new knob is `--prefill-max-wait-ms`, which releases held prefill, including during those steps, once the oldest waiting request has queued that long.
- No new collective. The delayer's state (10 int32s) rides on the existing DP state all-reduce (`ParallelConfig.sync_dp_state`), which then runs every step. Every rank reaches that sync on every step, so no rank can skip the exchange.
- Performance (preliminary: AgentX c128, 2x MI355X, InferenceX fast replay, one run per point; see Validation): with `--prefill-schedule-interval 16 --prefill-max-wait-ms 5000` and a 16K budget, total throughput is +8.4% and p90 interactivity +42% over the default scheduler (ITL p99 118 → 78 ms), at the cost of TTFT p90 1.4 → 5.5 s. <TODO: confirm at the full 3600 s replay.>

## Validation

**Unit tests (CPU)** on main `feb4cb019f`:

```bash
pytest tests/v1/core/test_prefill_delayer.py   # 22 passed
pytest tests/v1/core/test_prefill_delayer.py tests/v1/core/test_scheduler.py \
       tests/v1/core/test_async_scheduler.py tests/engine/test_arg_utils.py \
       tests/distributed/test_cpu_all_reduce.py \
       tests/v1/core/test_priority_preemption_bug.py   # 462 passed
pre-commit run --files <changed files>          # all hooks pass, mypy included
```


`test_prefill_delayer.py` covers:
- every FIRE/HOLD rule and bound;
- the scheduler's state report and hold gating, on a real `Scheduler`;
- the engine core syncing every step;
- a 2- and 3-rank gloo test, checking that `sync_dp_state(extra=...)` returns the exact sums and that every rank takes the same decision.

**GSM8K** (5-shot, lm_eval `local-completions`, 1319 questions, concurrency 64). DeepSeek-V4.1-Flash on 2x MI355X: DP2 with the MoE TP-sharded across both ranks, DSpark MTP (5 tokens), async scheduling, Model Runner V2.

| Policy | Flags | GSM8K (strict) |
|---|---|---:|
| `cadence` (default) | — | 0.9303 ± 0.0070 |
| `delayer`, 4 runs | `--prefill-schedule-interval 10 --prefill-max-wait-ms 5000` | 0.9242 / 0.9249 / 0.9409 / 0.9303 (mean 0.9301) |

No accuracy change and no errors or hangs.

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

**End-to-end performance (preliminary).** DeepSeek-V4.1-Flash on 2x MI355X: TP1 x DP2 with the MoE TP-sharded across both ranks, DSpark MTP (5 tokens, synthetic acceptance length 3.51), `--max-num-batched-tokens 16384` unless noted, Rust frontend, `vllm-router` (consistent hash on the session id). Workload: InferenceX AgentX multi-turn agentic trace (`infx.bench agentic`), concurrency 128, ~90% GPU prefix-cache hits. These are InferenceX fast replays (`AIPERF_EXPERIMENTAL_FAST=1`: 1200 s profile, one warmup request per lane), one run per point; the noise is about 7% on throughput and 4–5% on interactivity. Fast replays start with a colder prefix cache than the full 3600 s replay, so they overstate the default scheduler's ITL and understate absolute throughput.

| Arm | Total tok/s/GPU | p90 interactivity (1000 / ITL p90) | ITL p99 (ms) | TTFT p90 / p99 (s) | TTFT > 10 s | E2E-norm interactivity p90 |
|---|---:|---:|---:|---|---:|---:|
| default scheduler | 186,066 | 25.5 | 117.7 | 1.39 / 15.8 | 1.2% | 22.5 |
| delayer N6 W5000 | 200,036 (+7.5%) | 33.2 (+30%) | 58.8 | 4.05 / 10.5 | 1.1% | 23.3 |
| delayer N11 W5000 | 198,362 (+6.6%) | 35.7 (+40%) | 78.7 | 5.06 / 19.2 | 2.1% | 21.3 |
| **delayer N16 W5000** | **201,610 (+8.4%)** | **36.3 (+42%)** | 77.5 | 5.51 / 20.5 | 1.6% | 20.0 |
| delayer N16 W3000 | 195,304 (+5.0%) | 32.1 (+26%) | 96.3 | 4.12 / 17.6 | 1.7% | 20.1 |
| delayer N16 W8000 | 200,854 (+7.9%) | 39.0 (+53%) | 71.4 | 7.47 / 25.4 | 3.5% | 17.5 |
| delayer N16 W5000, 24K budget | 199,704 (+7.3%) | 35.6 (+40%) | 86.0 | 4.68 / 11.0 | 1.1% | 21.6 |

N is `--prefill-schedule-interval`, W is `--prefill-max-wait-ms`. Every delayer arm finished with no hang and no preemption; the delayer held prefill on 72–81% of steps, identically on both ranks.

Takeaways:
- The delayer trades TTFT for decode smoothness: every arm improves throughput by 5–8%, p90 interactivity by 26–53%, and ITL p99 by 18–50%.
- N sets the gain: p90 interactivity rises from N6 to N11 and flattens from N11 to N16. W sets the TTFT bound: W8000 pushes TTFT p90 past 7 s.
- To bound TTFT, a smaller N works better than a smaller W. N6 W5000 beats N16 W3000 on every metric. A small W releases more often from inside the protected steps, which hurts ITL p99.
- A 24K budget ties 16K on the main metrics and roughly halves TTFT p99.
- <TODO: confirm the default scheduler, N16 W5000 at 16K, and N16 W5000 at 24K with the full 3600 s replay, and validate at concurrency 64 and 256.>

**CI coverage:** the new tests are CPU tests (`pytest.mark.cpu_test`). The GPU serving path is not exercised in CI; it is covered by the GSM8K and performance runs above.

## Details

**Problem.** The DP ranks of a MoE model step in lockstep, so a step in which any rank prefills is a long step for every rank. With each rank prefilling independently in a fraction `p` of steps, `1 - (1 - p)^DP` of steps are long: 57% at `p = 10%` and DP8. `--prefill-schedule-interval` controls *when* ranks may prefill, but not *whether* the step is worth it:
- a cadence step runs whatever prefill each rank has, even one small chunk on one rank;
- a rank whose queue a release cannot drain (`prefill_capacity_bound`) stops throttling and prefills on every step.

**Decision** (`vllm/v1/core/sched/prefill_delayer.py`). Each rank reports these fields, and every rank runs the same state machine on their sums across the group:
- `prefillable`, `blocked`, `unreachable`
- `pending_tokens` (capped at the budget), `decodes`, `partial`
- `kv_high`, `queue_hot`
- `prefill_ran`, `arrived`

All counters advance per step. The only wall-clock check, `queue_hot`, compares each rank's own requests and is summed as a flag. The rules, in order:

1. After a step that ran prefill, protect `interval - 1` steps (HOLD), unless a queue is past `--prefill-max-wait-ms` (FIRE).
2. FIRE if no rank has prefill, no rank decodes, a rank's KV cache is nearly full, or a queue is past the wait limit.
3. FIRE after 200 held steps, or 100 with a prompt chunk in progress.
4. HOLD while `prefillable + blocked < dp_size`. A blocked rank (waiting work it cannot admit now) counts as aligned. While unaligned, a stall releases only if some missing rank cannot get work within the hold bound. A stall is 10 steps with no new pending tokens, no arrivals, and no rank turning (non-)prefillable.
5. FIRE once pending tokens reach 90% of the prefillable ranks' budgets, or on a stall. Otherwise HOLD.

The constants are module-level, not options, to keep the knob count at two. Decision counts are logged at DEBUG every 1000 steps.

**Scheduler.** `Scheduler.prefill_delayer_state()` builds the report. `prefill_ran` comes from the running loop's `prefill_scheduled` and the waiting loop's `prefill_admitted`, which uses the deferral gate's predicate, so full prefix hits and remote-KV resumes do not count. A hold goes through the existing `defer_prefills` path. It defers only on a rank with a decode to run, since an empty batch on one rank while its peers step can hang the engine. Under the delayer, `prefill_capacity_bound` does not lift a hold; the delayer's own bounds replace it.

**Engine core.** With the delayer, `DPEngineCoreProc._has_global_unfinished_reqs` syncs on every step, not every `dp_sync_interval` steps, and passes the delayer's buffer as `extra`. The decision sets `hold_prefills`, which `_should_throttle_prefills` returns. The delayer resets at each DP wave boundary.

**Default unaffected.** With `cadence`, no delayer is built, `_has_global_unfinished_reqs` and `sync_dp_state` take their original paths, and the scheduler's additions are guarded by `prefill_delayer_enabled=False`. Engines without MoE DP fall back to `cadence` and log once. The new fields are excluded from the compile cache key.

**Limitations:**
- TTFT is the tradeoff. At low concurrency the ranks rarely align, and prompts wait up to the hold bound.
- `--prefill-max-wait-ms` is a queue-age release, not a hard TTFT limit.
- The token budget sets the TTFT cost. At 8K, prompts need more prefill steps; at 32K, prefill steps are long. The docs recommend 16K for multi-turn agentic traffic.
- Under investigation: in the fast replays above, delayer arms dropped 0–4 requests (about 0.04%) whose responses carried no content (AIPerf `InvalidInferenceResultError`); the default-scheduler arm dropped none.
- `pending_tokens` counts waiting prompts before their prefix-cache lookup, so on cache-heavy workloads a release can carry less compute than the fill target implies. Left for a follow-up.
- Not supported: elastic EP, `--enable-fault-tolerance` (both change the DP size at runtime), and custom scheduler classes.
- About 400 lines of main code, most of it in a separate module. Follow-ups: cost-based release, and per-rank prefill cost balancing on the same exchange.

**Related.** #61834 is the predecessor, closed by its author in favor of this PR. #51199 coordinates prefill asynchronously through the DP coordinator; this PR uses a synchronous per-step exchange in the engine loop and adds decode protection and a fill target.

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
