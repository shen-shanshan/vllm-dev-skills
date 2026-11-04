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

**Unit tests (CPU)**, branch based on main `790be385c8`:

```bash
pytest tests/v1/core/test_prefill_delayer.py   # 21 passed
pytest tests/v1/core/test_prefill_delayer.py tests/v1/core/test_scheduler.py \
       tests/v1/core/test_async_scheduler.py tests/engine/test_arg_utils.py \
       tests/distributed/test_cpu_all_reduce.py   # 451 passed
pre-commit run --files <changed files>          # all hooks pass, mypy included
```

`tests/v1/core/test_prefill_delayer.py` covers:
- each FIRE/HOLD rule and bound;
- the arrival-driven stall restart;
- blocked and unreachable ranks;
- the scheduler's state report and hold gating, on a real `Scheduler`;
- the engine core syncing every step with the delayer's state;
- a multi-process gloo test (2 and 3 ranks, 300 steps of per-rank random states) that checks `sync_dp_state(extra=...)` returns the exact per-field sums and that every rank takes the decision a single reference delayer takes from those sums.

**End-to-end (not run yet):**
- <TODO: GPU serving benchmark on DeepSeek-V3/V4-class MoE, DP8 + EP8, multi-turn agentic trace, same node, ≥2 runs per arm with variance.>
  - Arms: `cadence` with interval ∈ {1, 8, 16, 32} vs `delayer` with interval ∈ {1, 10, 20}, with and without `--prefill-max-wait-ms`.
  - Concurrency: 16 / 64 / 128 / 256.
  - Metrics: output throughput vs E2E-normalized interactivity, TTFT p50/p90, ITL p50/p90.
- <TODO: gsm8k smoke test with `--prefill-schedule-policy delayer` (scheduling only; outputs should match).>
- <TODO: run with async scheduling and MTP (V1 and V2 model runner), the combination in which #61834 once hung.>

**CI coverage:** the new tests are CPU tests (`pytest.mark.cpu_test`). The GPU serving path is not exercised in CI; validation outside CI is the TODO above.

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
