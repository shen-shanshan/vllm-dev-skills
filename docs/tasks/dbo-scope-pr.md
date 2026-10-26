# [Feature][DBO] Add `--dbo-scope` to limit DBO to prefill batches

<!-- markdownlint-disable -->

## Overview

Adds a `--dbo-scope {all,prefill}` option (`ParallelConfig.dbo_scope`) so users can choose which batches Dual Batch Overlap runs on. The default `all` keeps today's behavior. `prefill` only microbatches batches that contain a prefill. Decode-only batches always run as a single batch, and no microbatched decode CUDA graphs are captured.

## Claims

- **New option:** `--enable-dbo --dbo-scope prefill` turns on DBO for prefill batches only. `--dbo-scope all` (the default) behaves exactly like current `--enable-dbo`.
- **No ubatched decode graphs under `prefill`:** microbatched decode CUDA graphs are not captured, which saves capture time and graph memory.
- **One decision point:** the scope is applied in `check_ubatch_thresholds()`, which every path already uses to decide whether to microbatch:
  - V1 runner DP sync and CUDA graph capture.
  - V2 runner DP sync and CUDA graph candidates.
- **Backward compatible and platform-agnostic:** `--enable-dbo` is unchanged, and the default keeps current behavior on every backend.

## Validation

**Unit tests:** `tests/v1/worker/test_dbo_scope.py` (new, CPU-only).

```bash
pytest tests/v1/worker/test_dbo_scope.py tests/v1/worker/test_ubatch_offsets.py -q
# 21 passed
```

The new tests cover:
- `check_ubatch_thresholds` for each scope, decode vs. prefill batches, and token counts just above and below the thresholds:
  - `prefill` never microbatches a uniform-decode batch, even at 1M tokens.
  - Prefill batches still follow `--dbo-prefill-token-threshold` under both scopes.
  - `all` keeps the existing decode threshold behavior.
  - Nothing is microbatched when DBO is off.
- V2 CUDA graph candidates: a FULL uniform-decode descriptor (64 reqs × 4 tokens) gets a microbatched twin under `all` and none under `prefill`.
- CLI and config:
  - The default is `all`.
  - `--dbo-scope prefill` and `--dbo-scope all` are passed through to `EngineArgs`.
  - Unknown values (e.g. `decode`) are rejected by argparse.

**Pre-commit:** `pre-commit run --files <changed files>` passes (including ruff check/format).

**E2E:** I haven't run an E2E check. The planned DeepSeek-V4-Pro DP8 + MTP run on MI355X couldn't go ahead because the checkpoint isn't available on the test machine. A reproducer for reviewers:

```bash
vllm serve <MoE model> --data-parallel-size 2 --enable-expert-parallel \
  --enable-dbo --dbo-scope prefill --all2all-backend <backend>
```

Under `prefill`, only steps that contain a prefill should be split into two microbatches; decode steps run as one batch.

## Details

**Motivation.** DBO hides one microbatch's communication behind the other microbatch's compute. For that to pay off, communication must be a large share of the step and each microbatch must have enough compute to cover it. Prefill batches meet both conditions. Decode batches often don't, particularly in single-node DP/EP deployments where communication is cheap (e.g. 8 × MI355X with `allgather_reducescatter`).

*Little to gain:*
- **Decode communication is mostly fixed latency.** Each rank sends only tens to hundreds of tokens per layer, so all-gather / reduce-scatter time is dominated by launch and sync latency, not bandwidth. Splitting into two microbatches pays that latency twice and does not halve communication time.
- **Little compute to hide it behind.** Each decode microbatch's MoE and attention work takes only tens of microseconds, often too short to cover the other microbatch's communication.

*Real costs:*
1. **MoE weights are read twice.** Decode MoE GEMMs are bound by memory bandwidth, so their cost is mostly reading expert weights. With two microbatches, an expert activated in both is read from HBM twice, nearly doubling weight traffic. Prefill is compute-bound and amortizes weight reads over many tokens, so it isn't affected.
2. **Lower GPU utilization.** Halving an already small batch cuts the parallelism of attention/MLA and GEMM kernels, leaving compute units underused.
3. **Fixed per-step overhead.** Switching between microbatch threads, event waits between the compute and comm streams, two sets of attention metadata and twice the kernel launches cost the same regardless of batch size. That overhead is a much larger share of a short decode step than of a prefill step.
4. **Padding waste.** Microbatching is all-or-none across DP ranks, which are padded to the same token count, and CUDA graphs add padding to align both microbatches to capture sizes. On small decode batches that padding is a large fraction of the work.
5. **Extra CUDA graphs.** A second set of microbatched FULL decode graphs has to be captured, adding startup time and graph memory.
6. **MTP amplifies all of the above.** With 3 speculative tokens, each decode request carries 4 tokens, so the default `--dbo-decode-token-threshold` of 32 already splits a rank with just 8 requests. Decode state under MTP + DBO is also more involved, and per-microbatch draft state is easy to get wrong (see the MTP hidden-buffer offset fix in #57773).

In our DeepSeek-V4-Pro DP8 + MTP runs on MI355X, the DBO gain (+7% output throughput, −3.5% TPOT vs. no DBO) was measured with decode microbatching effectively disabled (`--dbo-decode-token-threshold 8192`). We have not yet run a dedicated decode-DBO on/off comparison; `--dbo-scope` makes that A/B a one-flag change. Decode DBO can still win when communication is expensive or the batch is large:
- multi-node EP, where cross-node all-to-all is costly;
- very large decode batches per rank;
- DeepEP low-latency kernels designed for decode overlap.

So the default stays `all`.

Today the only way to keep DBO on prefill alone is to set `--dbo-decode-token-threshold` to a very large value. That is indirect, easy to get wrong when MTP multiplies the decode token count, and hard to read in deployment recipes. `--dbo-scope prefill` states the intent directly.

**Design.**
- The option is a new field rather than a value on `--enable-dbo`, so existing configs and the Python API (`enable_dbo: bool`) don't change.
- The check lives in `check_ubatch_thresholds()`, which every runtime and capture path already calls. Nothing else needs to change:
  - V1 capture only builds ubatched graphs when this function returns True for a uniform decode batch.
  - V2's `_maybe_ubatch_twin()` asks it with `uniform_decode=True`.
- `dbo_scope` is included in `ParallelConfig.compute_hash()` like the existing DBO thresholds, so DP ranks with different scopes are caught by the config hash check.

**Semantics.** "Decode" means a uniform decode batch: every request has the same query length (1 + number of speculative tokens). This is the same split `--dbo-decode-token-threshold` and `--dbo-prefill-token-threshold` already use. Non-uniform decode-only batches stay on the prefill side and follow the prefill threshold.

DP consistency is unchanged. Microbatching is still decided all-or-none across DP ranks from the synced uniform-decode flag, so all ranks reach the same answer.

**Limitations.** Only `all` and `prefill` are offered. A `decode`-only scope can be added later if there's a use case.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
