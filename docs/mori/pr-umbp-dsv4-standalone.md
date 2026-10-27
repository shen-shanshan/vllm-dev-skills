# [ROCm][UMBP] Support DeepSeek-V4-Pro with UMBP standalone mode (compressed KV + SWA replay)

## Purpose

This PR lets DeepSeek-V4-Pro on AMD Instinct (MI355X, TP8 or DP8) offload its prefix KV cache to a node-local MORI-UMBP `umbp_standalone_server` and restore it later. The pool is shared by every rank and engine on the host and survives engine restarts. It works with DSpark MTP speculative decoding (the InferenceX `mi355x-fp4-mtp` recipe) and runs on the MORI release the ROCm image pins (`v1.2.3.post1`). Only single-node standalone mode is covered; RDMA / distributed mode is out of scope.

It builds on the UMBP connector stack from RFC #57187:
- #58568 = `[UMBP][1/10]` … `[UMBP][10/10]`
- `[1/10]` #61789, `[2/10]` #61790 and `[3/10]` #61791 are already merged
- `[4/10]` … `[10/10]` are still under review (#58568 head `078c7bf335`, rebased on the merged `[3/10]`); standalone mode is `[UMBP][7/10]` #61795

It does **not** depend on the Kimi-K3 oriented UMBP PRs (#59276, #59281). **This PR must land after #58568's slices.** Until then the diff below is the delta on top of #58568.

### Why DeepSeek-V4 needs more than the generic connector

DeepSeek-V4 runs on the hybrid KV cache manager with several cache groups:
- compressed C4/C128 latents and the sparse indexer cache (block 256)
- the sliding-window cache
- the C4/C128 compressor states

Every one of these is an MLA-style spec (`MLAAttentionSpec` / `SlidingWindowMLASpec`). The UMBP connector already handles these groups generically: per-layer page sizes, the packed block-outermost layout, multi-group hit coordination, and forced async loads for indexer layers. Three things still made it a poor fit:

0. **The sliding-window and compressor-state groups turn into many tiny objects.**
   - Prefix hashes drop to 4 tokens, and the state groups have 4- and 8-token blocks, so a 128K-token prompt becomes tens of thousands of small objects per rank.
   - SGLang's UMBP integration for DeepSeek-V4 stores only the compressed pools. After a hit it recomputes the last sliding window, which rebuilds the SWA cache and compressor states.
   - vLLM already has this mechanism as *SWA bounded replay*, but it was wired only for DeepSeek-V4.1 and was off on ROCm.

1. **TP-replicated KV was stored `tp_size` times.**
   - UMBP keys carry the TP rank, and a lookup requires every rank's key.
   - MLA latents are byte-identical on all TP ranks, so at TP8 the shared standalone pool held 8 copies of every DeepSeek-V4 block and every lookup sent 8x the keys.
   - SGLang's UMBP integration for DeepSeek-V4 already uses rank-replicated keys for the same reason.
2. **Any KV connector turned off the ROCm DeepSeek-V4 fast paths.**
   - `amd/rocm.py` disables `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` and the gfx950 "extra cache NaN-free" path whenever `kv_transfer_config` is set. The reason is that a connector might put unscrubbed bytes (e.g. NaNs in unwritten slots) into the cache.
   - UMBP never does this: it restores only whole blocks that the same cache layout stored, and its namespace fingerprints every per-layer spec.

## Changes

**Store only compressed KV; rebuild the sliding window by replay (DeepSeek-V4, ROCm)**
- `KVConnectorBase_V1.requests_sliding_window_replay(kv_transfer_config)`: new classmethod, default `False`.
  - `MultiConnector` returns true only if every child does.
  - `UMBPStoreConnector` returns `True`, with opt-out `rebuild_sliding_window: false`.
- `vllm/models/deepseek_v4/compressor.py`: `dsv4_swa_replay_enabled()` is true only on ROCm, with the V2 model runner, `swa_bounded_replay` on, no PCP, and a connector that requests replay.
  - When true, the DeepSeek-V4 SWA cache uses `bounded_replay=True`.
  - The C4/C128 compressor-state caches are marked `rebuilt_by_replay`.
- `SlidingWindowMLASpec.rebuilt_by_replay` (new, default `False`): keeps a group out of prefix caching without declaring its own replay window. The compressor states (windows 8/128) cannot use `bounded_replay`, because the scheduler requires a single replay window.
  - The 128-token SWA replay from a 256-aligned hit rewrites the C4 overlap and any partial C128 group before they are read.
- `vllm/models/deepseek_v4/amd/model.py`: when replay is on, installs the existing replay-aware model state (`DeepseekV41ModelState`).
  - It forwards `replay_start` to the SWA metadata builders and pads the replayed tokens' slots in the prefix-cacheable groups, so restored compressed KV is not rewritten.
  - The ROCm DeepSeek-V4 SWA builder already clamps both prefill gather windows and decode indices at `replay_start`. No kernel change is needed.
- Result:
  - UMBP's prefix-cacheable groups are only the C4/C128 latents and the C4 indexer (256-token objects).
  - The scheduler rounds a UMBP hit to a block and recomputes its last window.
  - Without a UMBP connector, nothing changes. CUDA is untouched.
  - As with V4.1, replayed tokens see the sliding window only from `replay_start`. The rebuilt window KV is therefore an approximation, while compressed and sparse top-k context stays exact.

**TP-replicated keys (UMBP connector, `umbp/{data,connector,scheduler,worker}.py`, `umbp/runtime/{base,standalone}.py`)**
- `tp_replicated_group_ids()` marks a cache group as TP-replicated when every layer spec is MLA-style, `tp_size > 1`, and there is no decode/prefill context parallelism (CP shards the latent).
- `BlockIdentityCodec` gives replicated groups one TP-agnostic key (`tp*`):
  - `keys_for_topology` emits one key per object instead of one per TP rank;
  - `keys_per_object` drives hit accounting in lookup, and in every residency check (restored-block stores, lazy offload candidates, KV-event reconciliation), since replicated objects have one key instead of one per rank.
- The worker stores replicated groups only on TP rank 0. The other ranks drop those plans, still complete the store event, and load the shared object.
- Dedup is enabled only when the runtime shares one pool across ranks (`IUMBPRuntime.shares_pool_across_ranks`, true for standalone). Embedded mode keeps rank-private pools and per-rank keys unchanged.
- Workers report the replicated groups in `UMBPHandshakeMetadata`, so the scheduler, which only sees flattened specs, uses the same keys. This also holds when `key_namespace` is configured.
- New option `dedup_tp_replicated_kv` (default `true`) to opt out.
- Replicated keys use `tp*` and therefore can never alias objects stored under per-rank keys.

**Keep the ROCm DeepSeek-V4 fast paths with UMBP**
- `KVConnectorBase_V1.only_restores_canonical_kv(kv_transfer_config)`: new classmethod, default `False`, following the `supports_sleep_mode` pattern.
  - `MultiConnector` returns true only if every child does.
  - `UMBPStoreConnector` returns `True`.
- `vllm/models/deepseek_v4/amd/rocm.py`: `_has_foreign_kv_writer()` replaces the `kv_transfer_config is not None` checks feeding the aiter sparse MLA decode and prefill gates (`_aiter_sparse_mla_enabled`, `_aiter_sparse_mla_prefill_enabled`) and `_trust_dsv4_extra_cache_nan_free`. The DeepSeek-V4.1 backend keeps its own `kv_transfer_config` check.
- Only the ROCm DeepSeek-V4 backend consumes the new hook. CUDA paths and every other connector behave exactly as before.

**Speculative decoding: DSpark MTP (UMBP connector, `umbp/connector.py`)**
- The DeepSeek-V4-Pro MTP recipe uses `"method": "dspark"`. Its 3 draft layers have `compress_ratio=0`, so each one has only a sliding-window cache, and with replay on that cache becomes a `bounded_replay` group too.
- DSpark builds a position's draft KV from the target's hidden states alone (`precompute_and_store_context_kv`; `needs_eagle_block_drop()` is false). So the replayed window that rebuilds the target's SWA cache rebuilds the draft's as well, with no extra code. UMBP stores none of the draft KV.
- The connector's existing check only rejects EAGLE-style draft groups (`is_eagle_group`) that are not prefix-cacheable, so DSpark already passes it. This PR documents that and pins it with a test.
- EAGLE-style MTP (`"method": "mtp"`) draft groups are not rebuilt by the replay and are still rejected at startup. The error now says to set `rebuild_sliding_window: false`, which stores every group instead.

**Run on released MORI (`umbp/runtime/standalone.py`)**
- The standalone runtime in #58568 required `UMBPStandaloneProcessConfig.lookup_timeout_ms`. Neither MORI `v1.2.3.post1` (pinned by `docker/Dockerfile.rocm_base`) nor ROCm/mori main has it, so the connector could not start on the ROCm image at all.
- Now, when the binding is missing and `lookup_timeout_ms` is not configured, the connector logs a warning and attaches with unbounded lookups. Setting `lookup_timeout_ms` on such a MORI is still an error.

**Docs**
- `docs/features/umbp_standalone.md`: TP-replicated groups, the new option, a DeepSeek-V4-Pro on MI355X section (server + `vllm serve` example), speculative decoding (DSpark vs. EAGLE-style MTP), and the `lookup_timeout_ms` behavior on released MORI.

## Not duplicating existing work

- #58568 / the `[UMBP][N/10]` slices add the generic connector and standalone mode, but have no TP-replication awareness and no DeepSeek-V4 handling.
- #59275 / #59276 / #59281 cover an older snapshot, distributed mode, and layer-wise loading for Kimi-K3.
- No open PR changes the ROCm DeepSeek-V4 connector gating.

## Test Plan

Unit tests (ROCm; the UMBP suites mock the MORI client):

```bash
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_umbp_*.py -v
.venv/bin/python -m pytest tests/v1/core/test_prefix_replay.py -v
.venv/bin/python -m pytest tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py -v
# regression for the touched shared code
.venv/bin/python -m pytest tests/v1/core/test_contiguous_kv_packing.py \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/attention/test_deepseek_v4_swa_visible.py \
  tests/v1/kv_connector/unit/test_multi_connector.py -v
```

New tests:
- `test_prefix_replay.py::test_state_rebuilt_by_replay_is_not_prefix_cached`: a `rebuilt_by_replay` group is neither cached nor a second replay window.
- `test_rocm_aiter_triton_sparse_mla.py::test_dsv4_swa_replay_only_with_a_connector_that_asks`
- `test_umbp_standalone.py::test_only_tp_replicated_mla_groups_are_deduplicated`
- `::test_tp_replicated_blocks_are_stored_once_and_loaded_by_every_rank`: TP2; one `tp*` object per block; both ranks complete the store event; rank 1 restores byte-exact data.
- `::test_tp_dedup_can_be_disabled`, `::test_embedded_rank_private_pools_are_not_deduplicated`
- `test_rocm_aiter_triton_sparse_mla.py::test_only_non_canonical_kv_connectors_disable_aiter_sparse_mla`
- `test_umbp_standalone.py::test_dspark_draft_rebuilt_by_replay_is_accepted`: a non-EAGLE draft SWA group rebuilt by replay passes the draft-group check.
- `::test_eagle_draft_rebuilt_by_replay_is_rejected_with_a_hint`: an EAGLE draft group rebuilt by replay still raises, and the error points at `rebuild_sliding_window: false`.
- `::test_released_mori_without_lookup_timeout_still_attaches`, `::test_lookup_timeout_needs_mori_with_bounded_lookups` (replaces `test_standalone_requires_mori_with_bounded_lookups`)

E2E accuracy with DSpark MTP: the InferenceX recipe `dsv4/vllm/mi355x-fp4-mtp/agentic.yaml` (base + `override_dep8_c128`) is launched twice with identical flags, once without and once with the UMBP connector. Then GSM8K is run against each. With UMBP it runs twice: pass 1 stores the KV, pass 2 resends the same prompts so they are restored from the pool.

```bash
# server (128 GiB pool is plenty for GSM8K)
UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp/standalone.grpc.sock \
UMBP_DRAM_CAPACITY=$((128 * 1024**3)) UMBP_SSD_ENABLED=0 umbp_standalone_server

# engine: recipe args, + --kv-transfer-config for the UMBP run
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768 OMP_NUM_THREADS=1 \
vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 --trust-remote-code --async-scheduling \
  --tensor-parallel-size 1 --data-parallel-size 8 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 --max-num-seqs 128 --gpu-memory-utilization 0.86 \
  --moe-backend aiter --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 6, "draft_sample_method": "probabilistic"}' \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --prefill-schedule-interval 8 --long-prefill-token-threshold 16384 \
  --kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///tmp/umbp/standalone.grpc.sock"}}'

lm_eval --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Pro-0813,base_url=http://0.0.0.0:8333/v1/completions,num_concurrent=64,max_retries=10,max_gen_toks=2048,timeout=60000,tokenized_requests=False \
  --batch_size auto --tasks gsm8k --num_fewshot 5
```

Planned E2E on 8x MI355X, DeepSeek-V4-Pro, TP8, fp8 KV, MORI built with `BUILD_UMBP=ON` (not run yet, see Test Result):

```bash
# 1. standalone server (1 TiB DRAM pool)
UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp/standalone.grpc.sock \
UMBP_DRAM_CAPACITY=$((1024 * 1024**3)) UMBP_SSD_ENABLED=0 umbp_standalone_server

# 2. engine with --enable-prefix-caching and the UMBP connector (see docs)
# 3. restore check: cold request -> POST /reset_prefix_cache -> same request;
#    expect external prefix-cache hits and lower TTFT (outputs may differ
#    slightly: the SWA replay rebuilds the window approximately)
# 4. GSM8K (lm_eval, 5-shot) with and without the connector
HF_DATASETS_CACHE=/shared/models/huggingface/datasets lm_eval ...
```

## Test Result

### Unit tests

Environment: MI355X, ROCm 10.0, torch 2.13.0+rocm10.0.0. The branch is rebased on `main` @ `803bc13cf9` (which has `[UMBP][1/10]`–`[3/10]`) plus the `[UMBP][4/10]`–`[10/10]` stack (#58568 head `078c7bf335`).

| Suite | Result |
|---|---|
| `tests/v1/kv_connector/unit/test_umbp_{core,async_lookup,hybrid,events,embedded,standalone,lazy}.py` | **289 passed** |
| `tests/v1/core/test_prefix_replay.py` | **16 passed** |
| `tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py` | **19 passed** |
| Regression: `tests/v1/core/test_contiguous_kv_packing.py`, `tests/v1/core/test_kv_cache_utils.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `tests/v1/kv_connector/unit/test_multi_connector.py` | 200 passed, 2 failed (environment-only, see below) |

All new tests ran (none skipped) and passed, including the four added for DSpark MTP and released MORI:
- `test_umbp_standalone.py::test_dspark_draft_rebuilt_by_replay_is_accepted`
- `test_umbp_standalone.py::test_eagle_draft_rebuilt_by_replay_is_rejected_with_a_hint`
- `test_umbp_standalone.py::test_released_mori_without_lookup_timeout_still_attaches`
- `test_umbp_standalone.py::test_lookup_timeout_needs_mori_with_bounded_lookups`
- `test_umbp_standalone.py::test_only_tp_replicated_mla_groups_are_deduplicated`
- `test_umbp_standalone.py::test_tp_replicated_blocks_are_stored_once_and_loaded_by_every_rank`
- `test_umbp_standalone.py::test_tp_dedup_can_be_disabled`
- `test_umbp_standalone.py::test_embedded_rank_private_pools_are_not_deduplicated`
- `test_prefix_replay.py::test_state_rebuilt_by_replay_is_not_prefix_cached`
- `test_rocm_aiter_triton_sparse_mla.py::test_only_non_canonical_kv_connectors_disable_aiter_sparse_mla`
- `test_rocm_aiter_triton_sparse_mla.py::test_dsv4_swa_replay_only_with_a_connector_that_asks`

The two `test_multi_connector.py` failures come from the test environment and do not touch this change:
- `test_multi_example_connector_consistency`: the gated HF model `meta-llama/Llama-3.2-1B-Instruct` returns 401 without an HF token.
- `test_multi_connector_mixed_hma_disables_hybrid_kv_cache`: `ray` is not installed.

### End-to-end accuracy with DSpark MTP (DeepSeek-V4-Pro-0813, DP8, MI355X)

Environment:
- 8x MI355X, ROCm 10.0, torch 2.13.0+rocm10.0.0.
- MORI `v1.2.3.post1` built with `BUILD_UMBP=ON`.
- Recipe `dsv4/vllm/mi355x-fp4-mtp/agentic.yaml` with `override_dep8_c128`: DP8 x TP1, DSpark with 6 speculative tokens and probabilistic drafting, fp8 KV, model runner V2, prefix caching.
- The recipe's `vllm-router` front end is replaced by the engine's built-in DP load balancer, identically in both runs. The router only picks a DP rank.
- Every other flag and env var is identical between the two servers. The UMBP run adds only `--kv-transfer-config` (standalone, `kv_load_failure_policy: recompute`). Its startup log shows `DeepSeek-V4 SWA bounded replay: enabled`.
- GSM8K: `lm_eval` 0.4.13, 5-shot, all 1319 questions, greedy decoding, `num_concurrent=64`.

| Run | strict-match | flexible-extract | Draft acceptance | Mean acceptance length | Prompt tokens from UMBP / GPU prefix cache |
|---|---|---|---|---|---|
| Baseline (no connector) | 0.9515 ± 0.0059 | 0.9507 ± 0.0060 | 57.97% | 4.48 | 0 / 0 |
| UMBP, pass 1 (stores) | 0.9538 ± 0.0058 | 0.9530 ± 0.0058 | 58.04% | 4.48 | 0.2% / 0 |
| UMBP, pass 2 (same prompts restored) | 0.9522 ± 0.0059 | 0.9515 ± 0.0059 | 58.36% | 4.50 | **72.6%** / 12.5% |

- Accuracy is unchanged. Every difference is at most 0.23 points, well inside one standard error (about 0.6 points).
- Pass 2 is the stress case. 835,584 of 1,150,678 prompt tokens were restored from the UMBP pool (`vllm:external_prefix_cache_hits_total`), and 143,360 more hit the GPU prefix cache. Almost every request went through "restore compressed KV → recompute the last 128-token window".
- That recomputation rebuilds the SWA cache and compressor states of the target, and the sliding-window cache of the three DSpark draft layers.
- Draft acceptance is the signal for the draft side. Probabilistic rejection sampling keeps the output distribution exact even with bad drafts, so wrong draft KV would show up as lower acceptance, not lower accuracy. Acceptance stays at 58% (4.5 tokens per step) in every run, so the replay rebuilds the draft KV correctly.
- In the baseline, GSM8K prompts get no GPU prefix hits: with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768`, the SWA groups keep checkpoints only every 32K tokens. With UMBP the SWA groups leave prefix caching, so GPU hits also become possible (pass 2).

> TODO (not run yet, do not fill in from memory):
> - TP8 restore check: cold vs. warm TTFT, external hit tokens, and output comparison.
> - UMBP server key count and bytes per request, with `dedup_tp_replicated_kv` true vs. false (expected ~1/8) and with `rebuild_sliding_window` true vs. false.
> - A NaN-free run with `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`.
> - Replay accuracy on a long-context restore (e.g. 32K/128K prefix).

## Limitations / follow-ups

- Speculative decoding is validated with DSpark only. EAGLE-style `"method": "mtp"` with replay is rejected at startup; it needs `rebuild_sliding_window: false`, which stores every group. That combination has not been run end to end. The 0813 checkpoint's `mtp.*` weights are the DSpark layers, so DSpark is the MTP path for this model.
- On released MORI (no `lookup_timeout_ms`), scheduler lookups have no deadline: a stalled server blocks the lookup. The connector logs a warning.
- With a 1 TiB `UMBP_DRAM_CAPACITY` (hugepages off), `umbp_standalone_server` spent more than 10 minutes faulting in the pool before it opened its gRPC socket. Size the pool for the workload, or back it with hugepages.
- Distributed (RDMA) mode is not covered. It can reuse the same replicated keys once a shared pool is visible to all ranks.
- While the connector is configured, GPU prefix-cache hits also replay the last window, since the SWA groups leave prefix caching.
- The replay needs model runner V2. With `VLLM_USE_V2_MODEL_RUNNER=0` (or DBO), DeepSeek-V4 falls back to storing every group.

---

AI assistance (Claude Code) was used to prepare this change. The submitter has reviewed every changed line and ran the tests listed above.
