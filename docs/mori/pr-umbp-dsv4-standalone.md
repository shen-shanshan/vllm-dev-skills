# [ROCm][UMBP] Support DeepSeek-V4-Pro with UMBP standalone mode (compressed KV + SWA replay)

## Purpose

This PR lets DeepSeek-V4-Pro on AMD Instinct (MI355X, TP8) offload its prefix KV cache to a node-local MORI-UMBP `umbp_standalone_server` and restore it later. The pool is shared by every rank and engine on the host and survives engine restarts. Only single-node standalone mode is covered; RDMA / distributed mode is out of scope.

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
- `vllm/models/deepseek_v4/amd/rocm.py`: `_has_foreign_kv_writer()` replaces the `kv_transfer_config is not None` checks feeding `_aiter_sparse_mla_enabled` and `_trust_dsv4_extra_cache_nan_free`.
- Only the ROCm DeepSeek-V4 backend consumes the new hook. CUDA paths and every other connector behave exactly as before.

**Docs**
- `docs/features/umbp_standalone.md`: TP-replicated groups, the new option, and a DeepSeek-V4-Pro on MI355X section (server + `vllm serve` example).

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

E2E on 8x MI355X, DeepSeek-V4-Pro, TP8, fp8 KV, MORI built with `BUILD_UMBP=ON`:

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

Environment: MI355X, ROCm 10.0, torch 2.13.0+rocm10.0.0. The branch is rebased on `main` @ `fe5fe8ba5f` (which has `[UMBP][1/10]`–`[3/10]`) plus the `[UMBP][4/10]`–`[10/10]` stack (#58568 head `078c7bf335`).

| Suite | Result |
|---|---|
| `tests/v1/kv_connector/unit/test_umbp_{core,async_lookup,hybrid,events,embedded,standalone,lazy}.py` | **275 passed** |
| `tests/v1/core/test_prefix_replay.py` | **16 passed** |
| `tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py` | **12 passed** |
| Regression: `tests/v1/core/test_contiguous_kv_packing.py`, `tests/v1/core/test_kv_cache_utils.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `tests/v1/kv_connector/unit/test_multi_connector.py` | 200 passed, 2 failed (environment-only, see below) |

All seven new tests ran (none skipped) and passed:
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

### End-to-end (DeepSeek-V4-Pro, TP8, MI355X)

> TODO: this needs MORI built with `BUILD_UMBP=ON`; the installed `amd_mori 1.2.3.post1` has no UMBP bindings. Do not fill in from memory:
> - Restore check: cold vs. warm TTFT, external hit tokens, and output comparison. The SWA replay rebuilds the window approximately, so outputs may differ.
> - UMBP server key count and bytes per request, with `dedup_tp_replicated_kv` true vs. false (expected ~1/8) and with `rebuild_sliding_window` true vs. false.
> - GSM8K strict/flexible match with vs. without UMBP, plus a NaN-free run with `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`.
> - Replay accuracy on a long-context restore (e.g. 32K/128K prefix).

## Limitations / follow-ups

- Speculative decoding (MTP) is not validated yet.
- Distributed (RDMA) mode is not covered. It can reuse the same replicated keys once a shared pool is visible to all ranks.
- While the connector is configured, GPU prefix-cache hits also replay the last window, since the SWA groups leave prefix caching.
- The replay needs model runner V2. With `VLLM_USE_V2_MODEL_RUNNER=0` (or DBO), DeepSeek-V4 falls back to storing every group.
- An MTP draft's SWA layers would also become replayed groups, which the UMBP connector rejects for draft models today.

---

AI assistance (Claude Code) was used to prepare this change. The submitter has reviewed every changed line and ran the tests listed above.
