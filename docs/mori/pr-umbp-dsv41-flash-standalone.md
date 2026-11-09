**Title:** `[ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4.1-Flash`

## Overview

Run DeepSeek-V4.1-Flash on ROCm with the standalone `UMBPStoreConnector`, backed by a node-local MORI UMBP server. The aiter sparse MLA fast paths stay on, and the pool keeps a single copy of the TP-replicated MLA KV. RFC: #57187. Builds only on the `[UMBP]` series already on `main` (up to #61795, `[UMBP][7/10]`); it does not include or depend on #61797 (`[UMBP][9/10]`, lazy offload). The connector changes match the shared part of #63002 (DeepSeek-V4-Pro).

## Claims

- On MI355X, DeepSeek-V4.1-Flash runs the InferenceX agentic recipe with standalone UMBP (DSpark MTP, monokernel, mxfp4 indexer). Prefixes evicted from the GPU prefix cache are restored from the pool.
- With UMBP, the ROCm attention keeps `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` and the NaN-free extra-cache path. Before this PR, any KV connector turned them off.
- At TP > 1, each group whose layers declare `max_tp_shards=1` (on V4.1: compressed KV, SWA, indexer K) is stored once instead of `tp_size` times.
- GSM8K accuracy is unchanged, even with 93% of prompt tokens restored from the pool.
- On AgentX, the gain grows with how far the working set exceeds GPU KV: +46% throughput when it does not fit (TP2 c192), +3.8% when it barely fits (DP2 c256). UMBP adds no overhead when everything fits (TP2 c128).
- No behavior change for other connectors, embedded UMBP, or non-ROCm platforms, except that V4.1's KV specs now declare `max_tp_shards=1` like V4's (see Details).

## Validation

Setup: single node, 2x MI355X, MORI v1.2.3.post1 built with UMBP, 128 GiB standalone DRAM pool with default connector options. The engine uses the InferenceX recipe `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`, `override_tp2_c64`:

```bash
UMBP_DRAM_CAPACITY=$((128 * 1024**3)) UMBP_SSD_ENABLED=0 \
  umbp_standalone_server unix:///tmp/umbp/standalone.grpc.sock

VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash --tensor-parallel-size 2 --language-model-only \
  --tokenizer-mode deepseek_v41 --moe-backend aiter --block-size 128 --monokernel \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --engram-config '{"cpu_offload":true}' --max-num-batched-tokens 16384 \
  --kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///tmp/umbp/standalone.grpc.sock"}}'
```

**Smoke test.**
- Procedure: send a ~20K-token two-turn chat, call `/reset_prefix_cache`, then resend turn 2.
- Result: 24,320 tokens are restored from the pool, and the greedy answer is identical to the GPU-served one.
- In every run below, the engine log has no NaN, no Traceback, and no `AITER_TRITON_SPARSE_MLA is ignored` warning.

**Unit tests.** `pytest tests/v1/kv_connector/unit/test_umbp_*.py`: 341 passed. That includes new tests for replicated-group detection, `tp*` keys and key counts, TP rank > 0 skipping replicated stores, handshake agreement, the canonical-KV hook, and the separate namespace of engines whose other connectors may load foreign KV. `pytest tests/v1/core/test_kv_cache_planning.py tests/v1/kv_connector/unit/offloading_connector/test_config.py`: 144 passed.

**Accuracy: GSM8K.**
- `lm_eval` 0.4.13, 5-shot, all 1319 questions, `local-completions`, `num_concurrent=64`, `max_gen_toks=2048`.
- Code base: `main` @ `5cae6fd0c9`.
- The baseline has no connector and runs once. The UMBP deployment runs twice: pass 1 stores, then `/reset_prefix_cache`, then pass 2 restores the same prompts from the pool.

| Run | strict-match | flexible-extract | Mean acceptance length | Prompt tokens from UMBP |
|---|---|---|---|---|
| Baseline | 0.9212 ± 0.0074 | 0.9204 ± 0.0075 | 3.43 | 0 |
| UMBP pass 1 | 0.9333 ± 0.0069 | 0.9325 ± 0.0069 | 3.42 | 0 |
| UMBP pass 2 | 0.9295 ± 0.0071 | 0.9287 ± 0.0071 | 3.44 | **92.7%** |

- Every gap is within about 1.2 combined standard errors.
- Draft acceptance stays flat. Corrupted restored KV would lower it, because probabilistic rejection sampling keeps outputs exact even with bad drafts.

**Performance: InferenceX AgentX.**
- Client: `python3 -m infx.bench agentic` replaying the `semianalysis_cc_traces_weka_062126` trace, `DURATION=3600`. ISL mean is about 120K tokens (p90 about 250K), OSL mean about 900.
- Engine changes, identical on both arms:
  - `rejection_sample_method=synthetic` with `synthetic_acceptance_length=3.51`, the golden AL that InferenceX throughput runs use;
  - `VLLM_USE_RUST_FRONTEND=1`;
  - `--no-enable-extensible-kv-cache` (see Limitations).
- The UMBP arm adds only `--kv-transfer-config`. Each arm is a fresh deployment, run once.
- The recipe's own TP2 points stop at c64, where GPU KV holds every session. So the tests push concurrency past that point, in three setups:

| Setup | Code base | Notes |
|---|---|---|
| TP2 c128 / c192 | `main` @ `51c6dfd8c8` | Recipe flags as above. GPU KV is 29.4M tokens. |
| DP2 c256 | `main` @ `77d2790e78` | `--tensor-parallel-size 1 --data-parallel-size 2`, without expert parallelism, so attention runs DP2 and MoE runs TP2. `--monokernel` is dropped because mono decode rejects DP. A `vllm-router` `consistent_hash` front end keeps each session on one rank. GPU KV is 26.8M tokens per rank. |

The DP2 c256 deployment as run, UMBP arm. The baseline drops `--kv-transfer-config`, and the UMBP server is started as above:

```bash
# Engine: attention DP2 + MoE TP2 on 2x MI355X
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
VLLM_USE_RUST_FRONTEND=1 OMP_NUM_THREADS=1 HIP_VISIBLE_DEVICES=0,1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash --port 8333 \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --language-model-only --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v41 --moe-backend aiter \
  --gpu-memory-utilization 0.9 --block-size 128 --max-model-len 1048576 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}' \
  --engram-config '{"cpu_offload":true}' --max-cudagraph-capture-size 1024 \
  --max-num-batched-tokens 16384 --no-enable-extensible-kv-cache \
  --kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///tmp/umbp/standalone.grpc.sock"}}'

# Front end: session-affine router over the two DP ranks (vllm-router==0.1.14)
vllm-router --worker-urls http://localhost:8333 --intra-node-data-parallel-size 2 \
  --host 0.0.0.0 --port 30000 --policy consistent_hash \
  --request-timeout-secs 14400 --disable-retries

# Client: InferenceX AgentX replay through the router
# (from InferenceX/inferencex-e2e, after `source benchmarks/runtime_settings.sh`)
IS_AGENTIC=1 SCENARIO_TYPE=agentic-coding KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=umbp \
TOTAL_CPU_DRAM_GB=128 KV_OFFLOAD_BACKEND_METADATA='{"name":"umbp","version":"mori-1.2.3.post1"}' \
CONC=256 DURATION=3600 TP=2 EP_SIZE=1 DP_ATTENTION=true SPEC_DECODING=mtp \
MODEL=deepseek-ai/DeepSeek-V4.1-Flash SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash \
MODEL_PREFIX=dsv41flash FRAMEWORK=vllm PRECISION=fp4 \
AIPERF_SERVER_URL=http://localhost:30000 AIPERF_SERVER_METRICS_URLS=http://localhost:8333/metrics \
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1 AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:' \
  python3 -m infx.bench agentic
```

For the baseline arm, the client sets `KV_OFFLOADING=none` and drops `KV_OFFLOAD_BACKEND`, `KV_OFFLOAD_BACKEND_METADATA` and `TOTAL_CPU_DRAM_GB`. The TP2 runs differ in three ways:
- the engine uses `--tensor-parallel-size 2 --monokernel`;
- there is no router;
- the client uses `CONC=128` or `CONC=192` and `DP_ATTENTION=false`, and sends to the engine at `:8333`.

Each cell is baseline → UMBP (Δ).

| Metric | TP2 c128 | TP2 c192 | DP2 c256 |
|---|---|---|---|
| Total throughput / GPU (tok/s) | 245.3K → 246.8K (+0.6%) | 179.3K → 262.5K (**+46%**) | 270.3K → 280.6K (**+3.8%**) |
| Output throughput (tok/s) | 4,097 → 4,120 (+0.6%) | 2,778 → 3,954 (+42%) | 4,053 → 4,194 (+3.5%) |
| TTFT mean / p90 (s) | 0.50 / 0.96 → 0.47 / 0.87 | 7.12 / 19.46 → 1.59 / 4.18 (−78% / −79%) | 1.69 / 2.66 → 1.40 / 2.47 (−17% / −7%) |
| TPOT mean (ms) | 19.5 → 19.2 | 58.0 → 37.0 (−36%) | 54.8 → 52.4 (−4%) |
| E2E latency mean (s) | 19.9 → 19.6 | 59.9 → 36.3 (−39%) | 50.2 → 48.0 (−4%) |
| Prompt tokens from GPU prefix cache | 94.7% → 94.8% | 87.8% → 88.1% | 92.9% → 93.1% |
| Prompt tokens from UMBP | 0.14% | 4.8% | 0.2% |
| Prompt tokens computed | 5.3% → 5.1% | 12.2% → 7.1% (−42%) | 7.1% → 6.7% (−6%) |

- **TP2 c128.** The sessions fit in GPU KV, so UMBP has almost nothing to restore. Throughput and TPOT stay within single-run noise, so UMBP adds no overhead.
- **TP2 c192.** The sessions overflow GPU KV and the prefill-bound baseline is overloaded. UMBP restores prefixes that were evicted while their sessions sat between turns. That cuts computed prefill by 42% and drains the queue. AgentX replays in a closed loop, so a faster turn issues the next one sooner and the latency gain also shows up as throughput.
- **DP2 c256.** Each rank serves about 128 sessions against 26.8M tokens of KV, so the overflow is small and so is the gain. Both ranks hit the pool, with load split about 50/50.
- **Common to all runs:**
  - Mean acceptance length is 3.51, with no preemptions.
  - The pool never filled, so nothing was evicted.
  - Failed requests are 0.03% or fewer, all client-side `ClientOSError`.
  - At TP2 c192 the two arms ran on different days, with the same code, `vllm-rs` build, and host. Decode steps above about 170 running requests exceed the CUDA graph capture size on both arms.

## Details

**Why V4.1 needs fewer changes than V4.**
- The compressor state is a `CircularBufferSpec`, which is not prefix-cacheable. Hits are 128-aligned, so the open compression group is empty on restore.
- Prefix-cache SWA replay is off on ROCm for V4.1. So the SWA group stays prefix-cacheable, and UMBP stores it alongside the compressed KV and the indexer K.

**Key changes.**
- **`KVConnectorBase_V1.only_restores_canonical_kv`** (`base.py`): a new classmethod hook, default `False`.
  - `MultiConnector` returns `True` only when every child does.
  - `UMBPStoreConnector` returns `True` only with the derived key namespace (`key_namespace: auto`, the default). Its objects are whole blocks written by this cache layout, and the derived namespace fingerprints every per-layer spec. An explicit `key_namespace` skips that fingerprint, so it keeps the fast paths off.
  - **Provenance.** The derived namespace also records whether any connector of the engine may load foreign KV, for example a `MultiConnector` sibling of UMBP. If so, `"foreign_kv": true` joins the fingerprint, so bytes such an engine stores can never be restored by a UMBP-only engine that trusts the pool. The child UMBP connector only sees its own `kv_transfer_config`, so it reads the engine config, which workers set while they create their connector. The field is added only when set, so UMBP-only namespaces are unchanged.
- **`kv_connector_may_load_foreign_kv(vllm_config)`** (`kv_connector/utils.py`): a connector-agnostic helper, `not only_restores_canonical_kv(...)` of the configured connector, `False` without one.
- **`deepseek_v41/amd/rocm.py`**: the helper replaces the four `kv_transfer_config is not None` guards on the aiter sparse MLA and NaN-free extra-cache paths. The aiter kernel does not scrub NaNs, so these paths stay off for connectors that may load foreign bytes.
- **UMBP TP dedup** (`umbp/{connector,data,scheduler,worker}.py`, `runtime/{base,standalone}.py`):
  - The standalone runtime sets `shares_pool_across_ranks = True`, and workers report the TP-replicated groups in the handshake.
  - A group is TP-replicated when every layer's spec has `max_tp_shards == 1`, the existing contract behind `KVCacheConfig.kv_tp_replicas`. `deepseek_v41/attention.py` now sets it on the compressed-KV and indexer `MLAAttentionSpec`s, as V4 does; the V4.1 SWA spec already did.
  - Those groups get one `tp*` key. TP rank 0 stores it and every rank loads it.
  - Residency checks use `keys_per_object()`.
  - Opt out with `dedup_tp_replicated_kv: false`. Dedup is skipped under PCP/DCP.
  - `docs/features/umbp_store_connector_usage.md` documents the option, the TP-agnostic keys, and the `key_namespace` caveat.

**Other backends are unaffected.** The hook's only caller is the new helper, used only by the ROCm-only `deepseek_v41/amd/rocm.py`. Embedded UMBP keeps per-rank keys. Every other connector still turns the ROCm fast paths off. The one shared change is `max_tp_shards=1` on V4.1's specs: V4.1's `kv_tp_replicas` becomes `tp_size`, so `OffloadingConnector` can use its TP-replicated layout for V4.1, as it already does for V4. It also changes V4.1's UMBP namespace, so objects stored before this PR miss once. Engines that pair UMBP with a connector that may load foreign KV get their own namespace and no longer share objects with UMBP-only engines.

**Limitations.**
- Single node only.
- Without SWA replay on ROCm, the SWA group is stored too, so there are more objects than on V4-Pro.
- The connector changes overlap with #63002; whichever PR merges second needs a rebase.
- Extensible KV cache:
  - It allocates through VMM, which the standalone server cannot map over HIP IPC.
  - Current main turns the extensible KV cache off automatically for UMBP on ROCm.
  - The perf runs still pass `--no-enable-extensible-kv-cache` on both arms. Otherwise the connector-free baseline would get a larger KV cache than the UMBP arm.
- Lazy offload (#61797) is not part of this PR. Once it lands, its residency check needs `keys_per_object()` for TP-replicated groups. In our tests with it applied, its scan found no hashed free blocks on V4.1 and stored nothing; this is left as follow-up. The perf runs used the default eager store, which this does not affect.

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
