**Title:** `[ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4.1-Flash`

## Overview

Run DeepSeek-V4.1-Flash on ROCm with the standalone `UMBPStoreConnector`, backed by a node-local MORI UMBP server. The aiter sparse MLA fast paths stay on, and the pool keeps a single copy of the TP-replicated MLA KV. RFC: #57187. Depends on #61797 (`[UMBP][9/10]`); #61795 (`[UMBP][7/10]`) has merged. The connector changes match the shared part of #63002 (DeepSeek-V4-Pro).

## Claims

- On MI355X, DeepSeek-V4.1-Flash runs the InferenceX agentic recipe with standalone UMBP (DSpark MTP, monokernel, mxfp4 indexer). Prefixes evicted from the GPU prefix cache are restored from the pool.
- With UMBP, the ROCm attention keeps `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` and the NaN-free extra-cache path. Before this PR, any KV connector turned them off.
- At TP > 1, each MLA-style group (compressed KV, SWA, indexer K) is stored once instead of `tp_size` times.
- GSM8K accuracy is unchanged, even with 93% of prompt tokens restored from the pool.
- On AgentX, the gain grows with how far the working set exceeds GPU KV: +46% throughput when it does not fit (TP2 c192), +3.8% when it barely fits (DP2 c256). UMBP adds no overhead when everything fits (TP2 c128).
- No behavior change for other connectors, embedded UMBP, or non-ROCm platforms.

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
  - `UMBPStoreConnector` returns `True`: its objects are whole blocks written by this cache layout, and the key namespace fingerprints every per-layer spec.
- **`deepseek_v41/amd/rocm.py`**: `_has_foreign_kv_writer()` replaces the four `kv_transfer_config is not None` guards on the aiter sparse MLA and NaN-free extra-cache paths. The aiter kernel does not scrub NaNs, so these paths stay off for connectors that may load foreign bytes.
- **UMBP TP dedup** (`umbp/{connector,data,scheduler,worker}.py`, `runtime/{base,standalone}.py`):
  - The standalone runtime sets `shares_pool_across_ranks = True`, and workers report the TP-replicated groups in the handshake.
  - Those groups get one `tp*` key. TP rank 0 stores it and every rank loads it.
  - Residency checks use `keys_per_object()`.
  - Opt out with `dedup_tp_replicated_kv: false`. Dedup is skipped under PCP/DCP.

**Other backends are unaffected.** The hook's only caller is the ROCm-only `deepseek_v41/amd/rocm.py`. Embedded UMBP keeps per-rank keys. Every other connector still turns the ROCm fast paths off.

**Limitations.**
- Single node only.
- Without SWA replay on ROCm, the SWA group is stored too, so there are more objects than on V4-Pro.
- The connector changes overlap with #63002; whichever PR merges second needs a rebase.
- Extensible KV cache:
  - It allocates through VMM, which the standalone server cannot map over HIP IPC.
  - Current main turns the extensible KV cache off automatically for UMBP on ROCm.
  - The perf runs still pass `--no-enable-extensible-kv-cache` on both arms. Otherwise the connector-free baseline would get a larger KV cache than the UMBP arm.
- `lazy_offload` stores nothing on V4.1 in our tests, because its scan finds no hashed free blocks. The default eager store is unaffected. This is left as follow-up.

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
