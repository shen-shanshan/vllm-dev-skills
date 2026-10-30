**Title:** `[ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4.1-Flash`

## Overview

Run DeepSeek-V4.1-Flash on ROCm with the standalone `UMBPStoreConnector`, which uses a node-local MORI UMBP server for a single node. The aiter sparse MLA fast paths stay on, and the pool keeps one copy of the TP-replicated MLA KV. RFC: #57187. Depends on #61795 (`[UMBP][7/10]`) and #61797 (`[UMBP][9/10]`). The connector-side changes match the shared part of #63002 (DeepSeek-V4-Pro).

## Claims

- DeepSeek-V4.1-Flash on MI355X runs the InferenceX TP2 agentic recipe with standalone UMBP: DSpark MTP, monokernel, and the mxfp4 indexer. A prefix dropped from the GPU prefix cache is restored from the UMBP pool.
- With UMBP, the DeepSeek-V4.1 ROCm attention keeps `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` and the NaN-free extra-cache path. Before this PR, any KV connector disabled them.
- With standalone UMBP at TP > 1, each MLA-style group (compressed KV, SWA, indexer K) is stored once under a TP-agnostic key. Before this PR it was stored `tp_size` times.
- GSM8K accuracy is unchanged with UMBP, including when almost every prompt is restored from the pool (93% of prompt tokens).
- No behavior change for other connectors, for embedded UMBP, or for non-ROCm platforms.

## Validation

Single node, 2x MI355X, MORI v1.2.3.post1 built with UMBP. Server and engine scripts: `umbp-dsv41-flash/` in the dev notes.

```bash
# UMBP standalone server (128 GiB DRAM pool, no SSD tier)
UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp/standalone.grpc.sock \
UMBP_DRAM_CAPACITY=$((128 * 1024**3)) UMBP_SSD_ENABLED=0 \
  umbp_standalone_server unix:///tmp/umbp/standalone.grpc.sock

# Engine: InferenceX dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml, override_tp2_c64
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_SERVER_DEV_MODE=1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash --tensor-parallel-size 2 --language-model-only \
  --tokenizer-mode deepseek_v41 --moe-backend aiter --block-size 128 --monokernel \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --engram-config '{"cpu_offload":true}' --max-num-batched-tokens 16384 \
  --kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///tmp/umbp/standalone.grpc.sock"}}'
```

Smoke test: send a ~20K-token two-turn chat, call `/reset_prefix_cache`, then resend turn 2.

| Check | Result |
|---|---|
| `vllm:external_prefix_cache_hits_total` delta on the resent turn | 24320 tokens (190 blocks of 128) |
| Restored answer == GPU-served answer (greedy) | identical |
| No `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA is ignored` warning | none (aiter sparse MLA stays on) |
| No NaN / Traceback / ERROR in engine log | none |

E2E accuracy with DSpark MTP:
- Environment: same host and software. The branch is `main` @ `5cae6fd0c9` plus the cherry-picked #61795 and #61797 and this PR.
- Deployments: the recipe above is launched twice with identical flags (both with `VLLM_SERVER_DEV_MODE=1`). The baseline deployment has no connector; the UMBP deployment adds only `--kv-transfer-config`.
- GSM8K: `lm_eval` 0.4.13, 5-shot, all 1319 questions, `local-completions`, `num_concurrent=64`, `max_gen_toks=2048`.
- Passes: the baseline runs once. The UMBP deployment runs twice: pass 1 stores the KV, then `/reset_prefix_cache` drops the GPU prefix cache, and pass 2 resends the same prompts so they are restored from the pool.

```bash
lm_eval --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://0.0.0.0:8333/v1/completions,num_concurrent=64,max_retries=10,max_gen_toks=2048,timeout=60000,tokenized_requests=False \
  --batch_size auto --tasks gsm8k --num_fewshot 5
```

| Run | strict-match | flexible-extract | Draft acceptance | Mean acceptance length | Prompt tokens from UMBP / GPU prefix cache |
|---|---|---|---|---|---|
| Baseline (no connector) | 0.9212 ± 0.0074 | 0.9204 ± 0.0075 | 48.69% | 3.43 | 0 / 0.04% |
| UMBP, pass 1 (stores) | 0.9333 ± 0.0069 | 0.9325 ± 0.0069 | 48.35% | 3.42 | 0 / 0.04% |
| UMBP, pass 2 (same prompts restored) | 0.9295 ± 0.0071 | 0.9287 ± 0.0071 | 48.87% | 3.44 | **92.7%** / 0 |

- Accuracy does not drop. Both UMBP passes score at or above the baseline. The largest gap (baseline vs. pass 1, +1.2 points) is about 1.2 combined standard errors, i.e. run-to-run noise under MTP and batching.
- Pass 2 is the stress case. 1,066,752 of 1,150,678 prompt tokens were restored from the UMBP pool (`vllm:external_prefix_cache_hits_total`). Every request loaded its compressed KV, SWA KV and indexer K from the pool and recomputed only the tail after the last full 128-token block.
- Draft acceptance stays at about 48.5% (3.4 tokens per step) in every run. Probabilistic rejection sampling keeps outputs exact even with bad drafts, so corrupted restored KV would show up here as lower acceptance; it does not.
- The engine logs have no NaN, Traceback, or `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA is ignored` warning.

AgentX performance is not part of this PR; it follows in a separate change.

## Details

**Why DeepSeek-V4.1 needs fewer changes than V4.**
- Its compressor state is a `CircularBufferSpec`, which is not prefix-cacheable. Hits are 128-aligned, so the open compression group is empty on restore, the same as on a GPU prefix-cache hit.
- Prefix-cache SWA replay is off on ROCm for V4.1 (`deepseek_v41/attention.py`). So the SWA group stays prefix-cacheable, and UMBP stores it next to the compressed KV and the indexer K.
- The generic HMA lookup, the asynchronous load (forced for indexer models), and lazy offload work unchanged.

Key changes:
- **`KVConnectorBase_V1.only_restores_canonical_kv`** (`base.py`): a new classmethod hook, default `False`. `MultiConnector` returns `True` only when every child does. `UMBPStoreConnector` returns `True`: its objects are whole blocks written by this cache layout, and the key namespace fingerprints every per-layer spec.
- **`deepseek_v41/amd/rocm.py`**: `_has_foreign_kv_writer()` replaces the four `kv_transfer_config is not None` guards on the aiter sparse MLA (metadata builder and attention) and NaN-free extra-cache paths. The aiter kernel does not scrub NaNs, so these paths stay off for connectors that may load foreign bytes.
- **UMBP TP dedup** (`umbp/{connector,data,scheduler,worker}.py`, `runtime/{base,standalone}.py`):
  - The standalone runtime sets `shares_pool_across_ranks = True`. Workers then report the MLA/SWA-MLA groups (`tp_replicated_group_ids`) in the handshake.
  - These groups get one `tp*` key, which every TP rank loads and only TP rank 0 stores.
  - Residency checks use `keys_per_object()` instead of `rank_count`.
  - Opt out with `dedup_tp_replicated_kv: false`. PCP/DCP shard the latents, so dedup is skipped there.

Other backends are unaffected:
- The hook's default keeps the existing behavior, and its only caller is the ROCm-only `deepseek_v41/amd/rocm.py`.
- Embedded UMBP keeps per-rank keys, because `shares_pool_across_ranks` defaults to `False`.
- NIXL, Mooncake, LMCache, and offloading connectors still disable the ROCm fast paths, as before.

Limitations:
- Single node only; distributed or RDMA UMBP is out of scope.
- Without SWA replay on ROCm, V4.1 stores the SWA group as well. That means more and smaller objects than V4-Pro, which uses replay.
- The connector changes overlap with #63002. Whichever PR merges second needs a rebase.

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
