# [ROCm][DSv4] Paged MXFP4 dense sparse indexer for DeepSeek-V4 on aiter's MQA-logits kernel

## Purpose

#58671 added a paged MXFP4 sparse indexer for DeepSeek-V4.1 on ROCm gfx950. The indexer K cache is stored as MXFP4 (68 B per token instead of 132 B for FP8) in the preshuffled page order that aiter's `pa_mqa_logits_mxfp4` reads. That kernel scores the cache in place, so prefill no longer gathers K first. The PR has two layers:

- the **dense** path, which scores the whole context. It is model-agnostic.
- the **candidate-pool** path, which only exists because of V4.1's two-level indexer.

This PR brings the dense path to DeepSeek-V4 (e.g. DeepSeek-V4-Pro: 30 indexer layers at compress ratio 4, 64 heads × 128, top-k 1024). Today V4 on ROCm can only use the FP8 indexer cache, because:

- `dsa_indexer_uses_fp4()` rejects every model except `deepseek_v41` on ROCm;
- `SparseAttnIndexer.forward_hip` asserts that the cache is not FP4;
- V4's own MXFP4 Q/K Triton kernels pack FP4 with CUDA PTX (`cvt.rn.satfinite.e2m1x2.f32`), so they cannot run on ROCm.

The feature is opt-in, as on V4.1:

```bash
VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-V4-Pro -tp 8 \
  --attention-config '{"indexer_kv_dtype": "mxfp4"}'
```

The default path (FP8 indexer cache) is unchanged.

## Design

### K write: ROCm pooling kernel + aiter's cache op

In V4 the indexer K is produced by the indexer's own `DeepseekCompressor`. One fused Triton kernel does pool → RMSNorm → RoPE → MXFP4 → cache store, and nothing before the norm ever reaches memory. On ROCm we split it into two launches:

1. A new Triton kernel, `_compress_pool_kernel`, does the softmax-gated overlap pooling. It is copied from V4's fused kernel, up to the norm. It writes one bf16 row per boundary token into a scratch buffer that all indexer layers share (`[max_num_batched_tokens, 128]`).
2. aiter's public `k_norm_rope_mxfp4_cache(..., shuffle=layout)` then applies norm → RoPE → MXFP4 and stores into the preshuffled pages. This is the same op V4.1 uses, through the same `rocm_mxfp4_indexer_k_store` wrapper.

This keeps the page layout and the quantization defined only in aiter, which is the principle #58671 follows (`rocm_paged_mxfp4_cache_layout` is the only place vLLM reads the layout).

The cost is one extra small launch per indexer layer, plus a bf16 round trip of the pooled row before the norm. That rounding error (~2⁻⁹) is two orders of magnitude below the MXFP4 quantization step. A fully fused ROCm kernel can follow if profiles show the extra launch matters.

### Q quant

The Q path uses aiter's `q_rope_mxfp4_quant`, through the existing `rocm_mxfp4_indexer_q_quant`. It produces fp32 weights with the softmax and head scales folded in, as the aiter MQA-logits kernel expects.

### Scoring and metadata

Both are reused unchanged:

- `RocmSparseAttnIndexer`, which calls `rocm_mxfp4_sparse_attn_indexer` with no candidate blocks;
- the model-agnostic `RocmMxfp4IndexerMetadataBuilder`.

A new `DeepseekV4RocmMxfp4IndexerBackend` subclasses `DeepseekV4IndexerBackend`, so V4's 256-token blocks and ROCm adaptive-verification support carry over. At compress ratio 4 that gives 64-entry pages, which is aiter's preferred page size.

### Keeping CUDA untouched

Shared V4 code only gains extract-method / ClassVar hooks. The code inside them is moved verbatim, and the ROCm subclasses override the hooks:

| Hook | Location | Default (all platforms) | ROCm MXFP4 override |
|---|---|---|---|
| `DeepseekV4Attention._indexer_cls()` | `deepseek_v4/attention.py` | `DeepseekV4Indexer` | `DeepseekV4RocmMxfp4Indexer` when `dsa_indexer_uses_fp4()` |
| `DeepseekV4Indexer.cache_cls / compressor_cls / attn_cls` | `deepseek_v4/attention.py` | `DeepseekV4IndexerCache` / `DeepseekCompressor` / `SparseAttnIndexer` | ROCm cache (new backend) / ROCm compressor / `RocmSparseAttnIndexer` |
| `DeepseekV4Indexer._q_rope_quant()` | `deepseek_v4/attention.py` | `fused_indexer_q_rope_quant` | aiter `q_rope_mxfp4_quant` |
| `DeepseekV4Indexer._register_q_quant_warmup()` | `deepseek_v4/attention.py` | existing Triton Q-kernel warmup | no-op (the PTX kernel never runs) |
| `DeepseekCompressor._compress_norm_rope_store()` | `deepseek_v4/compressor.py` | existing cutedsl / two-stage / Triton dispatch | pooling kernel + aiter cache op |
| `DeepseekCompressor._register_store_warmup()` | `deepseek_v4/compressor.py` | existing warmups | no-op (the PTX kernel never runs) |

### Other changes

- `dsa_indexer_uses_fp4()`: the ROCm branch now accepts `deepseek_v4` as well as `deepseek_v41`. The CDNA4 and aiter checks are unchanged, and the CUDA branch is untouched.
- `_aiter_indexer_cache_ops`, `rocm_mxfp4_indexer_k_store` and `rocm_mxfp4_indexer_q_quant` move from `models/deepseek_v41/amd/rocm.py` to `v1/attention/ops/rocm_paged_mxfp4_indexer.py`, so V4 and V4.1 share them. They are still importable from the old module.

### Files

| File | Change |
|---|---|
| `vllm/models/deepseek_v4/amd/mxfp4_indexer.py` (new) | pooling kernel, ROCm compressor / indexer cache / indexer subclasses |
| `vllm/models/deepseek_v4/amd/rocm.py` | `_indexer_cls` override on `DeepseekV4ROCMAiterMLAAttention` |
| `vllm/models/deepseek_v4/attention.py`, `compressor.py` | hooks only (verbatim moves) |
| `vllm/v1/attention/backends/mla/rocm_paged_mxfp4_indexer.py` | `DeepseekV4RocmMxfp4IndexerBackend` |
| `vllm/v1/attention/backends/mla/indexer.py` | ROCm gate accepts `deepseek_v4` |
| `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py`, `vllm/models/deepseek_v41/amd/rocm.py` | shared helper move |
| `tests/kernels/attention/test_rocm_dsv4_mxfp4_indexer.py` (new) | K-writer tests at V4-Pro shapes |

## Test Plan

Hardware: 8× MI350X (gfx950), ROCm, `VLLM_ROCM_USE_AITER=1`.

1. New unit tests (CDNA4 only), at V4-Pro shapes: 64 heads, ratio 4 with overlap, 64-entry pages.
   - `test_pooled_rows_match_torch`: the pooling kernel against a torch reference.
   - `test_k_pages_match_aiter_fed_by_torch`: the full ROCm compressor store step against aiter's writer fed by torch pooling, compared byte for byte in the preshuffled pages.
   ```bash
   pytest tests/kernels/attention/test_rocm_dsv4_mxfp4_indexer.py
   ```
2. Regression for the V4.1 path after the helper move, and for V4 ROCm tests that touch the compressor or indexer:
   ```bash
   pytest tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py \
          tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py
   pytest tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py \
          tests/models/test_deepseek_v4_rocm_wo_a.py \
          tests/models/test_deepseek_v4_dspark_rocm.py \
          tests/v1/attention/test_deepseek_v4_rocm_adaptive.py \
          tests/kernels/test_compressor_kv_cache.py \
          tests/kernels/test_deepseek_v4_cpu_kernels.py
   ```
3. End to end, DeepSeek-V4-Pro, TP8, FP8 indexer (baseline, default) vs MXFP4 indexer (this PR). Server flags: `--kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 128 --max-num-batched-tokens 8192 --no-enable-prefix-caching`.
   - GSM8K, 5-shot, all 1319 questions, through chat completions:
     ```bash
     python tests/evals/gsm8k/gsm8k_eval.py --port 8333 --use-chat-completions \
       --max-tokens 2048 --temperature 0 --max-concurrency 64
     ```
   - Long-context passkey retrieval. GSM8K prompts are shorter than `topk × ratio = 4096` tokens, so they never reach the indexer's top-k; this test does. Contexts of 8K/16K/32K tokens, needle depths 0.1–0.9, 60 prompts.
   - `vllm bench serve` on a random dataset: ISL 8192 / OSL 1024 at concurrency 64 (256 prompts), and ISL 32000 / OSL 128 at concurrency 8 (32 prompts), both with `--ignore-eos`.

## Test Result

**Unit tests:** all pass.

- New tests: 4 passed. The ROCm compressor's pages match aiter's writer fed by torch pooling byte for byte (0 mismatched bytes). As a sanity check, feeding a shifted window mismatches 21.8% of bytes, so the test does detect wrong pooling.
- V4.1 MXFP4 indexer tests: 37 passed.
- V4 ROCm / compressor tests: 99 passed, 137 skipped (platform-gated).

**Accuracy (DeepSeek-V4-Pro, TP8):**

| Indexer cache | GSM8K (5-shot, 1319) | Passkey 8K | Passkey 16K | Passkey 32K |
|---|---|---|---|---|
| FP8 (baseline) | 0.9151 | 100% | 100% | 100% |
| MXFP4 (this PR) | 0.9181 | 100% | 100% | 100% |

**Capacity:** the GPU KV cache grows from 394,705 to 430,756 tokens (+9.1%), because the indexer cache shrinks from 132 B to 68 B per compressed token.

**Serving performance** (one run per config; differences of about 1–2% are within run-to-run noise):

| Config | Metric | FP8 | MXFP4 | Δ |
|---|---|---|---|---|
| 8K/1K, conc 64 | Output throughput (tok/s) | 1413.25 | 1438.02 | +1.8% |
| | Mean / median TTFT (ms) | 4801.20 / 1760.71 | 4488.67 / 1731.95 | −6.5% / −1.6% |
| | Mean / median TPOT (ms) | 40.51 / 42.25 | 40.03 / 41.89 | −1.2% / −0.9% |
| 32K/128, conc 8 | Total throughput (tok/s) | 19462.14 | 19742.89 | +1.4% |
| | Mean / median TTFT (ms) | 4369.27 / 4074.80 | 4289.94 / 4000.45 | −1.8% / −1.8% |
| | Mean / median TPOT (ms) | 69.21 / 71.28 | 68.35 / 70.37 | −1.2% / −1.3% |

## Notes

- **CUDA:** the shared-code edits are behaviour-preserving moves into overridable methods. CUDA was not run locally, so CUDA CI is relied on.
- **Not in scope:** V4.1's candidate-pool (sparse-logits) path. DeepSeek-V4 has no two-level indexer.
- **Duplicate-work check:** _TODO before opening the PR. Search open PRs for "ROCm DeepSeek V4 MXFP4 indexer" and confirm no existing PR covers this._
- **AI assistance:** this change was written with AI assistance (Claude Code). The tests and evaluations above were run on the hardware listed. _TODO before opening the PR: the submitter reviews every changed line._

🤖 Generated with [Claude Code](https://claude.com/claude-code)
