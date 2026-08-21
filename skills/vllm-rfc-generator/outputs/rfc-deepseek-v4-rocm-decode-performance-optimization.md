# [RFC]: DeepSeek-V4 ROCm Decode Performance Optimization

> **Labels**: RFC
> **Status**: Open for Feedback
> **Feedback Period**: At least 1 week

---

## Motivation

DeepSeek-V4 introduces CSA (Compressed Sparse Attention, `compress_ratio=4`), HCA layers, and a large MoE backbone. A side-by-side trace comparison of the vLLM decode path against vLLM-ATOM on ROCm shows that vLLM still leaves measurable performance on the table. The gaps cluster into three areas:

1. **Unfused mHC kernels.** vLLM runs `mhc_post` and `mhc_pre` as separate kernels, and `mhc_pre_big_fuse` + `add_rmsnorm_quant` as two kernels, while ATOM fuses each pair into a single kernel (`mhc_fused_post_pre` and `mhc_pre_big_fuse_rmsnorm`). Every extra kernel launch on the decode critical path costs launch latency and cache/register thrash.
2. **Serial execution where overlap is possible.** ATOM forks auxiliary HIP streams to run compressor GEMMs (CSA pre-attention), SWA KV insertion (HCA), and the MoE router/shared experts concurrently with the main stream. vLLM executes all of these serially.
3. **Slower attention support kernels.** ATOM replaces the vLLM CSA top-k selection kernels and the post-attention batched GEMMs (`torch.mm` + `manual_unroll`) with dedicated optimized kernels (e.g., `batched_gemm_bf16_kernel`).

An earlier attempt at closing gap #2 — PR #51794, which enables three-stream overlap for CSA pre-attention — is currently a draft: the overlapped region itself is faster when there is no bubble, but some ranks see a much slower tail all-reduce due to inter-rank latency jitter. The un-optimized kernels running in the parallel streams contend for hardware resources and desynchronize ranks. This suggests the correct order of work is: **replace the kernels with ATOM's optimized versions first, then enable stream overlap on top of them.**

## Proposed Change

Port the ATOM-optimized decode path into vLLM's ROCm DeepSeek-V4 implementation, using the ATOM trace as the reference for each change. The work is organized per module:

### 1. CSA

- **Fuse mHC kernels** (started in #52737): merge `mhc_post` + `mhc_pre` into `mhc_fused_post_pre`, and `mhc_pre_big_fuse` + `add_rmsnorm_quant` into `mhc_pre_big_fuse_rmsnorm`.
- **Pre-attention three-stream overlap** (based on #51794): before the input GEMMs, fork three HIP streams:
  - **default stream**: fused wqa+wkv GEMM → q/kv RMSNorm → wq_b → qnorm/RoPE → SWA KV insert;
  - **aux stream 0**: main-compressor wkv_gate GEMM → compress kernels (compressed KV cache write);
  - **aux stream 1**: indexer-compressor wkv_gate GEMM → K-cache write.

  After the join, the indexer weights GEMM, the indexer q-side (wq_b + fused q RoPE+quant), the sparse indexer op (top-k selection), and the MLA attention run serially on the main stream. Each aux stream is a self-contained chain starting from `hidden_states`, so no fine-grained cross-stream dependencies are required. **This item stays disabled (draft) until the kernels in the parallel streams are replaced with ATOM's**, to avoid the regression observed in #51794.
- **Replace top-k kernels**: between `gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` and the core `sparse_attention`, replace the vLLM top-k implementation with ATOM's fused sparse-indexer kernels.
- **Replace post-attention batched MM**: after `inverse_rope_gptj_kernel` (and before fp8 quant / `cross_device_reduce_2stage`), replace `torch.mm` + `manual_unroll` with `batched_gemm_bf16_kernel`.

### 2. HCA

- **Pre-attention two-stream overlap**: for `c128a` layers (which have no indexer), run the main KV compression concurrently with SWA token insertion on an auxiliary stream.
- **Kernel replacement**: port ATOM's HCA kernels (details to follow in a follow-up analysis).

### 3. MoE

- **Router and shared-experts dual-stream overlap**: run the router computation concurrently with the shared experts on a second stream, then join before the routed experts.
- **Kernel replacement**: port ATOM's MoE kernels (details to follow in a follow-up analysis).

### Key Design Decisions

- **Kernel replacement before stream overlap.** PR #51794 showed that overlapping un-optimized kernels causes hardware resource contention and inter-rank latency jitter, which can make the tail all-reduce slower than the serial baseline. We therefore land kernel replacement first and keep multi-stream overlap disabled until the parallel-stream kernels are ATOM-optimized.
- **ATOM trace as the reference.** Each change is validated against the corresponding ATOM trace segment (per-kernel duration, stream assignment, bubble visibility) and against end-to-end decode benchmarks.
- **Per-module, incremental PRs.** Each item above lands as a self-contained PR (fusion, overlap, or kernel replacement), so regressions can be bisected at the kernel level.

## Related PRs

**Implementation:**
- [#52737](https://github.com/vllm-project/vllm/pull/52737): [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER — the first item being worked on.

**Background / Motivation:**
- [#51794](https://github.com/vllm-project/vllm/pull/51794): [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4 — kept as draft until the parallel-stream kernels are replaced with ATOM's optimized versions.

## Feedback Period

We plan to keep this RFC open for at least 1 week (until 2026-08-26). Please share your thoughts in the comments.

## CC List

@<github-handles-of-reviewers>  <!-- TODO: fill in -->

## Any Other Things

- The full trace comparison (vLLM vs vLLM-ATOM decode traces with screenshots) lives in the analysis doc `vllm/model_performance/deepseek_v4/vllm_vs_vllm_atom_trace.md` and the internal wiki: https://amd.atlassian.net/wiki/x/hwHAbw.
- Open questions:
  - **Upstreaming strategy for ATOM kernels**: naming, licensing, and whether fused kernels should be gated behind an env var or enabled by default on ROCm.
  - **Quantified perf targets**: per-item decode latency / token throughput targets pending benchmark runs on MI300-series GPUs.
  - **HCA / MoE kernel replacement details**: these two items need their own detailed trace analysis before implementation.

---

> Please take a look at previous [RFCs](https://github.com/vllm-project/vllm/issues?q=label%3ARFC+sort%3Aupdated-desc) for reference.
