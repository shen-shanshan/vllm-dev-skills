**Title:** `[ROCm][Perf][DSv4.1] Quantize the Mega-MHC layer input to MXFP8 for fused_wqa_wkv and the shared expert`

## Overview

On gfx950, DeepSeek-V4.1's fused mHC seam (AITER Mega-MHC) writes a bf16 layer input, which `fused_wqa_wkv` and the shared expert's `gate_up_proj` then quantize to MXFP8 in separate launches. This PR has Mega-MHC also write that input as MXFP8 (row-major E8M0 scales, `out_dtype="mxfp8"`), so both GEMMs read it directly. Output is bit-identical.

## Claims

- Removes the standalone MXFP8 quant launch before `fused_wqa_wkv` and before the shared expert's `gate_up_proj` on the prefill path. On a DEP2 MI355X prefill step (~16.4k tokens per rank), standalone quant kernels drop from 211 (4.73 ms) to 131 (2.58 ms).
- Per full-length layer, the two Mega-MHC launches plus the two quants go from 885-896 us to 859-876 us, about 0.5 ms per ~480 ms prefill step. Mega-MHC costs about 25-30 us more per seam in MXFP8 mode, which takes back part of the 41 us quant it replaces.
- Bit-identical outputs: the quantized values and scales equal `mxfp8_e4m3_quantize` of the same bf16 layer input. GSM8K is unchanged within noise.
- Only ROCm-only files change; every other backend, and ROCm configs the gating rules out, keep the current path.

## Validation

**Bit-exactness (MI355X).**

- AITER `mega_mhc(..., out_dtype="mxfp8")` against `mxfp8_e4m3_quantize` of the bf16 layer input, at T = 1, 7, 64, 300, 4096 and 16384: the e4m3 values and E8M0 scales are identical. The bf16 norm, the gates and the next pre-mix are identical to the bf16 mode.
- `mhc_pre_delayed_mxfp8` against `MHCPreDelayedOp` (Mega-MHC seam), at T = 1, 300 and 16384, with and without the folded post block: all five outputs are identical, and the attached `QuantizedActivation` equals `mxfp8_e4m3_quantize` of the returned layer input.

**Prefill profile.**

Setup:

- DeepSeek-V4.1-Flash on 2x MI355X, DEP2 (`--data-parallel-size 2 --enable-expert-parallel --moe-backend aiter_mega_moe --all2all-backend mori`), AITER on, DSpark k=5, `--max-num-batched-tokens 16384`.
- Load: `vllm bench serve` with random 8192 in / 1024 out at concurrency 256.
- Capture: torch profiler on the first iterations after `/start_profile`, comparing the same full prefill steps (~16.4k tokens per rank).

| Per prefill step | Base | This PR |
|---|---:|---:|
| Mega-MHC → quant → `fused_wqa_wkv` | 40 | 0 |
| quant → shared expert `gate_up_proj` | 40 | 0 |
| Standalone MXFP8 quant kernels | 211 (4.73 ms) | 131 (2.58 ms) |
| Full-length layer: 2x Mega-MHC + seam quants (median, layers 2-19) | 885-896 us | 859-876 us |

Mega-MHC microbenchmark (T = 16384, H = 5120): 380 us in bf16 mode against 396 us in MXFP8 mode, while a separate `mxfp8_e4m3_quantize` takes 39 us.

<details>
<summary>Server command</summary>

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 --language-model-only \
  --data-parallel-size 2 --enable-expert-parallel \
  --moe-backend aiter_mega_moe --all2all-backend mori \
  --engram-config '{"cpu_offload":true}' \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 --kv-cache-dtype fp8_e4m3 --max-model-len 32768 \
  --max-num-batched-tokens 16384 --max-cudagraph-capture-size 1024 \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

</details>

**Accuracy (GSM8K).** Same server, settings of `.buildkite/lm-eval-harness/configs/DeepSeek-V4-Flash-MXFP4.yaml` (8-shot, all 1319 questions):

```bash
lm_eval --model local-completions \
  --model_args "model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None,num_concurrent=64,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 8 --limit 1319 --batch_size 1
```

| Run | strict-match | flexible-extract |
|---|---:|---:|
| Base | 0.9257 ± 0.0072 | 0.9272 ± 0.0072 |
| This PR | 0.9356 ± 0.0068 | 0.9348 ± 0.0068 |

The kernels are bit-exact, so the gap is run-to-run noise from DSpark's probabilistic drafting and batching. It is about one standard error of the difference between two runs.

`pre-commit run` passes on the changed files.

## Details

**Why.** On the AITER FlyDSL MXFP8 path (#67377), every dense linear without a pre-quantized input runs `mxfp8_e4m3_quantize` before its GEMM. `RocmDotScaledMxfp8LinearKernel` already reads a `QuantizedActivation` with row-major `[M, K / 32]` E8M0 scales, and Mega-MHC can write exactly that layout while it writes the norm. So the producer can hand over the quantized copy, and the GEMMs need no change.

**How the quantized copy reaches the GEMM.** `mhc_pre_delayed_mxfp8` attaches the `QuantizedActivation` to the bf16 layer input tensor as a Python attribute (`attach_mxfp8_input`). `fused_wqa_wkv` (ROCm attention `_fused_wqa_wkv_gemm`) and the shared expert (`DeepseekV4MLP.forward`) read it back with `mxfp8_input`.

- Any reshape, slice or gather creates a new tensor without the attribute, so a consumer can only see the copy of the exact rows it projects, and otherwise quantizes as before.
- This keeps the shared `DeepseekV4Attention` and MoE `forward` signatures unchanged.
- The bf16 layer input is still produced for the other readers: router, compressor, indexer `weights_proj` and MegaMoE.

**Key changes.**

- `vllm/_aiter_ops.py`: the Mega-MHC wrapper body moves into `_mega_mhc_seam`. A new `mhc_mega_post_pre_delayed_rms_norm_mxfp8` calls it with `out_dtype="mxfp8", scale_row_major=True`. The existing bf16 call is unchanged.
- `vllm/models/deepseek_v4/amd/model.py`: adds `attach_mxfp8_input` / `mxfp8_input` and the `takes_rocm_mxfp8_input` eligibility check. `DeepseekV4MLP.forward` feeds an attached copy to `gate_up_proj`.
- `vllm/models/deepseek_v41/amd/mhc.py`: adds `mhc_pre_delayed_mxfp8`, which runs the Mega-MHC seam in MXFP8 mode under the same conditions `MHCPreDelayedOp.forward_hip` uses for Mega-MHC, and otherwise defers to `MHCPreDelayedOp`.
- `vllm/models/deepseek_v41/amd/model.py`: the decoder layer decides once, after weight loading, whether each seam's consumer can read the copy, and then routes the attention seams (including `write_kv`) and the FFN seam through `mhc_pre_delayed_mxfp8`.
- `vllm/models/deepseek_v41/amd/rocm.py`: `_fused_wqa_wkv_gemm` reads the attached copy.

**When it applies.** Every condition below must hold; otherwise the layer keeps the current path.

- Mega-MHC is in use (gfx950, AITER on, `hc_mult == 4`, no ubatching).
- The consumer linear is `RocmDotScaledMxfp8LinearKernel` or `AiterFlyDSLMxfp8LinearKernel` with an MXFP8 input key and `supports_quantized_input`.
- No sequence parallelism.
- For the FFN seam: MegaMoE with an unfused shared expert.

**Limitation.** Decode steps with at most `MHC_OVERLAP_MAX_TOKENS` (128) tokens per rank inside FULL CUDA-graph capture use the mHC overlap path, whose layer input comes from the Triton collapse rather than Mega-MHC. Those steps keep the separate quant, so the gain is in prefill and in larger steps.

**Other backends.** Only ROCm-only modules change: `vllm/_aiter_ops.py`, `vllm/models/deepseek_v4/amd/`, `vllm/models/deepseek_v41/amd/`. The shared `MHCPreDelayedOp`, `DeepseekV4Attention`, linear and quantization code are untouched, so CUDA, XPU and CPU behave exactly as before. DeepSeek-V4 (not V4.1) shares `DeepseekV4MLP` but never attaches a copy, so it is unaffected.

**Not a duplicate.** The open PRs on nearby quants each fuse a different one:

- #66572 and #65837 fuse the q-norm quant before `wq_b`;
- #63821 and #66856 fuse the `wo_a` epilogue quant before `wo_b`.

None of them touches the Mega-MHC seam inputs.

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
