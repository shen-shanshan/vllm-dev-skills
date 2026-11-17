**Title:** `[ROCm][Perf][DSv4.1] Fuse the MXFP8 input quant of fused_wqa_wkv and the shared expert into their producers`

## Overview

On gfx950, DeepSeek-V4.1's MXFP8 linears quantize their bf16 input in a separate `mxfp8_e4m3_quantize` launch. This PR moves that quant into the producing kernel for three inputs:

- `fused_wqa_wkv` and the shared expert's `gate_up_proj`: AITER Mega-MHC also writes its layer input as MXFP8;
- the shared expert's `down_proj`: a new Triton kernel fuses `silu_and_mul_with_clamp` with the quant.

Outputs are bit-identical.

## Claims

On 2x MI355X, DEP2, prefill steps of about 16.4k tokens per rank:

- **Standalone MXFP8 quant kernels:** 211 (4.73 ms) → 91 (2.06 ms) per step.
- **Fused kernel against the two unfused kernels** (median per full-length layer, in-trace):
  - attention seam (Mega-MHC + quant): about 5-22 us faster;
  - FFN seam (Mega-MHC + quant): about 10-20 us faster;
  - shared-expert activation + quant: 67 → 38 us (−43%).
- **Bit-identical outputs**, and GSM8K is unchanged within noise.
- **Scope:** only ROCm-only files change, and every other backend keeps its current path.

## Validation

**Unit test (MI355X).** `tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py` checks both producers against the unfused path, bit for bit: 22 passed.

```bash
pytest tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py
```

- `mhc_pre_delayed_mxfp8` against `MHCPreDelayedOp` on the Mega-MHC seam, with and without the folded post block, at T = 1, 300 and 4096:
  - all five outputs are identical;
  - the attached MXFP8 copy equals `mxfp8_e4m3_quantize` of the layer input.
- `silu_and_mul_mxfp8` against `SiluAndMul` / `SiluAndMulWithClamp(10)` followed by `mxfp8_e4m3_quantize`, at M = 1, 17, 300 and 5000 and D = 576 and 2304.

**Kernel microbenchmark (MI355X, CUDA-graph replay, M = 16384).**

| Fusion | Unfused (two kernels) | Fused | Δ |
|---|---:|---:|---:|
| Mega-MHC seam (H = 5120) | 380 + 39 us | 396 us | −23 us |
| SiLU-and-mul with clamp (D = 2304) | 64.9 + 18.3 us | 35.6 us | −47.6 us (−57%) |

For the SiLU-and-mul kernel, smaller M from 1 to 4096 saves 2.2-12.8 us per call. At every M measured, the fused kernel is no slower than the unfused activation kernel alone.

**Prefill profile.**

Setup:

- DeepSeek-V4.1-Flash on 2x MI355X, DEP2 (`--data-parallel-size 2 --enable-expert-parallel --moe-backend aiter_mega_moe --all2all-backend mori`), AITER on, DSpark k=5, `--max-num-batched-tokens 16384`.
- Load: `vllm bench serve` with random 8192 in / 1024 out at concurrency 256.
- Capture: torch profiler on the first iterations after `/start_profile`.

The table compares the same three full prefill steps in each run, as the median over full-length layers 2-19. The baseline ran on GPUs 4,5, the #2 + #3 run also on GPUs 4,5, and the #2 + #3 + #4 run on GPUs 0,1.

| Per full-length layer | Base | #2 + #3 | #2 + #3 + #4 |
|---|---:|---:|---:|
| #2 attention seam: Mega-MHC + quant → Mega-MHC (MXFP8) | 434-443 us | 422-433 us | 413-420 us |
| #3 FFN seam: Mega-MHC + `gate_up` quant → Mega-MHC (MXFP8) | 449-459 us | 440-443 us | 428-436 us |
| #4 `silu_and_mul_with_clamp` + quant → fused kernel | 66.8-67.2 us | 67.9-68.2 us | 37.9-38.5 us |
| Standalone quant kernels per step | 211 (4.73 ms) | 131 (2.58 ms) | 91 (2.06 ms) |

A few notes on the numbers:

- In MXFP8 mode, Mega-MHC costs about 15 us more per launch in the microbenchmark and 25-30 us more in place. That cost takes back much of the 41 us quant it removes, so #2 and #3 gain less than #4.
- The spread between the two fused runs for #2 and #3 is about 10 us, part of which is the different GPU pair.
- MegaMoE dominates the step (about 70%), so end-to-end prefill time moves by roughly 1 ms out of about 480 ms, within run-to-run noise.

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
| #2 + #3 | 0.9356 ± 0.0068 | 0.9348 ± 0.0068 |
| #2 + #3 + #4 (this PR) | 0.9265 ± 0.0072 | 0.9272 ± 0.0072 |

The kernels are bit-exact, so the spread is run-to-run noise from DSpark's probabilistic drafting and batching. It stays within about one standard error of the difference between two runs.

`pre-commit run` passes on the changed files.

## Details

**Why.** On the AITER FlyDSL MXFP8 path (#67377), every dense linear without a pre-quantized input runs `mxfp8_e4m3_quantize` before its GEMM. `RocmDotScaledMxfp8LinearKernel` already reads a `QuantizedActivation` with row-major `[M, K / 32]` E8M0 scales. So the producer can hand over the quantized copy, and the GEMMs need no change.

**Attention and FFN seams (#2, #3).** Mega-MHC can write the MXFP8 copy alongside the bf16 norm (`out_dtype="mxfp8"`).

- `mhc_pre_delayed_mxfp8` attaches the `QuantizedActivation` to the bf16 layer input tensor as a Python attribute (`attach_mxfp8_input`).
- `fused_wqa_wkv` (ROCm attention `_fused_wqa_wkv_gemm`) and the shared expert's `gate_up_proj` (`DeepseekV4MLP.forward`) read it back with `mxfp8_input`.
- Any reshape, slice or gather creates a new tensor without the attribute, so a consumer only ever sees the copy of the exact rows it projects. This also keeps the shared `DeepseekV4Attention` and MoE `forward` signatures unchanged.
- The bf16 layer input is still produced for the router, compressor, indexer `weights_proj` and MegaMoE.

**Shared-expert activation (#4).** `silu_and_mul_mxfp8` does the gate/up clamp, SiLU and multiply in fp32, rounds to bf16 (the value the unfused path materializes), and quantizes it with the same arithmetic as `_mxfp8_quant_triton_kernel`. It writes `down_proj`'s MXFP8 input in one launch.

- Besides removing the quant launch, it no longer writes the bf16 activation and reads it back (about 150 MB at M = 16384). That is why it beats the unfused activation kernel alone at large M.
- The tile tiers (`_config`) were swept on MI355X for D = 2304.

**Key changes.**

- `vllm/_aiter_ops.py`: the Mega-MHC wrapper body moves into `_mega_mhc_seam`. A new `mhc_mega_post_pre_delayed_rms_norm_mxfp8` calls it with `out_dtype="mxfp8", scale_row_major=True`. The existing bf16 call is unchanged.
- `vllm/models/deepseek_v4/amd/silu_and_mul_mxfp8.py` (new): the fused activation + MXFP8 quant Triton kernel.
- `vllm/models/deepseek_v4/amd/model.py`:
  - adds `attach_mxfp8_input` / `mxfp8_input` and the `takes_rocm_mxfp8_input` eligibility check;
  - `DeepseekV4MLP.forward` feeds an attached copy to `gate_up_proj`;
  - it also feeds `silu_and_mul_mxfp8`'s output to `down_proj` when `down_proj` reads MXFP8 as is.
- `vllm/models/deepseek_v41/amd/mhc.py`: adds `mhc_pre_delayed_mxfp8`, which runs the Mega-MHC seam in MXFP8 mode under the same conditions `MHCPreDelayedOp.forward_hip` uses for Mega-MHC, and otherwise defers to `MHCPreDelayedOp`.
- `vllm/models/deepseek_v41/amd/model.py`: the decoder layer decides once, after weight loading, whether each seam's consumer can read the copy, and then routes the attention seams (including `write_kv`) and the FFN seam through `mhc_pre_delayed_mxfp8`.
- `vllm/models/deepseek_v41/amd/rocm.py`: `_fused_wqa_wkv_gemm` reads the attached copy.
- `tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py` (new): bit-exactness tests for both producers.

**When it applies.** Each fusion requires the consumer linear to be `RocmDotScaledMxfp8LinearKernel` or `AiterFlyDSLMxfp8LinearKernel` with an MXFP8 input key and `supports_quantized_input`. On top of that:

- #2 / #3 require Mega-MHC (gfx950, AITER on, `hc_mult == 4`, no ubatching) and no sequence parallelism. #3 additionally requires MegaMoE with an unfused shared expert.
- #4 requires a contiguous 2-D `gate_up` output and `down_proj.input_is_parallel`.

Anything else keeps the current path.

**Limitation.** Decode steps with at most `MHC_OVERLAP_MAX_TOKENS` (128) tokens per rank inside FULL CUDA-graph capture use the mHC overlap path, whose layer input comes from the Triton collapse rather than Mega-MHC. Those steps keep the separate quant for #2 and #3; #4 still applies there.

**Other backends.** Only ROCm-only modules change: `vllm/_aiter_ops.py`, `vllm/models/deepseek_v4/amd/`, `vllm/models/deepseek_v41/amd/`. The shared `MHCPreDelayedOp`, `DeepseekV4Attention`, activation, linear and quantization code are untouched, so CUDA, XPU and CPU behave exactly as before. DeepSeek-V4 (not V4.1) shares `DeepseekV4MLP`, but its linears are not MXFP8, so it keeps its current path.

**Not a duplicate.** The open PRs on nearby quants each fuse a different one:

- #66572 and #65837 fuse the q-norm quant before `wq_b`;
- #63821 and #66856 fuse the `wo_a` epilogue quant before `wo_b`.

None of them touches the Mega-MHC seam inputs or the shared expert.

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
