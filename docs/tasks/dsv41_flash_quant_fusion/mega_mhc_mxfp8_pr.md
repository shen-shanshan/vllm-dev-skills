**Title:** `[ROCm][Perf][DSv4.1] Fuse the MXFP8 input quant of fused_wqa_wkv and the shared expert into their producers`

## Overview

On gfx950 (AITER FlyDSL MXFP8 path, #67377), DeepSeek-V4.1's MXFP8 linears quantize their bf16 input in a separate `mxfp8_e4m3_quantize` launch. This PR moves that quant into the producing kernel for three linears:

- **#1 `fused_wqa_wkv`**: AITER Mega-MHC also writes its attention-seam layer input as MXFP8.
- **#2 shared expert `gate_up_proj`**: likewise for the FFN-seam layer input.
- **#3 shared expert `down_proj`**: a new Triton kernel fuses `silu_and_mul_with_clamp` with the MXFP8 quant.

Outputs are bit-identical to the unfused path.

## Claims

- Removes three standalone MXFP8 quant launches per layer, and the bf16 round trip of the shared expert's activation.
- On 2x MI355X DEP2, the forward step is 0.23-0.38% faster for 8k / 16k-token prefill and 768-token decode steps.
- Bit-identical outputs; GSM8K unchanged.
- ROCm-only change; other backends keep their current path.

## Validation

**Unit test (MI355X)**: both producers are bit-identical to the unfused path (22 passed).

```bash
pytest tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py
```

**Forward step time** (`vllm bench forward`, DeepSeek-V4.1-Flash, 2x MI355X, DEP2, AITER MegaMoE, fp8 KV, DSpark k=5). The times are per DP rank, median graph-replay time of one full step, including the sampler and drafter.

| Case (per DP rank) | Base | This PR | Δ |
|---|---:|---:|---:|
| prefill-16k: 1 × 16384-token prefill | 393.36 ms | 392.46 ms | −0.23% |
| prefill-8k: 1 × 8192-token prefill | 214.41 ms | 213.62 ms | −0.37% |
| decode-128: 128 × verify (1 + 5 drafts), KV ≈ 16k | 71.81 ms | 71.54 ms | −0.38% |

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
torchrun --nproc-per-node=2 -m vllm.entrypoints.cli.main bench forward \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer-mode deepseek_v41 --language-model-only \
  -dp 2 --enable-expert-parallel --moe-backend aiter_mega_moe --all2all-backend mori \
  --engram-config '{"cpu_offload":true}' \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 --kv-cache-dtype fp8_e4m3 --max-model-len 65536 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}' \
  --routing balanced \
  --case prefill16k=16384 --case prefill8k=8192 --case "decode128=128*6@16384:5"
```

End to end on AgentX (c256, DEP2), throughput and interactivity before and after the fusion are on par.

**Trace before / after (prefill step, one full-length layer)**

<!-- TODO: before / after trace screenshots -->

**Accuracy (GSM8K)**: 8-shot, all 1319 questions, settings of `.buildkite/lm-eval-harness/configs/DeepSeek-V4-Flash-MXFP4.yaml`, against a DEP2 server.

| Run | strict-match | flexible-extract |
|---|---:|---:|
| Base | 0.9257 | 0.9272 |
| This PR | 0.9265 | 0.9272 |

```bash
lm_eval --model local-completions \
  --model_args "model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None,num_concurrent=64,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 8 --limit 1319 --batch_size 1
```

## Details

`RocmDotScaledMxfp8LinearKernel` already reads a `QuantizedActivation` with row-major `[M, K / 32]` E8M0 scales, so the producers hand over the quantized copy and the GEMMs are unchanged.

- **#1 / #2:** `mhc_pre_delayed_mxfp8` runs the Mega-MHC seam with `out_dtype="mxfp8"`. It attaches the `QuantizedActivation` to the bf16 layer input as an attribute (`attach_mxfp8_input`), which `fused_wqa_wkv` and the shared expert's `gate_up_proj` read back. Any reshape, slice or gather drops the attribute, so a consumer only sees the copy of the rows it projects. The bf16 input stays available for the router, compressor, indexer and MegaMoE, and the shared `forward` signatures are unchanged.
- **#3:** `silu_and_mul_mxfp8` computes the clamped SiLU-and-mul in fp32, rounds it to bf16 as the unfused path does, and quantizes it with `_mxfp8_quant_triton_kernel`'s arithmetic.

Key changes:

- `vllm/_aiter_ops.py`: `mhc_mega_post_pre_delayed_rms_norm_mxfp8`.
- `vllm/models/deepseek_v41/amd/mhc.py`: `mhc_pre_delayed_mxfp8`.
- `vllm/models/deepseek_v41/amd/model.py`: per-seam opt-in, decided once after weight loading.
- `vllm/models/deepseek_v41/amd/rocm.py`: `_fused_wqa_wkv_gemm` reads the MXFP8 copy.
- `vllm/models/deepseek_v4/amd/silu_and_mul_mxfp8.py` (new): the #3 Triton kernel.
- `vllm/models/deepseek_v4/amd/model.py`: helpers, plus the shared-expert wiring for #2 and #3.
- `tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py` (new).

**When it applies:** gfx950 with AITER, the consumer linear on `RocmDotScaledMxfp8LinearKernel` / `AiterFlyDSLMxfp8LinearKernel`, and no sequence parallelism. In addition:

- #1 / #2 need Mega-MHC; #2 also needs MegaMoE with an unfused shared expert.
- Decode steps of at most 128 tokens per rank inside FULL graphs use the mHC overlap path, where #1 / #2 do not apply.

**Other backends:** only ROCm-only modules change (`vllm/_aiter_ops.py`, `models/deepseek_v4/amd/`, `models/deepseek_v41/amd/`). Shared layers are untouched, so CUDA, XPU and CPU are unaffected. DeepSeek-V4 shares `DeepseekV4MLP`, but its linears are not MXFP8, so it keeps its current path.

**Not a duplicate:** open PRs #66572 / #65837 (q-norm quant before `wq_b`) and #63821 / #66856 (`wo_a` epilogue quant before `wo_b`) fuse different quants.

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
