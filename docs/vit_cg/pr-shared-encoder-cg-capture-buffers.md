**Title:** `[Perf][Multimodal] Share encoder CUDA graph capture buffers across token budgets`

## Overview

`EncoderCudaGraphManager` keeps a full set of input and output buffers for every captured token budget. This PR lets the smaller budgets record prefix views of the largest budget's buffers, which halves the persistent capture-buffer memory for every model that implements `SupportsEncoderCudaGraph`. Follow-up to the ViT CUDA graph RFC #38175.

## Claims

- Encoder CUDA graph capture buffers (inputs listed in `buffer_keys`, plus the output buffer) are shared across the token budgets of the same path and capture-axis keys. Their memory drops by about half because budgets are powers of two. Measured on MI355X:
  - Qwen3-VL-2B-Instruct: 974.1 → 488.0 MiB.
  - DeepSeek-V4.1-Flash: 724.1 → 362.8 MiB per TP rank.
- The freed memory goes to the KV cache: +5,120 tokens on Qwen3-VL-2B and +8,927 tokens on DeepSeek-V4.1-Flash TP2.
- Encoder outputs are bit-identical to unshared capture. This holds for every replayed batch, in any replay order across budgets.
- No model code changes; the change is confined to `vllm/v1/worker/encoder_cudagraph.py`.

## Validation

**Unit tests.** New `TestEncoderCudaGraphSharedBuffers` in `tests/v1/cudagraph/test_encoder_cudagraph.py` covers five cases:
- Smaller budgets record prefix views of the largest budget's buffers.
- Ineligible buffers stay separate per budget: CPU scalars and buffers whose trailing dims vary with the budget.
- Interleaved-budget replay is bit-identical to a manager captured without sharing.
- Replay that omits a shared key raises.
- `clear()` releases the shared buffers.

```bash
pytest tests/v1/cudagraph/test_encoder_cudagraph.py tests/v1/worker/test_encoder_runner.py
# 100 passed, 7 skipped (MI355X, ROCm)
```

**End-to-end A/B on MI355X (Model Runner V2, `cudagraph_mm_encoder=True`, auto-inferred budgets 64 … 16384, 9 encoder graphs).** The harness first sends 7 image prompts at 5 resolutions, including one 3-image prompt, as a batch. It then sends 7 more prompts with different images one at a time, so each image replays its own budget graph.

Memory comes from the unique storages behind every captured graph's buffers. KV cache size comes from the startup log and is identical across repeated runs.

| Model / config | Capture buffers, before → after | KV cache tokens, before → after | Encoder graph capture log |
|---|---|---|---|
| Qwen3-VL-2B-Instruct, TP1, `max_num_batched_tokens=16384` | 974.1 → **488.0 MiB** | 1,508,352 → **1,513,472** | 4.00 → 3.48 GiB (3/3 runs) |
| DeepSeek-V4.1-Flash, TP2, aiter MoE, `max_num_batched_tokens=16384` | 724.1 → **362.8 MiB** per rank | 3,468,391 → **3,477,318** | 5.94 / 5.44 → 5.71 / 5.21 GiB in 2 of 3 runs; under TP this device-level number varies by ±0.5 GiB between runs |

**Bit-exactness.** End-to-end greedy outputs are not a usable signal here: two runs of the same build already disagree on 3 of 14 Qwen3-VL prompts and 12 of 14 DeepSeek-V4.1 prompts, because the batch composition varies between runs. So the harness compares the encoder directly:
1. Record the inputs of every batch that was replayed during generation.
2. Capture a second manager in the same worker with sharing disabled.
3. Replay every batch on both managers in five orders (forward, reverse, budget-descending, budget-ascending, shuffled) and compare the outputs with `torch.equal`.

| Model | Replayed batches | Budgets hit | Mismatches |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | 16 | 64, 128, 256, 512, 1024, 2048 | 0 in all 5 orders |
| DeepSeek-V4.1-Flash (both TP ranks) | 9 | 256, 1024, 4096 | 0 in all 5 orders |

<details>
<summary>Reproduce</summary>

The harness builds the `LLM`, reads buffer stats and installs the recorder through `collective_rpc`, generates, then runs the A/B replay check.

```bash
# build under test (with this PR) and baseline (without it)
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python harness.py qwen3_vl out/qwen3_vl_final.json
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python harness.py dsv41 out/dsv41_final.json   # TP2
python compare.py qwen3_vl base final
```

DeepSeek-V4.1-Flash engine args: `tokenizer_mode=deepseek_v41, tensor_parallel_size=2, moe_backend=aiter, block_size=128, attention_config={"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}, engram_config={"cpu_offload":true}, max_model_len=32768`. Environment: `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True`.
</details>

**Not validated here.** NVIDIA GPUs (CI covers the unit tests), and models other than Qwen3-VL and DeepSeek-V4.1. The sharing logic is model-agnostic, but only those two were run end to end.

## Details

**Why sharing is safe.**
- Graphs of one path and capture-axis keys never replay concurrently.
- Before every replay, `_run_budget_graph` zeroes or pads each input buffer and copies the batch into it.
- `_execute_local` clones the output (`clone=True`) before the next replay.

So no graph depends on buffer contents left by another budget. Budgets are already captured largest first, so the first buffer seen for a slot is the largest, and smaller budgets get `shared[:n]` views. The smaller budget's dummy capture data is copied into its view before capture.

**What is shared.** A buffer is shared only if:
- its key is in `EncoderCudaGraphConfig.buffer_keys`;
- it is a contiguous, non-0-dim tensor on the device;
- it differs from the shared buffer only by a shorter leading dim, with the same dtype and device.

Everything else keeps its own buffer as before. This covers:
- CPU scalars such as `max_seqlen`, whose value is baked into the graph;
- buffers whose trailing dims vary with the budget;
- per-axis-key shapes (Gemma4 pixel grids, MiniCPM-V patch grids), since slots are keyed by `(path, axis_keys, key)`.

The output buffer uses the same slot scheme. Paths never alias each other, so dual-path encoders (Step3-VL, DeepSeek-OCR) are unaffected.

**Guard.** A buffer counts as shared only after a second graph records a view of it; this is computed after all budgets are captured. A model that relied on capture-time contents of such a buffer would read another budget's data. So `_run_budget_graph` raises if replay omits a shared key, instead of silently using stale data. No in-tree model omits one today; MiniMax-M3 drops `max_seqlen`, which is a CPU scalar and never shared.

**Why not `WorkspaceManager`.** Its `get_persistent` resources outlive `EncoderCudaGraphManager.clear()`. V1's `profile_cudagraph_memory` captures into a throwaway manager and clears it, so persistent resources would skew that estimate. A manager-owned dict keeps the buffers' lifetime tied to the graphs.

**Key changes** (`vllm/v1/worker/encoder_cudagraph.py`):
- `_get_shared_buffer()` returns a prefix view of a slot's shared buffer, or `None` when the buffer is ineligible.
- `_capture_budget_graph()` routes `buffer_keys` inputs and the output buffer through `_get_shared_buffer()`.
- `capture()` finalizes `BudgetGraphMetadata.shared_keys`.
- `_run_budget_graph()` adds the guard; `clear()` drops the shared buffers.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
