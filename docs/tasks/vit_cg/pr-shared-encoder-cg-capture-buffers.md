**Title:** `[Perf][Multimodal] Share encoder CUDA graph capture buffers across token budgets`

## Overview

`EncoderCudaGraphManager` allocates a full set of input/output buffers per token budget. This PR makes smaller budgets use prefix views of the largest budget's buffers, which roughly halves persistent capture-buffer memory for every `SupportsEncoderCudaGraph` model. It also fixes Step3-VL returning graph-output views despite `clone=True`. Follow-up to RFC #38175.

## Claims

- Capture buffer memory is roughly halved, and the freed memory goes to the KV cache (MI355X):

  | Model | Capture buffers | KV cache tokens |
  |---|---|---|
  | Qwen3-VL-2B-Instruct | 974.1 → 488.0 MiB | +5,120 |
  | DeepSeek-V4.1-Flash (TP2, per rank) | 724.1 → 362.8 MiB | +8,927 |
  | Gemma4-E2B-it (capture axes) | 1356.3 → 539.7 MiB | +54,202 |
  | Step3-VL-10B (dual path) | 1018.7 → 430.1 MiB | +11,264 |

- Encoder outputs match unshared capture: bit-exact for Qwen3-VL, DeepSeek-V4.1 and Gemma4, and within Step3-VL's own replay noise.
- Fixes Step3-VL: an image without local patches got a view of the `global` graph output. A later batch in the same `execute()` overwrote it before it was returned. This already happens on main when both batches hit the same budget; sharing would extend it to any budget.

## Validation

**Unit tests** (`tests/v1/cudagraph/test_encoder_cudagraph.py`, +59 lines). Each new test fails when its fix is reverted:

- `test_shared_capture_buffers_match_unshared_capture` checks two things:
  - smaller budgets alias the largest budget's buffers;
  - one `execute()` that replays budget 64 then 16 returns every item bit-identical to an unshared capture.

  It fails if sharing is disabled or if the manager stops cloning outputs.
- `test_step3_vl_postprocess_clones_global_only_item` covers the Step3-VL fix.

```bash
pytest tests/v1/cudagraph/test_encoder_cudagraph.py tests/v1/worker/test_encoder_runner.py  # 97 passed, 7 skipped
```

**End-to-end A/B on MI355X** (Model Runner V2, auto-inferred budgets).

Greedy outputs vary between runs of the same build because batch composition differs, so the harness compares the encoder directly, inside each worker:
1. Record every replayed batch and every `execute()` call during generation.
2. Capture a second manager with sharing disabled.
3. Rerun the recorded inputs on both managers and compare with `torch.equal`:
   - each batch, in five orders across budgets;
   - each `execute()` call as a whole.

| Model | Batches / budgets | Batch A/B | `execute()` A/B |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | 11 / 7 | exact | — |
| DeepSeek-V4.1-Flash (both ranks) | 9 / 3 | exact | — |
| Gemma4-E2B-it | 13 / 2 | exact | exact (9 calls, 1 multi-budget) |
| Step3-VL-10B | 21 / 6 | max \|Δ\| 0.031 | max \|Δ\| 0.031; **14.13 with the fix reverted** |

Gemma4 and Step3-VL ran with `encoder_cudagraph_max_vision_items_per_batch=2`, so a 3-image prompt spans two budgets.

Step3-VL's 0.031 is not caused by sharing. Replaying one unshared graph twice on the same input differs by the same amount, on the same smallest budgets.

**Not covered:**
- NVIDIA beyond CI unit tests.
- DeepSeek-OCR, which fails on ROCm before capture, independent of this PR (`deepencoder.py` Triton kernel does not compile).

## Details

**Why sharing is safe.** Graphs of one path and capture-axis keys never replay concurrently. Every input is rewritten before each replay, and `_execute_local` asks `postprocess_encoder_output` to clone outputs. Budgets are already captured largest first, so smaller budgets get `shared[:n]` views keyed by `(path, axis_keys, key)`.

**What is shared.** Only the output buffer and `buffer_keys` inputs that are contiguous, non-0-dim device tensors with matching dtype and trailing dims. Everything else stays per budget, e.g. the CPU `max_seqlen` scalar baked into the graph, or per-axis-key shapes.

**Guard.** Slots handed out as views are recorded in `_aliased_slots`. `_run_budget_graph` raises if replay omits a value for one, since its capture-time contents now belong to another budget.

**`clone` contract.** Sharing the output buffer requires `postprocess_encoder_output(clone=True)` to return copies. All in-tree implementations already did, except Step3-VL's single-part branch, which this PR fixes. The requirement is now documented in the protocol docstring and in `docs/design/cuda_graphs_multimodal.md`.

**Why not `WorkspaceManager`.** Its persistent resources outlive `EncoderCudaGraphManager.clear()`, which would skew V1's `profile_cudagraph_memory` (it captures into a throwaway manager and clears it).

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
