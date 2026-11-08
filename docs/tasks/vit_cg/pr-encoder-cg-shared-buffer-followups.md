**Title:** `[Bugfix][Multimodal] Fix Step3-VL encoder output aliasing and guard shared CUDA graph capture buffers`

## Overview

Follow-up to #65084, which shares encoder CUDA graph capture buffers across token budgets. Sharing introduced two issues, and this PR fixes both:

1. Step3-VL returns views of the graph output, which a later replay overwrites.
2. The model protocol was tightened without saying so: buffers written only at capture time can be overwritten by other budgets.

## Claims

- **Step3-VL embeddings no longer get overwritten.** A no-patch image's embedding was a view of the `global` graph output, even with `clone=True`. A later replay in the same `execute()` overwrote it. Before #65084 this needed two batches on the *same* budget; with sharing, any budget on the path does it. Fix: honor `clone`.
- **Capture-time-only buffers are guarded.**
  - Before: 0-dim buffers were shared wholesale, so after capture every budget's graph saw the smallest budget's value. A shared buffer that replay did not refill silently kept another budget's contents.
  - After: 0-dim buffers stay per budget. These scalars cost no memory, so nothing is lost. Replay raises if it omits a buffer that smaller budgets view.
  - The constraints are documented: replay must refill shared buffers, `padding_logics` must write the whole destination, and `postprocess_encoder_output(clone=True)` must not alias its outputs.

## Validation

**Unit tests** (`tests/v1/cudagraph/test_encoder_cudagraph.py`, `tests/v1/worker/test_encoder_runner.py`): 118 passed.

| Test | Covers | Fails without the fix |
|---|---|---|
| `test_step3_vl_postprocess_clones_global_only_item` (new) | Step3-VL honors `clone=True` | ✔ |
| `test_replay_without_shared_buffer_raises` (new) | Replay guard | ✔ |
| `test_scalar_buffers_keep_own_value` (was `test_scalar_buffers_are_shared`) | Each budget's graph keeps its own scalar | ✔ |

**End-to-end on MI355X** (Model Runner V2, default budgets). A 3-image prompt with `encoder_cudagraph_max_vision_items_per_batch=2` splits one `execute()` into several batches. Each returned embedding is compared with the same image encoded alone (relative L2; bf16 noise is ≤ 0.05).

| Model | Corrupted embeddings, before → after | Capture buffers, before → after |
|---|---|---|
| Step3-VL-10B | **5/11 → 0/11** (max rel. error 0.62 → 0.04) | 430.1 → 430.1 MiB |
| Qwen3-VL-2B-Instruct | 0/11 → 0/11 | 488.0 → 488.0 MiB |
| Gemma4-E2B-it | 0/11 → 0/11 | 344.1 → 344.1 MiB |
| MiniMax-M3 (TP2, `mm_encoder_tp_mode=data`) | — → 0/11; guard not triggered | — → 554.0 MiB |

Step3-VL on main hits both overwrite forms:
- two batches on global budget 338;
- budget 338 followed by 169, which overwrites only because the buffer is shared.

## Details

- **Why main is safe today.** The only capture-time-only key in-tree is the CPU `max_seqlen`. ViT attention reads it with `.item()` at capture, so later overwrites never reach a graph. The guard and the docs keep future or out-of-tree models from silently depending on this.
- **Why the fix is in Step3-VL, not the manager.** Every other `postprocess_encoder_output` already returns fresh tensors. Cloning in the manager would add a full copy for all of them.

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
