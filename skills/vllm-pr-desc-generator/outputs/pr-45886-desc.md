## Purpose

In dual-path vision encoder CUDA Graph mode (e.g., DeepSeek-OCR), the global and local token budgets were previously auto-generated as power-of-2 multiples of the model's `global_token_per_image` and `local_token_per_patch` values, respectively. Users had no way to independently control each path's budget levels — the only knob was the shared `encoder_cudagraph_token_budgets`, which set the overall `max_budget` cap.

In practice, the global path produces a fixed number of tokens per image (e.g., 272 for DeepSeek-OCR), making multiple budget levels wasteful. Users need to keep only one global budget while keeping several local budgets to cover varying patch counts. Conversely, some deployments may want fine-grained global budgets but sparse local budgets.

This PR adds two new fields to `CompilationConfig` — `encoder_cudagraph_global_token_budgets` and `encoder_cudagraph_local_token_budgets` — that allow users to independently specify per-path token budgets. When non-empty, these override auto-generation for that path; when empty (default), behavior is unchanged.

### Key Changes

- **`vllm/config/compilation.py` — New config fields**: Added `encoder_cudagraph_global_token_budgets` and `encoder_cudagraph_local_token_budgets` (both `list[int]`, default `[]`) with positive-value validation in `__post_init__`.
- **`vllm/v1/worker/encoder_cudagraph.py` — Dual-path budget selection logic**: In `EncoderCudaGraphManager.__init__`, when `enable_dual_path_graph` is `True`, the manager now checks user-provided per-path budgets first. If non-empty, they are sorted and used directly; if empty, falls back to auto-generation via `_generate_budgets()`. The `0` budget auto-insertion for local budgets only applies in auto-generation mode. Also moved attribute initialization (`budget_graphs`, `graph_pool`, etc.) earlier in the constructor and consolidated redundant `mm_config` variable.
- **`docs/design/cuda_graphs_multimodal.md` — Documentation update**: Updated the dual-path budget generation section to describe the new config fields. Added a complete "Dual-path inference" usage guide with CLI and Python API examples. Updated the configuration reference table to list all five encoder CUDA Graph fields.
- **`tests/v1/cudagraph/test_encoder_cudagraph.py` — New `TestDualPathBudgetConfig` class**: Added 8 unit tests covering: user global budgets used directly, user local budgets used directly, both specified, auto-generation fallback when empty, no auto-insertion of `0` when local budgets are user-provided, mixed user/auto modes, and validation of negative/zero values. Also added `_MockDualPathModel` helper and extended `_MockVllmConfig`/`_make_mgr` to support dual-path parameters.
- **`tests/models/multimodal/generation/test_vit_cudagraph.py` — Integration test update**: Updated the DeepSeek-OCR test case to use the new per-path budget fields, un-skipped the test, and added `gpu_memory_utilization: 0.6` to avoid CI OOM.

## Test Plan

CI integration test:

```bash
pytest -sv tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_image[deepseek_ocr]
```

Unit tests:

```bash
pytest -sv tests/v1/cudagraph/test_encoder_cudagraph.py
```

## Test Result

CI (before vs. after):

```bash
# Before:
# Set {"encoder_cudagraph_token_budgets": [1152]}
# -> only supports auto-inference for global/local token budgets
EncoderCudaGraphManager dual-path mode: global_budgets=[272, 544, 1088, 1152], local_budgets=[0, 100, 200, 400, 800, 1152]

# After:
# Set {"encoder_cudagraph_global_token_budgets": [272], "encoder_cudagraph_local_token_budgets": [0, 100, 200, 400, 800]}
EncoderCudaGraphManager dual-path mode: global_budgets=[272], local_budgets=[0, 100, 200, 400, 800]
```

Unit tests:

```bash
--- 60 passed, 17 warnings in 15.75s ---
```

## (Optional) Documentation Update

- Updated `docs/design/cuda_graphs_multimodal.md` with the new dual-path budget configuration fields and a complete usage guide with CLI and Python examples.
