## Purpose

This PR implements the `SupportsEncoderCudaGraph` protocol for `DeepseekOCRForCausalLM`, enabling full CUDA graph capture of the vision encoder with a **dual-path graph** architecture. DeepSeek-OCR uses a two-tower ViT (SAM + CLIP) with a dynamic tiling mechanism — global images at 1024×1024 and optional local patches at 640×640.

Rather than capturing a single monolithic graph for both paths, this PR introduces a dual-path design (`enable_dual_path_graph=True`): two independent graph sets are captured — one for global images (constant 272 tokens each) and one for local patches (100 tokens each). During inference, the manager independently selects the smallest fitting budget per path, enabling partial graph fallback (one path hits while the other falls back to eager), and skipping local-path graphs entirely when no patches are present. This avoids wasted compute on zero-padded patch buffers for untiled images and avoids graphs that would otherwise be invalidated by variable `crop_shape` per image.

Related paper: [DeepSeek-OCR technical report](https://arxiv.org/pdf/2510.18234).

### Key Changes

- **`DeepseekOCRForCausalLM`** (`vllm/model_executor/models/deepseek_ocr.py`): Implements the full `SupportsEncoderCudaGraph` protocol — `get_encoder_cudagraph_config`, `get_encoder_cudagraph_budget_range`, `get_encoder_cudagraph_item_specs`, `select_encoder_cudagraph_items`, `prepare_encoder_cudagraph_capture_inputs`, `prepare_encoder_cudagraph_replay_buffers`, `encoder_cudagraph_forward`, `encoder_eager_forward`, and `postprocess_encoder_output`. All forward/capture methods accept a `path` parameter (`"global"` or `"local"`) for dual-path dispatch.
- **`_batched_encoder_forward_global_path` / `_batched_encoder_forward_local_path`**: New batched encoder forward methods. The global path bakes newline token insertion directly into the captured computation, outputting `[B * 272, n_embed]`. The local path outputs raw projected features `[P * 100, n_embed]` without newlines (added later in `_assemble_patch_grid`).
- **`_assemble_patch_grid` helper**: Extracted from `_encode_local_features` so local patch grid assembly (tile arrangement + per-row newline insertion) can be reused in both eager and CUDA-graph post-processing paths.
- **`postprocess_encoder_output`**: Updated to accept optional `local_output: torch.Tensor | None`. Splits the flat global output into per-image portions, splits local output into per-image patch groups, assembles patch grids with `_assemble_patch_grid`, and merges as `local_tiled + global + view_separator`.
- **`IMAGE_SIZE` import**: Switches from a module-local constant to importing `IMAGE_SIZE` from `vllm.transformers_utils.processors.deepseek_ocr`, removing the duplication.
- **`SupportsEncoderCudaGraph` protocol** (`vllm/model_executor/models/interfaces.py`): Adds `path: str | None = None` parameter to `prepare_encoder_cudagraph_capture_inputs`, `prepare_encoder_cudagraph_replay_buffers`, `encoder_cudagraph_forward`, and `encoder_eager_forward`. Adds `local_output: torch.Tensor | None = None` to `postprocess_encoder_output`.
- **`EncoderCudaGraphConfig`** (`vllm/v1/worker/encoder_cudagraph_defs.py`): Adds `enable_dual_path_graph: bool`, `global_token_per_image: int`, and `local_token_per_patch: int` fields for dual-path token budget calculation.
- **`EncoderItemSpec`** (`vllm/v1/worker/encoder_cudagraph_defs.py`): Adds `global_output_tokens: int` and `local_output_tokens: int` fields for dual-path packing decisions.
- **`EncoderCudaGraphManager`** (`vllm/v1/worker/encoder_cudagraph.py`): Major dual-path support — maintains separate `global_budget_graphs` and `local_budget_graphs` dicts, captures independent graph sets per path, and implements `_execute_local_dual_path` for dual-path greedy packing with partial eager fallback. The original single-path logic is preserved in `_execute_local_single_path`.

## Test Plan

E2E functional test:

```bash
python examples/generate/multimodal/vision_language_offline.py \
    -m deepseek_ocr --modality "image" --enable-vit-cuda-graph
```

Benchmark:

```bash
vllm bench mm-processor \
    --model /shared/models/modelscope/models/deepseek-ai/DeepSeek-OCR \
    --max-model-len 8192 \
    --dataset-name random-mm \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --random-mm-bucket-config '{(896, 896, 1): 1.0}' \
    --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --num-prompts 100 \
    --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

## Test Result

E2E functional test results (eager vs CUDA graph — outputs are consistent):

```
--------------------------------------------------
The image captures the majestic Tokyo Skytree, the tallest tower in Japan...
--------------------------------------------------
```

Benchmark results: dual-path CUDA graph vs eager baseline.

| Input Size | Tiled | Mean Latency | P99 Latency |
| :--------: | :---: | :----------: | :---------: |
| (224, 224) | No | -13.93% (20.74ms → 17.85ms) | -15.49% (22.59ms → 19.09ms) |
| (448, 448) | No | -20.96% (23.24ms → 18.37ms) | -27.10% (29.11ms → 21.22ms) |
| (896, 896) | Yes | -4.59% (42.08ms → 40.15ms) | -8.12% (44.07ms → 40.49ms) |
| (1024, 1024) | Yes | -6.56% (42.96ms → 40.14ms) | -10.31% (45.00ms → 40.36ms) |

> **Note:** When `image_width <= 640 and image_height <= 640`, the mm inputs will only contain a global image, without generating local patches.

## Documentation Update

- Updated `docs/design/cuda_graphs_multimodal.md` with DeepSeek-OCR in the support matrix.
