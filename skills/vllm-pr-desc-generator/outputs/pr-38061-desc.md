## Purpose

Following #35963 (ViT full CUDA graph support for image inference), this PR extends the encoder CUDA graph framework to support **video inference for Qwen3-VL**. Previously, the CUDA graph capture/replay path only handled image inputs (`pixel_values` + `image_grid_thw`). Video inputs use different keys (`pixel_values_videos` + `video_grid_thw`) and require larger `cu_seqlens` buffers because each video item contributes multiple frames (T attention sequences). This PR generalizes the protocol and manager to handle both modalities through a single shared graph manager.

**Note:** Video CUDA graphs are automatically disabled when EVS (Efficient Video Sampling) pruning is enabled, since EVS makes the token count data-dependent and incompatible with CUDA graph capture.

Related: #35963

### Key Changes

- **`EncoderCudaGraphConfig`** (`vllm/v1/worker/encoder_cudagraph_defs.py`): Replace single `input_key` field with `input_key_by_modality` dict (e.g., `{"image": "pixel_values", "video": "pixel_values_videos"}`) to support per-modality input tensor routing.
- **`SupportsEncoderCudaGraph` protocol** (`vllm/model_executor/models/interfaces.py`): Add `get_input_modality(mm_kwargs)` method to determine whether inputs are image or video. Add `max_frames_per_batch` parameter to `prepare_encoder_cudagraph_capture_inputs()` and `prepare_encoder_cudagraph_replay_buffers()`.
- **`Qwen3VLForConditionalGeneration`** (`vllm/model_executor/models/qwen3_vl.py`):
  - Implement `get_input_modality()` to route based on `mm_kwargs` keys.
  - Add `_get_pixel_values_by_modality()` and `_get_grid_thw_by_modality()` helpers to abstract modality-specific key access across all protocol methods.
  - Update `prepare_encoder_cudagraph_capture_inputs()` to build video-format grid configs (T>1 per item) when `max_frames_per_batch` exceeds `max_batch_size`, sizing `cu_seqlens` buffer for video replays.
  - Add replay buffer caching (`_replay_buffer_cache`) keyed by `(modality, grid_thw)` to avoid redundant CPU-side NumPy computation for repeated grid shapes.
  - Update `prepare_encoder_metadata()` to accept `max_frames_per_batch` for `cu_seqlens` padding, allowing video frames to exceed `max_batch_size`.
- **`EncoderCudaGraphManager`** (`vllm/v1/worker/encoder_cudagraph.py`):
  - Add `max_frames_per_batch` field to `BudgetGraphMetadata` and manager initialization.
  - Rename `encoder_cudagraph_max_images_per_batch` → `encoder_cudagraph_max_mm_items_per_batch` for generality.
  - Route `input_key` lookup through `get_input_modality()` during replay instead of using a fixed key.
- **`CompilationConfig`** (`vllm/config/compilation.py`): Add `encoder_cudagraph_max_frames_per_batch` config option (0 = auto-infer). Rename `encoder_cudagraph_max_images_per_batch` → `encoder_cudagraph_max_mm_items_per_batch`.
- **Tests** (`tests/v1/cudagraph/test_encoder_cudagraph.py`): Add `SimpleMockViTVideoModel` with dual-modality support, `TestGetInputModality` (no GPU), and `TestEncoderCudaGraphVideoReplay` (GPU) covering video capture/replay, fallback, counters, chunking, and mixed image+video through a shared manager. (+316 lines)

## Test Plan

**Unit tests:**

```bash
pytest tests/v1/cudagraph/test_encoder_cudagraph.py -v
```

**Functional test (video inference):**

```bash
# Add compilation_config to EngineArgs in run_qwen3_vl():
# compilation_config={
#     "cudagraph_mm_encoder": true,
#     "encoder_cudagraph_token_budgets": [128, 256, 512, 1024, 1536, 2048],
#     "encoder_cudagraph_max_images_per_batch": 8,
#     "encoder_cudagraph_max_frames_per_batch": 64,
# }
python examples/offline_inference/vision_language.py -m qwen3_vl --modality "video"
```

**Benchmark (single GPU, video):**

```bash
vllm bench mm-processor \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --max-model-len 16384 \
    --dataset-name random-mm \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --random-mm-bucket-config '{(224, 224, 8): 1.0}' \
    --random-mm-limit-mm-per-prompt '{"image": 0, "video": 1}' \
    --num-prompts 100 \
    --seed 42 \
    --mm-encoder-attn-backend FLASH_ATTN \
    --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [128, 256, 512, 1024, 1536, 2048], "encoder_cudagraph_max_mm_items_per_batch": 4, "encoder_cudagraph_max_frames_per_batch": 32}'
```

## Test Result

**Unit tests:** 36 passed (10.04s)

**Functional test:** Video inference produces correct descriptive output for test videos, matching eager-mode baseline.

## (Optional) Documentation Update

- "Vision Encoder (ViT) CUDA Graphs" docs update pending to cover the new video support and `encoder_cudagraph_max_frames_per_batch` configuration option.
