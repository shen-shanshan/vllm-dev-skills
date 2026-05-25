## Purpose

The previous auto-inference logic for `encoder_cudagraph_max_frames_per_batch` in the ViT encoder CUDA graph path used a hardcoded formula of `max_batch_size * 2`, which was both inaccurate and noted as a TODO. For models like Qwen3-VL, the actual max frames per video can be far greater than 2, causing the pre-allocated `cu_seqlens` buffer to be either too small (risking OOM or incorrect behavior) or unnecessarily large.

This PR replaces the hardcoded multiplier with a model-aware auto-inference path. The default value of `encoder_cudagraph_max_frames_per_batch` is changed from `0` to `None`, cleanly separating "auto-infer" from "user explicitly disabled video" (value `0`). When auto-inference is triggered, the manager calls the new `get_max_frames_per_video()` protocol method on the model, which queries `processing_info` to compute the actual per-video frame budget — making the buffer size accurate and model-specific. Image-only mode (video count per prompt limited to 0) is also correctly handled by forcing `max_frames_per_batch = 0`.

### Key Changes

- **`CompilationConfig.encoder_cudagraph_max_frames_per_batch`** (`vllm/config/compilation.py`): Type changed from `int = 0` to `int | None = None`. Validation guard updated to skip the non-negative check when the value is `None`.
- **`SupportsEncoderCudaGraph` protocol** (`vllm/model_executor/models/interfaces.py`): Added `get_max_frames_per_video() -> int` as a new required protocol method for models that opt into encoder CUDA graphs.
- **`Qwen3VLEncoder.get_max_frames_per_video()`** (`vllm/model_executor/models/qwen3_vl.py`): Implements the new protocol method by calling `processing_info.get_num_frames_with_most_features()` with the model's `max_model_len` and per-prompt video limit, returning an accurate model-specific max frame count.
- **`MultiModalRegistry.get_processing_info()`** (`vllm/multimodal/registry.py`): Exposes a thin public wrapper around `_create_processing_info()` (without caching, to be safe across multi-model scenarios) so models can access their processing info without internal API access.
- **`EncoderCudaGraphManager.__init__()`** (`vllm/v1/worker/encoder_cudagraph.py`): Updated frame budget selection logic — checks video-disabled condition first, then respects explicit user value (`is not None`), then falls back to `max_batch_size * model.get_max_frames_per_video()`.
- **`docs/design/cuda_graphs_multimodal.md`**: Updated field documentation for `encoder_cudagraph_max_frames_per_batch` and added `get_max_frames_per_video()` to the protocol method list.

## Test Plan

```bash
# Test-1: Image-only mode (video disabled)
# Set compilation_config with encoder_cudagraph_max_frames_per_batch=128
# but run inference on image modality only — video count per prompt = 0
python examples/offline_inference/vision_language.py -m qwen3_vl --modality "image"
# Expected: max_frames_per_batch=0 (image-only fallback)

# Test-2: Video mode with user-specified max_frames_per_batch=128
python examples/offline_inference/vision_language.py -m qwen3_vl --modality "video"
# Expected: max_frames_per_batch=128 (user value respected)

# Test-3: Video mode with auto-inferred max_frames_per_batch (None)
python examples/offline_inference/vision_language.py -m qwen3_vl --modality "video"
# Expected: max_frames_per_batch=<model-inferred value, e.g. 64>
```

## Test Result

```bash
# Test-1: Video modality disabled, max_frames_per_batch forced to 0
EncoderCudaGraphManager initialized with ..., max_frames_per_batch=0, ...

# Test-2: User explicitly set encoder_cudagraph_max_frames_per_batch=128
EncoderCudaGraphManager initialized with ..., max_frames_per_batch=128, ...

# Test-3: Auto-infer via get_max_frames_per_video() from processing_info
EncoderCudaGraphManager initialized with ..., max_frames_per_batch=64, ...
```

## (Optional) Documentation Update

- Updated `docs/design/cuda_graphs_multimodal.md` to document the new `get_max_frames_per_video()` protocol method and the updated semantics of `encoder_cudagraph_max_frames_per_batch` (default `None` = auto-infer; `0` = image-only mode; positive = user override).
