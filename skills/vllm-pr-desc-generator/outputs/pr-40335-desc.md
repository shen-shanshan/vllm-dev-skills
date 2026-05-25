# [MM][Misc] Support image+video mixed inputs (per prompt) for VLM examples

## Purpose

The VLM offline inference example (`examples/offline_inference/vision_language.py`) previously only accepted a single modality per run — either `image` or `video`. Users who wanted to test vision-language models on prompts that combine both an image and a video in the same context had no way to do so through the example script.

This PR adds a new `--modality "image+video"` option to the example, enabling per-prompt mixed image+video inputs. When this modality is selected, `limit_mm_per_prompt` is set to `{"image": 1, "video": 1}` and the appropriate combined placeholder token string is constructed for each model's chat template. The change covers ~20 VLM model runner functions and also fixes several pre-existing placeholder wrapping bugs (e.g., Qwen-series models had placeholder tokens double-wrapped inside vision boundary markers).

### Key Changes

- **`mm_limit` pattern** (all updated model runners): Replaces the hard-coded `{modality: 1}` with `{"image": 1, "video": 1} if modality == "image+video" else {modality: 1}` so `EngineArgs.limit_mm_per_prompt` correctly allows one image and one video per prompt.
- **`elif modality == "image+video":` placeholder branches** (all updated model runners): Concatenates the model-specific image placeholder and video placeholder tokens into a single prompt prefix string.
- **`run_hyperclovax_seed_vision`**: Adds a message-content-list branch that inserts both `{"type": "image", ...}` and `{"type": "video"}` content blocks; also widens `max_model_len` to `16384` for `image+video` (matching video-only behavior).
- **`run_minicpmv_base`**: Replaces the `modality_placeholder` dict lookup with an explicit `content_prefix` variable that concatenates image and video template strings for `image+video`.
- **`run_llava_onevision`**: Adds `image+video` prompt template branch before the `EngineArgs` construction block.
- **Placeholder correctness fixes** (EXAONE-4.5, Keye-VL/1.5, Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni, Qwen3-VL, Qwen3-VL-MoE): Moves vision boundary markers (`<|vision_start|>...<|vision_end|>`, `<|vision_bos|>...<|vision_eos|>`) from the outer prompt template string into the per-modality placeholder variable, so all three branches (image / video / image+video) are consistent.

**Models with added `image+video` support:**
ERNIE-4.5-VL, EXAONE-4.5, GLM-4.1V, GLM-4.5V, GLM-4.5V-FP8, GLM-OCR, HyperCLOVAX-SEED-Vision, Intern-S1, Intern-S1-Pro, InternVL3, Keye-VL, Keye-VL-1.5, LLaVA-OneVision, MiniCPM-V series, Molmo2, openPangu-VL, Ovis2.5, Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni, Qwen3-VL, Qwen3-VL-MoE, Tarsier2.

## Test Plan

Run the example with any supported model and the new mixed modality flag:

```bash
# Qwen3-VL (recommended quick test)
python examples/offline_inference/vision_language.py \
    -m qwen3_vl --modality "image+video"

# Qwen2.5-VL
python examples/offline_inference/vision_language.py \
    -m qwen2_5_vl --modality "image+video"

# GLM-4.1V
python examples/offline_inference/vision_language.py \
    -m glm4_1v --modality "image+video"

# MiniCPM-o-2.6
python examples/offline_inference/vision_language.py \
    -m minicpmo --modality "image+video"
```

Verify:
1. Script runs to completion without assertion errors or `ValueError: Unsupported modality`.
2. Each prompt receives a response that references both the image content and the video content.
3. Existing `--modality image` and `--modality video` modes are unaffected for all updated models.

## Test Result

```
--------------------------------------------------
The image shows a baby sitting on a bed, wearing glasses, and reading a book...
In the video, the baby continues to read the book, occasionally looking up and smiling...
--------------------------------------------------
The image shows a baby wearing glasses, sitting on a bed and reading a book...
In the video, the baby continues to read the book, turning the pages and occasionally looking up...
--------------------------------------------------
The image shows a baby girl sitting on a bed, wearing glasses and reading a book...
In the video, the baby girl continues to read the book, turning the pages and occasionally looking up...
--------------------------------------------------
```

Model successfully processes the combined image+video prompt and produces a response describing both modalities in the same output.
