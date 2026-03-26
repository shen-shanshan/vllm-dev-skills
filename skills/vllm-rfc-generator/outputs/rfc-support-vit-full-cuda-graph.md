# [RFC]: Support ViT Full CUDA Graph

> **Labels**: RFC
> **Status**: Open for Feedback
> **Feedback Period**: At least 2 weeks

---

## Motivation

Multimodal large language models (e.g., Qwen3-VL, Qwen3.5, GLM-V, Kimi K2.5) rely on a Vision Transformer (ViT) encoder to process visual inputs before feeding them into the language model backbone. In production serving scenarios, the ViT forward pass involves launching a large number of small CUDA kernels — including patch embedding, layer normalization, multi-head self-attention, and MLP projections — each of which incurs non-trivial kernel launch overhead on the host side. As request rates grow and latency budgets shrink, this overhead becomes a significant bottleneck, especially for video inference workloads where multiple frames must be processed per request.

Currently, vLLM supports CUDA graph capture for the decoder (LLM) portion of the model, which has proven effective at reducing kernel launch costs and improving throughput. However, the ViT encoder is still executed eagerly, meaning every forward pass re-launches all kernels from scratch. This asymmetry leaves substantial performance on the table, particularly for vision-heavy workloads.

Extending full CUDA graph support to the ViT encoder would allow the entire encoder forward pass to be captured and replayed as a single graph, eliminating per-kernel launch overhead and enabling more consistent, low-latency inference for multimodal models. This is especially critical for video inference pipelines where the ViT is invoked repeatedly across many frames within a single request.

## Proposed Change

We propose adding full CUDA graph capture and replay support for the ViT encoder in vLLM's multimodal inference pipeline. The key changes include:

### 1. ViT CUDA Graph Manager

Introduce a dedicated CUDA graph manager for the ViT encoder, responsible for capturing, caching, and replaying encoder graphs. The manager should handle:

- **Graph capture**: Capture the full ViT forward pass (patch embedding through final projection) as a CUDA graph during warmup or on first invocation for each distinct input shape.
- **Shape bucketing**: Since image/video inputs may vary in resolution and frame count, the manager must support multiple captured graphs indexed by input tensor shapes, or use padded fixed-size buckets to reduce the number of distinct graphs.
- **Memory management**: Pre-allocate input/output buffers for graph replay, ensuring that replayed graphs write to stable memory addresses.

### 2. Image and Video Inference Support

The implementation must support both modalities:

- **Image inference**: Single-frame ViT encoding with CUDA graph replay. Input shapes are relatively predictable (e.g., fixed crop sizes), making graph caching straightforward.
- **Video inference**: Multi-frame ViT encoding where the number of frames varies per request. The graph manager must efficiently handle variable frame counts, either via dynamic batching of frames or by capturing graphs for common frame-count buckets.

### 3. Model Coverage

The initial implementation should support the following model families:

- **Qwen3-VL / Qwen3.5**: Primary target, as these models are widely deployed and have well-defined ViT architectures.
- **GLM-V**: ChatGLM's vision variant with its specific encoder structure.
- **Kimi K2.5**: Moonshot AI's multimodal model.

Each model may have different ViT configurations (patch size, hidden dim, number of layers), so the graph manager must be model-agnostic in its capture logic, relying on the model's forward method to define the graph.

### 4. Invariant Checks and Budget Management

Add auto-inferred budget and max batch size validation in the ViT CUDA graph manager to prevent OOM errors and ensure captured graphs remain valid under varying load conditions.

### Key Design Decisions

- **Eager-fallback mode**: When an input shape has no matching captured graph, fall back to eager execution rather than failing. This ensures correctness while progressively caching more shapes.
- **Warmup strategy**: Capture graphs for the most common input shapes during model loading warmup, trading startup time for runtime performance.
- **Unified manager interface**: Reuse the existing CUDA graph infrastructure (capture/replay APIs) so that encoder graphs and decoder graphs are managed consistently.

## Related PRs

**Implementation:**
- [#35963](https://github.com/vllm-project/vllm/pull/35963): [Feature] ViT Full CUDA Graph — core implementation of full CUDA graph capture for the ViT encoder
- [#38061](https://github.com/vllm-project/vllm/pull/38061): [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference — extends ViT CUDA graph support to video inference for Qwen3-VL

**Bug Fixes / Improvements:**
- [#38040](https://github.com/vllm-project/vllm/pull/38040): [Fix] Invariant Check for Auto-Inferred Budgets/Max Batch Size in ViT CUDA Graph Manager — adds validation for auto-inferred budgets and max batch size

**Documentation:**
- [#37914](https://github.com/vllm-project/vllm/pull/37914): [Docs] Add Encoder (ViT) CUDA Graphs section to CUDA Graphs design doc — documents the encoder CUDA graph design

## Feedback Period

We plan to keep this RFC open for at least 2 weeks. Please share your thoughts, concerns, or alternative approaches in the comments.

## CC List

<!-- Add relevant maintainers and stakeholders here -->

## Any Other Things

- **Performance benchmarks**: We expect significant latency reduction for multimodal inference, especially for video workloads with many frames. Concrete benchmark numbers will be shared as the implementation matures.
- **Compatibility**: This change should be backward-compatible — models without ViT CUDA graph support will continue to use eager execution as before.
- **Future work**: Extend CUDA graph support to other encoder architectures (e.g., audio encoders for speech models) using the same manager infrastructure.

---

> Please take a look at previous [RFCs](https://github.com/vllm-project/vllm/issues?q=label%3ARFC+sort%3Aupdated-desc) for reference.
