# Example: Well-Written vLLM PR Description

Below is an example of a high-quality vLLM PR description generated from code changes. Use this as a style reference.

---

## Purpose

Speed up model loading and fully utilize the bandwidth of high-speed storage (e.g., 400 Gbps networked storage).

This PR adds a new weight loader `InstantTensor` that bypasses the safetensors deserialization overhead by pre-converting model weights into a memory-mapped format optimized for direct GPU transfer. Related RFC: #36091.

### Key Changes

- **New `InstantTensorLoader` class** (`vllm/model_executor/model_loader/instant_tensor.py`): Implements the `BaseModelLoader` interface. On first load, converts safetensors to `.it` format; subsequent loads use zero-copy mmap.
- **`--load-format instanttensor` CLI flag**: Registers the new loader in the `LoadFormat` enum and argument parser.
- **Shard-aware weight distribution**: Integrates with `tensor-parallel-size` and `enable-expert-parallel` to load only the relevant shard per rank, avoiding redundant I/O.
- **Fallback behavior**: If `.it` cache is missing, transparently falls back to safetensors loading and generates the cache in background.

## Test Plan

Load any model with any parallelism setting:

```bash
# Single GPU
vllm serve Qwen/Qwen3-30B-A3B --load-format instanttensor

# Multi-GPU with expert parallelism
vllm serve deepseek-ai/DeepSeek-R1 --load-format instanttensor \
    --tensor-parallel-size 8 --enable-expert-parallel
```

Verify:
1. Model loads successfully and serves requests
2. Second load is significantly faster (cache hit)
3. Output matches baseline `--load-format auto` for the same prompts

## Test Result

| Model | Format | Load Time | Throughput |
|-------|--------|-----------|------------|
| Qwen3-30B-A3B | safetensors | 45s | 2.1 GB/s |
| Qwen3-30B-A3B | instanttensor (cold) | 48s | 2.0 GB/s |
| Qwen3-30B-A3B | instanttensor (warm) | 12s | 8.3 GB/s |

## (Optional) Documentation Update

- Updated `docs/source/serving/engine_args.md` to document the new `--load-format instanttensor` option.
