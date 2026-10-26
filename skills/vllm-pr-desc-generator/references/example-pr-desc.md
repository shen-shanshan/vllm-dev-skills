# Example: Well-Written vLLM PR Description

Below is an example of a high-quality vLLM PR title and description generated from code changes. Use this as a style reference.

---

**Title:** `[Feature][Model Loader] Add InstantTensor weight loader for fast model loading`

## Overview

Add an `InstantTensor` weight loader (`--load-format instanttensor`) that skips safetensors deserialization and fully utilizes high-speed storage bandwidth (e.g., 400 Gbps networked storage). Related RFC: #36091.

## Claims

- New `--load-format instanttensor` option; works with TP and expert parallelism.
- ~3.7x faster warm model loading for Qwen3-30B-A3B (45s → 12s) with no change in outputs.
- Transparent fallback to safetensors when the `.it` cache is missing.

## Validation

Load and serve with different parallelism settings:

```bash
# Single GPU
vllm serve Qwen/Qwen3-30B-A3B --load-format instanttensor

# Multi-GPU with expert parallelism
vllm serve deepseek-ai/DeepSeek-R1 --load-format instanttensor \
    --tensor-parallel-size 8 --enable-expert-parallel
```

Verified that:

1. The model loads successfully and serves requests.
2. Outputs match the `--load-format auto` baseline for the same prompts (greedy decoding).
3. The second load is significantly faster (cache hit).

Load time on 1x H100, 400 Gbps networked storage (mean of 3 runs):

| Model | Format | Load Time | Throughput |
|-------|--------|-----------|------------|
| Qwen3-30B-A3B | safetensors (baseline) | 45s | 2.1 GB/s |
| Qwen3-30B-A3B | instanttensor (cold) | 48s | 2.0 GB/s |
| Qwen3-30B-A3B | instanttensor (warm) | 12s | 8.3 GB/s |

## Details

The bottleneck in loading large checkpoints from fast storage is safetensors deserialization, not I/O. On first load, `InstantTensorLoader` converts the safetensors weights into a memory-mapped `.it` format laid out for direct GPU transfer; subsequent loads use zero-copy mmap.

Key changes:

- **`InstantTensorLoader`** (`vllm/model_executor/model_loader/instant_tensor.py`): implements `BaseModelLoader`; handles conversion and mmap loading.
- **`LoadFormat` / arg parser**: registers `instanttensor` and the `--load-format instanttensor` flag.
- **Shard-aware loading**: integrates with `tensor-parallel-size` and `enable-expert-parallel` so each rank reads only its own shard.
- **Docs**: documents the new option in `docs/source/serving/engine_args.md`.

Tradeoffs and limitations:

- The cold load is slightly slower (~7%) because the `.it` cache is generated during the first load.
- The `.it` cache uses extra disk space roughly equal to the checkpoint size.

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
