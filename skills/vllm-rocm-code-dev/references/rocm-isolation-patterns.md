# ROCm Isolation Patterns in vLLM

How to put a ROCm-only change into vLLM without affecting CUDA/XPU/CPU/TPU. Pick the first pattern that fits; move down the list only if the earlier ones can't express the change. Verify names against the current tree before using them (they drift).

## Contents

1. ROCm-only files (no shared-code change)
2. Platform hook: base-class default + ROCm override
3. `CustomOp.forward_hip`
4. aiter ops and `VLLM_ROCM_*` env gating
5. Config-time defaults in `RocmPlatform.check_and_update_config`
6. Inline platform guard (last resort)
7. Comment style
8. Impact checklist

## 1. ROCm-only files

Change these first; they are only reached on ROCm:

| Area | Location |
|---|---|
| Platform behavior, arch helpers, backend selection | `vllm/platforms/rocm.py` (`on_gfx942()`, `on_gfx950()`, `on_mi3xx()`, `get_cdna_version()`, `RocmPlatform.get_attn_backend_cls`, ...) |
| aiter op wrappers and enable switches | `vllm/_aiter_ops.py` (`rocm_aiter_ops.is_*_enabled()`) |
| Attention backends | `vllm/v1/attention/backends/rocm_*.py`, `mla/rocm_*`, `mla/aiter_*` |
| MoE / quant kernels | `rocm_aiter_*` files under `vllm/model_executor/layers/fused_moe/` and `.../quantization/` |
| Env vars | `VLLM_ROCM_*` entries in `vllm/envs.py` (adding a ROCm env var is ROCm-only) |

## 2. Platform hook: base-class default + ROCm override

Use when shared code (model runner, layer, scheduler, config) needs ROCm-specific behavior.

1. Add a classmethod to `Platform` in `vllm/platforms/interface.py` whose default reproduces **exactly** the current behavior.
2. Override it in `RocmPlatform` (`vllm/platforms/rocm.py`).
3. Call `current_platform.<hook>(...)` from the shared code in place of the hard-coded behavior.

Real example — `enable_multi_stream_overlap`:

```python
# vllm/platforms/interface.py
@classmethod
def enable_multi_stream_overlap(cls, aux_stream_list, attn_metadata) -> bool:
    """Whether the current platform should enable multi-stream overlap
    for the given aux streams and attention metadata."""
    return False

# vllm/platforms/rocm.py
@classmethod
def enable_multi_stream_overlap(cls, aux_stream_list, attn_metadata) -> bool:
    # HIP event waits can hang in eager breaks, so only overlap where
    # fork/join become static graph edges.
    return aux_stream_list is not None and (
        torch.cuda.is_current_stream_capturing()
        or not isinstance(attn_metadata, dict)
    )
```

Rules:

- The default must be behavior-preserving for every other platform (including `CudaPlatform`, which may already override nearby hooks — check it doesn't need the new hook).
- Keep the hook narrow (one decision or one value), named after *what* it decides, not "rocm_xxx".
- Short docstring on the base method; the ROCm override carries the "why on ROCm" comment.

## 3. `CustomOp.forward_hip`

`CustomOp.dispatch_forward` routes to `forward_hip` on ROCm; its default calls `forward_cuda`. For a layer that is a `CustomOp` (norms, activations, rotary embedding, ...), override `forward_hip` with the ROCm kernel and leave `forward_cuda` untouched:

```python
def forward_hip(self, x):
    # aiter's fused kernel avoids an extra HBM round trip on MI3xx.
    if rocm_aiter_ops.is_xxx_enabled():
        return rocm_aiter_ops.xxx(x)
    return self.forward_cuda(x)
```

## 4. aiter ops and env gating

- Add new aiter wrappers to `rocm_aiter_ops` in `vllm/_aiter_ops.py`, following the existing register/lazy-import pattern, with an `is_<feature>_enabled()` gate that combines `VLLM_ROCM_USE_AITER` and a feature-specific env var.
- Import aiter lazily (inside functions or behind `rocm_aiter_ops`) so non-ROCm imports never touch it.
- A new feature env var defaults to the safe/old behavior unless the user asked to turn it on by default.

## 5. Config-time defaults

For "turn on X by default on ROCm" changes, rewrite the config once in `RocmPlatform.check_and_update_config` through a small helper (example: `_maybe_default_dsv4_mxfp4_indexer`):

- only rewrite `"auto"` values — an explicit user choice always wins;
- check all prerequisites (arch, aiter enabled, model type, parallel config) and return silently if any fails, so previously working configs never start failing;
- log once with `logger.info_once`, telling the user how to opt out.

## 6. Inline platform guard (last resort)

`if current_platform.is_rocm():` inside shared code is acceptable only for a tiny, local divergence where a hook would be over-engineered. Keep the CUDA branch textually unchanged, and put the ROCm branch first-class (not interleaved with CUDA logic). If you need two or more such guards for the same feature, switch to a hook (pattern 2).

## 7. Comment style

- 1–2 lines, explain *why* ROCm differs: hardware (wave64, LDS size, MFMA layout, FNUZ FP8), kernel availability/perf in aiter, numerics, HIP runtime behavior.
- Good: `# FNUZ FP8 on gfx942 has a different max, so rescale before the aiter GEMM.`
- Bad: multi-paragraph background, restating the code, mentioning other inference engines by name (never "ATOM").

## 8. Impact checklist

Before finishing, for each non-ROCm file in the diff answer:

- Does the default/CUDA path call the same functions with the same arguments as before?
- Are new parameters optional with defaults that preserve old behavior?
- Is any new import ROCm-only and lazy?
- Would `git diff` on CUDA-only code be a no-op if the ROCm branch were deleted?
