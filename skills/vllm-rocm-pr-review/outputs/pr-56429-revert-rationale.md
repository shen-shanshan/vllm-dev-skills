# Revert "[Rocm][Kimi-k3] Add pipeline_parallel support for the kimik3 DCP mode"

## Purpose

This PR reverts #53664 ("[Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode"). That PR added a fallback in `MLAAttentionLayer.build()` that derives `dcp_local_seq_lens` from `seq_lens` whenever `CommonAttentionMetadata` does not carry it, together with an assert and two test-registry entries. The fallback is dead code: in vLLM's in-tree paths, `dcp_local_seq_lens` is always populated before the MLA decode path runs whenever DCP is enabled.

Evidence:

- The monolithic V1 model runner populates `CommonAttentionMetadata.dcp_local_seq_lens` unconditionally when `decode_context_parallel_size > 1` (`vllm/v1/worker/gpu_model_runner.py:2456-2466`), and this has been in place since before #53664 was authored.
- The modular model runner (MRV2) does the same via `maybe_prepare_dcp_local_seq_lens()` right before attention metadata is built, after PCP batch partitioning (`vllm/v1/worker/gpu/model_runner.py:1728`, helper in `vllm/v1/worker/gpu/cp_utils.py`). The helper returns `None` only when `dcp_size == 1`, in which case the MLA layer's `dcp_world_size` is also 1 and the branch is never taken, so the fallback is unreachable.
- Both runners always set `seq_lens` on `CommonAttentionMetadata`, so the assert added by #53664 is likewise unreachable.

Keeping the fallback would also mask future regressions in runner-side population: if a path ever stopped populating `dcp_local_seq_lens`, the layer would silently derive it instead of failing loudly. Removing it restores a single source of truth for DCP-local sequence lengths (the model runner). The two test-registry entries added by #53664 are removed along with the revert.

## Test Plan

- `pytest tests/models/test_registry.py`
- Manual ROCm smoke test: Kimi-K3 DCP decode with PP (e.g. PP2×TP4×DCP4) to confirm decode metadata is unaffected.

## Test Result

TODO: paste results
