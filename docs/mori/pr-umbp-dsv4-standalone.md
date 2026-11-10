# [ROCm][UMBP] Support DeepSeek-V4-Pro with UMBP standalone mode (compressed KV + SWA replay)

## Overview

Lets DeepSeek-V4-Pro on MI355X offload its prefix KV to a node-local MORI-UMBP `umbp_standalone_server` (`UMBPStoreConnector`, standalone mode) and restore it later. Only the compressed KV goes to the pool. The sliding-window state is rebuilt by SWA bounded replay. RFC #57187; builds on the merged UMBP connector (`[UMBP][1/10]`–`[8/10]`) and on #64715 (DeepSeek-V4.1-Flash), whose TP dedup and `kv_connector_may_load_foreign_kv()` it reuses.

## Claims

- DeepSeek-V4 on ROCm stores only the C4/C128 latents and the C4 indexer (256-token objects). Without this, the SWA cache and compressor states (4/8-token blocks) become tens of thousands of tiny objects per long request.
- The aiter sparse MLA and NaN-free extra-cache fast paths stay on with UMBP, as on DeepSeek-V4.1.
- DSpark MTP works with the replay.
- GSM8K accuracy is unchanged with UMBP (DP8 + DSpark), and AgentX output throughput is +14.4%.
- No change without a connector that asks for replay, and no change on CUDA.

## Validation

**Unit tests** (MI355X, ROCm; the UMBP suites mock the MORI client): all pass.

| Suite | Result |
|---|---|
| `tests/v1/kv_connector/unit/test_umbp_*.py` | 343 passed |
| `tests/v1/core/test_prefix_replay.py` | 16 passed |
| `tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py` | 18 passed |

New tests:
- `test_prefix_replay.py::test_state_rebuilt_by_replay_is_not_prefix_cached`: a `rebuilt_by_replay` group is neither cached nor a second replay window.
- `test_rocm_aiter_triton_sparse_mla.py::test_dsv4_swa_replay_only_with_a_connector_that_asks`
- `test_umbp_standalone.py::test_dspark_draft_rebuilt_by_replay_is_accepted`
- `test_umbp_standalone.py::test_eagle_draft_rebuilt_by_replay_is_rejected_with_a_hint`

**E2E accuracy: GSM8K with DSpark MTP.**
- Setup: DeepSeek-V4-Pro-0813 on 8x MI355X, MORI `v1.2.3.post1` with `BUILD_UMBP=ON`.
- Recipe: InferenceX `dsv4/vllm/mi355x-fp4-mtp/agentic.yaml` + `override_dep8_c128`. The UMBP run differs only by `--kv-transfer-config`.
- Eval: `lm_eval` 5-shot, all 1319 questions.

| Run | strict-match | flexible-extract | Mean acceptance length | Prompt tokens from UMBP |
|---|---|---|---|---|
| No connector | 0.9515 ± 0.0059 | 0.9507 ± 0.0060 | 4.48 | 0 |
| UMBP, pass 1 (stores) | 0.9538 ± 0.0058 | 0.9530 ± 0.0058 | 4.48 | 0.2% |
| UMBP, pass 2 (restores) | 0.9522 ± 0.0059 | 0.9515 ± 0.0059 | 4.50 | **72.6%** |

Pass 2 restores almost every prompt from the pool and replays the last window. Accuracy and DSpark acceptance are unchanged, so the replay rebuilds both the target's window and the draft's.

**E2E agentic performance with DSpark MTP.**
- Same recipe point, behind the recipe's `vllm-router`.
- AgentX trace `semianalysisai/cc-traces-weka-062126`, 128 sessions.
- Profiling window of 600 s, after the warmup.
- UMBP pool: 512 GiB.

| Metric | No connector | UMBP | Δ |
|---|---|---|---|
| Output throughput (tok/s) | 1,157 | 1,324 | **+14.4%** |
| TTFT mean / p90 (s) | 4.50 / 11.54 | 3.65 / 7.88 | −18.9% / −31.7% |
| TPOT mean (ms) | 78.8 | 70.2 | −10.9% |
| Prefill tokens computed per request | 13,937 | 11,203 | −19.6% |

About half of the prefill saving comes from UMBP restores. The other half comes from extra GPU prefix hits: once the SWA groups leave prefix caching, a hit no longer has to land on an SWA checkpoint.

<details>
<summary>Reproduce</summary>

```bash
# UMBP server
UMBP_STANDALONE_ADDRESS=unix:///tmp/umbp/standalone.grpc.sock \
UMBP_DRAM_CAPACITY=$((128 * 1024**3)) UMBP_SSD_ENABLED=0 umbp_standalone_server

# engine: recipe args (dep8_c128, DSpark), plus for the UMBP run:
vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 ... \
  --kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///tmp/umbp/standalone.grpc.sock"}}'

# accuracy (run twice against the UMBP server)
lm_eval --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Pro-0813,base_url=http://0.0.0.0:8333/v1/completions,num_concurrent=64,max_gen_toks=2048,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5

# unit tests
pytest tests/v1/kv_connector/unit/test_umbp_*.py tests/v1/core/test_prefix_replay.py \
  tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py
```
</details>

## Details

**SWA replay instead of storing every group**
- `KVConnectorBase_V1.requests_sliding_window_replay()`: new classmethod, default `False`.
  - `MultiConnector` returns `True` only if every child does.
  - `UMBPStoreConnector` returns `True`; opt out with `rebuild_sliding_window: false`.
- `deepseek_v4/compressor.py`: `dsv4_swa_replay_enabled()` is `True` only when all of these hold: ROCm, `swa_bounded_replay` on, no PCP, and the connector asks for replay. Startup logs whether it is enabled, or why not.
- When enabled:
  - The SWA cache uses `bounded_replay=True`.
  - The C4/C128 compressor states use the new `SlidingWindowMLASpec.rebuilt_by_replay`. This keeps them out of prefix caching without declaring a second replay window, since the scheduler allows only one. The replay of 128 or more tokens from a 256-aligned hit rewrites them before they are read.
- `deepseek_v4/amd/model.py` installs the existing replay-aware `DeepseekV41ModelState`. It forwards `replay_start` to the SWA metadata and pads the replayed slots of restored groups. The ROCm SWA builder already clamps at `replay_start`, so no kernel changes are needed.
- Tradeoffs:
  - Replayed tokens see the sliding window only from `replay_start`, so the rebuilt window KV is approximate. Compressed and sparse top-k context stays exact. This is the same as on V4.1.
  - While the connector is configured, GPU prefix hits replay the last window too.

**Fast paths**
- `deepseek_v4/amd/rocm.py` now gates aiter sparse MLA (decode and prefill) and `extra_cache_nan_free` on `kv_connector_may_load_foreign_kv()`, instead of on `kv_transfer_config is not None`.
- UMBP with a derived namespace restores only whole blocks that the same cache layout wrote, so the fast paths stay on. Other connectors fall back as before.

**Speculative decoding**
- DSpark draft layers are SWA-only. They derive their window KV from the target's hidden states, so the same replay rebuilds them.
- EAGLE-style `"method": "mtp"` draft groups are not rebuilt by the replay and are rejected at startup. The error now suggests `rebuild_sliding_window: false`.

**Limitations**
- Single-node standalone mode only. Distributed (RDMA) mode is not covered.
- Each E2E configuration was run once, so run-to-run variance is not measured. A full-length (`DURATION=3600`) agentic run and a TP8 restore check are still to do.

**Docs**
- `docs/features/umbp_store_connector_usage.md`: adds the `rebuild_sliding_window` option and a "DeepSeek-V4 on ROCm" section.

---

AI assistance (Claude Code) was used to prepare this change. The submitter has reviewed every changed line and ran the tests listed above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
