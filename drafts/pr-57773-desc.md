## Purpose

This PR enables prefill dual-batch-overlap (DBO) for DeepSeek-V4 on ROCm in data-parallel deployments: attention DP, experts TP via flattened TP, no EP (e.g. `--data-parallel-size 8`).

Today `--enable-dbo` splits a step into two ubatches and runs them as a two-thread ping-pong on a compute stream and a comm stream. However, only the DeepEP-HT prepare/finalize moves its collectives to the comm stream. The naive DP/EP path (`allgather_reducescatter`, used by DP deployments without `--enable-expert-parallel`) runs its MoE all-gather (dispatch) and reduce-scatter (combine) synchronously on the compute stream. Both ubatches therefore serialize, and DBO gives no benefit in prefill. On ROCm this path was also blocked by the microbatching config check.

This PR makes the naive DP/EP prepare/finalize hand each collective to the comm stream, using the same compute/comm handoff as DeepEP-HT. While one ubatch's all-gather/reduce-scatter is in flight, the peer ubatch runs attention/MoE compute. It also fixes several DBO correctness issues that DeepSeek-V4 hits on this path.

### Key Changes

**MoE prepare/finalize (DBO comm overlap)**
- **`MoEPrepareAndFinalizeNaiveDPEPModular`** (`fused_moe/prepare_finalize/naive_dp_ep.py`): `dispatch()` / `combine_into_output()` now run inside a new `_comm_region()` context. Under an active DBO context, it yields the CPU to the peer ubatch, switches to the comm stream for the collective, and syncs back to the compute stream on exit. Outside DBO, and during cudagraph capture, it does nothing: the collective stays on the calling stream, so non-DBO behavior is unchanged on every platform. The class reports `supports_dbo()` only on ROCm, the only platform where this path has been validated.
- **`FusedMoEPrepareAndFinalize.supports_dbo()`** (`modular_kernel.py`): New capability, defaulting to `supports_async()`, so existing implementations behave exactly as before. The synchronous `_prepare` / `_finalize` paths now assert `not dbo_enabled() or supports_dbo()`. The naive DP/EP class keeps `supports_async()` as `False`. This avoids overloading `supports_async()`, which also drives MK-internal shared-expert overlap on every forward.
- **`VllmConfig`** (`config/vllm.py`): Allows `allgather_reducescatter` with microbatching on ROCm.

> [!NOTE]
> **`supports_dbo()` vs. `dbo_enabled()`**
>
> - `supports_dbo()` is a static **capability** of the prepare/finalize class: *can* it run correctly under DBO? It does not depend on the current step. For the naive DP/EP class it is `current_platform.is_rocm()`. It is used only as a guard in the modular kernel: `assert not dbo_enabled() or supports_dbo()`.
> - `dbo_enabled()` is per-thread **runtime state**: is the calling thread *currently* inside a ubatch context? It is True only inside the ubatch threads of a step that was actually split. It is used in `_comm_region()` to decide, per call, whether to hand the collective to the comm stream.
>
> | Platform | Step | `supports_dbo()` | `dbo_enabled()` | Behavior |
> |---|---|---|---|---|
> | ROCm | not split (below threshold, a DP rank opted out, or DBO off) | True | False | Collective stays on the calling stream (unchanged) |
> | ROCm | split prefill | True | True | Collective runs on the comm stream, overlapped with the peer ubatch |
> | Others | any | False | False | Unchanged. Microbatching with `allgather_reducescatter` is still rejected at config time. |
>
> Neither replaces the other. With only `dbo_enabled()`, a synchronous prepare/finalize that never handled DBO could silently run under it. With only `supports_dbo()`, `_comm_region()` could not tell a split step from a normal one. Previously the PR overloaded `supports_async()` to return `dbo_enabled()`, which mixed the two concepts. It also changed MK-internal shared-expert scheduling per thread, because `can_overlap_shared_experts` reads `supports_async()`.

**Ubatching fixes**
- **`UBatchWrapper._run_ubatches`** (`v1/worker/gpu_ubatch_wrapper.py`): Eager ubatch threads now call `torch.accelerator.set_device_index(self.device)`, as `_capture_ubatches` already did. A new thread starts on device 0. Without this call, each thread's first `current_stream()` created a stream on GPU 0 and waited on it, so every DP rank put work on GPU 0. That slowed rank 0, and every rank then waited on rank 0 at each MoE collective. This was the main cause of the earlier ~30% regression with DBO.
- **`maybe_create_ubatch_slices`** (`v1/worker/ubatch_utils.py`): The last split point (`cu_num_tokens[-1]`, a `np.int32`) is cast to `int` at the source. Previously the numpy scalar reached the DeepSeek-V4 C128A metadata builder and failed Triton specialization (`TypeError: failed to specialize argument of type: numpy.int32`). An explicit `list[int]` `split_point` is now handled as explicit split points instead of being list-multiplied.
- **`_make_metadata_with_slice`** (`v1/worker/ubatch_utils.py`): Slices `CommonAttentionMetadata.positions` per ubatch. Without this, startup fails with `positions is required for C128A metadata build`.
- **`UBatchContext.token_offset` / `dbo_current_ubatch_token_offset()`** (`v1/worker/ubatching.py`): Exposes each ubatch's row offset in the parent batch. It returns 0 outside DBO.

**DeepSeek-V4 (AMD)**
- **`DeepseekV4Model.forward`** (`models/deepseek_v4/amd/model.py`): Writes `_mtp_hidden_buffer` at the current ubatch's token offset. Previously both ubatch threads wrote from row 0, so the MTP drafter read misaligned hidden states for the step.

**Tests**
- `tests/kernels/moe/test_naive_dp_ep_dbo.py`: Covers `_comm_region` handoff, the no-op without DBO, skip during graph capture, compute-stream restore on error, `supports_dbo()` being ROCm-only while `supports_async()` stays `False`, and the default `supports_dbo()` contract.
- `tests/v1/worker/test_ubatch_offsets.py`: Covers per-thread ubatch token offsets and Python-int ubatch slice bounds.

### Trace Comparison (Conc=64, Prefill)

Main (DBO disabled):

<img alt="main" src="https://github.com/user-attachments/assets/158e1292-7b2f-482e-815b-5fa970a52834" />

This PR (DBO enabled, **9.25%** faster per layer):

<img alt="dbo-new" src="https://github.com/user-attachments/assets/6afc09c2-753c-4f25-a4c6-d05dcbe1fd92" />

## Test Plan

Unit tests:

```bash
pytest tests/kernels/moe/test_naive_dp_ep_dbo.py tests/v1/worker/test_ubatch_offsets.py -v
```

Benchmark with SA InferenceX `8k1k` (with `OSL=1`) and `agentic` workloads on 8x MI355X, using the server launch command below. Main runs use the same command without the three `--*dbo*` flags.

```bash
export VLLM_ENGINE_READY_TIMEOUT_S='10800'
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL='32768'
export VLLM_ROCM_USE_AITER='1'
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION='INT4'
export VLLM_ROCM_USE_AITER_MOE='1'
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS='0'
export OMP_NUM_THREADS='1'

vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --port 8333 \
  --served-model-name 'deepseek-ai/DeepSeek-V4-Pro' \
  --trust-remote-code \
  --async-scheduling \
  --distributed-executor-backend 'mp' \
  --kv-cache-dtype 'fp8' \
  --max-num-batched-tokens 8192 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --gpu-memory-utilization 0.86 \
  --moe-backend 'aiter' \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.49}' \
  --tokenizer-mode 'deepseek_v4' \
  --tool-call-parser 'deepseek_v4' \
  --reasoning-parser 'deepseek_v4' \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --prefill-schedule-interval 8 \
  --long-prefill-token-threshold 16384 \
  --max-num-seqs 64 \
  --load-format fastsafetensors \
  --enable-dbo \
  --dbo-prefill-token-threshold 8192 \
  --dbo-decode-token-threshold 8192
```

> [!NOTE]
> `--dbo-decode-token-threshold` defaults to 32. With MTP, a 64-request decode step has 256 tokens and would be microbatched, and microbatched decode is ~1.8x slower per token. Setting it to 8192 keeps decode on the regular FULL cudagraph path, so DBO only applies to prefill.

Accuracy test with `gsm8k` using `lm_eval`:

```bash
lm_eval \
  --model local-completions \
  --model_args model=/shared/models/deepseek-ai/DeepSeek-V4-Pro,base_url=http://0.0.0.0:8333/v1/completions,num_concurrent=64,max_retries=10,max_gen_toks=2048,timeout=60000 \
  --batch_size auto \
  --tasks gsm8k \
  --num_fewshot 5 \
  --output_path . 2>&1 | tee -a eval.log
```

## Test Result

### Unit Tests

All tests in `tests/kernels/moe/test_naive_dp_ep_dbo.py` and `tests/v1/worker/test_ubatch_offsets.py` pass.

### Benchmark

**Fixed sequence length (prefill only):**

> Uses the SA InferenceX `8k1k` workload with `OSL=1` to measure prefill only. This way, the requests received across DP ranks meet the DBO gate in more steps (microbatching is all-or-none across DP ranks).

| Concurrency | Version | Mean TTFT (ms) | Change | P99 TTFT (ms) | Change |
|---|---|---|---|---|---|
| 1 | Main (DBO disabled) | 922.68 | — | 922.68 | — |
| | This PR (DBO enabled) | 1000.28 | 8.41% ↑ | 1000.28 | 8.41% ↑ |
| 4 | Main (DBO disabled) | 1005.18 | — | 1220.48 | — |
| | This PR (DBO enabled) | 1016.20 | 1.10% ↑ | 1289.38 | 5.65% ↑ |
| 16 | Main (DBO disabled) | 2140.47 | — | 3912.48 | — |
| | This PR (DBO enabled) | 2205.26 | 3.03% ↑ | 3767.09 | 3.72% ↓ |
| 64 | Main (DBO disabled) | 5858.52 | — | 10182.92 | — |
| | This PR (DBO enabled) | 5477.98 | 6.49% ↓ | 9482.81 | 6.87% ↓ |
| 128 | Main (DBO disabled) | 10858.97 | — | 19695.63 | — |
| | This PR (DBO enabled) | 10042.61 | 7.52% ↓ | 18934.63 | 3.86% ↓ |

DBO helps most at high concurrency (`conc=64/128`), where it reduces both mean and tail TTFT by ~4–8%. At low concurrency it adds a small regression, largest at `conc=1` (+8.4%). There the batch is too small to hide the collective, and the split adds overhead. Users can raise `--dbo-prefill-token-threshold` to keep small batches unsplit.

**Agentic replay (conc=64):**

> DeepSeek-V4-Pro FP4 on 8x MI355X, DP8, MTP (3 speculative tokens), prefix caching on. Requests average ~97K prompt tokens and ~620 output tokens. 729 (Main) and 766 (DBO) requests were profiled, with no errors.

| Version | Output Token Throughput (tok/s) | Change | Mean TTFT (ms) | Change | Mean TPOT (ms) | Change |
|---|---|---|---|---|---|---|
| Main (DBO disabled) | 703.40 | — | 2,173.93 | — | 44.69 | — |
| This PR (DBO enabled) | 752.46 | 6.97% ↑ | 2,112.09 | 2.85% ↓ | 43.15 | 3.45% ↓ |

The earlier ~30% throughput regression with DBO is gone, mainly because of the device fix in `_run_ubatches`. DBO now matches or slightly beats Main on all three metrics. Flat TPOT confirms that decode is not microbatched.

Each configuration was run once. An earlier DBO run with the identical config and code gave 727.63 tok/s and 2,488.60 ms mean TTFT, so run-to-run variance is comparable to the deltas above. Read these results as parity or a modest gain rather than a definitive speedup. Because microbatching is all-or-none across DP ranks, only a small share of steps are split in this workload.

### Accuracy Test

| Task | Filter | n-shot | Metric | Main (DBO disabled) | This PR (DBO enabled) | Change |
|---|---|---|---|---|---|---|
| gsm8k | flexible-extract | 5 | exact_match ↑ | 0.9492 ± 0.0060 | 0.9530 ± 0.0058 | +0.0038 (+0.40%) ↑ |
| gsm8k | strict-match | 5 | exact_match ↑ | 0.9500 ± 0.0060 | 0.9538 ± 0.0058 | +0.0038 (+0.40%) ↑ |

Accuracy matches Main within the standard error.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>
