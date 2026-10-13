# [ROCm][DSv4] Enqueue the default chain first in the CSA/HCA multi-stream fork at small batch sizes

## Purpose

#57407 added a `default_first` option to `execute_in_parallel` (`vllm/utils/multi_stream_utils.py`). When it is set, the default-stream chain is enqueued before the aux-stream chains. The aux streams still wait only on `start_event`, which marks the default-stream work before the fork, so they keep overlapping the default chain. DeepSeek-V4.1 on ROCm already uses this option.

The DeepSeek-V4 ROCm multi-stream paths still enqueue the aux chains first:

- HCA dual-stream, from #56853: the main compressor runs on aux stream 0.
- CSA three-stream, from #51794: the main and indexer compressors run on aux streams 0 and 1.

On the default stream, both paths run fused `wq_a`/`wkv` GEMM → split + norm → `wq_b` → q-norm/RoPE/KV insert. This is the longest chain in the fork. Starting it first shortens the tail of the step when the device is underfilled.

Benchmarks show that this is not true for every batch size. Default-first is faster at small token counts and slightly slower once the batch saturates the device. This PR therefore enables `default_first` only when the step has at most `_DEFAULT_FIRST_TOKEN_THRESHOLD = 96` tokens.

## Changes

All changes are in `vllm/models/deepseek_v4/amd/rocm.py`:

- Add the module-level constant `_DEFAULT_FIRST_TOKEN_THRESHOLD = 96`.
- In `DeepseekV4ROCMAiterMLAAttention._prepare_and_attn`, compute `default_first = hidden_states.shape[0] <= _DEFAULT_FIRST_TOKEN_THRESHOLD` and pass it to the HCA and CSA `execute_in_parallel` calls.

```python
default_first = hidden_states.shape[0] <= _DEFAULT_FIRST_TOKEN_THRESHOLD
...
execute_in_parallel(..., enable=True, default_first=default_first)
```

Notes:

- **No runtime cost.** On ROCm, the multi-stream path only runs inside CUDA graph capture or with full-cudagraph metadata (`RocmPlatform.enable_multi_stream_overlap`). `hidden_states.shape[0]` is a concrete Python int at capture time, so each capture size bakes in its own launch order.
- **No change to correctness.** Every aux chain only reads `hidden_states` and its own compressor state, both produced before the fork. The join, where the `done_events` are waited on before returning, is unchanged.
- **Scope.** The threshold is a hard-coded constant, not an env var. It only affects the DeepSeek-V4 ROCm attention class, so DeepSeek-V4.1, CUDA, and other models are untouched. DeepSeek-V4-Flash shares this class and is also gated by the threshold.

### Choosing the threshold

The tests ran with MTP and `num_speculative_tokens=2`, so each decode step runs `3 × conc` tokens. With MTP, capture sizes are rounded to multiples of 3, and both 96 and 144 are capture sizes.

| Conc | Tokens / step | TPOT vs. main | Throughput vs. main |
| ---: | ---: | ---: | ---: |
| 1 | 3 | -7.26% | +7.22% |
| 4 | 12 | -1.26% | +2.99% |
| 16 | 48 | -0.21% | +0.40% |
| 32 | 96 | -0.73% | +0.92% |
| 48 | 144 | +0.65% | -0.49% |
| 64 | 192 | +0.69% | -0.26% |

The sign flips between 96 and 144 tokens. The threshold is set to 96, the last measured point with a gain. The range from 99 to 141 tokens was not measured.

## Test Plan

Model: DeepSeek-V4-Pro on ROCm. `<!-- TODO: GPU model, TP/DP/EP, image / commit -->`

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro <!-- TODO: parallel / other flags --> \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'

vllm bench serve <!-- TODO: dataset / input-output lens / num-prompts --> \
  --max-concurrency {1,4,16,32,48,64}
```

Accuracy: `<!-- TODO: e.g. lm_eval gsm8k, main vs. this PR -->`

## Test Result

### Performance

"main" is the version before this PR, where aux chains are always enqueued first.

| Conc | Version | Output token throughput (tok/s) | Change vs. main | Mean TTFT (ms) | Change vs. main | Mean TPOT (ms) | Change vs. main |
| ---: | :------ | ------------------------------: | --------------: | -------------: | --------------: | -------------: | --------------: |
|    1 | main    |                          149.11 |               — |         593.98 |               — |           6.06 |               — |
|    1 | default_first  |                   159.87 |         7.22% ↑ |         588.87 |         0.86% ↓ |           5.62 |         7.26% ↓ |
|    4 | main    |                          456.45 |               — |         492.52 |               — |           7.95 |               — |
|    4 | default_first  |                   470.10 |         2.99% ↑ |         464.21 |         5.75% ↓ |           7.85 |         1.26% ↓ |
|   16 | main    |                         1028.75 |               — |         674.62 |               — |          14.31 |               — |
|   16 | default_first  |                  1032.88 |         0.40% ↑ |         697.59 |         3.40% ↑ |          14.28 |         0.21% ↓ |
|   32 | main    |                         1361.78 |               — |         960.48 |               — |          21.98 |               — |
|   32 | default_first  |                  1374.31 |         0.92% ↑ |         981.40 |         2.18% ↑ |          21.82 |         0.73% ↓ |
|   48 | main    |                         1549.52 |               — |        1296.85 |               — |          29.04 |               — |
|   48 | default_first  |                  1541.91 |         0.49% ↓ |        1279.48 |         1.34% ↓ |          29.23 |         0.65% ↑ |
|   64 | main    |                         1659.41 |               — |        1583.06 |               — |          36.16 |               — |
|   64 | default_first  |                  1655.04 |         0.26% ↓ |        1599.42 |         1.03% ↑ |          36.41 |         0.69% ↑ |

The rows above compare always-`default_first` with main; they were used to pick the threshold. With the threshold, conc ≤ 32 keeps the default-first results, and conc ≥ 48 falls back to the same launch order as main.

`<!-- TODO: optionally re-run conc=48/64 with this PR to confirm parity with main -->`

TTFT changes are small and do not follow the batch size: conc=16/32 go up, conc=48 goes down. Prefill does not take the captured multi-stream path, so these changes are most likely queueing noise. The threshold was chosen on TPOT.

### Accuracy

`<!-- TODO -->`

## Notes

- Duplicate check: `<!-- TODO: gh pr list --repo vllm-project/vllm --state open --search "default_first deepseek rocm" -->`, and no open PR covers this change.
- No new tests. This only changes the launch order of an existing, tested utility.
- AI assistance (Claude Code) was used to prepare this change. The submitter reviewed every changed line and ran the benchmarks above.
