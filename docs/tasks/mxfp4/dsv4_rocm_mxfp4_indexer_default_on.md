# [ROCm][Perf] Enable the paged MXFP4 sparse indexer by default for DeepSeek-V4

## Purpose

#60834 brought the paged MXFP4 sparse indexer (dense path) to DeepSeek-V4 on ROCm gfx950. It was opt-in:

```bash
--attention-config '{"indexer_kv_dtype": "mxfp4"}'
```

Its measurements show the MXFP4 indexer is better than the FP8 one on almost every axis:

- lower TTFT (up to -30.7%) and TPOT;
- +9.2% KV cache capacity (68 B vs 132 B per compressed token);
- near-lossless accuracy: GSM8K -0.16pt, and GPQA Diamond is unchanged at 0.7121.

The results are copied below. This PR therefore makes MXFP4 the **default** indexer K cache dtype for DeepSeek-V4 (`model_type == "deepseek_v4"`, i.e. V4-Pro / V4-Flash) on ROCm CDNA4, and logs that it is enabled.

Nothing else changes:

- CUDA, XPU and CPU behave as before;
- DeepSeek-V4.1 keeps its current opt-in;
- every other model is untouched.

## Changes

The feature is selected only by `attention_config.indexer_kv_dtype`. About 10 call sites read it through `dsa_indexer_uses_fp4()`, which resolves `"auto"` to FP8: model construction, KV-cache spec, metadata builder and JIT warmup. So this PR rewrites `"auto"` → `"mxfp4"` **once at config time**. Every call site, and every worker that receives the pickled config, then sees one consistent value.

- `vllm/platforms/rocm.py`: `RocmPlatform.check_and_update_config` now calls a new helper, `_maybe_default_dsv4_mxfp4_indexer(vllm_config)`.
  - It lives in the ROCm-only override of the existing `Platform.check_and_update_config` hook, so no shared code path changes.
  - The hook runs after the model config hooks and DCP defaults are resolved.
  - It switches to MXFP4 only if **all** of the conditions below hold. Otherwise it silently keeps FP8, so a configuration that worked before never starts failing.

  | Condition | Why |
  |---|---|
  | `hf_config.model_type == "deepseek_v4"` | Scope: V4 only (the same field the `dsa_indexer_uses_fp4` gate checks). V4.1 stays opt-in. |
  | `indexer_kv_dtype == "auto"` | An explicit user choice (e.g. `"fp8"`) always wins. |
  | `not indexer_sparse_logits` | V4 has no candidate pool, so the sparse-logits path would raise. |
  | `get_cdna_version() == 4` | aiter's paged MXFP4 kernels need CDNA4 (MI350X/MI355X). |
  | `rocm_aiter_ops.is_enabled()` | The path runs on aiter's kernels (`VLLM_ROCM_USE_AITER=1`). |
  | DCP == 1 and PCP == 1 | `RocmMxfp4IndexerMetadataBuilder` does not support decode/prefill context parallel. |

  When the default is applied, it logs once:

  ```
  DeepSeek-V4 on ROCm CDNA4: using aiter's paged MXFP4 sparse indexer K cache by default. Pass --attention-config '{"indexer_kv_dtype": "fp8"}' to use the FP8 indexer cache.
  ```

- `vllm/config/attention.py`: docstring only. The `"auto"` description of `indexer_kv_dtype` now mentions the ROCm DeepSeek-V4 default.

**Opt out:** `--attention-config '{"indexer_kv_dtype": "fp8"}'`.

## Test Plan

- `pre-commit run --files vllm/platforms/rocm.py vllm/config/attention.py` (ruff, format, mypy).
- A quick logic check of the helper on MI355X (gfx950, `VLLM_ROCM_USE_AITER=1`) with stub configs.
- Performance and accuracy of the MXFP4 indexer itself were measured in #60834. The model was DeepSeek-V4-Pro on 8× MI355X: `8k1k` and agentic replay with InferenceX, GSM8K (5-shot) and GPQA Diamond. Those numbers are reproduced below; with this PR the "MXFP4" column is what users get without passing any flag.

## Test Result

### Lint / logic check

`pre-commit` passes. Helper behaviour:

| Input | Resulting `indexer_kv_dtype` |
|---|---|
| `deepseek_v4`, `"auto"` | `mxfp4` (log printed) |
| `deepseek_v41`, `"auto"` | `auto` (unchanged → FP8) |
| `deepseek_v4`, explicit `"fp8"` | `fp8` (user choice kept) |
| `deepseek_v4`, `"auto"`, DCP=2 | `auto` (unchanged → FP8) |

### Benchmark (from #60834, DeepSeek-V4-Pro)

Fixed sequence length (`8k1k`, TP8, MTP=2):

| Conc | Indexer | Output token throughput (tok/s) |  Change | Mean TTFT (ms) |   Change | Mean TPOT (ms) |  Change |
| ---: | :------ | ------------------------------: | ------: | -------------: | -------: | -------------: | ------: |
|    1 | FP8 (old default)   |                          145.99 |       — |         597.24 |        — |           6.24 |       — |
|    1 | MXFP4 (new default) |                          156.34 | 7.09% ↑ |         413.91 | 30.70% ↓ |           5.96 | 4.49% ↓ |
|    4 | FP8 (old default)   |                          454.91 |       — |         504.07 |        — |           8.02 |       — |
|    4 | MXFP4 (new default) |                          466.92 | 2.64% ↑ |         437.29 | 13.25% ↓ |           7.95 | 0.87% ↓ |
|   16 | FP8 (old default)   |                         1020.90 |       — |         744.88 |        — |          14.51 |       — |
|   16 | MXFP4 (new default) |                         1030.30 | 0.92% ↑ |         681.89 |  8.46% ↓ |          14.46 | 0.34% ↓ |
|   64 | FP8 (old default)   |                         1629.53 |       — |        1637.36 |        — |          36.85 |       — |
|   64 | MXFP4 (new default) |                         1656.28 | 1.64% ↑ |        1622.58 |  0.90% ↓ |          36.29 | 1.52% ↓ |

Agentic replay (DP8, MTP=3, prefix caching):

| Indexer | Output Token Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | KV Cache Capacity (tokens) |
|---|---|---|---|---|
| FP8 (old default)   | 697.25 | 2,442.24 | 46.01 | 4,589,684 |
| MXFP4 (new default) | 675.83 (3.07% ↓) | 2,321.05 (4.96% ↓) | 43.42 (5.62% ↓) | 5,010,749 (9.17% ↑) |

- **Latency improves:** mean TPOT -5.62% and mean TTFT -4.96%.
  - TPOT is the most consistent signal: both MXFP4 runs (44.60 / 42.24 ms) had lower TPOT than both FP8 runs (46.98 / 45.03 ms).
  - TTFT varied by up to 13% between runs of the same configuration, so treat its gain as indicative.
- **Output throughput is within noise (-3.07%).** On this workload it is driven mainly by the trace replay and by output lengths.
- **KV cache capacity: +9.17%.** Peak KV usage stayed below 40% here, so this workload does not yet turn the extra capacity into throughput. Higher-concurrency or longer-context workloads should benefit more.

### Accuracy (from #60834, DeepSeek-V4-Pro)

GSM8K:

| Task  | Filter           | n-shot | Metric        | FP8 (old default) | MXFP4 (new default) |   Change |
| ----- | ---------------- | -----: | ------------- | ----------------: | ------------------: | -------: |
| gsm8k | flexible-extract |      5 | exact_match ↑ |            0.9492 |              0.9477 | -0.16% ↓ |
| gsm8k | strict-match     |      5 | exact_match ↑ |            0.9500 |              0.9484 | -0.17% ↓ |

GPQA Diamond:

| Indexer | GPQA Diamond | Empty Responses (reasoning over budget) |
|---|---|---|
| FP8 (old default)   | 0.7121 ± 0.0323 | 13 / 198 |
| MXFP4 (new default) | 0.7121 ± 0.0323 | 11 / 198 |

- **Accuracy is unchanged on GPQA:** both indexers score 0.7121. Per question, 124 are correct in both runs, and the disagreements are symmetric (17 vs 17, McNemar p = 1.0).
- **This exercises the indexer's long-context path:** reasoning traces run to tens of thousands of tokens, far past the 4096 tokens where sparse top-k selection starts.
- **The GSM8K change is negligible** (< 0.2% relative).

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
