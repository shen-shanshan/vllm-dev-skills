# GSM8K: Mega-MHC MXFP8 layer input (before / after)

- Model: DeepSeek-V4.1-Flash, 2x MI355X, DEP2 (attention DP2 + EP2, AITER MegaMoE), DSpark k=5; server in `serve.sh`.
- Eval: lm-eval 0.4.13 `local-completions`, `gsm8k`, 8-shot, all 1319 questions (settings of `.buildkite/lm-eval-harness/configs/DeepSeek-V4-Flash-MXFP4.yaml`); `run_gsm8k.sh`.
- Base: vLLM `60bd6dd2ed`. After: the same commit plus the change.

| Run | strict-match | flexible-extract |
|---|---:|---:|
| Base (`results_base/`) | 0.9257 ± 0.0072 | 0.9272 ± 0.0072 |
| Mega-MHC MXFP8 (`results_mhc_mxfp8/`) | 0.9356 ± 0.0068 | 0.9348 ± 0.0068 |

The change is bit-exact at the kernel level (the fused quant equals `mxfp8_e4m3_quantize`
of the same bf16 layer input), so the gap is run-to-run noise from DSpark's probabilistic
drafting and batch composition: the gap (about 0.01) is about one standard error of the
difference between two runs (about 0.0099).
