# GSM8K: MXFP8 input quant fusion (before / after)

- Model: DeepSeek-V4.1-Flash, 2x MI355X, DEP2 (attention DP2 + EP2, AITER MegaMoE), DSpark k=5; server in `serve.sh`.
- Eval: lm-eval 0.4.13 `local-completions`, `gsm8k`, 8-shot, all 1319 questions (settings of `.buildkite/lm-eval-harness/configs/DeepSeek-V4-Flash-MXFP4.yaml`); `run_gsm8k.sh`.
- Base: vLLM `60bd6dd2ed`. The other rows are the same commit plus the change.

| Run | strict-match | flexible-extract |
|---|---:|---:|
| Base (`results_base/`) | 0.9257 ± 0.0072 | 0.9272 ± 0.0072 |
| #2 + #3, Mega-MHC MXFP8 (`results_mhc_mxfp8/`) | 0.9356 ± 0.0068 | 0.9348 ± 0.0068 |
| #2 + #3 + #4, also fused SiLU-and-mul MXFP8 (`results_quant_fusion/`) | 0.9265 ± 0.0072 | 0.9272 ± 0.0072 |

Every fused kernel is bit-exact against the unfused path (same quant of the same bf16
values), so the spread is run-to-run noise from DSpark's probabilistic drafting and batch
composition; it stays within about one standard error of the difference between two runs
(about 0.01).
