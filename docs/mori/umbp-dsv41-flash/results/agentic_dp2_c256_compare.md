# AgentX attention DP2 + expert TP2, c256, DURATION=3600, synthetic AL 3.51 (2026-11-03, base 77d2790e78, vllm-rs 40941f2c4)

| Metric | b0_dp2_c256 | s1_dp2_c256 |
|---|---|---|
| Requests (profiled) | 16344 | 16846 |
| Input tput (tok/s) | 536,592 | 557,020 |
| Output tput (tok/s) | 4,053 | 4,194 |
| Output tput / GPU (tok/s) | 2,026.7 | 2,097.0 |
| Total tput / GPU (tok/s) | 270,323 | 280,607 |
| QPS (mean) | 4.51 | 4.65 |
| TTFT mean / p50 / p90 / p95 (s) | 1.69 / 0.62 / 2.66 / 5.36 | 1.40 / 0.59 / 2.47 / 4.83 |
| TPOT mean / p50 / p90 (ms) | 54.8 / 51.6 / 77.0 | 52.4 / 50.6 / 72.2 |
| E2E latency mean / p50 / p90 (s) | 50.2 / 21.8 / 115.6 | 48.0 / 21.0 / 109.7 |
| Interactivity p50 (tok/s/user) | 19.36 | 19.75 |
| Server GPU prefix hit (window) | 95.0% | 95.1% |
| Server external (UMBP) hit (window) | 0.0% | 0.2% |
| Prompt tokens from GPU cache (whole run) | 92.9% | 93.1% |
| Prompt tokens from UMBP (whole run) | 0.0% | 0.2% |
| Draft acceptance (whole run) | 50.2% | 50.2% |
| Mean acceptance length (whole run) | 3.51 | 3.51 |
| Prompt tokens computed (whole run) | 7.1% | 6.7% |
| Preemptions (whole run) | 0 | 0 |
