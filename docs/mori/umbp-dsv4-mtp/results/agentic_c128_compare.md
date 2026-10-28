# AgentX override_dep8_c128, DURATION=600 (2026-10-28)

| Metric | baseline | umbp_replay |
|---|---|---|
| Requests (profiled) | 1155 | 1315 |
| Input tput (tok/s) | 148,395 | 170,036 |
| Output tput (tok/s) | 1,157 | 1,324 |
| Output tput / GPU (tok/s) | 144.7 | 165.4 |
| QPS (mean) | 1.84 | 2.10 |
| TTFT mean / p50 / p90 (s) | 4.50 / 2.49 / 11.54 | 3.65 / 2.25 / 7.88 |
| TPOT mean / p50 / p90 (ms) | 78.8 / 71.9 / 125.8 | 70.2 / 62.0 / 112.2 |
| E2E latency mean / p50 / p90 (s) | 52.5 / 32.6 / 120.2 | 45.6 / 27.5 / 105.9 |
| Interactivity p50 (tok/s/user) | 13.91 | 16.14 |
| Server GPU prefix hit (window) | 92.4% | 93.0% |
| Server external (UMBP) hit (window) | 0.0% | 0.8% |
| Prompt tokens from GPU cache (whole run) | 85.4% | 87.1% |
| Prompt tokens from UMBP (whole run) | 0.0% | 1.0% |
| Draft acceptance (whole run) | 41.9% | 43.0% |
| Mean acceptance length (whole run) | 3.51 | 3.58 |
| Preemptions (whole run) | 0 | 0 |
