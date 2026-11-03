# AgentX TP2 c192, DURATION=3600, synthetic AL 3.51 (B0 2026-11-02, S1 2026-11-03; same code and vllm-rs 76c6f3f2a)

| Metric | b0_c192 | s1_c192 |
|---|---|---|
| Requests (profiled) | 11300 | 15289 |
| Input tput (tok/s) | 355,832 | 521,082 |
| Output tput (tok/s) | 2,778 | 3,954 |
| Output tput / GPU (tok/s) | 1,388.8 | 1,977.2 |
| Total tput / GPU (tok/s) | 179,305 | 262,518 |
| QPS (mean) | 3.12 | 4.21 |
| TTFT mean / p50 / p90 / p95 (s) | 7.12 / 2.45 / 19.46 / 25.75 | 1.59 / 0.42 / 4.18 / 8.19 |
| TPOT mean / p50 / p90 (ms) | 58.0 / 58.9 / 88.5 | 37.0 / 35.2 / 54.2 |
| E2E latency mean / p50 / p90 (s) | 59.9 / 28.5 / 135.0 | 36.3 / 15.8 / 81.5 |
| Interactivity p50 (tok/s/user) | 16.98 | 28.39 |
| Server GPU prefix hit (window) | 88.7% | 88.8% |
| Server external (UMBP) hit (window) | 0.0% | 5.7% |
| Prompt tokens from GPU cache (whole run) | 87.8% | 88.1% |
| Prompt tokens from UMBP (whole run) | 0.0% | 4.8% |
| Draft acceptance (whole run) | 50.2% | 50.2% |
| Mean acceptance length (whole run) | 3.51 | 3.51 |
| Prompt tokens computed (whole run) | 12.2% | 7.1% |
| Preemptions (whole run) | 0 | 0 |
