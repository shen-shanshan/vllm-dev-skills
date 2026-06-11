# vLLM Benchmark Comparison Report

> Generated: 2026-03-23 15:08
> vLLM Benchmark Comparison

---

## Performance Metrics

| Metric | Before this PR | After this PR | Comparison |
| :----- | :------------- | :------------ | :--------- |
| **Throughput** | | | |
| Request throughput (req/s) | 6.36 | 6.46 | +1.57% ↑ |
| Output token throughput (tok/s) | 781.78 | 794.40 | +1.61% ↑ |
| Total token throughput (tok/s) | 1,207.07 | 1,225.91 | +1.56% ↑ |
| Peak output token throughput (tok/s) | 2,475 | 2,691 | +8.73% ↑ |
| **Latency** | | | |
| Benchmark duration (s) | 78.58 | 77.44 | -1.45% ↓ |
| Mean TTFT (ms) | 7,116.24 | 6,888.64 | -3.20% ↓ |
| Median TTFT (ms) | 4,295.84 | 4,128.82 | -3.89% ↓ |
| P99 TTFT (ms) | 18,370.87 | 17,487.94 | -4.81% ↓ |
| Mean TPOT (ms) | 245.78 | 240.14 | -2.29% ↓ |
| Median TPOT (ms) | 264.03 | 259.18 | -1.84% ↓ |
| P99 TPOT (ms) | 334.38 | 313.15 | -6.35% ↓ |
| Mean ITL (ms) | 246.99 | 241.84 | -2.09% ↓ |
| Median ITL (ms) | 117.71 | 121.08 | +2.86% ↑ |
| P99 ITL (ms) | 1,327.55 | 1,470.33 | +10.76% ↑ |

---

## AI Summary

- The most notable improvement is in P99 TPOT, which dropped **-6.35%** from 334.38ms → 313.15ms, indicating reduced tail latency for per-token generation under heavy load.
- TTFT improved across all percentiles: mean dropped **-3.20%** (7,116ms → 6,889ms), median **-3.89%** (4,296ms → 4,129ms), and P99 **-4.81%** (18,371ms → 17,488ms), reflecting faster time-to-first-token across the board.
- TPOT also improved consistently, with mean down **-2.29%** (245.78ms → 240.14ms) and median down **-1.84%** (264.03ms → 259.18ms), showing a modest but steady reduction in per-token generation time.
- Throughput saw a slight uplift of roughly **+1.6%** across request, output token, and total token throughput. Peak output token throughput jumped **+8.73%** (2,475 → 2,691 tok/s), suggesting better burst handling capacity.
- P99 ITL increased **+10.76%** (1,328ms → 1,470ms), the largest regression in the run. Median ITL also ticked up **+2.86%** (117.71ms → 121.08ms). These tail-latency spikes may reflect scheduling variability under peak concurrency and could be within run-to-run noise, but are worth monitoring.
- Overall, the PR delivers a consistent improvement in both throughput and latency, with the caveat that P99 inter-token latency regressed — likely a transient effect given that mean ITL still improved by **-2.09%**.
