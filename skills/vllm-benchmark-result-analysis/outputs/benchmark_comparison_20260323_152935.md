# vLLM Benchmark Comparison Report

> Generated: 2026-03-23 15:29
> vLLM Benchmark Comparison

---

## Performance Metrics

| Metric | Before this PR | After this PR | Comparison |
| :----- | :------------- | :------------ | :--------- |
| **Throughput** | | | |
| Request throughput (req/s) | 6.09 | 6.46 | +6.08% ↑ |
| Output token throughput (tok/s) | 748.84 | 794.40 | +6.08% ↑ |
| Total token throughput (tok/s) | 1,155.93 | 1,225.91 | +6.05% ↑ |
| Peak output token throughput (tok/s) | 2,945 | 2,691 | -8.62% ↓ |
| **Latency** | | | |
| Benchmark duration (s) | 82.09 | 77.44 | -5.66% ↓ |
| Mean TTFT (ms) | 10,365.81 | 6,888.64 | -33.54% ↓ |
| Median TTFT (ms) | 6,141.34 | 4,128.82 | -32.77% ↓ |
| P99 TTFT (ms) | 22,606.41 | 17,487.94 | -22.64% ↓ |
| Mean TPOT (ms) | 265.83 | 240.14 | -9.66% ↓ |
| Median TPOT (ms) | 277.96 | 259.18 | -6.76% ↓ |
| P99 TPOT (ms) | 611.59 | 313.15 | -48.80% ↓ |
| Mean ITL (ms) | 254.68 | 241.84 | -5.04% ↓ |
| Median ITL (ms) | 95.97 | 121.08 | +26.16% ↑ |
| P99 ITL (ms) | 1,787.15 | 1,470.33 | -17.73% ↓ |

---

## AI Summary

- The standout improvement is P99 TPOT, which dropped by **-48.80%** from 612ms to 313ms, indicating significantly reduced tail latency for token generation and a much more consistent decoding experience.

- TTFT saw substantial gains across all percentiles: mean improved by **-33.54%** (10,366ms → 6,889ms), median by **-32.77%** (6,141ms → 4,129ms), and P99 by **-22.64%** (22,606ms → 17,488ms), meaning requests start generating tokens much sooner.

- Overall throughput improved consistently by roughly **+6%** across request, output token, and total token throughput, with benchmark duration dropping from 82.09s to 77.44s — a solid end-to-end speedup.

- Median ITL increased by **+26.16%** (95.97ms → 121.08ms), which is a notable regression. However, mean ITL improved by **-5.04%** and P99 ITL improved by **-17.73%**, suggesting the median shift may reflect a change in the latency distribution shape rather than a broad degradation.

- Peak output token throughput decreased by **-8.62%** (2,945 → 2,691 tok/s) and peak concurrent requests dropped from 407 to 369. This may indicate the PR smooths out burst behavior, trading peak capacity for more consistent sustained performance.
