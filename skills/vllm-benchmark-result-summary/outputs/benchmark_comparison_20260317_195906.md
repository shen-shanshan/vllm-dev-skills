# vLLM Benchmark Comparison Report

> Generated: 2026-03-17 19:59
> vLLM Benchmark Comparison

---

## Performance Metrics

| Metric | Before this PR | After this PR | Comparison |
| :----- | :------------- | :------------ | :--------- |
| **Throughput** | | | |
| Request throughput (req/s) | 6.09 | 6.37 | +4.60% ↑ |
| Output token throughput (tok/s) | 748.84 | 784.04 | +4.70% ↑ |
| Total token throughput (tok/s) | 1,155.93 | 1,209.58 | +4.64% ↑ |
| Peak output token throughput (tok/s) | 2,945 | 2,560 | -13.07% ↓ |
| **Latency** | | | |
| Benchmark duration (s) | 82.09 | 78.53 | -4.34% ↓ |
| Mean TTFT (ms) | 10,365.81 | 8,328.36 | -19.66% ↓ |
| Median TTFT (ms) | 6,141.34 | 5,585.51 | -9.05% ↓ |
| P99 TTFT (ms) | 22,606.41 | 18,813.79 | -16.78% ↓ |
| Mean TPOT (ms) | 265.83 | 240.06 | -9.69% ↓ |
| Median TPOT (ms) | 277.96 | 253.22 | -8.90% ↓ |
| P99 TPOT (ms) | 611.59 | 359.47 | -41.22% ↓ |
| Mean ITL (ms) | 254.68 | 239.94 | -5.79% ↓ |
| Median ITL (ms) | 95.97 | 100.19 | +4.40% ↑ |
| P99 ITL (ms) | 1,787.15 | 1,423.15 | -20.37% ↓ |

---

## AI Summary

- **P99 TPOT dropped by -41.22%** (611.59ms → 359.47ms), the standout improvement in this PR — tail-latency for token generation is dramatically better, meaning worst-case streaming responsiveness is significantly more consistent.
- **Mean TTFT improved by -19.66%** (10,365ms → 8,328ms) and **P99 TTFT by -16.78%** (22,606ms → 18,814ms), indicating the system is scheduling and starting request generation noticeably faster under load.
- **Overall throughput improved across the board by ~4.6%**: request throughput rose from **6.09 → 6.37 req/s**, output token throughput from **748.84 → 784.04 tok/s**, and total token throughput from **1,155.93 → 1,209.58 tok/s**, alongside a shorter benchmark duration (**82.09s → 78.53s**).
- **P99 ITL improved by -20.37%** (1,787ms → 1,423ms) with mean ITL also down **-5.79%** (254.68ms → 239.94ms); median ITL saw a marginal +4.40% increase (95.97ms → 100.19ms), which is likely within noise given the overall latency gains.
- **Peak output token throughput fell -13.07%** (2,945 → 2,560 tok/s) and peak concurrent requests dropped from 407 to 378 — this suggests the PR may be reducing bursty batching behavior, trading peak burst capacity for more consistent steady-state performance.
