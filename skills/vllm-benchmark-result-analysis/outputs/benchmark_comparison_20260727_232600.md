# vLLM Benchmark Comparison Report

> Generated: 2026-07-27 23:26
> vLLM Benchmark Comparison

---

## Performance Metrics

| Metric | Before this PR | After this PR | Comparison |
| :----- | :------------- | :------------ | :--------- |
| **Throughput** | | | |
| Request throughput (req/s) | 0.89 | 0.89 | +0.00% ↓ |
| Output token throughput (tok/s) | 822.26 | 819.69 | -0.31% ↓ |
| Total Token throughput (tok/s) | 7,411.66 | 7,388.54 | -0.31% ↓ |
| **Latency** | | | |
| Benchmark duration (s) | 717.45 | 719.69 | +0.31% ↑ |
| Mean TTFT (ms) | 2,111.71 | 2,119.92 | +0.39% ↑ |
| Median TTFT (ms) | 587.27 | 586.98 | -0.05% ↓ |
| P99 TTFT (ms) | 25,604.31 | 25,705.54 | +0.40% ↑ |
| Mean TPOT (ms) | 73.98 | 74.17 | +0.26% ↑ |
| Median TPOT (ms) | 75.46 | 75.64 | +0.24% ↑ |
| P99 TPOT (ms) | 86.79 | 86.74 | -0.06% ↓ |
| Mean ITL (ms) | 74.10 | 74.29 | +0.26% ↑ |
| Median ITL (ms) | 46.87 | 46.92 | +0.11% ↑ |
| P99 ITL (ms) | 493.10 | 491.91 | -0.24% ↓ |
| **Other** | | | |
| P90 TTFT (ms) | 1,958.06 | 1,885.92 | -3.68% ↓ |
| P99.9 TTFT (ms) | 27,956.05 | 28,029.97 | +0.26% ↑ |
| P90 TPOT (ms) | 79.19 | 79.36 | +0.21% ↑ |
| P99.9 TPOT (ms) | 90.89 | 91.58 | +0.76% ↑ |
| P90 ITL (ms) | 50.90 | 50.94 | +0.08% ↑ |
| P99.9 ITL (ms) | 502.28 | 502.30 | +0.00% ↑ |
| Mean E2EL (ms) | 70,337.49 | 70,524.21 | +0.27% ↑ |
| Median E2EL (ms) | 70,634.43 | 70,615.49 | -0.03% ↓ |
| P90 E2EL (ms) | 80,822.88 | 80,846.30 | +0.03% ↑ |
| P99 E2EL (ms) | 98,176.12 | 98,389.37 | +0.22% ↑ |
| P99.9 E2EL (ms) | 102,349.33 | 102,233.14 | -0.11% ↓ |

---

## AI Summary

- The standout improvement is P90 TTFT, down **3.68%** (**1,958.06ms → 1,885.92ms**), while median TTFT was effectively unchanged at **-0.05%**.
- Other TTFT results were slightly weaker: mean increased **0.39%** (**2,111.71ms → 2,119.92ms**), P99 increased **0.40%**, and P99.9 increased **0.26%**.
- TPOT was broadly flat with small regressions at the center of the distribution; mean rose **0.26%** and P99.9 rose **0.76%** (**90.89ms → 91.58ms**), while P99 improved **0.06%**.
- ITL changes were negligible: mean increased **0.26%**, median increased **0.11%**, and P99 improved **0.24%** (**493.10ms → 491.91ms**).
- Throughput was effectively flat to slightly lower: request throughput stayed at **0.89 req/s**, while output and total token throughput each declined **0.31%** and benchmark duration increased **0.31%**.
- End-to-end latency remained essentially flat, with every change below **0.3%**; these movements and most other sub-1% differences are likely within normal run-to-run noise.

---
