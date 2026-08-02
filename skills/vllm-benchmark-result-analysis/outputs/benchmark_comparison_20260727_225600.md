# vLLM Benchmark Comparison Report

> Generated: 2026-07-27 22:56
> vLLM Benchmark Comparison

---

## Performance Metrics

| Metric | Before this PR | After this PR | Comparison |
| :----- | :------------- | :------------ | :--------- |
| **Throughput** | | | |
| Request throughput (req/s) | 0.87 | 0.89 | +2.30% ↑ |
| Output token throughput (tok/s) | 803.73 | 819.69 | +1.99% ↑ |
| Total Token throughput (tok/s) | 7,244.62 | 7,388.54 | +1.99% ↑ |
| **Latency** | | | |
| Benchmark duration (s) | 733.99 | 719.69 | -1.95% ↓ |
| Mean TTFT (ms) | 2,173.39 | 2,119.92 | -2.46% ↓ |
| Median TTFT (ms) | 603.66 | 586.98 | -2.76% ↓ |
| P99 TTFT (ms) | 25,710.82 | 25,705.54 | -0.02% ↓ |
| Mean TPOT (ms) | 75.71 | 74.17 | -2.03% ↓ |
| Median TPOT (ms) | 76.56 | 75.64 | -1.20% ↓ |
| P99 TPOT (ms) | 95.24 | 86.74 | -8.92% ↓ |
| Mean ITL (ms) | 75.84 | 74.29 | -2.04% ↓ |
| Median ITL (ms) | 46.69 | 46.92 | +0.49% ↑ |
| P99 ITL (ms) | 501.88 | 491.91 | -1.99% ↓ |
| **Other** | | | |
| P90 TTFT (ms) | 2,283.44 | 1,885.92 | -17.41% ↓ |
| P99.9 TTFT (ms) | 29,860.12 | 28,029.97 | -6.13% ↓ |
| P90 TPOT (ms) | 82.46 | 79.36 | -3.76% ↓ |
| P99.9 TPOT (ms) | 100.09 | 91.58 | -8.50% ↓ |
| P90 ITL (ms) | 50.68 | 50.94 | +0.51% ↑ |
| P99.9 ITL (ms) | 855.14 | 502.30 | -41.26% ↓ |
| Mean E2EL (ms) | 72,004.30 | 70,524.21 | -2.06% ↓ |
| Median E2EL (ms) | 71,954.11 | 70,615.49 | -1.86% ↓ |
| P90 E2EL (ms) | 84,177.75 | 80,846.30 | -3.96% ↓ |
| P99 E2EL (ms) | 108,286.85 | 98,389.37 | -9.14% ↓ |
| P99.9 E2EL (ms) | 111,884.79 | 102,233.14 | -8.63% ↓ |

---

## AI Summary

- The standout improvement is P99.9 ITL, down **41.26%** (**855.14ms → 502.30ms**), indicating substantially fewer extreme inter-token latency spikes.
- TTFT improved across most of the distribution: mean fell **2.46%** (**2,173.39ms → 2,119.92ms**), P90 fell **17.41%** (**2,283.44ms → 1,885.92ms**), and P99.9 fell **6.13%**; P99 was effectively flat at **-0.02%**.
- Token-generation latency also improved: mean TPOT dropped **2.03%** (**75.71ms → 74.17ms**) and P99 TPOT dropped **8.92%** (**95.24ms → 86.74ms**).
- Serving throughput improved consistently, with request throughput up **2.30%** and output and total token throughput both up **1.99%**; benchmark duration decreased **1.95%**.
- End-to-end latency improved from mean through the tail, including P99 **-9.14%** (**108,286.85ms → 98,389.37ms**) and P99.9 **-8.63%** (**111,884.79ms → 102,233.14ms**).
- Median and P90 ITL increased slightly (**+0.49%** and **+0.51%**), but these sub-1% changes are likely within normal run-to-run noise given the strong mean and tail improvements.

---
