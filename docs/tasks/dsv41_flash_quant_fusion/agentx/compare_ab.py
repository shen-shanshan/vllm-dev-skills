"""Compare two AgentX runs (base vs fused) with the metric set of
vllm-dev-skills/docs/agentx/results/compare_concurrency.py.

Usage: python3 compare_ab.py <base_run_dir> <fused_run_dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/shashen/vllm/vllm-dev-skills/docs/agentx/results")
from compare_concurrency import ROWS, load  # noqa: E402

base, fused = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
print("| Metric | Base | Fused | Δ |")
print("|---|---:|---:|---:|")
for label, key, fmt, _ in ROWS:
    b, f = base[key], fused[key]
    delta = f"{100 * (f - b) / b:+.1f}%" if b else "-"
    print(f"| {label} | {fmt.format(b)} | {fmt.format(f)} | {delta} |")
