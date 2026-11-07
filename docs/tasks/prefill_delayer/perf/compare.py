"""Markdown table of the prefill delayer perf points under <root>/<tag>/.

    python3 compare.py /tmp/prefill_delayer_perf base cad4 del4 del4_wait
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/prefill_delayer_perf")
TAGS = sys.argv[2:] or ["base", "cad4", "del4", "del4_wait"]
NUM_GPUS = 2


def load(tag):
    out = ROOT / tag
    agg = json.loads(next(out.glob("dsv41flash_vllm_*_c*.json")).read_text())
    eng = dict(
        line.split() for line in (out / "engine_metrics.txt").read_text().splitlines()
    )
    eng = {k.removeprefix("vllm:"): float(v) for k, v in eng.items()}
    # Last decision counters the delayer logged (identical on both ranks).
    decisions = None
    for line in (out / "engine.log").read_text(errors="replace").splitlines():
        if "Prefill delayer decisions: " in line:
            decisions = ast.literal_eval(line.split("Prefill delayer decisions: ")[1])
    # The aggregate JSON stops at p95; AIPerf's own export has p99 (in ms).
    aiperf = json.loads((out / "aiperf_artifacts/profile_export_aiperf.json").read_text())
    agg["ttft_p99"] = aiperf["time_to_first_token"]["p99"] / 1000
    return agg, eng, decisions


def lat(a, key, stats, scale=1.0, fmt="{:.2f}"):
    d = a["request_metrics"]["latency"][key]
    return " / ".join(fmt.format(d[s] * scale) for s in stats)


def holds(d):
    if not d:
        return "—"
    total = sum(d.values())
    held = sum(v for k, v in d.items() if k.startswith("hold_"))
    return f"{held / total:.0%} ({', '.join(f'{k} {v / total:.0%}' for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if k.startswith('hold_'))})"


def fires(d):
    if not d:
        return "—"
    total = sum(d.values())
    return ", ".join(
        f"{k} {v / total:.0%}"
        for k, v in sorted(d.items(), key=lambda kv: -kv[1])
        if not k.startswith("hold_")
    )


rows = [
    ("Requests profiled / errors", lambda a, e, d: f"{a['request_accounting']['records_profiled']} / {a['request_accounting']['records_error_dropped']}"),
    ("Output tput / GPU (tok/s)", lambda a, e, d: f"{a['request_metrics']['throughput']['output']['tokens_per_second'] / NUM_GPUS:,.0f}"),
    ("Total tput / GPU (tok/s)", lambda a, e, d: f"{(a['request_metrics']['throughput']['input']['tokens_per_second'] + a['request_metrics']['throughput']['output']['tokens_per_second']) / NUM_GPUS:,.0f}"),
    ("TTFT mean / p50 / p90 / p95 / p99 (s)", lambda a, e, d: lat(a, "ttft", ("mean", "p50", "p90", "p95")) + f" / {a['ttft_p99']:.2f}"),
    ("TPOT p50 / p90 (ms)", lambda a, e, d: lat(a, "tpot", ("p50", "p90"), 1000, "{:.1f}")),
    ("Interactivity p50 / p90 (tok/s/user)", lambda a, e, d: lat(a, "intvty", ("p50", "p90"))),
    ("E2E-norm interactivity p50 / p90", lambda a, e, d: lat(a, "e2e_norm_intvty", ("p50", "p90"))),
    ("E2E latency mean / p90 (s)", lambda a, e, d: lat(a, "e2el", ("mean", "p90"), fmt="{:.1f}")),
    ("Prompt tokens from GPU cache (run)", lambda a, e, d: f"{e['prefix_cache_hits_total'] / e['prompt_tokens_total']:.1%}"),
    ("Preemptions (run)", lambda a, e, d: f"{e['num_preemptions_total']:.0f}"),
    ("Delayer holds (share of steps)", lambda a, e, d: holds(d)),
    ("Delayer releases", lambda a, e, d: fires(d)),
]

data = {}
for tag in TAGS:
    try:
        data[tag] = load(tag)
    except (StopIteration, FileNotFoundError):
        pass
print("| Metric | " + " | ".join(data) + " |")
print("|---|" + "---|" * len(data))
for name, fn in rows:
    print(f"| {name} | " + " | ".join(fn(*data[t]) for t in data) + " |")
