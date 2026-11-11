"""Compare B0 (no offload) and S1 (UMBP) AgentX runs across concurrencies.

Usage: python3 compare_concurrency.py <runs_root> <prefix> <conc> [<conc> ...]
Reads <runs_root>/<prefix>_{b0,s1}_dp2_c<conc>/ and prints one markdown table
per metric group, with the S1 vs B0 change for each concurrency.
"""

import csv
import json
import sys
from pathlib import Path

NUM_GPUS = 2


def load(out: Path) -> dict:
    a = json.loads(next(out.glob("dsv41flash_vllm_*_c*.json")).read_text())
    rm = a["request_metrics"]
    tput = rm["throughput"]
    lat = rm["latency"]
    eng = dict(line.split() for line in (out / "engine_metrics.txt").read_text().splitlines())
    eng = {k.removeprefix("vllm:"): float(v) for k, v in eng.items()}
    prompt = eng.get("prompt_tokens_total", 0.0)
    gpu_hit = eng.get("prefix_cache_hits_total", 0.0)
    ext_hit = eng.get("external_prefix_cache_hits_total", 0.0)
    m = {
        "requests": a["request_accounting"]["records_profiled"],
        "total_tput_gpu": (tput["input"]["tokens_per_second"] + tput["output"]["tokens_per_second"]) / NUM_GPUS,
        "output_tput_gpu": tput["output"]["tokens_per_second"] / NUM_GPUS,
        "ttft_p50": lat["ttft"]["p50"],
        "ttft_p90": lat["ttft"]["p90"],
        "tpot_mean": lat["tpot"]["mean"] * 1000,
        "tpot_p90": lat["tpot"]["p90"] * 1000,
        "intvty_p50": lat["intvty"]["p50"],
        "intvty_p90": lat["intvty"]["p90"],
        "e2e_p90": lat["e2el"]["p90"],
        "gpu_hit_pct": 100 * gpu_hit / prompt if prompt else 0.0,
        "umbp_hit_pct": 100 * ext_hit / prompt if prompt else 0.0,
        "computed_pct": 100 * (prompt - gpu_hit - ext_hit) / prompt if prompt else 0.0,
        "preemptions": eng.get("num_preemptions_total", 0.0),
    }
    m.update(step_share(out))
    return m


def step_share(out: Path) -> dict:
    """Share of engine steps with 1024-2048 tokens, summed over DP ranks (engine endpoint)."""
    path = out / "aiperf_artifacts" / "server_metrics_export.csv"
    total = over = 0.0
    with path.open() as f:
        rows = csv.reader(line for line in f if not line.startswith("#"))
        header = next(rows)
        for row in rows:
            r = dict(zip(header, row))
            if r.get("Metric") != "vllm:iteration_tokens_total" or not r["Endpoint"].endswith(":8333"):
                continue
            buckets = dict(kv.split("=") for kv in row[-1].split(";") if "=" in kv)
            le1024 = float(buckets["1024.0"])
            le2048 = float(buckets["2048.0"])
            count = float(r["avg"])
            total += count
            over += le2048 - le1024
    return {"steps_1k_2k_pct": 100 * over / total if total else 0.0}


ROWS = [
    ("Total tput / GPU (tok/s)", "total_tput_gpu", "{:,.0f}", True),
    ("Output tput / GPU (tok/s)", "output_tput_gpu", "{:,.0f}", True),
    ("Requests (profiled)", "requests", "{:,}", True),
    ("TTFT p50 (s)", "ttft_p50", "{:.2f}", False),
    ("TTFT p90 (s)", "ttft_p90", "{:.2f}", False),
    ("TPOT mean (ms)", "tpot_mean", "{:.1f}", False),
    ("TPOT p90 (ms)", "tpot_p90", "{:.1f}", False),
    ("Interactivity p50 (tok/s/user)", "intvty_p50", "{:.2f}", True),
    ("Interactivity p90 (tok/s/user)", "intvty_p90", "{:.2f}", True),
    ("E2E p90 (s)", "e2e_p90", "{:.1f}", False),
    ("Prompt from GPU prefix cache (%)", "gpu_hit_pct", "{:.1f}", None),
    ("Prompt from UMBP (%)", "umbp_hit_pct", "{:.2f}", None),
    ("Prompt computed (%)", "computed_pct", "{:.1f}", False),
    ("Steps with 1024-2048 tokens (%)", "steps_1k_2k_pct", "{:.1f}", None),
    ("Preemptions", "preemptions", "{:,.0f}", None),
]


def main() -> None:
    root, prefix, concs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    data = {c: (load(root / f"{prefix}_b0_dp2_c{c}"), load(root / f"{prefix}_s1_dp2_c{c}")) for c in concs}
    head = "| Metric | " + " | ".join(f"c{c} B0 | c{c} S1 | c{c} Δ" for c in concs) + " |"
    print(head)
    print("|---|" + "---|" * (3 * len(concs)))
    for label, key, fmt, higher_better in ROWS:
        cells = []
        for c in concs:
            b, s = data[c][0][key], data[c][1][key]
            delta = ""
            if higher_better is not None and b:
                delta = f"{100 * (s - b) / b:+.1f}%"
            cells += [fmt.format(b), fmt.format(s), delta]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
