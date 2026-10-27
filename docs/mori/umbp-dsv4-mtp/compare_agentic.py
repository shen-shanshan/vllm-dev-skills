"""Print a markdown table comparing AgentX points under /tmp/umbp/agentic/<tag>/."""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/shashen/vllm/umbp_runs/agentic")
TAGS = sys.argv[2:] or ["baseline", "umbp_replay", "umbp_store_all"]


def load(tag):
    out = ROOT / tag
    agg = json.loads(next(out.glob("dsv4_vllm_dep8_c*.json")).read_text())
    eng = dict(
        line.split() for line in (out / "engine_metrics.txt").read_text().splitlines()
    )
    return agg, {k.removeprefix("vllm:"): float(v) for k, v in eng.items()}


rows = [
    ("Requests (profiled)", lambda a, e: f"{a['request_accounting']['records_profiled']}"),
    ("Input tput (tok/s)", lambda a, e: f"{a['request_metrics']['throughput']['input']['tokens_per_second']:,.0f}"),
    ("Output tput (tok/s)", lambda a, e: f"{a['request_metrics']['throughput']['output']['tokens_per_second']:,.0f}"),
    ("Output tput / GPU (tok/s)", lambda a, e: f"{a['request_metrics']['throughput']['output']['tokens_per_second'] / 8:,.1f}"),
    ("QPS (mean)", lambda a, e: f"{a['request_metrics']['qps']['mean']:.2f}"),
    ("TTFT mean / p50 / p90 (s)", lambda a, e: "{mean:.2f} / {p50:.2f} / {p90:.2f}".format(**a['request_metrics']['latency']['ttft'])),
    ("TPOT mean / p50 / p90 (ms)", lambda a, e: " / ".join(f"{a['request_metrics']['latency']['tpot'][k] * 1000:.1f}" for k in ("mean", "p50", "p90"))),
    ("E2E latency mean / p50 / p90 (s)", lambda a, e: "{mean:.1f} / {p50:.1f} / {p90:.1f}".format(**a['request_metrics']['latency']['e2el'])),
    ("Interactivity p50 (tok/s/user)", lambda a, e: f"{a['request_metrics']['latency']['intvty']['p50']:.2f}"),
    ("Server GPU prefix hit (window)", lambda a, e: f"{a['server_metrics']['cache']['gpu_cache_hit_rate']:.1%}"),
    ("Server external (UMBP) hit (window)", lambda a, e: f"{a['server_metrics']['cache']['external_cache_hit_rate']:.1%}"),
    ("Prompt tokens from GPU cache (whole run)", lambda a, e: f"{e['prefix_cache_hits_total'] / e['prompt_tokens_total']:.1%}"),
    ("Prompt tokens from UMBP (whole run)", lambda a, e: f"{e['external_prefix_cache_hits_total'] / e['prompt_tokens_total']:.1%}"),
    ("Draft acceptance (whole run)", lambda a, e: f"{e['spec_decode_num_accepted_tokens_total'] / e['spec_decode_num_draft_tokens_total']:.1%}"),
    ("Mean acceptance length (whole run)", lambda a, e: f"{1 + e['spec_decode_num_accepted_tokens_total'] / e['spec_decode_num_drafts_total']:.2f}"),
    ("Preemptions (whole run)", lambda a, e: f"{e['num_preemptions_total']:.0f}"),
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
