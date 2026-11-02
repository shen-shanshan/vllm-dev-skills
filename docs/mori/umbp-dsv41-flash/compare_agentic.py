"""Print a markdown table comparing DeepSeek-V4.1-Flash TP2 AgentX points under <root>/<tag>/."""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/umbp_runs/agentic_dsv41")
NUM_GPUS = 2
TAGS = sys.argv[2:] or ["b0_c128", "s1_c128", "s2_decode_c128", "s3_lazy_c128"]


def load(tag):
    out = ROOT / tag
    agg = json.loads(next(out.glob("dsv41flash_vllm_tp2_c*.json")).read_text())
    eng = dict(
        line.split() for line in (out / "engine_metrics.txt").read_text().splitlines()
    )
    return agg, {k.removeprefix("vllm:"): float(v) for k, v in eng.items()}


rows = [
    ("Requests (profiled)", lambda a, e: f"{a['request_accounting']['records_profiled']}"),
    ("Input tput (tok/s)", lambda a, e: f"{a['request_metrics']['throughput']['input']['tokens_per_second']:,.0f}"),
    ("Output tput (tok/s)", lambda a, e: f"{a['request_metrics']['throughput']['output']['tokens_per_second']:,.0f}"),
    ("Output tput / GPU (tok/s)", lambda a, e: f"{a['request_metrics']['throughput']['output']['tokens_per_second'] / NUM_GPUS:,.1f}"),
    ("Total tput / GPU (tok/s)", lambda a, e: f"{(a['request_metrics']['throughput']['input']['tokens_per_second'] + a['request_metrics']['throughput']['output']['tokens_per_second']) / NUM_GPUS:,.0f}"),
    ("QPS (mean)", lambda a, e: f"{a['request_metrics']['qps']['mean']:.2f}"),
    ("TTFT mean / p50 / p90 / p95 (s)", lambda a, e: "{mean:.2f} / {p50:.2f} / {p90:.2f} / {p95:.2f}".format(**a['request_metrics']['latency']['ttft'])),
    ("TPOT mean / p50 / p90 (ms)", lambda a, e: " / ".join(f"{a['request_metrics']['latency']['tpot'][k] * 1000:.1f}" for k in ("mean", "p50", "p90"))),
    ("E2E latency mean / p50 / p90 (s)", lambda a, e: "{mean:.1f} / {p50:.1f} / {p90:.1f}".format(**a['request_metrics']['latency']['e2el'])),
    ("Interactivity p50 (tok/s/user)", lambda a, e: f"{a['request_metrics']['latency']['intvty']['p50']:.2f}"),
    ("Server GPU prefix hit (window)", lambda a, e: f"{a['server_metrics']['cache']['gpu_cache_hit_rate']:.1%}"),
    ("Server external (UMBP) hit (window)", lambda a, e: f"{a['server_metrics']['cache']['external_cache_hit_rate']:.1%}"),
    ("Prompt tokens from GPU cache (whole run)", lambda a, e: f"{e['prefix_cache_hits_total'] / e['prompt_tokens_total']:.1%}"),
    ("Prompt tokens from UMBP (whole run)", lambda a, e: f"{e['external_prefix_cache_hits_total'] / e['prompt_tokens_total']:.1%}"),
    ("Draft acceptance (whole run)", lambda a, e: f"{e['spec_decode_num_accepted_tokens_total'] / e['spec_decode_num_draft_tokens_total']:.1%}"),
    ("Mean acceptance length (whole run)", lambda a, e: f"{1 + e['spec_decode_num_accepted_tokens_total'] / e['spec_decode_num_drafts_total']:.2f}"),
    ("Prompt tokens computed (whole run)", lambda a, e: f"{1 - (e['prefix_cache_hits_total'] + e['external_prefix_cache_hits_total']) / e['prompt_tokens_total']:.1%}"),
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
