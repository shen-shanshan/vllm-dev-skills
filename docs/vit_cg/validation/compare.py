"""compare.py <qwen3_vl|dsv41> <ref_tag> <tag>... : memory, budget hits, KV size,
and end-to-end greedy outputs of each run against the first tag."""
import json, os, sys, re
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
def load(p):
    d = json.load(open(p)); return d
def logline(p, pat):
    for l in open(p):
        if re.search(pat, l): return l.strip().split("] ",1)[-1]
which = sys.argv[1]; tags = sys.argv[2:]
base = load(f"{OUT}/{which}_{tags[0]}.json")
for t in tags:
    d = load(f"{OUT}/{which}_{t}.json")
    s = d["stats"][0]
    print(f"== {t}: runner={s.get('runner')} graphs={s.get('num_graphs')} "
          f"view_bytes={s.get('view_bytes',0)/2**20:.1f}MiB unique={s.get('unique_storage_bytes',0)/2**20:.1f}MiB")
    print("   hits:", d["hits"][0] and d["hits"][0].get("budget_hits"))
    log = f"{OUT}/{which}_{t}.log"
    print("  ", logline(log, "Graph capturing finished"))
    print("  ", logline(log, "GPU KV cache size"))
    print("  ", logline(log, "Available KV cache memory"))
    if t == tags[0]: continue
    same_tok = sum(a["token_ids"] == b["token_ids"] for a, b in zip(base["results"], d["results"]))
    maxdiff = 0.0
    exact_lp = True
    for a, b in zip(base["results"], d["results"]):
        for sa, sb in zip(a["logprobs"], b["logprobs"]):
            for k, v in sa.items():
                if k in sb:
                    maxdiff = max(maxdiff, abs(v - sb[k]))
                    if v != sb[k]: exact_lp = False
    print(f"   vs {tags[0]}: identical token seqs {same_tok}/{len(d['results'])}, "
          f"logprobs bit-exact={exact_lp}, max |dlogprob|={maxdiff:.3g}")
