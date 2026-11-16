"""Per full-M layer (2..19): mega_mhc time, the quant right after each mega_mhc
and after ep_combine (shared gate_up input), for one prefill step.
Usage: python seam_cost.py <trace.json.gz> <step-name>"""
import gzip, json, statistics, sys
d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
a = max((e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"] == sys.argv[2]), key=lambda e: e["dur"])
k = sorted((e for e in ev if e.get("cat") == "kernel" and a["ts"] <= e["ts"] <= a["ts"] + a["dur"]), key=lambda e: e["ts"])
s2 = [i for i, e in enumerate(k) if "megamoe_stage2" in e["name"]]
rows = []
for L in range(1, 19):  # layers 2..19 (skip layer 1: engram)
    seg = k[s2[L] + 1 : s2[L + 1] + 1]
    mhc = [e for e in seg if "mega_mhc_kernel" in e["name"]]
    q_after = 0.0
    for i, e in enumerate(seg[:-1]):
        if e["name"] == "_kernel.kd" and ("mega_mhc_kernel" in seg[i - 1]["name"] or "ep_combine" in seg[i - 1]["name"]):
            q_after += e["dur"]
    rows.append((sum(e["dur"] for e in mhc), q_after))
m = statistics.median(r[0] for r in rows); q = statistics.median(r[1] for r in rows)
print(f"{sys.argv[2]}: per layer median  mega_mhc x2 = {m:6.1f} us, seam quants = {q:5.1f} us, sum = {m + q:6.1f} us")
