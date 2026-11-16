"""Summarize one prefill step of a DSv4.1-Flash ROCm torch-profiler trace.

Usage: python analyze_prefill.py <trace.json.gz> [step-annotation-name]

Without a step name, the longest `execute_context_*` step whose context token
count is >= 16000 is used. Prints the kernel breakdown, the standalone MXFP8
quant kernels (`_kernel.kd`) with their producer and consumer kernels, and the
per-layer quant count / time.
"""

import collections
import gzip
import json
import re
import statistics
import sys


def short(name: str) -> str:
    name = re.sub(r"^void ", "", name).replace(".kd", "")
    for pat, label in [
        ("flydsl_mxpsh", "FLYDSL_GEMM"),
        ("flydsl_bpreshuffle_mxfp8", "FLYDSL_GEMM"),
        ("bmm_a8w8_mxscale", "wo_a_bmm(aiter)"),
        ("mega_mhc_kernel", "aiter mega_mhc"),
        ("mhc_post_kernel", "aiter mhc_post"),
        ("act_and_mul", "act_and_mul"),
        ("megamoe_stage1", "megamoe_stage1"),
        ("megamoe_stage2", "megamoe_stage2"),
        ("megamoe_prepare", "megamoe_prepare"),
        ("_sparse_mla", "_sparse_mla*"),
        ("Cijk_", "hipBLASLt"),
    ]:
        if pat in name:
            return label
    return re.split(r"[<(]", name)[0][:60]


def main() -> None:
    trace = json.load(gzip.open(sys.argv[1]))
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    steps = [
        e
        for e in events
        if e.get("cat") == "gpu_user_annotation"
        and e["name"].startswith("execute_context")
    ]
    if len(sys.argv) > 2:
        step = max((e for e in steps if e["name"] == sys.argv[2]), key=lambda e: e["dur"])
    else:
        full = [
            e
            for e in steps
            if int(re.match(r"execute_context_\d+\((\d+)\)", e["name"]).group(1))
            >= 16000
        ]
        # Skip a first step stretched by warmup: take the shortest full step.
        step = min(full, key=lambda e: e["dur"] if e["dur"] > 1e5 else 1e12)
    t0, t1 = step["ts"], step["ts"] + step["dur"]
    kernels = sorted(
        (
            e
            for e in events
            if e.get("cat") == "kernel" and t0 <= e["ts"] <= t1
        ),
        key=lambda e: e["ts"],
    )
    total = sum(e["dur"] for e in kernels)
    print(f"step {step['name']}: wall {step['dur'] / 1e3:.1f} ms, "
          f"kernel-sum {total / 1e3:.1f} ms, kernels {len(kernels)}")

    by_name: dict[str, float] = collections.Counter()
    count: dict[str, int] = collections.Counter()
    for e in kernels:
        by_name[short(e["name"])] += e["dur"]
        count[short(e["name"])] += 1
    print("\n== top kernels")
    for name, dur in by_name.most_common(15):
        print(f"{dur / 1e3:8.2f} ms {100 * dur / total:5.1f}% x{count[name]:4d}  {name}")

    print("\n== standalone MXFP8 quant (_kernel.kd): producer -> _kernel -> consumer")
    by_stream = collections.defaultdict(list)
    for e in kernels:
        by_stream[e["tid"]].append(e)
    sites = collections.defaultdict(list)
    for seq in by_stream.values():
        for i, e in enumerate(seq):
            if e["name"] != "_kernel.kd":
                continue
            prod = short(seq[i - 1]["name"]) if i else "-"
            cons = seq[i + 1] if i + 1 < len(seq) else None
            sites[(prod, short(cons["name"]) if cons else "-")].append(
                (e["dur"], cons["dur"] if cons else 0.0)
            )
    for (prod, cons), v in sorted(sites.items(), key=lambda kv: -sum(a for a, _ in kv[1])):
        print(f"x{len(v):3d} quant med {statistics.median(a for a, _ in v):6.1f} us "
              f"sum {sum(a for a, _ in v) / 1e3:5.2f} ms | consumer med "
              f"{statistics.median(b for _, b in v):7.1f} us | {prod} -> {cons}")
    quant = [e["dur"] for e in kernels if e["name"] == "_kernel.kd"]
    print(f"total _kernel: x{len(quant)} {sum(quant) / 1e3:.2f} ms "
          f"({100 * sum(quant) / total:.2f}% of kernel time)")

    print("\n== per layer (window between megamoe_stage2 kernels)")
    stage2 = [i for i, e in enumerate(kernels) if "megamoe_stage2" in e["name"]]
    for layer, (a, b) in enumerate(zip(stage2, stage2[1:]), start=1):
        seg = kernels[a + 1 : b + 1]
        q = [e["dur"] for e in seg if e["name"] == "_kernel.kd"]
        mhc = [e["dur"] for e in seg if "mega_mhc_kernel" in e["name"]]
        print(f"L{layer:2d} quant x{len(q)} {sum(q):6.0f} us | mega_mhc x{len(mhc)} "
              f"{sum(mhc):6.0f} us")


if __name__ == "__main__":
    main()
