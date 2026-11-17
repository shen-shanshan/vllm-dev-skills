"""Per-fusion kernel cost in one DSv4.1 prefill step, full-length layers only.

For each full-length layer window (between consecutive `megamoe_stage2`
kernels, layers 2-19) it measures, as medians over the layers:

- #2 attention seam: Mega-MHC + the `_kernel` quant right after it
  (`fused_wqa_wkv` input), or the Mega-MHC alone once fused.
- #3 FFN seam: Mega-MHC + the `_kernel` quant after `ep_combine`
  (shared expert `gate_up_proj` input), or the Mega-MHC alone once fused.
- #4 shared expert activation: `act_and_mul` + the `_kernel` quant after it
  (`down_proj` input), or `_silu_and_mul_mxfp8_kernel` once fused.

Usage: python fusion_cost.py <trace.json.gz> <step-name>
"""

import gzip
import json
import statistics
import sys


def main() -> None:
    trace = json.load(gzip.open(sys.argv[1]))
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    step = max(
        (
            e
            for e in events
            if e.get("cat") == "gpu_user_annotation" and e["name"] == sys.argv[2]
        ),
        key=lambda e: e["dur"],
    )
    kernels = sorted(
        (
            e
            for e in events
            if e.get("cat") == "kernel"
            and step["ts"] <= e["ts"] <= step["ts"] + step["dur"]
        ),
        key=lambda e: e["ts"],
    )
    stage2 = [i for i, e in enumerate(kernels) if "megamoe_stage2" in e["name"]]
    cols: dict[str, list[float]] = {k: [] for k in ("#2", "#3", "#4")}
    for layer in range(1, 19):
        seg = kernels[stage2[layer] + 1 : stage2[layer + 1] + 1]

        def quant_after(i: int) -> float:
            nxt = seg[i + 1] if i + 1 < len(seg) else None
            return nxt["dur"] if nxt is not None and nxt["name"] == "_kernel.kd" else 0.0

        mhc = [i for i, e in enumerate(seg) if "mega_mhc_kernel" in e["name"]]
        attn, ffn = mhc[0], mhc[1]
        cols["#2"].append(seg[attn]["dur"] + quant_after(attn))
        combine = next(i for i, e in enumerate(seg) if "ep_combine" in e["name"])
        cols["#3"].append(seg[ffn]["dur"] + quant_after(combine))
        act = next(
            i
            for i, e in enumerate(seg)
            if "act_and_mul" in e["name"] or "silu_and_mul_mxfp8" in e["name"]
        )
        cols["#4"].append(
            seg[act]["dur"]
            + (quant_after(act) if "act_and_mul" in seg[act]["name"] else 0.0)
        )
    print(
        f"{sys.argv[2]}: "
        + ", ".join(f"{k} {statistics.median(v):6.1f} us" for k, v in cols.items())
    )


if __name__ == "__main__":
    main()
