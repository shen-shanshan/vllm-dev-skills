"""Per-minute completed requests / tokens of an AgentX run, from AIPerf's
profile_export.jsonl, to spot stalls. Usage: python3 timeline.py <run_dir>..."""

import collections
import json
import sys
from pathlib import Path

for run in sys.argv[1:]:
    recs = [json.loads(l) for l in (Path(run) / "aiperf_artifacts/profile_export.jsonl").open()]
    start = min(r["metadata"]["request_start_ns"] for r in recs)
    per_min = collections.defaultdict(lambda: [0, 0, 0])
    for r in recs:
        m = r["metadata"]
        minute = int((m["request_end_ns"] - start) / 60e9)
        cell = per_min[minute]
        cell[0] += 1
        cell[1] += r["metrics"].get("input_sequence_length", {}).get("value", 0)
        cell[2] += r["metrics"].get("output_sequence_length", {}).get("value", 0)
    phases = collections.Counter(r["metadata"]["benchmark_phase"] for r in recs)
    print(f"== {Path(run).name}  records {len(recs)} {dict(phases)}")
    for minute in range(max(per_min) + 1):
        n, isl, osl = per_min.get(minute, [0, 0, 0])
        print(f"  min {minute:3d}: {n:4d} req  {isl / 60 / 2:10,.0f} in tok/s/GPU  {osl / 60 / 2:8,.0f} out tok/s/GPU")
