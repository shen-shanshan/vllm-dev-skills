#!/usr/bin/env python3
"""Deterministic extractor for vLLM/ATOM torch-profiler decode traces.

Slices one decode step out of a Chrome-trace JSON, aggregates per-kernel
statistics, and emits two artifacts for downstream analysis:
  <prefix>_digest.md   - human-readable roll-up (primary artifact)
  <prefix>_slice.json  - exact names + full statistics

Only aggregates. Semantic interpretation (grouping kernels into logical
operations, comparing engines) is done by the consumer, not here.

Stdlib-only. Optional ijson support for very large files.
"""

import argparse
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

EXEC_RE = re.compile(r"^execute_context_")
NAME_CUT = 97  # truncation length for display rows (full names in dictionary/JSON)


def eprint(msg):
    print(msg, file=sys.stderr)


def load_events(path):
    """Return (top_level_dict, events_list). Supports .json and .json.gz."""
    opener = gzip.open if path.endswith(".gz") else open
    size = os.path.getsize(path)
    if size <= 300 * 1024 * 1024:
        with opener(path, "rb") as f:
            data = json.load(f)
        return data, extract_events(data)
    try:
        import ijson  # optional, only needed for huge files
    except ImportError:
        eprint(f"WARNING: file is {size/1e6:.0f} MB but ijson is not installed; "
               "falling back to json.load (may use a lot of memory).")
        with opener(path, "rb") as f:
            data = json.load(f)
        return data, extract_events(data)
    with opener(path, "rb") as f:
        events = list(ijson.items(f, "traceEvents.item"))
        top = {}
        for key in ("schemaVersion", "baseTimeNanoseconds", "displayTimeUnit",
                    "vllm_version", "vllm_version_tuple", "roctracer_version",
                    "rocprofiler-sdk_version", "hip_runtime_version",
                    "hip_driver_version", "host_name", "traceName"):
            try:
                with opener(path, "rb") as g:
                    top[key] = next(ijson.items(g, key), None)
            except Exception:
                top[key] = None
        return top, events


def extract_events(data):
    """traceEvents may be a list or a dict keyed by 'pid,tid'."""
    te = data.get("traceEvents")
    if isinstance(te, list):
        return te
    if isinstance(te, dict):
        flat = []
        for v in te.values():
            flat.extend(v)
        return flat
    return []


def detect_engine(top, forced):
    if forced:
        return forced
    if "vllm_version" in top:
        return "vllm"
    if "roctracer_version" in top:
        return "atom"
    return None


def fmt_us(v):
    return "n/a" if v is None else f"{v:.1f}"


def short_name(name):
    if len(name) <= NAME_CUT:
        return name
    tail = ".kd" if name.endswith(".kd") else ""
    return name[:NAME_CUT - len(tail)] + "..." + tail


def annotation_windows(events, channel):
    """Return sorted [(ts, dur)] of execute_context annotations in channel."""
    anns = [e for e in events
            if e.get("cat") == channel and e.get("ph") == "X"
            and EXEC_RE.match(e.get("name", ""))]
    anns.sort(key=lambda e: e["ts"])
    return [(e["ts"], e.get("dur", 0.0)) for e in anns]


def all_annotation_names(events):
    out = {}
    for channel in ("user_annotation", "gpu_user_annotation"):
        names = Counter(e["name"] for e in events if e.get("cat") == channel)
        out[channel] = dict(sorted(names.items()))
    return out


def kernel_events_by_pid(events):
    kernels = [e for e in events
               if e.get("cat") == "kernel" and e.get("ph") == "X"]
    by_pid = defaultdict(list)
    for k in kernels:
        by_pid[k["pid"]].append(k)
    return by_pid


def stream_of(e):
    s = (e.get("args") or {}).get("stream")
    if s is None:
        s = e.get("tid")
    return s


def build_windows(anns, mode, kernels):
    """Return list of (start, end, note). kernels = list sorted by ts."""
    n = len(anns)
    gaps = [anns[i + 1][0] - anns[i][0] for i in range(n - 1)]
    med_gap = statistics.median(gaps) if gaps else 0.0
    windows = []
    if mode == "gpu-annot":
        for i in range(n - 1):
            windows.append((anns[i][0], anns[i + 1][0], ""))
        trailing_end = anns[-1][0] + max(anns[-1][1], med_gap)
        windows.append((anns[-1][0], trailing_end, "trailing"))
    else:  # cpu-annot: per-window lead shift, approximate
        leads = []
        for i in range(n):
            ts_i = anns[i][0]
            after = [k["ts"] for k in kernels
                     if ts_i < k["ts"] <= ts_i + 20000]
            lead = min(after) - ts_i if after else 0.0
            leads.append(max(0.0, lead))
        for i in range(n - 1):
            windows.append((anns[i][0] + leads[i],
                            anns[i + 1][0] + leads[i], "approx"))
        trailing_end = anns[-1][0] + leads[-1] + max(anns[-1][1], med_gap)
        windows.append((anns[-1][0] + leads[-1], trailing_end, "trailing+approx"))
    return windows


def window_kernels(kernels, start, end):
    return [k for k in kernels if start <= k["ts"] < end]


def rollup(ks, window_start):
    names = defaultdict(list)
    for k in ks:
        names[k["name"]].append(k)
    rows = []
    for name, evs in names.items():
        durs = sorted(e.get("dur", 0.0) for e in evs)
        rels = sorted(e["ts"] - window_start for e in evs)
        rows.append({
            "name": name,
            "count": len(evs),
            "sum_us": sum(durs),
            "mean_us": sum(durs) / len(durs),
            "median_us": statistics.median(durs),
            "min_us": durs[0],
            "max_us": durs[-1],
            "streams": sorted({stream_of(e) for e in evs}, key=str),
            "first_rel_us": rels[0],
            "last_rel_us": rels[-1],
        })
    rows.sort(key=lambda r: r["first_rel_us"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--engine", choices=["auto", "vllm", "atom"], default="auto")
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "outputs"))
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--step-index", default="middle")
    ap.add_argument("--occurrence", default="middle")
    ap.add_argument("--bracket", choices=["gpu-annot", "cpu-annot"],
                    default="gpu-annot")
    ap.add_argument("--layer-count", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    top, events = load_events(args.trace)
    engine = detect_engine(top, None if args.engine == "auto" else args.engine)
    if engine is None:
        eprint("ERROR: cannot auto-detect engine (no vllm_version / "
               "roctracer_version top-level keys). Re-run with --engine.")
        sys.exit(1)

    prefix = args.prefix or os.path.basename(args.trace).replace(".json.gz", "").replace(".json", "")
    os.makedirs(args.output_dir, exist_ok=True)
    digest_path = os.path.join(args.output_dir, f"{prefix}_digest.md")
    slice_path = os.path.join(args.output_dir, f"{prefix}_slice.json")

    # --- basics ---
    by_pid = kernel_events_by_pid(events)
    if not by_pid:
        eprint("ERROR: no kernel events found in trace.")
        sys.exit(1)
    rank_pid = max(by_pid, key=lambda p: len(by_pid[p]))
    kernels = sorted(by_pid[rank_pid], key=lambda k: k["ts"])
    other_pids = {p: len(v) for p, v in by_pid.items() if p != rank_pid}

    ph_counts = Counter(e.get("ph") for e in events)
    cat_counts = Counter(e.get("cat") for e in events)
    versions = {k: top.get(k) for k in
                ("vllm_version", "vllm_version_tuple", "roctracer_version",
                 "rocprofiler-sdk_version", "hip_runtime_version",
                 "hip_driver_version", "schemaVersion")}
    devices = top.get("deviceProperties") or []
    dist = top.get("distributedInfo")
    dist_str = json.dumps(dist)[:300] if dist is not None else "n/a"

    # --- annotations & windows ---
    ann_names = all_annotation_names(events)
    gpu_anns = annotation_windows(events, "gpu_user_annotation")
    cpu_anns = annotation_windows(events, "user_annotation")
    mode = args.bracket
    warnings = []
    if mode == "gpu-annot" and not gpu_anns:
        if cpu_anns:
            mode = "cpu-annot"
            warnings.append("gpu channel has no execute_context annotations; "
                            "fell back to CPU channel (approximate).")
        else:
            eprint("ERROR: no execute_context user annotations found in either "
                   "channel. Capture decode with torch profiler step "
                   "annotations enabled, then re-run.")
            sys.exit(2)
    if mode == "cpu-annot":
        warnings.append("STEP ATTRIBUTION IS APPROXIMATE: CPU annotations lead "
                        "the GPU by several ms; windows are lead-shifted "
                        "heuristics. Prefer gpu-annot brackets when available.")
    source_anns = gpu_anns if mode == "gpu-annot" else cpu_anns
    if not source_anns:
        eprint("ERROR: no execute_context annotations in the chosen bracket channel.")
        sys.exit(2)

    windows = build_windows(source_anns, mode, kernels)
    n_windows = len(windows)
    per_window = [window_kernels(kernels, ws, we) for (ws, we, _n) in windows]

    # --- replay stability (over fully-bracketed steady-state windows only;
    # window 0 may hold warmup/init kernels and is compared separately) ---
    n_full = n_windows - 1  # last window is the trailing window by construction
    full_counts = [Counter(k["name"] for k in per_window[i]) for i in range(n_full)]
    ref_counts = full_counts[1] if n_full >= 2 else (full_counts[0] if n_full == 1 else Counter())
    replay_diffs = {}
    for i in range(2 if n_full >= 2 else 1, n_full):
        c = full_counts[i]
        if c != ref_counts:
            replay_diffs[i] = {n: (ref_counts.get(n, 0), c.get(n, 0))
                               for n in set(ref_counts) | set(c)
                               if ref_counts.get(n, 0) != c.get(n, 0)}
    if n_full >= 2 and full_counts[0] != ref_counts:
        warnings.append("window 0 differs from steady-state windows "
                        "(warmup/init kernels); excluded from replay check.")
    replay_stable = not replay_diffs

    # --- step selection (trailing window excluded by default) ---
    if args.step_index == "middle":
        step = n_full // 2 if n_full >= 1 else 0
        if n_full == 1:
            warnings.append("only one fully-bracketed decode step; using it.")
    else:
        step = max(0, min(int(args.step_index), n_windows - 1))
        if step == n_windows - 1:
            warnings.append("explicitly selected the trailing window; its end "
                            "is extrapolated, not bracket-verified.")
    if replay_diffs:
        busy = [sum(k.get("dur", 0.0) for k in per_window[i]) for i in range(n_full)]
        med = statistics.median(busy)
        step = min(range(n_full), key=lambda i: abs(busy[i] - med))
        warnings.append(f"replay unstable; chose window {step} by median busy time.")

    ws, we, wnote = windows[step]
    chosen_kernels = per_window[step]
    wall_us = we - ws
    busy_us = sum(k.get("dur", 0.0) for k in chosen_kernels)
    busy_pct = busy_us / wall_us * 100 if wall_us else 0.0
    if not 50.0 <= busy_pct <= 150.0 and chosen_kernels:
        warnings.append(f"busy/wall = {busy_pct:.1f}% is outside [50%, 150%]; "
                        "suspicious timestamp units — verify µs assumption.")

    # per-stream roll-up
    stream_map = defaultdict(lambda: [0, 0.0])
    for k in chosen_kernels:
        s = stream_of(k)
        stream_map[s][0] += 1
        stream_map[s][1] += k.get("dur", 0.0)

    # --- per-name rollup & occurrence slice ---
    rows = rollup(chosen_kernels, ws)
    try:
        occ_int = int(args.occurrence)
    except ValueError:
        occ_int = None

    # per-window per-name occurrence lists: occurrence k within a window
    # corresponds to layer k of that kernel's per-layer family
    occ_by_window = defaultdict(dict)
    for i, wks in enumerate(per_window):
        by_name = defaultdict(list)
        for k in wks:
            by_name[k["name"]].append(k)
        for name, evs in by_name.items():
            evs.sort(key=lambda e: e["ts"])
            occ_by_window[name][i] = evs

    mids = {}
    for r in rows:
        name, c = r["name"], r["count"]
        if c < 2:
            continue
        occ = (c // 2) if occ_int is None else occ_int
        occ = max(0, min(occ, c - 1))
        ev = occ_by_window[name][step][occ]
        dur_by_step = []
        for i in range(n_windows):
            evs = occ_by_window[name].get(i, [])
            dur_by_step.append(evs[occ].get("dur", 0.0) if len(evs) > occ else None)
        valid = [d for d in dur_by_step if d is not None]
        mids[name] = {
            "count": c,
            "occ_index": occ,
            "ts_rel_us": ev["ts"] - ws,
            "dur_us": ev.get("dur", 0.0),
            "stream": stream_of(ev),
            "dur_by_step_us": dur_by_step,
            "dur_median_us": statistics.median(valid) if valid else None,
            "dur_min_us": min(valid) if valid else None,
            "dur_max_us": max(valid) if valid else None,
        }

    once_kernels = [{"name": k["name"], "ts_rel_us": k["ts"] - ws,
                     "dur_us": k.get("dur", 0.0), "stream": stream_of(k)}
                    for k in sorted(chosen_kernels, key=lambda k: k["ts"])
                    if next(r["count"] for r in rows if r["name"] == k["name"]) == 1]

    # short ids, by first appearance in the step (ts order)
    id_map = {}
    for i, r in enumerate(rows, 1):
        id_map[r["name"]] = f"k{i}"

    # --- digest markdown ---
    L = []
    A = L.append
    A(f"# {engine.upper()} Trace Digest — {os.path.basename(args.trace)}")
    A("")
    A("## 1. Trace basics")
    A("")
    A("| Field | Value |")
    A("|---|---|")
    A(f"| Engine | {engine} |")
    for k, v in versions.items():
        if v is not None:
            A(f"| {k} | {v} |")
    dev_names = [f"{d.get('name','?')}(sm{d.get('computeMajor','?')}.{d.get('computeMinor','?')}×{d.get('numSms','?')})"
                 for d in devices[:8]]
    A(f"| deviceProperties | {len(devices)} GPU(s): {', '.join(dev_names) if dev_names else 'n/a'} |")
    A(f"| distributedInfo | {dist_str} |")
    A(f"| baseTimeNanoseconds | {top.get('baseTimeNanoseconds')} |")
    A(f"| Events (ph) | {dict(sorted(ph_counts.items(), key=lambda kv: str(kv[0])))} |")
    A(f"| Events (cat) | {dict(sorted(cat_counts.items(), key=lambda kv: str(kv[0])))} |")
    A(f"| Kernel events (all) | {len(kernels)} on pid {rank_pid} |")
    if other_pids:
        A(f"| Other kernel pids | {other_pids} (ignored; {rank_pid} = captured rank) |")
    A("")
    A("## 2. Step structure")
    A("")
    for channel, names in ann_names.items():
        A(f"- {channel} annotation names: {list(names.items())}")
    A("")
    A("| Win | start_rel_us | end_rel_us | wall_us | kernels | note |")
    A("|---|---|---|---|---|---|")
    base0 = windows[0][0]
    for i, ((w_s, w_e, note), wks) in enumerate(zip(windows, per_window)):
        mark = " **<-- CHOSEN**" if i == step else ""
        A(f"| {i} | {w_s - base0:.1f} | {w_e - base0:.1f} | {w_e - w_s:.1f} | "
          f"{len(wks)}{mark} | {note} |")
    A("")
    A(f"- windows: {n_windows} | chosen step: {step} | bracket mode: {mode}")
    A(f"- replay_stable: {replay_stable}")
    if replay_diffs:
        A(f"- replay diffs: {json.dumps(replay_diffs, indent=2)}")
    A(f"- occurrence rule: {args.occurrence}"
      + " (per kernel name, occurrence index in ts order)")
    A("")
    A("## 3. Chosen window summary")
    A("")
    A(f"- wall_us: {wall_us:.1f} | kernels: {len(chosen_kernels)} | "
      f"distinct names: {len(rows)} | busy_us: {busy_us:.1f} | "
      f"busy/wall: {busy_pct:.1f}%")
    if args.layer_count:
        A(f"- layer-count hint: {args.layer_count} (see per_layer column in §4)")
    A("")
    A("| stream | events | busy_us | % of busy |")
    A("|---|---|---|---|")
    for s in sorted(stream_map, key=str):
        ev, bu = stream_map[s]
        A(f"| {s} | {ev} | {bu:.1f} | {bu / busy_us * 100:.1f}% |")
    A("")
    A("## 4. Per-kernel roll-up (chosen window)")
    A("")
    hdr = ("| id | kernel | C | sum_us | mean_us | median_us | min_us | max_us | "
           "pct_busy | streams | first_rel_us | last_rel_us |")
    if args.layer_count:
        hdr = hdr[:-1] + " | per_layer |"
    A(hdr)
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        row = (f"| {id_map[r['name']]} | {short_name(r['name'])} | {r['count']} | "
               f"{r['sum_us']:.1f} | {r['mean_us']:.1f} | {r['median_us']:.1f} | "
               f"{r['min_us']:.1f} | {r['max_us']:.1f} | "
               f"{r['sum_us'] / busy_us * 100:.2f} | "
               f"{','.join(str(s) for s in r['streams'])} | "
               f"{r['first_rel_us']:.1f} | {r['last_rel_us']:.1f}")
        if args.layer_count:
            row += f" | {r['count'] / args.layer_count:.2f}"
        A(row + " |")
    A("")
    A("## 5. Middle-occurrence slice (chosen window, sorted by ts_rel)")
    A("")
    A("| id | kernel | C | occ | ts_rel_us | dur_us | dur_median_across_steps_us | stream |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["count"] < 2:
            continue
        m = mids[r["name"]]
        A(f"| {id_map[r['name']]} | {short_name(r['name'])} | {m['count']} | "
          f"{m['occ_index']} | {m['ts_rel_us']:.1f} | {m['dur_us']:.1f} | "
          f"{fmt_us(m['dur_median_us'])} | {m['stream']} |")
    A("")
    A("## 6. Non-repeating kernels (C==1 in chosen window)")
    A("")
    if once_kernels:
        A("| name | ts_rel_us | dur_us | stream |")
        A("|---|---|---|---|")
        for k in once_kernels:
            A(f"| {short_name(k['name'])} | {k['ts_rel_us']:.1f} | "
              f"{k['dur_us']:.1f} | {k['stream']} |")
    else:
        A("(none)")
    A("")
    A("## 7. Name dictionary (id -> full kernel name)")
    A("")
    A("```")
    for name, sid in sorted(id_map.items(), key=lambda kv: int(kv[1][1:])):
        A(f"{sid} = {name}")
    A("```")
    A("")
    A("## 8. Notes / warnings")
    A("")
    if warnings:
        for w in warnings:
            A(f"- WARNING: {w}")
    else:
        A("(none)")
    A("")
    A(f"Full data: `{os.path.basename(slice_path)}` (rollup / middle_slice / "
      "once_kernels / windows / streams / annotations)")

    with open(digest_path, "w") as f:
        f.write("\n".join(L))

    slice_data = {
        "engine": engine,
        "file": args.trace,
        "trace_keys": sorted(top.keys()),
        "versions": {k: v for k, v in versions.items() if v is not None},
        "base_time_ns": top.get("baseTimeNanoseconds"),
        "bracket_mode": mode,
        "n_windows": n_windows,
        "chosen_step": step,
        "window_us": [ws, we],
        "replay_stable": replay_stable,
        "replay_diffs": replay_diffs,
        "kernel_pid": rank_pid,
        "windows": [{"index": i, "start_us": w_s, "end_us": w_e,
                     "kernels": len(wks), "note": note}
                    for i, ((w_s, w_e, note), wks) in enumerate(zip(windows, per_window))],
        "streams": {str(s): {"events": ev, "busy_us": bu}
                    for s, (ev, bu) in stream_map.items()},
        "rollup": {r["name"]: {k: v for k, v in r.items() if k != "name"}
                   for r in rows},
        "middle_slice": mids,
        "once_kernels": once_kernels,
        "annotations": {
            "gpu_exec_us": [[t, d] for t, d in gpu_anns],
            "cpu_exec_us": [[t, d] for t, d in cpu_anns],
            "other_gpu_annotations": list(ann_names.get("gpu_user_annotation", {}).keys()),
            "other_cpu_annotations": list(ann_names.get("user_annotation", {}).keys()),
        },
    }
    with open(slice_path, "w") as f:
        json.dump(slice_data, f, indent=1)

    print(f"digest: {digest_path}")
    print(f"slice:  {slice_path}")
    print(f"engine: {engine} | steps: {n_windows} | chosen_step: {step} | "
          f"window_us: {ws:.1f}..{we:.1f} | kernels: {len(chosen_kernels)} | "
          f"distinct: {len(rows)} | replay_stable: {replay_stable} | "
          f"busy_pct: {busy_pct:.1f}")


if __name__ == "__main__":
    main()
