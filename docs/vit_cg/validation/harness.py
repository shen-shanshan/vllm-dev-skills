"""A/B harness for shared encoder CUDA graph capture buffers.

Usage: python harness.py <model> <out.json> [eager_vit]
  model: qwen3_vl | dsv41 | step3_vl | gemma4 | deepseek_ocr

After generation, each worker captures a second EncoderCudaGraphManager with
buffer sharing disabled and compares it with the live (shared) manager:
  - "batch" A/B: every recorded batch replayed alone, in five orders.
  - "execute" A/B: every recorded execute() call rerun whole, so a later batch
    overwriting an earlier batch's not-yet-returned outputs shows up.
"""

import json
import os
import random
import sys
from collections import Counter

from PIL import Image, ImageOps

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def _find_manager(worker):
    runner = worker.model_runner
    mgr = getattr(runner, "encoder_cudagraph_manager", None)
    if mgr is None and hasattr(runner, "model_state"):
        enc = getattr(runner.model_state, "encoder_runner", None)
        mgr = getattr(enc, "cudagraph_manager", None)
    return runner, mgr


def _clone_kwargs(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_kwargs(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_kwargs(v) for v in obj)
    return obj


def rpc_stats(worker):
    import torch

    runner, mgr = _find_manager(worker)
    if mgr is None:
        return {"runner": type(runner).__module__, "manager": None}
    storages = {}
    naive = 0
    graphs = []
    for path, gset in mgr.budget_graphs.items():
        for key, meta in gset.items():
            for b in [*meta.input_buffers.values(), meta.output_buffer]:
                if not isinstance(b, torch.Tensor) or b.device.type == "cpu":
                    continue
                naive += b.numel() * b.element_size()
                st = b.untyped_storage()
                storages[st.data_ptr()] = st.nbytes()
            graphs.append({"path": path, "key": str(key)})
    return {
        "runner": type(runner).__module__,
        "budgets": dict(mgr.path_token_budgets),
        "capture_axes": [list(map(str, a)) for a in mgr._capture_axes],
        "max_batch_size": mgr.max_batch_size,
        "num_graphs": len(graphs),
        "graphs": graphs,
        "aliased_slots": sorted(map(str, getattr(mgr, "_aliased_slots", ()))),
        "view_bytes": naive,
        "unique_storage_bytes": sum(storages.values()),
    }


def rpc_install_recorder(worker):
    """Count budget hits and record the inputs of every batch and execute()."""
    _, mgr = _find_manager(worker)
    if mgr is None:
        return False
    mgr._hit_budgets = Counter()
    mgr._recorded = []
    mgr._recorded_execs = []
    run_budget_graph = mgr._run_budget_graph
    execute = mgr.execute

    def recording_run_budget_graph(mm_kwargs, token_budget, path="default", axis_keys=()):
        mgr._hit_budgets[f"{path}:{token_budget}"] += 1
        mgr._recorded.append((_clone_kwargs(mm_kwargs), token_budget, path, axis_keys))
        return run_budget_graph(mm_kwargs, token_budget, path=path, axis_keys=axis_keys)

    def recording_execute(mm_kwargs):
        mgr._recorded_execs.append(_clone_kwargs(mm_kwargs))
        return execute(mm_kwargs)

    mgr._run_budget_graph = recording_run_budget_graph
    mgr.execute = recording_execute
    return True


def rpc_hits(worker):
    _, mgr = _find_manager(worker)
    if mgr is None:
        return None
    return {
        "budget_hits": dict(getattr(mgr, "_hit_budgets", {})),
        "graph_hits": mgr.graph_hits,
        "graph_misses": mgr.graph_misses,
    }


def rpc_ab_check(worker):
    import torch

    with torch.inference_mode():
        return _ab_check(worker)


def _ab_check(worker):
    import torch

    from vllm.distributed.parallel_state import graph_capture
    from vllm.platforms import current_platform
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    _, mgr = _find_manager(worker)
    if mgr is None:
        return None
    # Drop the instance-level recorders so the class methods run below.
    for name in ("_run_budget_graph", "execute"):
        mgr.__dict__.pop(name, None)

    class _NeverShared(dict):
        def get(self, *args, **kwargs):
            return None

    def capture_unshared():
        m = EncoderCudaGraphManager(mgr.vllm_config, mgr.device, mgr.dtype, mgr.model)
        m._get_shared_buffer = lambda *a, **k: None
        # PR #65084's implementation keeps its shared buffers in these dicts.
        for name in ("_shared_input_buffers", "_shared_output_buffers"):
            if hasattr(m, name):
                setattr(m, name, _NeverShared())
        with graph_capture(device=mgr.device):
            m.capture(graph_pool=current_platform.graph_pool_handle())
            torch.accelerator.synchronize()
        return m

    base = capture_unshared()

    # Batch A/B: each recorded batch alone, in several orders across budgets.
    recs = mgr._recorded
    order = list(range(len(recs)))
    orders = {
        "forward": order,
        "reverse": order[::-1],
        "by_budget_desc": sorted(order, key=lambda i: -recs[i][1]),
        "by_budget_asc": sorted(order, key=lambda i: recs[i][1]),
        "shuffled": random.Random(0).sample(order, len(order)),
    }

    def path_tokens(kw, path):
        return sum(spec.get_path_output_tokens(path) for spec in mgr._get_item_specs(kw))

    ref = {}
    for i, (kw, budget, path, axis) in enumerate(recs):
        ref[i] = base._run_budget_graph(kw, budget, path=path, axis_keys=axis).clone()
    report = {
        "num_batches": len(recs),
        "budgets": sorted({f"{r[2]}:{r[1]}" for r in recs}),
        "axis_keys": sorted({str(r[3]) for r in recs if r[3]}),
    }
    for name, idxs in orders.items():
        mism = []
        for i in idxs:
            kw, budget, path, axis = recs[i]
            out = mgr._run_budget_graph(kw, budget, path=path, axis_keys=axis)
            n = path_tokens(kw, path)
            if not torch.equal(out[:n], ref[i][:n]):
                mism.append((i, path, budget, float((out[:n] - ref[i][:n]).abs().max())))
        report[name] = {"mismatches": mism}

    # Execute A/B: whole execute() calls, every item compared after all of the
    # call's batches ran.
    exec_mism = []
    single_item_max = []
    replays = []
    multi_budget = 0
    for i, kw in enumerate(mgr._recorded_execs):
        budgets_used = []
        run = mgr._run_budget_graph

        def counting_run(mm_kwargs, token_budget, path="default", axis_keys=()):
            budgets_used.append((path, token_budget))
            return run(mm_kwargs, token_budget, path=path, axis_keys=axis_keys)

        mgr._run_budget_graph = counting_run
        # Clone right away: a model that ignores clone=True returns views of
        # the graph output, which the runs below would overwrite.
        out = [o.clone() for o in mgr.execute(kw)]
        mgr.__dict__.pop("_run_budget_graph")
        ref_out = [o.clone() for o in base.execute(kw)]
        # Several batches of one path on different budgets in a single call.
        per_path = Counter(p for p, _ in set(budgets_used))
        if any(n > 1 for n in per_path.values()):
            multi_budget += 1
        for j, (a, b) in enumerate(zip(out, ref_out, strict=True)):
            if not torch.equal(a, b):
                exec_mism.append((i, j, float((a.float() - b.float()).abs().max())))
        # Each item alone in its own execute(): no later batch can overwrite
        # it, so this catches overwrites that the unshared manager shares too
        # (several batches replaying the same budget graph).
        if len(out) > 1:
            for j in range(len(out)):
                single = mgr.execute(mgr._select_items(kw, [j])[0])[0].float()
                rel = float((out[j].float() - single).norm() / single.norm())
                single_item_max.append((i, j, rel))
            replays.append((i, budgets_used))
    # Control: a second unshared capture against the first. Differences here
    # come from capture itself (e.g. kernel selection), not from sharing.
    if os.environ.get("AB_CONTROL") == "1":
        base2 = capture_unshared()
        ctrl = []
        for i, (kw, budget, path, axis) in enumerate(recs):
            out = base2._run_budget_graph(kw, budget, path=path, axis_keys=axis)
            n = path_tokens(kw, path)
            if not torch.equal(out[:n], ref[i][:n]):
                ctrl.append((i, path, budget, float((out[:n] - ref[i][:n]).abs().max())))
        # Same graph replayed twice: replay itself is deterministic.
        again = []
        for i, (kw, budget, path, axis) in enumerate(recs):
            a = base._run_budget_graph(kw, budget, path=path, axis_keys=axis).clone()
            b = base._run_budget_graph(kw, budget, path=path, axis_keys=axis)
            n = path_tokens(kw, path)
            if not torch.equal(a[:n], b[:n]):
                again.append((i, path, budget))
        report["control_unshared_vs_unshared"] = {"mismatches": ctrl}
        report["control_replay_twice"] = {"mismatches": again}
        base2.clear()
    report["execute"] = {
        "num_executes": len(mgr._recorded_execs),
        "num_multi_budget_executes": multi_budget,
        "num_items": sum(len(mgr._get_item_specs(kw)) for kw in mgr._recorded_execs),
        "mismatches": exec_mism,
        # (execute, item, relative L2 error) of multi-item calls vs. the item
        # run alone; bf16 noise from a different budget graph stays ~1e-2.
        "vs_single_item": sorted(single_item_max, key=lambda m: -m[2]),
        # (path, budget) replays of each multi-item call, in order.
        "multi_item_replays": replays,
    }
    base.clear()
    return report


PROMPTS = {
    "qwen3_vl": lambda n, q: (
        "<|im_start|>user\n"
        + "<|vision_start|><|image_pad|><|vision_end|>" * n
        + f"{q}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "dsv41": lambda n, q: (
        "<｜begin▁of▁sentence｜><｜User｜>"
        + "<｜deepseek_image｜>\n\n" * n
        + f"{q}<｜Assistant｜></think>"
    ),
    "step3_vl": lambda n, q: (
        "<｜begin▁of▁sentence｜> You are a helpful assistant.<|BOT|>user\n "
        + "<im_patch>" * n
        + f"{q} <|EOT|><|BOT|>assistant\n"
    ),
    "gemma4": lambda n, q: (
        "<bos><start_of_turn>user\n"
        + "<|image|>" * n
        + f"\n{q}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "deepseek_ocr": lambda n, q: "<image>" * n + f"\n{q}",
}

ENGINES = {
    "qwen3_vl": dict(
        model="Qwen/Qwen3-VL-2B-Instruct",
        max_model_len=16384,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.6,
    ),
    "dsv41": dict(
        model="/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
        "314113f8a7dadb388215396ba5fbee2c610a04d6",
        tokenizer_mode="deepseek_v41",
        tensor_parallel_size=2,
        moe_backend="aiter",
        block_size=128,
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
        engram_config={"cpu_offload": True},
        max_model_len=32768,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
    ),
    # Dual-path encoder (global / local patches). Two items per batch makes
    # a 3-image prompt split into batches on different global budgets.
    "step3_vl": dict(
        model="stepfun-ai/Step3-VL-10B",
        trust_remote_code=True,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        # Leaves room for the two extra unshared captures of AB_CONTROL=1.
        gpu_memory_utilization=0.5,
        compilation_config={"encoder_cudagraph_max_vision_items_per_batch": 2},
    ),
    # Capture axes (one graph per budget per pooled row width); two items
    # per batch as for Step3-VL.
    "gemma4": dict(
        model="google/gemma-4-E2B-it",
        max_model_len=16384,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.6,
        compilation_config={"encoder_cudagraph_max_vision_items_per_batch": 2},
    ),
    # Dual-path encoder (global / local tiles).
    "deepseek_ocr": dict(
        model="deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.6,
    ),
}


def build_requests(make_prompt, variant=0):
    stop = ImageAsset("stop_sign").pil_image.convert("RGB")
    cherry = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    sizes = [(224, 224), (448, 336), (640, 480), (896, 896), (1300, 876)]
    if variant:
        # Distinct pixels and sizes so the encoder cache cannot serve them.
        stop, cherry = ImageOps.mirror(stop), ImageOps.mirror(cherry)
        sizes = [(w + 32 * variant, h + 32 * variant) for w, h in sizes]
    reqs = []
    for i, (w, h) in enumerate(sizes):
        img = (stop if i % 2 == 0 else cherry).resize((w, h), Image.BICUBIC)
        reqs.append(
            {
                "prompt": make_prompt(1, "Describe this image in detail."),
                "multi_modal_data": {"image": [img]},
            }
        )
    reqs.append(
        {
            "prompt": make_prompt(1, "What is in this image?"),
            "multi_modal_data": {"image": [cherry]},
        }
    )
    multi = [
        stop.resize((320, 240)),
        cherry.resize((512, 512)),
        stop.resize((800, 600)),
    ]
    reqs.append(
        {
            "prompt": make_prompt(3, "Compare these three images."),
            "multi_modal_data": {"image": multi},
        }
    )
    return reqs


def main():
    which, out_path = sys.argv[1], sys.argv[2]
    eager_vit = len(sys.argv) > 3 and sys.argv[3] == "eager_vit"
    kwargs = dict(ENGINES[which])
    cc = {"cudagraph_mm_encoder": not eager_vit, **kwargs.pop("compilation_config", {})}
    make_prompt = PROMPTS[which]

    llm = LLM(
        compilation_config=cc,
        limit_mm_per_prompt={"image": 3},
        enable_prefix_caching=False,
        max_num_seqs=8,
        **kwargs,
    )
    stats = llm.collective_rpc(rpc_stats)
    llm.collective_rpc(rpc_install_recorder)

    sp = SamplingParams(temperature=0.0, max_tokens=48, logprobs=5)
    outs = llm.generate(build_requests(make_prompt), sp)
    # One request at a time so each image size replays its own budget graph.
    for req in build_requests(make_prompt, variant=1):
        outs += llm.generate([req], sp)
    hits = llm.collective_rpc(rpc_hits)
    ab = llm.collective_rpc(rpc_ab_check) if not eager_vit else None

    results = []
    for o in outs:
        c = o.outputs[0]
        results.append(
            {
                "text": c.text,
                "token_ids": list(c.token_ids),
                "logprobs": [
                    {str(t): lp.logprob for t, lp in step.items()}
                    for step in (c.logprobs or [])
                ],
            }
        )
    with open(out_path, "w") as f:
        json.dump({"stats": stats, "hits": hits, "ab": ab, "results": results}, f, indent=1)
    print("WROTE", out_path)
    print(json.dumps({"hits": hits, "ab": ab}, indent=1)[:6000])


if __name__ == "__main__":
    main()
