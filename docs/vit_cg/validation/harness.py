"""A/B harness for shared encoder CUDA graph capture buffers.

Usage: python harness.py <qwen3_vl|dsv41> <out.json>
"""

import json
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
            bufs = [
                b
                for b in [*meta.input_buffers.values(), meta.output_buffer]
                if isinstance(b, torch.Tensor) and b.device.type != "cpu"
            ]
            for b in bufs:
                naive += b.numel() * b.element_size()
                st = b.untyped_storage()
                storages[st.data_ptr()] = st.nbytes()
            graphs.append(
                {
                    "path": path,
                    "key": str(key),
                    "shared_keys": sorted(getattr(meta, "shared_keys", ())),
                }
            )
    return {
        "runner": type(runner).__module__,
        "budgets": {p: b for p, b in mgr.path_token_budgets.items()},
        "max_batch_size": mgr.max_batch_size,
        "num_graphs": len(graphs),
        "graphs": graphs,
        "view_bytes": naive,
        "unique_storage_bytes": sum(storages.values()),
    }


def rpc_install_counter(worker):
    _, mgr = _find_manager(worker)
    if mgr is None or hasattr(mgr, "_hit_budgets"):
        return False
    mgr._hit_budgets = Counter()
    orig = mgr._run_budget_graph

    def wrapped(mm_kwargs, token_budget, path="default", axis_keys=()):
        mgr._hit_budgets[f"{path}:{token_budget}"] += 1
        return orig(mm_kwargs, token_budget, path=path, axis_keys=axis_keys)

    mgr._run_budget_graph = wrapped
    return True


def _clone_kwargs(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_kwargs(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_kwargs(v) for v in obj)
    return obj


def rpc_install_recorder(worker):
    _, mgr = _find_manager(worker)
    if mgr is None:
        return False
    mgr._recorded = []
    orig = mgr._run_budget_graph
    mgr._orig_run_budget_graph = orig

    def wrapped(mm_kwargs, token_budget, path="default", axis_keys=()):
        mgr._recorded.append((_clone_kwargs(mm_kwargs), token_budget, path, axis_keys))
        return orig(mm_kwargs, token_budget, path=path, axis_keys=axis_keys)

    mgr._run_budget_graph = wrapped
    return True


def rpc_ab_check(worker):
    """Replay every recorded batch on the shared-buffer graphs and on a fresh
    manager captured without sharing; compare outputs bit for bit."""
    import torch

    with torch.inference_mode():
        return _ab_check(worker)


def _ab_check(worker):
    import random

    import torch

    from vllm.distributed.parallel_state import graph_capture
    from vllm.platforms import current_platform
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    _, mgr = _find_manager(worker)
    if mgr is None:
        return None
    base = EncoderCudaGraphManager(mgr.vllm_config, mgr.device, mgr.dtype, mgr.model)
    base._get_shared_buffer = lambda *a, **k: None
    with graph_capture(device=mgr.device):
        base.capture(graph_pool=current_platform.graph_pool_handle())
        torch.accelerator.synchronize()
    base_unique = {
        b.untyped_storage().data_ptr()
        for g in base.budget_graphs.values()
        for m in g.values()
        for b in [*m.input_buffers.values(), m.output_buffer]
        if isinstance(b, torch.Tensor) and b.device.type != "cpu"
    }

    run_shared = mgr._orig_run_budget_graph
    recs = mgr._recorded
    order = list(range(len(recs)))
    orders = {
        "forward": order,
        "reverse": order[::-1],
        "by_budget_desc": sorted(order, key=lambda i: -recs[i][1]),
        "by_budget_asc": sorted(order, key=lambda i: recs[i][1]),
        "shuffled": random.Random(0).sample(order, len(order)),
    }
    ref = {}
    for i, (kw, budget, path, axis) in enumerate(recs):
        ref[i] = base._run_budget_graph(kw, budget, path=path, axis_keys=axis).clone()
    report = {"num_batches": len(recs), "budgets": sorted({r[1] for r in recs})}
    for name, idxs in orders.items():
        mism = []
        for i in idxs:
            kw, budget, path, axis = recs[i]
            out = run_shared(kw, budget, path=path, axis_keys=axis)
            n_tok = sum(mgr._get_per_item_out_tokens(kw))
            if not torch.equal(out[:n_tok], ref[i][:n_tok]):
                mism.append((i, budget, float((out[:n_tok] - ref[i][:n_tok]).abs().max())))
            # also check the replay matches the baseline graph of the same budget
            # when baseline replays right after (cross-budget contamination)
        report[name] = {"mismatches": mism}
    report["base_buffer_storages"] = len(base_unique)
    base.clear()
    return report


def rpc_hits(worker):
    _, mgr = _find_manager(worker)
    if mgr is None:
        return None
    return {
        "budget_hits": dict(getattr(mgr, "_hit_budgets", {})),
        "graph_hits": mgr.graph_hits,
        "graph_misses": mgr.graph_misses,
    }


QWEN_IMG = "<|vision_start|><|image_pad|><|vision_end|>"
DS_IMG = "<｜deepseek_image｜>\n\n"


def qwen_prompt(n_img: int, q: str) -> str:
    return (
        f"<|im_start|>user\n{QWEN_IMG * n_img}{q}<|im_end|>\n<|im_start|>assistant\n"
    )


def ds_prompt(n_img: int, q: str) -> str:
    return (
        f"<｜begin▁of▁sentence｜><｜User｜>{DS_IMG * n_img}{q}<｜Assistant｜></think>"
    )


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
    cc = {"cudagraph_mm_encoder": not eager_vit}
    if which == "qwen3_vl":
        kwargs = dict(
            model="/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/"
            "snapshots/89644892e4d85e24eaac8bacfd4f463576704203",
            max_model_len=16384,
            max_num_batched_tokens=16384,
            max_num_seqs=8,
            gpu_memory_utilization=0.6,
        )
        make_prompt = qwen_prompt
    else:
        kwargs = dict(
            model="/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
            "314113f8a7dadb388215396ba5fbee2c610a04d6",
            tokenizer_mode="deepseek_v41",
            tensor_parallel_size=2,
            moe_backend="aiter",
            block_size=128,
            attention_config={
                "indexer_kv_dtype": "mxfp4",
                "indexer_sparse_logits": True,
            },
            engram_config={"cpu_offload": True},
            max_model_len=32768,
            max_num_batched_tokens=16384,
            max_num_seqs=8,
            gpu_memory_utilization=0.85,
        )
        make_prompt = ds_prompt

    llm = LLM(
        compilation_config=cc,
        limit_mm_per_prompt={"image": 3},
        enable_prefix_caching=False,
        **kwargs,
    )
    stats = llm.collective_rpc(rpc_stats)
    llm.collective_rpc(rpc_install_counter)
    llm.collective_rpc(rpc_install_recorder)

    reqs = build_requests(make_prompt)
    sp = SamplingParams(temperature=0.0, max_tokens=48, logprobs=5)
    outs = llm.generate(reqs, sp)
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
        json.dump(
            {"stats": stats, "hits": hits, "ab": ab, "results": results}, f, indent=1
        )
    print("WROTE", out_path)
    print(json.dumps({"hits": hits, "ab": ab}, indent=1)[:6000])


if __name__ == "__main__":
    main()
