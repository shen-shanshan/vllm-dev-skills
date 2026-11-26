#!/usr/bin/env python3
"""One-click updater for the vllm-vit-cg-pr-review skill.

Re-harvests the review comments of the ViT CUDA graph maintainers
(shen-shanshan, Isotr0py) from every PR linked in the tracker issue
vllm-project/vllm#38175 into references/upstream/reviewer-comments.md, and
records the harvest in references/upstream/.update-metadata.json.

With --check-vllm-tree it also verifies that the framework paths and symbols
referenced by the rules file still exist in vllm main (OK / MISSING) — a
tripwire for interface drift (renamed protocol methods, moved files).

This script NEVER rewrites references/vit-cg-review-rules.md,
references/reviewer-playbook.md or SKILL.md — distilling new review patterns
is a curation step performed by Claude with user approval.

Requires the gh CLI (authenticated).
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

REPO = "vllm-project/vllm"
TRACKER_ISSUE = 38175
REVIEWERS = ("shen-shanshan", "Isotr0py")

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_DIR = os.path.join(SKILL_DIR, "references", "upstream")
CORPUS_FILE = os.path.join(UPSTREAM_DIR, "reviewer-comments.md")
METADATA_FILE = os.path.join(UPSTREAM_DIR, ".update-metadata.json")

PR_REF = re.compile(r"(?:github\.com/vllm-project/vllm/pull/|#)(\d{4,6})\b")

# Framework files referenced by the rules file (path -> symbols that must exist).
VLLM_FRAMEWORK = {
    "vllm/v1/worker/encoder_cudagraph.py": ["class EncoderCudaGraphManager", "def _generate_budgets"],
    "vllm/v1/worker/encoder_cudagraph_defs.py": [
        "class EncoderItemSpec",
        "class EncoderCudaGraphPathConfig",
        "class EncoderCudaGraphConfig",
        "class EncoderCudaGraphCaptureInputs",
        "class EncoderCudaGraphReplayBuffers",
        "padding_logics",
        "capture_axes",
        "ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG",
    ],
    "vllm/model_executor/models/interfaces.py": [
        "class SupportsEncoderCudaGraph",
        "def get_encoder_cudagraph_config",
        "def get_encoder_cudagraph_budget_range",
        "def get_encoder_cudagraph_item_specs",
        "def select_encoder_cudagraph_items",
        "def postprocess_encoder_output",
        "def prepare_encoder_cudagraph_capture_inputs",
        "def prepare_encoder_cudagraph_replay_buffers",
        "def encoder_cudagraph_forward",
        "def encoder_eager_forward",
    ],
    "vllm/model_executor/models/encoder_cudagraph_utils.py": ["class Qwen2VLEncoderCudaGraphMixin"],
    "vllm/model_executor/models/utils.py": ["def scatter_output_slices"],
    "vllm/config/compilation.py": [
        "cudagraph_mm_encoder",
        "encoder_cudagraph_token_budgets",
        "encoder_cudagraph_max_token_budget",
        "encoder_cudagraph_max_vision_items_per_batch",
        "encoder_cudagraph_max_frames_per_batch",
    ],
    "docs/design/cuda_graphs_multimodal.md": ["## Supported Models"],
    "tests/models/multimodal/generation/test_vit_cudagraph.py": ["MODEL_CONFIGS", "class VitCudagraphTestConfig"],
    "tests/v1/cudagraph/test_encoder_cudagraph.py": [],
    "examples/generate/multimodal/vision_language_offline.py": ["MODELS_SUPPORT_VIT_CUDA_GRAPH"],
}


def gh_json(args: List[str]) -> Any:
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def gh_list(path: str) -> List[Any]:
    pages = gh_json(["api", "--paginate", "--slurp", path])
    return [item for page in pages for item in page]


def login(item: Dict[str, Any]) -> str:
    return (item.get("user") or {}).get("login", "ghost")


def tracker_prs() -> List[int]:
    issue = gh_json(["api", f"repos/{REPO}/issues/{TRACKER_ISSUE}"])
    comments = gh_list(f"repos/{REPO}/issues/{TRACKER_ISSUE}/comments")
    text = "\n".join([issue.get("body") or ""] + [c.get("body") or "" for c in comments])
    return sorted({int(n) for n in PR_REF.findall(text)})


def harvest(number: int) -> Optional[Dict[str, Any]]:
    """Collect the reviewers' comments on one PR, with the author's replies for context."""
    try:
        pr = gh_json(["api", f"repos/{REPO}/pulls/{number}"])
    except RuntimeError:
        return None
    state = "merged" if pr.get("merged") else pr["state"]
    inline = gh_list(f"repos/{REPO}/pulls/{number}/comments")
    threads: Dict[int, List[Dict[str, Any]]] = {}
    for c in inline:
        threads.setdefault(c.get("in_reply_to_id") or c["id"], []).append(c)
    blocks = []
    for thread in threads.values():
        if not any(login(c) in REVIEWERS for c in thread):
            continue
        first = thread[0]
        hunk = "\n".join((first.get("diff_hunk") or "").splitlines()[-6:])
        line = first.get("line") or first.get("original_line")
        blocks.append(f"### `{first['path']}:{line}`\n```diff\n{hunk}\n```")
        for c in thread:
            blocks.append(f"- **@{login(c)}**: {c['body'].strip()}")
    for r in gh_list(f"repos/{REPO}/pulls/{number}/reviews"):
        if login(r) in REVIEWERS and (r.get("body") or "").strip():
            blocks.append(f"### review @{login(r)} [{r['state']}]\n{r['body'].strip()}")
    for c in gh_list(f"repos/{REPO}/issues/{number}/comments"):
        body = (c.get("body") or "").strip()
        if login(c) in REVIEWERS and body and not body.startswith("/ci "):
            blocks.append(f"### comment @{login(c)}\n{body}")
    return {
        "number": number,
        "title": pr["title"],
        "author": login(pr),
        "state": state,
        "count": len(blocks),
        "text": "\n\n".join(blocks),
    }


def write_corpus(results: List[Dict[str, Any]]) -> None:
    today = datetime.date.today().isoformat()
    parts = [
        "# ViT CUDA graph reviewer comments (raw corpus)",
        "",
        f"> Harvested {today} by `scripts/update_skill.py` from the PRs linked in "
        f"[#{TRACKER_ISSUE}](https://github.com/{REPO}/issues/{TRACKER_ISSUE}). "
        f"Reviewers: {', '.join('@' + r for r in REVIEWERS)}; author replies are kept for context. "
        "Distilled rules live in `../vit-cg-review-rules.md` and `../reviewer-playbook.md` — "
        "this file is the evidence they were built from. Do not edit by hand.",
        "",
    ]
    for r in sorted(results, key=lambda item: -item["count"]):
        if not r["count"]:
            continue
        parts.append(f"## PR #{r['number']}: {r['title']} (@{r['author']}, {r['state']})")
        parts.append("")
        parts.append(r["text"])
        parts.append("")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def check_vllm_tree(ref: str) -> bool:
    ok = True
    for path, symbols in VLLM_FRAMEWORK.items():
        try:
            blob = gh_json(["api", f"repos/{REPO}/contents/{path}?ref={ref}"])
        except RuntimeError:
            print(f"MISSING  {path}")
            ok = False
            continue
        content = base64.b64decode(blob.get("content", "")).decode(errors="replace")
        if not content and blob.get("download_url"):  # files > 1 MB come without inline content
            content = subprocess.run(["curl", "-sL", blob["download_url"]], capture_output=True, text=True).stdout
        missing = [s for s in symbols if s not in content]
        print(f"{'OK     ' if not missing else 'CHANGED'}  {path}" + (f"  (missing: {', '.join(missing)})" if missing else ""))
        ok = ok and not missing
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-vllm-tree", action="store_true", help="verify framework paths/symbols in vllm main")
    parser.add_argument("--ref", default="main", help="vllm ref for --check-vllm-tree (default: main)")
    parser.add_argument("--skip-harvest", action="store_true", help="only run --check-vllm-tree")
    args = parser.parse_args()

    os.makedirs(UPSTREAM_DIR, exist_ok=True)
    if not args.skip_harvest:
        numbers = tracker_prs()
        print(f"Harvesting reviewer comments from {len(numbers)} PRs linked in #{TRACKER_ISSUE}...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [r for r in pool.map(harvest, numbers) if r]
        old = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, encoding="utf-8") as f:
                old = json.load(f).get("prs", {})
        new = {str(r["number"]): {"state": r["state"], "comments": r["count"]} for r in results}
        changed = sorted(k for k in new if old.get(k) != new[k])
        write_corpus(results)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"harvested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                       "tracker_issue": TRACKER_ISSUE, "reviewers": list(REVIEWERS), "prs": new}, f, indent=2)
        total = sum(r["count"] for r in results)
        print(f"Wrote {CORPUS_FILE} ({total} comment blocks).")
        print(f"PRs new/changed since last harvest: {', '.join('#' + n for n in changed) or 'none'}")
    if args.check_vllm_tree:
        print(f"\nChecking framework paths in {REPO}@{args.ref}...")
        if not check_vllm_tree(args.ref):
            print("\nSome paths/symbols moved: update references/vit-cg-review-rules.md (framework map).")
            sys.exit(2)


if __name__ == "__main__":
    main()
