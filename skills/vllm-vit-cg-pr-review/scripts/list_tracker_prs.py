#!/usr/bin/env python3
"""List the PRs linked from the ViT full CUDA graph tracker issue
(vllm-project/vllm#38175, body + comments) and show which are still open.

For each PR it also reports whether a review report already exists in this
skill's outputs/ directory and whether the PR head moved since that review.

Requires the gh CLI (authenticated).

Usage:
    python3 list_tracker_prs.py              # open PRs only
    python3 list_tracker_prs.py --all        # every linked PR with its state
    python3 list_tracker_prs.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

REPO = "vllm-project/vllm"
TRACKER_ISSUE = 38175
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(SKILL_DIR, "outputs")

PR_REF = re.compile(r"(?:github\.com/vllm-project/vllm/pull/|#)(\d{4,6})\b")


def gh_json(args: List[str]) -> Any:
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def tracker_text() -> str:
    issue = gh_json(["api", f"repos/{REPO}/issues/{TRACKER_ISSUE}"])
    pages = gh_json(["api", "--paginate", "--slurp", f"repos/{REPO}/issues/{TRACKER_ISSUE}/comments"])
    comments = [c for page in pages for c in page]
    return "\n".join([issue.get("body") or ""] + [c.get("body") or "" for c in comments])


def pr_info(number: int) -> Optional[Dict[str, Any]]:
    try:
        pr = gh_json(["api", f"repos/{REPO}/pulls/{number}"])
    except RuntimeError:
        return None  # an issue number, or a deleted PR
    state = "MERGED" if pr.get("merged") else pr["state"].upper()
    return {
        "number": number,
        "state": state,
        "draft": bool(pr.get("draft")),
        "author": (pr.get("user") or {}).get("login", "ghost"),
        "title": pr["title"],
        "head_sha": pr["head"]["sha"],
    }


def existing_review(number: int) -> Optional[str]:
    """Return the head_sha recorded in an existing report, '' if unknown, None if absent."""
    path = os.path.join(OUTPUTS_DIR, f"pr-{number}-review.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        head = f.read(2000)
    match = re.search(r"^head_sha:\s*([0-9a-f]{7,40})", head, re.M)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="include merged/closed PRs")
    parser.add_argument("--json", help="also write the result to this JSON file")
    args = parser.parse_args()

    numbers = sorted({int(n) for n in PR_REF.findall(tracker_text())})
    rows = []
    for number in numbers:
        info = pr_info(number)
        if info is None or (not args.all and info["state"] != "OPEN"):
            continue
        reviewed = existing_review(number)
        if reviewed is None:
            info["report"] = "none"
        elif reviewed and info["head_sha"].startswith(reviewed):
            info["report"] = "up-to-date"
        else:
            info["report"] = "stale"
        rows.append(info)

    for row in rows:
        draft = " [draft]" if row["draft"] else ""
        print(f"#{row['number']}\t{row['state']}{draft}\treport={row['report']}\t@{row['author']}\t{row['title']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n{len(rows)} PR(s) from tracker #{TRACKER_ISSUE}", file=sys.stderr)


if __name__ == "__main__":
    main()
