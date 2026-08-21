#!/usr/bin/env python3
"""Fetch all data for a vllm-project/vllm PR using the gh CLI or GitHub REST API,
for ROCm-focused review by the vllm-rocm-pr-review skill."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Union

REPO = "vllm-project/vllm"


def run_gh(args: List[str]) -> Any:
    """Run a gh CLI command and return parsed JSON output."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gh_available() -> bool:
    try:
        subprocess.run(["gh", "auth", "status"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def api_request(url: str, token: Optional[str] = None) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "vllm-rocm-pr-review"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}: {e.read().decode()}")


def fetch_all_pages(url: str, token: Optional[str] = None) -> List[Any]:
    items = []
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        data = api_request(f"{url}{sep}per_page=100&page={page}", token)
        if not data:
            break
        items.extend(data)
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.5)
    return items


def fetch_via_gh_cli(pr_number: int) -> Dict[str, Any]:
    print(f"Using gh CLI to fetch PR #{pr_number} from {REPO}...")

    # PR metadata + body
    pr = run_gh([
        "pr", "view", str(pr_number),
        "--repo", REPO,
        "--json",
        "number,title,body,author,state,createdAt,mergedAt,closedAt,"
        "labels,assignees,reviewRequests,additions,deletions,changedFiles,"
        "baseRefName,headRefName,url,isDraft,milestone,comments,reviews",
    ])

    # File-level diff
    diff_result = subprocess.run(
        ["gh", "pr", "diff", str(pr_number), "--repo", REPO],
        capture_output=True,
        text=True,
    )
    diff = diff_result.stdout if diff_result.returncode == 0 else ""

    # Review comments (inline code comments) via REST
    review_comments_raw = run_gh([
        "api",
        f"repos/{REPO}/pulls/{pr_number}/comments",
        "--paginate",
    ])

    # Issue comments (general discussion)
    issue_comments_raw = run_gh([
        "api",
        f"repos/{REPO}/issues/{pr_number}/comments",
        "--paginate",
    ])

    # Top-level reviews (approvals / change requests)
    reviews_raw = run_gh([
        "api",
        f"repos/{REPO}/pulls/{pr_number}/reviews",
        "--paginate",
    ])

    # Files changed
    files_raw = run_gh([
        "api",
        f"repos/{REPO}/pulls/{pr_number}/files",
        "--paginate",
    ])

    return {
        "pr": pr,
        "diff": diff,
        "review_comments": review_comments_raw,
        "issue_comments": issue_comments_raw,
        "reviews": reviews_raw,
        "files": files_raw,
    }


def fetch_via_rest_api(pr_number: int, token: Optional[str]) -> Dict[str, Any]:
    base = f"https://api.github.com/repos/{REPO}"
    print(f"Using REST API to fetch PR #{pr_number} from {REPO}...")

    pr = api_request(f"{base}/pulls/{pr_number}", token)
    issue = api_request(f"{base}/issues/{pr_number}", token)
    pr["labels"] = issue.get("labels", [])

    # Raw diff
    headers = {"Accept": "application/vnd.github.diff", "User-Agent": "vllm-rocm-pr-review"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base}/pulls/{pr_number}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            diff = resp.read().decode(errors="replace")
    except Exception:
        diff = ""

    review_comments = fetch_all_pages(f"{base}/pulls/{pr_number}/comments", token)
    issue_comments = fetch_all_pages(f"{base}/issues/{pr_number}/comments", token)
    files = fetch_all_pages(f"{base}/pulls/{pr_number}/files", token)
    reviews = fetch_all_pages(f"{base}/pulls/{pr_number}/reviews", token)

    return {
        "pr": pr,
        "diff": diff,
        "review_comments": review_comments,
        "issue_comments": issue_comments,
        "files": files,
        "reviews": reviews,
    }


def fetch_ci_checks(pr_number: int, token: Optional[str] = None) -> List[Any]:
    """Fetch CI check status for the PR (non-fatal on failure)."""
    try:
        if gh_available():
            return run_gh([
                "pr", "checks", str(pr_number),
                "--repo", REPO,
                "--json", "name,state,conclusion",
            ])
        base = f"https://api.github.com/repos/{REPO}"
        pr = api_request(f"{base}/pulls/{pr_number}", token)
        head_sha = pr.get("head", {}).get("sha")
        if not head_sha:
            return []
        return fetch_all_pages(f"{base}/commits/{head_sha}/check-runs", token)
    except Exception as e:
        print(f"Warning: could not fetch CI checks: {e}")
        return []


def truncate_diff(diff: str, max_chars: int = 80000) -> str:
    """Truncate diff if too large, keeping the file headers for context."""
    if len(diff) <= max_chars:
        return diff
    lines = diff.splitlines(keepends=True)
    result = []
    total = 0
    for line in lines:
        if total + len(line) > max_chars:
            result.append(f"\n... [diff truncated at {max_chars} chars, total diff length: {len(diff)}] ...\n")
            break
        result.append(line)
        total += len(line)
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch vllm-project/vllm PR data for ROCm-focused review")
    parser.add_argument("pr_number", type=int, help="PR number to fetch")
    parser.add_argument("--token", help="GitHub personal access token (optional if gh CLI is available)")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: /tmp/vllm_rocm_pr_<PR>.json)")
    parser.add_argument("--max-diff-chars", type=int, default=80000,
                        help="Max characters for diff (default: 80000)")
    parser.add_argument("--include-ci", action="store_true",
                        help="Also fetch CI check status into the 'checks' key")
    args = parser.parse_args()

    output = args.output or f"/tmp/vllm_rocm_pr_{args.pr_number}.json"

    try:
        if gh_available():
            data = fetch_via_gh_cli(args.pr_number)
        else:
            data = fetch_via_rest_api(args.pr_number, args.token)

        data["diff"] = truncate_diff(data.get("diff", ""), args.max_diff_chars)

        if args.include_ci:
            print("Fetching CI check status...")
            data["checks"] = fetch_ci_checks(args.pr_number, args.token)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        pr = data["pr"]
        title = pr.get("title", "N/A")
        state = pr.get("state", "N/A")
        additions = pr.get("additions", "?")
        deletions = pr.get("deletions", "?")
        n_files = len(data.get("files", []))
        n_comments = len(data.get("issue_comments", [])) + len(data.get("review_comments", []))

        print(f"\n✅ PR #{args.pr_number}: {title}")
        print(f"   State: {state}  |  +{additions} -{deletions}  |  {n_files} files  |  {n_comments} comments")
        print(f"   Data saved to: {output}")
        print(f"\nNow ask Claude to review this PR (run the ROCm relevance gate first).")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
