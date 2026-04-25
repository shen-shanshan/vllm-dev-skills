#!/usr/bin/env python3
"""
Fetch and organize multimodal-related open issues from vllm-project/vllm.

Categorizes issues by problem type and generates a structured Markdown report.
Prefers GitHub CLI (gh) for authentication. Falls back to raw API with optional token.

Usage:
    python3 fetch_multimodal_issues.py [OPTIONS]

Options:
    --token       GitHub personal access token (optional, bypassed if gh CLI is available)
    --output-dir  Directory to save the report (default: ./outputs)
    --keywords    Extra comma-separated keywords to search (appended to defaults)
    --stdout      Print report to stdout instead of saving to file

Examples:
    python3 fetch_multimodal_issues.py
    python3 fetch_multimodal_issues.py --output-dir /tmp/reports
    python3 fetch_multimodal_issues.py --keywords "Kimi,DeepSeek-VL"
    python3 fetch_multimodal_issues.py --stdout
"""

import sys
import json
import argparse
import subprocess
import time
import urllib.request
import urllib.parse
import urllib.error
import os
import re
from datetime import datetime
from pathlib import Path

REPO = "vllm-project/vllm"

# ── Keyword groups (batched to reduce GitHub Search API calls) ──────────────
KEYWORD_GROUPS = [
    # Generic multimodal terms
    ["multimodal", "multi-modal", "VLM", "vision language", "image input"],
    # VL/Vision-Language models
    ["Qwen-VL", "Qwen2-VL", "Qwen2.5-VL", "LLaVA", "InternVL"],
    ["MiniCPM-V", "Phi-3-vision", "Phi-4-vision", "PaliGemma", "Florence"],
    ["LLaMA-Vision", "Llama3-vision", "Pixtral", "Idefics", "CogVLM"],
    # Omni / Speech / Audio models
    ["Qwen-Omni", "Qwen2-Omni", "Qwen2.5-Omni", "audio", "whisper", "speech"],
    # Video understanding
    ["video understanding", "video input", "video model", "temporal"],
    # Multimodal feature keywords
    ["ViT", "visual encoder", "image encoder", "vision tower"],
    ["cuda graph", "CUDA graph", "vit cuda", "mm cuda"],
    ["EPD", "encoder-decoder disaggregat", "prefill disaggregat"],
    ["prefix caching multimodal", "mm prefix", "image cache", "visual cache"],
    ["image token", "image patch", "pixel", "vision token"],
]

ALL_KEYWORDS = [kw for group in KEYWORD_GROUPS for kw in group]

# ── Problem-type classification rules ─────────────────────────────────────
# Each rule: (category_name, [title/label keywords that signal this category])
PROBLEM_TYPE_RULES = [
    ("Bug / 错误报告", [
        "bug", "error", "crash", "fail", "broken", "wrong", "incorrect",
        "exception", "traceback", "oom", "out of memory", "segfault",
        "hang", "stuck", "timeout", "not work", "doesn't work",
    ]),
    ("Feature Request / 功能请求", [
        "feature", "request", "support", "add", "implement", "enable",
        "allow", "proposal", "enhancement", "wish", "want",
    ]),
    ("Performance / 性能问题", [
        "performance", "slow", "latency", "throughput", "speed", "optimize",
        "bottleneck", "memory usage", "overhead", "benchmark", "oom",
        "out of memory",
    ]),
    ("CUDA Graph / 计算图", [
        "cuda graph", "vit cuda", "cuda graph multimodal", "mm cuda graph",
        "graph capture", "graph mode",
    ]),
    ("EPD / 编解码分离", [
        "EPD", "encoder prefill", "encoder-decoder disaggregat",
        "disaggregat", "prefill disaggregat", "decode disaggregat",
    ]),
    ("Prefix Caching / 前缀缓存", [
        "prefix caching", "prefix cache", "kv cache multimodal",
        "image cache", "visual cache", "mm prefix",
    ]),
    ("ViT / 视觉编码器", [
        "vit", "visual encoder", "image encoder", "vision tower",
        "clip", "siglip", "visual backbone",
    ]),
    ("Video Multimodal / 视频多模态", [
        "video", "temporal", "frame", "video understanding",
    ]),
    ("Audio / Speech / 音频语音", [
        "audio", "whisper", "speech", "asr", "tts", "voice",
    ]),
    ("VL Model: Qwen-VL系列", [
        "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwenvl",
    ]),
    ("VL Model: Qwen-Omni系列", [
        "qwen-omni", "qwen2-omni", "qwen2.5-omni", "qwenomni",
    ]),
    ("VL Model: LLaVA系列", [
        "llava", "llava-next", "llava-onevision",
    ]),
    ("VL Model: InternVL系列", [
        "internvl", "intern-vl", "internlm-xcomposer",
    ]),
    ("VL Model: 其他模型", [
        "minicpm", "phi-3-vision", "phi-4-vision", "paligemma", "florence",
        "pixtral", "idefics", "cogvlm", "llama-vision", "mllama",
    ]),
    ("Image Token / 输入处理", [
        "image token", "image patch", "pixel", "vision token",
        "image input", "image process",
    ]),
    ("Compatibility / 兼容性", [
        "compat", "incompatible", "version", "upgrade", "deprecat",
        "migrat", "backward",
    ]),
    ("Documentation / 文档", [
        "doc", "document", "readme", "example", "tutorial", "guide",
    ]),
]


def gh_available():
    try:
        result = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def search_with_gh(query):
    cmd = [
        "gh", "api",
        "-H", "Accept: application/vnd.github.v3+json",
        f"/search/issues?q={urllib.parse.quote(query)}&per_page=100&sort=created&order=desc",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"gh CLI error: {result.stderr}", file=sys.stderr)
        return []
    data = json.loads(result.stdout)
    return data.get("items", [])


def search_with_api(query, token=None):
    encoded = urllib.parse.quote(query)
    url = f"https://api.github.com/search/issues?q={encoded}&per_page=100&sort=created&order=desc"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "vllm-multimodal-issue-fetcher/1.0",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("items", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        if e.code == 403:
            print(
                "\nRate limit exceeded. Use --token or install gh CLI.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        return []
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}", file=sys.stderr)
        return []


def build_query(keywords):
    kw_or = " OR ".join(f'"{kw}"' for kw in keywords)
    return f"repo:{REPO} is:open is:issue ({kw_or}) in:title,body"


def fetch_all_multimodal_issues(extra_keywords=None, token=None):
    use_gh = gh_available()
    if use_gh:
        print("  Using gh CLI for authenticated requests.", file=sys.stderr)
    elif token:
        print("  Using API token.", file=sys.stderr)
    else:
        print("  Warning: No auth. Rate limits apply (10 req/min).", file=sys.stderr)

    groups = list(KEYWORD_GROUPS)
    if extra_keywords:
        groups.append(extra_keywords)

    seen = {}
    for i, group in enumerate(groups):
        query = build_query(group)
        print(f"  [{i+1}/{len(groups)}] Searching: {group[:3]}...", file=sys.stderr)

        if use_gh:
            items = search_with_gh(query)
        else:
            items = search_with_api(query, token)
            if i < len(groups) - 1:
                time.sleep(6)

        for issue in items:
            num = issue["number"]
            if num not in seen:
                seen[num] = issue

    print(f"\n  Found {len(seen)} unique multimodal-related open issues.", file=sys.stderr)
    return sorted(seen.values(), key=lambda x: x["number"], reverse=True)


def format_date(iso_str):
    try:
        return datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
    except Exception:
        return iso_str[:10]


def classify_issue(issue):
    """Return list of matched problem-type category names for an issue."""
    title = issue.get("title", "").lower()
    body = (issue.get("body") or "")[:500].lower()
    label_names = " ".join(l["name"].lower() for l in issue.get("labels", []))
    text = f"{title} {label_names} {body}"

    matched = []
    for category, signals in PROBLEM_TYPE_RULES:
        for signal in signals:
            if signal.lower() in text:
                matched.append(category)
                break  # only add each category once

    return matched if matched else ["其他 / Other"]


def render_issue_row(issue):
    num = issue["number"]
    title = issue["title"].replace("|", "\\|")
    url = issue["html_url"]
    author = issue.get("user", {}).get("login", "unknown")
    created = format_date(issue.get("created_at", ""))
    labels = ", ".join(f"`{l['name']}`" for l in issue.get("labels", [])) or "-"
    return f"| [#{num}]({url}) | {title} | @{author} | {created} | {labels} |"


def render_markdown(issues):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# vllm-project/vllm 多模态相关 Open Issues",
        f"",
        f"> 仓库: [{REPO}](https://github.com/{REPO})  ",
        f"> 生成时间: {now}  ",
        f"> 共 **{len(issues)}** 个 open issue",
        f"",
    ]

    # ── Section 1: Full issue table ────────────────────────────────────────
    lines += [
        "## 一、Issue 总览",
        "",
        "| # | 标题 | 作者 | 创建日期 | Labels |",
        "|---|------|------|----------|--------|",
    ]
    for issue in issues:
        lines.append(render_issue_row(issue))

    # ── Section 2: By problem type ─────────────────────────────────────────
    lines += ["", "---", "", "## 二、按问题类型分类", ""]

    # Build category → issues mapping
    category_map: dict[str, list] = {}
    for issue in issues:
        for cat in classify_issue(issue):
            category_map.setdefault(cat, []).append(issue)

    # Render in the order defined by PROBLEM_TYPE_RULES, then "其他"
    rendered_cats = set()
    ordered_cats = [cat for cat, _ in PROBLEM_TYPE_RULES] + ["其他 / Other"]
    for cat in ordered_cats:
        if cat not in category_map:
            continue
        rendered_cats.add(cat)
        cat_issues = category_map[cat]
        lines.append(f"### {cat} ({len(cat_issues)})")
        lines.append("")
        for issue in cat_issues:
            num = issue["number"]
            title = issue["title"]
            url = issue["html_url"]
            created = format_date(issue.get("created_at", ""))
            lines.append(f"- [#{num}]({url}) **{title}** _(created: {created})_")
        lines.append("")

    # Any leftover categories not in the predefined list
    for cat, cat_issues in category_map.items():
        if cat in rendered_cats:
            continue
        lines.append(f"### {cat} ({len(cat_issues)})")
        lines.append("")
        for issue in cat_issues:
            num = issue["number"]
            title = issue["title"]
            url = issue["html_url"]
            created = format_date(issue.get("created_at", ""))
            lines.append(f"- [#{num}]({url}) **{title}** _(created: {created})_")
        lines.append("")

    # ── Section 3: Category summary table ─────────────────────────────────
    lines += ["---", "", "## 三、分类统计"]
    lines += [
        "",
        "| 问题类型 | Issue 数量 |",
        "|----------|-----------|",
    ]
    for cat in ordered_cats:
        if cat in category_map:
            lines.append(f"| {cat} | {len(category_map[cat])} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=f"Fetch and categorize multimodal open issues from {REPO}"
    )
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save Markdown report (default: <skill>/outputs/)",
    )
    parser.add_argument(
        "--keywords",
        help="Extra comma-separated keywords",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print report to stdout instead of saving to file",
    )
    args = parser.parse_args()

    extra_keywords = None
    if args.keywords:
        extra_keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    print(f"Fetching multimodal open issues from {REPO}...", file=sys.stderr)
    issues = fetch_all_multimodal_issues(extra_keywords, token=args.token)

    report = render_markdown(issues)

    if args.stdout:
        print(report)
        return

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        # Default: <skill_root>/outputs/
        skill_root = Path(__file__).parent.parent
        out_dir = skill_root / "outputs"

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"vllm_multimodal_issues_{ts}.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_file}", file=sys.stderr)
    print(f"Total issues: {len(issues)}", file=sys.stderr)


if __name__ == "__main__":
    main()
