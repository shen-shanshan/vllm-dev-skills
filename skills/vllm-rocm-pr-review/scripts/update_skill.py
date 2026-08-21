#!/usr/bin/env python3
"""One-click updater for the vllm-rocm-pr-review skill.

Downloads the latest upstream reference material into references/upstream/:
  - ATOM review-pr SKILL.md
  - aiter review-pr SKILL.md
  - mori README.md
  - mori repo tree snapshot (path + type only)

Idempotent: sources whose SHA is unchanged are skipped.
Optionally verifies that the vLLM ROCm backbone paths referenced by the skill
still exist in vllm main (--check-vllm-tree).

This script NEVER rewrites references/vllm-rocm-review-rules.md or SKILL.md —
rule distillation is a curation step performed by Claude with user approval.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

USER_AGENT = "vllm-rocm-pr-review"

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_DIR = os.path.join(SKILL_DIR, "references", "upstream")
METADATA_FILE = os.path.join(UPSTREAM_DIR, ".update-metadata.json")

DEFAULT_REF = "main"

# (name, raw_url_template, output_filename, kind)
# kind: "text" (save raw bytes as text) | "tree" (mori repo tree via contents/trees API)
SOURCES = [
    {
        "name": "atom-review-pr-skill.md",
        "raw_url": "https://raw.githubusercontent.com/ROCm/ATOM/{ref}/.claude/skills/review-pr/SKILL.md",
        "contents_api": "https://api.github.com/repos/ROCm/ATOM/contents/.claude/skills/review-pr/SKILL.md?ref={ref}",
        "kind": "text",
    },
    {
        "name": "aiter-review-pr-skill.md",
        "raw_url": "https://raw.githubusercontent.com/ROCm/aiter/{ref}/.claude/skills/review-pr/SKILL.md",
        "contents_api": "https://api.github.com/repos/ROCm/aiter/contents/.claude/skills/review-pr/SKILL.md?ref={ref}",
        "kind": "text",
    },
    {
        "name": "mori-readme.md",
        "raw_url": "https://raw.githubusercontent.com/ROCm/mori/{ref}/README.md",
        "contents_api": "https://api.github.com/repos/ROCm/mori/contents/README.md?ref={ref}",
        "kind": "text",
    },
    {
        "name": "mori-structure.json",
        "raw_url": None,  # built from trees API
        "contents_api": None,
        "trees_api": "https://api.github.com/repos/ROCm/mori/git/trees/{ref}?recursive=1",
        "kind": "tree",
    },
]

# vLLM ROCm backbone paths referenced by the skill's rules file / SKILL.md.
# Tripwire for vLLM layout drift.
VLLM_BACKBONE_PATHS = [
    "vllm/platforms/rocm.py",
    "vllm/_aiter_ops.py",
    "vllm/kernels/aiter_ops.py",
    "vllm/v1/attention/backends/rocm_attn.py",
    "vllm/v1/attention/backends/rocm_aiter_fa.py",
    "vllm/v1/attention/backends/rocm_aiter_unified_attn.py",
    "vllm/v1/attention/backends/mla/rocm_aiter_mla.py",
    "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py",
    "vllm/v1/attention/backends/mla/aiter_triton_mla.py",
    "vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py",
    "vllm/model_executor/kernels/linear/scaled_mm/rocm.py",
    "vllm/model_executor/kernels/linear/scaled_mm/aiter.py",
    "vllm/model_executor/kernels/linear/mxfp4/aiter.py",
    "vllm/model_executor/kernels/linear/mxfp8/rocm_native.py",
    "vllm/model_executor/kernels/mhc/aiter.py",
    "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/aiter_mxfp8_moe.py",
    "vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py",
    "vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py",
    "vllm/distributed/device_communicators/aiter_custom_all_reduce.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py",
    "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
    "requirements/rocm.txt",
    "requirements/kv_connectors_rocm.txt",
    ".buildkite/ci_config_rocm.yaml",
]

VLLM_TREES_API = "https://api.github.com/repos/vllm-project/vllm/git/trees/{ref}?recursive=1"


def api_request(url: str, token: Optional[str] = None, timeout: int = 60) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}: {e.read().decode()[:300]}")


def download_text(url: str, token: Optional[str] = None, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            headers = {"User-Agent": USER_AGENT}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def load_metadata() -> Dict[str, Any]:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(meta: Dict[str, Any]) -> None:
    write_file(METADATA_FILE, json.dumps(meta, ensure_ascii=False, indent=2))


def fetch_source_sha(source: Dict[str, Any], ref: str, token: Optional[str]) -> Optional[str]:
    """Return the current upstream SHA for a source, or None if unavailable."""
    if source["kind"] == "tree":
        # Use the trees API sha itself as the version marker.
        try:
            data = api_request(source["trees_api"].format(ref=ref), token)
            return data.get("sha")
        except Exception as e:
            print(f"  ⚠️  cannot get SHA for {source['name']}: {e}")
            return None
    try:
        data = api_request(source["contents_api"].format(ref=ref), token)
        return data.get("sha")
    except Exception as e:
        print(f"  ⚠️  cannot get SHA for {source['name']}: {e}")
        return None


def fetch_mori_structure(ref: str, token: Optional[str]) -> str:
    data = api_request(
        "https://api.github.com/repos/ROCm/mori/git/trees/{ref}?recursive=1".format(ref=ref),
        token,
        timeout=120,
    )
    entries = [
        {"path": t["path"], "type": t["type"]}
        for t in data.get("tree", [])
    ]
    # Keep only interesting top dirs to bound size; cap at 1000 entries.
    keep = {"src", "python", "include", "docs"}
    filtered = [e for e in entries
                if e["path"].split("/")[0] in keep or e["path"] in ("README.md", "setup.py", "pyproject.toml")]
    if len(filtered) > 1000:
        filtered = filtered[:1000]
    return json.dumps({
        "sha": data.get("sha"),
        "fetched_from": "https://api.github.com/repos/ROCm/mori/git/trees/{ref}?recursive=1".format(ref=ref),
        "entries": filtered,
    }, ensure_ascii=False, indent=2)


def update_sources(ref: str, token: Optional[str]) -> Dict[str, Any]:
    meta = load_metadata()
    sources_meta = meta.setdefault("sources", {})

    for source in SOURCES:
        name = source["name"]
        prev = sources_meta.get(name, {})
        prev_sha = prev.get("sha")

        print(f"\n[{name}]")
        new_sha = fetch_source_sha(source, ref, token)
        if new_sha is None:
            sources_meta[name] = {
                **prev,
                "status": "error",
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            continue

        if new_sha == prev_sha and os.path.exists(os.path.join(UPSTREAM_DIR, name)):
            print(f"  ↻ unchanged (sha {new_sha[:10]}) — skipping")
            sources_meta[name] = {**prev, "status": "unchanged",
                                  "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            continue

        try:
            if source["kind"] == "text":
                content = download_text(source["raw_url"].format(ref=ref), token)
                out_path = os.path.join(UPSTREAM_DIR, name)
                write_file(out_path, content)
                size = os.path.getsize(out_path)
            else:
                content = fetch_mori_structure(ref, token)
                write_file(os.path.join(UPSTREAM_DIR, name), content)
                size = os.path.getsize(os.path.join(UPSTREAM_DIR, name))

            sources_meta[name] = {
                "url": (source["raw_url"] or source["trees_api"]).format(ref=ref),
                "sha": new_sha,
                "previous_sha": prev_sha,
                "status": "updated",
                "size_bytes": size,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            print(f"  ✅ updated {prev_sha[:10] if prev_sha else 'none'} → {new_sha[:10]} ({size} bytes)")
        except Exception as e:
            print(f"  ❌ download failed: {e}")
            sources_meta[name] = {
                **prev,
                "status": "error",
                "error": str(e)[:300],
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

    meta["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["ref"] = ref
    return meta


def check_vllm_tree(ref: str, token: Optional[str]) -> bool:
    print(f"\n=== vLLM backbone check (ref={ref}) ===")
    try:
        data = api_request(VLLM_TREES_API.format(ref=ref), token, timeout=180)
    except Exception as e:
        print(f"  ❌ cannot fetch vllm tree: {e}")
        return False
    paths = {t["path"] for t in data.get("tree", []) if t.get("type") == "blob"}
    all_ok = True
    for p in VLLM_BACKBONE_PATHS:
        if p in paths:
            print(f"  ✅ {p}")
        else:
            print(f"  ❌ MOVED/MISSING: {p}")
            all_ok = False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh upstream reference material for vllm-rocm-pr-review")
    parser.add_argument("--ref", default=DEFAULT_REF, help=f"Upstream branch/tag (default: {DEFAULT_REF})")
    parser.add_argument("--token", help="GitHub PAT (optional; helps rate limits)")
    parser.add_argument("--check-vllm-tree", action="store_true",
                        help="Verify vLLM ROCm backbone paths still exist in vllm main")
    args = parser.parse_args()

    print(f"Updating upstream snapshots (ref={args.ref}) → {UPSTREAM_DIR}")
    meta = update_sources(args.ref, args.token)
    save_metadata(meta)

    updated = sum(1 for s in meta["sources"].values() if s.get("status") == "updated")
    errors = sum(1 for s in meta["sources"].values() if s.get("status") == "error")
    print(f"\nSummary: {updated} updated, "
          f"{sum(1 for s in meta['sources'].values() if s.get('status') == 'unchanged')} unchanged, "
          f"{errors} errors")
    print(f"Metadata: {METADATA_FILE}")

    tree_ok = True
    if args.check_vllm_tree:
        tree_ok = check_vllm_tree(args.ref, args.token)

    if errors == len(SOURCES):
        print("❌ All sources failed — nothing was updated.")
        return 1

    print("\nNext: curation loop — if any source SHA changed since the last curation, "
          "diff the refreshed upstream docs against references/vllm-rocm-review-rules.md "
          "and propose updates to the user before rewriting curated files.")
    if not tree_ok:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
