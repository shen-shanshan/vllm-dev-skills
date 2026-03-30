---
name: vllm-pr-desc-generator
description: Generate a vLLM-style PR description (Purpose / Test Plan / Test Result) from a GitHub PR's code changes. Use when the user provides a vLLM PR link or number and asks to generate, write, or draft a PR description. Triggered by requests like "帮我生成PR描述", "generate PR description for vllm PR 12345", "帮我写vllm PR的描述", "draft a PR desc for https://github.com/vllm-project/vllm/pull/12345".
---

# vllm-pr-desc-generator

Generate a vLLM-style PR description from code changes and save to `./outputs/`.

Take `./references/example-pr-desc.md` as a style reference for the output.

## Workflow

1. **Parse PR number** — extract from URL or direct number
2. **Fetch PR data** — run `scripts/fetch_pr_data.py`
3. **Read the JSON** — load into context
4. **Analyze code changes** — understand the diff, purpose, and impact
5. **Generate PR description** — write vLLM-format description
6. **Save** — write to `./outputs/pr-<NUMBER>-desc.md`
7. **Confirm** — tell the user the output path

## Step 1: Parse PR Number

Extract `<PR_NUMBER>` from user input. Accepted formats:
- `https://github.com/vllm-project/vllm/pull/12345` → `12345`
- `vllm PR 12345` or `PR #12345` → `12345`
- Plain number `12345` → `12345`

For PRs from other repos (e.g., `vllm-project/vllm-ascend`), adjust the `--repo` flag in the fetch script accordingly. Default repo is `vllm-project/vllm`.

## Step 2: Fetch PR Data

```bash
python3 /Users/shanshan-shen/.claude/skills/vllm-pr-desc-generator/scripts/fetch_pr_data.py <PR_NUMBER> \
    --output /tmp/vllm_pr_<PR_NUMBER>.json
```

Optional flags:
- `--token <token>` — GitHub PAT (not needed if `gh` CLI is authenticated)
- `--max-diff-chars <N>` — limit diff size (default 80 000)

## Step 3: Analyze Code Changes

Read `/tmp/vllm_pr_<PR_NUMBER>.json`. Focus on:

- **`diff`** and **`files`**: Understand what code changed, which modules are affected
- **`pr.title`**: Infer the PR category prefix (e.g., `[Feature]`, `[BugFix]`, `[Misc]`, `[CI/Build]`, `[Doc]`)
- **`pr.body`**: Check if the author already provided context or linked issues
- **`issue_comments` / `review_comments`**: Extract reviewer feedback, design decisions, test suggestions

Categorize the change type:
- **Feature**: New functionality, new model support, new API
- **BugFix**: Correctness fix, crash fix, regression fix
- **Refactor**: Code reorganization without behavior change
- **Performance**: Optimization, memory reduction, throughput improvement
- **Docs**: Documentation only
- **CI/Build**: Build system, CI pipeline, dependency changes

## Step 4: Generate PR Description

Write the description following the vLLM PR template. The output MUST include these sections:

```markdown
## Purpose

<1-3 paragraphs explaining WHY this change is needed and WHAT it does>

<If there are linked issues, reference them: "Fixes #1234" or "Related: #5678">

### Key Changes

<Bulleted list of the most important code changes, organized by module/file>
<Each bullet: **bold file/class name** followed by a concise description>
<Group related changes together>

## Test Plan

<Commands or steps to verify the change>
<Use code blocks for CLI commands>

## Test Result

<Expected results, benchmarks, or before/after comparisons>
<Use tables for benchmark data when applicable>

## (Optional) Documentation Update

<Only include if docs were changed or should be changed>
```

### Writing Guidelines

- **Purpose section**: Lead with the problem/motivation, then describe the solution approach. Be specific — "Fix OOM when loading large MoE models" is better than "Fix a bug".
- **Key Changes**: Group by module. Use bold for file/class names. Each bullet should be self-contained and informative.
- **Test Plan**: Provide concrete, copy-pasteable commands. Include model names, flags, and expected behavior.
- **Test Result**: Prefer tables for quantitative results. For bug fixes, describe the before/after behavior.
- **Language**: Match the PR title language. If title is in English, write in English. If Chinese, write in Chinese. Default to English.
- **Tone**: Technical, concise, factual. No filler words.

## Step 5: Save Output

Save to `/Users/shanshan-shen/.claude/skills/vllm-pr-desc-generator/outputs/pr-<NUMBER>-desc.md`.

Tell the user the file path and briefly summarize what the PR does.

## Authentication

- **gh CLI available** (recommended): script auto-detects, no token needed
- **No gh CLI**: pass `--token <PAT>` with `repo` scope
- **No auth**: unauthenticated REST API used; rate limit applies (60 req/hr)
