---
name: vllm-pr-desc-generator
description: Generate a vLLM-style PR title and description (Overview / Claims / Validation / Details, following the current `.github/PULL_REQUEST_TEMPLATE.md`) from a GitHub PR's code changes. Use when the user provides a vLLM PR link or number and asks to generate, write, or draft a PR description. Triggered by requests like "帮我生成PR描述", "generate PR description for vllm PR 12345", "帮我写vllm PR的描述", "draft a PR desc for https://github.com/vllm-project/vllm/pull/12345".
---

# vllm-pr-desc-generator

Generate a vLLM-style PR title and description from code changes and save to `./outputs/`.

The format follows vLLM's current PR template ([`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/vllm-project/vllm/blob/main/.github/PULL_REQUEST_TEMPLATE.md)) and the "Pull Request Contents" section of vLLM's `pr-checklist` skill (`.agents/skills/pr-checklist/SKILL.md`). If either has changed upstream, follow the upstream version.

Take `./references/example-pr-desc.md` as a style reference for the output.

## Workflow

1. **Parse PR number** — extract from URL or direct number
2. **Fetch PR data** — run `scripts/fetch_pr_data.py`
3. **Read the JSON** — load into context
4. **Analyze code changes** — understand the diff, purpose, and impact
5. **Generate PR title + description** — write vLLM-format title and description
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
python3 ~/.claude/skills/vllm-pr-desc-generator/scripts/fetch_pr_data.py <PR_NUMBER> \
    --output /tmp/vllm_pr_<PR_NUMBER>.json
```

Optional flags:
- `--token <token>` — GitHub PAT (not needed if `gh` CLI is authenticated)
- `--max-diff-chars <N>` — limit diff size (default 80 000)

## Step 3: Analyze Code Changes

Read `/tmp/vllm_pr_<PR_NUMBER>.json`. Focus on:

- **`diff`** and **`files`**: Understand what code changed, which modules are affected
- **`pr.title`**: Check the existing title and its bracketed tags (see Step 4 for the tag conventions)
- **`pr.body`**: Check if the author already provided context or linked issues
- **`issue_comments` / `review_comments`**: Extract reviewer feedback, design decisions, test suggestions

Categorize the change type:
- **Feature**: New functionality, new model support, new API
- **BugFix**: Correctness fix, crash fix, regression fix
- **Refactor**: Code reorganization without behavior change
- **Performance**: Optimization, memory reduction, throughput improvement
- **Docs**: Documentation only
- **CI/Build**: Build system, CI pipeline, dependency changes

## Step 4: Generate PR Title and Description

### PR Title

PR titles must start with one or more bracketed tags, followed by a space and a concise description, e.g. `[Bugfix][Scheduler] Fix priority handling`. Tags are stacked without spaces. Prefer these well-known tags; use a more specific one only when none fits:

- **Type tags**: `[Bugfix]`, `[Feature]`, `[Perf]`, `[Refactor]`; `[CI]` / `[CI/Build]`, `[Test]`, `[Doc]` / `[Docs]`; `[Misc]` when nothing else fits.
- **Scope tags**: `[Model]`, `[Frontend]`, `[Rust Frontend]`, `[Core]`, `[Kernel]`; `[Model Runner V2]` / `[MRV2]`, `[Attention]`, `[Multimodal]`, `[Quantization]`, `[MoE]`, `[Spec Decode]`, `[LoRA]`, `[KV Connector]`, `[KV Offload]`; `[ROCm]`, `[XPU]`, `[CPU]`, or `[Hardware][Vendor]` (e.g. `[Hardware][AMD]`).

If the existing title already follows these rules and matches the change, keep it; otherwise propose a corrected one.

### PR Description

The output file MUST start with the title, followed by the four template sections in this order, then the template's checklist:

```markdown
**Title:** `[Type][Scope] Concise description`

## Overview

<1-2 lines summarizing the change. Link relevant issues and predecessor/blocking PRs, e.g. "Fixes #1234", "Follow-up to #5678".>

## Claims

<A few concrete bullets stating what this PR accomplishes: speedups, feature support, bug fixes, etc.>

## Validation

<Evidence that each claim is met, plus steps to reproduce: commands, test runs, accuracy evals, benchmarks.>
<Compact tables with configuration, baseline, changed result, and reproduction commands.>

## Details

<Motivation, design decisions, root cause, tradeoffs, and limitations — concisely.>
<Optional "Key changes" bullets grouped by module/file when the diff spans many files.>
<Collapse lengthy logs or supporting material with <details>.>

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [ ] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
```

Copy the checklist verbatim and leave every box unchecked — the author ticks them. Do not include the `BEFORE SUBMITTING ...` footer; GitHub Actions strips it anyway.

### Writing Guidelines

- **Overview**: 1-2 lines only. Lead with what the change does and why; put everything else in Details.
- **Claims**: Concrete and verifiable — "Fix OOM when loading large MoE models with TP=8" beats "Fix a bug". Each claim should be backed by something in Validation. By change type:
  - Bugfix — which issue is resolved and how it can be verified.
  - Feature — what functionality is added and how to use it.
  - Refactor — the motivation and expected improvements.
  - Perf — targeted model / hardware / serving config / workload, and the measured gain.
- **Validation**: Copy-pasteable commands with model names and flags.
  - Bugfix: the reproducer, showing it fails before and passes after (including any new test).
  - Feature: a validated usage example.
  - Perf: prefer end-to-end serving benchmarks; for repeated runs report run count and variability; if only microbenchmarks are available, say so explicitly.
  - If CI cannot cover the changed path (hardware / checkpoint unavailable), state the gap and what was validated outside CI.
  - Only use results that appear in the PR body/comments or that the user provides. Never invent numbers — if evidence is missing, leave a clearly marked placeholder such as `<TODO: add benchmark result>` and tell the user.
- **Details**: For bug fixes, include the root cause. For PD-disaggregation / NIXL connector changes, add a short numbered flow of the request path across P and D instances. Highlight non-obvious choices, tradeoffs, limitations, compatibility impacts, and areas needing maintainer attention.
- **Conciseness**: Keep every section tight; link full logs or wrap them in `<details>` instead of pasting them inline.
- **Language**: Match the PR title language. If title is in English, write in English. If Chinese, write in Chinese. Default to English.
- **Tone**: Technical, concise, factual. No filler words.

## Step 5: Save Output

Save to `~/.claude/skills/vllm-pr-desc-generator/outputs/pr-<NUMBER>-desc.md`.

Tell the user the file path, the proposed title, a brief summary of what the PR does, and any `<TODO>` placeholders they need to fill in.

## Authentication

- **gh CLI available** (recommended): script auto-detects, no token needed
- **No gh CLI**: pass `--token <PAT>` with `repo` scope
- **No auth**: unauthenticated REST API used; rate limit applies (60 req/hr)
