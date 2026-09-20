---
name: vllm-rocm-pr-review
description: Review AMD/ROCm-related pull requests from the vllm-project/vllm GitHub repository and produce a single combined Chinese report: a detailed PR summary (summary, background & motivation, code-change analysis with Mermaid diagrams, technical principles, discussion highlights, risk table) followed by severity-sorted, type-categorized ROCm review findings, a verdict, and copy-paste English review comments with file + diff line numbers ready for GitHub PR review. Use when the user provides a vllm PR number or link and asks to review it, especially for ROCm/AMD/aiter/mori related changes. Triggered by requests like "review vllm PR 12345", "帮我review vllm的PR 12345", "vllm rocm PR review", "review这个rocm PR".
---

# vLLM ROCm PR Review

Review a `vllm-project/vllm` PR through an AMD/ROCm lens, then write a **single combined Chinese report** to `./outputs/pr-<NUMBER>-review.md`: Part A (§1–6) is a detailed PR summary in the style of the `vllm-pr-summary` skill, Part B (§7–8) is the ROCm review findings + verdict, and §9 holds 2–5 copy-paste English comments with file:line targets for the GitHub PR review UI. The `vllm-pr-summary` skill itself remains a separate, unchanged skill for standalone summary requests.

Knowledge sources (kept in `references/`):

| File | Content | When to read |
|------|---------|--------------|
| `references/vllm-rocm-review-rules.md` | Curated rule catalog adapted from the ATOM/aiter review skills to vLLM's ROCm surface | The mandatory checklist (Step 7) |
| `references/upstream/atom-review-pr-skill.md` | Full upstream ATOM review-pr skill | Deep reviews / new rule discovery |
| `references/upstream/aiter-review-pr-skill.md` | Full upstream aiter review-pr skill | Deep reviews / new rule discovery |
| `references/upstream/mori-readme.md` + `mori-structure.json` | mori (AMD comm library) README + repo tree snapshot | PRs touching mori / MORI-EP / moriio |
| `references/example-report.md` | Style anchor for the combined report (Part A detailed summary + Part B findings) | Before writing the report |

**Context economy**: for quick reviews, read only the rule categories mapped by the PR type (Step 5) plus the core discipline section of the rules file. For deep reviews (Tier-1 backbone files touched, mori changes, perf claims), read the full rules file.

## Workflow

1. Fetch PR data — `scripts/fetch_pr_data.py`
2. ROCm relevance gate
3. Summary analysis → Part A content (detailed summary)
4. Semantic understanding (5 questions)
5. PR-type classification → mandatory rule categories
6. Backbone-file risk assessment
7. Apply the rule checklist
8. AI-code diagnostic + blind-spot check
9. Write the combined report (incl. §9 copy-paste English comments) to `./outputs/` and tell the user the path

Non-ROCm path (gate says 不相关): Steps 1 → 2 → 3, then only housekeeping (HK) checks + verdict — no ROCm rule categories — then Step 9.

## Step 1: Fetch PR Data

```bash
python3 /Users/shanshan-shen/.claude/skills/vllm-rocm-pr-review/scripts/fetch_pr_data.py <PR_NUMBER> \
    --output /tmp/vllm_rocm_pr_<PR_NUMBER>.json --include-ci
```

The JSON contains:

| Key | Content |
|-----|---------|
| `pr` | PR metadata: title, body, author, state, labels, additions/deletions, branch names, linked reviews |
| `diff` | Full unified diff of all changed files |
| `files` | Per-file stats: filename, status, additions, deletions, patch |
| `issue_comments` | General discussion comments |
| `review_comments` | Inline code review comments |
| `reviews` | Top-level reviews (approvals, change requests) |
| `checks` | CI check status (only when `--include-ci` was passed; absent = CI status unknown) |

For large diffs, group `files[].patch` by module and focus on the ROCm-touched files first.

## Step 2: ROCm Relevance Gate

Scan the PR title, body, and changed file paths for ROCm markers: `rocm`, `amd`, `hip`, `aiter`, `mori`, `gfx9`, `mi300/mi350/mi355`, `mxfp`, `fp8`, `rccl`, `hipblaslt`, `hipgraph`, plus known paths (`vllm/platforms/rocm.py`, `vllm/_aiter_ops.py`, `vllm/v1/attention/backends/rocm_*`, `vllm/models/*/amd/`, `requirements/rocm.txt`, `requirements/kv_connectors_rocm.txt`, `.buildkite/ci_config_rocm.yaml`, `docker/Dockerfile.rocm*`, `csrc/rocm/`).

- **Fully ROCm-related** → full checklist (Steps 3–8).
- **Not ROCm-related** → state this in the report's intro line, but still produce the full Part A detailed summary (§1–6) plus a light Part B: housekeeping (HK) findings + verdict only. Note in §7 that aiter/mori/ROCm-specific rules were skipped.
- **Partially related** (e.g. a CUDA PR touching a shared file also used by ROCm) → full checklist, but scope per file: name exactly which files are ROCm-exposed and check shared paths for cross-backend regressions (A2).

## Step 3: Summary Analysis (Part A Content)

Synthesize the PR into the detailed-summary sections before running the ROCm checklist. From the fetched JSON (PR body, linked issues, `files[].patch` grouped by module, `issue_comments`, `review_comments`, `reviews`):

1. **总结**: 2–4 sentences — what problem, what core approach.
2. **背景与动机**: why now, what it enables, linked issues / roadmap context.
3. **代码修改分析**: group changed files by module with one-line descriptions (the module table), identify the architecture or control flow worth one or more Mermaid diagrams, and bullet the key implementation details (new classes, changed APIs, algorithm changes, config additions).
4. **技术原理**: which background concepts a reviewer needs (SPMD, paged attention, PD disaggregation, MoE, scaled_mm/mxfp, CUDA graphs, RDMA/GPUDirect, …) — 2–5 short paragraphs.
5. **讨论亮点**: substantive reviewer comments / design debates (skip lgtm/thanks).
6. **风险表**: generic High/Medium/Low risks across 正确性 / 性能 / 兼容性 / 测试覆盖 / 可维护性. Keep rows terse; ROCm-specific actionable findings belong in Part B — a row may reference a finding by title instead of repeating the argument.

This step and the 5 semantic questions of Step 4 overlap deliberately: Step 3 answers "what is this PR" for the reader; Step 4 answers "what could break on ROCm" for the checklist.

## Step 4: Semantic Understanding (5 Questions)

Answer before applying any rules:

1. **What changed computationally?** Which kernel path / data flow / formula — not "improves attention".
2. **Hardware + model scope?** gfx942 / gfx950? Which model families (DeepSeek V3/V3.2/V4, Kimi K3, Inkling, MiniMax M3, GPT-OSS)? TP/EP configs?
3. **API surface changes?** New aiter/mori op usage, new kwargs, changed signatures, new env vars?
4. **Performance claim mechanism?** WHY faster — fewer HBM round-trips, fewer launches, better tiling?
5. **WHY or WHAT?** Does the description explain the mechanism or only the action? Surface-only → elevated AI-code risk.

## Step 5: PR-Type → Mandatory Rule Categories

| PR touches | Mandatory rules (see rules file) |
|---|---|
| aiter op call / `vllm/_aiter_ops.py` / `vllm/kernels/aiter_ops.py` | A1, B1–B6, C1–C4, D1, D1b, D5–D9, E1 |
| Attention backend (`rocm_attn.py`, `rocm_aiter_*`, mla/) | A2, B1–B3, C1–C3, D1, D8, G1, P1–P4 |
| MoE / quantization (scaled_mm, mxfp4/mxfp8, aiter_moe experts) | C2, D1, D4, A1, P1–P5 |
| mori / comm / kv_transfer (`moriio/`, `aiter_custom_all_reduce.py`, `prepare_finalize/mori.py`) | E1–E5, G1, G1b, F1 + mori checklist (rules file appendix) |
| Env vars / platform (`vllm/envs.py`, `vllm/platforms/rocm.py`) | D2, B1, B5/B6, HK (env-var registration) |
| torch.compile pass / custom op | B3, D6, D7 |
| requirements / CI / docker | E4, HK |
| Any PR | HK always |

## Step 6: Backbone-File Risk Assessment

Tier summary (full table in the rules file):

- **Tier 1 — silent numeric corruption possible**: `vllm/v1/attention/backends/rocm_attn.py`, `rocm_aiter_fa.py`, `rocm_aiter_unified_attn.py`, `mla/rocm_aiter_mla*.py`, `vllm/model_executor/kernels/linear/scaled_mm/{rocm,aiter}.py`, `mxfp4/aiter.py`, `mxfp8/rocm_native.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`. Every PR touching these gets a deep review.
- **Tier 2 — dispatch/plumbing, wrong dispatch = silent bypass**: `vllm/platforms/rocm.py`, `vllm/_aiter_ops.py`, `vllm/kernels/aiter_ops.py`, `fused_moe/router/aiter_shared_routed_fused_moe_router.py`, `vllm/model_executor/kernels/mhc/aiter.py`.
- **Tier 3 — structural**: `requirements/rocm.txt`, `requirements/kv_connectors_rocm.txt`, `.buildkite/ci_config_rocm.yaml`, `docker/Dockerfile.rocm*`, `fused_moe/configs/*.json`.

Heuristics: (Q1) Tier-1 file touched but only tested on CUDA? (Q2) dispatch gate changed without exercising the new path on ROCm HW? (Q3) shared helper used by both backends changed without cross-backend validation?

**Discipline**: cross-file verification — read the full function/symbol family (including callers and headers), not just the diff hunk, before claiming "no sync" / "missing branch" / "wrong dispatch". "Compiles" ≠ "runs correctly on MI300/MI355".

## Step 7: Rule Checklist

Read `references/vllm-rocm-review-rules.md`. For deep reviews read it fully; otherwise read the core discipline section + the categories mapped in Step 5. The rules come from the ATOM/aiter review skills (rule codes A/B/C/D/E/F/G/P/HK are internal — never show them in the report).

Key gates that always apply:

- **🔴 gate**: before firing any blocking finding, name the concrete triggering input (shape / dtype / arch / config). If you cannot, downgrade to ⚠️ or drop it.
- **CI-failure classification**: a red check is not automatically the PR's fault. Compare against main; infra flakes exist; expired logs on old runs are meaningless.
- **Number provenance**: trace every benchmark number in the PR to its source (script output / config). Untraceable numbers are `[unverified]` — never repeat them as fact.

## Step 8: AI-Code Diagnostic + Blind-Spot Check

Run the structural checks from the rules file (AI 诊断 section): hallucinated-symbol sweep (does every aiter/mori symbol referenced actually exist — check against `mori-structure.json` / aiter docs), twin divergence (near-identical blocks in two files that diverged), safety theater, tests calibrated to pass, magic constants (fnuz 240/448, LDS limits).

Finish with the blind-spot question: *is there any correctness risk, resource hazard, or behavioral edge case in this diff that none of the above caught?* If yes, add it.

## Step 9: Write the Report

Save to `/Users/shanshan-shen/.claude/skills/vllm-rocm-pr-review/outputs/pr-<NUMBER>-review.md`. Chinese (Simplified) unless the user requests English. One file, two parts separated by `---`. Target ≤ ~750 lines: Part A ~250–400, Part B ~150–300, §9 ~40–60. 3–10 findings (fewer on the non-ROCm path).

```markdown
# PR #<NUMBER>: <Title>

> **Author**: @author | **State**: OPEN/MERGED/CLOSED | **Date**: YYYY-MM-DD
> **Branch**: `head` → `base` | **Labels**: ...
> **Changes**: +X -Y lines across N files | **ROCm 相关性**: 完全相关 / 部分相关 / 不相关
> 本报告分两部分：§1–6 为 PR 详细总结，§7–8 为 ROCm review 意见。

---

## 1. 总结 (Summary)

2–4 sentences: what problem, what core approach.

## 2. 背景与动机 (Background & Motivation)

From the PR body + linked issues: why now, why this approach. (Supersedes the old 动机 section.)

## 3. 代码修改分析 (Code Change Analysis)

### 3.1 修改的模块

Table or bullets grouped by module, one line each. (Supersedes the old 代码改动总结 section.)

### 3.2 架构 / 流程图

At least one Mermaid diagram — this part inherits the diagram style of `vllm-pr-summary`:
- **flowchart TD** for execution flow / decision logic
- **sequenceDiagram** for component interactions
- **classDiagram** for new classes / interface changes
- **graph LR** for data / dependency relationships

Chinese labels; one diagram per concept. For very large PRs (> 20 files), draw a high-level module dependency graph rather than per-function flow.

### 3.3 关键实现细节

Bullets: new classes, changed APIs, algorithm changes, config additions.

## 4. 涉及的技术原理 (Technical Principles)

2–5 short paragraphs/bullets on the background concepts needed to understand the PR.

## 5. 评论区讨论亮点 (Discussion Highlights)

Only substantive reviewer comments / design debates from `issue_comments`, `review_comments`, `reviews`. Skip lgtm/thanks. (Supersedes the old 现有讨论 section.)

## 6. 风险与潜在问题 (Risk Analysis)

| 风险 | 严重程度 | 说明 |
|------|---------|------|

Generic High/Medium/Low risks (正确性 / 性能 / 兼容性 / 测试覆盖 / 可维护性). Terse rows; ROCm-specific actionable findings go in §7 — reference them by title instead of repeating the argument.

---

## 7. Review 意见 (Findings)

Summary table first (意见类型 × 数量), then findings **sorted by severity**: 🔴 必须修复 → ⚠️ 建议修复 → 📝 建议/备注.
Each finding:

**🔴【类型】标题** `[已验证]/[推测]`
- **问题**: what exactly, with file/line
- **影响**: what breaks at runtime if unfixed
- **行动**: ends with "作者应当…" / "建议作者…" / "建议 review 时追问…"

类型: 【正确性】【性能】【设计】【注释/文档】【测试】【兼容性】【可维护性】【安全】.

Non-ROCm path: only HK findings (env-var registration, CI/requirements/docker consistency, docs), and add a line: "非 ROCm 相关，已跳过 ROCm 专项规则（A/B/C/D/E/F/G/P）。"

## 8. 结论 (Verdict)

✅ LGTM / ⚠️ NEEDS WORK / 🔴 BLOCK + 1–2 sentence rationale.

## 9. 英文 Review 评论 (Copy-Paste English Comments)

2–5 comments derived from the top findings of §7 (🔴 first, then the strongest ⚠️; never from `[推测]` findings), ready to paste into the GitHub PR review UI. Format per comment:

**C<N>** `path/to/file.py:<start>-<end>` — 🔴 blocking / ⚠️ comment

```text
<English comment text>
```
```

Verdict rules: 🔴 BLOCK requires at least one finding with a concrete triggering input. ⚠️ NEEDS WORK when any ⚠️ findings exist. ✅ LGTM otherwise. No findings at all → omit section 7 entirely and say so in 结论.

§9 guidance:

- **Line numbers** are new-file (right side of the diff) line numbers, computed from the `files[].patch` hunk headers `@@ -a,b +c,d @@` — `+c` is the hunk's first new-file line; count forward from there, added and context lines included. Single line: `file.py:42`; range: `file.py:42-48` (start–end).
- **GitHub constraint**: inline comments can only attach to lines present in the diff. If the offending line is unchanged, anchor on the nearest changed line in the same hunk and say so in the comment.
- One comment per finding; keep each 3–8 sentences in English: conclusion first, then why it matters, then the ask ("I think …" / "Could you …" / "Consider …"). Polite and direct; no severity emojis inside the comment text.
- Non-ROCm path: derive from the HK findings (usually fewer comments), same format.

Also give the user a one-line summary in chat with the output path.

## Updating the Skill (一键更新)

The skill refreshes its upstream knowledge with one command:

```bash
python3 /Users/shanshan-shen/.claude/skills/vllm-rocm-pr-review/scripts/update_skill.py
```

What it does:
1. Downloads the latest ATOM review-pr SKILL.md, aiter review-pr SKILL.md, mori README, and mori repo tree into `references/upstream/` (skips sources whose SHA is unchanged — idempotent).
2. Writes `.update-metadata.json` (fetch date + per-source SHAs).
3. With `--check-vllm-tree`: verifies the ~25 backbone paths in the rules file still exist in vllm main (OK/MOVED/MISSING) — a tripwire for vLLM layout drift.

It **never rewrites** `references/vllm-rocm-review-rules.md` or this SKILL.md — rule distillation is a judgment task. After running it, the curation loop:

1. If `.update-metadata.json` shows no SHA change since the last curation, skip.
2. Otherwise diff the refreshed upstream docs against the curated rules file: new rules upstream? renamed rule triggers? moved vLLM paths? updated aiter/mori APIs?
3. **Present the proposed changes to the user and only rewrite curated files after approval.** Append a changelog row (date, source SHAs, what changed) to the rules file's 更新日志 section.

Users can also trigger this by saying "更新 vllm-rocm-pr-review skill" or "update the rocm review skill".

## Authentication

- **gh CLI available** (recommended): fetch script auto-detects, no token needed
- **No gh CLI**: pass `--token <PAT>` with `repo` scope
- **No auth**: unauthenticated REST API used; rate limit applies (60 req/hr)
