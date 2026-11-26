---
name: vllm-vit-cg-pr-review
description: Review vision-encoder (ViT) CUDA graph pull requests in vllm-project/vllm — the model-integration, framework, bugfix and doc/test PRs tracked by issue #38175 ("Support ViT Full CUDA Graph") — and produce a single combined Chinese report: a detailed PR summary (summary, background, code-change analysis with Mermaid diagrams, technical principles, discussion highlights, risk table) followed by severity-sorted ViT CG review findings checked against the latest vllm main (SupportsEncoderCudaGraph protocol, EncoderCudaGraphManager budgets/shared buffers, capture/replay/eager equivalence, doc/example/CI integration, benchmark validity), a verdict, copy-paste English review comments with file + diff line numbers, and a ≤100-word English overall summary. Review judgment follows the learned review habits of the ViT CG maintainers (@shen-shanshan, @Isotr0py). Use when the user gives a vllm PR number/link about encoder/ViT CUDA graphs, or asks to review the open PRs of tracker #38175. Triggered by requests like "review vllm PR 63074", "帮我 review 这个 ViT CUDA graph PR", "review #38175 里 open 的 PR", "vit cg pr review", "encoder cudagraph PR review".
---

# vLLM ViT CUDA Graph PR Review

Review a `vllm-project/vllm` PR through the ViT (encoder) CUDA graph lens and write a **single combined Chinese report** to `./outputs/pr-<NUMBER>-review.md`. The report format is the same as `vllm-rocm-pr-review` (Part A §1–6 detailed summary, Part B §7–8 findings + verdict, §9 copy-paste English comments, §10 English overall summary), so the Pingboard dashboard renders both the same way; the ROCm/aiter/mori rules are replaced by ViT CG rules.

Knowledge sources (kept in `references/`):

| File | Content | When to read |
|------|---------|--------------|
| `references/vit-cg-review-rules.md` | Rule catalog + framework map (main snapshot) distilled from the maintainers' reviews | The mandatory checklist (Step 7) |
| `references/reviewer-playbook.md` | How @shen-shanshan and @Isotr0py review ViT CG PRs: division of focus, thinking order, English phrasing templates, severity habits | Before Step 4 and before writing §9/§10 |
| `references/upstream/reviewer-comments.md` | Raw corpus: every review comment of the two maintainers on tracker PRs (with diff hunks and author replies) | Looking for precedent on a specific model/pattern (`grep` it) |
| `references/example-report.md` | Style anchor for the combined report | Before writing the report |

**Context economy**: for a doc/test-only PR read the core discipline + framework map + D/T/HK of the rules file. For model integrations and framework changes read the rules file fully.

## Workflow

1. Fetch PR data — `scripts/fetch_pr_data.py`
2. Sync vllm main + relevance gate + PR type
3. Summary analysis → Part A content
4. Semantic understanding (5 questions)
5. PR type → mandatory rule categories
6. Cross-path verification against main (capture / replay / eager / manager)
7. Apply the rule checklist
8. AI-code diagnostic + blind-spot check
9. Write the combined report to `./outputs/` and tell the user the path

**Batch mode** ("review #38175 里 open 的 PR"): list candidates first, then run Steps 1–9 per PR (one PR per sub-agent works well; they are independent):

```bash
python3 ~/.claude/skills/vllm-vit-cg-pr-review/scripts/list_tracker_prs.py   # open PRs + report status (none / up-to-date / stale)
```

Skip drafts that are explicitly someone's exploration (e.g. a maintainer's `[Draft]` refactor) or unrelated stale PRs unless the user asks; include drafts that the tracker lists as model integrations. Re-review only `none` / `stale` PRs unless asked.

## Step 1: Fetch PR Data

```bash
python3 ~/.claude/skills/vllm-vit-cg-pr-review/scripts/fetch_pr_data.py <PR_NUMBER> \
    --output /tmp/vllm_vit_cg_pr_<PR_NUMBER>.json --include-ci
```

| Key | Content |
|-----|---------|
| `pr` | Metadata: title, body, author, state, isDraft, labels, additions/deletions, branch names, `headRefOid` |
| `head_sha` | Normalized 40-char head SHA (use it in the front matter) |
| `diff` | Unified diff (truncated at 80k chars — for big PRs read `files[].patch` or `gh pr diff`) |
| `files` | Per-file stats + `patch` |
| `issue_comments` / `review_comments` / `reviews` | Discussion, inline comments, approvals |
| `checks` | CI status (only with `--include-ci`) |

To read the PR's full new files (not just hunks): `gh pr checkout` is **not** allowed in `~/vllm/vllm` if it is dirty and never create worktrees there — instead `git fetch upstream pull/<N>/head:pr-<N>` and read with `git show pr-<N>:<path>`.

## Step 2: Sync Main + Relevance Gate + PR Type

```bash
cd ~/vllm/vllm && git fetch upstream main -q && git log -1 --format='%h %cd' upstream/main
```

All framework reads use `git show upstream/main:<path>` (the local `origin` may be a stale personal fork). Note the main SHA in §7's intro line.

Relevance markers: `SupportsEncoderCudaGraph`, `encoder_cudagraph*`, `EncoderCudaGraph*`, `cudagraph_mm_encoder`, `get_encoder_cudagraph_*`, `prepare_encoder_cudagraph_*`, `postprocess_encoder_output`, `test_vit_cudagraph.py`, `test_encoder_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`, `MODELS_SUPPORT_VIT_CUDA_GRAPH`, a reference to #38175.

- **Fully related** → full checklist.
- **Partially related** (e.g. a vision-tower refactor that touches a model already supporting encoder CG) → full checklist scoped to the CG-exposed code; check that the existing graph path still matches the eager path.
- **Not related** → full Part A, then HK-only Part B with the line "非 ViT CUDA graph 相关，已跳过 ViT CG 专项规则。"

PR type (decides Step 5): **model integration** (a model implements the protocol) / **framework** (manager, defs, interfaces, compilation config) / **bugfix** / **doc-test-CI**.

## Step 3: Summary Analysis (Part A)

From the JSON (body, linked issues, `files[].patch` by module, comments, reviews):

1. **总结**: 2–4 sentences — which model/framework piece, what core approach (what is inside the graph, what stays eager).
2. **背景与动机**: tracker context (#38175), which earlier PR it follows (Qwen3-VL #35963, InternVL #41759, DeepSeek-OCR multi-path #43586, …), claimed results.
3. **代码修改分析**: module table; at least one Mermaid diagram — for model integrations draw the capture / replay / eager-fallback flow showing which sub-modules are in the graph and which buffers are recorded; for framework PRs the manager control flow.
4. **技术原理**: the model's vision encoder structure (patching, tiling/crops, position embedding, attention kind, merger/projector), budget-based capture, buffer padding, multi-path / capture axes as relevant — 2–5 paragraphs.
5. **讨论亮点**: substantive comments (bot reviews such as claude[bot]/copilot only if they raised something real; say whether it was addressed).
6. **风险表**: terse High/Medium/Low rows (正确性 / 性能 / 兼容性 / 测试覆盖 / 可维护性); reference Part B findings by title.

## Step 4: Semantic Understanding (5 Questions)

Read `references/reviewer-playbook.md` §2 first, then answer:

1. **What is inside the graph?** Exact sub-modules captured (patch embed? pos-embed interpolation? merger/projector/resampler?) and what remains eager (preprocessing, post-merge, newline tokens, audio tower).
2. **What makes the input shape dynamic, and how is it fixed?** Variable patches/tiles/frames → token budget, `max_batch_size`, `max_frames_per_batch`, `paths`, `capture_axes`; which buffers are recorded and how they are padded.
3. **API surface vs main?** Which protocol methods are implemented/overridden; any new config fields, interface methods, manager changes; any use of removed interfaces.
4. **Performance mechanism and evidence?** Launch-bound regime (small images/few tiles) vs padding cost (large/fixed tiles); does the benchmark actually send images/videos to the encoder?
5. **WHY or WHAT?** Does the description explain the design choices (why this budget range, why this path split) or only list actions? Surface-only → elevated AI-code risk.

## Step 5: PR Type → Mandatory Rule Categories

| PR type | Mandatory rules (see rules file) |
|---|---|
| Model integration | I1–I3, B1–B4, N1–N4, V1–V5, C1–C5, T1–T3, D1–D4, P1–P3, H1–H2, HK |
| Framework (manager / defs / interfaces / config) | M1–M4, B2–B4, C1–C3, C5, T4, D1, D4, P1–P4, HK |
| Bugfix | the rules of the touched area + T4 (regression test) + HK |
| Doc / test / CI | D1–D4, T1–T3, HK |

## Step 6: Cross-Path Verification Against Main

Before claiming any correctness finding, for model integrations read in full (PR head version) and compare:

- `get_encoder_cudagraph_config()` — `modalities`, `buffer_keys`, `paths`, `capture_axes`, `padding_logics`, `max_frames_per_video`, `out_hidden_size`
- `prepare_encoder_cudagraph_capture_inputs()` vs `prepare_encoder_cudagraph_replay_buffers()` — same key set, same dtypes/trailing dims (shared prefix buffers!), same per-item layout
- `encoder_cudagraph_forward()` vs the original eager `forward()` / `encoder_eager_forward()` — same math (interpolation mode, RoPE, merge order, projector), no host sync inside
- `postprocess_encoder_output()` / `select_encoder_cudagraph_items()` — token accounting matches `EncoderItemSpec.output_tokens`; clone semantics
- Manager call sites in `upstream/main:vllm/v1/worker/encoder_cudagraph.py` (`_capture_budget_graph`, `_run_budget_graph`, `_execute_local`, `_dp_shard`) — the PR must match how main actually calls it

Find the closest already-merged sibling (same tower family / same pattern) on main and diff the approach; divergence without a reason is a finding. Search precedent with `grep -n "<model or pattern>" references/upstream/reviewer-comments.md`.

**Discipline**: "compiles and the e2e test passes" ≠ "graph output equals eager output for all supported shapes". A graph-vs-eager test that only uses one image size proves little.

## Step 7: Rule Checklist

Read `references/vit-cg-review-rules.md` (fully for model integration / framework). Rule codes are internal — never show them in the report. Always-on gates:

- **🔴 gate**: name the concrete triggering input (resolution / tile count / frames / budget / batch size / TP-DP config), else downgrade.
- **Number provenance**: benchmarks must exercise the encoder (`vllm bench latency` does not send images). Untraceable → `[unverified]`.
- **Main drift**: if the PR is behind main's interfaces, say exactly which merged PR changed what.
- **CI classification**: compare failures with main before blaming the PR.

## Step 8: AI-Code Diagnostic + Blind-Spot Check

Run the AI 诊断 section of the rules file (hallucinated protocol methods/config fields, image↔video and capture↔replay twin divergence, claims vs tests, unreachable guards, self-calibrated tests, magic constants). Then ask: *is there any correctness risk, resource hazard, or behavioral edge case in this diff that none of the above caught?* If yes, add it.

## Step 9: Write the Report

Save to `~/.claude/skills/vllm-vit-cg-pr-review/outputs/pr-<NUMBER>-review.md`. Chinese (Simplified) unless the user asks for English. Read `references/example-report.md` first. Target ≤ ~750 lines; 3–10 findings.

**Front matter (required)** — read by the Pingboard dashboard (verdict + "PR changed since review"). Plain `key: value` lines only.

| Key | Value |
|-----|-------|
| `pr` | PR number |
| `repo` | `vllm-project/vllm` |
| `head_sha` | `head_sha` from the Step 1 JSON (40 chars) |
| `verdict` | `block` / `needs-work` / `lgtm` — must match §8 |
| `reviewed_at` | ISO 8601 UTC (`date -u +%Y-%m-%dT%H:%M:%SZ`) |
| `skill` | `vllm-vit-cg-pr-review` |
| `main_sha` | short SHA of `upstream/main` the review was checked against |

```markdown
---
pr: <NUMBER>
repo: vllm-project/vllm
head_sha: <40-char head SHA>
verdict: block | needs-work | lgtm
reviewed_at: YYYY-MM-DDTHH:MM:SSZ
skill: vllm-vit-cg-pr-review
main_sha: <short SHA>
---

# PR #<NUMBER>: <Title>

> **Author**: @author | **State**: OPEN/MERGED/CLOSED (draft?) | **Date**: YYYY-MM-DD
> **Branch**: `head` → `base` | **Labels**: ...
> **Changes**: +X -Y lines across N files | **ViT CG 相关性**: 完全相关 / 部分相关 / 不相关 | **类型**: 模型集成 / 框架 / Bugfix / 文档测试
> 本报告分两部分：§1–6 为 PR 详细总结，§7–8 为 ViT CUDA graph review 意见（对照 vllm main `<short SHA>`）。

---

## 1. 总结 (Summary)
## 2. 背景与动机 (Background & Motivation)
## 3. 代码修改分析 (Code Change Analysis)
### 3.1 修改的模块
### 3.2 架构 / 流程图        (≥1 Mermaid; Chinese labels; one diagram per concept)
### 3.3 关键实现细节
## 4. 涉及的技术原理 (Technical Principles)
## 5. 评论区讨论亮点 (Discussion Highlights)
## 6. 风险与潜在问题 (Risk Analysis)
| 风险 | 严重程度 | 说明 |

---

## 7. Review 意见 (Findings)
## 8. 结论 (Verdict)
## 9. 英文 Review 评论 (Copy-Paste English Comments)
## 10. 英文总结评论 (Overall Review Summary)
```

§1 must work as a stand-alone summary: the dashboard shows it in the PR's AI Review panel.

**§7 Findings**: a summary table first (意见类型 × 数量), then findings sorted 🔴 必须修复 → ⚠️ 建议修复 → 📝 建议/备注. Each finding:

**🔴【类型】标题** `[已验证]/[推测]`
- **问题**: what exactly, with file:line
- **影响**: what breaks at runtime / for maintainers if unfixed
- **行动**: ends with "作者应当…" / "建议作者…" / "建议 review 时追问…"

类型: 【正确性】【性能】【设计】【接口同步】【测试】【注释/文档】【兼容性】【可维护性】. Add a short "集成清单" table before the findings for model-integration PRs (接口实现 / 文档表 / example / `test_vit_cudagraph.py` / 单测 / benchmark / 精度 → ✅/❌/⚠️).

**§8 Verdict**: ✅ LGTM / ⚠️ NEEDS WORK / 🔴 BLOCK + 1–2 sentences. 🔴 requires a finding with a concrete trigger; ⚠️ when any ⚠️ finding exists; ✅ otherwise. Follow the maintainers' habit: initial support with honest ❌︎ in the doc table and follow-ups is acceptable; wrong outputs / capture overflow / aliasing never are.

**§9 English comments** — 2–5 comments from the top findings (🔴 first, then the strongest ⚠️; never `[推测]` ones), phrased like the playbook templates (short, "I suppose…", "Can we…?", "FYI, #NNNNN"). Format:

**C<N>** `path/to/file.py:<start>-<end>` — 🔴 blocking / ⚠️ comment

```text
<English comment text>
```

- Line numbers are new-file (right side) numbers computed from the `@@ -a,b +c,d @@` hunk headers of `files[].patch`; single line `file.py:42`, range `file.py:42-48`.
- Inline comments only attach to lines in the diff; for unchanged lines anchor on the nearest changed line of the same hunk and say so.
- 3–8 sentences each, conclusion first, then why, then the ask; no emojis or rule codes in the text.

**§10 Overall summary** — always the last section: one `text` block, English, ≤ 100 words (count before saving), no headings/bullets/emojis/rule codes/line numbers. Structure: what the PR does and what is done well (specific) → the 1–3 most important asks (matching §9) → stance consistent with the verdict (`lgtm` supportive; `needs-work` names what must change; `block` names the blocking issue and its trigger). Only `[已验证]` content.

Finally give the user a one-line summary in chat with the output path. Optionally remind them that the report shows in the Pingboard dashboard (PR detail → AI Review; this skill's report takes priority over a `vllm-rocm-pr-review` report for the same PR) once pushed:

```bash
cd <vllm-dev-skills checkout> && git add skills/vllm-vit-cg-pr-review/outputs/pr-<NUMBER>-review.md \
  && git commit -m "Add ViT CG review for vllm PR #<NUMBER>" && git push
```

## Updating the Skill (一键更新)

```bash
python3 ~/.claude/skills/vllm-vit-cg-pr-review/scripts/update_skill.py --check-vllm-tree
```

1. Re-harvests every comment of @shen-shanshan / @Isotr0py on the PRs linked in #38175 into `references/upstream/reviewer-comments.md` and prints the PRs whose comment set changed since the last run (`.update-metadata.json`).
2. `--check-vllm-tree` verifies the framework files and symbols in the rules file's framework map still exist on vllm main (OK / CHANGED / MISSING).

It **never rewrites** the rules file, the playbook or this SKILL.md. Curation loop: read the new comments of the changed PRs and the CHANGED framework items → propose new/changed rules, playbook phrasing and framework-map rows to the user → rewrite only after approval, and append a row to the rules file's 更新日志. Triggered by "更新 vllm-vit-cg-pr-review skill" / "update the vit cg review skill".

## Authentication

- **gh CLI** (required by `list_tracker_prs.py` / `update_skill.py`, recommended for `fetch_pr_data.py`): `gh auth login`.
- **No gh CLI**: `fetch_pr_data.py --token <PAT>`; unauthenticated REST works but is limited to 60 requests/hour — not enough for batch reviews.
