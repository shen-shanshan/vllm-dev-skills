---
name: vllm-rfc-generator
description: Generate a vLLM-style RFC (Request for Comments) document based on user input. Use when the user wants to create an RFC for a major architectural change or design decision in vLLM. Triggered by requests like "帮我生成一个vLLM RFC", "create a vLLM RFC for ...", "写一个RFC关于...", "生成RFC文档".
---

# vLLM RFC Generator

Generate a vLLM-style RFC Markdown document from user-provided information and save it to `./outputs/`.

## Inputs

Collect the following from the user (ask if not provided):

| Field | Required | Description |
|-------|----------|-------------|
| **RFC Title** | Yes | Short, descriptive title (will be prefixed with `[RFC]:`) |
| **Main Content** | Yes | Description of the motivation and proposed changes |
| **Related PRs** | No | List of PR numbers or URLs; classify them by category |
| **Feedback Period** | No | Duration for feedback (default: "At least 1 week") |
| **CC List** | No | GitHub usernames to CC (e.g., `@user1`, `@user2`) |
| **Other Notes** | No | Any additional context or notes |

## Workflow

1. **Collect inputs** — gather all necessary information from the user
2. **Classify PRs** — if PRs are provided, group them by category (see PR Classification below)
3. **Generate RFC content** — write each section following the template
4. **Save output** — write to `./outputs/rfc-<slug>.md`
5. **Confirm** — tell the user the output path

## PR Classification

When the user provides related PRs, classify them into appropriate categories such as:

- **Background / Motivation PRs** — existing issues, bugs, or prior discussions that motivate this RFC
- **Implementation PRs** — concrete code changes implementing the proposed design
- **Dependency PRs** — prerequisite changes that must land first
- **Related / Follow-up PRs** — tangential improvements or future work inspired by this RFC

Format each category as a bullet list:

```markdown
### Related PRs

**Background / Motivation:**
- [#XXXX](https://github.com/vllm-project/vllm/pull/XXXX): Brief description

**Implementation:**
- [#YYYY](https://github.com/vllm-project/vllm/pull/YYYY): Brief description

**Dependencies:**
- [#ZZZZ](https://github.com/vllm-project/vllm/pull/ZZZZ): Brief description
```

If a PR number has no description, fetch its title via the GitHub API or `gh` CLI:
```bash
gh pr view <NUMBER> --repo vllm-project/vllm --json title,url -q '"\(.url): \(.title)"'
```

## RFC Document Template

Save to `/Users/shanshan-shen/.claude/skills/vllm-rfc-generator/outputs/rfc-<slug>.md`
where `<slug>` is the title converted to lowercase with hyphens (e.g., `rfc-async-engine-redesign.md`).

```markdown
# [RFC]: <Title>

> **Labels**: RFC
> **Status**: Open for Feedback
> **Feedback Period**: <period>

---

## Motivation

<2–4 paragraphs explaining WHY this change is needed>

- What problem does it solve?
- What are the current limitations or pain points?
- Why is this the right time to address it?

## Proposed Change

<3–6 paragraphs or structured subsections describing the proposed design>

Key points to cover:
- What is being changed / added / removed?
- High-level design or architecture
- API surface changes (if any)
- Migration path or backward compatibility

### Key Design Decisions

- **Decision 1**: Rationale
- **Decision 2**: Rationale

## Related PRs

<Classified PR list — see PR Classification section>

## Feedback Period

<Feedback period statement, e.g.: "We plan to keep this RFC open for at least 2 weeks (until YYYY-MM-DD). Please share your thoughts in the comments.">

## CC List

<@username1>, <@username2>

## Any Other Things

<Optional: open questions, alternatives considered, links to design docs, benchmarks, etc.>

---

> Please take a look at previous [RFCs](https://github.com/vllm-project/vllm/issues?q=label%3ARFC+sort%3Aupdated-desc) for reference.
```

## Writing Guidelines

- **Motivation**: Be concrete. Reference real pain points, GitHub issues, or user feedback. 2–4 short paragraphs.
- **Proposed Change**: Be specific enough that reviewers can evaluate the design, but avoid implementation minutiae. Use subsections for complex proposals.
- **Tone**: Professional, neutral, open to feedback. This is a proposal, not a decree.
- **Length**: Aim for 400–800 words total across Motivation and Proposed Change. Quality over quantity.
- **Language**: Write in **English** by default. If the user requests Chinese, write in Chinese.

## Example Output Path

```
/Users/shanshan-shen/.claude/skills/vllm-rfc-generator/outputs/rfc-async-engine-redesign.md
```
