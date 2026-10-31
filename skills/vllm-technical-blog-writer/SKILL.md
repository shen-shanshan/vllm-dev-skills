---
name: vllm-technical-blog-writer
description: Write or complete Chinese vLLM technical blog posts in the author's established Zhihu style. Use when the user provides a vLLM feature, model, architecture, optimization, or other topic and asks for a full blog post, or provides an existing Markdown outline/draft plus references and asks to research current vllm-project/vllm code and finish the missing sections. Produces concise technical diagrams that reproduce the author's hand-drawn draw.io style (Comic Sans, draw.io color pairs, nested containers, red request / blue output edges) and stores each new article with its images under this skill's outputs directory.
---

# vLLM Technical Blog Writer

Write technically accurate vLLM articles that sound like the author, not a generic tutorial generator.

## Required references

Before writing:

1. Read [references/style-profile.md](references/style-profile.md).
2. Read [references/diagram-guide.md](references/diagram-guide.md) and open [assets/diagram-style-template.svg](assets/diagram-style-template.svg) when the article needs any diagram. Every diagram must reproduce the author's hand-drawn draw.io style described there.
3. Treat the user's draft, outline, references, and explicit wording as higher priority than the default patterns in those files.

## Select the workflow

- **New article**: the user supplies a topic but no article draft. Follow "Create a new article".
- **Complete an article**: the user supplies an outline, template, or partially written Markdown article. Follow "Complete an existing article".
- If the request mixes both, preserve supplied content and treat it as an existing article.

## Research rules

Research before drafting. Do not rely on model memory for current vLLM behavior.

1. Inspect the latest `main` branch of `vllm-project/vllm`, unless the user specifies a release, commit, or PR.
2. Record the analyzed commit SHA or exact version in working notes. Mention it in the article only when version context matters.
3. Find the actual entry points, core classes/functions, data structures, and call path. Read implementations rather than inferring behavior from names.
4. Cross-check with first-party sources in this order:
   - vLLM source code and tests
   - vLLM official documentation, design docs, PRs, issues, and discussions
   - upstream papers and dependency documentation
   - high-quality third-party material
5. Prefer merged code and current docs. Clearly label behavior that exists only in an open PR, proposal, or older release.
6. Keep a source list while researching. Every benchmark number, version-sensitive statement, or non-obvious claim must be traceable.
7. If sources disagree, use the code as the source of truth and explain the version difference.
8. Never invent code paths, API names, benchmark data, design rationale, or citations.

## Create a new article

1. Clarify only genuinely blocking ambiguity, such as two unrelated features sharing a name.
2. Decide the article's single main question and intended reader.
3. Build a compact outline that moves from motivation and basic principle to concrete flow, implementation, and trade-offs. Do not force every article into the same template.
4. Choose one running example when the topic is abstract. Reuse it across sections.
5. Research the current implementation and collect only the code snippets, diagrams, and data needed to explain the main question.
6. Draft in Chinese following the style profile. Keep English for proper nouns, code identifiers, APIs, and established technical terms.
7. Add diagrams only where they reduce explanation cost. Follow the diagram guide and render every referenced image.
8. Add a short summary and a references section. Avoid a generic "future outlook" unless the topic genuinely needs one. End the summary section with the fixed authorship statement from the style profile ("总结与收尾").
9. Run the quality checklist, then save the article.

### New article output

Create:

```text
outputs/<article-slug>/
├── <article-slug>.md
└── images/
    ├── <diagram-name>.svg
    └── <diagram-name>.mmd   # only for Mermaid diagrams
```

- Resolve `outputs/` relative to this `SKILL.md`, not relative to the user's repository.
- Use a short lowercase `kebab-case` article slug.
- Store each article in its own directory.
- Put every image referenced by the article in that article's `images/` directory.
- Reference images from Markdown with `./images/<filename>.svg` or `./images/<filename>.png`.
- Keep reproducible diagram source beside the rendered image when applicable.

## Complete an existing article

1. Read the entire article and supplied references before editing.
2. Identify:
   - completed sections that must remain unchanged,
   - explicit placeholders or empty headings,
   - claims that need current-code verification,
   - terminology, heading depth, numbering, and local writing patterns.
3. Preserve the author's existing wording, opinions, examples, title hierarchy, and section order unless the user explicitly requests restructuring.
4. Research only what is needed to fill gaps, while checking interfaces affected by the latest `main` branch.
5. Continue from the surrounding prose. Match its detail level and transitions; do not paste a separate tutorial style into the middle of the draft.
6. Replace placeholders with finished prose. Do not leave TODOs, fabricated citations, or notes to the author.
7. Edit the existing Markdown file in place. If the draft does not already contain the fixed authorship statement from the style profile, add it at the end of the summary section (before the references).
8. Put newly created images in an `images/` directory beside that article and use relative links. If the draft already follows another image convention, preserve it unless the user asks to migrate.
9. Do not create a duplicate article under this skill's `outputs/` for this workflow.

## Writing constraints

- Default to Simplified Chinese except for professional terms, identifiers, commands, and quoted source names.
- Use full natural paragraphs. Do not hard-wrap prose at 80 characters or any fixed width.
- Explain the "why" before deep implementation details.
- Prefer concrete examples, tensor shapes, call chains, and before/after comparisons over abstract adjectives.
- Quote only short, necessary code fragments from the analyzed version. Annotate the repository-relative file path.
- Keep claims calibrated. State uncertainty or limited evidence directly.
- Do not pad the article to appear comprehensive. Omit sections that do not help explain the topic.
- Avoid repetitive summaries, canned transitions, excessive callout boxes, emoji decoration, and marketing language.
- Use Markdown headings, lists, tables, code blocks, formulas, and images only when they improve comprehension.

## Diagram workflow

1. Decide whether the relationship is best shown as architecture, flow, sequence, state, or data-layout diagram.
2. Hand-author `images/<name>.svg` by default, starting from the `<defs>` and `<style>` of `assets/diagram-style-template.svg`: white canvas, Comic Sans MS 12px, draw.io fill/stroke color pairs, nested containers with top-left bold titles, 1px orthogonal edges colored by data flow (request red, output blue), and a 10px grid. Use Mermaid (saved as `images/<name>.mmd` with the themed init line from the guide) only for sequence or state diagrams that are impractical to lay out by hand.
3. Render the SVG to PNG (for example with Playwright + Chromium) and inspect it. Compare against the template and the author's original figures; fix style drift, clipped labels, crossing lines, unreadable text, or unnecessary nodes before referencing it. If SVG is unsuitable for the target platform, ship the PNG as well.
4. Use a descriptive Chinese caption in the article.

Never reference an image file that was not created successfully.

## Quality checklist

Before finishing, verify:

- [ ] The article answers one clear technical question.
- [ ] Current implementation claims were checked against the requested vLLM version.
- [ ] Class, function, CLI option, config, and file names exactly match the source.
- [ ] Version-sensitive statements and performance numbers have sources.
- [ ] The prose follows the style profile without copying sentences mechanically.
- [ ] Existing user-written content was preserved in completion mode.
- [ ] All diagrams are necessary, simple, rendered, and linked with valid relative paths.
- [ ] New-article output is `outputs/<article-slug>/<article-slug>.md`.
- [ ] Every article-owned image is under its `images/` directory.
- [ ] No fixed-width hard wrapping was introduced.
- [ ] No placeholders or unsupported claims remain.
- [ ] Every list item ends with “；” except the last item of each list, which ends with “。” (see style profile). Reference-list items take no trailing punctuation and use `[text](URL)` links.
- [ ] English words in prose start with a capital letter; code, paths, URLs, brand names with fixed casing (vLLM) and quoted UI text (in inline code) are left as is.
- [ ] The summary ends with the fixed authorship statement (本文是先由我的 `vllm-technical-blog-writer` Skill …微调完成的), placed before the references section, link intact.
- [ ] The references section is numbered like the other top-level sections (e.g. `八、参考资料`).
- [ ] Diagrams follow the author's draw.io style from the diagram guide: white background, Comic Sans MS font stack, draw.io color pairs (dark red `#a20025` key nodes, red request path / blue output path), nested containers, 1px orthogonal edges.
