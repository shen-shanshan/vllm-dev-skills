# vLLM Dev Skills

**English** | [简体中文](./README_zh.md)

<div align='left'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src="https://img.shields.io/badge/License-Apache--2.0-turquoise.svg">
  <img src=https://img.shields.io/badge/Maintained%3F-yes-green.svg >
  <img src=https://img.shields.io/badge/Contributions-welcome-blue >
  <img src=https://img.shields.io/github/watchers/shen-shanshan/vllm-dev-skills?color=9cc >
  <img src=https://img.shields.io/github/forks/shen-shanshan/vllm-dev-skills.svg?style=social >
  <img src=https://img.shields.io/github/stars/shen-shanshan/vllm-dev-skills.svg?style=social >
</div align='left'>

## 📌 Overview

A curated collection of Claude Code agent skills that accelerate the entire vLLM development lifecycle — from understanding codebases and triaging issues to designing features, contributing changes, and analyzing performance. Each skill encapsulates a specific development task as a reusable, composable workflow, enabling vLLM developers and community contributors to work more efficiently with AI-assisted tooling.

## 🚀 Skills

| Name | Category | Description | Rec |
| :--- | :------- | :---------- | :-- |
| [vllm-rocm-env-prepare](./skills/vllm-rocm-env-prepare/SKILL.md) | Env Setup | Set up the vLLM ROCm dev environment inside a `vllm/vllm-openai-rocm` nightly container: clone repos, link skills, init git/node/claude config, install the GitHub CLI (`gh`, no login), build a no-compile editable vLLM `.venv` aligned to the image binaries, and optionally build MORI with UMBP. Already-done steps are skipped. | ⭐️⭐️⭐️ |
| [vllm-dev-task-discovery](./skills/vllm-dev-task-discovery/SKILL.md) | Demand Discovery | Analyze contribution opportunities in the vLLM repository for community developers. | ⭐️ |
| [vllm-feature-tutorial](./skills/vllm-feature-tutorial/SKILL.md) | Code Reading | Generate comprehensive Chinese technical tutorial documents for vLLM features and modules. | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-model-tutorial](./skills/vllm-model-tutorial/SKILL.md) | Code Reading | Generate comprehensive Chinese technical tutorial documents for vLLM models. | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-feature-design](./skills/vllm-feature-design/SKILL.md) | Change Proposing | Design and implement vLLM features according to user requirements. | ⭐️ |
| [vllm-rfc-generator](./skills/vllm-rfc-generator/SKILL.md) | Change Proposing | Generate a vLLM-style RFC document based on user input. | ⭐️ |
| [vllm-rocm-code-dev](./skills/vllm-rocm-code-dev/SKILL.md) | Code Contribution | Implement ROCm-only code changes in vLLM from a development requirement (no impact on CUDA or other backends, concise "why on ROCm" comments), and keep an English PR description in `docs/tasks/` in sync with every code update. | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-pr-desc-generator](./skills/vllm-pr-desc-generator/SKILL.md) | Code Contribution | Generate a vLLM-style PR title and description (Overview / Claims / Validation / Details, following the latest PR template) from a GitHub PR's code changes; kept short and final-state only, with the checklist ticked by default. | ⭐️⭐️ |
| [vllm-test-generator](./skills/vllm-test-generator/SKILL.md) | Code Contribution | Generate unit tests or end-to-end tests for vllm. | ⭐️ |
| [vllm-benchmark-result-analysis](./skills/vllm-benchmark-result-analysis/SKILL.md) | Performance Optimization | Compare serving benchmark outputs before and after a code change. | ⭐️⭐️ |
| [vllm-vs-atom-decode-trace-comparison](./skills/vllm-vs-atom-decode-trace-comparison/SKILL.md) | Performance Optimization | Compare decode-phase kernel implementations of vLLM vs ATOM from torch-profiler traces, with per-layer-type comparison tables and a vLLM optimization TODO list. | ⭐️⭐️ |
| [infx-bench-script-generator](./skills/infx-bench-script-generator/SKILL.md) | Performance Optimization | Generate local bash scripts (docker run / server start / AgentX client replay) to run an InferenceX srt-slurm recipe benchmark without Slurm, with optional torch profiler prefill/decode step capture. | ⭐️⭐️⭐️ |
| [vllm-pr-summary](./skills/vllm-pr-summary/SKILL.md) | Code Review | Fetch and analyze a PR from vllm, then generate a report covering PR overview, code change analysis, technical principles, discussion highlights, and risk assessment. | ⭐️⭐️⭐️⭐️ |
| [vllm-rocm-pr-review](./skills/vllm-rocm-pr-review/SKILL.md) | Code Review | Review AMD/ROCm-related PRs from vllm (aiter/mori/ROCm kernels) and generate a single combined Chinese report: a detailed PR summary (with Mermaid diagrams, technical principles, discussion highlights, and risk table) followed by severity-sorted, type-categorized ROCm review findings, a verdict, copy-paste English comments with file + diff line numbers ready for GitHub PR review, and a short (≤100-word) English overall summary to post as the PR's top-level comment. | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-vit-cg-pr-review](./skills/vllm-vit-cg-pr-review/SKILL.md) | Code Review | Review ViT (encoder) CUDA graph PRs from vllm — the model integrations, framework changes and fixes tracked by [#38175](https://github.com/vllm-project/vllm/issues/38175) — against the latest vllm main, following the review habits learned from the feature's maintainers (@shen-shanshan, @Isotr0py). Same combined Chinese report format as `vllm-rocm-pr-review` (detailed summary, severity-sorted findings with an integration checklist, verdict, copy-paste English comments, ≤100-word English summary); can batch-review all open tracker PRs. | ⭐️⭐️⭐️⭐️ |
| [vllm-multimodal-open-issue-analyzer](./skills/vllm-multimodal-open-issue-analyzer/SKILL.md) | Issue Analysis | Fetch and organize multimodal-related open issues from vllm. | ⭐️ |
| [vllm-technical-blog-writer](./skills/vllm-technical-blog-writer/SKILL.md) | Technical Writing | Write or complete Chinese vLLM technical blog posts in the author's established style (two-segment `XXX｜XXX` titles; outline headings and points are polished, and sections may be adjusted unless told otherwise), with concise architecture, flow, and sequence diagrams drawn in the author's hand-drawn draw.io style. | ⭐️⭐️ |

## 📖 Usages

Skills accept prompts in English or Chinese. Skills marked "Chinese" in the table above write their reports in Chinese regardless of the prompt language.

**📚 vllm-rocm-env-prepare**

Prompt:

```text
I just entered a vllm/vllm-openai-rocm:nightly-rocm100 container. Please set up the vLLM ROCm dev environment (optional: also install MORI with UMBP).
/vllm-rocm-env-prepare
```

Output: cloned / synced repos, linked skills, git / node / claude config, the GitHub CLI (`gh`; run `gh auth login` yourself), and a no-compile editable vLLM in `/home/shashen/vllm/vllm/.venv` (optional: MORI with UMBP installed into the venv), followed by a report of which steps ran and which were skipped.

**📚 vllm-dev-task-discovery**

Prompt:

```text
What are the recent contribution opportunities for community developers in vLLM's multimodal module? (Optional: list the current top 20 tasks.)
/vllm-dev-task-discovery
```

Output: [vllm_dev_task_multimodal_20260611](./skills/vllm-dev-task-discovery/outputs/vllm_dev_task_multimodal_20260611.md).

**📚 vllm-feature-tutorial**

Prompt:

```text
I want to learn about the EPD (Encode-Prefill-Decode) feature in vLLM. Please write a technical document about it.
/vllm-feature-tutorial
```

Output: [disaggregated_encoder_epd](./skills/vllm-feature-tutorial/outputs/disaggregated_encoder_epd.md).

**📚 vllm-model-tutorial**

Prompt:

```text
I want to learn about the DeepSeek-OCR model in vLLM, especially how its ViT processes multimodal inputs. Please write a technical document about it.
/vllm-model-tutorial
```

Output: [deepseek_ocr](./skills/vllm-model-tutorial/outputs/deepseek_ocr.md).

**📚 vllm-feature-design**

Prompt:

```text
Background:
vLLM's EPD feature currently only supports ExampleECConnector. We want to integrate a MooncakeECConnector and use Mooncake to manage multiple transfer backends, e.g. TCP / RDMA / SHM.
Related link: https://github.com/vllm-project/vllm/pull/33714#issuecomment-3882716972

Goal:
Following how MooncakeConnector is registered and used for PD disaggregation (KV cache transfer), implement MooncakeECConnector to transfer the encoder cache for EPD.
Only the TCP backend is needed for now, but keep the interface extensible for more backends such as RDMA and SHM. For example, implement a MooncakeECConnector base class with a TCPMooncakeECConnector subclass, so RDMAMooncakeECConnector, SHMMooncakeECConnector, etc. can be added later.
Users select the Mooncake transfer backend in the vLLM launch config; switching after startup is not supported, and TCP is the default.
You also need to rethink how to manage the multi-modal embeddings internally inside vLLM (You might need to introduce "embed blocks" based on hash + position).

Notes:
Do not write tests. Only implement the core code, and generate a Markdown design document in the project root.

References:
https://kvcache-ai.github.io/Mooncake/getting_started/supported-protocols.html
https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html

/vllm-feature-design
```

Output: [Support Mooncake Based ECConnector for EPD](https://github.com/vllm-project/vllm/issues/39766) (with manual adjustment and optimization).

**📚 vllm-rfc-generator**

Prompt:

```text
RFC title:
Support ViT Full CUDA Graph

Main content:
Add full CUDA graph for the ViT to reduce kernel launch overheads.
Must support image and video inference, and mainstream VLMs such as Qwen3-VL / Qwen3.5 / GLM-V / Kimi K2.5.

Related PRs:
https://github.com/vllm-project/vllm/pull/35963
https://github.com/vllm-project/vllm/pull/37914
https://github.com/vllm-project/vllm/pull/38040
https://github.com/vllm-project/vllm/pull/38061

/vllm-rfc-generator
```

Output: [Support ViT Full CUDA Graph](https://github.com/vllm-project/vllm/issues/38175) (with manual adjustment and optimization).

**📚 vllm-rocm-code-dev**

Prompt:

```text
[Background]
https://github.com/vllm-project/vllm/pull/51794 implemented DeepSeek-V4 CSA multi-stream overlap for the ROCm backend in vLLM. Building on that PR, implement dual-stream overlap for the DeepSeek-V4 HCA layer on the vLLM ROCm backend.

[Task]
Follow the DeepSeek-V4 HCA dual-stream overlap implementation in /Users/shanshan-shen/Documents/GitHub/ATOM:

For the HCA layer, first run serially: mhc_fused_post_pre_gemm_sqrsum_kernel → mhc_pre_big_fuse_rmsnorm_kernel.

Then launch two streams:
1. default stream: fused wqa+wkv GEMM → q/kv RMSNorm → wq_b → qnorm/RoPE → SWA KV insert
2. aux stream 0: HCA main compressor

Finally, join the two streams (end of the overlap) and run sparse attention and the rest sequentially.

[Notes]
1. Add a capture-only gate to the HCA multi-stream fork on the vLLM ROCm path (matching ATOM's in_hipgraph rule).
/vllm-rocm-code-dev
```

Output: ROCm-only code changes in the local vLLM repo, plus an English PR description saved to `docs/tasks/<task-slug>.md`, which is updated after every rebase / bugfix / review fix.

**📚 vllm-pr-desc-generator**

Prompt:

```text
https://github.com/vllm-project/vllm/pull/38061
/vllm-pr-desc-generator
```

Output: [pr-38061-desc](./skills/vllm-pr-desc-generator/outputs/pr-38061-desc.md) (related PR: [#38061](https://github.com/vllm-project/vllm/pull/38061)).

**📚 vllm-test-generator**

Prompt:

```text
Based on the changes in https://github.com/vllm-project/vllm/pull/38061/changes, generate the corresponding test cases in tests/v1/cudagraph/test_encoder_cudagraph.py.
Reuse the SimpleMockViTModel code as much as possible to reduce redundancy.
/vllm-test-generator
```

Output: [test_encoder_cudagraph](https://github.com/vllm-project/vllm/pull/38061/changes#diff-e8bf25855192d2972d13d0f0263a121462a181c4c297981d2395394666dc7979).

**📚 vllm-benchmark-result-analysis**

Prompt:

```text
Before this PR:
============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Request rate configured (RPS):           10.00     
Benchmark duration (s):                  78.58     
Total input tokens:                      33418     
Total generated tokens:                  61431     
Request throughput (req/s):              6.36      
Output token throughput (tok/s):         781.78    
Peak output token throughput (tok/s):    2475.00   
Peak concurrent requests:                383.00    
Total token throughput (tok/s):          1207.07   
---------------Time to First Token----------------
Mean TTFT (ms):                          7116.24   
Median TTFT (ms):                        4295.84   
P99 TTFT (ms):                           18370.87  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          245.78    
Median TPOT (ms):                        264.03    
P99 TPOT (ms):                           334.38    
---------------Inter-token Latency----------------
Mean ITL (ms):                           246.99    
Median ITL (ms):                         117.71    
P99 ITL (ms):                            1327.55   
==================================================

After this PR:
============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Request rate configured (RPS):           10.00     
Benchmark duration (s):                  77.44     
Total input tokens:                      33418     
Total generated tokens:                  61522     
Request throughput (req/s):              6.46      
Output token throughput (tok/s):         794.40    
Peak output token throughput (tok/s):    2691.00   
Peak concurrent requests:                369.00    
Total token throughput (tok/s):          1225.91   
---------------Time to First Token----------------
Mean TTFT (ms):                          6888.64   
Median TTFT (ms):                        4128.82   
P99 TTFT (ms):                           17487.94  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          240.14    
Median TPOT (ms):                        259.18    
P99 TPOT (ms):                           313.15    
---------------Inter-token Latency----------------
Mean ITL (ms):                           241.84    
Median ITL (ms):                         121.08    
P99 ITL (ms):                            1470.33   
==================================================

/vllm-benchmark-result-analysis
```

Output: [benchmark_comparison_20260323_150830](./skills/vllm-benchmark-result-analysis/outputs/benchmark_comparison_20260323_150830.md) (related PR: [#7104](https://github.com/vllm-project/vllm-ascend/pull/7104)).

**📚 vllm-vs-atom-decode-trace-comparison**

Prompt:

```text
Compare the decode traces of vLLM and ATOM on MiniMax-M3.
Model: MiniMaxAI/MiniMax-M3
vllm trace: /path/to/vllm-minimax-m3-tp4-8k1k-conc8.json
atom trace: /path/to/atom-minimax-m3-tp4-8k1k-conc8.json
Report language: Chinese
/vllm-vs-atom-decode-trace-comparison
```

Output: [minimax-m3-trace-comparison_zh_20260905_162716](./skills/vllm-vs-atom-decode-trace-comparison/outputs/minimax-m3-trace-comparison_zh_20260905_162716.md).

**📚 vllm-pr-summary**

Prompt:

```text
https://github.com/vllm-project/vllm/pull/35963
/vllm-pr-summary
```

Output: [pr-35963-summary](./skills/vllm-pr-summary/outputs/pr-35963-summary.md).

**📚 vllm-rocm-pr-review**

Prompt:

```text
https://github.com/vllm-project/vllm/pull/29304
/vllm-rocm-pr-review
```

Output: [pr-29304-review](./skills/vllm-rocm-pr-review/outputs/pr-29304-review.md). The report is a single file in two parts: §1–6 is a detailed PR summary (merged from vllm-pr-summary, with Mermaid diagrams, technical principles, discussion highlights, and a risk table); §7–8 are the ROCm review findings (sorted by severity) and the verdict; §9 holds English inline comments (with file and diff line numbers) ready to paste into a GitHub PR review; §10 is an English overall summary of at most 100 words to post as the PR's top-level comment. See [example-report.md](./skills/vllm-rocm-pr-review/references/example-report.md) for the style.

**📚 vllm-vit-cg-pr-review**

Prompt (single PR):

```text
https://github.com/vllm-project/vllm/pull/63074
/vllm-vit-cg-pr-review
```

Prompt (batch: every open PR of the tracker):

```text
Review the open PRs in https://github.com/vllm-project/vllm/issues/38175 and its comments.
/vllm-vit-cg-pr-review
```

Output: [pr-63074-review](./skills/vllm-vit-cg-pr-review/outputs/pr-63074-review.md) (others in [outputs/](./skills/vllm-vit-cg-pr-review/outputs/)). Same report format as vllm-rocm-pr-review (§1–6 detailed summary / §7–8 findings + verdict / §9 English inline comments / §10 English summary), with ViT CUDA graph–specific review rules: checked against vllm main's `SupportsEncoderCudaGraph` interface and `EncoderCudaGraphManager` for interface sync, capture/replay/eager numerical equivalence, budget computation, modality coverage, docs/example/CI integration, and benchmark validity. Review criteria and English wording are learned from @shen-shanshan's and @Isotr0py's reviews on tracker PRs. `python3 scripts/update_skill.py --check-vllm-tree` re-collects reviewer comments and checks for framework interface drift in one step. When a PR has reports from both this skill and vllm-rocm-pr-review, the Pingboard dashboard shows this skill's report first.

**📚 vllm-multimodal-open-issue-analyzer**

Prompt:

```text
/vllm-multimodal-open-issue-analyzer
```

Output: [vllm_multimodal_issues_20260425_030830](./skills/vllm-multimodal-open-issue-analyzer/outputs/vllm_multimodal_issues_20260425_030830.md).

**📚 vllm-technical-blog-writer**

Scenario 1 — write a complete article from a topic:

```text
Write a complete Chinese technical blog post on "Automatic Prefix Caching in vLLM" in my usual writing style, based on the latest code and related materials. Add concise architecture and flow diagrams where useful.
/vllm-technical-blog-writer
```

Scenario 2 — complete an existing Markdown draft:

```text
Using my article template <article-dir>/index.md and the references I provided, together with the latest vllm-project/vllm main branch, fill in the remaining content as required by the [Agent TODO] markers. Put new images in the ./images directory next to the article.
/vllm-technical-blog-writer
```

Output: New articles are saved under [`./skills/vllm-technical-blog-writer/outputs/<article-slug>/`](./skills/vllm-technical-blog-writer/outputs/), with referenced images stored in the article's `images/` directory. Existing drafts are updated in place. Article titles always use the two-segment `XXX｜XXX` form (with "vLLM" when the post is explicitly about vLLM). Outline headings and points are treated as references and may be polished; sections may be added, removed, or reordered unless you explicitly ask to keep the section design.

**📚 infx-bench-script-generator**

Prompt:

```text
I want to run the override_dep8_c64 config of dsv4/vllm/mi355x-fp4-mtp/agentic.yaml on my local server and capture one complete prefill step with the torch profiler. Put all results under /home/shashen/vllm/dsv4_benchmark. Please generate the scripts.
/infx-bench-script-generator
```

Output: one folder per `<recipe>_<override>` combination, containing `01_docker_run.sh` (start the container on the host), `02_start_server.sh` (start the server inside the container, with optional profiler config), `03_run_client.sh` (run the AgentX replay inside the container), and `variant.json` / `variant.env`. Example: [dsv4_vllm_mi355x-fp4-mtp_agentic_override_dep8_c64](./skills/infx-bench-script-generator/outputs/dsv4_vllm_mi355x-fp4-mtp_agentic_override_dep8_c64/).

## 🛠️ Configuration

**Option 1: install with the [`skills`](https://github.com/vercel-labs/skills) CLI** (requires Node.js).

```bash
# List the skills available in this repo.
npx skills add shen-shanshan/vllm-dev-skills --list

# Install one or more skills for Claude Code at user level (~/.claude/skills).
npx skills add shen-shanshan/vllm-dev-skills -g -a claude-code -s vllm-pr-desc-generator

# Install all skills for Claude Code at user level.
npx skills add shen-shanshan/vllm-dev-skills -g -a claude-code -s '*' -y

# Update installed skills later.
npx skills update -g
```

Drop `-g` to install into the current project (`./.claude/skills`) instead, or change `-a` to target other agents (`'*'` for all). The CLI installs a snapshot of the GitHub repo, so run `npx skills update` to pick up new changes.

**Option 2: soft-link a local clone** (edits in the clone take effect immediately).

```bash
# Manually soft-link single skill to claude-code's official skill dir.
ln -sfn /path/to/vllm-dev-skills/skills/skill-name ~/.claude/skills/skill-name

# Manually soft-link all skills to claude-code's official skill dir.
for dir in /path/to/vllm-dev-skills/skills/*/; do
  name=$(basename "$dir")
  ln -sfn "$dir" ~/.claude/skills/"$name"
done
```

## 📖 References

- [Agent Skills](https://agentskills.io/home#why-agent-skills)
- [Extend Claude with skills](https://code.claude.com/docs/en/skills)
- [Configure Claude Code in VSCode](https://api.lingyaai.cn/model_intro/Claude-Code.html)
- [Configure Claude Code CLI with Claude HUD](https://github.com/jarrodwatts/claude-hud)
- [Awesome DeepSeek Agent](https://github.com/deepseek-ai/awesome-deepseek-agent)

## ©️ Citation

```bibtex
@misc{vllm-dev-skills@2026,
  title  = {vllm-dev-skills},
  url    = {https://github.com/shen-shanshan/vllm-dev-skills},
  note   = {Open-source software available at https://github.com/shen-shanshan/vllm-dev-skills},
  author = {shen-shanshan},
  year   = {2026}
}
```

## 📜 License

Apache-2.0 License, find more details [<u>here</u>](./LICENSE).

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=shen-shanshan/vllm-dev-skills&type=Timeline)](https://www.star-history.com/#shen-shanshan/vllm-dev-skills&Timeline)
