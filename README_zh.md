# vLLM Dev Skills

[English](./README.md) | **简体中文**

<div align='left'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src="https://img.shields.io/badge/License-Apache--2.0-turquoise.svg">
  <img src=https://img.shields.io/badge/Maintained%3F-yes-green.svg >
  <img src=https://img.shields.io/badge/Contributions-welcome-blue >
  <img src=https://img.shields.io/github/watchers/shen-shanshan/vllm-dev-skills?color=9cc >
  <img src=https://img.shields.io/github/forks/shen-shanshan/vllm-dev-skills.svg?style=social >
  <img src=https://img.shields.io/github/stars/shen-shanshan/vllm-dev-skills.svg?style=social >
</div align='left'>

## 📌 项目简介

一套精心整理的 Claude Code Agent Skill，覆盖 vLLM 研发的完整流程：从阅读代码、分析 Issue，到设计特性、贡献代码和分析性能。每个 Skill 都把一类具体的开发任务封装成可复用、可组合的工作流，帮助 vLLM 开发者和社区贡献者借助 AI 工具更高效地工作。

## 🚀 Skill 列表

| 名称 | 类别 | 描述 | 推荐度 |
| :--- | :--- | :--- | :----- |
| [vllm-rocm-env-prepare](./skills/vllm-rocm-env-prepare/SKILL.md) | 环境准备 | 在 `vllm/vllm-openai-rocm` nightly 容器内搭建 vLLM ROCm 开发环境：Clone 代码仓、链接 Skill、初始化 git / node / claude 配置、安装 GitHub CLI（`gh`，不代为登录）、构建与镜像二进制对齐的免编译 editable vLLM `.venv`，并可选构建带 UMBP 的 MORI。已完成的步骤会自动跳过。 | ⭐️⭐️⭐️ |
| [vllm-dev-task-discovery](./skills/vllm-dev-task-discovery/SKILL.md) | 需求发现 | 为社区开发者分析 vLLM 仓库中可以贡献的任务。 | ⭐️ |
| [vllm-feature-tutorial](./skills/vllm-feature-tutorial/SKILL.md) | 代码阅读 | 为 vLLM 的特性和模块生成详尽的中文技术教程。 | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-model-tutorial](./skills/vllm-model-tutorial/SKILL.md) | 代码阅读 | 为 vLLM 中的模型生成详尽的中文技术教程。 | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-feature-design](./skills/vllm-feature-design/SKILL.md) | 方案设计 | 根据用户需求设计并实现 vLLM 特性。 | ⭐️ |
| [vllm-rfc-generator](./skills/vllm-rfc-generator/SKILL.md) | 方案设计 | 根据用户输入生成 vLLM 风格的 RFC 文档。 | ⭐️ |
| [vllm-rocm-code-dev](./skills/vllm-rocm-code-dev/SKILL.md) | 代码贡献 | 根据开发需求在 vLLM 中实现仅限 ROCm 路径的代码改动（不影响 CUDA 等其他后端，附简洁的"为什么 ROCm 需要这样做"注释），并在每次代码更新后同步维护 `docs/tasks/` 下的英文 PR 描述。 | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-pr-desc-generator](./skills/vllm-pr-desc-generator/SKILL.md) | 代码贡献 | 根据 GitHub PR 的代码改动，按 vLLM 最新 PR 模板（Overview / Claims / Validation / Details）生成 PR 标题与描述；描述保持精简、只写最终结论，Checklist 默认全部勾选。 | ⭐️⭐️ |
| [vllm-test-generator](./skills/vllm-test-generator/SKILL.md) | 代码贡献 | 为 vLLM 生成单元测试或端到端测试。 | ⭐️ |
| [vllm-benchmark-result-analysis](./skills/vllm-benchmark-result-analysis/SKILL.md) | 性能优化 | 对比代码改动前后的 serving benchmark 结果。 | ⭐️⭐️ |
| [vllm-vs-atom-decode-trace-comparison](./skills/vllm-vs-atom-decode-trace-comparison/SKILL.md) | 性能优化 | 基于 torch profiler trace 对比 vLLM 与 ATOM 在 decode 阶段的 kernel 实现，按层类型输出对比表，并给出 vLLM 优化 TODO 列表。 | ⭐️⭐️ |
| [infx-bench-script-generator](./skills/infx-bench-script-generator/SKILL.md) | 性能优化 | 生成本地 bash 脚本（docker run / 启动 server / AgentX client 回放），无需 Slurm 即可跑 InferenceX srt-slurm recipe benchmark，可选用 torch profiler 采集 prefill / decode step。 | ⭐️⭐️⭐️ |
| [vllm-pr-summary](./skills/vllm-pr-summary/SKILL.md) | 代码检视 | 获取并分析 vLLM 的 PR，生成包含 PR 概览、代码改动分析、技术原理、讨论要点和风险评估的报告。 | ⭐️⭐️⭐️⭐️ |
| [vllm-rocm-pr-review](./skills/vllm-rocm-pr-review/SKILL.md) | 代码检视 | 检视 vLLM 中 AMD / ROCm 相关的 PR（aiter / mori / ROCm kernel），生成一份合并的中文报告：详细的 PR 总结（含 Mermaid 图、技术原理、讨论要点和风险表），按严重程度排序、按类型分类的 ROCm 检视意见和结论，可直接复制到 GitHub PR review 的英文评论（含文件与 diff 行号），以及一段 100 词以内、可作为 PR 顶层评论的英文总结。 | ⭐️⭐️⭐️⭐️⭐️ |
| [vllm-vit-cg-pr-review](./skills/vllm-vit-cg-pr-review/SKILL.md) | 代码检视 | 对照 vLLM 最新 main 分支，检视 [#38175](https://github.com/vllm-project/vllm/issues/38175) 跟踪的 ViT（encoder）CUDA graph PR（模型接入、框架改动和修复），检视习惯学习自该特性的维护者（@shen-shanshan、@Isotr0py）。报告格式与 `vllm-rocm-pr-review` 相同（详细总结、按严重程度排序并附接入检查清单的检视意见、结论、可直接复制的英文评论、100 词以内的英文总结）；支持批量检视 tracker 中所有 open 的 PR。 | ⭐️⭐️⭐️⭐️ |
| [vllm-multimodal-open-issue-analyzer](./skills/vllm-multimodal-open-issue-analyzer/SKILL.md) | Issue 分析 | 获取并整理 vLLM 中与多模态相关的 open issue。 | ⭐️ |
| [vllm-technical-blog-writer](./skills/vllm-technical-blog-writer/SKILL.md) | 技术写作 | 按作者既有风格撰写或补全中文 vLLM 技术博客（标题统一为 `XXX｜XXX` 两段式；大纲中的标题和要点会被润色，除非另有说明，章节也可能调整），并按作者手绘 draw.io 风格绘制简洁的架构图、流程图和时序图。 | ⭐️⭐️ |

## 📖 使用示例

**📚 vllm-rocm-env-prepare**

Prompt：

```text
我刚进入 vllm/vllm-openai-rocm:nightly-rocm100 容器，请帮我初始化 vllm rocm 开发环境（Optional：并安装带 UMBP 的 MORI）。
/vllm-rocm-env-prepare
```

输出：已克隆 / 已同步的代码仓、已链接的 skill、git / node / claude 配置、已安装的 GitHub CLI（`gh`，需自行 `gh auth login`），以及 `/home/shashen/vllm/vllm/.venv` 中的免编译 editable vLLM（可选：venv 中装上带 UMBP 的 MORI），最后汇报哪些步骤执行了、哪些被跳过。

**📚 vllm-dev-task-discovery**

Prompt：

```text
我想了解 vLLM 的多模态模块中最近还有哪些可以给社区开发者贡献的事情。（Optional：请帮我分析并整理出当前 top 20 的 tasks）
/vllm-dev-task-discovery
```

输出：[vllm_dev_task_multimodal_20260611](./skills/vllm-dev-task-discovery/outputs/vllm_dev_task_multimodal_20260611.md)。

**📚 vllm-feature-tutorial**

Prompt：

```text
我想了解 vLLM 中的 EPD（Encode-Prefill-Decode）特性，请帮我总结一份技术文档。
/vllm-feature-tutorial
```

输出：[disaggregated_encoder_epd](./skills/vllm-feature-tutorial/outputs/disaggregated_encoder_epd.md)。

**📚 vllm-model-tutorial**

Prompt：

```text
我想了解 vLLM 中的 DeepSeek-OCR 模型，尤其是其 ViT 部分对多模态输入的处理流程，请帮我总结一份技术文档。
/vllm-model-tutorial
```

输出：[deepseek_ocr](./skills/vllm-model-tutorial/outputs/deepseek_ocr.md)。

**📚 vllm-feature-design**

Prompt：

```text
需求背景：
vLLM 的 EPD 特性目前只支持 ExampleECConnector，我们希望集成 MooncakeECConnector，并通过 Mooncake 来统一管理多种传输后端，比如：TCP/RDMA/SHM。
相关链接：https://github.com/vllm-project/vllm/pull/33714#issuecomment-3882716972

设计目标：
请你参考 vLLM 中 MooncakeConnector 在 PD 分离场景（传输 KV Cache）中的注册和使用方式，实现 MooncakeECConnector，用于支持 EPD 特性下的 Encoder Cache 传输。
目前，你只需要实现 TCP 传输方式，但需要考虑将来可能支持更多的传输后端，比如：RDMA、SHM 等（保证接口的可扩展性），比如：你可以实现 MooncakeECConnector 基类，TCPMooncakeECConnector 继承该基类，未来需要实现其它传输后端时，可以直接添加 RDMAMooncakeECConnector、SHMMooncakeECConnector 等子类。
用户通过 vLLM 启动配置来指定使用哪种 Mooncake 传输后端，不支持启动后切换传输方式，默认使用 TCP 传输。
你还需要重新思考 vLLM 内部如何管理多模态 embedding（可能需要引入基于 hash + position 的 "embed blocks"）。

注意事项：
你不需要写测试用例，只写核心代码实现，并生成一份 markdown 格式的设计文档，放到当前项目的根目录下。

参考资料：
https://kvcache-ai.github.io/Mooncake/getting_started/supported-protocols.html
https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html

/vllm-feature-design
```

输出：[Support Mooncake Based ECConnector for EPD](https://github.com/vllm-project/vllm/issues/39766)（经过人工调整和优化）。

**📚 vllm-rfc-generator**

Prompt：

```text
RFC 标题：
Support ViT Full CUDA Graph

主要内容：
为 ViT 加入 full CUDA graph，减少 kernel launch 开销。
需要支持图像、视频推理，需要支持 Qwen3-VL/Qwen3.5/GLM-V/Kimi K2.5 等主流 VLM 模型。

相关 PR：
https://github.com/vllm-project/vllm/pull/35963
https://github.com/vllm-project/vllm/pull/37914
https://github.com/vllm-project/vllm/pull/38040
https://github.com/vllm-project/vllm/pull/38061

/vllm-rfc-generator
```

输出：[Support ViT Full CUDA Graph](https://github.com/vllm-project/vllm/issues/38175)（经过人工调整和优化）。

**📚 vllm-rocm-code-dev**

Prompt：

```text
【需求背景】：
https://github.com/vllm-project/vllm/pull/51794 为 vllm 中的 rocm 后端实现了 deepseek-v4 csa multi-stream overlap。现在需要你基于该 PR，继续为 vllm rocm 后端实现 deepseek-v4 hca layer 的 dual-stream overlap。

【你的任务】：
参考 /Users/shanshan-shen/Documents/GitHub/ATOM 中 deepseek-v4 hca 的 dual-stream overlap 实现：

对于 HCA 层，先串行执行：mhc_fused_post_pre_gemm_sqrsum_kernel → mhc_pre_big_fuse_rmsnorm_kernel。

然后，launch 两条 stream：
1.default stream：fused wqa+wkv GEMM → q/kv RMSNorm → wq_b → qnorm/RoPE → SWA KV insert
2.aux stream 0：HCA main compressor

最后，join 两条 stream（多流 overlap 部分结束），接下来顺序执行 sparse attention 等后续流程。

【注意事项】：
1.给 vllm rocm 路径的 hca 多流 fork 加 capture-only gate（对齐 ATOM 的 in_hipgraph 规则）；
/vllm-rocm-code-dev
```

输出：本地 vLLM 仓库中仅限 ROCm 路径的代码改动，以及保存在 `docs/tasks/<task-slug>.md` 的英文 PR 描述；之后每次 rebase、修复 bug 或根据 review 修改代码，都会同步更新这份描述。

**📚 vllm-pr-desc-generator**

Prompt：

```text
https://github.com/vllm-project/vllm/pull/38061
/vllm-pr-desc-generator
```

输出：[pr-38061-desc](./skills/vllm-pr-desc-generator/outputs/pr-38061-desc.md)（对应 PR：[#38061](https://github.com/vllm-project/vllm/pull/38061)）。

**📚 vllm-test-generator**

Prompt：

```text
请根据 https://github.com/vllm-project/vllm/pull/38061/changes 中的改动，在 tests/v1/cudagraph/test_encoder_cudagraph.py 中生成对应的测试用例。
要求尽可能复用 SimpleMockViTModel 的代码，减少冗余。
/vllm-test-generator
```

输出：[test_encoder_cudagraph](https://github.com/vllm-project/vllm/pull/38061/changes#diff-e8bf25855192d2972d13d0f0263a121462a181c4c297981d2395394666dc7979)。

**📚 vllm-benchmark-result-analysis**

Prompt：

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

输出：[benchmark_comparison_20260323_150830](./skills/vllm-benchmark-result-analysis/outputs/benchmark_comparison_20260323_150830.md)（对应 PR：[#7104](https://github.com/vllm-project/vllm-ascend/pull/7104)）。

**📚 vllm-vs-atom-decode-trace-comparison**

Prompt：

```text
帮我对比分析 vLLM 和 ATOM 在 MiniMax-M3 上的 decode trace 差异。
模型：MiniMaxAI/MiniMax-M3
vllm trace：/path/to/vllm-minimax-m3-tp4-8k1k-conc8.json
atom trace：/path/to/atom-minimax-m3-tp4-8k1k-conc8.json
报告语言：中文
/vllm-vs-atom-decode-trace-comparison
```

输出：[minimax-m3-trace-comparison_zh_20260905_162716](./skills/vllm-vs-atom-decode-trace-comparison/outputs/minimax-m3-trace-comparison_zh_20260905_162716.md)。

**📚 vllm-pr-summary**

Prompt：

```text
https://github.com/vllm-project/vllm/pull/35963
/vllm-pr-summary
```

输出：[pr-35963-summary](./skills/vllm-pr-summary/outputs/pr-35963-summary.md)。

**📚 vllm-rocm-pr-review**

Prompt：

```text
https://github.com/vllm-project/vllm/pull/29304
/vllm-rocm-pr-review
```

输出：[pr-29304-review](./skills/vllm-rocm-pr-review/outputs/pr-29304-review.md)。新版报告为单文件两大部分：§1–6 PR 详细总结（整合自 vllm-pr-summary，含 Mermaid 图、技术原理、讨论亮点、风险表），§7–8 ROCm review 意见（按严重程度排序的 findings + verdict），§9 为可直接复制到 GitHub PR review 的英文评论（含文件与 diff 行号），§10 为 100 词以内的英文总结评论（本次 review 的整体结论，可直接发到 PR 评论区）；样式示例见 [example-report.md](./skills/vllm-rocm-pr-review/references/example-report.md)。

**📚 vllm-vit-cg-pr-review**

Prompt（单个 PR）：

```text
https://github.com/vllm-project/vllm/pull/63074
/vllm-vit-cg-pr-review
```

Prompt（批量：tracker 中所有 open 的 PR）：

```text
review https://github.com/vllm-project/vllm/issues/38175 及其评论区中 open 的 PR
/vllm-vit-cg-pr-review
```

输出：[pr-63074-review](./skills/vllm-vit-cg-pr-review/outputs/pr-63074-review.md)（其余见 [outputs/](./skills/vllm-vit-cg-pr-review/outputs/)）。报告格式与 vllm-rocm-pr-review 相同（§1–6 详细总结 / §7–8 findings + verdict / §9 英文行内评论 / §10 英文总结），review 规则改为 ViT CUDA graph 专项：对照 vllm main 的 `SupportsEncoderCudaGraph` 接口与 `EncoderCudaGraphManager`，检查接口同步、capture/replay/eager 数值等价、budget 计算、模态覆盖、文档/example/CI 集成与 benchmark 有效性；判断标准与英文措辞学习自 @shen-shanshan、@Isotr0py 在 tracker PR 中的 review 意见。`python3 scripts/update_skill.py --check-vllm-tree` 一键重新采集 reviewer 评论并检查框架接口是否漂移。同一个 PR 同时有本 skill 与 vllm-rocm-pr-review 的报告时，Pingboard dashboard 优先展示本 skill 的报告。

**📚 vllm-multimodal-open-issue-analyzer**

Prompt：

```text
/vllm-multimodal-open-issue-analyzer
```

输出：[vllm_multimodal_issues_20260425_030830](./skills/vllm-multimodal-open-issue-analyzer/outputs/vllm_multimodal_issues_20260425_030830.md)。

**📚 vllm-technical-blog-writer**

场景 1：根据主题写一篇完整文章：

```text
请以“vLLM 中的 Automatic Prefix Caching”为主题，分析最新代码和相关资料，并参考我过往的写作风格生成一篇完整的中文技术博客。请按需补充简洁的架构图和流程图。
/vllm-technical-blog-writer
```

场景 2：补全已有的 Markdown 草稿：

```text
请基于我提供的“vLLM多模态推理｜ViT_Full_CUDA_Graph\index.md”文章模板和参考资料，结合 vllm-project/vllm 最新 main 分支的代码，按 [Agent TODO] 的要求补充剩余内容。新增图片请放在文章同级的 ./images 目录中。
/vllm-technical-blog-writer
```

输出：新文章保存在 [`./skills/vllm-technical-blog-writer/outputs/<article-slug>/`](./skills/vllm-technical-blog-writer/outputs/) 下，引用的图片放在文章的 `images/` 目录中；已有草稿直接原地更新。文章标题统一采用 `XXX｜XXX` 两段式（明确讲 vLLM 的文章会带上“vLLM”）。大纲中的标题和要点仅作参考，可能会被润色；除非明确要求保留章节设计，章节也可能增删或调整顺序。

**📚 infx-bench-script-generator**

Prompt：

```text
我想在本地服务器上跑 dsv4/vllm/mi355x-fp4-mtp/agentic.yaml 的 override_dep8_c64 配置，并且需要用 torch profiler 采集一个完整的 prefill step，所有测试结果放到 /home/shashen/vllm/dsv4_benchmark 目录下，请帮我生成对应的测试脚本。
/infx-bench-script-generator
```

输出：每个 `<recipe>_<override>` 组合一个文件夹，包含 `01_docker_run.sh`（host 起容器）、`02_start_server.sh`（容器内起 server，可选 profiler 配置）、`03_run_client.sh`（容器内跑 AgentX replay）以及 `variant.json`/`variant.env`，示例见 [dsv4_vllm_mi355x-fp4-mtp_agentic_override_dep8_c64](./skills/infx-bench-script-generator/outputs/dsv4_vllm_mi355x-fp4-mtp_agentic_override_dep8_c64/)。

## 🛠️ 安装配置

**方式 1：使用 [`skills`](https://github.com/vercel-labs/skills) CLI 安装**（需要 Node.js）。

```bash
# 列出本仓库中可用的 skill
npx skills add shen-shanshan/vllm-dev-skills --list

# 为 Claude Code 安装一个或多个 skill（用户级，~/.claude/skills）
npx skills add shen-shanshan/vllm-dev-skills -g -a claude-code -s vllm-pr-desc-generator

# 为 Claude Code 安装全部 skill（用户级）
npx skills add shen-shanshan/vllm-dev-skills -g -a claude-code -s '*' -y

# 之后更新已安装的 skill
npx skills update -g
```

去掉 `-g` 则安装到当前项目（`./.claude/skills`）；修改 `-a` 可以安装给其他 agent（`'*'` 表示全部）。CLI 安装的是 GitHub 仓库的快照，仓库有更新时需要运行 `npx skills update` 才会生效。

**方式 2：软链接本地 clone**（修改 clone 中的文件后立即生效）。

```bash
# 将单个 skill 软链接到 Claude Code 的官方 skill 目录
ln -sfn /path/to/vllm-dev-skills/skills/skill-name ~/.claude/skills/skill-name

# 将全部 skill 软链接到 Claude Code 的官方 skill 目录
for dir in /path/to/vllm-dev-skills/skills/*/; do
  name=$(basename "$dir")
  ln -sfn "$dir" ~/.claude/skills/"$name"
done
```

## 📖 参考资料

- [Agent Skills](https://agentskills.io/home#why-agent-skills)
- [Extend Claude with skills](https://code.claude.com/docs/en/skills)
- [Configure Claude Code in VSCode](https://api.lingyaai.cn/model_intro/Claude-Code.html)
- [Configure Claude Code CLI with Claude HUD](https://github.com/jarrodwatts/claude-hud)
- [Awesome DeepSeek Agent](https://github.com/deepseek-ai/awesome-deepseek-agent)

## ©️ 引用

```bibtex
@misc{vllm-dev-skills@2026,
  title  = {vllm-dev-skills},
  url    = {https://github.com/shen-shanshan/vllm-dev-skills},
  note   = {Open-source software available at https://github.com/shen-shanshan/vllm-dev-skills},
  author = {shen-shanshan},
  year   = {2026}
}
```

## 📜 许可证

本项目采用 Apache-2.0 许可证，详见 [<u>LICENSE</u>](./LICENSE)。

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=shen-shanshan/vllm-dev-skills&type=Timeline)](https://www.star-history.com/#shen-shanshan/vllm-dev-skills&Timeline)
