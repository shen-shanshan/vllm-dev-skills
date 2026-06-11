---
name: vllm-dev-task-discovery
description: Analyze contribution opportunities in the vllm-project/vllm repository for community developers. Given a module, feature, or model area, this skill collects information from open issues, recent PRs, GitHub discussions, code TODOs/FIXMEs, roadmap labels, and maintainer activity to generate a structured Markdown report of actionable tasks (feature development, model support, performance optimization, bug fixes, documentation, refactoring). Each task includes difficulty, prerequisite knowledge, related maintainers/contributors, urgency, and importance ratings. Use when the user wants to find contribution opportunities in vllm, understand what tasks are available in a specific area, or get a developer demand analysis. Triggered by requests like "我想了解vllm的多模态模块中还有哪些可以贡献的事情", "帮我分析vllm XXX模块的社区需求", "整理vllm中XXX相关的task", "vllm contribution opportunities in XXX", "what can I work on in vllm's XXX area", "vllm developer demand analysis for XXX".
---

# vLLM 社区开发需求观察分析

## 概述

本 skill 帮助社区开发者在 vllm-project/vllm 仓库中找到可贡献的 task/idea。通过系统性搜集多渠道信息（Issue、PR、Discussion、代码 TODO、Roadmap、文档），生成按类型归类的结构化需求清单报告。

## 工作流

```
用户指定目标模块/特性/模型  →  多源数据搜集  →  需求提炼与分类  →  评估与标注  →  生成报告
```

### 第一步：明确目标范围

从用户输入中提取关键信息：

- **目标模块/特性/模型**：例如 "多模态"、"Prefix Caching"、"Qwen3-VL"、"Scheduler"、"V1 Engine"
- **Top N**：用户可指定生成 top N 个 task，默认 20～100（建议默认 30）
- **任务类型偏好**（可选）：特性开发 / 模型支持 / 性能优化 / Bugfix / 文档补充 / 代码重构

如果用户描述模糊，用简短问题确认范围（例如"你指的是多模态中的视觉编码器 ViT 部分，还是整个多模态 pipeline？"）。

### 第二步：多源数据搜集

**必须**从以下所有渠道平行搜集信息。每个渠道的搜索关键词应基于用户指定的目标模块确定。

#### 2.1 Open Issues（未关闭的 Issue）

```bash
# 搜索开放 issue（最近更新的优先）
gh search issues --repo vllm-project/vllm --state open --limit 50 --sort updated --match title,body "<关键词1>" "<关键词2>"

# 按 label 筛选
gh issue list --repo vllm-project/vllm --state open --limit 50 --label "<相关label>" --sort updated

# 获取 issue 详情（含评论区）
gh issue view --repo vllm-project/vllm <issue-number> --comments
```

关注以下信号：
- label 为 `good first issue` / `good second issue` / `help wanted` 的 issue
- 评论区中有 maintainer 明确指出的待做事项
- 近 30 天内活跃的 issue
- 带有 `roadmap` / `feature request` / `bug` label 的 issue

#### 2.2 近期合并的 PR

```bash
# 搜索近期合并的 PR（了解模块正在做什么）
gh search prs --repo vllm-project/vllm --state merged --limit 30 --sort updated --match title,body "<关键词>"

# 按 label 筛选 PR
gh pr list --repo vllm-project/vllm --state merged --limit 30 --label "<相关label>" --sort updated
```

关注以下信号：
- PR 中 maintainer 提到的"后续可以做的优化"（follow-up）
- PR 评论区中标记为 TODO 的讨论项
- 合并后仍存在的已知限制

#### 2.3 GitHub Discussions

```bash
# 搜索讨论
gh search discussions --repo vllm-project/vllm --match title,body "<关键词>" --limit 20
```

关注：
- 社区用户提出的需求或痛点
- maintainer 对某些特性规划的表态
- Roadmap 讨论

#### 2.4 仓库 Roadmap 与里程碑

```bash
# 获取里程碑信息
gh api repos/vllm-project/vllm/milestones --jq '.[] | select(.state=="open") | {title, description, due_on}'

# 搜索 Roadmap 相关 issue
gh issue list --repo vllm-project/vllm --state open --label roadmap --limit 20
```

关注：
- 带有季度/版本里程碑的 issue
- 带 `roadmap` / `priority` / `p0` / `p1` label 的 issue

#### 2.5 代码中的 TODO/FIXME/HACK

```bash
# 必须在本地 clone 的 vllm 仓库中执行
# 如果本地没有 clone，先 clone:
# git clone https://github.com/vllm-project/vllm.git /tmp/vllm --depth 1

grep -rn "TODO\|FIXME\|HACK\|XXX" /tmp/vllm/vllm/ --include="*.py" | grep -i "<目标相关>"
```

#### 2.6 文档中的缺失项

```bash
# 搜索 vllm docs 目录
ls /tmp/vllm/docs/source/
# 检查是否有相关模型/特性的文档缺失
```

关注：
- `docs/source/features/` 下缺失的特性说明
- `docs/source/models/` 下缺失的模型文档
- README 中标注为 "coming soon" 或 "experimental" 的内容

#### 2.7 社区活跃贡献者与 Maintainer 识别

```bash
# 相关模块最近的 contributor
gh api repos/vllm-project/vllm/commits --jq '.[].author.login' | sort | uniq -c | sort -rn | head -20

# 查看相关 PR 的 reviewer
gh pr list --repo vllm-project/vllm --state merged --limit 30 --search "<关键词>" --json reviews --jq '.[].reviews[].author.login' | sort | uniq -c | sort -rn
```

### 第三步：需求提炼与分类

将搜集到的信息提炼为具体的、可执行的 task。每个 task 必须满足：

- **具体可执行**：不是模糊的"优化性能"，而是"为 Qwen3-VL 的图像预处理添加 batch 支持"
- **范围适中**：一个 task 应在 1～3 周内可完成（对社区开发者而言）
- **有关联来源**：来源于具体的 issue、PR 讨论或代码注释

按以下类型归类：

| 类型 | 说明 | 典型来源 |
|------|------|---------|
| 🚀 特性开发 | 新功能/新能力的开发 | Feature Request / Roadmap / PR follow-up |
| 🧩 模型支持 | 新模型的适配和接入 | Model Request issue / PR |
| ⚡ 性能优化 | 延迟/吞吐/内存优化 | Performance issue / Benchmark PR |
| 🐛 Bug 修复 | 已知 bug 的修复 | Bug issue / 代码 HACK/FIXME |
| 📖 文档补充 | 文档新增或改进 | 缺失文档 / Doc issue label |
| 🔧 代码重构 | 代码质量/架构改进 | 代码 TODO / Maintainer 反馈 |
| ✅ 测试补充 | 测试覆盖/CI 改进 | Testing label / Coverage gap |

### 第四步：评估与标注

对每个 task 给出以下维度的评估：

#### 难度评估

| 难度 | 标记 | 说明 |
|------|------|------|
| Beginner | ⭐ | 适合首次贡献者，涉及文档/简单配置/单文件修改 |
| Intermediate | ⭐⭐ | 需要一定的领域知识，涉及多文件修改或简单逻辑 |
| Advanced | ⭐⭐⭐ | 需要深入理解 vllm 架构，涉及核心模块修改 |

#### 前置知识与技能

列出完成该 task 需要掌握的：
- Python 相关技能（如 async/await、类型注解、CUDA 基础）
- vLLM 相关知识（如 Scheduler 机制、Model Runner 流程、Multimodal Pipeline）
- 外部知识（如特定模型架构、HuggingFace Transformers）
- 工具链（如 CUDA profiling、GitHub Actions）

#### 社区联系人

标注相关模块的：
- 主要 maintainer（GitHub ID）
- 活跃 contributor（GitHub ID）
- 关联的 issue/PR 编号

#### 紧急程度

| 级别 | 说明 |
|------|------|
| 🔴 High | Roadmap 中有明确 deadline、与特定版本/季度强绑定 |
| 🟡 Medium | 社区讨论活跃、多人关注、短期内理应完成 |
| 🟢 Low | 长期规划项、无明确时间要求 |

#### Roadmap 归属

如果可以从 label 或里程碑中推断，标注属于哪一年哪个季度的 Roadmap（如 `2026 Q3 Roadmap`）。

#### 重要程度

| 级别 | 说明 |
|------|------|
| ⭐⭐⭐ | 核心功能缺失/影响面广/社区强烈需求 |
| ⭐⭐ | 有价值的改进，但非阻塞 |
| ⭐ | Nice to have，锦上添花 |

### 第五步：生成报告

报告必须严格遵循下方模板结构，保存至 `<skill_root>/outputs/vllm_dev_demand_<target>_<timestamp>.md`。

## 报告模板

```markdown
# vLLM 社区开发需求分析报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 🎯 目标模块 | <用户指定的模块/特性/模型> |
| 📅 分析日期 | YYYY-MM-DD |
| 📊 Task 总数 | N（若用户指定 top N） |
| 🔍 数据来源 | Issues / PRs / Discussions / Code TODOs / Roadmap / Docs |
| 💡 分析范围 | 近 X 个月活跃的内容（建议默认 6 个月） |

---

## 📋 需求总览（Top N Tasks）

| # | 类型 | 标题 | 难度 | 紧急度 | 重要度 | 来源 |
|---|------|------|------|--------|--------|------|
| 1 | 🚀 | <task 标题> | ⭐⭐ | 🔴 | ⭐⭐⭐ | #12345 |

---

## 🚀 特性开发

### Task N: <task 标题>

| 维度 | 详情 |
|------|------|
| **难度** | ⭐⭐ Intermediate |
| **前置知识** | • Python async/await<br>• vLLM Scheduler 机制<br>• ... |
| **社区联系人** | @maintainer_id (核心 reviewer)<br>@contributor_id (相关 PR 作者) |
| **紧急程度** | 🔴 High — 属于 2026 Q3 Roadmap |
| **重要程度** | ⭐⭐⭐ 核心功能 |
| **关联来源** | • Issue: [#12345](https://github.com/vllm-project/vllm/issues/12345)<br>• PR: [#12350](https://github.com/vllm-project/vllm/pull/12350)（相关讨论） |
| **建议起点** | <具体的代码文件路径或入口函数> |

**背景与描述：**

<1-2 段说明这个 task 是什么、为什么需要做、社区当前的讨论状态>

**预期工作项：**

1. <具体步骤 1>
2. <具体步骤 2>
3. <具体步骤 3>

---

## ⚡ 性能优化

### Task N: <task 标题>
...

---

## 🐛 Bug 修复

### Task N: <task 标题>
...

---

## 🧩 模型支持

### Task N: <task 标题>
...

---

## 📖 文档补充

### Task N: <task 标题>
...

---

## 🔧 代码重构

### Task N: <task 标题>
...

---

## ✅ 测试补充

### Task N: <task 标题>
...

---

## 📊 附录

### A. 关键数据来源汇总

| 来源类型 | 数量 | 备注 |
|----------|------|------|
| Open Issues | X | 其中 good first issue: Y 个 |
| Related PRs | X | 近 6 个月合并 |
| Discussions | X | |
| Code TODOs | X | |
| Roadmap Items | X | |

### B. 活跃社区联系人

| GitHub ID | 角色 | 相关模块 | 活跃度 |
|-----------|------|---------|--------|
| @xxx | Maintainer | Multimodal | 高 |

### C. 参考链接

- [vLLM Roadmap](https://github.com/vllm-project/vllm/labels/roadmap)
- [Good First Issues](https://github.com/vllm-project/vllm/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
```

## 质量准则

1. **可执行性优先**：每个 task 必须具体到可以立即上手，避免模糊描述
2. **来源可追溯**：每个 task 必须关联到至少一个具体来源（issue/PR/代码行号）
3. **数量控制**：严格遵守用户指定的 top N，类型均衡分布，不集中在单一类型
4. **难度梯度**：确保 Beginner / Intermediate / Advanced 任务都有，方便不同水平的开发者
5. **时效性**：优先纳入近 3 个月内活跃的内容，超过 6 个月无更新的 issue/PR 降权
6. **诚实标注**：如果一个 task 的信息不完整（如无法判断紧急度），标注为"待确认"而非编造

## 搜索时的注意事项

- `gh search issues` 和 `gh search prs` 的 `--match` 参数支持 `title`、`body`、`comments`
- 关键词用空格分隔表示 AND 关系，用引号包裹实现精确短语匹配
- 如果 `gh search discussions` 不可用，使用 `gh api graphql` 或 WebFetch 替代
- 如果本地没有 vllm 仓库，先 `git clone https://github.com/vllm-project/vllm.git /tmp/vllm --depth 1`
- 代码搜索时优先搜索 `vllm/` 下的 `.py` 文件，忽略 `tests/` 目录（除非目标明确是测试补充）
- GitHub API 有速率限制，短时间内大量请求会被限制。有 `gh` CLI 时无限制，否则需要在请求间等待
