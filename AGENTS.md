# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project Layout

- `skills/<skill-name>/SKILL.md` — skill definitions; generated reports go to `skills/<skill-name>/outputs/`.
- `README.md` / `README_zh.md` — project README in English (default, shown on GitHub) and Simplified Chinese; keep them in sync.
- `docs/workflow/` — the AI-native workflow diagram shown in both READMEs: `gen_workflow.py` is the source, `workflow_en.svg` / `workflow_zh.svg` are generated from it.
- `website/` — GitHub Pages site (MkDocs Material), deployed by `.github/workflows/pages.yml` on push to `master`: `https://shen-shanshan.github.io/vllm-dev-skills/`. `hooks.py` generates the pages at build time — the home page from `README_zh.md`, and one page per file in `skills/vllm-feature-tutorial/outputs/` and `skills/vllm-model-tutorial/outputs/` (no other skill outputs are published); `catalog.yml` holds the group, card title and summary of each tutorial. Preview locally with `pip install -r website/requirements.txt && mkdocs serve -f website/mkdocs.yml`.
- `TODO.md` — roadmap, personal learning plan for technical reports, and blog writing tasks.

## Rules

### Keep `README.md` and `README_zh.md` in sync

README 分为英文版 `README.md`（默认展示）和中文版 `README_zh.md` 两份。任何时候修改 README（无论是否因为新增 / 更新 skill），都必须同时更新这两份文件：

1. 两份内容和结构保持一致（同样的章节、表格行、使用示例和链接），只是语言不同。
2. 每份只用一种语言，不要中英文混用：英文版中的 Prompt 示例也翻译成英文，中文版中的说明文字使用中文；代码、命令、路径、skill 名、专有名词和 benchmark 原始输出等保持原样。
3. 两份文件顶部的语言切换链接（`**English** | [简体中文](./README_zh.md)` / `[English](./README.md) | **简体中文**`）保持不变。
4. 网站主页由 `README_zh.md` 的“项目简介”“AI-native 工作流”“Skill 列表”“安装配置”四节自动生成。修改这几节的标题时，同步修改 `website/hooks.py` 中 `_build_home()` 引用的标题。

### Sync generated technical reports to `TODO.md`

每次通过 `vllm-feature-tutorial` 或 `vllm-model-tutorial` 生成技术报告后，必须同步更新 `TODO.md`：

1. 在 `## Technical Reports（个人学习计划）` 小节的末尾追加一行，格式如下（路径相对于仓库根目录）：

   ```markdown
   - [ ] skills/vllm-feature-tutorial/outputs/<report_name>.md (Generated: ✅ / Read: )
   - [ ] skills/vllm-model-tutorial/outputs/<report_name>.md (Generated: ✅ / Read: )
   ```

2. 如果该报告路径已在 `TODO.md` 中存在（例如重新生成同一份报告），不要重复添加。
3. 在 `website/catalog.yml` 的 `feature` / `model` 下为新报告登记一项（所属分组、卡片标题 `title`、一句话中文简介 `summary`），并确认报告开头有 `> **最后更新**: YYYY-MM-DD` 一行，网站据此显示更新时间和过期提醒。重新生成已有报告时更新这一行的日期。
4. 不要修改已有条目的 `Read` 状态和复选框，这些由用户自己维护。
5. 完成后在回复中告知用户 `TODO.md` 和 `website/catalog.yml` 已更新。

### Sync project docs when adding/updating skills

每次在 `skills/` 下新增或更新 skill 后，必须完成以下同步工作。这里的“更新 skill”仅指修改 skill 定义本身（`SKILL.md` 及其配套的 references / scripts / templates 等文件）；使用 skill 生成内容到 `skills/<skill-name>/outputs/` 下不算更新，不触发本规则。

1. **链接新 skill 到 Claude Code 官方 skill 目录**（仅新增 skill 时）：将其软链接到本机 `~/.claude/skills/` 下，使 Claude Code 能识别：

   ```bash
   ln -sfn "$(pwd)/skills/<skill-name>" ~/.claude/skills/<skill-name>
   ```

   完成后用 `ls -l ~/.claude/skills/<skill-name>` 确认链接指向正确。若同名链接已存在且指向本仓库，无需重复创建。
2. **更新 `README.md` 和 `README_zh.md`**（两份都要改，规则见 “Keep `README.md` and `README_zh.md` in sync” 一节）：
   - 新增 skill：在 skill 表格（英文版 `## 🚀 Skills`，中文版 `## 🚀 Skill 列表`）中按类别将其插入到合适位置（Name 链接到 `./skills/<skill-name>/SKILL.md`，填写类别 / 描述 / 推荐度），并在工作流一节（英文版 `## 🧭 AI-native Workflow`，中文版 `## 🧭 AI-native 工作流`）中，把它写进它所属的阶段（发现问题 / 学习原理 / 设计方案 / 开发代码 / 测试验证 / 提交代码 / 撰写博客，或 Maintainer、持续进化部分；没有合适阶段时可以新增或调整阶段），说明它在该阶段的用途、是否由 agent 自动调用，并附上示例 Prompt 或输出链接，保持与已有条目一致的格式。
   - 更新 skill：若 skill 的功能、描述、触发方式或用法有变化，同步修改两份 README 中表格的描述及工作流一节中的对应内容。
   - 重命名或删除 skill：同步修改表格、工作流一节和工作流图中的名称。
3. **更新工作流图**（新增 / 删除 / 重命名 skill，或 skill 所属阶段、用途变化时）：修改 `docs/workflow/gen_workflow.py` 中的 `STAGES` / `MAINTAINER` / `EVOLVE` 数据（中英文用途说明都要填，每行最多 3 个节点），然后运行 `python3 docs/workflow/gen_workflow.py`，同时重新生成 `workflow_en.svg`、`workflow_zh.svg` 以及博客用的 `docs/blogs/images/ai-native-workflow.png`（由中文版渲染）。不要手动编辑生成的 SVG。生成后用 `rsvg-convert` 渲染成 PNG 查看两张图（没有该命令时先 `sudo apt-get install -y librsvg2-bin fonts-noto-cjk`），确认文字没有溢出、重叠或缺字。
4. **更新项目介绍博客**：调用 `vllm-technical-blog-writer` skill，更新 `docs/blogs/` 下的博客（`introduction_to_vllm_dev_skills.md`、`zhihu.md` 与 `weixin.md` 需保持内容同步；`zhihu.md` 没有标题行、各级标题比前者高一级，Star 段落放在参考资料之前；`weixin.md` 按 `vllm-technical-blog-writer` 的 “WeChat version” 规则从前者重新生成），在工作流对应的阶段小节（“四、作为开发者”的 4.1 发现问题 ~ 4.7 撰写博客、“五、作为 vLLM Maintainer”或“六、让工作流持续进化”）中新增或修改该 skill 的介绍，必要时同步调整引言中的 skill 数量和总结等相关段落。
5. 完成后在回复中告知用户以上哪些内容已更新（软链接 / `README.md` 与 `README_zh.md` / 工作流图 / 博客）。
