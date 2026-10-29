# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project Layout

- `skills/<skill-name>/SKILL.md` — skill definitions; generated reports go to `skills/<skill-name>/outputs/`.
- `TODO.md` — roadmap, personal learning plan for technical reports, and blog writing tasks.

## Rules

### Sync generated technical reports to `TODO.md`

每次通过 `vllm-feature-tutorial` 或 `vllm-model-tutorial` 生成技术报告后，必须同步更新 `TODO.md`：

1. 在 `## Technical Reports（个人学习计划）` 小节的末尾追加一行，格式如下（路径相对于仓库根目录）：

   ```markdown
   - [ ] skills/vllm-feature-tutorial/outputs/<report_name>.md (Generated: ✅ / Read: )
   - [ ] skills/vllm-model-tutorial/outputs/<report_name>.md (Generated: ✅ / Read: )
   ```

2. 如果该报告路径已在 `TODO.md` 中存在（例如重新生成同一份报告），不要重复添加。
3. 不要修改已有条目的 `Read` 状态和复选框，这些由用户自己维护。
4. 完成后在回复中告知用户 `TODO.md` 已更新。

### Sync project docs when adding/updating skills

每次在 `skills/` 下新增或更新 skill 后，必须完成以下同步工作。这里的“更新 skill”仅指修改 skill 定义本身（`SKILL.md` 及其配套的 references / scripts / templates 等文件）；使用 skill 生成内容到 `skills/<skill-name>/outputs/` 下不算更新，不触发本规则。

1. **链接新 skill 到 Claude Code 官方 skill 目录**（仅新增 skill 时）：将其软链接到本机 `~/.claude/skills/` 下，使 Claude Code 能识别：

   ```bash
   ln -sfn "$(pwd)/skills/<skill-name>" ~/.claude/skills/<skill-name>
   ```

   完成后用 `ls -l ~/.claude/skills/<skill-name>` 确认链接指向正确。若同名链接已存在且指向本仓库，无需重复创建。
2. **更新 `README.md`**：
   - 新增 skill：在 `## 🚀 Skills` 表格中按 Category 将其插入到合适位置（Name 链接到 `./skills/<skill-name>/SKILL.md`，填写 Category / Description / Rec），并在 `## 📖 Usages` 中补充对应的使用示例（Prompt 等），保持与已有条目一致的格式。
   - 更新 skill：若 skill 的功能、描述、触发方式或用法有变化，同步修改表格中的 Description 及 `## 📖 Usages` 中的对应内容。
3. **更新项目介绍博客**：调用 `vllm-technical-blog-writer` skill，更新 `docs/blogs/introduction_to_vllm_dev_skills/` 下的博客（`introduction_to_vllm_dev_skills.md` 与 `zhihu.md` 需保持内容同步），在对应的研发环节小节（如 2.1 环境准备 ~ 2.9 技术写作）中新增或修改该 skill 的介绍，必要时同步调整“后续计划”等相关段落。同时将两篇博客开头 NOTE 中的日期（`本文将会随项目演进持续更新（Last Update：YYYY/MM/DD）`）更新为当天日期。
4. 完成后在回复中告知用户以上哪些内容已更新（软链接 / `README.md` / 博客）。
