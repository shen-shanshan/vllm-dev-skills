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
