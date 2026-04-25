---
name: vllm-multimodal-open-issue-analyzer
description: Fetch and organize multimodal-related open issues from vllm-project/vllm. Categorizes issues by problem type (Bug, Feature Request, Performance, CUDA Graph, EPD disaggregation, Prefix Caching, ViT/visual encoder, Video, Audio/Speech, specific VL models, etc.) and generates a structured Markdown report saved to the skill's ./outputs directory. Use when the user wants to collect, search, analyze, or summarize open issues related to multimodal features or models in the vllm repository. Triggered by requests like "搜集vllm中多模态相关的issue", "整理vllm的多模态open issue", "查一下vllm有哪些多模态问题还没解决", "帮我分析vllm多模态issue", "vllm multimodal open issues report".
---

# vllm 多模态 Open Issues 分析

## 工作流

1. 运行 `scripts/fetch_multimodal_issues.py`
2. 脚本自动抓取、去重、分类，生成 Markdown 报告
3. 报告默认保存至 `<skill_root>/outputs/vllm_multimodal_issues_<timestamp>.md`
4. 告知用户报告路径及 issue 总数

## 运行脚本

```bash
python3 scripts/fetch_multimodal_issues.py
```

可选参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--token` | GitHub token（有 gh CLI 时不需要） | `--token ghp_xxx` |
| `--output-dir` | 自定义报告输出目录 | `--output-dir /tmp/reports` |
| `--keywords` | 追加额外关键词（逗号分隔） | `--keywords "DeepSeek-VL,Kimi"` |
| `--stdout` | 输出到终端而非文件 | `--stdout` |

## 搜索关键词覆盖范围

**通用术语**: multimodal, multi-modal, VLM, vision language, image input

**VL 模型**: Qwen-VL/2/2.5, LLaVA, InternVL, MiniCPM-V, Phi-3/4-vision, PaliGemma, Florence, Pixtral, CogVLM, LLaMA-Vision, Idefics

**Omni/音频**: Qwen-Omni/2/2.5, audio, whisper, speech

**视频**: video understanding, video input, temporal

**特性关键词**: ViT, visual encoder, cuda graph, EPD, prefix caching, image token, image patch

## 问题类型分类逻辑

脚本对每个 issue 的标题、label、body（前 500 字符）做关键词匹配，归入以下类别（一个 issue 可属于多个类别）：

- Bug / 错误报告
- Feature Request / 功能请求
- Performance / 性能问题
- CUDA Graph / 计算图
- EPD / 编解码分离
- Prefix Caching / 前缀缓存
- ViT / 视觉编码器
- Video Multimodal / 视频多模态
- Audio / Speech / 音频语音
- VL Model: Qwen-VL系列 / Qwen-Omni系列 / LLaVA系列 / InternVL系列 / 其他模型
- Image Token / 输入处理
- Compatibility / 兼容性
- Documentation / 文档
- 其他 / Other

## 报告结构

生成的 Markdown 报告包含三部分：

1. **Issue 总览**（表格）：编号、标题、作者、创建日期、Labels
2. **按问题类型分类**：每个类别列出匹配 issue（含链接、创建日期）
3. **分类统计**：各类别 issue 数量汇总表

## 认证说明

- **有 gh CLI**（推荐）：自动使用，无速率限制
- **无 gh CLI，有 token**：使用 `--token`
- **无认证**：搜索批次间自动等待 6 秒，避免触发 GitHub 10 req/min 限制
