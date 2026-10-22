# DeepSeek-V4-Pro MXFP4 Indexer：GPQA Diamond 精度测试结果

> 测试日期：2026-10-22
> 对应 PR：#60834（[ROCm][Perf] Apply paged MXFP4 sparse indexer dense path for DeepSeek-V4）
> 测试方法见 `dsv4_mxfp4_indexer_gpqa_niah_eval.md`。本次只测 GPQA，NIAH 还没测。

## 测试环境

| 项目 | 值 |
|---|---|
| 硬件 | 8 × AMD Instinct MI350X（gfx950），TP8 |
| 代码 | `/home/shashen/vllm/vllm`，commit `8e008a3049` 加上处理 review 意见后尚未提交的改动（workspace manager、`compress_pool` JIT warmup、`ptr_range_bytes`） |
| 模型 | `/shared/models/deepseek-ai/DeepSeek-V4-Pro`（fp8 权重） |
| 服务参数 | `--kv-cache-dtype fp8 --max-model-len 163840 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9`，`VLLM_ROCM_USE_AITER=1`，默认开启 thinking；MXFP4 额外加 `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` |
| 评测工具 | lm_eval 0.4.13，`local-chat-completions --apply_chat_template` |
| 任务 | `gpqa_diamond_cot_zeroshot`（198 题），指标为 flexible-extract 的 `exact_match` |
| 生成参数 | `temperature=0`（贪心解码），`max_gen_toks=131072`，并发 64 |
| 数据 | `Idavidrein/gpqa`（gpqa_diamond），已缓存到 `/shared/models/huggingface/datasets/Idavidrein___gpqa` |
| 脚本 | `benchmark/gpqa/serve.sh`、`benchmark/gpqa/run_gpqa.sh` |
| 结果文件 | `benchmark/gpqa/results/gpqa_{fp8,mxfp4}_gen131072/`，日志在 `benchmark/gpqa/*.log` |

## 结果

| Indexer | GPQA Diamond（exact_match） | Stderr | 空回答（推理超出预算） | 非空回答上的准确率 | GPU KV 容量 | 用时 |
|---|---|---|---|---|---|---|
| Main（FP8 Indexer） | 0.7121 | ±0.0323 | 13 / 198 | 0.7622 | 1,878,431 tokens | 约 64 分钟 |
| This PR（MXFP4 Indexer） | 0.7121（持平） | ±0.0323 | 11 / 198 | 0.7540 | 2,051,155 tokens（+9.19%） | 约 66 分钟 |

### 逐题对比

| 情况 | 题数 |
|---|---|
| 两边都答对 | 124 |
| 只有 FP8 答对 | 17 |
| 只有 MXFP4 答对 | 17 |
| 两边提取出的答案完全相同 | 155 / 198 |
| McNemar 精确检验 | p = 1.0（两边没有差异） |

两边的分歧完全对称（17 比 17）。分歧来自很长的推理链：indexer 的数值精度略有不同，几万个 token 累积下来，贪心解码的路径会逐渐分叉，属于正常的波动。没有看到 MXFP4 系统性变差的迹象。

### 运行状态

两边服务的 198 个请求全部返回 200。MXFP4 服务的日志里没有 Traceback，没有 workspace lock 相关的报错，也没有 `unknown asm constraint`。

## 为什么这个测试能覆盖 indexer

indexer 只有在上下文超过 `topk × compress_ratio = 1024 × 4 = 4096` 个 token 时才真正做稀疏选择。GPQA 的 prompt 只有约 500 token，但开启 thinking 之后，推理链普遍有上万甚至十几万 token。所以在 decode 的大部分时间里，上下文都远超 4096，每一步都在走 MXFP4 indexer 的 top-k 选择。这正是 reviewer 担心 gsm8k 覆盖不到的路径。

## 踩过的坑：生成预算太小会让分数失真

第一次用 `max_gen_toks=32768` 跑 FP8，得分只有 **0.5202**，其中 **71 / 198** 题的回答是空的。原样重放其中一题，结果是 `finish_reason=length`：32,768 个 token 全部用在了推理上，还没来得及给出答案就被截断了。在有回答的 127 题里，准确率是 0.811，说明模型本身是正常的。

把预算提高到 131072 之后，空回答降到 11～13 题。剩下这些很可能是贪心解码在推理中陷入了循环。两边的设置完全一致，所以这不影响对比结论，但会压低绝对分数。

32K 那次的结果保存在 `benchmark/gpqa/results/gpqa_fp8_gen32768/`，只作为记录，不参与对比。

## 写进 PR 的内容（英文）

```markdown
**GPQA Diamond** (`gpqa_diamond_cot_zeroshot`, 198 questions, lm_eval 0.4.13, chat completions with thinking, greedy, `max_gen_toks=131072`), DeepSeek-V4-Pro TP8 on MI350X:

| Indexer | GPQA Diamond | GPU KV cache capacity |
|---|---|---|
| Main (FP8 Indexer) | 0.7121 ± 0.0323 | 1,878,431 tokens |
| This PR (MXFP4 Indexer) | 0.7121 ± 0.0323 | 2,051,155 tokens (9.19% ↑) |

With thinking, reasoning traces run to tens of thousands of tokens, so most decode steps are far past the 4096-token point where the indexer starts its sparse top-k selection. Per question, 124 are correct in both runs and the disagreements are symmetric (17 vs 17, McNemar p = 1.0).
```

## 待办

- [ ] NIAH（8K / 32K / 128K），步骤见 `dsv4_mxfp4_indexer_gpqa_niah_eval.md`
- [ ] 手动触发 `:amd: (MI355 DPX) Attention Kernels Shard %N` CI，并回复 review
