# PR #54682: [ROCm][Perf] Optimize MiniMax-M3 decode indexer and top-k

> **Author**: @Fangzhou-Ai | **State**: OPEN | **Date**: 2026-09-01
> **Branch**: fork → `main` | **Labels**: `rocm`, `minimax`
> **Changes**: +1877 -327 lines across 6 files | **ROCm 相关性**: 完全相关
> **Assignee**: @shen-shanshan
> **Tracking**: [#54681](https://github.com/vllm-project/vllm/issues/54681)

## 1. 动机 (Motivation)

MiniMax-M3 在 ROCm decode 上对每一层稀疏注意力都会重算 index score 和 top-k（不改 `index_topk_freq`）。gfx950 TP4 BF16 路径上，旧 scorer 按固定 `(num_reqs, num_kv_chunks)` 切分、selector 是 partial+merge 两阶段 bitonic，spec decode 多 query 行还会重复读同一块 index-K。本 PR 用 vLLM 自有 Triton 做三件事：按真实 block 数映射 CTA 并复用 K tile；融合 top-k / page-16 sparse table / context length；用 packed key 固定「分数降序、下标升序」。不合格的 arch/dtype/head_dim/请求数走旧路径。不改 AITER、不新增 AITER kernel；与 #52664（FP8 AITER）、#53448（prefill/FP8 store）路径不同。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/models/minimax_m3/amd/ops/index_topk.py` | 新增 `_decode_index_score_balanced_kernel`、`_decode_topk_fused_kernel`、资格函数；删除旧 partial/merge selector |
| `vllm/models/minimax_m3/amd/ops/sparse_pa.py` | 抽出 `_write_sparse_block_table_row_from_values`；decode AITER 可吃预构建 table |
| `vllm/models/minimax_m3/common/indexer.py` | ROCm 注册 `topk_completion_counter`；透传 fused table 与 counter |
| `vllm/models/minimax_m3/amd/model.py` | AITER sparse PA decode 时预分配 table，indexer 融合写出 |
| `vllm/models/minimax_m3/amd/sparse_attention_msa.py` | `forward(..., decode_sparse_table=)` |
| `tests/kernels/attention/test_minimax_m3.py` | 映射覆盖、bitwise 打分、图回放、launch policy、全序、fused table |

机制：更少的空转 CTA、一次加载 K 打完整 head×query tile、少一次 selector launch、AITER 路径少一次 table kernel。描述把 WHY 写清楚了，不是纯 WHAT。OpenAI Codex 辅助实现与数字，人工审阅清单未勾。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 性能 | — | 2 | — |
| 正确性 | — | 1 | — |
| 测试 | — | 2 | — |
| 可维护性 | — | 1 | 1 |
| 兼容性 | — | 1 | — |

未发现可写出具体触发输入的静默数值 🔴。原子汇合 + Graph 复位有单测覆盖，不把「AMD atomics 可见性」升到必须修复。

---

**⚠️【性能】balanced scorer 按 `seq_lens.shape[0] <= 11` 门控，与 CUDA Graph 填充 batch 以及作者给出的 c15 数字对不上** `[已验证]`（c15 仍走 fast scorer `[推测]`）

- **问题**: `_decode_score_program_budget`（`index_topk.py`）只在 gfx950 + BF16 + `head_dim==128` 且 **`1 <= num_reqs <= 11`** 时返回 budget；`num_reqs = seq_lens.shape[0]`。vLLM decode 的 `seq_lens` 在 FULL/PIECEWISE 图里通常是 **capture 时的 padded 行数**（`max_num_seqs`），不是 live 并发。`NUM_REQUESTS` 还是 `tl.constexpr`，图捕获会把这个行数烤进 kernel。HEAD 上并发 15 或 `max_num_seqs>11` 会 **静默回退** `_decode_index_score_kernel`。作者却报告「graph-real request-balanced scorer」在 c1/c5/c10/**c15** 中位加速 10.7%/11.1%/29.5%/**58.1%**——与 HEAD 的 `MAX_DECODE_SCORE_BALANCED_REQUESTS = 11` 矛盾，更像 rebase 前另一套门控，或 microbench 传的是未填充的 live 行数。
- **影响**: 触发输入：**gfx950 MiniMax-M3 BF16 decode，CUDA Graph 捕获 `seq_lens.shape[0] > 11`（常见 `max_num_seqs≥16`），或 live batch=15**。生产 serving 可能完全吃不到 scorer 收益，只剩下 fused selector；PR 正文把 scorer 加速和 ITL 写在一起，容易让 reviewer 以为 c15 已在 HEAD 上走 fast scorer。
- **行动**: 作者应当说明 indexer 传入的 `seq_lens` 是 live 行还是 padded 行；若是 padded，应按 live block 映射、资格看非零行，或把预算放到 capture 时的 `max_num_seqs` 并证明零长 padding 不会打乱 CTA 映射。建议 review 时追问 c15 scorer 58% 是在哪一版、`seq_lens.shape[0]` 当时是多少。

**⚠️【测试】serving ITL / GSM8K / 部分 kernel 数字不能当作当前 HEAD 的事实** `[已验证]`

- **问题**: 作者写明 ITL 来自 rebase 前 `5678bb1a68` vs `1dc464d426`，每档并发只有一对 conditioned A/B，**将在 rebase 后复测才标 ready**。清单「warmed serving A/B has been reconfirmed on the rebased head」未勾。kernel 12 例 A/B 有方法学（512 warmup、32 次 AB/BA、200 次 graph replay、HIP event），但仓库里没有 harness 脚本。GSM8K 与「newly enabled TP and query-length layouts」也未重跑。按数字溯源规则，ITL 9.8%–14.3%、scorer 58%、selector 37% 对 **当前 diff 均为 `[unverified]`**。
- **影响**: reviewer 无法用 HEAD 复现最亮眼的 serving 数字；窄 case 作者自己另进程只复现 3.36%，和表里 3.45%–10.7% 同量级，说明小 batch 收益本身就不稳。
- **行动**: 作者应当在当前 rebase 头上重跑 TP4 AgentX ITL（多对、报 TP-rank max），并把 kernel harness 以 gist/测试附件形式给出。未完成前不要勾 PR checklist 的 Test Result 当作已验证。

**⚠️【正确性】GSM8K 非逐样本等价，reasoning e2e 不在本 PR** `[已验证]`

- **问题**: 五-shot GSM8K 1319 题：baseline 1254 strict；candidate 两次 1247 / 1252。McNemar p=0.4638 / 0.8804。作者不宣称 exact-output 等价；同 commit 两次 session 有 53 题评分变化。@liuzijing2014 要的 reasoning bench 被指到 #54845，本 PR 无数据。packed key 会改变并列分数下的 block 次序，相对旧 bitonic 路径这是有意的全序，但仍可能改稀疏模式。
- **影响**: 触发输入：**MiniMax-M3 gfx950 BF16 decode + 本 selector**。聚合对齐不能排除 agentic/reasoning 轨迹漂移。并列 top-k 从「不稳定」变成「下标更小优先」，与 HuggingFace 参考实现若用另一种并列规则会系统性偏差。
- **行动**: 作者应当在本 PR 链回 #54845 的具体分数，或补一条 MiniMax reasoning/agentic 集；并写明并列分数的全序是否与 CUDA/参考一致。建议 review 时不要把 McNemar「不显著」读成「无差异」。

**⚠️【测试】新 GPU 路径在非 ROCm CI 上全部 skip，gfx950 无自动回归网** `[已验证]`

- **问题**: 新增用例大量 `skipif(not current_platform.is_rocm())`。vLLM 主 CI 是 CUDA；AMD 队列即使有也多为 gfx942，本 PR 的 fast 资格还要求 **gfx950**。CPU 侧只覆盖 mapping/launch policy。PR checks 对 fork 的 Claude review 是关的。
- **影响**: 合入后 fused kernel、atomic 复位、K-tile 多头点积只能靠作者本地 `118 passed, 13 skipped`。后续改 `index_topk.py` 的静默数值错误会先打到 MI355 用户。
- **行动**: 建议作者在描述里写明「必须 gfx950 人工跑 `tests/kernels/attention/test_minimax_m3.py`」；maintainer 合入前在 MI355 上重跑该文件，不要只看 CUDA CI 绿。

**⚠️【兼容性】与 #52664 同时改 MiniMax-M3 indexer 热路径，互斥边界只靠文字** `[已验证]`

- **问题**: #52664 把合格 FP8 scoring/top-k 切到 AITER；本 PR 改同一套 `minimax_m3_index_decode` / indexer kwargs / sparse table 所有权。HEAD 用 `is_gfx950` + BF16 决定 Triton fast path，没有和 AITER FP8 入口做显式互斥 assert。本 PR 还把 fused table 写进 indexer 签名（必须四参数成套，且 `num_idx_heads==1`）。
- **影响**: 两 PR 若先后合入，容易出现：FP8 走 AITER scorer 但 ROCm indexer 仍传 fused table；或 Triton fused selector 盖掉 AITER top-k。触发输入：**同时开启本优化与 #52664 的 FP8 indexer**。
- **行动**: 作者应当在 dispatch 处按 dtype 显式分支（BF16 Triton vs FP8 AITER），并在 PR 正文指定 rebase 顺序。建议 review 时让 #52664 作者对一下签名。

**⚠️【可维护性】近 2k 行 Triton 由 Codex 生成，人工审阅未完成；`ADAPTIVE_FINAL_MERGE` 对 2/4/8/16 复制同一 merge** `[已验证]`

- **问题**: 作者勾选「Codex assisted with implementation, benchmarks, validation, duplicate-work research, and this description」，且「human submitter has reviewed every changed line」未勾。`_decode_topk_fused_kernel` 里 `active_chunks <= 2/4/8` 四段几乎逐字复制 `_merge_store_decode_topk`。`SINGLE_TILE_GUARANTEED` 的 then/else 前半段也重复。
- **影响**: 原子汇合、volatile partial load、packed key、init/local 哨兵分（1e30/1e29）任何一处抄错都是静默错 top-k。未逐行人工审的情况下，合入成本在后续调试而不在当前 CI。
- **行动**: 作者应当勾完 accountability 后再标 ready；建议把 merge 宽度收成循环或一张表。建议 review 时对 fused kernel 做一次人工走读（尤其 `pid_chunk >= active_chunks` 早退是否漏 `atomic_add`——当前早退不加计数、单 chunk 不碰 counter，逻辑自洽，但要确认没有「部分 chunk 早退却仍等 `NUM_TOPK_CHUNKS`」的版本残留）。

**📝【设计】fused sparse table 仍是单 local head 合同，与 generalized scorer 的 1/2/4 head 宣传容易混读** `[已验证]`

- **问题**: scorer/selector 对 local index head 1/2/4 共用 fast 路径；`EMIT_SPARSE_TABLE` 在 `num_idx_heads != 1` 时 `ValueError`。AITER sparse PA 也要求 1 KV head。Triton sparse attention 仍可用多头 `topk_idx`。
- **行动**: 建议作者在 `amd/model.py` 调用点旁注明「仅 AITER decode 传 fused kwargs」；PR 描述已有这段，代码注释再钉一次即可。

## 4. 现有讨论 (Existing Discussion)

- **@liuzijing2014**：要求 reasoning e2e。作者回复结果发在 #54845，本 PR 未附。
- **@AndreasKaratzas**：其余多为 NIT；问 kernel launch 后是否必须在 Python 里显式 `synchronize`。测试里 `torch.accelerator.synchronize()` 是为了 host 断言，不是 HIP 对同 stream 后续 kernel 缺隐式序。生产 `minimax_m3_index_decode` 热路径没有这层 sync，方向正确。
- **claude[bot]**：fork 上自动 review 关闭。
- 作者自述：仍是 draft until 人工逐行审完 + rebase 后 serving A/B；GitHub 时间线已 `ready_for_review`，与正文 checklist 不一致。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**

这是一条机制清楚的 gfx950 MiniMax-M3 decode 热路径优化：工作量映射、K tile 复用、融合 selector、Graph 完成计数器都对症，kernel 单测（bitwise / 全序 / replay）也比平均 perf PR 厚。没有写成 🔴 的具体正确性触发器。

但 HEAD 上 **scorer 对 `seq_lens.shape[0]>11` 静默回退**、**ITL/GSM8K 不是当前 rebase 的数字**、**Codex 大 kernel 尚未勾人工审阅**、以及 **#52664 签名重叠**，都足以挡住「可以合入」。保持 open 可以，标 ready 应等到：① 澄清 padded `num_reqs`；② HEAD 上重跑 TP4 ITL；③ 人工走读 fused kernel 并勾 accountability。
