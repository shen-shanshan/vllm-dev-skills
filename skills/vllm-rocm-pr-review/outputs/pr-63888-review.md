---
pr: 63888
repo: vllm-project/vllm
head_sha: dab9209c05c4434e31eeaa6f18e40170874d2ed6
verdict: lgtm
reviewed_at: 2026-11-03T09:06:32Z
skill: vllm-rocm-pr-review
---

# PR #63888: [ROCm][Perf][DSv4.1] Skip the top-k -1 fill on the rows vLLM's decode top-k writes

> **Author**: @okorzh-amd | **State**: OPEN | **Date**: 2026-10-27
> **Branch**: `okorzh/dsv41-indexer-skip-decode-topk-fill` → `main` | **Labels**: rocm, DSv4.1
> **Changes**: +105 -16 lines across 3 files | **ROCm 相关性**: 完全相关（gfx950 paged MXFP4 sparse indexer，DeepSeek-V4.1 / V4）
> 本报告分两部分：§1–6 为 PR 详细总结，§7–8 为 ROCm review 意见。

---

## 1. 总结 (Summary)

gfx950 上的 paged MXFP4 稀疏 indexer（`rocm_mxfp4_sparse_attn_indexer` / `rocm_mxfp4_sparse_mqa_indexer`）每层在 top-k 之前把本 step 的 `topk_indices_buffer[:num_tokens]` 全部填 -1。对于走 vLLM 自家 `top_k_per_row_decode` 的 decode 行（`rows < 128 且 k < 2048`），该 kernel 本身就会写满每行全部 k 个槽位（不足 k 的尾部写 -1），所以这次 fill 是死写。本 PR 抽出 `_vllm_topk(rows, k)` 谓词与 `_topk` 的 dispatch 共用，新增 `_fill_topk_rows` 只对 decode 行之后的行（prefill、尾部 padding）做 fill，从而在 DSv4.1-Flash decode FULL graph 中每 step 少 8 个 `FillFunctor<int>` kernel。

## 2. 背景与动机 (Background & Motivation)

- gfx950 MXFP4 indexer（#58671）+ gfx950 k=512 vLLM top-k（#56743）组合下，每个 index 层在 decode graph 内多一个 fill kernel（~4.5 us）。gfx942 indexer 早已在纯 decode 且无 padding 的 step 上跳过 fill（#60731），本 PR 把同样的思路推广到 gfx950，并且扩大到 mixed / padded step 的 decode 行。
- 历史背景：#44527 曾把 ROCm FP8 indexer 的 fill 全部删掉，#53906 又恢复（因为有行不会被写，导致脏索引）。所以本 PR 的关键问题是 **"被跳过 fill 的每一行每一列是否都保证被写"**——这是本 review 的重点（见 §4.2 场景表）。
- 作者声称收益很小：ABBA 下 c1 -13 us（-0.25%），c2 -7 us（-0.11%）；profile 中 in-graph fill 时间约 37 us/step。这些数字来自作者本地 MI355X 运行，未见可追溯日志，属 `[unverified]`，但方向与机制吻合（8 个 kernel 的移除）。

## 3. 代码修改分析 (Code Change Analysis)

### 3.1 修改的模块

| 文件 | 改动 |
|------|------|
| `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py` (+19/-3) | 新增 `_vllm_topk(rows, k)`（`rows < 128 and k < 2048`），`_topk` 改为调用它；新增 `_fill_topk_rows(layer, num_tokens)`，两个 indexer 入口用它替换原来的 `topk_indices_buffer[:num_tokens] = -1` |
| `tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py` (+24/-1) | CPU 测试 `test_topk_fill_skips_the_rows_vllms_topk_writes`：4 个 case 校验 fill 起点（7/512→7，128/512→0，7/2048→0，无 decode→0） |
| `tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py` (+62/-12) | `_run_layers` 增加 `k / pad / pool_blocks` 参数和尾部断言（padding 行必须是 -1、buffer 之外保持哨兵 7）；新增 GPU 测试 `test_decode_rows_skip_the_fill_in_mixed_and_padded_steps`（decode+3 padding、mixed、mixed+padding × k=32/512/1024），decode 行里还包含一个长度 0 的 padding 请求 |

### 3.2 架构 / 流程图

```mermaid
flowchart TD
    A[indexer 入口<br/>rocm_mxfp4_sparse_attn/mqa_indexer] --> B[_fill_topk_rows]
    B --> C{metadata.decode 非空 且<br/>_vllm_topk&#40;len&#40;decode_row_lens&#41;, k&#41;?}
    C -- 是 --> D[start = decode 行数<br/>只 fill [start, num_tokens&#41;]
    C -- 否 --> E[start = 0<br/>与 main 相同：全量 fill]
    D --> F[prefill chunks: _dense_prefill / _gather_prefill<br/>写 topk_buffer[t0:t1]]
    E --> F
    F --> G{decode 路径}
    G --> H[_dense_decode / _gather_decode<br/>rows = decode_row_lens.shape[0]]
    H --> I[_topk&#40;logits, lengths, topk_buffer[:rows,:k], k&#41;]
    I --> J{width >= k 且 not _vllm_topk?}
    J -- 是 --> K[aiter top-k<br/>（此时 fill 未被跳过）]
    J -- 否 --> L[torch.ops._C.top_k_per_row_decode<br/>写满每行 k 个槽位]
    L --> M[gather 路径再经 _remap_compact_topk<br/>改写全部 k 槽]
```

### 3.3 关键实现细节

- **谓词共享**：`_fill_topk_rows` 与 `_topk` 共用 `_vllm_topk`，且两处用的行数完全一致——`_fill_topk_rows` 用 `decode_row_lens.shape[0]`，`_dense_decode` / `_gather_decode` 的 `rows = lengths.shape[0]`，传给 `_topk` 的 logits/compact 的 `shape[0]` 也是这个 `rows`；k 两处都是 `layer.topk_tokens`。`_vllm_topk` 为真时 `_topk` 一定走 vLLM kernel（与 width 无关），所以"跳过 fill ⇒ vLLM kernel 写这些行"是精确等价，而不是近似。
- **纯形状决策**：不读 device tensor，FULL graph 捕获时与回放时的决策一致；breakable PIECEWISE 每 step 重新决定。
- **保守面**：`rows >= 128`、`k >= 2048`（GLM-5/DSv3.2 #61985）、无 decode 的 step、decode 之后的所有行都保留 fill。
- **未改 kernel 源码**；`csrc/sampler.cu` 与 main 一致。

## 4. 涉及的技术原理 (Technical Principles)

### 4.1 背景

- **DSA / 稀疏 indexer**：DeepSeek-V3.2/V4/V4.1 的 indexer 对每个 query token 打分全部（压缩后的）KV 位置，取 top-k（V4.1/V4-Flash k=512，V4-Pro k=1024）作为稀疏 MLA 的索引。`topk_indices_buffer` 是模型级共享的 `torch.empty([max_num_batched_tokens, index_topk], int32)`，**跨层、跨 step 复用**——任何没被写的槽位都会保留上一层 / 上一 step 的合法范围内的索引，下游稀疏 attention 读到后不会崩，只会静默算错。这就是 fill -1 存在的原因，也是 "stale index" 风险的来源。
- **vLLM `top_k_per_row_decode`**：每个 block 处理一行；`topKPerRowJob` 要么走 `rowLen <= topK` 短路（前 rowLen 写索引，余下写 -1），要么在最后的 store loop 中对 `i in [0, topK)` 全部写出。gfx950 k=512 的 device-length-aware 路径：单 block 直接写，或 split + merge 两次 launch，两次 launch 用同一个 `gfx950TopK512ActiveBlocks(rowEnd, gridDim.x, maxBlocks)` 计算 `activeBlocks`，`activeBlocks == 1` 时 merge 提前返回、split 直接写——每行恰好写一次。`rowEnd = max(0, ...)`，长度 0 的行写全 -1。
- **CUDA graph padding**：FULL uniform-decode graph 回放时，batch 被 padding 到 capture size；padding 请求的 `seq_len = 0`。

### 4.2 "跳过 fill 的行是否都会被写"——逐场景核对（本 review 重点）

| 场景 | 涉及行 | 谁保证写满 k 列 | 结论 |
|------|-------|----------------|------|
| 纯 decode，rows<128，k<2048 | `[0, rows)` | vLLM kernel（final store loop / 短路） | ✅ `[已验证]` csrc/sampler.cu `topKPerRowJob` |
| decode 行中含 padding 请求（seq_len=0） | 这些行在 `[0, rows)` 内 | `rowEnd = max(0, …)` → 短路写全 -1；flatten 路径 `decode_seq_lens_buffer[actual_expanded:] = 0` | ✅ `[已验证]` mla/indexer.py `_prepare_decode_tensors` |
| CUDA-graph padded batch（FULL） | padding 行计入 `num_decode_tokens` 时同上；不计入时位于 decode 行之后 | 前者由 kernel 写 -1，后者由 `_fill_topk_rows` fill | ✅ 两种都覆盖；新 GPU 测试的 `pad=3` 覆盖后者 |
| uniform 路径 padding 请求长度为负（`seq_len - next_n + j + 1 < 0`） | `[0, rows)` | kernel 内 `max(0, …)` 钳位 | ✅ `[已验证]` |
| k > 有效 token 数（短上下文） | `[0, rows)` | 短路分支写 `[rowLen, k)` 为 -1 | ✅ |
| spec decode（DSpark k=6，7 行/请求） | flatten 后每 token 一行，`rows = num_decode_tokens` | 同上；adaptive verification 时 `native_next_n = 1` 走 flatten | ✅ |
| native 多行序列 dense decode（`decode_native`） | logits 仍是 `(rows, width)`，`_topk` 的 rows 不变 | 同上 | ✅ |
| candidate consumer gather decode | `out = topk_buffer[:rows, :k]` | vLLM kernel 写满后 `_remap_compact_topk` 对 `cols < K` 全部改写（`hit` 否则 -1） | ✅ |
| mixed step（decode + prefill） | prefill 行在 decode 之后 | fill 保留 | ✅ 与 main 一致 |
| rows ≥ 128 或 k ≥ 2048（aiter top-k） | 全部 | fill 保留 | ✅ 与 main 一致 |
| rows ≥ 128 但 width < k（回退到 vLLM kernel） | 全部 | fill 保留（保守，不受影响） | ✅ |
| NaN / ±inf logits | `[0, rows)` | final store loop 无条件写 k 个槽（内容可能来自 smem 垃圾，但 main 上 fill 后同样会被这次写覆盖） | ✅ 行为与 main 等价，fill 本来就保护不了这种情况 |
| DCP / PCP | — | builder `__init__` 直接 `NotImplementedError` | 不适用 |
| out 视图步长 | `topk_buffer[:rows, :k]` | kernel 按 `rowIdx * topK` 寻址、不读 out 的 stride(0)；所有调用方 buffer 宽度 = `index_topk` | ✅ 目前成立，但为隐式不变式（见 §7 📝） |

结论：在当前代码下，被跳过的行列 **全部** 由 vLLM kernel（以及 gather 路径的 remap）写满，未发现 stale-index 漏洞。

## 5. 评论区讨论亮点 (Discussion Highlights)

- **@frida-andersson**（approve）：认为 k=1024 已被 GPU 测试与哨兵扫描覆盖，DSv4 e2e 非必须。作者更正：V4-Flash 的 `index_topk` 是 512（与 V4.1 相同），1024 是 V4-Pro；自 #63669 起 MI355X 上 V4-Flash 默认（AITER on + indexer dtype auto）就会走到这个 op。
- **@jamesETsmith**（non-blocking）：问非 `--monokernel` 路径是否做过精度验证。作者答：fill 跳过发生在 op 内部，mono 与常规路径调用同一个 op；GPU op 测试不经过 mono；GSM8K 8 并发时 49–56 行的 step 与所有 mixed step 实际走的都是常规路径。reviewer 接受。
- Code owner（tjtanaa / AndreasKaratzas / dllehr-amd）review 仍待定；CI 状态未知（页面未显示）。

## 6. 风险与潜在问题 (Risk Analysis)

| 风险 | 严重程度 | 说明 |
|------|---------|------|
| 静默脏索引（未来改动破坏 "vLLM kernel 写满 k 列" 的隐式契约） | Medium | 当前成立；但契约只写在 PR 描述里，代码中 `_fill_topk_rows` 无任何说明，见 §7 📝 |
| out buffer 宽度 ≠ topk_tokens 时 kernel 按 stride k 写 | Low | 既有隐式假设；跳过 fill 后该假设也承担了防脏索引的职责，见 §7 📝 |
| GPU 测试只在可选 AMD CI job 中运行 | Medium | 默认 CI 只跑 CPU plan test；`test_top_k_per_row.py` 在 gfx950 无 CI |
| DSv4（V4-Flash 21 个 indexer 层）未做 e2e | Low | 同一 op 同一 k=512；op 测试与哨兵扫描已覆盖 |
| 性能收益很小 | Low | -0.25%（c1），数字 `[unverified]`，但无回退 |
| 验证用 commit 与当前 head 不同 | Low | 见 §7 📝 |

---

## 7. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 可维护性 | — | — | 2 |
| 测试 | — | — | 1 |
| 注释/文档 | — | — | 1 |

未发现 🔴 / ⚠️ 级问题：§4.2 的逐场景核对表明，在当前代码下所有被跳过 fill 的行与列都由 `top_k_per_row_decode`（及 gather 路径的 `_remap_compact_topk`）写满。

**📝【可维护性】`_fill_topk_rows` 依赖的跨函数不变式没有写进代码** `[已验证]`

- **问题**: `rocm_paged_mxfp4_indexer.py:456-466` 的 `_fill_topk_rows` 没有 docstring / 注释。它的正确性依赖三个不在同一函数里的事实：(1) `_dense_decode` / `_gather_decode` 写的恰好是 `topk_buffer[:decode_row_lens.shape[0], :topk_tokens]`；(2) `_vllm_topk` 为真时 `_topk` 必走 `top_k_per_row_decode`；(3) 该 kernel 对每行无条件写满 k 槽（短行尾部写 -1）。PR 描述里也承认 "共享谓词只保护对 `_vllm_topk` 的修改，`_topk` 其它分支的修改只能靠 GPU 测试发现"。
- **影响**: 将来有人改 decode 路径（例如只对部分行跑 top-k、引入新的 decode top-k kernel、或给 vLLM 分支加 arch/width 条件）时，buffer 是跨层跨 step 复用的 `torch.empty`，未写行会留下**合法范围内的旧索引**，稀疏 MLA 静默算错而不会崩溃——这正是 #44527→#53906 的历史问题。
- **行动**: 建议作者在 `_fill_topk_rows` 加 2–4 行注释写明上述契约（并指向 `test_decode_rows_skip_the_fill_in_mixed_and_padded_steps`），在 `_topk` 的 vLLM 分支旁注明 "this branch must write all k slots of every row; `_fill_topk_rows` relies on it"。

**📝【可维护性】out buffer 宽度 = `topk_tokens` 的假设现在也承担防脏索引职责** `[已验证]`

- **问题**: `_topk` 调用 `top_k_per_row_decode(logits, 1, lengths, out, rows, logits.stride(0), logits.stride(1), k)` 时传的是 `topk_buffer[:rows, :k]` 视图，但 kernel 按 `rowIdx * topK` 连续寻址、不读 out 的 `stride(0)`。目前 DSv4/V4.1 的所有 `topk_indices_buffer` 均为 `[max_num_batched_tokens, index_topk]`，所以成立。
- **影响**: 若将来某模型 buffer 宽度 > `topk_tokens`（例如多个 k 共享一个 buffer），main 上就已经会写错位置；本 PR 之后这些行还会额外留下未被覆盖的旧索引。属既有隐患，不是本 PR 引入。
- **行动**: 建议作者在 `_fill_topk_rows` 或 `_layer` 中加一个廉价的 host 侧断言 `topk_buffer.shape[1] == topk_tokens`（或 `out.is_contiguous()`），把这个假设显式化。

**📝【测试】GPU 侧验证只在可选 AMD CI job 中** `[已验证]`

- **问题**: 新 GPU 测试只在可选的 `:amd: (MI355 DPX) Attention Kernels` 中运行（需要 `/amd-ci run` + Unblock）；`test_top_k_per_row.py` 的 gfx950 用例无 CI；默认 CI 只跑 CPU plan test。作者已在描述中如实说明。
- **影响**: 本 PR 的正确性完全依赖 "kernel 写满 k 列" 的契约，而守护该契约的 GPU 测试默认不跑；未来 sampler.cu 的改动可能在 CI 绿的情况下破坏它。
- **行动**: 建议合入前由 AMD 侧触发一次 `/amd-ci run`（Attention Kernels + DSv4-Flash Disaggregated DP EP），并考虑把 `test_top_k_per_row.py -k gfx950` 加入某个 MI355 job。

**📝【注释/文档】PR 描述中的验证 commit 与当前 head 不一致** `[已验证]`

- **问题**: 描述引用的 commit 为 9fc85b7dda（CPU 测试）、4eddaa283f（扫描 / profile / ABBA / GSM8K）、f3dc96acdb（"current head"），而 PR 当前 head 为 `dab9209c05`（commits `1bca205db0` + `dab9209c05`），说明之后又 rebase / force-push 过。作者声称差异仅为注释。
- **影响**: 读者无法直接确认数据对应的代码；性能数字（-13 us / -7 us）本身也无可追溯日志，属 `[unverified]`。
- **行动**: 建议作者把描述中的 "current head" 更新为 `dab9209c05` 并注明与 4eddaa283f 的 diff 为空（或仅注释）。

## 8. 结论 (Verdict)

**✅ LGTM**

改动小且精确：跳过条件与 `_topk` 的 dispatch 共用同一谓词、同一行数和同一 k，所有被跳过的行列都由 vLLM decode top-k（短路写 -1 或 final store loop 写满）以及 gather 路径的 remap 覆盖；padding 请求、CUDA-graph padding、k > 有效长度、spec decode、mixed step 均已逐一核对并有 GPU 测试覆盖。建议补充代码内的不变式注释与 buffer 宽度断言（均为 📝），并在合入前触发一次可选 AMD GPU CI。

## 9. 英文 Review 评论 (Copy-Paste English Comments)

**C1** `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py:456-466` — 📝 comment

```text
I walked through the cases (zero-length padding requests inside the decode rows, FULL-graph padded batches, rowLen < k, flattened spec-decode rows, the gather path's remap, NaN logits) and every skipped row/column is written by top_k_per_row_decode, so this looks correct to me. One request: the correctness of `_fill_topk_rows` relies on an invariant that lives in three other places — `_dense_decode`/`_gather_decode` writing exactly `topk_buffer[:decode_row_lens.shape[0], :topk_tokens]`, `_topk` always taking vLLM's kernel when `_vllm_topk` holds, and that kernel storing all k slots of every row. Since `topk_indices_buffer` is a shared `torch.empty` reused across layers and steps, breaking any of these later would leave in-range stale indices (the #44527/#53906 failure mode) rather than crash. Could you add a short docstring here stating that contract, and a one-line note next to the vLLM branch in `_topk`?
```

**C2** `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py:308` — 📝 comment

```text
Minor, pre-existing: `top_k_per_row_decode` addresses its output at `rowIdx * topK` and never reads `out.stride(0)`, so passing `topk_buffer[:rows, :k]` is only correct while the buffer's width equals `topk_tokens` (true for every DSv4/V4.1 caller today). With the fill now skipped on these rows, that assumption also becomes what keeps them free of stale indices. Consider a cheap host-side `assert layer.topk_buffer.shape[1] == layer.topk_tokens` (in `_layer` or `_fill_topk_rows`) to make it explicit. (Anchored here because the kernel call itself is unchanged.)
```

**C3** `tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py:496-498` — 📝 comment

```text
Thanks for the mixed/padded test — it is exactly the coverage this skip needs. Since it only runs in the optional `:amd: (MI355 DPX) Attention Kernels` job and nothing runs `test_top_k_per_row.py -k gfx950` in CI, could someone trigger `/amd-ci run` before merge? Also, the description's "current head" (f3dc96acdb) no longer matches the PR head (dab9209c05); a one-line update would help readers map the numbers to the code.
```
