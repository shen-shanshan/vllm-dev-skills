# DeepSeek-V4.1-Flash（MI355X）MXFP8 激活量化融合：进度

更新时间：2026-11-16

## 背景

- vLLM upstream main `3494bb0417` 已合入 #67377：在 gfx950 上开启 AITER 时，DSv4.1 的全部 MXFP8 32x32 dense linear 都走 AITER FlyDSL GEMM（`gemm_a8w8_blockscale_bpreshuffle`），`wo_a` 走 AITER batched GEMM（`batched_gemm_a8w8_mxscale_bpreshuffle`）。
- FlyDSL GEMM 只接受 FP8 的 A 和 row-major `[M, K/32]` E8M0 scale，输出 BF16。没有预量化输入时，每个 GEMM 前都会单独跑一次 `mxfp8_e4m3_quantize`（trace 中的 `_kernel.kd`）。原来 Triton 路径在 M≤8 时把量化放在 GEMM 内部（#63179），切换到 FlyDSL 后这个能力没有了。
- vLLM 的 consumer 已能把 `QuantizedActivation`（FP8 + row-major E8M0 scale）交给 FlyDSL（`RocmDotScaledMxfp8LinearKernel.apply_weights` → `rocm_mxfp8_block32_gemm(..., weight_preshuffled)`）。

## 数据来源

- **prefill**：`vllm-profile-traces/dsv41_flash/dep2_8k1k_c256_prefill/`。MI355X ×2，DEP2（attention DP2 + EP2），AITER MegaMoE，8k/1k random，c256。取 rank 1 的 `execute_context_2(16372)` step（485 ms）中以完整 M（约 16.4k）执行的层（第 1–20 层）的单次耗时。
- **decode**：旧 trace `vllm-profile-traces/dsv41_flash/agentx_dep8_c256/`（DEP8，AgentX，#67377 之前，dense GEMM 还走 Triton，但量化 kernel 相同），仅作参考。

## 可与 FlyDSL dense GEMM 融合的量化站点

| # | GEMM（N×K） | 上游 producer → `_kernel` | prefill 量化 (µs) | decode 量化 (µs) | 涉及的层 | 在 producer 侧融合的做法 | 现状 |
|---|---|---|---:|---:|---|---|---|
| 1 | **`wo_b`**（5120×8192） | `wo_a`（AITER `bmm_a8w8_mxscale`，只输出 BF16） | **77** | 4.7 | 每层 | 改 aiter，让 bmm 的 epilogue 输出 MXFP8；或者 `wo_a` 改回 Triton，用 `QUANT_OUT` epilogue | #63821、#66856（均基于 Triton `wo_a`，需要和 FlyDSL 版对比） |
| 2 | **`wq_a+wkv`**（1792×5120） | Mega-MHC（输出 BF16） | 41 | 3–5 | 每层（第 0 层除外） | 给 `mega_mhc` 传 `out_dtype="mxfp8"`（aiter 已支持，gfx950 上输出 row-major scale） | **已完成（本地，未提交）**，见 `mega_mhc_mxfp8_pr.md` |
| 3 | **shared `gate_up`**（4608×5120） | pre-FFN x（Mega-MHC 输出） | 41 | 4.8 | 每层 | 同上，用 FFN 侧的 Mega-MHC 输出 mxfp8。MegaMoE 在 `megamoe_prepare` 里还会自己量化一次，不做 `forward_prequant`（compact 路径已在 `megamoe_prepare` 内融合量化） | **已完成（本地，未提交）**，见 `mega_mhc_mxfp8_pr.md` |
| 4 | **shared `down`**（5120×2304） | `act_and_mul`（BF16，本身 48 µs） | 20 | 5.9 | 每层 | 新写一个 `silu_and_mul + MXFP8 quant` kernel，两个 kernel 合成一个 | 无人在做，aiter 只有 MXFP4 版本 |
| 5 | **`wq_b`**（32768×1280） | q/kv RMSNorm（`kernel.kd`，35 µs） | 11 | 4.8 | 每层 | norm 和量化合成一个 kernel | #66572（Triton，bit-exact）、#65837（aiter，非 bit-exact） |
| 6 | **indexer `wq_b`**（4096×1280） | 同一份 qr 又量化一遍 | 13 | 约 5 | 有 indexer 的层 | 和 #5 共用同一份量化结果 | 上面两个 PR 都已覆盖 |
| 7 | **engram `wkv`**（25600×6144） | engram 查表（表本身是 MXFP8，被反量化成 BF16 后再量化） | 52–57 | 5.7 | 第 1、14 层 | `EngramBatch` 直接输出 FP8 和 row-major scale | 无人在做 |
| 8 | `wq_a+wkv`（第 0 层） | aiter `add_rmsnorm_quant` 之后又跑一次 `_kernel` | **213（异常）** | — | 只有第 0 层 | 第 0 层不走 Mega-MHC。这次量化比其他层的同类量化（41 µs）慢了 5 倍，原因待查 | — |

另外，DSpark draft 的 `main_proj`（5120×15360）也走这个 GEMM，但只在 decode step 中执行，这次 prefill 窗口里没有出现。

## 总量与占比

- **prefill**（DEP2，485 ms step）：独立量化 `_kernel` 合计约 4.7 ms，约占 1.0%。以完整 M 执行的层每层约 190 µs，约为 dense GEMM 时间（约 2.4 ms/层）的 8%。同一 step 里 MegaMoE（stage1 + stage2 + combine + prepare）约占 74%，dense GEMM（含 `wo_a`）约占 11%。
- **decode**（旧 DEP8 trace）：每层约 25–30 µs，每 step 约 1 ms，约占 5–6%。

## 两种融合方式

- **A. 在上游 producer 里融合（推荐，尤其是 prefill）。** producer 直接输出 `QuantizedActivation`，FlyDSL GEMM 不用改。代价是每个站点要单独实现：#2、#3 靠 aiter 现成能力接线，#5、#6 已有 PR，#4、#7 要新写 kernel，#1 依赖 aiter 或 Triton `wo_a`。
- **B. 把量化放进 GEMM 的 prologue。** 需要 aiter 给 FlyDSL GEMM 加一个"A 是 BF16、kernel 内量化"的变体，一次改动就能覆盖上表所有站点。对 decode 的小 M 最划算；对 prefill 的大 M 不划算，因为每个 N tile 都要把 A 重新量化一遍。

## 建议

- **prefill：** 量化融合整体只值约 1%。按单项收益排序是 #1（77 µs）> #2 + #3（82 µs，一个接口改动就能同时拿到）> #4。#8 第 0 层那 213 µs 的异常也值得顺手查一下。
- **decode：** 先补抓一份 FlyDSL 路径下的 decode trace，确认每个站点的实际耗时，再决定走 A 还是推动 aiter 做 B。

## #2 / #3 结果（2026-11-16）

- 代码：vLLM `60bd6dd2ed` 上的本地改动（5 个 ROCm-only 文件），PR 描述见 `mega_mhc_mxfp8_pr.md`。
- prefill trace：`vllm-profile-traces/dsv41_flash/dep2_8k1k_c256_prefill_mhc_mxfp8/`。
  - `mega_mhc → _kernel` 和 `ep_combine → _kernel`（shared `gate_up`）都从每 step 40 次降到 0 次；
  - 每 step 独立量化从 211 次 / 4.73 ms 降到 131 次 / 2.58 ms；
  - 以完整 M 执行的层，"2 次 Mega-MHC + 2 次 seam 量化"从 885–896 µs 降到 859–876 µs。Mega-MHC 的 mxfp8 模式每次多花约 25–30 µs，抵消了一部分收益，净省约 0.5 ms/step（约 0.1%）。
- GSM8K（`gsm8k/`）：基线 strict 0.9257 / flexible 0.9272，改动后 strict 0.9356 / flexible 0.9348，差异在噪声范围内（kernel 级 bit-exact）。
- 局限：decode 时每个 rank ≤128 token 且处于 FULL graph capture 中，会走 mHC overlap 路径（layer input 来自 Triton），这部分 decode 没有收益。

## 待办

- [ ] 补抓一份 FlyDSL 路径下的 decode trace（DEP2，与 prefill 配置相同）
- [x] 查 #8：第 0 层 `wq_a+wkv` 输入的量化 213 µs 没能复现，后两次 trace 都是约 39 µs，判断是首次运行的偶发情况
- [ ] 查启动日志中 aiter 对各 shape 报 "w_scale 1x32 is untuned" 的原因（trace 中 kernel 名带 `wb32x32`）
- [ ] 确定融合方向（A / B）和第一个要做的站点
