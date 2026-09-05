# vLLM vs ATOM Decode Trace 对比 — MiniMax-M3

> 数据来源：torch profiler Chrome-trace JSON，两份 trace 均采集自 MiniMax-M3、TP4、8k 输入 / 1k 输出、concurrency 8、MI300X（单 rank 捕获）。
> 分析方法：从 6 个 decode step 中取中间 step（窗口 2），按「occurrence-index」取每类层的中间层样本进行单层单次对比。
> 审计线索：`vllm-minimax-m3-tp4-8k1k-conc8_digest.md` / `_slice.json`、`atom-minimax-m3-tp4-8k1k-conc8_digest.md` / `_slice.json`（与本报告同目录）。

## 1. 基本信息

| 项目 | vLLM | ATOM |
|---|---|---|
| 引擎版本 | vLLM 0.28.1rc1.dev199+g7c5dc571c（rocprofiler-sdk 1.1） | ATOM（roctracer） |
| HIP runtime / driver | 70253211 / 70253211 | 70253211 / 70253211 |
| GPU | 4× MI300X（sm9.5 × 256 SMs，每卡 8 GCD） | 4× MI300X（同上） |
| 通信 | world_size 4、backend nccl、pg_size 4、rank 0（单 rank 捕获） | 同上 |
| Trace 事件（ph X） | 8807（kernel 6590 / cpu_op 1478 / cuda_runtime 714） | 7256（kernel 5832 / cpu_op 1062 / cuda_runtime 277 / gpu_memcpy 72） |
| Decode step 数 | 6（gpu 通道 `execute_context_*` 注解） | 6（同上） |
| 所选 step 窗口 | 窗口 2，wall 9013.8 µs | 窗口 2，wall 9183.2 µs |
| 窗口内 kernel | 1098 个 / 48 种，busy 8968.4 µs（99.5%） | 972 个 / 39 种，busy 9096.2 µs（99.1%） |
| Stream | stream 4（1096 kernel，99.9%）+ stream 2（2 个杂核）→ 单流 | stream 3（100%）→ 单流 |
| Replay 稳定性 | replay_stable = True | replay_stable = True |

**采集一致性说明**：两份 trace 为同模型、同 TP4/8k1k/conc8 配置（用户确认）。观察到一个差异：**vLLM 的 MSA 层 KV cache 为 fp8**（`fusedMiniMaxM3QNormRopeKVInsertKernel` 模板 `unsigned char` + `Fp8KVCacheDataType=1`），**ATOM 为 bf16**（`fusedQKNormIdxrQKNormKernel` 模板 `hip_bfloat16` + `Fp8KVCacheDataType=0`）。若这是 `kv_cache_dtype` 配置差异，则 KV 显存占用量不可比（fp8 减半），但对本报告的**计算路径对比**影响很小（该 kernel 两引擎耗时几乎相同，见 §3）。

## 2. 模型结构简介

MiniMax-M3（~428B 总参数 / ~23B 激活，hidden 6144）的 60 层 decoder 由两种层组成：

- **层 0–2（3 层）：dense GQA + MLP**。全 GQA 注意力（`kernel_unified_attention`）+ SwiGLU-OAI MLP（`dense_intermediate_size = 12288`）。
- **层 3–59（57 层）：MSA + MoE**。MiniMax Sparse Attention：Index Branch（4 个 query head + 1 个共享 key head，block size 128）按 block 做 Top-k 检索（topk 16 blocks → 注意力长度 ≤ 2048），Main Branch 对选中的 KV block 做精确 block-sparse softmax attention；MoE 部分 routed expert intermediate 3072 + 1 个 shared expert。

**dense 层 forward 流程**（trace 实测 kernel 序列，≈ 第 1 层）：

```
residual add + Gemma RMSNorm（输入）
 → QKV 投影 (hgemm_bf16_16x64x64x8)
 → QK-Norm + RoPE + KV 写 (fusedMiniMaxM3QNormRopeKVInsert / fusedQKNormIdxrQKNorm)
 → KV cache 写 (reshape_and_cache_kernel_flash)
 → dense GQA attention (kernel_unified_attention) → 输出归并 (reduce_segments)
 → O 投影 (_gemm_a16_w16, TP 切分) → allreduce
 → residual add + RMSNorm
 → gate 投影 → SiLU → up 投影 → allreduce(TP 切分) → residual add + RMSNorm
```

**MSA+MoE 层 forward 流程**（trace 实测 kernel 序列，≈ 第 31 层）：

```
residual add + RMSNorm（输入）
 → QKV 投影 (hgemm_bf16_16x64x64x7)
 → QK-Norm + RoPE + KV 写（同上，MSA 层 vLLM 用 fp8 KV）
 → index score 计算 → top-k 索引选择/合并 → block-sparse attention
 → 注意力输出归并 → O 投影 (_gemm_a16_w16) → allreduce → residual add + RMSNorm
 → MoE router 投影 → top-k 门控 → expert 排序 → grouped GEMM(gate+up, ck_tile MoeFlatmm)
 → SiLU → grouped GEMM(down, ck_tile MoeFlatmm) → allreduce → residual add + RMSNorm
```

两种层型均每层 2 次跨卡通信（O 投影输出 1 次 + MLP/MoE 输出 1 次），TP4 下 gate/up 侧权重做复制计算、O 投影与 down 侧输出做 allreduce。

## 3. 逐层类型对比

耗时单位为 µs，为**单层单次执行**的 GPU 时间。MSA 行取 57 层均值、dense 行取 3 层均值（digest §4 的 `mean_us`）；gate/up/down 与两处通信因同一 kernel 承担不同角色，按所选 step 的单层 occurrence 拆分取值（digest §5，跨 step 中位数偏差 <5%）。「实现差异」列使用固定标签；差异% 为 ATOM 相对 vLLM（负 = ATOM 更快）。

### 3.1 dense GQA + MLP 层（≈ 第 1 层）

| 逻辑算子 | vLLM kernel(s) | ATOM kernel(s) | 实现差异 | vLLM (µs) | ATOM (µs) | 差异% | 结论 |
|---|---|---|---|---|---|---|---|
| 残差加 + RMSNorm（输入） | `_gemma_fused_add_rmsnorm_kernel` | 无独立 kernel | **融合通信**（并入 allreduce_fusion） | 4.8 | 0 | — | 见通信行合计 |
| QKV 投影 | `hgemm_bf16_16x64x64x8_SPK6` | 同名 | 结构等价 | 9.0 | 9.2 | +2% | 持平 |
| QK-Norm + RoPE + KV 写 | `fusedMiniMaxM3QNormRopeKVInsertKernel`（bf16 KV） | `fusedQKNormIdxrQKNormKernel` | 结构等价 | 4.6 | 4.3 | −7% | ATOM 略快 |
| KV 布局/写 | `reshape_and_cache_kernel_flash` 4.6 | 3× `at::elementwise` 布局拷贝 12.6 + `reshape_and_cache_kernel_flash` 4.4 | 不同算子（ATOM 多 3 个 layout copy） | 4.6 | 17.0 | +270% | **vLLM 显著更快** |
| Attention | `kernel_unified_attention` | 同名 | 结构等价 | 27.4 | 27.2 | −1% | 持平 |
| Attn 输出归并 | `reduce_segments` | 同名 | 结构等价 | 4.7 | 4.6 | −2% | 持平 |
| O 投影 | `_gemm_a16_w16_...BLOCK_SIZE_N_32...` | `_gemm_a16_w16_...BLOCK_SIZE_N_16...` | **不同算子**（tile N_32 vs N_16） | 11.9 | 8.4 | −29% | **ATOM 显著更快** |
| 通信 + 残差加 + Norm（attn 块后） | `cross_device_reduce_1stage` 8.7 + `_gemma_fused_add_rmsnorm` 4.8 | `allreduce_fusion_kernel_1stage` | **split vs fused** | 13.5 | 10.0 | −26% | **ATOM 更快** |
| gate 投影 | `hgemm_bf16_16x64x128x5_SPK2` | 同名 | 结构等价 | 17.4 | 18.9 | +9% | ATOM 略慢 |
| SiLU | `_swiglu_oai_kernel` | 同名 | 结构等价 | 4.2 | 4.0 | −5% | 持平 |
| up 投影 | `hgemm_bf16_16x64x128x5_SPK2` | 同名 | 结构等价 | 10.7 | 10.5 | −2% | 持平 |
| 通信 + 残差加 + Norm（MLP 块后） | `cross_device_reduce_1stage` + `_gemma_fused_add_rmsnorm` | `allreduce_fusion_kernel_1stage` | **split vs fused** | 13.5 | 10.0 | −26% | **ATOM 更快** |
| **单层合计** | | | | **126.3** | **124.1** | **−1.7%** | ATOM 略快 |

要点解读：

- dense 层两引擎总体接近（差 1.7%）。ATOM 的收益全部来自两处：**allreduce+残差+RMSNorm 三合一融合**（每层两处合计省 7.0 µs）与 **O 投影 GEMM tile 差异**（省 3.5 µs）。
- ATOM 最大的短板是 KV 写之前多了 3 个 `at::elementwise` 布局拷贝（每层 ~12.6 µs，全部落在 dense 层，共 C=9），疑似为 unified attention 做的 contiguity 转换——vLLM 的 `reshape_and_cache_kernel_flash` 直接消化了这一步。

### 3.2 MSA + MoE 层（≈ 第 31 层）

| 逻辑算子 | vLLM kernel(s) | ATOM kernel(s) | 实现差异 | vLLM (µs) | ATOM (µs) | 差异% | 结论 |
|---|---|---|---|---|---|---|---|
| 残差加 + RMSNorm（输入） | `_gemma_fused_add_rmsnorm_kernel` | 无独立 kernel | **融合通信** | 4.8 | 0 | — | 见通信行合计 |
| QKV 投影 | `hgemm_bf16_16x64x64x7_SPK6` | 同名 | 结构等价 | 9.5 | 10.2 | +7% | ATOM 略慢 |
| QK-Norm + RoPE + KV 写 | `fusedMiniMaxM3QNormRopeKVInsertKernel`（**fp8 KV**） | `fusedQKNormIdxrQKNormKernel`（bf16 KV） | 结构等价 / **量化 KV 差异** | 4.9 | 5.0 | +2% | 持平 |
| Index score 计算 | `_decode_index_score_balanced_kernel` | `_decode_index_score_topk_partial_kernel` | 不同算子 | 7.7 | 11.1 | +44% | **vLLM 更快** |
| Top-k 索引合并 | `_decode_topk_fused_kernel` | `_topk_index_merge_kernel` | 不同算子（vLLM 已 fused） | 4.7 | 6.6 | +40% | **vLLM 更快** |
| Block-sparse attention | `_gqa_sparse_decode_kernel` | `paged_attention_decode_sliding_window_head_1` | 不同算子 | 7.9 | 11.6 | +47% | **vLLM 更快** |
| Attn 输出归并 | `_merge_topk_attn_out_kernel` | `pa_decode_ps_reduce_hip_kernel` | 不同算子 | 4.8 | 4.5 | −6% | 持平 |
| O 投影 | `_gemm_a16_w16_...N_32...` | `_gemm_a16_w16_...N_16...` | **不同算子**（tile） | 11.9 | 8.4 | −29% | **ATOM 显著更快** |
| 通信 + 残差加 + Norm（attn 块后） | `cross_device_reduce_1stage` 8.7 + `_gemma_fused_add_rmsnorm` 4.8 | `allreduce_fusion_kernel_1stage` | **split vs fused** | 13.5 | 10.0 | −26% | **ATOM 更快** |
| MoE Router 投影 | `_rocm_fp32_router_gemm_kernel`（fp32 计算） | `hgemm_bf16_16x64x64x6_SPK12`（bf16） | 不同算子 | 5.4 | 5.2 | −4% | 持平（ATOM 略快） |
| Top-k 门控 | `aiter::grouped_topk_kernel` | `aiter::topk_gating_kernel_opt` | 不同算子 | 4.5 | 4.2 | −7% | ATOM 略快 |
| Expert 排序 | `aiter::opus_moe_sorting_entry` | 同名 | 结构等价 | 6.9 | 6.7 | −3% | 持平 |
| MoE buffer fill | `at::vectorized_elementwise`（FillFunctor） | 同名 | 结构等价 | 4.4 | 4.3 | −2% | 持平 |
| grouped GEMM（gate+up） | `ck_tile::MoeFlatmm`（a16w16） | 同名（不同模板） | 结构等价 | 26.4 | 29.4 | +11% | ATOM 略慢 |
| SiLU | `aiter::swiglu_act_and_mul_kernel` | 同名 | 结构等价 | 4.5 | 4.2 | −7% | 持平 |
| grouped GEMM（down） | `ck_tile::MoeFlatmm`（第二变体） | 同名（不同模板） | 结构等价 | 14.6 | 15.9 | +9% | ATOM 略慢 |
| 通信 + 残差加 + Norm（MoE 块后） | `cross_device_reduce_1stage` 8.7 + `_gemma_fused_add_rmsnorm` 4.8 | `allreduce_fusion_kernel_1stage` | **split vs fused** | 13.5 | 10.0 | −26% | **ATOM 更快** |
| **单层合计** | | | | **149.9** | **147.3** | **−1.7%** | 基本持平 |

要点解读：

- MSA 层整体基本持平（−1.7%），但内部差异比 dense 层更两极：**vLLM 的稀疏注意力四段（index→topk→sparse decode→merge）合计 25.1 µs，比 ATOM 的四段（33.8 µs）快 26%**；而 ATOM 在通信融合与 O 投影上把差距补了回来。
- vLLM 的 `_decode_topk_fused_kernel` 把 index-topk 做了融合、`_decode_index_score_balanced_kernel` 用了 balanced 调度——这两点是 ATOM 的 `..._topk_partial` + `_topk_index_merge` 拆分结构所没有的，也是 40%+ 差距的来源。
- 两引擎的 MoE 主体（排序 / MoeFlatmm / SiLU）都复用同一套 aiter/ck_tile 算子，属于「同源实现」；ATOM 的 flatmm 略慢（+9~11%）可能来自模板参数或 expert 布局差异。
- KV 精度差异：vLLM MSA 层写 fp8 KV、ATOM 写 bf16，但该 kernel 耗时几乎相同（4.9 vs 5.0），说明 vLLM 的量化路径没有额外成本——vLLM 同时拿到一半的 KV 显存。

## 4. vLLM TODO 清单

按单 step 预期收益排序（收益 = 单层节省 × 层数，60 层）：

| # | 位置（逻辑算子） | 现状（vLLM） | ATOM 做法 | 证据 | 建议 | 预期收益 | 风险 |
|---|---|---|---|---|---|---|---|
| 1 | 通信 + 残差 + RMSNorm（每层 2 处） | `cross_device_reduce_1stage` + `_gemma_fused_add_rmsnorm` 分开发射（每层 4 个 kernel，27.0 µs） | `aiter::allreduce_fusion_kernel_1stage` 把 allreduce+残差+RMSNorm 三合一（每层 2 个 kernel，20.0 µs） | §3 两表通信行：每处省 3.5 µs，每层省 7.0 µs | 在 vLLM ROCm 侧对 MiniMax-M3 启用 AITER AR+RMS 融合：`allreduce_rms_fusion.py` 已有 `AiterAllreduceFusedAddRMSNormWithCopyPattern`（PR #43953 为 GemmaRMSNorm 系启用），需为 MiniMax-M3 amd 路径补齐 pattern 覆盖 | ~7.0 µs × 60 ≈ **420 µs/step（~4.7%）** | ① aiter 融合 kernel 有历史精度坑（累加 f32 + 先加残差再 round，aiter#2586 已修为 bf16 round-trip），接入需跑精度回归；② 1-stage 适用 token 数上限，MiniMax hidden 6144 → 12 KB/次，远低于 128 KB 阈值，1-stage 适用；③ 需验证 fused 模式与 fp8 KV 量化路径共存 |
| 2 | O 投影 GEMM（全部 60 层） | `_gemm_a16_w16` 使用 `BLOCK_SIZE_N_32` tile，11.9 µs | 同族 kernel 用 `BLOCK_SIZE_N_16` tile，8.4 µs | §3 两表 O 投影行：每层省 3.5 µs | 对 O 投影 shape（M=1/小 batch × N=6144×TP 分片）重新 autotune ck-tile GEMM tile 配置，重点验证 N_16/N_32 与 KSPLIT/SPLITK_BLOCK_SIZE 组合 | ~3.5 µs × 60 ≈ **210 µs/step（~2.3%）** | 小 M 场景 tile 选择与 batch 相关，需覆盖 decode 不同 batch 大小验证；收益可能随 batch 增大反转 |
| 3 | MoE Router 投影 | `_rocm_fp32_router_gemm_kernel`（fp32 累加）5.4 µs | `hgemm_bf16_16x64x64x6_SPK12`（bf16）5.2 µs | §3.2 router 行 | 评估 router 投影换 bf16 hgemm（SPK12）的精度影响，若 top-k 结果一致则替换 | ~0.2 µs × 57 ≈ 11 µs/step | router 输出决定 expert 选择，fp32→bf16 可能改变 top-k 边界，必须做端到端质量评估 |
| 4 | Top-k 门控 | `aiter::grouped_topk_kernel` 4.5 µs | `aiter::topk_gating_kernel_opt` 4.2 µs | §3.2 top-k 行 | 检查两者语义差异，若等价则换用 opt 版本 | ~0.3 µs × 57 ≈ 17 µs/step | 需确认 `_opt` 变体对 expert 数/分组参数的支持范围 |

**反向观察（ATOM 可向 vLLM 学习，vLLM 无需跟进）**：

- **稀疏注意力**：vLLM 四段合计 25.1 µs 比 ATOM 33.8 µs 快 26%，尤其 `_decode_index_score_balanced_kernel` 的 balanced 调度与 `_decode_topk_fused_kernel` 的融合是 ATOM 缺失的。此项不进 vLLM TODO。
- **dense 层 KV 布局拷贝**：ATOM 每 dense 层多 3 个 `at::elementwise` 拷贝（12.6 µs/层），vLLM 无此开销。
- 两引擎 step 总耗时：vLLM 9013.8 µs vs ATOM 9183.2 µs，vLLM 略快 1.9%——完成 TODO #1+#2 后（合计 ~7%）差距将进一步拉大。

## 5. 其它观察

- **kernel 发射数**：vLLM 每 step 1098 个 kernel（48 种）vs ATOM 972 个（39 种）。ATOM 靠融合把发射数降了 11%，但由于 §3 中各有胜负，总时间并未拉开。
- **step 尾部（非逐层算子）**：vLLM = 大 GEMM 129.9 µs（SPLITK_BLOCK_SIZE_6144，lm_head 相关）+ `ncclDevKernel_Generic_1` 53.8 µs + Gumbel 采样组 ~30 µs；ATOM = 大 GEMM 164.5 µs + `aiter::allgather_lastdim` 33.2 µs + argmax reduce 63.2 µs。ATOM 的 allgather 算子（33.2 µs）显著快于 vLLM 的 nccl kernel（53.8 µs），但其 argmax（63.2 µs）远慢于 vLLM 的 Gumbel 采样路径——两引擎的采样实现不同（vLLM 用 Gumbel 采样，ATOM 用全 vocab argmax）。尾部合计 vLLM 约轻 45 µs。
- **嵌入层通信**：两引擎在 step 头部各有 1 次 `cross_device_reduce_1stage`（vLLM 121 次 = 120 层内 + 1 嵌入；ATOM 同样 1 次独立 + 120 次融合版）。
- **host 侧**：两引擎 CPU 注解均领先 GPU ~7 ms（正常的主机流水线领先），cpu_op/cuda_runtime 数量差异（vLLM 1478/714 vs ATOM 1062/277）显示 ATOM 的 host 侧调度更精简，但未成为瓶颈。
- **单流执行**：两引擎 decode 均为单 GPU stream 串行（vLLM 仅 2 个杂核在 stream 2），本 trace 对中无 multi-stream 重叠优化。

## 6. 方法学与局限

- **步骤切分**：以 gpu 通道 `execute_context_*` 注解为界取中间窗口（窗口 2）；两 trace 均 `replay_stable=True`（各窗口 kernel 计数一致，CUDA Graph 回放），单步采样具有代表性。
- **层识别**：trace 无层索引，采用 occurrence-index 方法（kernel 名第 k 次出现 ≈ 第 k 层）；dense 行数据 = 3 个 dense 层均值（≈层 0–2），MSA 行数据 = 57 个 MSA 层均值（≈层 3–59），不区分具体层号。首层/末层与 step 尾部（lm_head/采样）已剔除出对比表。
- **单 rank 视角**：trace 只捕获 rank 0（TP4），通信耗时是单 rank 观测值；不同 rank 间的负载不均衡不在分析范围内。
- **配置差异**：MSA 层 KV cache 精度不同（vLLM fp8 vs ATOM bf16），若来自 `kv_cache_dtype` 配置则属不可比项（本报告仅对比计算路径，该差异已在 §1 标注）。
- **数据来源**：本报告所有数字均可回溯到本目录下两份 `_digest.md` / `_slice.json`（由 `scripts/extract_trace.py` 生成，可重复运行验证）。
