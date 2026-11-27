# Trace 对比分析指南

本指南是执行 vLLM vs ATOM decode trace 对比分析时的领域知识。分析前必须完整阅读本文件。
所有数字一律取自 digest / slice 产物，禁止编造或估算。

## 目录

1. Trace 数据速览
2. Decode 步骤切分
3. Layer 识别（occurrence-index 方法）
4. 逻辑算子分组方法
5. 多流检测
6. 对比表与 TODO 规则
7. Pitfalls
8. 报告模板
9. MiniMax-M3 worked example

---

## 1. Trace 数据速览

- **单位是 float 微秒（µs）**：`ph:"X"` 事件的 `ts`/`dur` 是 µs（torch profiler 的 Chrome trace 导出格式）。
  绝对时间 = `baseTimeNanoseconds + ts * 1000`。分析时一律用相对时间（窗口起点归零）。
  校验方法：digest §3 的 `busy/wall` 应在 ~90-100%（decode 几乎满负载）；若远低于 50% 或超过 150%，单位假设可疑。
- **引擎识别**：顶层有 `vllm_version` → vllm；有 `roctracer_version`（无 `vllm_version`）→ atom。
- **事件类别**（`cat`）：
  - `kernel`：GPU kernel（HIP），带 `args.stream`/`device`/`correlation`，atom 的还有 `grid`/`block`。分析主体。
  - `cpu_op`：CPU 侧算子（aten::*、forward 函数等），`ts`/`dur` 是 CPU 墙钟，与 GPU 时间轴有 ~7ms 领先偏移，**不能**直接和 kernel 时间比较，只在分析 host 启动开销时参考。
  - `cuda_runtime`：HIP runtime API 调用。
  - `user_annotation` / `gpu_user_annotation`：profiler 注解，用于切分 decode step。
  - `ac2g`（ph "f"/"s"）：activity→GPU 关联流事件，可忽略。
- **rank 范围**：每份 trace 通常只捕获 1 个 rank 的 GPU 事件（kernel 全在一个 pid 下，`args.device` == 该 pid）。
  `deviceProperties` 列出集群全部 GPU（如 4×MI300X 每卡 8 个 GCD，sm9.5×256），`distributedInfo` 给出 world_size/backend/pg 配置。对比结论基于单 rank 视角，通信算子的耗时同理。
- **stream 判定**：`args.stream` 可靠；缺失时回退 `tid`。digest §3 的 per-stream 表直接给出多流情况。

## 2. Decode 步骤切分

- 每轮 decode 由 `gpu_user_annotation` 通道中名为 `execute_context_*(*)_generation_*(*)` 的注解界定
  （提取脚本自动完成，见 digest §2）。
- **为什么不用 CPU 通道**：host 领先 GPU 数个 ms（实测 ~6.9ms），CPU 注解的时间戳不对齐 GPU 执行窗口，
  用它切分会把相邻 step 的 kernel 划错窗口。只有 gpu 通道无注解时才用 `--bracket cpu-annot` 近似模式
  （digest 会醒目警告 "STEP ATTRIBUTION IS APPROXIMATE"）。
- 注解名过滤：只匹配 `^execute_context_`。vllm trace 里 `nccl:_all_gather_base` 注解与 execute_context 交错出现，
  混入会得到错误窗口数。extract 脚本已过滤；若未来新版本出现其它注解名，digest §2 会列出全部注解名，注意甄别。
- N 个注解 → N-1 个完整窗口 + 1 个尾窗口（end 按中位步长外推）。分析默认用完整窗口。
- **replay_stable**：各窗口 kernel 名计数一致（= CUDA Graph 回放）。为 false 时单步采样代表性差，
  报告 §6 必须写明该局限。

## 3. Layer 识别（occurrence-index 方法）

trace 中 kernel 名/args 里**没有 layer 索引**。识别方法：

1. **计数族**：decode 是逐层顺序执行的，每类层内相同的 kernel 每 step 重复执行一次（或两次）。
   统计 chosen step 内每个 kernel 名的执行次数 C（digest §4）：
   - C = 层数（如 57）：该 kernel 每层执行一次 → 属于"每层型"算子（如每层一个 attention）；
   - C = 2×层数（如 120）：每层执行两次（典型：MoE 每层两次通信，或 fused_add_rmsnorm 每层两个）；
   - C = 某种层型的层数（如 3）：该 kernel 只在该层型内出现（如 dense 层专属的 hgemm 变体）；
   - C 很小且与层数无关：模型全局操作（embedding/sampler/调度杂核）。
2. **occurrence index → layer index**：同一 kernel 名的第 k 次执行（按 ts 排序，窗口内）对应第 k 层。
   extract 脚本默认取第 C//2 次（中间层），digest §5 的 `occ` 列即该索引；`dur_median_across_steps_us`
   是同一 occurrence 跨窗口的时长中位数，用于抑制回放抖动。
3. **层型判别**：
   - 计数族本身：不同层型的专属 kernel 计数不同（如 57 vs 3）；
   - kernel 模板变体：同一 kernel 名的不同模板实例化区分层型
     （如 vLLM 的 `fusedMiniMaxM3QNormRopeKVInsertKernel<..., unsigned char, Fp8KVCacheDataType 1, ...>`
     是 MSA 层（fp8 KV），`<..., __hip_bfloat16, Fp8KVCacheDataType 0, ...>` 是 dense 层（bf16 KV））；
   - **ts 位置**：digest §4/§5 按 ts_rel 排序，dense 层块通常位于 step 头部（层 0..2），
     其专属 kernel 的 first_rel_us 集中在窗口前几百 µs；
   - HF config 校验：用模型名查 HuggingFace config（`num_hidden_layers`、dense/sparse 层分布），
     与计数族互相印证。离线时直接用计数结构推断。
4. 层号只需"同类型中间层"粒度（如"MSA 层 #28"），不必绝对精确；报告措辞用"≈ 第 N 层"。

## 4. 逻辑算子分组方法

对比的最小单位是**逻辑算子**（logical operation），不是单个 kernel——vllm 和 atom 对同一逻辑算子的
kernel 拆分/融合方式不同，kernel 名永远对不上。按每层 forward 的标准算子序列归组：

**标准 decode 层 forward 序列**（适用于 GQA/稀疏 attention + MoE/dense 混合模型）：

```
RMSNorm(输入) → QKV 投影 → QK-Norm/RoPE → KV cache 写入 → 稀疏 attention decode
→ attention 输出归并 → MoE router gemm → top-k 选择 → expert 排序 → grouped GEMM(up+gate)
→ 激活(SiLU) → grouped GEMM(down) → fused allreduce(残差相加)  → [下一层]
```

**归组与命名规则**：

- 把每个 kernel 归到上述某个逻辑算子。可以按 kernel 名字典（digest §7 全名）在引擎源码里 grep 确认语义
  （vllm 仓库 `vllm/_custom_ops.py`/`csrc/`，ATOM 仓库 `atoms/`、aiter 的 `csrc/`）。
- 同一逻辑算子行的两引擎实现：
  - **split vs fused**：vLLM 用 4 个独立 kernel 完成 sparse attention（index score → topk → gqa decode → merge），
    ATOM 可能用 2-3 个融合度更高的 kernel；
  - **不同算子**：同一操作 vLLM 用 kernel A、ATOM 用 kernel B（如 router gemm：
    vLLM `_rocm_fp32_router_gemm_kernel` vs ATOM `hgemm_bf16_16x64x64x6_SPK12`）；
  - **融合通信**：ATOM `allreduce_fusion_kernel_1stage` 把 allreduce 与 add 残差融合；
    vLLM 可能拆成 allreduce + 单独 add，或调用次数不同（121 vs 120）；
  - **量化差异**：fp8 KV cache vs bf16 KV cache（看 kernel 模板参数里的 `unsigned char`/`Fp8KVCacheDataType`）。
- 每格写 kernel 名（可截断）+ `C×mean(µs)`。同一行 vLLM 侧多个 kernel 用 `<br>` 分隔并写合计。
- "合计"口径 = 该逻辑算子在**该层型内单层一次执行**的 GPU 时间 = Σ(count_i × mean_i) 按层归一，
  即对 C=每层型层数的 kernel 用 mean（已是单层单次），对 C=2×层数的除以 2。写清口径。

## 5. 多流检测

- 看 digest §3 的 per-stream 表：单流 → 一个 stream 占 ~100% busy；多流 → 多个 stream 各有显著 busy。
- 多流并行判定：slice 的 `middle_slice` 里带每 kernel 的 ts_rel/dur/stream，
  若 A stream 的 kernel 区间与 B stream 的 kernel 区间在时间上重叠，说明并行执行。
- 措辞规则：单流 trace（如 ATOM 全 stream 3）明确写"单流串行"；vLLM 主流 + 少量小核（如 2 个 kernel 在 stream 2）
  不是业务多流，写"单流 + 个别杂核"。
- 结论"某引擎用了多流重叠"必须给出重叠区间的证据（哪两个 kernel、各自 ts 区间）。

## 6. 对比表与 TODO 规则

- **差异维度标签**（"实现差异"列固定用以下词汇）：`split vs fused` / `不同算子` / `融合通信` /
  `多流重叠` / `量化 KV(f8 vs bf16)` / `激活融合` / `结构等价`。
- **快慢判定**：用单层单次的合计 µs 比较；差异 <5% 写"持平"，5-20% 写"略快/略慢"，>20% 写"显著"。
  只对可复现的差异下结论（dur_median_across_steps_us 跨步稳定、两引擎窗口可比）。
- **数字来源**：digest §4（sum/mean/C）与 §5（单次 dur、跨步中位数）。报告 §6 注明 digest/slice 文件名。
- **TODO 清单**（写给 vLLM，报告 §4）每条包含：
  | # | 位置(逻辑算子) | 现状(vLLM) | ATOM 做法 | 证据(表内数字) | 建议 | 预期收益 | 风险 |
  按收益排序。建议可指名 aiter 算子（如 `aiter::allreduce_fusion_kernel_1stage`、
  `paged_attention_decode_sliding_window_head_1`、`fusedQKNormIdxrQKNormKernel`）或融合策略。
  预期收益 = 该算子在单 step 的合计 µs 差值 × 60 层之类的量级换算，写清楚换算口径。
  风险列考虑：精度影响（fp8 vs bf16）、aiter 依赖、vllm 与 atom 的 batch/调度差异导致的适用性限制。
- 若 atom 反而更慢（存在这种行），如实标注"vLLM 更快"，不强行写 TODO。

## 7. Pitfalls

- **首尾层特殊**：窗口 0 可能含 warmup kernel（digest 会警告），尾窗口是外推的；默认中间窗口已避开。
- **C==1 的杂核不是业务算子**：`at::native::*`（fill/copy/arange 等）、sampler 尾部（`_gumbel_sample_kernel`、
  `_post_update_kernel` 等）属于调度/采样开销，可放"其它观察"，不放进逐层对比表。
- **大 GEMM 尾核**：step 尾部的大 gemm（如 SPLITK_BLOCK_SIZE_6144 的 `_gemm_a16_w16_kernel`）+ nccl/allgather
  是 logits/lm_head 相关跨 rank 操作，不是逐层算子；两引擎都有时可以作为独立一节对比（"step 尾部"）。
- **通信次数差异**：121 vs 120 这类差异要查清再下结论（多出的那次可能是 lm_head 或首层残差），
  可从 first_rel_us/last_rel_us 定位发生位置。
- **aiter 版本差异**：两引擎链接的 aiter 版本可能不同，kernel 名模板参数会有出入；只要语义相同就归同一行。
- **不跨模型下结论**：两 trace 必须同模型同配置（SKILL.md 第 1 步已与用户确认），报告 §1 必须显式写出该前提。
- **replay_stable=false**：单步代表性差，报告 §6 说明，且优先用 busy 中位窗口（脚本已自动回退）。
- **占用率异常**：busy/wall 不在 90-100% 附近时，先怀疑注解切分或单位问题，再分析。

## 8. 报告模板

标题：`# vLLM vs ATOM Decode Trace 对比 — <模型名>`（英文版译英；`both` 时生成两份文件）

```
## 1. 基本信息
两张表（vllm / atom 各一）：引擎版本、GPU（数量/型号/每卡 GCD）、world_size/rank/backend、
trace 文件、事件总数、step 数、所选 step 窗口(ms)、窗口内 kernel 数/去重数、busy/wall、streams、replay_stable。
附"采集一致性说明"：两引擎同模型同配置（用户确认）；列出任何不一致（版本/GPU 等）。

## 2. 模型结构简介
- 层数与层型分布（来源：HF config + 计数族印证，写清来源）
- 每类层的 forward 流程（文本步骤列表 + 关键 kernel 名）

## 3. 逐层类型对比
每类层一节，一节一张表：
| 逻辑算子 | vLLM kernel(s) | ATOM kernel(s) | 实现差异 | vLLM 单层(µs) | ATOM 单层(µs) | 差异% | 结论 |
表后 3-6 行要点解读（不是逐行复述）。

## 4. vLLM TODO 清单
| # | 位置(逻辑算子) | 现状(vLLM) | ATOM 做法 | 证据 | 建议 | 预期收益 | 风险 |
按收益排序。

## 5. 其它观察
step 时间线差异、host 侧开销、step 尾部（lm_head/采样）、噪音核等。

## 6. 方法学与局限
切步方法、occurrence-index 假设、层号近似、单步采样、单 rank 视角；
注明 digest/slice 文件名（数据审计线索）。
```

## 9. MiniMax-M3 worked example

60 层：层 0-2 为 dense GQA+MLP，层 3-59 为 MSA(MiniMax Sparse Attention)+MoE。
样例 trace：tp4、8k prompt 1k output、concurrency 8，MI300X（每卡 8 GCD，sm9.5×256），
vLLM 0.28.1rc1.dev199+g7c5dc571c vs ATOM（roctracer）。单 rank 捕获；6 个 decode step；
vLLM chosen step 1098 kernel / 48 种，ATOM 972 / 39；均 replay_stable；busy/wall ~99%。

**计数族含义（chosen step）**：

| C | 含义 | vLLM 例子 | ATOM 例子 |
|---|---|---|---|
| 57 | 每个 MSA+MoE 层一次 | `_gqa_sparse_decode_kernel.kd` 等 5 个 attention 核 | `paged_attention_decode_sliding_window_head_1` 等 |
| 3 | 每个 dense 层一次 | `kernel_unified_attention.kd` | 同名 |
| 60 | 全部 60 层一次 | `_gemm_a16_w16_...` (down proj) | 同名（模板参数不同） |
| 120 | 每层两次 | `_gemma_fused_add_rmsnorm_kernel.kd` | `aiter::allreduce_fusion_kernel_1stage` |
| 6 | dense 层每层两次 | `hgemm_bf16_16x64x128x5_SPK2` (dense gate/up) | 同名 |
| 121 | 每层两次+1 次额外 | `aiter::cross_device_reduce_1stage`（vLLM） | — |

**dense 层（≈层 1）单层单次流程**（first_rel_us 集中在窗口前 ~100-450µs）：
RMSNorm(`_gemma_rmsnorm_kernel` 1×，或 fused_add_rmsnorm 的第一次) → QKV 投影
(`hgemm_bf16_16x64x64x8_SPK6`) → QK-Norm+RoPE+KV 写
(vLLM `fusedMiniMaxM3QNormRopeKVInsertKernel<..., bf16 KV>` 4.6µs / ATOM `fusedQKNormIdxrQKNormKernel` 4.3µs)
→ `reshape_and_cache_kernel_flash` → dense attention `kernel_unified_attention` (~27µs) → `reduce_segments`
→ gate/up 投影 (`hgemm_bf16_16x64x128x5_SPK2`，每层 2 次) → SiLU 激活
(vLLM `_swiglu_oai_kernel` / ATOM 同名) → down 投影 (`_gemm_a16_w16_...` 60×之一) → 通信/残差。

**MSA+MoE 层（≈层 28）单层单次流程**：
QKV 投影 (`hgemm_bf16_16x64x64x7_SPK6`) → QK-Norm+RoPE+KV 写
(vLLM `fusedMiniMaxM3QNormRopeKVInsertKernel<..., fp8 KV>` / ATOM `fusedQKNormIdxrQKNormKernel`，注意 KV 格式差异)
→ sparse attention 四段（vLLM）：`_decode_index_score_balanced_kernel`(7.7) → `_decode_topk_fused_kernel`(4.7)
→ `_gqa_sparse_decode_kernel`(7.9) → `_merge_topk_attn_out_kernel`(4.8)；
ATOM 四段：`_decode_index_score_topk_partial_kernel`(11.1) → `_topk_index_merge_kernel`(6.6)
→ `paged_attention_decode_sliding_window_head_1`(11.6) → `pa_decode_ps_reduce_hip_kernel`(4.5)
→ MoE router（vLLM `_rocm_fp32_router_gemm_kernel` 5.4 / ATOM `hgemm_bf16_16x64x64x6_SPK12` 5.2）
→ top-k 门控（两者均 `aiter::grouped_topk_kernel` 系 / ATOM 为 `aiter::topk_gating_kernel_opt`）
→ expert 排序（均 `aiter::opus_moe_sorting_entry`）
→ grouped GEMM up+gate（均 `ck_tile MoeFlatmm`，~26-29µs，两引擎同名不同模板）
→ SiLU（均 `aiter::swiglu_act_and_mul_kernel`）→ grouped GEMM down（均 `ck_tile MoeFlatmm` 第二变体，~15µs）
→ 通信（vLLM `cross_device_reduce_1stage` 121×/step vs ATOM `allreduce_fusion_kernel_1stage` 120×/step）。

**通信行对比口径**：vLLM 121 次 × 8.7µs = 1053µs/step；ATOM 120 次 × 10.0µs = 1196µs/step。
单层口径 vLLM ≈ 2×8.7=17.4µs、ATOM ≈ 2×10.0=20.0µs（每层 2 次），但 ATOM 融合了残差相加而 vLLM 需另计 add。
结论须写清"每层 2 次"与融合差异。

**step 尾部**（非逐层）：vLLM 有 129.9µs 大 GEMM（SPLITK_BLOCK_SIZE_6144）+ `ncclDevKernel_Generic_1` 53.8µs
+ sampler 组（gumbel/argmax/post_update ~30µs）；ATOM 有 164.5µs 大 GEMM + `aiter::allgather_lastdim` 33.2µs
+ 63.2µs reduce + 采样组。可放"其它观察"对比。
