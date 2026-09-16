# DeepSeek-V4 gsm8k 精度回退：bisect 范围缩小分析（修正版）

> 日期：2026-09-16
> 环境：vLLM main @ `b0898a4937`，DeepSeek-V4（文本模型），AMD gfx950（MI350/MI355），expert_dtype = fp4，ROCm
> 现象：lm_eval gsm8k 精度跌至 0.00xx

## 一、bisect 范围（2026-09-16 修正）

经用户二分测试修正，问题 commit 位于以下两个 commit 之间：

| 端点 | Commit | 时间 (UTC) | 说明 |
|---|---|---|---|
| 下界 | `2a02f6efe3` — [CI][ROCm][Disagg] Add GLM-5.2-FP8 to MoRIIO model catalog (#53885) | 2026-09-10 04:12 | 时间上较早的 commit |
| 上界 | `eed1f3d0c6` — [Rust Frontend] Support `generation` blocks in HF chat template (#56378) | 2026-09-12 04:56 | 时间上较晚的 commit |

- 区间：`2a02f6efe3..eed1f3d0c6`，共 **115 个 commit**（2026-09-10 12:12 CST ～ 2026-09-12 12:56 CST）。
- 注意：用户口头给出的两个 hash 顺序与时间顺序相反（`2a02f6efe3` 是 `eed1f3d0c6` 的祖先），本报告按 git 时间序处理。

## 二、嫌疑清单（按嫌疑排序）

### 1. `b4da4d17ae` — [ROCm][Bugfix] Fix AITER preshuffled FP8 block-scale kernel (#56433)

**直接改 DSV4 的 FP8 权重加载和前向路径**（fp4 experts 之外的 FP8 层全在这条路径上）。

- `weight_already_preshuffled()` 跳过手工 shuffle 的条件取决于加载时选中的 kernel 是否声明 `preshuffles_weight`——若加载时与运行时 kernel 选择不一致 → 权重布局错配 → 静默垃圾输出。
- 新增实现 `apply_block_scaled_mm`，docstring 自述 *"a row-major one will not raise, it just returns wrong numbers for M > 1"*。
- `_wq_b_act_scale_transpose` 重写 fused norm+quant 的 scale 字节序协商（self.wq_b 与 indexer.wq_b 共享一个 qr_scale）。
- 文件：`scaled_mm/aiter.py`（+56）、`deepseek_v4/amd/model.py`（+17）、`deepseek_v4/amd/rocm.py`（+70）、`mla_attention.py`、`linear.py`。

**注意**：与 #51692（见下）成对出现——#51692 引入该 kernel，#56433 修它。若验证这对，两个一起 revert。

```bash
git revert b4da4d17ae
```

### 2. `7470082f57` — [ROCm][Perf] Add bpreshuffled blockscaled fp8 GEMM (#51692)

**新增** `AiterPreshuffledFp8BlockScaledMMKernel` 并写入 kernel 注册表（`linear/__init__.py` +4）——改变了 DSV4 FP8 GEMM 的 kernel 选择结果，是 #56433 所修 kernel 的引入者。若该 kernel 在您的形状/env 组合下选错或算错，即产生垃圾输出。

- 文件：`scaled_mm/aiter.py`（+128）、`_aiter_ops.py`（+32）、`kernels/linear/__init__.py`（+4）。

```bash
git revert 7470082f57   # 与 #56433 一起 revert 测试
```

### 3. `9163190dda` — [ROCm][Bugfix][Perf] Tune multi-stream shared experts use; wvSplitKrc fixes (#56098)

**命中 DSV4 ROCm 的 MLA wv GEMM**：

- `csrc/rocm/skinny_gemms.cu`（+248）重写 wvSplitKrc（split-K 分片）逻辑；`layers/utils.py`（+69）新增 `wvsplitkrc_dispatch` 启发式，注释明确要求 *"Both must pick the same chunkk or the workspace check here bounds the wrong k_rnd"*——Python 与 CUDA 侧分片选择若不一致 → workspace 越界或错值。
- 同时调整 shared experts 的多流使用（`shared_experts.py` -13、`moe_runner.py` -13）与 `parallel_state.py`（+6）。

```bash
git revert 9163190dda   # 含 C++，需增量编译
```

### 4. `be1cb9834b` — [Kernel] Optional Q-norm in fused DSv4 MLA epilogue; group_size=32 for packed FP8 quant (#56215)

**命中 V4 的 per-token-group FP8 激活量化**（已确认 `deepseek_v4/common/ops/fused_indexer_q.py` 使用 per-token-group quant）：

- `csrc/.../w8a8/fp8/per_token_group_quant.cu`（+151）改为 packed 布局 + group_size=32——该 kernel 在 ROCm 上经 hipify 编译，若 packed 布局与消费端不一致 → 量化值错乱。
- 另改 NVIDIA 的 `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`（+134，ROCm 路径不调用，影响小）。

```bash
git revert be1cb9834b   # 含 C++，需增量编译
```

### 5. `588a813a60` — [ROCm][BugFix] Fix AITER MXFP4 ASM-GEMM crash on unfused shared experts (#55213)

**命中 fp4 experts 路径**：`mxfp4/aiter.py`（+29）新增 ASM-GEMM 的 scale 形状检查与 Triton 回退（`use_asm_gemm` 在加载后按层实例改写）。若回退判定与实际 kernel 调用不一致 → 错值。

```bash
git revert 588a813a60
```

### 6. `73fb19151f` — [Fast Start] Support fp4 (#55465)

**命中 mxfp4 MoE 权重加载流程**：`quantization/mxfp4.py`（+29）把 MoE kernel 构建重构为 `_build_moe_kernel`，并新增 pre-processed 权重检查/提前返回分支。虽然面向 fast-start loader，但重构后的加载流程对普通加载同样生效——若某分支漏建 kernel → MoE 输出垃圾。

```bash
git revert 73fb19151f
```

### 7. `d0dfe587d5` — [Bugfix] Fall back to full decode graphs for noncompiled models (#55095)

**精确命中平台配置**：gfx950 + `DeepseekV4ForCausalLM` + 未显式指定 cudagraph_mode 时强制 `CUDAGraphMode.NONE`（eager），另加 `piecewise_capture_available` 降级逻辑。模式切换会改变运行时代码路径（metadata 形态、capture 状态门控——如 #51794 多流逻辑中的 `is_current_stream_capturing()`）。

- 零成本验证：`--cudagraph-mode full` 显式指定跑一次，无需 revert。

```bash
git revert d0dfe587d5
```

### 8. `1e1060f998` — [Kimi Perf] Group fp8 mla cache insertion (#55356)

共享 `csrc/libtorch_stable/cache_kernels.cu`（+117）模板重构为 `<scalar_t, cache_t, kv_dt>`——DSV4 的 fp8 KV cache 插入走这些 kernel。

```bash
git revert 1e1060f998   # 含 C++，需增量编译
```

### 9. `7e91760650` — [ROCm][Perf] Route large DSV4 sparse prefill to AITER OPUS (#54855)

`rocm_aiter_mla_sparse.py`（+107）新增 OPUS 预填充路由：gfx950 且 **q.shape[0] ≥ 1024** 时走 AITER OPUS kernel。gsm8k 批量较大时可能触发（取决于 chunk 内 query 数）。

```bash
git revert 7e91760650
```

### 10. `9dd969da09` — [Model][ROCm] Enable DeepSeek V4 Vision (#55107)

`amd/rocm.py`（±68）重写 ROCm prefill 的 `_combine_topk_swa_indices_kernel`：无图分支也引入 `gather_start` clamp 新语义、arange 改为 `next_power_of_2(swa_width)`。短 prompt 场景大多等价，但该 kernel 每个 prefill 都执行。

```bash
git revert 9dd969da09
```

### 11. `e77daef89e` — [Model] Support DeepSeek-V4.1-Flash (#56214)

gfx950 上 `rocm_aiter_sparse_attn_indexer` custom op 注册条件扩展（`on_gfx11() or on_gfx950()`）且 `mutates_args` 增加 `candidate_blocks`——mutates_args 标记错误会影响 cudagraph capture / buffer 复用假设。另改共享 `indexer.py`（+17）。

```bash
git revert e77daef89e
```

### 12. `9b959b8657` — [Model] DeepSeek-V4.1-Flash Model Definitions (#56228)

新模型定义，但改共享 `kernels/linear/__init__.py`（+18，kernel 注册表）与 mxfp8 kernel 文件——注册表变化可能影响现有模型的 kernel 选择。

```bash
git revert 9b959b8657
```

### 13. `1cf6555214` — [Bugfix] Pin EPLB and MLA host-to-device transfer buffers (#56138)

`sparse_mla_attention.py`（+16）改 MLA/indexer 权重 H2D buffer 的 pin 语义——异步传输时序类 bug 的典型来源。

```bash
git revert 1cf6555214
```

### 14. `120ec4ebd2` — [5/N][warmup][DSv4] Migrate NVIDIA CuTeDSL attention kernels (#53566)

改共享 warmup 机制（`jit_warmup.py` +23、`jit_warmup_triton_helper.py` ±58）——warmup 问题通常表现为崩溃而非错值，嫌疑较低。

```bash
git revert 120ec4ebd2
```

### 15. 低嫌疑

| Commit | PR | 说明 |
|---|---|---|
| `ae48466cf3` | #48247 | AITER AG/RS，仅 DP 通信；只跑 TP 无 DP 则排除 |
| `c191787a68` | #56446 | YaRN max_model_len 派生；gsm8k 短 prompt 基本无感 |
| `7de70fa7ae` | #56161 | ROCm linear bias `requires_grad=False`，仅影响加载细节 |
| `db723d245a` | #55710 | 空/零 CLI 参数解析，可能影响配置解析 |
| `6ff479e1f7` | #55236 | kv dtype 错误信息改进，纯报错文案 |

## 三、已排除（区间内其余 commit 中值得说明的）

| Commit | PR | 排除原因 |
|---|---|---|
| `2d75e586fc` | #56153 | constexpr 改动只在非 gfx950 分支，对 gfx950 无效 |
| `8a7f98c8c3` | #53280 | MoE align kernel，DSV4 不用 `moe_align_block_size`（已 grep 确认） |
| `46d2b23ac5` | #56503 | `_aiter_ops.py` 纯新增 `mhc_pre_delayed`（V4.1 专用），对 V4 惰性 |
| `9521c60bdc` | #54038 | Kimi-K3 KDA 专属 |
| `86aca66191` | #56159 | Kimi-K3 KDA 专属 |
| `06e57f622c` | #56526 | Kimi-K3 专属 |
| `0c1e89ceb9` | #55426 | Kimi-K3 专属 |
| `127143e27f` | #56429 | Kimi-k3 DCP revert |
| `828f4f19b4` | #55239 | GLM-5.3-Flash MTP 路由 |
| `c3ccc0e957` | #55355 | DSV4 CPU backend 新增（需确认未动共享代码，建议 grep 确认） |
| `8359e15aae` | #53695 | KV connector + AITER unified attn（未启用 KV connector 则惰性） |
| `cc5dd0a857` | #56312 | MRV1 + breakable cudagraph（需 env 显式开启） |
| `980c16c8e4` | #56107 | PCP + spec decode，lm_eval 不启用 |
| `6fe67cbbf3` | #46994 | MTP spec decode under PP，lm_eval 不启用 |
| `83990f5bcc` | #55212 | MRV2 DCP metadata |
| `2e0ee66cab` | #55819 | MRV2 UVA |
| `b28c3e1568` | #54713 | EAGLE spec decode |
| `6ee5bb0a0b` | #54889 | DCP-only kernel |
| `bdea2777ea` | #55579 | CUTLASS workspace（NVIDIA） |
| `7cdd9304ae` | #55531 | mamba DCP |
| `40e6042ec8` | #54574 | nemotron MTP |
| 其余 | — | 纯 multimodal/CPU/XPU/frontend/CI/docs 类 |

## 四、建议验证顺序

1. **零成本**：`--cudagraph-mode full` 显式指定（对应 #55095）；
2. revert `b4da4d17ae` + `7470082f57`（#56433 + #51692 成对，纯 Python）→ 跑 gsm8k；
3. revert `9163190dda`（#56098，含 C++，需增量编译）→ `be1cb9834b`（#56215，含 C++）；
4. revert `588a813a60`（#55213）→ `73fb19151f`（#55465）→ `1e1060f998`（#55356）；
5. 继续按 9→14 顺序逐个排查。

## 五、备注

- 已测试排除：#56562、#56323、#56200、#56160。
- 本区间内 #56433 与 #51692 成对（引入+修复），建议成对 revert。
- 涉及 C++ 的 revert（#56098、#56215、#55356）需按 `docs/contributing/incremental_build.md` 增量编译后再测试。
- 每次 revert 只影响本地分支，验证后可 `git revert <revert-commit>` 撤销。
