# DeepSeek-V4 gsm8k 精度回退排查报告

> 日期：2026-09-16
> 环境：vLLM main @ `b0898a4937`，DeepSeek-V4（文本模型），AMD gfx950（MI350/MI355），expert_dtype = fp4，ROCm
> 现象：lm_eval gsm8k 精度从正常值跌至 0.00xx（≈ 随机输出）
> 排查窗口：2026-09-11 18:00 CST ～ 2026-09-16 10:02 CST，共 **245** 个 commit

---

## 一、高度可疑 PR（按嫌疑排序）

### 1. `a6c5d6d0fc` — [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4 (#51794)

**嫌疑点：**

- 原代码中 ROCm 因 hang 问题**显式禁用** aux streams（`None if current_platform.is_rocm() else [torch.cuda.Stream() ...]`），该 PR 改为 ROCm 也无条件创建 3 条 aux stream，在 CSA 层的 Q 投影 / indexer / compressor 之间做多流 overlap。
- 新代码 `vllm/models/deepseek_v4/amd/rocm.py` 的 docstring 自述：*"Forking side streams there would rely on runtime HIP event sync, which is unreliable in this overlap on ROCm (event waits can hang)"* —— 这种 event 同步不可靠的场景正是产生竞态、拿到错序/半更新 tensor 的温床，表现就是输出全错。
- **没有环境变量开关**（已 grep 确认无 gate），只能 revert 或本地 patch 测试。

**改动文件：**

- `vllm/models/deepseek_v4/amd/model.py`（+11/-?）
- `vllm/models/deepseek_v4/amd/rocm.py`（+194）
- `vllm/models/deepseek_v4/attention.py`（±31）
- `vllm/models/deepseek_v4/cpu/cpu_sparse.py`

**Revert 命令：**

```bash
git revert a6c5d6d0fc
```

> 关联：`bbbd0a02c9`（#56382，dedicated stream 携带排队工作的 bugfix）与 #51794 在同一块多流逻辑上，排查时可一并关注。

---

### 2. `dffbb714e4` — [ROCm][Perf] Optimize DSV4.1 K=512 decode top-k on gfx950 (#56743)

**嫌疑点：**

- 动了两处：共享的 `csrc/libtorch_stable/sampler.cu`（+263 行，新增 `useTopK512Optimization` 模板路径，引入 `-INFINITY` sentinel 与新 finalize 逻辑）和 `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` 的 kernel 选择 gate。
- 该 gate 在 **gfx950 decode、`topk_tokens == 512`、rows ≤ 384** 时把 AITER top-k 换成新写的 native 路径——正好命中本环境硬件。**若 checkpoint 的 `index_topk == 512`，此 PR 落在每个 decode step 的候选选择关键路径上**，错一个就全错。
- 即使 `index_topk == 2048`，sampler.cu 的大改也值得怀疑（`kNumFinalItems`、finalize 逻辑均有重构）。

**改动文件：**

- `csrc/libtorch_stable/sampler.cu`（+263/-?）
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`（+12）
- `tests/kernels/test_top_k_per_row.py`

**Revert 命令：**

```bash
git revert dffbb714e4
```

> 注意：此 PR 含 C++ 改动，revert 后需按 `docs/contributing/incremental_build.md` 增量编译再测试。

---

### 3. `13e221f830` — [Perf] Fuse DSV4.1 input metadata preparation with Triton (#56562)

**嫌疑点：**

- 虽然 tag 是 DSV4.1，但改的是**共享路径**：
  - `vllm/v1/attention/backend.py` 的 `CommonAttentionMetadata.get_token_to_req_indices` 从 `torch.repeat_interleave` 换成全新 Triton kernel（`_token_request_mapping_kernel`）。token→request 映射若有一处 off-by-one / 边界搜索错误，所有 attention 后端的元数据全错 → 输出 garbage。
  - `vllm/v1/attention/backends/mla/indexer.py` 的 `DeepseekV32IndexerMetadataBuilder` 重构 121 行（V4 的 indexer 同用此 builder）。

**改动文件：**

- `vllm/v1/attention/backend.py`（±24）
- `vllm/v1/attention/backends/mla/indexer.py`（±121）
- `vllm/v1/attention/ops/metadata.py`（+87，新增 Triton kernel）

**Revert 命令：**

```bash
git revert 13e221f830
```

---

### 4. `aed894c190` — [6/N][warmup][DSv4] Migrate sampling, and DFlash JIT kernels (#56323)

**嫌疑点：**

- 重写 `vllm/v1/sample/ops/topk_topp_triton.py`（+333 行）；给 `GPUModelRunner.__init__` 加 `@JitWarmupRegistry.capture`；重构 sampler 初始化（`vllm/v1/worker/gpu/model_runner.py` 97 行改动）。
- gfx950 上采样默认走 AITER，Triton 采样 kernel 本身未必命中，但 warmup 注册/捕获机制改动影响启动与 CUDA graph capture 行为。
- 同系列隔天即出现 `2671fedfc7` *"Revert explicit Triton JIT warmup migration"*（#56654），说明该 warmup 迁移线不稳定。
- 依赖同系列 #53566 / #50178 等 commit，revert 可能出现冲突，需人工解决。

**改动文件（节选）：**

- `vllm/v1/sample/ops/topk_topp_triton.py`、`topk_topp_sampler.py`
- `vllm/v1/worker/gpu/model_runner.py`、`gpu_model_runner.py`
- `vllm/model_executor/warmup/jit_warmup.py`、`jit_warmup_triton_helper.py`

**Revert 命令：**

```bash
git revert aed894c190
# 若冲突，先 git revert --abort，再人工解决
```

---

### 5. `dc07f1638f` — [Bugfix][MLA] Read sparse model settings from text config (#56160)

**嫌疑点：**

- 把 sparse MLA 的 `topk_tokens` 读取源从 `hf_config.index_topk` 换成 `hf_text_config.index_topk`（`sparse_mla_attention.py`、`rocm_aiter_mla_sparse.py` 等 11 处）。
- 已确认**模型侧仍然读 `config.index_topk`**（`vllm/models/deepseek_v4/amd/model.py:966`、`attention.py:930`）。若 checkpoint 带 `text_config` 且其中 `index_topk` 与外层不一致（或缺失），metadata builder 与模型层将使用**不同的 topk** —— 候选数对不上，注意力直接错乱。这种静默数值不一致正是 0.00xx 精度的典型来源。

**Revert 命令：**

```bash
git revert dc07f1638f
```

---

## 二、中等可疑

| Commit | PR | 原因 |
|---|---|---|
| `24bfbd5d4d` | #56200 | deepseek_v4 parser 加 `wait_for_reasoning=thinking`，reasoning 解析重构可能截断/吞掉答案，lm_eval extract 后直接 0 分 |
| `c6fa1f05d1` | #56346 | `libtorch_stable/persistent_topk.cuh` + 新增 `sampled_topk.cuh`，ROCm 经 hipify 编译，DSA 候选 top-k 相关 |
| `ef5f7cd119` | #56560 | tag 是 DSV4.1，但改的是共享 `mxfp8/rocm_native.py` dequant 逻辑，若 V4 有 mxfp8 层则相关 |
| `71888f507a` | #56688 | FP8 dummy init 数值路径变更，影响 warmup/CUDA graph capture 初值 |
| `d4ee7fe7a9` | #51204 | quant 级 backend 选择重构，默认无行为变化（opt-in），但动了 `mxfp4/aiter.py`；如设置过 `linear_backend` 相关配置需留意 |

---

## 三、已排查排除

| Commit | PR | 排除原因 |
|---|---|---|
| `c8d1cf077a` | #56176 | GLM-5.3-Flash MXFP4，只改 glm5next 目录，不碰 DSV4 |
| `a4d2d9d95e` | #53940 | Kimi-K3 a4w4 flydsl，Kimi 专属 |
| `e6eb0d120c` | #56893 | MXFP8 KV 存储，V4.1 专属（deepseek_v41） |
| `c711f740b7` | #56349 | breakable CUDA graphs 仅对 DeepseekV41ForCausalLM 自动启用 |
| `46d2b23ac5` | #56503 | AITER mHC，V4.1 专属 |
| `00972dfd72` | #56513 | fold mHC post，V4.1 专属 |
| `5372e72a98` | #56633 | fold mHC post（NVIDIA），V4.1 专属 |
| `dabc4362b4` | #56628 | DSA decode candidate mask，V4.1 专属 |
| `dc2e8f1157` | #54965 | W4A16 zero-point，本环境是 fp4 experts，不适用 |
| `9dd969da09` | #55107 | V4 Vision，文本路径受 `vision_n_layers > 0` 门控，纯文本无行为变化 |
| `2d75e586fc` | #56153 | indexer gather kernel 去 constexpr，语义等价 |
| `03dc26e639` | #53793 | ReLU2+FP8 fusion，DSV4 不用 fused act-quant（已 grep 确认） |
| `8a7f98c8c3` | #53280 | MoE align cooperative writes，CUDA-only kernel |
| `7ee8a6dd01` | #56122 | watermarking，默认关闭 |
| spec decode / MRV2 系列 | — | lm_eval 默认不启用 |

---

## 四、具体排查步骤

### Step 0：准备

```bash
# 1. 建调试分支，保持 main 干净
git checkout -b dsv4-acc-debug

# 2. 记录基线（复现问题）
# 用与上周五完全相同的 eval 命令跑一次，确认 0.00xx 可复现，例如：
.venv/bin/python -m lm_eval --model vllm \
    --model_args pretrained=<模型路径>,tensor_parallel_size=...,max_model_len=...,gpu_memory_utilization=... \
    --tasks gsm8k --batch_size auto
# 记录输出，作为对照基准
```

> 快速迭代可用 `--limit 100` 先筛，确认嫌疑后跑全量 gsm8k 复核。

### Step 1：零成本检查（对应 #56160）

不需要 revert，先检查 checkpoint 配置一致性：

```bash
python - <<'EOF'
import json
cfg = json.load(open("<模型路径>/config.json"))
tc = cfg.get("text_config") or {}
print("outer index_topk:", cfg.get("index_topk"))
print("text_config.index_topk:", tc.get("index_topk"))
EOF
```

- 若两者**不一致或 text_config 缺失该字段** → 直接先做 Step 5（revert `dc07f1638f`）。
- 若一致 → 按 Step 2 顺序继续。

### Step 2：revert `a6c5d6d0fc`（#51794 CSA multi-stream）

```bash
git revert a6c5d6d0fc
```

- 纯 Python 改动，editable 安装下重启服务即可生效，无需重编译。
- 跑 gsm8k。**若精度恢复 → 结案**（问题在自己 PR 的多流 overlap；无 env 开关，需修复后重提）。
- 若仍坏 → 保留此 revert，继续 Step 3。

### Step 3：revert `dffbb714e4`（#56743 gfx950 top-k）

```bash
git revert dffbb714e4
```

- **含 C++ 改动（sampler.cu）**，需增量编译：
  ```bash
  # 按 docs/contributing/incremental_build.md 配置后
  cmake --build build --target vllm -j
  ```
- 跑 gsm8k。恢复 → 结案；仍坏 → 继续。

### Step 4：revert `13e221f830`（#56562 input metadata Triton fusion）

```bash
git revert 13e221f830
```

- 纯 Python（Triton kernel），重启服务即可。
- 跑 gsm8k。恢复 → 结案；仍坏 → 继续。

### Step 5：revert `dc07f1638f`（#56160 index_topk 读取源）

```bash
git revert dc07f1638f
```

- 纯 Python，重启服务即可。
- 跑 gsm8k。恢复 → 结案；仍坏 → 继续。

### Step 6：revert `aed894c190`（#56323 sampling warmup 迁移）

```bash
git revert aed894c190
# 若冲突：git revert --abort 后人工解决（该 PR 依赖同系列 #53566/#50178 等）
```

- 纯 Python，重启服务即可。
- 跑 gsm8k。恢复 → 结案。

### Step 7：中等可疑逐个排查

若以上 5 个都排除，按此顺序逐个 revert + 测试：

```bash
git revert 24bfbd5d4d   # #56200 reasoning parser
git revert c6fa1f05d1   # #56346 persistent top-k（含 C++，需重编译）
git revert ef5f7cd119   # #56560 mxfp8 dequant
git revert 71888f507a   # #56688 FP8 dummy init
git revert d4ee7fe7a9   # #51204 quant backend 选择
```

### Step 8：清理

```bash
# 保留结论（确定罪魁）后，撤销其余无关 revert：
git log --oneline | head          # 找到各 "Revert ..." commit 的 hash
git revert <无关-revert-commit>   # revert 的 revert，逐个撤销

# 调试分支确认无误后合回 main 或直接删除：
git checkout main
git branch -D dsv4-acc-debug      # 确认不需要后再删
```

---

## 五、备注

- 不建议直接 `git bisect`：245 个 commit 且每次要跑完整 gsm8k，人工锁定 5 个嫌疑后逐个 revert 验证成本更低。
- 若 revert 中途出现冲突且无法简单解决，`git revert --abort` 可安全回到上一步状态。
- 所有 revert 只影响本地调试分支，不会污染 main。
