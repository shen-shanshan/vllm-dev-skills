# vLLM ROCm PR Review 规则目录

> 来源：从 ATOM（ROCm/ATOM）与 aiter（ROCm/aiter）的 `review-pr` skill 提炼，并适配 vLLM ROCm 代码面。
> 上游原始文档快照：`references/upstream/`（由 `scripts/update_skill.py` 同步，SHA 记录在 `.update-metadata.json`）。
> 提炼日期：2026-08-20。规则编码（A/B/C/D/E/F/G/P/HK）是内部标签，**绝不出现在报告中**。

**何时读全文**：PR 触及 Tier-1 backbone 文件、mori 相关、或带性能声明 → 全文。快速 review → 只读「核心纪律」+ SKILL.md Step 4 映射到的分类。

---

## 1. 核心纪律

1. **跨文件验证（报告任何 kernel/dispatch 发现之前）**：diff 只显示改动行，不显示全貌。声称"没有同步"、"缺少分支"、"dispatch 错误"前，必须读完整函数及其调用者/兄弟变体（同目录 `_opt`/`_prefill`/`_decode`/`_v2` 后缀文件、相关 header）。只看 hunk 就下结论是误报的主要来源。
2. **CI 失败分类（把每个红叉归因后再怪 PR）**：读失败的 step 名；外部 docs/readthedocs 构建失败与代码无关；main 在同一窗口挂同一 job = 基线/flaky；旧 run 的日志过期（HTTP 410）= 对今日 main 无意义，要求 rebase + 重跑。**ROCm CI 被跳过且 PR 触及 ROCm 路径 = E4**。
3. **"编译通过 ≠ 在 MI300/MI355 上跑对"**：vLLM 主 CI 以 CUDA 为主，ROCm 路径经常只在特定 label/hardware 队列上跑。任何 ROCm 专属分支若未被 AMD hardware CI 覆盖，静默数值错误可以合入 main。
4. **🔴 gate**：任何"必须修复"级别的发现必须写出**具体触发输入**（shape / dtype / arch / 配置组合）。写不出 → 降级为 ⚠️（"值得检查"）或放弃。这是防止误报落在 maintainer PR 上的唯一手段。
5. **数字溯源**：PR 里最亮眼的那个性能数字，必须能追溯到来源（脚本输出 / 日志 / 配置文件）。追溯不到 = `[unverified]`，报告中标注并请作者给出处，**绝不当作事实转述**。

## 2. vLLM ROCm 表面地图

### 2.1 关键环境变量（`vllm/envs.py`）

| 变量 | 作用 |
|------|------|
| `VLLM_ROCM_USE_AITER` | 总开关：启用 aiter 算子（MLA、MoE、量化等） |
| `VLLM_ROCM_USE_AITER_MOE` | 启用 aiter MoE 算子 |
| `VLLM_ROCM_USE_AITER_MLA` | 启用 aiter MLA attention |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | 启用 aiter unified attention |
| `VLLM_ROCM_USE_AITER_CUSTOM_PAGED_ATTN` | 启用 aiter custom paged attention |
| `VLLM_ROCM_USE_AITER_RMSNORM` | aiter RMSNorm |
| `VLLM_ROCM_USE_AITER_ROPE` | aiter RoPE |
| `VLLM_ROCM_USE_AITER_GQA` / `_FP8` / `_BLOCKV` | GQA 相关开关 |
| `VLLM_ROCM_USE_AITER_FP8_BLOCK_MOE` | FP8 block MoE |

> 注：以上为常见集合的快照，review 时以 `vllm/envs.py` 当前定义为准（staleness guard）。新增 env var 必须注册到 `envs.py` 并在 PR 描述中说明（HK-ENV）。

### 2.2 依赖与版本 pin

- `requirements/rocm.txt`：aiter 版本 pin（`aiter==x.y.z` 或 git ref）。改动 aiter 版本 = 可能引入/移除算子，须对照 aiter 侧 PR（E1）。
- `requirements/kv_connectors_rocm.txt`：mori（`amd_mori`）pin。mori 从源码构建是推荐路径；RDMA 用户态库与内核/firmware 版本不匹配（`availDevices.size() > 0` 断言）是常见环境坑。
- `docker/Dockerfile.rocm*`：ROCm 基础镜像版本（如 7.0.x）；升级 ROCm 版本需重跑完整 AMD CI。

### 2.3 Backbone 文件分级（Tier 表）

**Tier 1 — 静默数值错误可能（触及必深查）**

| 文件 | 影响面 | 常见失败模式 |
|------|--------|-------------|
| `vllm/v1/attention/backends/rocm_attn.py` | 所有 ROCm attention 路由 | 错误 backend 选择、元数据构建错误 |
| `vllm/v1/attention/backends/rocm_aiter_fa.py` | aiter flash attention 路径 | 错误 kernel、mask 丢失 |
| `vllm/v1/attention/backends/rocm_aiter_unified_attn.py` | unified attention 路径 | dispatch 条件错误 |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | MLA decode/prefill（DeepSeek V3/V3.2/V4） | 错误 KV 布局、dtype |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | 稀疏 MLA（V4/Kimi） | 索引布局错误 |
| `vllm/model_executor/kernels/linear/scaled_mm/rocm.py` | ROCm scaled GEMM（FP8/INT4 等） | 错误 scale、错误 kernel 选择 |
| `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` | aiter scaled GEMM | dispatch 条件、contiguous |
| `vllm/model_executor/kernels/linear/mxfp4/aiter.py` | MXFP4 量化 GEMM | scale 布局 |
| `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py` | MXFP8 量化 GEMM | dtype 常量 |
| `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` | aiter MoE experts（所有 MoE 模型） | 专家路由、权重 pin |

**Tier 2 — dispatch/管道层（错 dispatch = 静默绕过）**

| 文件 | 影响面 |
|------|--------|
| `vllm/platforms/rocm.py` | ROCm 平台定义（is_hip、设备名、capability）— 被所有 ROCm 代码引用 |
| `vllm/_aiter_ops.py` | aiter 算子 Python 封装/注册（fake 实现也在这） |
| `vllm/kernels/aiter_ops.py` | aiter 算子 dispatch |
| `vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py` | MoE router |
| `vllm/model_executor/kernels/mhc/aiter.py` | MH 压缩 (multi-head compression) dispatch |
| `vllm/distributed/device_communicators/aiter_custom_all_reduce.py` | aiter 自定义 all-reduce |
| `vllm/compilation/passes/fusion/rocm_aiter_fusion.py` | torch.compile fusion pass |

**Tier 3 — 结构性（配置/CI/构建）**

- `requirements/rocm.txt`、`requirements/kv_connectors_rocm.txt`、`.buildkite/ci_config_rocm.yaml`、`docker/Dockerfile.rocm*`、`vllm/model_executor/layers/fused_moe/configs/*.json`（调优配置，device_name 为 AMD_Instinct_*）
- AMD 专属模型目录：`vllm/models/{deepseek_v32,deepseek_v4,inkling,kimi_k3,minimax_m3}/amd/`（模型专属但大量使用 aiter/Gluon 算子）
- `csrc/rocm/`（C++/HIP kernel）、`cmake/hipify.py`

**Tier-1 文件特殊规则**：
- `rocm_attn.py` / dispatch 类文件：新的 if/elif 分支必须每个分支都被测试覆盖；改了 dispatch 条件但只测了一个分支 → 必报。
- 任何 attention kernel 改动：确认 prefill 与 decode 两条路径都验证（P4：TP=4/8 切分后的 head 数）。

### 2.4 mori 集成面

- **MORI-EP**（MoE all2all，`--all2all-backend mori` + `--enable-expert-parallel`）：
  - `vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py`（`MoriPrepareAndFinalize`）
  - 需要 `VLLM_ROCM_USE_AITER=1`；EP 场景下每个设备持有 `num_experts/EP` 个专家
- **MORI-IO**（PD 分离 KV 传输，MoRIIOConnector）：
  - `vllm/distributed/kv_transfer/kv_connector/v1/moriio/`：`moriio_connector.py`（scheduler 侧）、`moriio_engine.py`（worker 侧）、`moriio_common.py`、`moriio_layout.py`
  - 注册点：`vllm/distributed/kv_transfer/kv_connector/factory.py`（新 connector 必须注册，否则 `--kv-transfer-config` 报错）
  - Python 依赖：`from mori.io import BackendType, EngineDesc, IOEngine, MemoryDesc, PollCqMode, RdmaBackendConfig, XgmiBackendConfig`
  - 配置（`--kv-transfer-config` JSON）：`kv_connector: "MoRIIOConnector"`、`kv_role: kv_producer|kv_consumer`、`kv_connector_extra_config`: `proxy_ip`、`proxy_ping_port`、`http_port`、`handshake_port`、`notify_port`（按 (DP,TP) rank 偏移）、`read_mode`（READ=consumer 拉取，默认 WRITE=producer 推送）、`backend: rdma|xgmi`；RDMA 调优：`qp_per_transfer`、`post_batch_size`、`num_workers`

### 2.5 硬件速查

- gfx942 = MI300X/MI325X，gfx950 = MI350/MI355，gfx1250 = RDNA4
- LDS per CU：gfx942/gfx950 = 64KB；gfx1250 = 320KB 但 `ds_read/ds_write` 立即数偏移 16-bit（超 64KB 会 VGPR spill）
- FP8：e4m3fnuz max=240（gfx942 默认 fnuz），OCP e4m3 max=448；**判断是否 fnuz 必须看 `tensor.dtype`（`fp8_e4m3fnuz`），不能看 arch 名**——同一 arch 上 fn 与 fnuz 可能同时存在（KV fnuz、Q fn）
- MFMA 硬件 tile（MI300X FP8）：M=16, N=16, K=32

## 3. 规则目录

严重程度：🔴 必须修复 / ⚠️ 建议修复 / 📝 备注。所有 🔴 都要过 🔴 gate（具体触发输入）。

### A — 覆盖缺口（修了一个路径，同 bug 活在兄弟路径里）

- **A1 兄弟变体未修复** ⚠️（Tier-1 文件则 🔴）：修复只改了 `rocm_aiter_fa.py`，`rocm_aiter_unified_attn.py` 或 mla 目录下的变体可能有同样 bug。扫描同目录 `_opt`/`_prefill`/`_decode`/`_v2`/`_sparse` 后缀文件。实例：aiter#3841 strided q_nope 修复漏了 `_prefill_opt`。
- **A2 共享路径无跨模型/跨后端验证** ⚠️：改动在 CUDA/ROCm 共享文件（如 `vllm/v1/attention/`、fused_moe 公共层）但只验证了 CUDA 或单个模型。ROCm 侧每个受影响模型族（DeepSeek V3/V3.2/V4、Kimi、Inkling、MiniMax、GPT-OSS）至少一个被测？
- **A3 激活条件宽于验证范围** ⚠️：新 dispatch 让某 kernel 对整族模型生效，但只在一种模型/配置上测过。实例：vLLM#16435 FusedMoE 错误地对整族激活，需要后续 restrict PR。

### B — 静默绕过（代码看起来完整，但某些输入悄悄走了错误路径）

- **B1 dispatch 门控参数未断言** 🔴：新 if/elif/else 中每个被挡掉的参数（`block_table`、`alibi_slopes`、`window_size`、`logits_soft_cap`、`dropout_p`、`is_causal`）必须被 assert 或转发。既不 assert 也不转发 = 静默错结果。实例：aiter#3576 `block_table is None` 假分支悄悄算了 dense attention。
- **B2 用尺寸代理判断 prefill/decode 相位** ⚠️：`max_q`、`q.shape[0]==1`、`seq_len<=1` 判 decode 会把短 prefill / chunked prefill 误判。用显式相位标志（`attn_metadata.is_prefill`）。
- **B3 cudagraph 捕获兼容性** 🔴：captured 区域内新增 Python if/loop、`tensor.item()`、`torch.empty/zeros` 会破坏捕获。触发点：model forward、`model_runner` 路径、rocm_aiter_fusion pass 生成的图内代码。静态变化（常量算术、既有 tensor 的 dtype cast、无分支 kwarg 转发）安全。
- **B4/B5 tl.constexpr 安全检查被关闭而无不变式证明** ⚠️：aiter shim 或 Triton kernel 中 `CHECK_BOUNDS=False` 之类必须由调用方不变式保证，PR 注释未说明 → 非法内存访问风险。实例：ATOM#1498 `CHECK_NEG_ONE_SENTINEL=False` 关闭 -1 slot 过滤。
- **B5/B6 API 传播不完整** 🔴/⚠️：
  | 子类型 | X（改动） | Y（未同步的下游） | Z（失败） |
  |---|---|---|---|
  | param-discard | 新参数加入签名 | 函数体没用它 | 值被接受但从未生效（正确性参数→🔴，性能旋钮→⚠️） |
  | param-removed | 参数删除 | 所有调用点 | TypeError |
  | rename | 公共符号改名 | 所有 import 点（含跨仓库） | AttributeError |
  | attr-missing | 新 getattr 属性 key | 未设置该属性的路径 | 静默零/None 回退 |
  | dispatch-silent | 多后端回退 | 调用方日志 | 无诊断的后端切换 |
  例外：基类强制签名、子类合法忽略参数 → 📝（结构性丢弃）。FP 自检：确认同 PR 没有加兼容 shim（同名 wrapper / alias / fc_name pin）再报。
- **B7 过度保守的 assert 挡住合法形状** ⚠️：`assert M % tileM == 0` 但 kernel 内部会 pad。实例：aiter#3998。

### C — 硬编码 arch/dtype（对 gfx942/bf16 正确，对 gfx950/fp8 悄悄崩）

- **C1 fnuz 判断用 arch 名** ⚠️：`if "gfx942" in arch: treat_as_fnuz()` 错误——用 `tensor.dtype == fp8_e4m3fnuz`。按 arch 门控*转换*可以，按 arch 判断*属性*不行。实例：aiter#4073。
- **C2 FP8 scale 上限硬编码** ⚠️：`fp8_max = 240.0` 对 fnuz 正确、对 OCP e4m3（448）错误。用 `get_dtype_max(dtype)`。FP 自检：常量所在路径若已被 runtime guard 限定单一 dtype/arch 则安全。
- **C3 dtype 硬编码** ⚠️：固定 `bf16`/`fp8_e8m0` 而路径实际处理多种 dtype。FP 自检：先搜同文件未改动行——同路径已有同样硬编码 = 既有风格，不报新违规。
- **C4 新 arch 字符串出现在 dispatch 条件** ⚠️：新 `+` 行引入 `'gfx942'`/`'gfx950'` 字面量。豁免：注释/docstring/目录名、集中注册表导入、kernel 专属包装函数内的 capability guard（如 `_detect_gfx1201()`）。FP 自检：同文件未改动行已有同一字符串 → 不报。

### D — 未初始化/边界状态（buffer 没准备好就用了）

- **D1 未初始化 buffer 上的原子归约** 🔴：`atomic_fmax(*ptr, val)` 读旧值——`torch.empty()`/`::empty()` 分配的垃圾值主导 max → 损坏的 amax → 损坏的 FP8 descale → 静默错误量化。vLLM 实例：给 aiter fused qk-norm-rope-quant 类算子传 `torch.empty` 的 amax 缓冲。atomicAdd 同理 🔴；零系数数学上抵消的 partial-sum 缓冲降 ⚠️（但 `0.0 × NaN = NaN` 仍要报）。
- **D1b 条件赋值的 UnboundLocalError** 🔴：变量只在 if/elif 分支赋值、块后无条件引用。检查：if 块前有没有 `var = None` 默认值。实例：ATOM#860。
- **D2 新默认路径无回退 env var** ⚠️：新实现未经充分验证就替换默认路径，应有临时 kill-switch（验证后移除；permanent feature-flag 会被 maintainer 拒）。
- **D3 hipblaslt 出现在调优配置** 🔴：任何 `+` 行含 `hipblaslt` 的 tuning CSV/YAML/JSON 都不能合入（不跨 Docker 持久、会 hang）。检查 `fused_moe/configs/*.json`、quantization utils configs。
- **D4 无引用地逆转安全不变式** 🔴：旧注释"must X because Y"→ 新代码删 X 称"X 不需要"但无 spec/asm/test 证明。触发：`torch.zeros→torch.empty`（旧注释有 must/required/read back as zero）、无解释删 assert、删 `.contiguous()`。实例：aiter#4043 两句注释直接矛盾。
- **D5 backbone 文件间逐字复制** ⚠️：同一算法块复制进 2+ Tier 1/2 文件仅改变量名（同公式、同注释结构、同魔法常量）。每个文件的不变式可能不同，未独立验证。AI 代码签名。实例：ATOM#1493 deepseek_v2/v4 chunked indexer。
- **D6/D7 torch.compile fake 实现** 🔴：`torch.library.custom_op`/`@compile_ops` 新算子无 `_fake`/`gen_fake`/`abstract_impl` → compiled 区域内 runtime crash 或静默 eager 回退；fake 返回 dtype/shape 与真算子不符 → torch.compile dtype 断言或静默错值。vLLM 实例位置：`vllm/_aiter_ops.py`。实例：aiter#4110 fake 返回 bf16、真算子返回 fp8。
- **D8 kernel 调用缺 contiguous 检查** ⚠️：可能 strided 的输入（slice、`.T`、非连续 view 输出）传给 C++/aiter kernel 前无 `.contiguous()`/`.is_contiguous()` 断言 → 从错误地址读，完全静默。
- **D9 int32 溢出** 🔴：`token_id * num_heads * head_dim` 在 int32 下 token_id > 16M 溢出（H=32, D=128）；`seq_start * K` 长上下文溢出。触发：token_id/seq_start/total_tokens/batch_offset 参与 buffer 地址/索引运算且乘法前未显式扩到 int64。vLLM 长上下文 + 大 batch 场景。

### E — 跨仓库同步（改动不完整，缺另一个仓库/文件的配套更新）

- **E1 新 aiter/mori 符号无上游 PR 链接** ⚠️：新 `from aiter import X` / 新 kwarg / mori 新 API 用法，PR 描述应链接 aiter/mori 侧 PR；新 kwarg 可能依赖未发布的 aiter 版本。用 `references/upstream/mori-structure.json` / aiter 仓库核符号存在性。
- **E2 新参数带向后兼容默认值 = 死代码** 📝：默认值保持旧行为 → 修复永远不会激活，谁传非默认值？实例：aiter#3773 `max_seqlen=-1`。
- **E3 插件桥未同步** ⚠️：ATOM 插件（`vllm/plugins/`、ATOM bridge）直接读 vLLM 数据结构（KV layout、签名）——改动需确认插件兼容。
- **E4 ROCm CI 未跑/被跳过** 🔴（触及下游消费面时）：`.buildkite/ci_config_rocm.yaml` 定义的 AMD 队列在 `checks` 里 skipped 或缺失，而 diff 触及 ROCm 路径 → 覆盖缺口，合入后 AMD 用户首当其冲。检查方法：fetch 时带 `--include-ci`，对照 PR 是否带 AMD CI 相关 label。CUDA CI 全绿不代表 ROCm 不崩。
- **E5 稳定契约改动需 owner sign-off** ⚠️→🔴：改变 `platforms/rocm.py`、`_aiter_ops.py`、attention dispatch 默认路径等广泛消费面的行为/签名，不应单人 approve 自合。用 `git log --format='%an' -- <file> | sort | uniq -c | sort -rn | head` 找 de-facto owner，建议 review 时 @ 提及。

### F — 资源重复（同一数据悄悄 pin 两次）

- **F1 新权重变体与原权重并存** ⚠️：新 `w13_preshuffled`/`w_quantized` 属性与原权重同时 pin → HBM 翻倍。大 MoE 模型（w13 7B+）只在峰值负载 OOM。检查：新变体创建后原张量是否释放（`del` / 换引用）。实例：ATOM#1469 "this will make us pin double weight"。

### G — 多流同步（stream A 写、stream B 读、中间无同步）

- **G1 缺 HIP/CUDA stream 同步** 🔴：HIP stream 默认并发。非默认 `torch.cuda.Stream`、显式 `stream=` 参数、或模型加载期在侧流准备的权重/KV 缓冲被 forward 在计算流消费——检查两次访问之间有没有 `synchronize()` / `wait_stream()` / `hipEventRecord`+`hipStreamWaitEvent`。缺 = 读垃圾，无 crash 无报错。
- **G1b 生产 serving 代码阻塞 queue.get() 无 timeout** ⚠️：`while True:` worker 循环里 `queue.get()` 无 `timeout=` 且无 Empty 处理 → 生产者异常退出时进程永久挂起。moriio 后台线程/代理循环是典型位置。豁免：测试代码、CLI 工具、一次性脚本。

### P — 性能证据（有性能声明时必查）

- **P1 性能 PR 无数值** ⚠️：触发词 perf/optimize/fuse/faster/+X%/replace kernel。必须有带单位的数字（ms、tokens/s、TFLOPS、%）。截图 ≠ 数字。
- **P2 benchmark 只有 toy shape** ⚠️：只测 M≤256、1 token、单一模型。生产：decode 单 token + prefill ISL≥4096、TP=4/8；MoE 模型用真实 E/topk。
- **P3 性能声明不可复现** ⚠️：缺脚本/命令、ROCm 版本、GPU 型号、TP/DP 配置、模型 checkpoint。
- **P4 TP 切分 shape 未覆盖** ⚠️：新 attention/norm 算子只在全 head 数（≈TP=1）测过。TP=4/8 时每设备 head 数除以 TP——H=128 过、H=32 可能 OOB。
- **P5 计时窗口隐藏了用户每次都要付的成本** ⚠️：per-call JIT 编译（未缓存）、每次冷启动在计时区内的 setup。FP 自检：成本是每次部署付一次（可摊销 → 不报）还是每次调用/每次冷启动都付（→ 报）。权重 shuffle 在计时外做一次是**正确方法学**。

### HK — 卫生检查（快速扫描）

| 检查 | 触发 | 标注 |
|---|---|---|
| 临时脚本 | `runperf-*.sh`、`test_local_*.py` 在 diff 里 | ⚠️ HK1 移除 |
| 无关文件 | 与 PR 目的无关的文件 | ⚠️ HK2 |
| module 级 sys.path | 非测试 .py 里 `sys.path.insert/append` | ⚠️ HK3 用相对导入 |
| TODO/stub 在新路径 | 新分支里 `# TODO`/`pass`/`NotImplementedError` | ⚠️ HK4 不完整实现 |
| 未注册的新 env var | `+` 行 `os.environ.get("VLLM_...")` | 📝 HK5 注册进 `vllm/envs.py` 并写文档 |
| 测试参考实现提升精度 | 参考用 Python float 字面量/`.float()`/`.double()` 提到 fp32，kernel 跑 bf16/fp8 | ⚠️ HK6 比较前 cast 回 kernel dtype |
| 新第三方依赖 | requirements 新增包；或新 import 非项目依赖。豁免：ROCm 系统包（`amdsmi`/`hip`/`rccl`）不在 PyPI，只需 try/except 守卫 + 注释 | 📝 HK7 |
| fp8_max/240/448 | 见 C2 | — |
| 未解释删除 assert | 见 D4 | — |

## 4. ROCm 专属附录

### 4.1 mori 集成检查清单（PR 触及 mori 时逐项过）

1. **注册**：新 connector/backend 是否在 `kv_connector/factory.py` 注册？`--all2all-backend mori` 是否在 CLI/`engine_args` 解析处加了分支且默认值行为不变？
2. **配置面**：`kv_connector_extra_config` 新 key 是否有文档（`docs/features/moriio_connector_usage.md`）？`read_mode`/`backend` 非法值的报错是否明确（B6 静默回退）？
3. **端口**：`notify_port` 按 (DP,TP) rank 偏移的计算是否正确（rank 0 与 rank N 端口冲突）？
4. **tensor 注册**：传给 `session.register_torch_tensor` / batch_read/batch_write 的 buffer 是否 pinned/registered 且生命周期覆盖传输完成（G1 同步）？
5. **HIP/CUDA 守卫**：mori 依赖仅 ROCm 可用，代码是否有 `current_platform.is_rocm()` 守卫或 import guard，CUDA 平台 import vllm 会不会因 mori 缺失而挂（`requirements/kv_connectors_rocm.txt` 与主 requirements 的分离逻辑）？
6. **回退**：mori 不可用（import 失败/环境不满足）时是否有明确报错或回退到默认 connector，而不是静默走错误的传输路径（B1/B6）？
7. **资源**：KV 传输 buffer 池是否重复分配（F1）；`queue.get()` 后台线程是否有 timeout（G1b）。

### 4.2 其他 ROCm 领域检查

- **LDS 预算**：新 Triton kernel / 改 tile size / blocking factor → LDS 是否超 64KB/CU（gfx942/gfx950）；超限 → VGPR spill → 性能退化或编译失败。
- **EP vs TP shape math**：EP 切专家（每设备 `num_experts/EP`），TP 切权重矩阵（列/行 slice）。dispatch 用 `world_size` 当分母时确认它是正确的并行度——用 TP 度去除专家数 = 静默路由到错误专家。
- **内存大小计算**：检查漏乘因子。实例：ATOM#1423 `swa_pages = num_swa_blocks` 漏乘 `* block_size`。
- **copy vs in-place**：有 `out=` 参数时用 `.copy_()` 浪费一次分配。
- **混用 fn/fnuz**：gfx942 KV cache 默认 fnuz，Q 量化可能产 fn（如 V4 Flash fused indexer）。混用路径需要显式 dtype dispatch——静默 dtype 不匹配编译得过但结果错。
- **魔法常量命名**：MFMA_M=16/N=16/K=32 等硬件常量应命名，不散落。

## 5. AI 代码诊断

预过滤表（3+ 命中 → 提示"AI 代码风险升高"）：

| 问题 | 警示信号 |
|------|---------|
| 描述解释机制（WHY）还是只讲动作（WHAT）？ | 只 WHAT → 风险升高 |
| 性能数字可疑地整（2.0x、1.5x）？ | 可能挑数据/编造 |
| 只有 trace 截图无数值？ | 截图 ≠ 数字 |
| 测试只有 token_num ≤ 256 或 M=1？ | AI 默认 toy shape |
| 门控参数被 assert 还是静默？ | 静默 → B1 |
| 无关文件被提交？ | HK2，AI commit 产物 |
| bug fix 根因没解释？ | "fix the issue" 无机制 → AI 猜测 |
| "Test Plan/Test Result" 留模板占位？ | 未测试的 PR |
| PR 描述 footer 有 "🤖 Generated with Claude Code" 之类署名？ | 作者可能不理解改动——提高 dispatch 与测试覆盖的审查优先级 |

**结构验证（diff 触及代码时强制）**：

1. **幻觉符号扫描**：列出 diff 中所有新符号（函数/kwarg/枚举/属性/import），逐一 grep 真实定义（aiter API、mori API、torch、vllm 模块）。找不到 = 缺陷直到证明存在。`🔴 [symbol] 在 [module] 中不存在/签名不符`
2. **孪生分歧**：镜像代码（prefill/decode、gfx942/gfx950、v2/v3、fwd/bwd）逐字段对比；一边 int64 一边 int32、一边 mask 一边没有、stride 顺序翻转 = 未完成的复制粘贴。
3. **声明/注释 ↔ 代码 + 数字溯源**：代码真的实现了描述/注释声称的不变式吗？最亮眼的数字追溯到来源；追溯不到标 `[unverified]` 并追问，不转述。
4. **安全戏法**：每个新 if/try/assert guard 可达吗？会触发吗？`except: pass` 吞真错误吗？
5. **为通过而校准的测试**：参考实现是否 kernel 的结构孪生（同 bug 两边都有，永远一致）？atol/rtol 无理由放宽？断言 kernel 自身输出？
6. **魔法常量无推导**：新 tile size/threshold/epsilon 有推导或调优依据吗？`📝 常量 [value] 无推导依据 — 追问来源`

## 6. 判定校准

- 每条 finding 三要素：**问题**（what/where，含 file:line）/ **影响**（不改会怎样：错结果/崩溃/OOM/性能退化）/ **行动**（以"作者应当…"/"建议作者…"/"建议 review 时追问…"结尾）。缺动词 = 不完整的 finding，不写。
- 每条标 `[已验证]`（追溯到代码/证据链）或 `[推测]`（合理但未证实，降级为"值得检查"）。只推断的根因绝不当作定论。
- 规则编码（P1、D2、A1…）绝不出现。
- 🔴 BLOCK 必须有具体触发输入；否则降级。
- 好的 finding 示例：
  - `🔴【正确性】fused_moe/experts/rocm_aiter_moe.py:xxx 将 amax 缓冲从 torch.zeros 改为 torch.empty，但 aiter 内核用 atomic_fmax 累积——垃圾值主导 max 导致 FP8 descale 静默错误。在 fp8_w8a8 + E≥128 配置下每个 token 的量化结果都会损坏。作者应当改回 torch.zeros 或提供内核内部零初始化的证据。` `[已验证]`
  - `⚠️【测试】新 MLA kernel 只在 TP=1 下测过；TP=8 时每设备 head 数为 1/8，shape math 未验证。建议作者补充 TP=8 decode + prefill 的测试。` `[推测]`
- 坏 finding 示例：`⚠️ 缺性能数字`（无影响无行动）；`🔴 D2 violation`（规则码无意义）。

## 7. 更新日志

| 日期 | 上游 SHA（ATOM/aiter/mori） | 变更 |
|------|---------------------------|------|
| 2026-08-20 | 728c3a3fa1 / 3fdb11afac / 2ab492a77f | 初始提炼：规则目录自 ATOM+aiter review-pr skill 建立，mori 集成面自 mori README + vllm PR #28664/#29304 调研 |
