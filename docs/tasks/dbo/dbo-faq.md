# vLLM DBO 常见问题整理

> 背景：在为 vLLM 新增 `--dbo-scope {all,prefill}`（见 [dbo-scope-pr.md](./dbo-scope-pr.md)）的过程中，围绕 DBO 的阈值语义、与 ATOM TBO 的区别、以及 decode DBO 的取舍做了几次讨论，整理如下。
>
> 代码位置以本地 `/home/shashen/vllm/vllm` 和 `/home/shashen/vllm/ATOM` 为准。

---

## Q1：`--dbo-prefill-token-threshold` 比较的是单个 prefill 请求的 token 数，还是这一 step 整个 batch 的 token 数？

**结论：比较的是本 DP rank 在这一个 step 调度的整个 batch 的 token 总数，不是单个请求的。并且开 DP 时要求所有 rank 都满足阈值。**

### 1. 比较的是哪个 token 数

`check_ubatch_thresholds(config, num_tokens, uniform_decode)`（`vllm/v1/worker/ubatch_utils.py`）的 `num_tokens` 来源：

- **V1 runner**：`num_tokens_unpadded`，即 scheduler 本 step 给本 rank 调度的总 token 数（`total_num_scheduled_tokens`），不含 CUDA graph padding。调用链是 `gpu_model_runner.py` 的 `_determine_batch_execution_and_padding` → `coordinate_batch_across_dp`。
- **V2 runner**：本 rank 的 `num_tokens`，同样是整个 batch 的总量（见 `vllm/v1/worker/gpu/dp_utils.py`）。

这个总数包括 batch 中所有请求：

- 每个 prefill 请求在本 step 被调度的 chunk token；
- 每个 decode 请求的 token。开 MTP 时，每个 decode 请求算 1 + 草稿 token 数个 token。

两个阈值按 batch 类型区分：

| 阈值 | 适用 batch | 默认值 | 条件 |
|---|---|---|---|
| `--dbo-prefill-token-threshold` | 含 prefill 的 batch（准确说是非 uniform decode） | 512 | batch 总 token ≥ 阈值 |
| `--dbo-decode-token-threshold` | 纯 uniform decode batch | 32 | batch 总 token ≥ 阈值 |

举例：一个 step 中有 1 个调度了 300 token 的 prefill chunk，另有 60 个 decode 请求、每个 4 token（共 240）。batch 总数是 540 ≥ 512，所以会尝试拆分，尽管单个 prefill 只有 300 token。

### 2. DP 下所有 rank 必须一致

拆分只能全部 rank 一起拆或一起不拆，因为各 rank 每个 ubatch 的集合通信必须对齐：

- **V1**：每个 rank 先用自己的 token 数判断，再在 `_post_process_ubatch` 中用 `torch.all(...)` 汇总，只要一个 rank 不满足就全部不拆。另外，如果最慢的 rank 第二个 ubatch 会是空的（`is_last_ubatch_empty`），也会放弃拆分。
- **V2**：直接拿所有 DP rank 中最小的 token 数 `num_tokens_across_dp.min()` 和阈值比较。

所以实际效果是：每个 DP rank 本 step 的 batch 总 token 数都 ≥ 阈值，才会拆分。

### 3. 对调参的影响

- 如果 `--dbo-prefill-token-threshold 8192` 和 `--max-num-batched-tokens 8192` 一样大，只有每个 rank 在这一 step 都打满 8192 token 才会拆分。负载不均或 batch 不满的 step 都不会走 DBO。
- 单个 prefill 请求的长度不直接参与判断。长 prompt 会被 chunked prefill 切成每 step 最多 `max-num-batched-tokens`，它影响的只是 batch 总量。

---

## Q2：开 DP 时，ATOM 对 TBO 阈值的判断也是 all-or-none 吗？它的开启条件和 vLLM 有什么区别？

**结论：最终结果同样是全部 rank 一起开或一起关（否则 RCCL 会因为每个 ubatch 的通信大小不一致而 hang），但判定方式不同：ATOM 的 min-token 门槛按 OR 汇总，可拆分性按 AND 汇总，所以比 vLLM 宽松很多。**

### ATOM 的判定（`atom/utils/tbo/ubatching.py`）

**第一步：各 rank 本地计算两个 bit**（`local_tbo_precompute`）

| | `meets_min_tokens`（值不值得拆） | `can_split`（能不能拆） |
|---|---|---|
| prefill step | 本 rank prefill 请求的 token 总数 ≥ `ATOM_TBO_PREFILL_MIN_TOKENS`（环境变量，默认 8192） | token 中点切分（`ATOM_TBO_PREFILL_TOKEN_SPLIT=1`，默认）：token ≥ 2；按请求边界切分：prefill 请求数 ≥ 2 |
| decode step | 恒为 True，decode 没有 token 门槛 | decode 请求数 > 2（`decode_bucket_splits`）；dummy run 为 False |
| decode step，且未开 `--enable-tbo all` | False | False |

**第二步：DP 同步**（`sync_dp_metadata`，一次 all_gather）

```
active = OR(meets_min_tokens) AND AND(can_split) AND uniform_mode
```

- `OR(meets_min_tokens)`：只要一个 rank 达到 8k，其余未达到但能拆的 rank 也会被强制拆分。
- `AND(can_split)`：只要有一个 rank 在结构上拆不了，就整体关闭。
- `uniform_mode`：所有 rank 必须同为 prefill 或同为 decode；如果有的 rank 在 prefill、有的在 decode，直接关闭。
- 各 rank 的 ub0/ub1 token 数取 MAX，让所有 rank 用同样大小的 per-ubatch buffer 或 graph。

### 与 vLLM 的对比

| 维度 | vLLM | ATOM |
|---|---|---|
| 范围开关 | `--dbo-scope {all,prefill}`（本次新增） | `--enable-tbo [prefill\|all]` |
| prefill 门槛 | `--dbo-prefill-token-threshold`，默认 512 | `ATOM_TBO_PREFILL_MIN_TOKENS`，默认 8192 |
| 门槛的统计对象 | 本 rank 本 step 整个 batch 的总 token（含混在其中的 decode token） | 本 rank 本 step prefill 请求的 token 总数 |
| decode 门槛 | `--dbo-decode-token-threshold`，默认 32（按 token 数） | 无 token 门槛，只要求请求数 > 2 |
| 门槛的 DP 汇总 | AND：所有 rank 都要达标（V1 用 `torch.all`，V2 用 `min(num_tokens)`） | OR：任一 rank 达标即可 |
| 可拆分性的 DP 汇总 | AND（任一 rank 第二个 ubatch 为空就放弃） | AND(`can_split`) |
| prefill/decode 混合 | 用同步后的 `uniform_decode` 决定用哪个阈值 | 要求各 rank 模式一致，否则关闭 |
| 最终结果 | 全开或全关 | 全开或全关 |

### 实际影响

- **负载不均时 vLLM 更难触发 DBO。** 例如 DP8 中 7 个 rank 各有 8k prefill token、1 个 rank 只有 2k：ATOM 会照常开启，2k 的那个 rank 被强制拆分；vLLM 在阈值为 8192 时整体关闭。
- **想在 vLLM 上接近 ATOM 的效果：**
  - prefill 阈值要设得比 ATOM 的 8192 低，因为 vLLM 比较的是所有 rank 中最小的那个；
  - decode 用 `--dbo-scope prefill` 关掉，或者把 decode 阈值调到大约“每 rank 请求数 > 2”对应的 token 数。
- **把 vLLM 的门槛也改成 OR 汇总：** 技术上可行，V2 只需把 `num_tokens_across_dp.min()` 改成 `max()`。但这会改变 vLLM 现有语义，需要单独提 PR 并做性能验证，目前没有改。

---

## Q3：在 vLLM 上，为什么关掉 decode 的 DBO 会比较好？

**结论：这不是所有场景都成立。但在单机 MI355X、DP8、naive allgather/reducescatter、开 MTP 这类部署下，decode DBO 能隐藏的通信很少，拆批的代价却是确定要付的，所以大概率不划算。**

DBO 的收益来自一个 ubatch 的通信和另一个 ubatch 的计算重叠。前提是通信占比高，且每个 ubatch 的计算足以盖住通信。prefill 两个条件都满足，decode 往往都不满足。

### 收益小

- **decode 通信主要是固定延迟。** 每个 rank 每层只有几十到几百个 token，all-gather / reduce-scatter 的数据量很小，耗时主要来自启动和同步延迟。拆成两个 ubatch 后这部分固定开销要付两次，通信时间不会减半。
- **能用来掩盖通信的计算太短。** 每个 ubatch 的 MoE 和 attention 只有几十微秒，很难完整盖住另一个 ubatch 的通信。

### 代价大

1. **MoE 权重读两遍（最主要）。** decode 的 MoE GEMM 受显存带宽限制，耗时主要花在读专家权重上。拆成两个 ubatch 后，同一个专家在两个 ubatch 里都被激活时，要从 HBM 读两次，权重流量接近翻倍。prefill 是计算密集型，权重读取被大量 token 摊薄，所以没有这个问题。
2. **GPU 利用率下降。** batch 拆小后，attention/MLA 和 GEMM 的并行度减半，CU 占用不足，单个 kernel 效率下降。
3. **固定的调度和同步开销。** ubatch 线程切换、comm 流与 compute 流之间的 event 等待、两份 attention metadata、翻倍的 kernel launch，这些开销都不随 batch 变小而减少。decode step 本身很短，这部分占比就很显著。
4. **padding 浪费。** DP 下全部 rank 必须一起拆或一起不拆，所以要 pad 到相同 token 数；CUDA graph 还要按 capture size 对齐两个 ubatch。小 batch 下 padding 占的比例更高。
5. **额外的 CUDA graph。** 需要多 capture 一套 ubatched FULL decode graph，增加启动时间和显存占用。ROCm 上这条路径也出过问题，PR #57773 曾专门禁用 FULL graph 下的微批。
6. **MTP 会放大以上问题。** 每个 decode 请求有 4 个 token（1 + 3 个草稿），默认阈值 32 下每个 rank 只要有 8 个请求就会拆分。MTP + DBO 时 decode 的状态更复杂，之前就出现过 `_mtp_hidden_buffer` 被两个 ubatch 互相覆盖、导致精度严重下降的问题。

### 实测依据

此前在 DeepSeek-V4-Pro DP8 + MTP 上的 benchmark 中，DBO 相对 main 吞吐 +6.97%，TPOT −3.45%。这组数据用的是 `--dbo-decode-token-threshold 8192`，decode 基本不会被拆分，相当于 `--dbo-scope prefill`。

注意：还没有单独做过“decode DBO 开 vs 关”的对照，上面关于 decode 的结论是推理得出的。要定量确认，可以在同一负载下分别跑 `--dbo-scope all` 和 `--dbo-scope prefill`，重点对比 TPOT。

### decode DBO 可能更有利的场景

- **多机 EP：** 跨节点 all-to-all 延迟高（RDMA，百微秒级），通信在 decode step 中占大头。DeepSeek / SGLang 的 decode TBO 就是这类场景。
- **每个 rank 的 decode batch 很大**（数百请求以上），拆分后每个 ubatch 的计算仍然能盖住通信。
- **使用 DeepEP low-latency 等专为 decode 重叠设计的通信后端**，并且通信与计算的 SM 资源已经隔离。

因此 `--dbo-scope` 的默认值保持 `all`，不改变现有行为；单机、通信成本低的部署建议使用 `--dbo-scope prefill`。
