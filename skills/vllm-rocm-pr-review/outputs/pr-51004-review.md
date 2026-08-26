# PR #51004: [Rocm] Fix LMcache support issues for the kimik3-dspark model

> **Author**: @haic0 | **State**: OPEN | **Date**: 2026-08-04（最近更新 2026-08-25）
> **Branch**: `haic0:rocm-kimik3-dspark` → `vllm-project:main` | **Labels**: rocm, kv-connector, kimi, k3, dflash, documentation, verified
> **Changes**: +167 -37 across 2 files | **ROCm 相关性**: 完全相关（Tier-1 文件 `mla/rocm_aiter_mla.py` + ROCm 启动脚本）

## 1. 动机 (Motivation)

Kimi-K3（Mamba+MLA 混合架构）在 ROCm 上跑 DSpark 推测解码（每步最多 2 个 draft token，多 token 验证）+ LMCache 时遇到两类问题：(1) `rocm_aiter_mla.py` 的小头（num_heads<16）Gluon 多 token 验证路径把 paged-KV 元数据按统一 `max_qo_len` 展开成逐验证行，而 DSpark 每请求每步的 draft token 数是**可变且不均匀**的，request→row 映射错位导致错误 KV 索引与 GPU 非法访问；小头 Gluon 验证中 `min_kv_seq_len` 推导为 0 时 AITER 选择 split-K 也会触发 ILA。(2) 部署缺少可复现方式：LMCache 需要 0.5.3 的 Mamba 统一视图 + subpaged MLA 视图 edit、ROCm GPU cache 注册依赖 CuPy、L1 池要按 `/dev/shm` 动态定尺寸。PR 将展开逻辑抽为带缓存的辅助函数（按真实 `qo_indptr` 展开、固定单 split），并新增一键启动脚本，在 8x MI355X 上验证通过（2/2 请求成功，接受率 98.98%）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` (+50/-37) | 新增 `_expand_dspark_mqa_metadata(decode, num_rows)`：按 `qo_indptr` 逐请求真实行数展开因果验证行，行数一致性断言，结果以 `num_rows` 为 key 缓存在 decode 元数据上供所有 MLA 层复用；`min_kv_seq_len` 固定为 1（单 split）；`forward_mqa` Gluon 验证分支改为调用该函数 |
| `examples/online_serving/start_kimik3_dspark_lmcache.sh` (+117) | 一键启动 LMCache server + vllm serve：AITER 环境变量、L1=60% 空闲 `/dev/shm` 动态计算、LMCache 0.5.3 能力校验（`_MambaUnifiedViewEdit`/`_SubpagedMLAAttentionViewEdit`）、端口就绪等待、trap 清理 |

交叉验证结论（对照本地 vllm main 基线）：

- `AiterMLADecodeMetadata` 是普通 dataclass（父类 `MLACommonDecodeMetadata` 无 slots），builder 每步新建实例且**总是**设置 `qo_indptr`（`rocm_aiter_mla.py:795`）→ 新断言与 `setattr` 缓存在结构上安全。
- `min_kv_seq_len` 是 metadata 既有字段，默认值即 1（注释："Minimum KV length used by Gluon to choose a safe split count"），普通 Gluon decode 路径一直传 1；旧验证代码传 `int(row_len.min())`（可为 0）才是偏离默认值的异常路径 → 固定为 1 的动机成立。
- 均匀 `qo_len` 情形下新公式 `per_req_len[r] - qo_len[r] + row_pos + 1` 与旧公式 `per_req_len[r] - (qlen-1) + t` 逐元素等价 → 无行为回归。
- 兄弟路径扫描：同目录 `rocm_aiter_mla_sparse.py`、`aiter_triton_mla.py` 及其他 backend 无同款 uniform-qlen 展开，无覆盖缺口；asm 验证路径直接消费 `qo_indptr`，天然支持变长。

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---------|------|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 5 |
| 📝 建议/备注 | 2 |

**⚠️【测试】PR 自带的运行时验证未覆盖被修改的 Gluon verify 分支** `[已验证]`

- **问题**: `use_gluon_verify`（`rocm_aiter_mla.py:988-993`）在 `is_quantized_kv_cache(kv_cache_dtype)` 为 True 时恒返回 False（"fp8 has one [asm kernel] via the q-row fold and must not come here"）。PR 的启动脚本与 benchmark 均使用 `--kv-cache-dtype fp8`，因此 8x MI355X 上实际执行的是 `forward_mqa` 第三分支（asm `mla_decode_fwd`），被修改的 Gluon verify 分支（bf16 KV 专属）在 PR 的 Test Plan 中只有"Python 语法编译"一项验证。若 Kimi-K3 的 num_heads ≥ 16，该分支对 K3 本身也不可达，作者遇到 GPU fault 的配置（bf16 + 小头）未出现在任何文档化验证中。
- **影响**: 修复声称解决的 split-K ILA 与元数据错位在真实硬件上未被回归验证，合入后该分支可能仍带病。
- **行动**: 作者应当补充 bf16 KV + num_heads<16 + DSpark verify 配置的 benchmark 证据，或说明触发原 GPU fault 的确切配置及在该配置下的复验结果。

**⚠️【正确性】`min_kv_seq_len=1` 下零长度行（cudagraph padding 请求）的 Gluon 单 split 行为未验证** `[推测]`

- **问题**: 展开后 `row_len` clamp 到 0 的行仍保留在新 CSR 中（旧注释明确 cudagraph padding 请求产生 0 行，seq_len=0）。旧代码 `min_kv_seq_len=int(row_len.min())`=0 触发 split-K 是 bug 本身；新代码固定 1 强制单 split，但 aiter Gluon 单 split 路径处理空行（indptr 相邻相等）的行为无证据。且 Gluon verify 分支不为 `o` 做 zero-fill——对比 asm 路径在 `max_qo_len > 1 and not has_persistent_metadata` 时显式 `o.zero_()`（注释："unwritten lanes cannot leak into logits"）。
- **影响**: 全 cudagraph + pad_uniform_mtp + 小头 + DSpark verify 场景下，padding 行的输出可能未写入（垃圾值）或触发越界读；PR 的 2 请求 benchmark 不含任何 padding 行。
- **行动**: 建议作者在跑满 uniform batch（含 padding 请求）的配置下验证，或提供 Gluon 对空行语义的说明。

**⚠️【测试】新辅助函数无单元测试，benchmark 规模极小** `[已验证]`

- **问题**: `_expand_dspark_mqa_metadata` 是纯张量逻辑（可由 `qo_indptr`/`paged_kv_indptr` 完全确定），未加 CPU 单测（变长 qo_len、0 长度请求、行数断言分支、缓存命中/失效路径均无覆盖）；benchmark 仅 2 请求（`--max-num-seqs 8` 未跑满），无法覆盖 batch 内 draft 长度混合与 prefix cache 命中后的 offload/load 路径。
- **影响**: 后续重构无保护网；元数据展开正确性只有 2 请求样本。
- **行动**: 建议作者为该函数补充 CPU 单测，并补充混合 draft 长度 + 并发请求的 benchmark。

**⚠️【CI/合并】pre-commit 失败且需 rebase** `[已验证]`

- **问题**: mergify 于 2026-08-25 报告 pre-commit 检查失败；`mergeable_state: unstable`、`rebaseable: false`。CI checks 明细无法获取（本地 gh 未认证、fetch 脚本 checks 字段为空），ROCm CI 队列是否覆盖此 PR 未知。
- **影响**: 阻塞合并。
- **行动**: 作者应当先运行 `pre-commit run --all-files` 修复格式问题并 rebase main 后重推。

**⚠️【兼容性】全 cudagraph 捕获下 `.item()` 同步与动态 `total` 的既有风险未缓解** `[推测]`

- **问题**: 展开中含 `int(new_indptr[-1].item())` + `torch.arange(total)`；cudagraph 捕获时 `total` 被烘焙为捕获期值，而 verify 步 `per_req_len` 随 seq 增长逐 step 变化，replay 期图形尺寸错配。PR 的缓存只把"每层一次"的同步降为"每步一次"，未移除同步本身。此外新增的行数一致性断言在 `pad_uniform_mtp` 合成 qo_indptr 时若 `B != sum(qo_len)`，会把静默错位变成捕获期 AssertionError（崩溃但至少不静默）。
- **影响**: 启用 cudagraph（非 enforce-eager）的小头 DSpark 验证配置下可能图形尺寸错配或崩溃——该风险基线代码同样存在（非本 PR 引入），PR 的验证配置以 `--enforce-eager` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 回避了它。
- **行动**: 建议 review 时追问作者该路径在 cudagraph 下的预期行为；至少应在注释中说明 enforce-eager 前提。

**📝【可维护性】缓存 `setattr` 的静默失败降级** — `MLACommonDecodeMetadata` 是普通 dataclass（无 slots），`setattr` 实际必然成功，try/except 纯防御性；失败时静默退化为逐层重算且无任何日志，排障时不易察觉。建议删除 try/except 或降级为 debug 日志。

**📝【脚本健壮性】L1 池下限未校验可用 shm** — `max(1, int(free*0.6))` 在 `/dev/shm` 小于 1 GiB 时（如 Docker 默认 64MB）仍输出 1 GiB，lmcache server 初始化会失败且报错不直观；脚本内嵌 Python 用 `python` 而非 `python3`。均为低优先级，可后续优化。

## 4. 现有讨论 (Existing Discussion)

- **mergify[bot]**：文档预览已生成；2026-08-25 最新评论报告 pre-commit 失败并要求修复后重推。
- **claude[bot]**：因 PR 来自 fork，自动审查被禁用，需 maintainer 评论 `@claude review` 触发。
- 暂无人工 review 意见；两位请求的 reviewer（@tjtanaa、@AndreasKaratzas）尚未响应。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 修复思路正确且与 metadata 既有默认值（`min_kv_seq_len=1`）自洽，代码结构无静默数值风险，但 PR 文档化的运行时验证（fp8 KV）恰好被 `use_gluon_verify` 的量化 KV 门控排除在被修改的分支之外，核心修复在真实硬件上的回归证据缺失；同时需处理 pre-commit 失败与 rebase。
