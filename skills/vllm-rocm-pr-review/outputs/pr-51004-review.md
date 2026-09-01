# PR #51004: [Rocm] Fix LMcache support issues for the kimik3-dspark model

> **Author**: @haic0 | **State**: OPEN | **Date**: 2026-08-04（最近更新 2026-08-25）
> **Branch**: `haic0:rocm-kimik3-dspark` → `vllm-project:main` | **Labels**: rocm, kv-connector, kimi, k3, dflash, documentation, verified
> **Changes**: +167 -37 across 2 files | **ROCm 相关性**: 完全相关（Tier-1 文件 `mla/rocm_aiter_mla.py` + ROCm 启动脚本）
> **合并状态**: `mergeable: false`（merge_state: dirty，与当前 main 冲突）

## 1. 动机 (Motivation)

Kimi-K3（Mamba+MLA 混合架构）在 ROCm 上跑 DSpark 推测解码（每步最多 2 个 draft token，多 token 验证）+ LMCache 时遇到两类问题：(1) `rocm_aiter_mla.py` 的小头（num_heads<16）Gluon 多 token 验证路径把 paged-KV 元数据按统一 `max_qo_len` 展开成逐验证行，而 DSpark 每请求每步的 verify 行数是**可变且不均匀**的，request→row 映射错位导致错误 KV 索引与 GPU 非法访问；小头 Gluon 验证中 split-K 也触发 ILA。(2) 部署缺少可复现方式：LMCache 需要 0.5.3 的 Mamba 统一视图 + subpaged MLA 视图 edit、ROCm GPU cache 注册依赖 CuPy、L1 池要按 `/dev/shm` 动态定尺寸。PR 将展开逻辑抽为带缓存的辅助函数（按真实 `qo_indptr` 展开、固定单 split），并新增一键启动脚本，在 8x MI355X 上验证（2/2 请求成功，接受率 98.98%）。

**注意**：本 review 更新于 2026-08-30，当日 #51171 "[ROCm][MLA] Reach FULL cudagraphs for AITER MLA speculative decoding" 已合入 main，恰好重写了本 PR 修改的代码路径（见 Finding 1）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` (+50/-37) | 新增 `_expand_dspark_mqa_metadata(decode, num_rows)`：按 `qo_indptr` 逐请求真实行数展开因果验证行，行数一致性断言，结果以 `num_rows` 为 key 缓存在 decode 元数据上供所有 MLA 层复用；`min_kv_seq_len` 固定为 1（单 split）；`forward_mqa` Gluon 验证分支改为调用该函数 |
| `examples/online_serving/start_kimik3_dspark_lmcache.sh` (+117) | 一键启动 LMCache server + vllm serve：AITER 环境变量、L1=60% 空闲 `/dev/shm` 动态计算、LMCache 0.5.3 能力校验（`_MambaUnifiedViewEdit`/`_SubpagedMLAAttentionViewEdit`）、端口就绪等待、trap 清理 |

交叉验证结论（对照 PR base 8fe9317f2e 与当前 main）：

- **展开数学正确**：新公式 `per_req_len[r] - qo_len[r] + row_pos + 1`（clamp≥0）在均匀 qo_len 下与旧公式逐元素等价；`src` 索引上界 = `old_indptr[r] + per_req_len[r] - 1 = old_indptr[r+1] - 1`，不越 `paged_kv_indices`。
- **缓存结构安全**：`AiterMLADecodeMetadata` 是普通 dataclass（无 slots），builder 每步新建实例且总是设置 `qo_indptr` → 断言与 `setattr` 缓存在结构上成立；同一对象在一步内跨 MLA 层共享，缓存命中语义正确。
- **`min_kv_seq_len=1` 自洽**：该字段默认值即 1（注释："Minimum KV length used by Gluon to choose a safe split count"），普通 Gluon decode 路径一直传 1（含 cudagraph padding 零长度行的常规组合）；旧验证代码传 `int(row_len.min())`（可为 0 或大值 → split-K）才是偏离默认值的异常路径。固定为 1 的动机成立。
- **兄弟路径扫描**：`rocm_aiter_mla_sparse.py` 直接把 `qo_indptr` 交给 aiter kernel（天然支持变长），无同款 bug；但 CUDA `triton_mla.py` 有同类均匀假设（见 Finding 3）。

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---------|------|
| 🔴 必须修复 | 1 |
| ⚠️ 建议修复 | 4 |
| 📝 建议/备注 | 3 |

**🔴【兼容性】目标代码路径已被今日合入的 #51171 重写，PR 需 rebase 并重新对齐方向** `[已验证]`

- **问题**: 三件事实叠加。(a) PR 与 main 冲突：GitHub 显示 `mergeable: false`、merge_state: dirty。(b) #51171（2026-08-30 合入，commit dbf662c9e8）删除了 `forward_mqa` 中 Python 侧逐 token 展开（即本 PR 替换的那段代码），改为把 `q_nope.unflatten(0, (num_reqs, qlen))` 的 4-D MTP 视图直接交给 `mla_gluon`，由 kernel 内做 per-position 因果边界（"score_end = min(split_kv_end, seq_len - qlen + q_pos + 1)"），并把 `.item()` 同步从 forward 移到 builder（main 注释明确点名"the per-layer syncs in forward_mqa did [abort capture]"）以支持 speculative decode 的 full cudagraph；`min_kv_seq_len` 改由 builder 按 active 请求计算。(c) main 新代码对非均匀 batch 显式拒绝：`num_reqs = B // qlen; if num_reqs * qlen != B: raise ValueError`（main `rocm_aiter_mla.py:1451-1456`）。
- **影响**: rebase 后本 PR 的 `_expand_dspark_mqa_metadata` 要么直接冲突、要么成为死代码；PR 保留的 forward 内 `.item()` 与 #51171 的 full-cudagraph 方向直接矛盾。更关键的是：PR 声称修复的非均匀 qo_len 场景在 main 上**仍未被支持**，只是从 GPU fault 变成了显式 ValueError 崩溃。触发输入：bf16 KV + 12-head K3 DSpark verify + batch 内某请求 verify 行数 < `max_qo_len`（例如刚完成 prefill 的请求只有 1 行而其他请求 3 行）→ main 当前代码直接 raise。
- **行动**: 作者应当 rebase 到最新 main，并与 #51171 作者及 ROCm maintainer 对齐：非均匀 qo_len 是放进 kernel（扩展 4-D MTP entry 的 per-position bound，需确认 mla_gluon/aiter 是否支持）还是保留 Python 展开（需解决 full-cudagraph 兼容，例如把展开移到 builder/metadata 构建阶段）；同时用具体 trace 说明非均匀 verify batch 在 vLLM v1 DSpark 调度下确实可达（这是整个 PR 的价值前提）。

**⚠️【测试】PR 的运行时验证（fp8 KV）不经过被修改的 Gluon verify 分支** `[已验证]`

- **问题**: `use_gluon_verify` 在 `is_quantized_kv_cache(kv_cache_dtype)` 为 True 时恒返回 False（"fp8 has one [asm kernel] via the q-row fold and must not come here"）。PR 的启动脚本与 benchmark 均使用 `--kv-cache-dtype fp8`，因此 8x MI355X 上实际执行的是 `forward_mqa` 的 asm 分支，被修改的 Gluon verify 分支（bf16 KV 专属）在 Test Plan 中只有"Python 语法编译"一项验证。且 benchmark 仅 2 请求、接受率 98.98%——几乎全程均匀 batch，非均匀 qo_len 这一核心场景反而没有出现。
- **影响**: 修复声称解决的元数据错位与 split-K ILA 在真实硬件上未被回归验证；"2/2 请求成功"不能证明修复生效。
- **行动**: 作者应当补充 bf16 KV + 小头 + DSpark verify 配置下的 benchmark（包含混合 verify 行数的 batch），或明确说明触发原 GPU fault 的确切配置及在该配置下的复验结果。

**⚠️【正确性】CUDA TRITON_MLA 非因果 DSpark 路径存在同类均匀 qlen 假设** `[已验证代码 / 触发场景未证实]`

- **问题**: main 的 `triton_mla.py:320-323` 在非因果 DSpark 展开中同样假设均匀：`query_len = num_decode_tokens // num_decodes` + `block_table.repeat_interleave(query_len)`。非均匀 batch 时整数除法截断，展开行数与真实 query 行数错配——与本 PR 在 ROCm 侧修复的是同一类 bug（注释称该路径满足 UNIFORM_BATCH 契约，但 eager 非均匀 batch 不受契约保护）。
- **影响**: 若非均匀 verify batch 真实可达，CUDA 侧同样存在错误 KV 视图风险；若不可达，则 ROCm 侧的修复前提也需要重新论证。两边的结论应一致。
- **行动**: 建议 review 时向作者（及 spec-decode maintainer）追问：非均匀 verify 行数在 vLLM v1 DSpark 调度下是否可达？若可达，CUDA 后端是否需要同类修复。

**⚠️【CI/合并】pre-commit 失败，且 PR 无 AMD CI 覆盖** `[已验证]`

- **问题**: mergify 于 2026-08-25 报告 pre-commit 失败（当前 head 的 check runs 中 pre-commit 仍为 failure）；head commit 的 status 仅有 readthedocs，无 buildkite AMD CI（PR 未加 amd label）。
- **影响**: 阻塞合并；ROCm 路径的合入验证完全依赖作者自报的 8x MI355X bench。
- **行动**: 作者应当运行 `pre-commit run --all-files` 修复并重推，同时请 maintainer 触发 AMD CI。

**⚠️【测试】新辅助函数无单元测试** `[已验证]`

- **问题**: `_expand_dspark_mqa_metadata` 是纯张量逻辑（可由 `qo_indptr`/`paged_kv_indptr` 完全确定），未加 CPU 单测：变长 qo_len、零长度请求、行数断言分支、缓存命中/失效路径均无覆盖。
- **影响**: 后续重构无保护网；该函数承载的正是最容易回归的索引数学。
- **行动**: 建议作者为该函数补充 CPU 单测（torch 张量即可，无需 GPU）。

**📝【可维护性】脚本导出两个 vllm main 中不存在的 env var** `[已验证]`

`VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION` 与 `SAFETENSORS_FAST_GPU` 均未出现在 vllm main 的 `envs.py`、kimi_k3 模型代码及 model_loader 中（前者查无此名，后者仅有同域但不同的 `VLLM_FASTSAFETENSORS_QUEUE_SIZE`）。对普通用户而言是误导性的死变量。建议作者确认它们是否只在自己的 fork/内部版本生效，否则从脚本移除或注册进 `envs.py`。

**📝【可维护性】缓存 `setattr` 的 try/except 是死代码** `[已验证]`

`AiterMLADecodeMetadata` 是普通 dataclass，`setattr` 必然成功；失败时静默退化为逐层重算且无任何日志。建议删除 try/except（或失败时打印 warning）。

**📝【脚本健壮性】L1 池下限未校验可用 shm** `[已验证]`

`max(1, int(free*0.6))` 在 `/dev/shm` 小于 1 GiB 时（如 Docker 默认 64MB）仍输出 1 GiB，lmcache server 初始化会失败且报错不直观；脚本内嵌 Python 用 `python` 而非 `python3`。低优先级，可后续优化。

## 4. 现有讨论 (Existing Discussion)

- **mergify[bot]**：文档预览已生成；2026-08-25 最新评论报告 pre-commit 失败并要求修复后重推。
- **claude[bot]**：因 PR 来自 fork，自动审查被禁用，需 maintainer 评论 `@claude review` 触发。
- 暂无人工 review 意见；请求的 reviewer（@tjtanaa、@AndreasKaratzas）尚未响应。

## 5. 结论 (Verdict)

🔴 **BLOCK** — 当前状态无法合并：`mergeable: false` 与 main 冲突，且目标路径刚被 #51171 以相反方向（kernel 内 4-D MTP 直传 + full cudagraph）重写，rebase 后本 PR 的实现将成为死代码；而 PR 修复的非均匀 verify 场景在 main 上仍以 `ValueError` 形式存在（触发输入明确：非均匀 verify batch）。作者应当 rebase 后与 #51171 作者/ROCm maintainer 对齐修复方向（kernel 内支持 per-request qo_len vs 把展开移到 builder），并补上 bf16 KV 配置的实测证据——PR 的修复思路本身（按真实 `qo_indptr` 展开）是对的，价值主张仍然成立，但需要在新基座上重新落地。
