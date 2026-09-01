# PR #51171: [ROCm][MLA] Reach FULL cudagraphs for AITER MLA speculative decoding

> **Author**: @yudigege86 | **State**: OPEN | **Date**: 2026-08-05（最近更新 2026-08-26）
> **Branch**: `yudigege86:rocm-mla-full-cudagraphs` → `vllm-project:main` | **Labels**: `rocm`, `nvidia`, `verified`
> **Changes**: +192 -125 行，5 个文件（8 commits）| **ROCm 相关性**: 完全相关（Tier-1 文件 `mla/rocm_aiter_mla.py`）
> **CI（最新 head 6ae0d00）**: pre-commit ✓ / pre-run-check ✓ / DCO ✓；无 AMD 硬件 CI run；okorzh-amd 已 APPROVE

## 1. 动机 (Motivation)

ROCm 上 MLA 模型（Kimi-K3 + DSpark 草稿，8x MI355X TP8）的推测解码无法达到 FULL cudagraph，两个独立原因：(1) `TritonMLAMetadataBuilder` 从类常量上报 `UNIFORM_SINGLE_TOKEN_DECODE`，引擎对全部 attention group 取 min，一个草稿组就把整引擎降级到 PIECEWISE；(2) `AiterMLAImpl.forward_mqa` 小头数（<16 heads/rank）verify 路径每层执行 4 次 device→host 同步 + 动态分配，中止 HIP 图捕获。

经过评审驱动重构，最终方案为：Triton builder 按 KV group 的 `non_causal_multi_token_decode` 标记上报 `UNIFORM_BATCH`；AITER verify 路径改为把 `q` unflatten 成 4-D 后直接走 `mla_gluon` 的原生 MTP 入口（因果界由 kernel 内施加），`min_kv_seq_len` 在 `_build_decode`（捕获区外）计算并排除 cudagraph padding 行。实测 c1 12.38 → 44.00 tok/s（padding 修复后 45.33），c16 175.93 → 340.76，GSM8K 准确率不变。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `mla/rocm_aiter_mla.py`（Tier-1） | 删除初版扁平 KV 展开机制（flat_kv_indptr/indices、QLEN kernel 泛化）；verify 走 4-D MTP 入口；`_build_decode` 计算 `min_kv_seq_len`（`pad_uniform_mtp` 下只取 active 行）；`AiterMLADecodeMetadata` 新增 `min_kv_seq_len` 字段 |
| `mla/triton_mla.py` | 新增 `get_cudagraph_support` 类方法：`non_causal_multi_token_decode` → `UNIFORM_BATCH` |
| `v1/worker/gpu_worker.py` | worker 侧填充 `kv_cache_size_tokens` / `kv_cache_max_concurrency`（`get_kv_cache_capacity`） |
| `tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py` | 回归测试改写：spy 断言 4-D q + 每请求 paged_kv 元数据 + `use_2d_view=False` |
| `tests/v1/attention/test_rocm_aiter_mla_mtp_split.py` | 新增 padding 行不污染 `min_kv_seq_len` 的测试 |

## 3. Review 意见 (Findings)

意见类型统计：🔴 0 | ⚠️ 5 | 📝 3

**⚠️【兼容性】4-D MTP 入口依赖 aiter ≥ v0.1.19，vLLM 无版本 pin 也无能力探测** `[已验证缺失/失败模式未验证]`
- **问题**: `mla_gluon` 的 4-D MTP 入口在 aiter v0.1.19 才文档化（tjtanaa 评审所指 SHA 3135022）；`vllm/requirements/rocm.txt`、`common.txt`、`pyproject.toml` 均无 aiter pin，`_gluon_mla_decode_supported()` 只探测 gfx950，不探测 aiter 版本。`forward_mqa:1338-1352` 无条件向 `mla_gluon` 传 4-D q。
- **影响**: 用户环境 aiter < 0.1.19（如旧版 ROCm docker）时，小头数 verify 路径（DSpark + Kimi-K3 TP8 + MI355X）会命中旧入口——具体失败模式（assert 崩溃 vs 静默错值）取决于旧版本实现，未验证；若是静默错值则升级为正确性问题。
- **行动**: 建议作者在 PR 中写明最低 aiter 版本要求，或在 `_gluon_mla_decode_supported()` 中加入 4-D 入口的特性探测；同时确认 nightly docker 已携带 ≥ 0.1.19。

**⚠️【性能/兼容性】`min_kv_seq_len` 语义随 aiter 版本漂移，当前注释对 ≥ v0.1.20 不成立** `[已验证]`
- **问题**: aiter ≤ v0.1.19：bh16 两个 regime 用 `cdiv(min_kv_seq_len, BLOCK_N)` 封顶 `NUM_KV_SPLITS`（okorzh-amd 的 padding 修复在此版本上是 load-bearing，splits 1→4）。aiter ≥ v0.1.20（ROCm/aiter#4555）：bh16 完全不再读 `min_kv_seq_len`（split 数只由 launch budget 决定，kernel 内从运行时 KV 长度推导实际分块），仅 bh64 保留一个 assert——而小头数路径永远到不了 bh64。作者自己的 commit `88abec7f7b` 精确记录了这条版本边界，但 4-D 重写（`bb87d717e4`）后，`_build_decode:790-792` 的注释退化为「mla_gluon still wants a lower bound on the KV length it is asked to split」——对 ≥ v0.1.20 不成立。
- **影响**: 在 ≥ v0.1.20 上该计算（含每步 1-2 次 device→host 同步）是纯开销；注释会误导后续维护者。由于 vLLM 不 pin aiter，两个语义在用户群中并存。
- **行动**: 建议作者把 `88abec7f7b` 的版本边界表述恢复到当前注释中；如考虑性能，可探讨在 ≥ v0.1.20 上跳过计算（需版本探测，权衡后也可接受现状）。

**⚠️【性能】单 token decode 的 `min_kv_seq_len` 恒为默认值 1，aiter ≤ v0.1.19 上 KV-split 塌缩** `[已验证 pre-existing]`
- **问题**: `_build_decode:796-806` 只在 `use_gluon_verify`（qlen>1）分支计算 `min_kv_seq_len`；`use_gluon_decode`（qlen=1）分支沿用元数据默认值 1（main 上即是如此，非本 PR 引入）。在 aiter ≤ v0.1.19 上，小头数单 token decode 的 split 数 = `max(1, min(256//batch, cdiv(1,64)))` = 1。
- **影响**: head 数整除 16 的配置（如 TP16 → 8 heads/rank 的 DeepSeek-V3 / Kimi-K3）在 gfx950 上单 token decode 走 Gluon 时 KV-split 并行度塌缩为 1——与本 PR 修复的 verify 场景同源的性能问题。PR 已具备全部机制（`per_req_len` 已在手），只差把 gate 放宽到也覆盖 decode 分支（并处理 zero-qo 行）。
- **行动**: 建议作者顺手修复（或在 PR 中说明 aiter ≥ v0.1.20 已使该值无关），否则 ≤ v0.1.19 用户群的非 spec decode 性能仍受损。

**⚠️【测试/CI】真实 4-D kernel 调用在任何 CI 中均不执行，且无 AMD 硬件 CI run** `[已验证]`
- **问题**: causal-mask 回归测试用 spy 替换 `mla_gluon` 并 mock 架构探测，只断言契约（shape/元数据），真实 kernel 语义（因果尾部、split 行为）不在 CI 覆盖内；最新 head 的 checks 中无任何 AMD 硬件队列 run（PR 未带 `amd` label）。reger-men 的独立数据来自其自研等价实现（late-July build），非本分支。
- **影响**: aiter 侧 4-D 入口语义若变化（如因果界公式调整），CI 无法发现；合入后 AMD 用户首当其冲。
- **行动**: 建议合并前请求 AMD CI（`amd` label）跑 ROCm 测试队列，或由 AMD maintainer 明确背书真实硬件验证。

**⚠️【兼容性/跨后端】`triton_mla.py` 的 UNIFORM_BATCH 提升对所有平台生效，仅 ROCm 侧验证** `[推测]`
- **问题**: `get_cudagraph_support` 的提升作用于 TRITON_MLA 的所有部署（含 NVIDIA 上使用 TRITON_MLA 的 MLA 模型），且是 group 级提升——共享草稿 KV group 的因果目标模型也被一并提升（docstring 已记录）。多 token 块的 decode-path 准入（`reorder_batch_threshold`）是 pre-existing 的，本 PR 改变的是图捕获模式。
- **影响**: NVIDIA 上 FULL cudagraph + spec decode 的整链行为（compilation 的 spec-decode 降级警告、capture/replay）未验证；若某平台/配置下 FULL 捕获与推测解码组合存在其他约束，会以运行时错误形式暴露。
- **行动**: 建议作者在 PR 中注明跨后端影响面，并请 NVIDIA 侧 reviewer（PR 已带 `nvidia` label）确认该提升在 CUDA 上的安全性。

**📝【文档】PR 描述整体过时** `[已验证]`
- **问题**: PR body 仍描述已删除的扁平展开方案（"That expansion moves into `_build_decode`…"）及初版数据（46.30 tok/s / 46.6 MiB 预留，后者对应的缓冲区已不存在）；最新数据（44.00 / 340.76 / 45.33）散落在评论中。
- **行动**: 建议作者在合并前把 Purpose 与 Test Result 更新为 4-D MTP 方案的最终版本，否则未来读者会被误导。

**📝【注释】`use_gluon_verify` docstring 与调用点不一致** `[已验证]`
- **问题**: docstring（`rocm_aiter_mla.py:1003`）写 `use_2d_view=True`，而 `forward_mqa:1349` 实际传 `False`，测试也断言 `False`。
- **行动**: 建议作者修正 docstring。

**📝【可维护性】`gpu_worker.py` 的 `kv_cache_size_tokens` 填充当前无仓库内消费者** `[已验证]`
- **问题**: 该修复的唯一读者（初版扁平缓冲区尺寸计算）已在重写中删除；现仓库内 attention backends、model runner 均不读取该字段。注释仍引用「AITER MLA verify view, for one」，指向一个已不存在的消费方。改动对所有平台生效（NVIDIA worker 也会填充）。
- **行动**: 建议作者更新注释说明保留理由（通用 gap 修复 / 未来消费者），或考虑从本 PR 拆出以缩小跨后端影响面。

## 4. 现有讨论 (Existing Discussion)

- **@tjtanaa（08-20）**: 指出 aiter v0.1.19 已文档化 4-D 张量输入，质疑扁平展开的必要性——作者采纳并整体重写为 4-D MTP 方案。
- **@claude[bot] 评审（08-20，由 @shen-shanshan 触发）**: 两条发现——扁平缓冲区尺寸未计入 `pad_uniform_mtp` padding 行（可溢出）；越界检查用 `assert` 会被 `python -O` 剥离。两点均随扁平方案移除而失效，但 `assert`/`raise` 教训已体现在新代码（`B % qlen` 校验用 `raise ValueError`）。
- **@okorzh-amd（08-25）**: 发现 `min_kv_seq_len` 被 padding 行钉在 qlen，c1 复现中 Gluon KV-split 从 17 塌缩到 1——即 PR 宣称的 c1 加速是在 split 关闭状态下测得的。作者 `afd132cfd` 修复（active 行掩码）+ 新增单测，复测 c1 = 45.33 tok/s。随后 **APPROVED**。
- **@yudigege86（08-25）**: nightly 复测 c1 12.38 → 44.00（ITL 174.66 → 35.14 ms）、c16 175.93 → 340.76 tok/s。
- **@reger-men（08-14）**: gfx950 独立复现同样两个缺陷；补充关键对照——spec 关闭时强制 PIECEWISE 比 FULL 慢 5.2x（c1），说明 decode 图价值独立于推测解码。
- **@seungrokj（08-26）**: 催促 tjtanaa 对重写后版本复审（尚未回复）。
- **mergify**: 08-22 报合并冲突（已 rebase）；08-24/25 报 pre-commit 失败——最新 head（08-25 19:44）pre-commit 已绿，此项已解决。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK**

实现质量高、评审循环健康（两次重大评审均被采纳落地，AMD maintainer 已 APPROVE），无阻断性正确性缺陷。剩余事项为合并前收尾：更新过时的 PR 描述与版本相关注释、明确 aiter 最低版本要求、确认 NVIDIA 跨后端影响、请求 AMD CI 验证，以及等待 tjtanaa 对重写版本的复审。
