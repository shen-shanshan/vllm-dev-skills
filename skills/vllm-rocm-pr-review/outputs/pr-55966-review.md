# PR #55966: [ROCm][Spec Decode] Add Aiter MLA decode support non-causal draft block

> **Author**: @ppalanga | **State**: OPEN | **Date**: 2026-09-08（更新至 2026-09-11）
> **Branch**: fork `cb834667d590` → `main e7edf17cea` | **Labels**: `rocm`, `ready`
> **Changes**: +225 -6 across 9 files | **ROCm 相关性**: 完全相关（Aiter MLA decode + DSpark 非 causal draft block）

## 1. 动机 (Motivation)

DSpark（DeepSeek 并行草稿）的 draft 块是**块内非 causal**的：块内 token t 可以看见 t+1，只是相对上下文 causal。此前 `AiterMLAMetadataBuilder` 不声明 `supports_non_causal_multi_token_decode`，DSpark draft 块路由到 AITER MLA 时会在 `build()` 中直接 `ValueError`，用户只能把 draft attention 钉在 Triton MLA（块被 flatten 成单 token 行）。本 PR 是 #53001 的 rebase，让 AITER MLA decode 走"传真实 mask"路线：把 `causal` 透传给 aiter 的 `get_mla_metadata_v1`（`is_causal`）与 `mla_decode_fwd`（`causal=`），使非 causal draft 块可以用 Aiter MLA 服务；同时守住三个边界：Gluon verify 保持 causal-only、DCP verify 保持 causal-only、fp8 2-token 非 causal 块直接拒绝（aiter fp8 dispatch 折叠表无 qlen=2 的非 causal 入口，缺 kernel 会 abort 进程而非 raise）。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | 核心。`AiterMLABackend.supports_non_causal()` 新增（探测 aiter 是否带 `causal=`，失败即 raise）；builder 声明 `supports_non_causal_multi_token_decode=True`；`_build_decode` 增加 `causal` 参数：非 causal 强制走 persistent 调度（`not causal or ...`）、`get_mla_metadata_v1` 第 6 个位置参数 `True` → `causal`、`use_gluon_verify` 非 causal 直接 False、fp8 + qlen==2 非 causal 抛 `ValueError`；`forward_mqa` 非 DCP 路径把 `attn_metadata.causal` 传给 `mla_decode_fwd` |
| `vllm/_aiter_ops.py` | `_rocm_aiter_mla_decode_fwd_impl` / 静态 `mla_decode_fwd` 增加 `causal: bool = True` 并**无条件**传给 aiter；新增 `mla_decode_supports_non_causal()` 探针（`inspect.signature` 检查 `causal`，缺失即 RuntimeError，fail-closed） |
| `vllm/model_executor/layers/attention/mla_attention.py` | 公共 `build()` 向 `_build_decode` 传 `causal=not non_causal_decode`；基类 `_build_decode` 签名加 `causal: bool = True` |
| `flashattn_mla.py` / `flashmla.py` / `dots3_note/nvidia/attention.py` | 仅为匹配基类签名补 `causal` 参数，函数体不使用 |
| 3 个测试文件 | `use_gluon_verify` 非 causal 路由断言、mask 透传断言（mock aiter）、fp8 2-token 拒绝断言、缺 `causal` 参数 fail-closed 断言、真实硬件上 `causal=` 存在性断言（CDNA3+） |

## 3. Review 意见 (Findings)

意见类型 × 数量：⚠️ 建议修复 ×4，📝 建议/备注 ×3；无 🔴。

### ⚠️【兼容性】`causal` 无条件传给 aiter，旧版 aiter 的 causal decode 也会崩 `[已验证]`

- **问题**: `_rocm_aiter_mla_decode_fwd_impl` 现在对**每次** `mla_decode_fwd` 调用都传 `causal=`（默认 True）。公开 aiter 仓库 v0.1.20 之前的 `mla_decode_fwd` 签名是显式参数（无 `**kwargs`），旧 wheel 上普通 causal decode 会直接 `TypeError: unexpected keyword argument 'causal'`——而探针 `mla_decode_supports_non_causal()` 只在 `use_non_causal=True`（即配了 DSpark）的 backend 校验路径（`vllm/v1/attention/backend.py:324`）被调用，非 DSpark 用户根本不会触发那个友好的 RuntimeError，只会逐 token 撞 TypeError。
- **影响**: 依赖旧 aiter wheel 的 ROCM_AITER_MLA 用户在升级 vllm 后所有 decode 请求失败，且报错信息不指向"升级 aiter"。`requirements/rocm.txt` 无 aiter pin（镜像内自带），PR 描述也未注明最低 aiter 版本或链接 aiter 侧 PR。
- **行动**: 建议作者在 PR 描述中注明最低 aiter 版本（v0.1.20，公开仓库已确认含 `causal=True`，见 aiter/mla.py:557）并附 aiter 侧 PR 链接；考虑在 `_rocm_aiter_mla_decode_fwd_impl` 首次调用失败时给出"upgrade aiter"的提示，而非裸 TypeError。维护者已在讨论中明确接受"未来镜像必须支持"，此项不阻塞，但文档化仍是合入前应补齐的。

### ⚠️【测试】小 head 数 bf16 非 causal 路径的真实硬件数值未验证 `[推测]`

- **问题**: 新条件 `not causal or ...` 强制 12-head bf16 qlen>4 的非 causal 块走 persistent 调度——这正好是旧注释所说"gfx950 fold 在 qlen>4 无 bf16 persistent kernel"的形状。新注释论证"fold 丢弃的是 causal staircase，非 causal 块没有 staircase，所以 schedule 成立"，这是对 aiter 内部 kernel 覆盖的推断。PR 内单测（`test_a_bf16_padded_rank_past_qlen4_keeps_the_schedule_when_non_causal` 等）全部 mock 掉 aiter，只断言路由决策（`has_persistent_metadata`），不验证数值。作者 MI355X 实测（Kimi-K3 TP8）是大 head 数配置，未覆盖小 head 数（如 12/8 head，DeepSeek-V3 高 TP 切分后）非 causal 路径。
- **影响**: 若该形状在 aiter 侧实际无正确非 causal persistent kernel，小 head 配置下 DSpark draft 块会静默产出错误 logits，直接污染 draft 验证。
- **行动**: 建议作者补充 12/8-head + bf16 + qlen>4 非 causal decode 在真实 MI300/MI355 上与 Triton MLA（flatten 路线）的数值对比，或在 PR 中给出 aiter 侧该形状 kernel 覆盖的证据。

### ⚠️【测试】精度验证尚未完成，非 causal verify 的正确性仍是开放项 `[已验证]`

- **问题**: PR body 的 #53001 bench/accuracy 计划 checkbox 未勾选；作者 agentic 实测 211 个请求中 12 个 `InvalidInferenceResultError`（作者归因于配置/负载差异，未归因 PR，但未解释）；计划中的 8k/1k serving 与 RULER `niah_single_2` A/B 结果截至最后一次评论仍未贴出。本 PR 的全部价值在于非 causal draft 块验证的正确性，而这一点的独立精度证据目前缺失。
- **影响**: 掩码传递错误（例如 `is_causal` 位置参数错位）在吞吐指标上不会立即暴露，只有精度基准能抓。
- **行动**: 建议作者在合入前贴出 RULER/8k-1k 对比结果，并说明 12 个 `InvalidInferenceResultError` 的根因。

### ⚠️【兼容性】fp8 KV + qlen==2 非 causal 块在运行中抛 ValueError，配置层无法预先拦截 `[推测]`

- **问题**: `_build_decode` 中 fp8 + 非 causal + `max_qo_len==2` 直接 `raise ValueError`。该 guard 只存在于运行时 metadata 构建路径，engine 启动校验（backend.py 的 `use_non_causal` 检查）不知道 aiter 的 kernel 折叠表缺 qlen=2。qlen==2 的非 causal 块可达：draft 块在接近 max_model_len/EOS 时被截断到剩余 2 token 即触发（`build()` 的 uniform 检查允许截断块）。
- **影响**: 一个通过了启动校验的 fp8 KV + DSpark 配置，在某个请求即将结束时整个 batch 崩溃（而非优雅拒绝该请求）。作者注释承认这是临时洞（"Drop this once the pinned AITER folds that length"），且缺 kernel 时 aiter 会直接 abort 进程，guard 是必要的——但 crash 时机在最差的时刻。
- **行动**: 建议作者确认 qlen=2 截断块在 DSpark 下确实可达；若可达，考虑在 engine 启动时对 (fp8 KV cache + DSpark 非 causal) 组合给出配置级告警，把失败前移到部署阶段。

### 📝【设计】`inspect.signature` 探针对 aiter 打包方式变化敏感 `[推测]`

- **问题**: `mla_decode_supports_non_causal()` 依赖 `inspect.signature(mla_decode_fwd)` 能看到 `causal`。当前 aiter 的 `mla_decode_fwd` 是纯 Python 包装（公开仓库确认），signature 可读；但若未来 aiter 把该入口改回 pybind/内置函数（signature 为 `(*args, **kwargs)`），探针会 fail-closed 报"upgrade aiter"，即使内核实际支持。`test_installed_aiter_mla_decode_accepts_causal` 在真实 CI 硬件上把这条链 pin 住了，风险可控。
- **影响**: 仅影响未来 aiter 打包变化时的可用性，不影响正确性（fail-closed 方向安全）。
- **行动**: 建议作者在探针 docstring 里注明对 aiter 入口保持 Python 包装的依赖（当前注释已隐含），或后续改为运行时 probe（用 try 调用一次最小 shape）消除对 signature 的依赖。

### 📝【设计】非 causal + DCP 被双保险拒绝（fail-fast，正确） `[已验证]`

- **问题**: `supports_non_causal_multi_token_dcp` 保持 False → `backend.py:326` 的 backend 校验和 `MLACommonMetadataBuilder._validate_dspark_dcp_support`（mla_attention.py:2187）双重拦截 DSpark + decode_context_parallel_size>1 的非 causal 组合，报错明确。`forward_mqa` 的 DCP 分支不传 `causal` 因此不可达，无静默错。对比 FlashInfer MLA（flatten + `dcp_tot_seq_lens`）已支持非 causal DCP。
- **影响**: 无正确性风险；仅功能缺口（AITER MLA + DSpark + DCP 用户需换 backend）。
- **行动**: 无需改动；若报错信息能点名可用 backend（如 FlashInfer MLA）会更友好。

### 📝【可维护性】三个子类为匹配基类签名"结构性地丢弃" `causal`，安全但需 sign-off `[已验证]`

- **问题**: `flashmla.py`、`flashattn_mla.py`、`dots3_note/nvidia/attention.py` 的 `_build_decode` 增加 `causal` 参数但不使用。已验证二者 builder 未声明 `supports_non_causal_multi_token_decode`（保持 False），非 causal 块在 `build()` 即被 ValueError 拦截，丢弃的参数不可达，无静默错误。代码库内全部 5 处 `_build_decode` 定义均已同步签名（grep 验证），无遗漏 override 会 TypeError。
- **影响**: 无功能影响；但 AndreasKaratzas 已要求 flashmla 改动需 TJ 批准，Fangzhou-Ai 已请 @njhill 看 NVIDIA 侧 dots3_note 改动（作者也提出可 revert 该文件改动，讨论中未最终决定）。
- **行动**: 建议作者等待 TJ / njhill 的 sign-off，或如讨论中所说直接 revert dots3_note 的改动以减少合入门槛。

## 4. 现有讨论 (Existing Discussion)

- **探针设计之争（核心讨论）**: Fangzhou-Ai 质疑 `supports_non_causal()` 的 gate 必要性（"AITER 在 ROCm 路径总是存在"），主张直接失败而非静默回退；AndreasKaratzas 认为动态探测是正确方向，但建议探测为 false 时记日志。最终三方就 fail-closed 收敛：探测失败直接 RuntimeError（"upgrade aiter rather than falling back"），Fangzhou-Ai 对最终代码 LGTM。当前 diff 中 raise 位于探针内，与讨论结论一致。
- **dots3_note 改动**: Fangzhou-Ai 质疑"为什么 NVIDIA 侧需要这个改动"，作者解释纯为基类签名一致性；作者提出可 revert，未决。
- **flashmla 审批**: AndreasKaratzas 明确此 PR 因触碰 `flashmla.py` 需要 TJ 批准。
- **过程提示**: AndreasKaratzas 提醒作者"let's not reply with AI to reviewers"。
- **CI 状态**: mergify 曾报 pre-commit 失败；Buildkite #88209 有 5 个失败 job 重试中，最新 commit 触发新 build #88252（结果未出）。按 CI 归因纪律，未将具体失败归因于本 PR。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK**

路由与门控设计整体扎实：mask 透传两条路径（`get_mla_metadata_v1` 的 `is_causal` / `mla_decode_fwd` 的 `causal=`）都有测试断言，非 causal+DCP 与 Gluon verify 被双保险挡在 causal-only 之内，所有 fail 点均为 fail-closed。主要缺口在验证侧：精度基准（RULER 等）未贴出、小 head 数非 causal 路径无真实硬件数值验证、CI 未绿（pre-commit + 5 个 job 重试中）、旧 aiter 兼容性未文档化（最低版本 v0.1.20 + aiter PR 链接缺失），以及 flashmla（TJ）与 dots3_note（njhill）的 sign-off 待定。建议作者补齐上述验证与文档后再合入。
