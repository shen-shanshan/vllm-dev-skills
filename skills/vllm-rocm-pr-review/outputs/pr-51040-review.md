# PR #51040: [ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts

> **Author**: @xiaohuguo2023 | **State**: OPEN | **Date**: 2026-08-04(最近更新 2026-08-26)
> **Branch**: `xguo/rocm-mla-fp8-prefill-independent` → `main` | **Labels**: rocm, ready, verified, k3
> **Changes**: +85 -28 lines across 2 files | **ROCm 相关性**: 完全相关(AITER FP8 MLA prefill,gfx950)

## 1. 动机 (Motivation)

AITER 的 FP8 MLA prefill 汇编路径(`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`)硬性要求 16 对齐头数。Kimi-K3 在 TP8 下每 rank 仅 12 头(96 头 / kv_lora_rank=512),不满足条件时回退到 BF16 FMHA 解压路径——该回退构建的 bf16 working set 不受 FP8 KV pool 显存核算覆盖,实测 ~197k tokens 长上下文 prefill 时 OOM(KV pool 使用率 <4%)。本 PR 是已合并 #50578(decode 侧 pad 12→16)的 prefill 对应物:利用 MLA 注意力逐 query head 独立的性质,把 Q/K/V 复制填充到 16 头跑汇编 kernel 再切回真实头数(数学上精确),同时按填充后的 head 数构建 PS 元数据,顺带把部分 tile 数从 ~4032 降到 ~960(gcd(16,256)=16 vs gcd(12,256)=4),省出约 6 GiB workspace。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` — builder 门控 (L420) | 放宽 `_fp8_prefill_enabled`:heads 条件加 `0 < num_heads < 16`;新增 `kv_cache_dtype_str == "fp8"` 与 `model_config.dtype == bf16` 两个 gate(后者是 review 过程中发现 fp16 静默错数后补上的) |
| 同文件 — `_init_fp8_prefill_ps_buffers` (L487) | `num_head_k = max(16, num_heads)`;新参数 `attn_out_dtype`;`num_heads < 16` 时向 workspace 额外预留 `[max_num_batched_tokens, 16, v_head_dim]` 输出 scratch(lock 前预留 → 固定地址) |
| 同文件 — `_build_fp8_prefill_ps_metadata` (L593) | 同样 `num_head_k = max(16, num_heads)`,work/reduce map 与填充后的 16 头张量匹配 |
| 同文件 — impl 门控 (L1067) | 用 `is_quantized_kv_cache(kv_cache_dtype)` + 同样的 heads 条件(与 builder 谓词不完全一致,见 Finding 4) |
| 同文件 — `_mla_fp8_prefill_attn` (L1119–1205) | `_pad16` 时经 `AiterMLAHelper.get_mla_padded_q` 把 q/k/v 填到 16 头,输出改从 workspace scratch 取 `[total_q, 16, v_head_dim]`,kernel 跑完后 `copy_` 回调用方的 12 头输出 |
| `tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py` | 引入 `ATTN_OUT_DTYPE = torch.bfloat16` 常量统一 dtype;`_build_prefill_metadata` 增加 `attn_out_dtype` 参数传入真实 builder |

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 正确性 | 0 | 0 | 0 |
| 测试 | 0 | 1 | 0 |
| 兼容性 | 0 | 1 | 0 |
| 性能 | 0 | 1 | 1 |
| 可维护性 | 0 | 0 | 2 |

**⚠️【测试】pad-to-16 路径——本 PR 的全部核心逻辑——没有任何自动化测试覆盖** `[已验证]`

- **问题**: `tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py:47` 的 `NUM_HEADS = 16` 是 16 对齐的,`_pad16` 分支、`max(16, num_heads)` 的 workspace 预留分支、以及 `out_3d[:, :_real_nhead]` 的切回逻辑永远不会执行;文件头 docstring 还写着"FP8 prefill requires 16-aligned heads"(已过时)。此外该测试 `skipif` 依赖 gfx950(MI355),vLLM 常规 ROCm CI 队列(gfx942)整文件跳过——即 CI 对这条路径的覆盖为零。改动本身仅是 dtype 常量重构,对 16 头场景行为不变。
- **影响**: 12 头 pad 路径的正确性目前完全依赖作者在 MI355X 上的手工验证(GSM8K + 长上下文)。若未来 #50578 的 `get_mla_padded_q` 或 builder 逻辑演化,pad 路径的回归不会被任何 CI 捕获,合入后第一个踩坑的是 AMD 用户。
- **行动**: 作者应当参照 decode 侧 `test_rocm_aiter_mla_head_padding.py::test_h12_aiter_mla_decode_matches_reference` 的模式,把测试参数化为 `NUM_HEADS ∈ {12, 16}`(12 即 K3 TP8 的真实头数),并顺手更新 docstring。

**⚠️【兼容性】K3 在 TP4 下(24 heads/rank)仍落在回退区间,原 OOM 问题在该配置下依旧存在** `[已验证]`

- **问题**: 门控条件是 `num_heads % 16 == 0 or 0 < num_heads < 16`(L420、L1067)。K3 TP4 → 96/4 = 24 头,24 不整除 16 且不小于 16 → gate 为 False → 回到 BF16 FMHA 解压路径。而 decode 侧的工具 `AiterMLAHelper.get_actual_mla_num_heads`(main 版)明确支持非对齐头数 pad 到 128(`is_valid_num_heads`: `num_heads <= 128 or num_heads % 16 == 0`),即 pad 技术在 decode 侧已覆盖 24 头,prefill 侧却自限于 <16。触发输入:K3、TP4、`--kv-cache-dtype fp8`,长上下文 prefill ≥ ~197k tokens。
- **影响**: 若 TP4 是目标部署形态,本 PR 声明的动机("K3 长上下文 OOM")在 TP4 上并未解决——只是把问题留在了 17–31、33–47… 等非整除区间。不是回归(回退行为与现状一致),但修复覆盖不完整。
- **行动**: 建议 review 时追问作者:TP4 是否为支持配置?若 24 头确实有硬件验证需求,建议把 prefill 的 pad 上限与 decode 对齐(复用 `get_actual_mla_num_heads` 的 128 上限),或在 PR 描述中明确声明支持矩阵(仅 TP8/TP2/TP1)。

**⚠️【性能】chunked prefill 的后续 chunk 会走 FA 回退,590k 长 prompt 的收益归因需要澄清** `[分发逻辑已验证 / 收益构成推测]`

- **问题**: `forward_mha`(L1247–1249)在 `has_context`(chunked_context 非空,即长 prompt 的延续 chunk)时无条件回退到 `super().forward_mha`——BF16 FA 解压路径,这正是本 PR 声称要消灭的 OOM 根源路径。该分发是本 PR 之前就存在的逻辑(PR 未改),但 PR 正文的核心战果"590k tokens fresh prefill 28.6s OK"是在 `--max-num-batched-tokens 4096` 下测的:590k prompt ≈ 144 个 chunk,只有第一个 chunk 是"fresh"、走 FP8 汇编路径,其余 ~143 个延续 chunk 按此分发都应走 FA 回退。
- **影响**: 若延续 chunk 确实全部回退,则 (a) 原 OOM 场景(长 context 下 FA 解压的 bf16 working set)在延续 chunk 上为何不再触发、以及 (b) 28.6s 的构成(FP8 只加速了首个 chunk?)都需要说明。存在两种可能:延续 chunk 实际以某种方式仍走 FP8 路径(则 590k 数据可信),或 FA 回退在延续 chunk 上的显存是逐 chunk 有界的(则原 OOM 的根因描述需要修正)。无论哪种,PR 描述与代码行为之间存在未解释的间隙。
- **行动**: 建议 review 时追问作者:590k 场景下延续 chunk 的实际执行路径与每 chunk 的 bf16 working set 大小;建议在 PR 描述中补充 chunked prefill 行为说明(哪怕一句话),避免后续读者误以为 FP8 路径覆盖整个长 prompt。

**📝【可维护性】builder 与 impl 的两处门控谓词语义不一致,建议收敛为单一谓词** `[已验证]`

- **问题**: builder(L420–423)用 `kv_cache_dtype_str == "fp8"`(main 已有归一化,`fp8_e4m3/e5m2` 会映射为 `"fp8"`,所以别名没问题)+ `model_config.dtype == bf16`;impl(L1067–1069)用 `is_quantized_kv_cache(kv_cache_dtype)`(任意非 "auto" 的量化 dtype,如 fp8_inc 也为真)+ **没有** bf16 检查。当前行为安全:实际分发由 builder 生成的 `fp8_prefill_qo_indptr` 元数据驱动,impl 门控为真但元数据缺失时走 FA 回退——但两个谓词可以各自漂移,未来的 dtype/量化扩展(如 fp8_inc)会悄悄制造 builder=False / impl=True 的不一致状态。
- **影响**: 无即时运行时故障;属状态一致性负债,这类双门控在本文件历史上有过教训(`use_gluon_decode` 的注释明确写了"builder 和 impl 必须看到同一个答案")。
- **行动**: 建议作者把门控提取为单一谓词(如 `_fp8_mla_prefill_enabled(num_heads, kv_cache_dtype, model_dtype)`),builder 与 impl 共用;顺带可去掉 L1065 处与 L30 重复的局部 import。

**📝【可维护性】魔数 16 散布 4 处,且同一段注释块在两处逐字重复** `[已验证]`

- **问题**: `max(16, self.num_heads)`(L487、L593)、`_pad16 = _real_nhead < 16`(L1119)、`nhead = 16 if _pad16 else self.num_heads`(L1124)——而 `AiterMLAHelper._AITER_MIN_MLA_HEADS = 16` 已存在(main 版),正是这个语义。L487 与 L593 的两段解释性注释块几乎逐字相同。
- **影响**: 未来若 kernel 对齐要求变化(如 32),需改 4+ 处,漏一处即 builder/impl 不一致。
- **行动**: 建议作者复用 `_AITER_MIN_MLA_HEADS` 常量(或加 `padded_num_heads` 属性),并把重复注释收敛到一处定义。

**📝【性能】pad 拷贝与多算的 head 计算是每 token 都要付的成本,AITER 侧 follow-up 已提出** `[作者讨论中已确认]`

- **问题**: 12→16 头意味着 q/k/v 三份 pad 拷贝(约 28 KB/token)+ 输出 slice-back(约 6 KB/token)+ 33% 的多余 head 计算。作者在讨论中已给出量化并提议 AITER 增加 `num_real_heads` 参数一次性消除全部四份拷贝。
- **影响**: 相对 OOM 回退是净收益,但 head 数越接近 16 浪费越大,对吞吐敏感场景不可忽略。
- **行动**: 记录即可——建议在 PR 描述中链接该 AITER follow-up issue 以便追踪。

## 4. 现有讨论 (Existing Discussion)

- **Rohan138(2026-08-10)**:质疑能否去掉额外输出 copy、AITER 是否可在 kernel 内做 padding+slicing;并指出 `out_3d` 动态初始化与 CUDA Graph / persistent MLA / prefix caching 的兼容性,建议改为构造函数里的静态内存。**作者回应**:copy 无法从 vLLM 侧消除(out 按 12 头打包,16 头行接不住);q/k/v 的 pad 拷贝(28 KB/token)远大于输出 copy(6 KB/token),AITER `num_real_heads` 可一次消除四份拷贝;`out_3d` 已并入 workspace `get_simultaneous`(lock 前预留最大形状 → 固定地址,无需第二套 sizing 机制,代价仅 32 MiB @8K budget 且仅 `num_heads<16` 时)。
- **AndreasKaratzas(2026-08-21)**:追问测试里为何硬编码 `torch.bfloat16`。**作者回应**顺带发现真实 bug:此前 fp16 模型 + fp8 KV 会误入该路径,bf16 位模式被当 fp16 解读——输出错数且无报错;已新增 `model_config.dtype == bf16` 门控修复。这是本 PR 最有价值的 review 产出之一。
- **hongxiayang**:流程性要求(rebase 解决冲突、补 vllm serve 命令与精度结果、随 #51011 合并后重跑),作者均已补齐。
- **CI 状态**:pre-commit 曾失败后修复;Buildkite #84031、#84484 之后,最新 #85576 于 2026-08-26 对 head `d05ffe8e` 触发,截至撰写时仍在运行,当前 head 尚无绿 CI 记录。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**

无 🔴 级正确性缺陷:跨文件验证确认了新路径的关键事实——`get_mla_padded_q` 对 3D prefill 张量有效且物化 contiguous、slice-back 数学上精确、bf16/fp8 门控闭环、workspace 预留形状与运行时请求一致、无未初始化缓冲的原子归约、无 UnboundLocalError。但 PR 的核心逻辑(pad 路径)零自动化测试覆盖、TP4 K3(24 头)场景未决、590k 长 prompt 的 chunked 收益归因与代码分发存在未解释间隙——建议补齐 12 头测试用例并对上述两点作出书面澄清后再合入。
