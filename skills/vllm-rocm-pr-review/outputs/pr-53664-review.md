# PR #53664: Add pipeline_parallel support for the kimik3 model

> **Author**: @haic0 | **State**: OPEN | **Date**: 2026-08-25
> **Branch**: `haic0:add-kimi-k3-pipeline-parallel-support` → `vllm-project:main` | **Labels**: `rocm`, `kimi`, `k3`, `kv-cache-manager`
> **Changes**: +293 -43 lines across 9 files | **ROCm 相关性**: 完全相关（Tier-1 文件 `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`，Kimi K3 + AITER + gfx950 Gluon）
> **CI 状态**: 未知（本次 fetch 未取到 check-runs 数据；PR 正文 Test plan 自述 ROCm GPU 测试被跳过）

## 1. 动机 (Motivation)

Kimi-K3 是 MLA + KDA/GDN（Mamba 类循环层）混合架构，要支持 PP2×TP4×DCP4 部署，需要三块能力：① DCP 下 attention 的 KV cache 沿序列切分到各 rank，跨 rank 合并必须拿到每个 rank 的 LSE——但 vLLM 的 `rocm_aiter_ops.mla_decode_fwd` 包装层丢弃了 AITER 原生 kernel 的 LSE 返回值；② DCP 下 query 头数被 all-gather 放大为 `num_heads × dcp_world_size`，而原代码仍按单 rank 头数做 padding 与 kernel 路由；③ KV 组几何一刀切——MLA 组应切分、Mamba/GDN 组应逐 rank 复制，原 coordinator 对所有组统一用进程级 DCP。本 PR 是 #51705 大补丁的「最小非投机 DCP 子集」，刻意排除 DSpark、投机验证、segmented MLA 与 FULL-cudagraph，以收敛审查面。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`（Tier-1，+133/-32） | DCP>1 时 Gluon 路径传 `return_lse=True`；ASM 路径绕过 vLLM 包装层直调 `aiter.mla.mla_decode_fwd` 拿 LSE；全部按汇聚头数 `_decode_num_heads` 做 padding/路由；`use_gluon_decode` 增加 `dcp_world_size` 参数并用 `inspect.getsource` 正则探测 Gluon bh16 头数上限 |
| `vllm/v1/core/kv_cache_coordinator.py` + `kv_cache_utils.py` | 新增 `dcp_world_size_for_kv_cache_spec()`：FullAttention/MLA 组用进程 DCP，Mamba 等组为 1；manager 构造与 prefix cache 查找改按组取 dcp/pcp world size |
| `vllm/model_executor/layers/attention/mla_attention.py` | 元数据 build 时若上游未提供 DCP 本地 seq_lens，则用 `get_dcp_local_seq_lens()` 推导 |
| `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | decode 卷积索引张量补 `.contiguous()` |
| 4 个测试文件 | 注册表 PP 断言（两个 Kimi 架构）、LSE 反 padding 单测、DCP 路由单测、按组 KV 几何 + prefix cache 单测 |

## 3. Review 意见 (Findings)

| 意见类型 | 数量 |
|---------|------|
| 🔴 必须修复 | 0 |
| ⚠️ 建议修复 | 5 |
| 📝 建议/备注 | 2 |

### ⚠️【测试】ROCm kernel 路径无任何 CI/单测覆盖 `[已验证]`

- **问题**: 新增的两条 `return_lse=True` 分支（Gluon `mla_gluon(..., return_lse=True)` 与直调 `_get_aiter_mla_decode()(..., return_lse=True)`）在 CI 中零执行：单测全是 CPU 逻辑测试（路由、反 padding、KV 几何），PR 正文明确「ROCm AITER routing tests…module skipped because the test container had no GPU device」；硬件验证是「PP2×TP4×DCP4…previously validated on 8 AMD GPUs with the equivalent runtime patch set」——**等价补丁集 ≠ 本 diff**，`kv_buffer.view(-1, 1, 1, D)` 的 4D 布局、`return_lse` 的实际返回形状均未被本 commit 验证。
- **影响**: 若直调签名的参数顺序/布局与 aiter 实际不符，合入后 AMD DCP 用户在运行时崩，且无回归测试兜底。
- **行动**: 建议作者在带 GPU 的 AMD CI 队列（kimi/rocm label）跑本 commit 的 E2E，或贴出 8-GPU 验证的等价补丁与本 diff 的逐文件对照说明；至少补一个 `return_lse` 契约的 mock 级测试。

### ⚠️【兼容性】aiter 上游 `return_lse` 依赖无 PR 链接与版本 pin `[已验证]`

- **问题**: PR 正文要求「AITER build whose `mla_decode_fwd` supports `return_lse=True`」，但未链接 aiter 侧 PR、未改 `requirements/rocm.txt`、无版本下限声明。老版本 aiter 下 DCP>1 的行为是运行期 assert 崩溃（报错信息本身清晰），而 `mla_gluon` 的 `return_lse` kwarg 若不被接受则是 TypeError。
- **影响**: 社区用户用现有 pinned aiter 版本跑 DCP 直接失败，无任何安装期提示；未来 aiter 上游若调整 kwarg，两条路径（包装层 vs 直调）会行为分叉。
- **行动**: 建议作者链接 aiter 侧 PR、在 PR 描述或 `requirements/rocm.txt` 注释中写明最低 aiter 版本，或做版本探测（如 `inspect.signature` 检查 `return_lse` 是否可接受）后优雅报错。

### ⚠️【正确性】DCP + 投机解码组合缺少配置层拦截，verify 分支存在静默错结果风险 `[推测]`

- **问题**: ASM decode 分支对 `max_qo_len != 1` 抛 `NotImplementedError` 是唯一防线，但该 raise 位于 verify 分支**之后**；DCP>1 且启用投机验证（Kimi K3 的 DSpark）时，先进入 verify 分支，该分支内部仍按 `self.num_heads`（未汇聚）构建 metadata，而 query 已被 DCP 层 all-gather 为 `num_heads × dcp_world_size`——头数不匹配，若形状恰好凑上则每个 rank 用自己的 KV 分片算局部 attention 且无 LSE 合并，输出静默错误。diff 中未见 DCP 与投机解码互斥的配置层校验（需作者确认上游 model_runner/config 层是否已有守卫）。
- **影响**: 用户以 DCP>1 启动且投机解码开启（或默认开启）时，可能得到错误输出而非报错。
- **行动**: 建议作者确认上游是否存在「DCP × spec decode」守卫；若无，在 engine 配置校验处显式拒绝该组合（而非依赖运行时 NotImplementedError），并在 PR 描述中写明该限制。

### ⚠️【可维护性】`inspect.getsource` + 正则探测 aiter 头数上限脆弱，静默回退掩盖版本差异 `[已验证]`

- **问题**: `_gluon_mla_max_bh16_heads()` 依赖 aiter 包装函数源码中出现字面量 `requires nhead <= (\d+)`；aiter 改文案、改 assert 措辞、或包装层编译为 C 扩展时探测失败，静默回退到 16（`_AITER_MIN_MLA_HEADS`）。且 `max(fallback, int(match))` 会无视探测到的更低上限。回退只影响性能（Gluon → ASM），不伤正确性，但没有任何日志暴露「探测失败」这一事实。
- **影响**: aiter 升级后 DCP 场景悄悄从 Gluon 掉到 ASM 路径，性能回退难以归因；若 aiter 真实上限低于 16 则会触发 kernel 内 assert。
- **行动**: 建议作者在探测失败时记录一次 warning 日志（说明回退值），并考虑改用 aiter 提供的版本化 capability API（若存在）或环境变量覆盖。

### ⚠️【正确性】Gluon 路径 LSE 直接 `reshape`、无 unpadding、无单测 `[推测]`

- **问题**: ASM 路径对 LSE 做了 `get_mla_unpadded_lse()` 反 tile-padding（有单测覆盖），但 Gluon 路径是 `lse.reshape(B, num_q_heads)` 直接返回——隐含假设 aiter Gluon kernel 的 `return_lse=True` 恰好返回 `B × num_q_heads` 个元素且头序与 `o` 一致。该契约未验证且无任何测试。若 Gluon 返回 padded LSE（如 tile 到 16 头），reshape 元素数不符会 RuntimeError（响亮崩溃）；若元素数巧合吻合但头序不同，则 DCP 合并静默错值。
- **影响**: 8 头 × DCP2=16 头场景（16 % 16 == 0，Gluon 路由可达）下存在静默错误的可能性窗口。
- **行动**: 建议作者对照 aiter Gluon `return_lse` 实现确认返回布局，并为 Gluon 路径补一个与 ASM 路径对称的 unpadding + 头序单测（哪怕用 mock）。

### 📝【注释/文档】精度数字与并行配置不一致 `[已验证]`

- **问题**: 正文标题与 Test plan 宣称 PP2×TP4×**DCP4**，但 GSM8K 日志目录为 `pp2-tp4-dcp2-gsm8k-5shot`——0.9545 的 exact_match 是在 **DCP2** 下测得，DCP4 仅有「serving 验证通过」无精度数据。
- **行动**: 建议作者在 PR 描述中明确标注精度验证的并行配置，或补充 DCP4 的精度结果。

### 📝【可维护性】`use_gluon_decode` / `use_gluon_verify` 签名分化无注释 `[已验证]`

- **问题**: decode 路由函数新增 `dcp_world_size` 参数而 verify 路由函数保持 3 参数（仍用 `self.num_heads`），两者行为边界（verify 不支持 DCP）只在 PR 正文体现、代码内无说明；`_build_decode` 中两处调用并存，后续维护者容易误改。
- **行动**: 建议作者在 `use_gluon_verify` 处加一行注释说明「verify 路径不支持 DCP（见 #51705 范围）」，或将 DCP 检查提前。

## 4. 现有讨论 (Existing Discussion)

- 0 条 issue comment、0 条 inline review comment。
- `claude[bot]` 自动 review：仅注明「PR 来自 fork，自动 review 默认关闭，维护者可评论 `@claude review` 触发一次性审查」。
- 16 位 reviewer 被请求（含 WoosukKwon、DarkLight1337、njhill、LucasWilkinson、tdoublep、robertgshaw2-redhat、ywang96 等），尚无任何人工 review 或 approval；`mergeable_state: unstable`（需要 rebase）。
- 与 #51705 的重叠关系由作者主动声明（本 PR 为最小非投机子集），是合理的拆分策略，但后续两个 PR 合入顺序会互相影响 rebase。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**

核心设计方向正确且非 DCP 路径行为严格保持等价（已逐条件核对 `use_gluon_decode` 的 DCP=1 分支），无已证实的具体触发输入能导致静默数值错误；但新 kernel 路径（两条 `return_lse=True` 分支）在 CI 中零执行、aiter 上游依赖未 pin 无链接、DCP×投机解码组合缺少配置层确认——建议作者补齐上述三项后合入。
