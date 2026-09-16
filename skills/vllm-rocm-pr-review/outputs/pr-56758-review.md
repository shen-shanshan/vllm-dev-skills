# PR #56758: [Scheduler] Add --max-num-active-seqs to cap RUNNING admission

> **Author**: @ChuanLi1101 | **State**: OPEN | **Date**: 2026-09-14
> **Branch**: `ChuanLi1101:chuanli/max-num-active-seqs` → `vllm-project:main` | **Labels**: `rocm`, `scheduler`
> **Changes**: +51 -1 行，5 个文件 | **ROCm 相关性**: 部分相关（rocm 标签 + AMD reviewer，动机来自 ROCm 部署；但改动文件均为平台无关的调度器/配置代码，未触及任何 ROCm 专属路径）

---

## 1. 动机 (Motivation)

`max_num_seqs` 目前身兼两职：既决定 model runner 的 per-request 缓冲区与 CUDA Graph 捕获档位（执行容量），又限制调度器从 WAITING 准入 RUNNING 的请求数（准入上限）。想在保持大容量图捕获的同时运行更小的 decode batch（ROCm 部署诉求：为小 batch 调小 `max_num_seqs` 需改变捕获档位、缩小静态缓冲区，各轮调优实验执行环境不一致且容量被永久缩小），当前无法实现。本 PR 新增可选参数 `--max-num-active-seqs`，只降低准入上限，runner/图容量仍按 `max_num_seqs` 配置；默认 `None` 时行为与现状完全一致。作者同时明确区分了它与 `--max-num-queued-reqs`（API 层跨 DP rank 的排队上限，超限 503 拒绝）——本参数只在单 engine 内限制 WAITING → RUNNING 的准入时机，不拒绝请求。

## 2. 代码改动总结 (Change Summary)

| 文件 | 说明 |
|------|------|
| `vllm/config/scheduler.py` | 新增 `max_num_active_seqs` 字段（`int \| None`, `ge=1`）；`verify_max_model_len()` 中校验必须 `<= max_num_seqs`，否则 `ValueError` |
| `vllm/engine/arg_utils.py` | `EngineArgs` 新字段、`--max-num-active-seqs` CLI 参数、`create_engine_config()` 透传 |
| `vllm/v1/core/sched/scheduler.py` | 新增 `max_num_active_reqs`（`None` 时回退 `max_num_running_reqs`）；准入循环 break 条件由 `max_num_running_reqs` 改为 `max_num_active_reqs`（唯一运行时行为变更点，+8 -1） |
| `tests/v1/core/test_scheduler.py` + `utils.py` | 新增单测：cap=3 时 10 请求仅准入 3、7 留 WAITING、runner 槽位仍为 16 |

## 3. Review 意见 (Findings)

意见类型 × 数量：⚠️ 建议修复 ×2，📝 建议/备注 ×4，🔴 0。

**交叉验证说明**（对照本地 vllm 代码树 @ f6b055425f，≥ PR base dc36fcce）：已确认 `self.running.append()` 全文件仅一处（scheduler.py:1242，位于主准入循环内），cap 覆盖全部准入路径，无旁路；`async_scheduler.py` 无独立准入循环；`HiSparseConnectorScheduler` 仅是 KV connector 的元数据构建器，不直接准入请求；`verify_max_model_len` 由 `SchedulerConfig.__init__` 调用，`ValueError` 校验可达。aiter/mori 专属规则不适用（无相关改动）。

---

**⚠️【设计】暂停流式会话计入 active cap，小 cap 下可能饿死新请求** `[已验证]`

- **问题**：准入循环的计数口径为 `num_running = len(self.running) + self.num_waiting_for_streaming_input`（scheduler.py:866-867），PR 只是把比较阈值换成了 `max_num_active_reqs`。`WAITING_FOR_STREAMING_REQ` 的暂停会话（等待用户下一段输入的聊天中间态）不参与计算，却仍占用 cap 名额。触发输入：`--max-num-active-seqs 16` + 10 个暂停的流式会话 → 最多只有 6 个请求能真正运行，即使 `max_num_seqs=128` 意味着 118 个 runner 槽位空闲。
- **影响**：交互式多轮流式负载下，新请求长期滞留 WAITING，服务表现为"卡住"——与用户"只限制 decode 计算量"的初衷相悖（暂停会话不产生计算）。该计数在旧语义下无感（暂停会话本就占 runner 槽位），但小 cap 使其成为显性瓶颈。
- **行动**：建议作者在 PR 描述/文档中明确此计数语义；或与 reviewer 讨论是否只以 `len(self.running)` 作为 active cap 的计数口径（runner 槽位上限仍按原口径）。

**⚠️【流程】DCO 检查 action_required，阻塞合入** `[已验证]`

- **问题**：head commit `063722a` 的 CI 检查中 DCO 为 `action_required`——commit 缺少 `Signed-off-by`。
- **影响**：vLLM 强制要求 DCO，当前状态无法合入。
- **行动**：作者应当对 commit 补充签名并推送更新。

**📝【注释/文档】PR 描述 "without recapturing graphs" 措辞有误导性** `[已验证]`

- **问题**：CUDA Graph 在**每次引擎启动时都会重新捕获**（`capture_model()`，档位由 `max_num_seqs` 派生，见 `vllm/config/vllm.py:2157`）。改 `max_num_active_seqs` 重启同样要付完整捕获时间，该参数并不省启动成本；其真实含义只是"捕获的图集内容与 buffer/KV 布局无需改变"。同理，cap 生效期间 RUNNING 无法超过 cap——大容量不是自动可用的突发余量，而是"改 flag 即可释放"的储备。
- **行动**：建议作者修正描述措辞，动机部分强调真实收益：调优实验的执行环境一致性（干净 A/B 对比）与配置语义解耦（容量 vs 策略），而非节省捕获时间。

**📝【注释/文档】新参数无任何文档**

- **问题**：`docs/` 未更新。`max_num_seqs`（容量 + 准入）、`max_num_active_seqs`（准入）、`max_num_queued_reqs`（API 层排队上限，503）三者语义相近，容易误配。
- **行动**：建议作者在引擎参数文档中补充本参数，并明确与另两者的区别（PR 描述中的对比文字很适合搬进文档）。

**📝【测试】边界用例缺失**

- **问题**：现有单测只覆盖了基本 cap 行为。缺：(a) 暂停流式会话计入 cap 的行为（即上述 ⚠️ 场景）；(b) `max_num_active_seqs > max_num_seqs` 触发 `ValueError`；(c) 调低 cap 后"无抢占、随请求完成收敛"的语义。
- **行动**：建议作者补充 (a)(b) 两个低成本用例——(a) 恰好能把设计语义固化成测试。

**📝【可维护性】引擎状态仍上报原始 `max_num_seqs`**

- **问题**：`vllm/v1/engine/core.py:1665` 的 `EngineStatus` 上报 `scheduler_config.max_num_seqs`。启用 cap 后，外部观测工具（控制台/监控）会以为准入容量是 `max_num_seqs`，而实际生效上限是 `max_num_active_seqs`。
- **行动**：可选——建议在状态中额外暴露生效的准入上限，避免排障误判。

**📝【兼容性】PD 分离部署共享配置时，prefill 侧同样受限**

- **问题**：参数是 engine 级（每进程一份 `SchedulerConfig`）。PD 分离（`kv_role=producer/consumer`）是双进程部署，decode 实例可单独设小值——但若两个实例共用同一份启动配置模板，prefill 实例的准入并发也会被同一个小 cap 限制，而 prefill 需要高并发来维持 KV 产出。
- **行动**：建议在文档中提示"该参数应只配置在 decode 实例上"。

**CI 说明**：本 PR 无 AMD 硬件 CI 运行；但改动为平台无关的准入策略，CUDA CI + CPU 单测覆盖可接受。另注：PR 未作性能声明（正确——这是行为参数而非优化），ROCm 侧"避免 hipGraph 重捕获"的收益也没有基准数据支撑，属动机陈述而非实测结论。

## 4. 现有讨论 (Existing Discussion)

- **@Fangzhou-Ai**：`Thanks @ChuanLi1101 LGTM! cc @shen-shanshan` —— 已 LGTM，等待 assignee（@shen-shanshan）继续推进。
- **claude[bot]**：fork PR 自动 review 被禁用，提示维护者可评论 `@claude review` 触发一次性 review。
- 无行内 review comments，无设计争议。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK** —— 核心逻辑经交叉验证无正确性问题（准入路径唯一、校验可达、默认行为完全向后兼容），但 DCO 签名缺失是硬性合入门槛；建议同时明确暂停流式会话的计数语义并补充文档与两个低成本测试用例。修复 DCO 后即可进入合并流程。
