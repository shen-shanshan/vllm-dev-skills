# vLLM PR #61834（`--dp-prefill-delayer`）目前不能合入的原因分析

> PR：[vllm-project/vllm#61834](https://github.com/vllm-project/vllm/pull/61834) `[DARFT][Core][DP] Add an opt-in cross-DP prefill delayer (--dp-prefill-delayer)`
> 作者：@benenzhu | 状态：**Draft（OPEN）** | Labels：`scheduler` | 规模：+368 −4，6 个文件
> 整理日期：2026-11-04
> 相关文档：[PR 总结](../../../skills/vllm-pr-summary/outputs/pr-61834-summary.md)、[ATOM PrefillDelayer 技术报告](../../../skills/vllm-feature-tutorial/outputs/atom_prefill_delayer.md)

---

## 0. 结论先行

截至 2026-11-04，**还没有任何 vLLM maintainer 在这个 PR 上留下 review 或明确的拒绝意见**（reviews 为空；CODEOWNERS 中的 WoosukKwon、robertgshaw2-redhat、njhill 等都还在待 review 状态）。所以本文说的"不能合入"，并不是 maintainer 已经给出的结论，而是对照下面这些依据，归纳出它**现在还达不到合入门槛**的原因：

1. vLLM 的贡献规范：`AGENTS.md`、`.agents/skills/pr-checklist/SKILL.md`、`.github/PULL_REQUEST_TEMPLATE.md`、`docs/contributing/README.md`；
2. PR 自身的状态和评论区讨论（@functionstackx、@maslan-coop26、作者本人）；
3. **同类 PR #51199 已经收到的 maintainer 意见**，这是最直接的先例。

| # | 原因 | 类别 | 严重程度 | 是否阻塞 |
|---|------|------|---------|---------|
| 1 | 作者自己标为 Draft，标题前缀写成 `[DARFT]`，checklist 全部没勾 | 流程 | 高 | **阻塞**（Draft 不会进入 review） |
| 2 | 这个分支本身没有 GPU 端到端验证，Claims 用的是"早期开发分支"的数据 | 验证 | 高 | **阻塞** |
| 3 | 没有证明比现有 `--prefill-schedule-interval` 好到值得新增一套机制（低并发下反而更差） | 设计取舍 | 高 | **阻塞**（maintainer 在 #51199 上提过同样的要求） |
| 4 | TTFT 劣化严重，而且没有基于时间的上界 | 设计取舍 | 高 | **阻塞**（作者自己也承认） |
| 5 | 和 #51199 解决的是同一个问题，存在重复 / 竞争 | 社区规范 | 中 | 需要先协调 |
| 6 | 改动了核心组件（scheduler、engine core busy loop），新增了每步一次的集合通信，增加了 hang 风险 | 设计 | 中 | 需要 maintainer 认可设计 |
| 7 | 实现仍在变化：ATOM 的关键修正还没移植，后续功能还在陆续加入 | 成熟度 | 中 | 需要先收敛范围 |
| 8 | 测试只有 CPU 单测，没有多 rank DP 测试，也没有 CI 覆盖说明 | 测试 | 中 | 需要补齐或说明 |
| 9 | PR 描述不符合 AI 辅助贡献的规范（没运行 `/pr-checklist`，checklist 没勾选，benchmark 不够严谨） | 社区规范 | 中 | 需要修正 |

---

## 1. Draft 状态与 PR 流程问题

- **作者自己标了 Draft**：标题是 `[DARFT]`（`DRAFT` 的拼写错误），GitHub 状态也是 Draft。vLLM 的 maintainer 一般不会 review Draft PR。在 #51199 上，@xinli-sw 的第一条意见就是"先转成 ready for review"。
- **作者明确说 PR 处于早期阶段**（评论原话："This PR is at an early stage. Our next goal is to lower TTFT…"），并且列出了 3 项还没完成的后续工作（见第 4、7 节）。
- **PR checklist 一项都没勾**，包括：
  - `I used vLLM's /pr-checklist skill. (Mandatory for agents, optional for humans)`：正文写了 "AI assistance (Claude Code) was used for this PR"，所以按 `AGENTS.md` 的要求，这一项是**必须做的**（"Before opening a PR (drafts included) … run the pr-checklist skill and address its findings"）；
  - `AI assistance was used…`：正文里写了，但没有勾选；
  - Design Fit / Testing and Validation / Code Quality / Pull Request Contents 四项也都没有勾。
- 期间 mergify 提示过 merge conflict。作者是通过 merge main 解决的，没有 rebase。

## 2. 验证不足：这个分支没有在 GPU 上跑过

PR 的 Validation 一节写得很清楚：

> GPU: no run of this branch yet.
> Measured on a development branch with an earlier version of this change … single runs per configuration.
> Validation on this upstream branch is still to be added.

对照 `pr-checklist` 第 2 节和第 4.2 节，有以下几个缺口：

| 要求（pr-checklist） | 现状 |
|---|---|
| "new features or performance optimizations must be run locally and/or benchmarked" | 当前分支**没有**在 GPU 上跑过；性能数据来自早期版本，之后合入的 #6（只 HOLD 有 decode 的 rank）、`cpu_all_reduce_`、fault tolerance 检查都改变了行为 |
| "For repeated benchmarks, report the run count and variability" | 大部分配置只跑了 1 次；c64 baseline 是 2 次的平均，**没有给出方差** |
| 对比要可复现、条件要一致 | c128 的 baseline 和实验组**跑在不同节点**（D vs C），而且 recipe "with other changes" |
| "reproduction commands" | 没有给出 `vllm serve` 和 benchmark 的完整命令 |
| "Model evaluation results when the change affects … serving"（AGENTS.md） | 改动只影响调度、不改数值，但没有任何 accuracy smoke test（例如 gsm8k）证明调度变化不影响输出 |
| 投机解码 / V2 model runner 等组合 | 只有第三方（@maslan-coop26）在 DSv4.1-Flash + V2 runner + DSpark 上跑过，**而且跑出过 hang**，修复之后在 GPU 上的验证也是第三方完成的 |

## 3. 没有证明比现有 `--prefill-schedule-interval` 更值得（设计取舍）

这是**最核心的一点**。vLLM 已经有 `--prefill-schedule-interval`（#44558），以及它在 DP=1 下的本地版本（#54627）。新增一套平行机制，首先要回答"为什么现有机制不够用，以及新机制的收益是否抵得上它增加的复杂度"。

### 3.1 先例：maintainer 对 #51199 的意见

#51199（@kjiang249，Adaptive Prefill Delayer for DP，同样是 Draft，标签 `needs-rebase`）是解决同一个问题的另一个 PR。@njhill 在 2026-08-07 的意见可以直接套用到 #61834：

- 现有的 `--prefill-schedule-interval` 是否已经能解决这个问题？
- 需要 **e2e performance benchmark comparisons**；
- 这个 PR "**adds multiple extra tuning knobs and additional complexity to the scheduler**"。

作者后来只给了一组和 SGLang 的横向对比（42,276 vs 27,047 tok/s），没有和 vLLM 自己的 `prefill-schedule-interval` 做同条件对比。这个 PR 到现在也没有合入。

### 3.2 #61834 在这一点上的现状

- **@functionstackx** 提了同样的问题："is there any throughput vs e2el improvement? won't `--prefill-schedule-interval` already solve the issue"，并附上了官方 DeepSeek-V4-Pro recipe 中的调度建议链接。
- 作者的回复数据（DSv4-Pro，8×MI355X，同节点）：

  | | c64 interval=32 | c64 delayer | c128 interval=32 | c128 delayer |
  |---|---:|---:|---:|---:|
  | Output tok/s/GPU | 158 | 145（**−8%**） | 189 | 350（+85%） |
  | E2EL mean (s) | 34.1 | 40.9（**+20%**） | 68.3 | 40.4（−41%） |
  | TTFT mean (s) | 5.0 | 25.3（**+406%**） | 10.0 | 10.7 |
  | E2E-normalized interactivity P90 | 9.5 | **3.3** | 4.5 | 6.9 |

- **作者自己承认**："this PR is overfitting to optimize only on throughput vs interactivity P90 now"，并认为 E2E-normalized interactivity 更重要。
- **对比基线只有 interval=32 这一个点**。作者把 "A sweep against interval 8/16 (with and without the long-prefill threshold) across concurrency" 列为**还没做**的下一步。没有这组 sweep，就无法排除"把 interval 调小就能拿到同样收益"的可能。

> **小结**：目前的证据只能说明"在 c128 + interval=32 这一个点上更好"，同时在 c64 上更差。这还不够支撑"给核心 scheduler 增加一套新机制"。

## 4. TTFT 劣化，而且没有时间上界

- HOLD 的上界按**步数**计算（`TTFT_MAX_HOLD_STEPS=200`、`PARTIAL_MAX_HOLD_STEPS=100`），而且 protected steps 期间的 HOLD **不计入** `_hold_steps`，所以实际等待的墙钟时间和每步耗时相关，**没办法对应到 TTFT SLA**。
- 作者在 PR 正文中写明了这个代价："The trade-off is TTFT: prefill waits for the other ranks, and long prompts wait for alignment chunk by chunk."
- 作者在最新回复中也认为：基于等待时长的上界 "is the TTFT bound this PR is missing … so it belongs in this PR"。也就是说，**作者本人认为这个 PR 还缺一个必要组件**。
- 与之对照，ATOM 原版有 `ATOM_PREFILL_DELAYER_MAX_QUEUE_MS`（按请求到达时间计算排队时长、lockstep 安全）、cost gate、unreachable 判定和 adaptive 策略，这些都是用来控制 TTFT 的，移植版一个都没有。

## 5. 和 #51199 重复 / 竞争

`AGENTS.md` 的 Duplicate-work checks 要求：

> If an open PR already addresses the same fix, do not open another. If your approach is materially different, explain the difference in the issue.

- #51199 早于 #61834，同样是 Draft，同样针对"DP 各 rank 的 prefill 对齐"这个问题。
- #61834 在正文 Details 里用一段文字和 CPU 仿真结果说明了两者的区别（同步 all-reduce vs 通过 coordinator 异步协调；这里的 mixed step 少 2–4 倍，但 #51199 的 TTFT 低约 3 倍），并 cc 了 @kjiang249，表示 "The two are complementary, and I am happy to merge efforts"。
- 但是：
  - 规范要求在 **issue** 中说明两者的区别，而不只是写在 PR 正文里；
  - 两个 PR 都没有和 maintainer 对齐过"社区要哪一种"；
  - 从 maintainer 的角度看，**两个 Draft PR 同时给 scheduler 增加相似但互不兼容的机制**，合理的做法是先确定方向（比如开一个 RFC 或 issue 讨论），而不是分别 review。

## 6. 改动核心组件，并新增每步一次的集合通信

`pr-checklist` 第 1.1 / 1.3 / 1.4 节要求：核心组件（`vllm/v1/core/sched/*`、engine core）的改动要尽量少侵入，非默认功能的非平凡逻辑不应该放进公共路径。

| 关注点 | 现状 |
|---|---|
| 侵入范围 | 改了 `Scheduler.schedule()` 中 `defer_prefills` 的公共条件、在 schedule 末尾新增 `ran_prefill` 判定、在 `DPEngineCoreProc.run_busy_loop` 中新增每步一次的 all-reduce，以及 wave reset、`_should_throttle_prefills` 分支 |
| 复杂度 | 主代码约 230 行（不含测试），属于 pr-checklist 中的"Medium complexity（50–250 行）"上限附近。加上作者计划补的 TTFT 上界、统计日志、DP 代价均衡等功能后，会超过 250 行 |
| 集合通信安全 | 新 collective 要求"所有 rank 每步都执行到这里"，这个不变量目前**只靠注释保证**。之前已经出过一次 hang（没有 decode 的 rank 被 HOLD 后调度空 batch，见 benenzhu/vllm#6），说明这条路径很脆弱，而 CI 中**没有**多 rank 测试能防止回归 |
| 兼容性 | 显式拒绝了 elastic EP、fault tolerance（#46370 的 scale-down）、自定义 scheduler 类，并且和 `prefill_schedule_interval` 互斥，相当于又多了一组需要维护的互斥矩阵 |
| 状态耦合 | `local_state()` 读取后会清零 `scheduler.ran_prefill`（读操作有副作用）；`ran_prefill` 的判定用的是启发式规则 `n > 1 + len(spec_tokens)` |
| 异步调度 | 决策基于"上一步执行后的状态"，作用于下一次 schedule。在 async scheduling 下的时序没有分析，也没有测试 |
| 硬编码常量 | 0.9 填充率、200/100/10 步、0.9 KV 阈值都不能配置；但如果改成可配置，又会碰到 @njhill 批评的 "multiple extra tuning knobs" |

## 7. 实现还没收敛，范围还在变

- 最近一轮更新（2026-10-31）合入了第三方修复（#6）、`cpu_all_reduce_`、拒绝 fault tolerance。作者还邀请 @maslan-coop26 继续往 `dp-prefill-delayer` 分支提交两个新功能：
  1. 每 2000 步输出一次 FIRE/HOLD 统计日志；
  2. ATOM 风格的等待时长上界和 DP 间 prefill 代价均衡。

  作者的态度是"I will fold them in here"。**PR 的范围还在扩大**，maintainer 很难在这个时候开始 review。
- 和 ATOM 最新实现相比，移植版缺少 ATOM 在 #2727 中修掉的若干问题：
  - stall 判定只看 `pending <= prev_pending`，而 pending 被截断到 budget。在闭环负载下，pending 不变不代表没有新工作，ATOM 曾因此在 CI 中**误放行 218 次**未对齐的 prefill；
  - 没有 `blocked`（head 分不到 KV）和 `unreachable`（在 TTFT 预算内无法对齐）的区分；
  - 没有新到达请求的信号。

  这些都可能在 review 阶段被指出，或在部署后表现为碎片化 prefill 和 TTFT 抖动。

## 8. 测试覆盖不足

- 新增的 `tests/v1/core/test_prefill_delayer.py` 全部是 CPU 单测：决策状态机、`local_state`、单 scheduler 的 HOLD 门控、配置互斥。
- **缺少的部分**：
  - 多 rank DP 测试：最关键的 lockstep 和 collective 不变量没有被测到；
  - 和 async scheduling、投机解码、V2 model runner 组合的测试（这正是 hang 出现的场景）；
  - `pr-checklist` 第 2.3 节要求：CI 覆盖不到的路径，要**写明缺少哪些覆盖、受什么资源限制、在 CI 之外做了哪些验证**。PR 中没有这段说明。
- 此外，`test_local_state` 使用了 `SimpleNamespace` 模拟 scheduler，而 pr-checklist 2.1 节提倡 "Avoid using mock objects, especially when highly coupled with internal logic"。

## 9. PR 描述不符合 AI 辅助贡献的规范

`AGENTS.md` 对 AI 辅助的 PR 描述有强制要求，逐项对照如下：

| 要求 | 现状 |
|---|---|
| 说明为什么不是重复 PR | 正文 Details 中说明了和 #51199 的区别，基本满足，但没有在 issue 中说明 |
| 写出运行过的测试命令和结果 | pytest / pre-commit 命令和结果都有，满足 |
| 影响 serving 时给出模型评估结果 | **缺失** |
| 明确声明使用了 AI 辅助 | 正文有声明，但 checklist 没勾 |
| 运行 `/pr-checklist` 并处理其发现的问题（Draft 也需要） | **没有证据表明运行过** |
| 用户可见的改动（新 CLI 参数、config 字段）要更新渲染文档 | config docstring 和 CLI 帮助会自动生成参考页，基本满足；但没有 feature 级别的使用文档，没有说明适用场景、和 interval 怎么选、TTFT 代价，也没有 `examples/` |

---

## 10. 要推动合入，建议补齐的工作

按优先级排列：

1. **先和 maintainer 对齐方向**：开 issue 或 RFC，把 #51199、#61834 和现有 `prefill-schedule-interval` 放在一起讨论，确定社区要哪种机制，或者把两者合并（作者已表示愿意合并）。
2. **补一组同条件的端到端对比**：
   - 同一节点、同一 recipe、多次运行并给出方差；
   - 基线至少包括 interval ∈ {1, 8, 16, 32}，各自带/不带 long-prefill threshold；
   - 并发覆盖低、中、高（如 c16 / c64 / c128 / c256）；
   - 指标：throughput vs E2E-normalized interactivity、throughput vs TTFT，ITL-only 的 P90 只作为参考。
3. **补上 TTFT 的时间上界**（类似 ATOM 的 `MAX_QUEUE_MS`），并证明加上它之后在低并发下 TTFT 和 E2E 指标不再明显变差，或者给出"只在某个并发以上开启"的明确建议。
4. **在当前分支上做 GPU 验证**：至少 DSv4-Pro / DSv4.1-Flash DEP8，覆盖 V1/V2 model runner、async scheduling、MTP/投机解码；附上一次 accuracy smoke（如 gsm8k），以及完整的复现命令。
5. **收敛范围**：明确这个 PR 包含什么、哪些功能放到 follow-up PR（统计日志、代价均衡）；同时移植 ATOM #2727 中的 stall 和到达信号修正。
6. **降低对核心路径的侵入，加强不变量保护**：
   - 考虑把 delayer 的 collective 和 `_has_global_unfinished_reqs` 的 all-reduce **合并成一次**，这样既省一次通信，也从结构上保证不会被单个 rank 跳过；
   - 把 `ran_prefill` 读后清零的副作用改成显式接口；
   - 增加一个多进程 DP 的 CPU/gloo 测试，覆盖 lockstep 不变量。
7. **把 PR 描述整理到规范要求**：
   - 运行 `/pr-checklist` 并勾选；
   - 修正标题拼写；
   - 补 feature 文档，说明它是什么、怎么开启、默认值、和 `--prefill-schedule-interval` 怎么选、TTFT 代价；
   - 补上 CI 覆盖缺口的说明；
   - 准备好后转为 Ready for review，并 @ scheduler 的 code owner。

---

## 附：信息来源

- PR #61834 正文和评论：@functionstackx（10-15）、@benenzhu（10-16、10-31）、@maslan-coop26（10-26、10-29），mergify（10-15）
- PR #51199 及 @njhill、@xinli-sw 的意见（2026-08-07）
- vLLM 仓库：`AGENTS.md`（Contribution Policy）、`.agents/skills/pr-checklist/SKILL.md`、`.github/PULL_REQUEST_TEMPLATE.md`、`docs/contributing/README.md#documenting-user-visible-changes`
- ATOM 仓库：`atom/model_engine/prefill_delayer.py`，以及提交 #1611、#2016、#2727、#2897
