# 面向 Agentic Workload 的大模型推理优化

## 引言

## 什么是 Agentic Workload？

Agent 是怎么工作的？（画图说明，尤其需要展示出过程中不同阶段 kv cache（token 上下文）长度的变化）

1. 用户输入一个 prompt（比如：帮我实现 XXX 功能）
2. Model call（在 GPU 上执行）：LLM 执行推理（45.7k tokens）
3. Tool call（在 CPU 上执行）：理解当前项目
4. Model call（在 GPU 上执行）：LLM 执行推理（48.4k tokens = history + 少量 new tokens）
5. Tool call（在 CPU 上执行）：写代码
6. ……（中间省略 N 轮）
7. Model call（在 GPU 上执行）：LLM 执行推理（108.5k tokens = history + 少量 new tokens）
8. 分裂（fan out）出 sub-agent（同时，主 LLM 等待 sub-agent 执行完成）
   1. sub-agent 1：
      1. Model call（在 GPU 上执行）：LLM 执行推理（10k tokens：上一轮主 LLM 提供的信息）
      2. Tool call（在 CPU 上执行）：查阅技术文档（29.4k tokens）
      3. 返回结果
   2. sub-agent 2：
      1. Model call（在 GPU 上执行）：LLM 执行推理（10k tokens：上一轮主 LLM 提供的信息）
      2. Tool call（在 CPU 上执行）：执行测试（59.3k tokens）
      3. 返回结果
9. 主 LLM 收到各个 sub-agent 返回的结果
10. Model call（在 GPU 上执行）：LLM 执行推理（141.5k tokens = 108.5k tokens + sub-agent 返回的 new tokens）
11. agent 总结当前进展并通知用户
12. 用户根据 agent 返回的结果输入下一步指令（新的 prompt）
13. ……（如此循环）

特点：
1.一个任务是多轮迭代、多个模型共同完成的（某些 sub-agent 可能会用一些更快、更轻量的模型去执行简单的任务）。
2.每一轮对话，都会带上上一轮对话的全部上下文，历史信息越滚越多。
3.GPU 是忽忙忽闲的：agent 在执行 tool call 时，GPU 空闲；agent 同时 fan out 大量 sub-agent 时，多个 LLM 并行推理，出现 GPU 负载洪峰。

挑战：
1.每一轮 LLM 推理，都需要计算（prefill）大量的、重复的历史上下文，需要通过 prefix cache 尽量减少重算 kv（如果全部重算，TTFT 会变得很久，用户体验差），因此 kv cache hit 率这个指标变得非常重要。prefill 完成后，LLM 开始 decode（吐字），这时 interactivity（output tokens/s per user）这个指标很重要。
2.agent 同时 fan out 大量 sub-agent 时，多个 LLM 并行推理，GPU 的计算压力突然变得很大（大并发，流量激增，此时是 compute bound）。注意：在 sub-agent 执行时，主 LLM 在等待；在 sub-agent 全部执行完成之前，此前主 LLM 已经算出来的 kv cache 还不能删掉（keep it warm），因为最后主 LLM 汇总 sub-agent 返回的结果并继续执行推理时，还需要复用之前的上下文（否则就需要重算，开销很大）。要求 kernel 的泛化性要好，在大、小并发的场景下性能都要好。

参考 https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html 中的 What Agentic Applications Demand 这一节，介绍下面对 Agentic Workload 我们应该关注什么？

## Agentic 推理优化的方法有哪些？

显存优化：

1.prefix reuse (kv offloading: lmcache / mooncake / umbp(AMD))，这里需要参考 https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html 简单介绍下什么是 UMBP，以及 UMBP 在 agentic workload 下的优势。
画图说明：GPU HBM / CPU DRAM / NVMe SSD 三级 KV 缓存架构，以 AMD MI355X GPU 为例，包括：各级的容量和带宽分别是多少？级间的传输方式（协议）分别是什么？

2.DCP (each rank keeps 1/8 of MLA KV)，这里需要参考 https://vllm.ai/blog/2026-08-07-decode-context-parallelism 解析下为什么纯 TP MLA 会复制 KV？对比下 TP 和 DCP 各自的优劣？为什么 agentic workload 需要 DCP（缓解 KV 缓存压力，agent 推理可以少算 kv，通信换显存）？画图说明。
再简要介绍下 TP8 + DCP8 部署模式下，DCP 的通信流程（gather all q heads → attention on each gpu's 1/8 kv -> all-to-all merge outputs）画图说明。

计算优化：

1.mono decode kernel → 为什么主要用于小并发场景？（极致 fusion 解决 memory-bound，大并发时已经是 compute-bound）
介绍下什么是 mono kernel？每个 layer 只有两个 kernel，分别是什么？为什么需要 mono kernel？解决的是 agent 推理的哪个场景？怎么设计和实现的？画图说明。

调度优化：

1.`--prefill-schedule-interval` 参数（delay prefill, hold then batch），这里需要参考 https://vllm.ai/blog/2026-09-08-vllm-agentx#scheduling-mixed-agentic-traffic-at-two-levels 介绍下 vLLM 中的这个参数是用来做什么的？为什么需要它（为什么组 batch 做一次 prefill 比分开多次做单个 prefill 好）？解决的是 agent 推理的哪个场景？优化的是什么指标？画图说明。

## 怎么评估 Agentic 推理的性能？

推理引擎在 Agentic Workload 下的推理性能怎么评估？——InferenceX AgentX

简单介绍下这个跑分榜单怎么看？顺便介绍下什么是 pareto frontier（帕累托曲线）？怎么看一个曲线的好坏？（画图说明）。

某个曲线越在右上角越好。
1.向上：more tokens per GPU（挑战：keep and reuse more history kv）
2.向右：faster for each user（挑战：调度和计算）
一条曲线代表：某个推理引擎（比如：vLLM、SGLang）跑某个模型（比如：DeepSeek-V4-Pro），在不同并发点下的性能连线。

核心指标：
1.用户体验（横坐标）：P90 interactivity（即大部分用户感受到的推理速度）
2.硬件厂商（纵坐标）：更关注成本（单位美元、单位时间能提供多少 token），因此更关注整个服务器的吞吐（即 total token per 1$ TCO）

> 解释 TCO 是什么的缩写？代表什么意思？

简单介绍下 InferenceX AgentX 用来评测的 cc 数据集的特征。
单个请求平均输入长度（142k），单个请求平均输出长度（444）。压力大部分在 prefill，因此，主要成本不是 decode（生成），而是反复地处理、保存超长上下文。

## 总结

## 参考资料

https://vllm.ai/blog/2026-09-08-vllm-agentx
https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
https://vllm.ai/blog/2026-08-07-decode-context-parallelism

自行补充其它相关的参考资料。
