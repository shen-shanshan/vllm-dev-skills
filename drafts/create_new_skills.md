我想写一个用于 review https://github.com/vllm-project/vllm 项目中 amd (rocm) 相关 PR 的 skill，名叫 `vllm-rocm-pr-review`。

使用方式：
用户给出 vllm PR 链接，调用本 skill，你需要给出这个 PR 的 motivation、代码改动的总结（可以参考 `vllm-pr-summary` 这个 skill，但要比它生成的内容更精简一点），然后给出你的 review 意见。review 意见需要根据重要程度进行排序，并按意见类型（比如：是 bug、缺少注释、设计不合理、……等等）进行分类，其它细节你自行补充。

你需要参考以下材料：
1.ATOM 是 AMD 内部对标 vllm 的推理引擎，也可以作为插件植入到 vllm 中，https://github.com/ROCm/ATOM/blob/main/.claude/skills/review-pr/SKILL.md 是它的 `pr-review` skill，里面记录了一些在 rocm 上开发的注意事项，你也可以参考这个 skill 的结构来生成我们的 skill。
2.aiter 是 AMD 用于大模型推理的算子库，vllm 中也大量使用了 aiter 中的算子，https://github.com/ROCm/aiter/blob/main/.claude/skills/review-pr/SKILL.md 是它的 `pr-review` skill，你需要参考里面的一些算子使用的注意事项。
3.mori 是 AMD 用于大模型推理的通信库，代码仓库：https://github.com/ROCm/mori，你需要了解关于该仓库的一些基本信息，用于 review vllm 中关于 mori 的 PR。

其它注意事项：
1.你需要提供一个更新接口，可以一键调用，然后自动查看以上文档和项目的最新情况，并更新本 skill。
2.将每次 PR 分析的结果总结为一份中文的 markdown 文档，放到当前 skill 的 `outputs` 目录下。
3.本次创建的 skill 相关文件需要放到本机的 `/Users/shanshan-shen/Documents/GitHub/vllm-dev-skills/skills/vllm-rocm-pr-review` 目录下，然后再通过 `ln -sfn /Users/shanshan-shen/Documents/GitHub/vllm-dev-skills/skills/vllm-rocm-pr-review ~/.claude/skills/vllm-rocm-pr-review` 链接到 claude 可识别的 skill 目录中。
4.最后，将这个 skill 更新到 `/Users/shanshan-shen/Documents/GitHub/vllm-dev-skills/README.md` 文档中。

---

【需求背景】：
请帮我创建一个 skill，名叫 vllm-vs-atom-decode-trace-comparison，用于对比分析 https://github.com/vllm-project/vllm 和 https://github.com/ROCm/ATOM 这两个推理引擎基于同样的配置运行同一个大模型时，使用 torch profiler 采集到的 trace 差异。
用户输入：
1.跑的是哪个模型（比如：MiniMaxAI/MiniMax-M3）；
2.一份 vllm trace 的路径（比如：/Users/shanshan-shen/Documents/GitHub/rocm-beginner-to-master/vllm/models/minimax_m3/traces/vllm-minimax-m3-tp4-8k1k-conc8.json）；
3.一份 atom trace 的路径（比如：/Users/shanshan-shen/Documents/GitHub/rocm-beginner-to-master/vllm/models/minimax_m3/traces/atom-minimax-m3-tp4-8k1k-conc8.json）；
4.输出报告使用中文（默认）、英文、还是中英文各生成一份。

【你的任务】：
1.每一个 trace 文件都包含 decode 阶段的多个 step，数据量很大，因此你可以只取其中某一个 step，不同类型的层各取一个 layer 的 trace 数据用于对比和分析。比如：对于 MiniMax-M3 模型，包含：(1) dense GQA + MLP 和 (2) MSA + MoE 这两种类型的层，你就可以对 (1) 和 (2) 各取某一个 step 中某一个 layer 的 trace 数据，从而减少了需要处理的数据量。
2.对于每一种 layer，对比 vllm 和 atom 的实现差异。比如：对于同样的一个操作，vllm 和 atom 使用的算子分别是什么？有没有使用融合算子？有没有使用 multi-stream？哪些操作 vllm 和 atom 使用的算子不同？等等。请将这些信息总结为 markdown 表格，每种 layer 一个表，表格的每一行为某一个操作在 vllm / atom 上的实现对比。
3.请根据上一步生成的表格，创建一份给 vllm 的 todo list。比如：对于某个操作，vllm 和 atom 的实现（算子）有差异，并且 atom 的更快，那就说明需要将 vllm 中的这些算子替换为 atom 中的算子。
4.请将第二步和第三步中分析的结果全部汇总到一份 markdown 报告中，包括：这两条 trace 的基本信息、当前跑的这个模型结构的基本介绍（有哪些层？每一层内部的 forward 流程是什么？）、第二步和第三步的分析结果，其它信息你看情况补充。

【注意事项】：
1.将最后输出的报告保存到当前 skill 目录下的 outputs 目录中（比如：xxx/vllm-vs-atom-decode-trace-comparison/outputs/xxx.md）。
2.该 skill 需要能够处理并分析基于 vllm / atom 运行的各种大模型，而不是只针对某一种模型的 skill。
