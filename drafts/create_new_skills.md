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
