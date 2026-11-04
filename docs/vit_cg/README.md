# ViT CUDA graph：跨 token budget 共享 capture buffer

`EncoderCudaGraphManager` 原来会为每个 token budget 各自保留一整套输入/输出 buffer。
本次改动让较小的 budget 直接使用最大 budget 那份 buffer 的前缀视图，capture buffer
的常驻显存约减半。同时修复了 Step3-VL 在 `clone=True` 时仍返回 graph 输出视图的问题。

- 基线 commit：`67791dd94d`（vllm 分支 `vit-cg`）。第一版已提交为 `9ef59a3892`；`_aliased_slots` 精简和 review 意见的修改尚未提交。`shared-capture-buffers.patch` 是相对基线的完整改动。
- 相关 RFC：https://github.com/vllm-project/vllm/issues/38175
- PR：#65351；review 意见见 `skills/vllm-rocm-pr-review/outputs/pr-65351-review.md`

## 文件

| 路径 | 内容 |
|---|---|
| `pr-shared-encoder-cg-capture-buffers.md` | PR 标题与描述（英文，vLLM 模板） |
| `shared-capture-buffers.patch` | 代码、单元测试和 `docs/design/cuda_graphs_multimodal.md` 的改动（`git apply` 可用） |
| `validation/harness.py` | A/B 验证脚本，详见下文 |
| `validation/run.sh` | 启动脚本：`run.sh <qwen3_vl\|dsv41\|step3_vl\|gemma4\|deepseek_ocr> <tag> [eager_vit]`，结果写到 `validation/out/` |
| `validation/compare.py` | 汇总对比：`compare.py <model> <基准 tag> <tag>...` |
| `validation/out/` | 全部运行结果（`*.json` 为统计与输出，`*.log` 为完整日志） |

`harness.py` 做三件事：
- 统计 capture buffer 显存、各 budget 的命中情况和 greedy 输出。
- 在 worker 内额外 capture 一个关闭共享的 manager，做两种 A/B：
  - **逐 batch A/B**：每个 batch 单独 replay，按 5 种顺序。
  - **execute 级 A/B**：整次 `execute` 重跑，用来抓后一个 batch 覆盖前一个 batch 输出的问题。
- 设置 `AB_CONTROL=1` 时，再加对照组：比较两个都不共享的 manager，以及同一个 graph 连续 replay 两次的结果。

## `validation/out/` 中各次运行

| 文件 | 代码版本 | 说明 |
|---|---|---|
| `*_base.*` | 改动前（`67791dd94d` 的 `encoder_cudagraph.py` / `step3_vl.py`） | 显存、KV 的基线 |
| `qwen3_vl_shared*` / `dsv41_shared*` | 第一版共享实现 | 同一代码连跑两次，用来衡量端到端输出的运行间噪声 |
| `qwen3_vl_final` / `dsv41_final` | 第二版（提交 `9ef59a3892`） | Qwen3-VL、DSv4.1 的显存和 KV 数据 |
| `qwen3_vl_aliased` / `dsv41_aliased` | `_aliased_slots` 版本，容器重启后重跑 | Qwen3-VL、DSv4.1 的逐 batch A/B 数据（旧版 harness，没有 execute 级 A/B） |
| `gemma4_shared` | 最终版，`max_vision_items_per_batch=2` | Gemma4（capture axis）的显存、KV 和 A/B 数据 |
| `gemma4_shared_autobatch` | 最终版，batch 大小自动推断 | 只命中单 budget，仅作参考 |
| `step3_vl_control` | 最终版，`AB_CONTROL=1` | Step3-VL（双路径）的显存、KV、A/B 和对照组数据 |
| `step3_vl_nofix` | 共享实现，撤掉 Step3-VL 修复 | 复现 bug：execute 级 A/B 最大偏差 14.13 |
| `step3_vl_main` | main（`dd4e40142a`），不共享，未修复 Step3 | 与单独执行对比：2/11 个 item 被污染（相对误差 0.62），证明 main 上已有此 bug |
| `step3_vl_pr65084` | PR #65084 的 `encoder_cudagraph.py`，未修复 Step3 | 与单独执行对比：6/11 个 item 被污染（相对误差 1.34） |
| `step3_vl_pr65351` | PR #65351 head | 与单独执行对比：0/11 |
| `step3_vl_shared` | 最终版，`gpu_memory_utilization=0.85` | 早期运行，KV 与其他 Step3 运行不可比 |
| `step3_vl_eager` | 最终版，`cudagraph_mm_encoder=False` | 证明 Step3-VL 的乱码输出与 CUDA graph 无关 |
| `deepseek_ocr_shared` | 最终版 | 启动失败（ROCm 上 Triton kernel 编译错误，与本改动无关） |
| `qwen3_vl_eager` | 改动后，`cudagraph_mm_encoder=False` | 仅作参考 |
| `qwen_shared.*` | 第一版 harness | 早期试跑，保留备查 |

## 结果摘要（MI355X，Model Runner V2）

| 模型 | capture buffer | KV cache tokens | 逐 batch A/B | execute 级 A/B |
|---|---|---|---|---|
| Qwen3-VL-2B-Instruct（TP1） | 974.1 → 488.0 MiB | 1,508,352 → 1,513,472 | 11 个 batch、7 个 budget，0 处不一致 | 未测 |
| DeepSeek-V4.1-Flash（TP2，每个 rank） | 724.1 → 362.8 MiB | 3,468,391 → 3,477,318 | 9 个 batch、3 个 budget、两个 rank，0 处不一致 | 未测 |
| Gemma4-E2B-it（capture axis，27 个 graph） | 1356.3 → 539.7 MiB | 8,759,154 → 8,813,356 | 13 个 batch，0 处不一致 | 9 次（1 次跨 budget），0 处不一致 |
| Step3-VL-10B（双路径，17 个 graph） | 1018.7 → 430.1 MiB | 586,752 → 598,016 | 最大偏差 0.031（replay 噪声） | 修复后最大 0.031；未修复时 14.13 |

注意：

- **端到端输出不能当判据。** greedy 输出不能用来判断正确性：同一份代码连跑两次，Qwen3-VL 只有 11/14 条一致，DSv4.1 只有 2/14 条一致，原因是每次的 batch 组合不同。
- **Step3-VL 的小偏差是 replay 噪声。** Step3-VL 的 vision graph 在最小的两个 budget（global 169、local 162）上本身就不确定：同一个不共享的 graph 对同一输入连续 replay 两次，结果也不一致，偏差量级相同（≤ 0.031）。所以 Step3-VL 不能用逐位相等判断，要看偏差量级。
- **Step3-VL 生成乱码与本改动无关。** eager ViT 下也一样，可能是 prompt 或 ROCm 平台的问题；A/B 比较的是 encoder 输出张量，不受影响。
- **DeepSeek-OCR 在 ROCm 上跑不起来。** `deepencoder.py:170` 的 Triton kernel 编译失败，与本改动无关。
- **DSv4.1 的 graph 显存日志不可靠。** 在 TP 下，日志里的 "Graph capturing finished … took" 数值在不同运行之间会波动约 ±0.5 GiB，显存收益以 capture buffer 的唯一 storage 字节数和 KV cache 容量为准。
- **DSv4.1 的 KV 数据来自容器重启前。** KV 对比用的是重启前同一环境下的 `base` 和 `final`；`aliased` 在重启后跑，KV 为 3,477,243 tokens，相差的 75 个 token 来自环境差异。

## 复现

```bash
cd /home/shashen/vllm/vllm
git apply ../vllm-dev-skills/docs/vit_cg/shared-capture-buffers.patch   # 若尚未应用
V=../vllm-dev-skills/docs/vit_cg/validation
$V/run.sh qwen3_vl final                          # GPU 0（可用 GPUS= 覆盖）
GPUS=6,7 $V/run.sh dsv41 final                    # TP2
GPUS=1 $V/run.sh gemma4 shared
AB_CONTROL=1 GPUS=3 $V/run.sh step3_vl control
# 基线：临时恢复改动前的文件
git stash && git checkout 67791dd94d -- vllm/v1/worker/encoder_cudagraph.py vllm/model_executor/models/step3_vl.py
GPUS=3 $V/run.sh step3_vl base
git checkout HEAD -- vllm/v1/worker/encoder_cudagraph.py vllm/model_executor/models/step3_vl.py && git stash pop
python3 $V/compare.py step3_vl base control
```

前置条件：

- 权重：
  - HF 模型由 vLLM 按 repo id 自动下载到 `/root/.cache/huggingface/hub`：Qwen/Qwen3-VL-2B-Instruct、google/gemma-4-E2B-it、stepfun-ai/Step3-VL-10B、deepseek-ai/DeepSeek-OCR。
  - DSv4.1-Flash 在 `/shared/models--deepseek-ai--DeepSeek-V4.1-Flash`。
- 单元测试：`pytest tests/v1/cudagraph/test_encoder_cudagraph.py -k "shared_capture or step3"`。
