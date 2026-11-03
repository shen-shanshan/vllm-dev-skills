# ViT CUDA graph：跨 token budget 共享 capture buffer

`EncoderCudaGraphManager` 原来会为每个 token budget 各自保留一整套输入/输出 buffer。
本次改动让较小的 budget 直接使用最大 budget 那份 buffer 的前缀视图，capture buffer
的常驻显存约减半，encoder 输出与改动前逐位一致。

- 基线 commit：`67791dd94d`（vllm 分支 `vit-cg`，改动尚未提交）
- 相关 RFC：https://github.com/vllm-project/vllm/issues/38175

## 文件

| 路径 | 内容 |
|---|---|
| `pr-shared-encoder-cg-capture-buffers.md` | PR 标题与描述（英文，vLLM 模板） |
| `shared-capture-buffers.patch` | 代码与单元测试改动（`git apply` 可用） |
| `validation/harness.py` | A/B 验证脚本：统计 capture buffer 显存、各 budget 命中、greedy 输出，并在 worker 内逐位对比共享与不共享两个 manager 的 replay 结果 |
| `validation/run.sh` | 启动脚本：`run.sh <qwen3_vl\|dsv41> <tag> [eager_vit]`，结果写到 `validation/out/` |
| `validation/compare.py` | 汇总对比：`compare.py <model> <基准 tag> <tag>...` |
| `validation/out/` | 全部运行结果（`*.json` 为统计与输出，`*.log` 为完整日志） |

## `validation/out/` 中各次运行

| tag | 代码版本 | 说明 |
|---|---|---|
| `base` | 改动前（`git stash`） | 基线 |
| `shared` / `shared2` | 第一版共享实现 | 同一代码连跑两次，用来衡量端到端输出的运行间噪声 |
| `final` | 最终版（`shared_keys` 改为 capture 结束后统计） | PR 中引用的数据 |
| `eager` | 改动后，`cudagraph_mm_encoder=False` | 仅作参考（Qwen3-VL） |
| `qwen_shared.*` | 第一版 harness | 早期试跑，只有整批请求，没有 A/B 对比；保留备查 |

## 结果摘要（MI355X，Model Runner V2）

| 模型 | capture buffer | KV cache tokens | 逐位 A/B |
|---|---|---|---|
| Qwen3-VL-2B-Instruct（TP1） | 974.1 → 488.0 MiB | 1,508,352 → 1,513,472 | 16 个 batch、6 个 budget、5 种 replay 顺序，0 处不一致 |
| DeepSeek-V4.1-Flash（TP2，每个 rank） | 724.1 → 362.8 MiB | 3,468,391 → 3,477,318 | 9 个 batch、3 个 budget、5 种顺序、两个 rank，0 处不一致 |

注意：

- 端到端 greedy 输出不能用来判断正确性。同一份代码连跑两次，Qwen3-VL 只有 11/14 条一致，DSv4.1 只有 2/14 条一致，原因是每次的 batch 组合不同。
- DSv4.1 在 TP 下，日志里的 "Graph capturing finished … took" 数值在不同运行之间会波动约 ±0.5 GiB，所以显存收益以 capture buffer 的唯一 storage 字节数和 KV cache 容量为准。

## 复现

```bash
cd /home/shashen/vllm/vllm
git apply ../vllm-dev-skills/docs/vit_cg/shared-capture-buffers.patch   # 若尚未应用
V=../vllm-dev-skills/docs/vit_cg/validation
$V/run.sh qwen3_vl final                 # GPU 0（可用 GPUS= 覆盖）
GPUS=6,7 $V/run.sh dsv41 final           # TP2
git stash && $V/run.sh qwen3_vl base; git stash pop
python3 $V/compare.py qwen3_vl base final
```

前置条件：

- Qwen3-VL-2B-Instruct 权重：`huggingface_hub.snapshot_download('Qwen/Qwen3-VL-2B-Instruct')`，会下载到 `/root/.cache/huggingface/hub`。
- DSv4.1-Flash 权重：`/shared/models--deepseek-ai--DeepSeek-V4.1-Flash`。
- 单元测试：`pytest tests/v1/cudagraph/test_encoder_cudagraph.py -k SharedBuffers`。
