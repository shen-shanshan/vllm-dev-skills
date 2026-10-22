# DeepSeek-V4-Pro MXFP4 Indexer 长上下文精度测试：GPQA 与 NIAH

> 背景：PR #60834（ROCm DeepSeek-V4 MXFP4 dense indexer）的 reviewer 提出：
> *"Should we add gpqa and niah to ensure the implementation works at longer context? Don't think gsm8k would catch a potential issue in the indexer."*
>
> 本文整理这两个数据集是什么、本机已有哪些数据，以及 FP8 和 MXFP4 两种 indexer 的对比测试步骤。

## 为什么 gsm8k 测不到 indexer

DeepSeek-V4 的 sparse indexer 只有在上下文长度超过 `topk × compress_ratio = 1024 × 4 = 4096` 个 token 时，才会真正做 top-k 稀疏选择。不超过这个长度时，代码走 short-context 分支，所有压缩块都会被选中。

gsm8k 的 prompt 只有 1K 左右，**根本到不了 indexer 的 top-k 选择这一步**，所以 indexer 的精度问题在 gsm8k 上体现不出来。

这两个数据集正好补上这个缺口：

- **NIAH**：覆盖长 prefill（8K 到 128K）下的检索能力。
- **GPQA**：开启 thinking 后，推理过程往往有上万个 token，可以覆盖长 decode 过程中的 top-k 选择。

## 1. 两个数据集是什么，本机有没有

| 简称 | 全称 | 在 lm_eval 里的位置 | 用到的数据 | 本机是否已有 |
|---|---|---|---|---|
| **GPQA** | **GPQA: A Graduate-Level Google-Proof Q&A Benchmark**，研究生难度的生物、物理、化学选择题，靠搜索引擎也很难查到答案 | `gpqa_diamond_*` 系列任务。diamond 是质量最高的子集，共 198 题 | HF 数据集 `Idavidrein/gpqa`，**需要登录并同意使用条款才能下载** | ❌ 没有。`/shared/models/huggingface/datasets` 里只有 mmlu-pro、gsm8k、cc-traces，整个 `/shared/models/huggingface` 和 `~/.cache` 下也都找不到 |
| **NIAH** | **Needle In A Haystack**（大海捞针）。lm_eval 里用的是 NVIDIA **RULER** 基准中的 NIAH 子集 | `niah_single_1/2/3`、`niah_multikey_1/2/3`、`niah_multiquery`、`niah_multivalue` 共 8 个任务 | 题目由本地程序合成，不需要题库。其中 `niah_single_2/3` 和 `multikey_1` 的"草堆"文本来自 HF 数据集 `baber/paul_graham_essays` | ✅ "草堆"数据已经缓存好了，在 `/shared/models/huggingface/hub/datasets--baber--paul_graham_essays/`（`essays.parquet` 完整）。不过它放在 `hub/` 目录下，不在 `datasets/` 下 |

### NIAH 各任务说明

NIAH 的做法是把一个或多个"针"（形如 `One of the special magic numbers for <key> is: <value>.` 的句子）埋进很长的"草堆"文本里，再让模型把对应的值找出来。

| 任务 | 草堆 | 难点 |
|---|---|---|
| `niah_single_1` | 重复句子（"The grass is green..."） | 单根针，最基础 |
| `niah_single_2` | Paul Graham 文章 | 单根针，草堆是真实文本 |
| `niah_single_3` | Paul Graham 文章 | 单根针，值是 UUID |
| `niah_multikey_1/2/3` | 文章或针 | 有多根干扰针，要找到对应 key 的那一根 |
| `niah_multiquery` | 文章 | 一次查询多个 key |
| `niah_multivalue` | 文章 | 一个 key 对应多个 value，要全部找出来 |

### 本机的其他依赖情况

| 依赖 | 状态 |
|---|---|
| lm_eval | 没装，需要安装 `lm-eval[api,ruler]` |
| nltk 的 `punkt_tab` 分词数据 | 没有，NIAH 运行时会自动下载，也可以提前下 |
| HF token | 没有，下载 GPQA 时需要 |
| 网络 | HF、PyPI、GitHub 都能访问 |
| V4-Pro 的 tokenizer | 能用 `AutoTokenizer` 正常加载，NIAH 需要用它来控制 prompt 长度。但这个 tokenizer 里没有 chat_template，所以 chat 格式必须交给服务端处理 |

## 2. 测试命令和执行顺序

整体流程：先启动 Main（FP8 indexer）服务，跑完 GPQA 和 NIAH；再启动 This PR（MXFP4 indexer）服务，原样再跑一遍；最后对比两组结果。

### 第 0 步：环境准备（只做一次）

```bash
cd /home/shashen/vllm/vllm
# ruler 这个 extra 会带上 nltk、wonderwords、scipy
uv pip install --python .venv/bin/python "lm-eval[api,ruler]"

# 让 HF 的 hub 缓存和 datasets 缓存都指向 /shared，这样能直接复用已有的 paul_graham_essays，
# 之后下载的 gpqa 也会存到这里
export HF_HOME=/shared/models/huggingface

# NIAH 需要 nltk 的 punkt_tab，先下好
.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"

# GPQA 是受限数据集：先在 https://huggingface.co/datasets/Idavidrein/gpqa 页面点击同意条款，然后登录
huggingface-cli login            # token 默认保存在 /root/.cache/huggingface/token
# HF_HOME 改了之后，HF 会到 $HF_HOME/token 去找 token，所以要用 HF_TOKEN_PATH 指回登录时保存的位置
export HF_TOKEN_PATH=/root/.cache/huggingface/token
.venv/bin/python -c "from datasets import load_dataset; print(load_dataset('Idavidrein/gpqa','gpqa_diamond'))"
# 下载成功后会缓存在 /shared/models/huggingface/datasets/Idavidrein___gpqa，共 198 条
```

注意：

- `.venv` 最近被重建过好几次，`tblib` 已经丢过两次。如果之后又重建了，lm_eval 也要重新装。
- 已经实测过：lm_eval 0.4.13 可以正常运行 GPQA。

### 测试脚本

下面两个脚本已经实际跑过，放在 `/home/shashen/vllm/vllm/benchmark/gpqa/` 下，第 1、2、4 步直接调用即可。

`benchmark/gpqa/serve.sh`：启动服务，参数 `fp8` 或 `mxfp4`，两种配置唯一的区别是 `--attention-config`。

```bash
#!/usr/bin/env bash
# Usage: serve.sh fp8|mxfp4 -- DeepSeek-V4-Pro TP8 for the GPQA/NIAH accuracy runs.
set -euo pipefail
MODE=${1:?fp8|mxfp4}
cd /home/shashen/vllm/vllm
EXTRA=()
if [[ "$MODE" == "mxfp4" ]]; then
  EXTRA=(--attention-config '{"indexer_kv_dtype":"mxfp4"}')
fi
export VLLM_ROCM_USE_AITER=1
exec .venv/bin/vllm serve /shared/models/deepseek-ai/DeepSeek-V4-Pro \
  -tp 8 --port 8333 --trust-remote-code --served-model-name dsv4-pro \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --kv-cache-dtype fp8 --load-format fastsafetensors \
  --max-model-len 163840 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  "${EXTRA[@]}"
```

`benchmark/gpqa/run_gpqa.sh`：跑 GPQA，参数是标签，以及可选的生成预算（默认 131072）。

```bash
#!/usr/bin/env bash
# Usage: run_gpqa.sh <tag> [max_gen_toks] -- GPQA Diamond CoT zero-shot against the server on :8333.
set -euo pipefail
TAG=${1:?tag}
MAX_GEN=${2:-131072}
cd /home/shashen/vllm/vllm
export HF_HOME=/shared/models/huggingface HF_TOKEN_PATH=/root/.cache/huggingface/token
exec .venv/bin/lm_eval --model local-chat-completions --apply_chat_template \
  --model_args model=dsv4-pro,base_url=http://127.0.0.1:8333/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=10800,tokenized_requests=False \
  --tasks gpqa_diamond_cot_zeroshot \
  --gen_kwargs max_gen_toks=$MAX_GEN,temperature=0 \
  --output_path benchmark/gpqa/results/gpqa_${TAG}_gen${MAX_GEN} --log_samples
```

关键配置及原因：

| 配置 | 值 | 原因 |
|---|---|---|
| `max_gen_toks` | **131072** | 32768 不够用，会让大量题目的回答为空，详见后面的"踩坑记录" |
| `--max-model-len` | 163840 | 要能装下 prompt 加 131072 的生成预算；NIAH 最长测 128K，也够用 |
| lm_eval 的 `timeout` | **10800**（3 小时） | 单个请求推理十几万 token，可能要跑几十分钟，原来设的 3600 秒有超时风险 |
| `temperature` | 0 | 贪心解码，结果可以复现，便于对比 FP8 和 MXFP4 |

### 第 1 步：启动 Main（FP8 indexer）服务

```bash
cd /home/shashen/vllm/vllm/benchmark/gpqa
./serve.sh fp8 > server_fp8.log 2>&1 &
# 等服务就绪，并确认用的是 FP8 indexer
until curl -sf localhost:8333/health >/dev/null; do sleep 10; done
grep -m1 "indexer cache for Lightning" server_fp8.log     # 应为 Using FP8 indexer cache ...
grep -m1 "GPU KV cache size" server_fp8.log               # 记录 KV 容量
```

### 第 2 步：跑 GPQA

```bash
cd /home/shashen/vllm/vllm
benchmark/gpqa/run_gpqa.sh fp8 > benchmark/gpqa/gpqa_fp8_gen131072.log 2>&1
tail -5 benchmark/gpqa/gpqa_fp8_gen131072.log             # 结果表
```

实测每种配置大约需要 **65 分钟**，时间主要花在少数推理链十几万 token 的长尾请求上。跑完之后，建议先用后面"踩坑记录"里的检查脚本看一下空回答的数量。

### 第 3 步：跑 NIAH，每个长度单独跑一次

```bash
TAG=fp8   # 跑 MXFP4 时改成 mxfp4
NIAH=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue
for L in 8192 32768 131072; do
  .venv/bin/lm_eval --model local-completions \
    --model_args model=dsv4-pro,base_url=http://127.0.0.1:8333/v1/completions,tokenizer=/shared/models/deepseek-ai/DeepSeek-V4-Pro,tokenizer_backend=huggingface,num_concurrent=32,max_retries=3,timeout=10800,tokenized_requests=False,max_length=163840 \
    --tasks $NIAH \
    --metadata "{\"max_seq_lengths\":[$L]}" \
    --limit 100 \
    --output_path results/niah_${TAG}_$L --log_samples
done
```

为什么每个长度要单独跑一次：RULER 每个任务、每个长度都固定生成 500 条样本，这个数写死在 `prepare_niah.py` 里，没法通过参数改。而 `--limit` 截取的是拼接后的前 N 条。如果在一次运行里同时传多个长度，`--limit 100` 只会截到最短那个长度的样本。拆开跑，就能保证每个长度各取 100 条。

用 `local-completions` 而不是 chat 接口，是因为 RULER 的 prompt 本身就以答案前缀结尾（`... is:`），设计上就是让模型直接续写。如果 V4 用纯续写的效果很差（两边分数都很低），可以改成 `local-chat-completions --apply_chat_template`，但 FP8 和 MXFP4 必须用同一种方式。

### 第 4 步：停掉服务，换 This PR（MXFP4）再跑一遍

停掉第 1 步的服务（`kill %1`，或者 kill 掉 `vllm serve` 进程），等 GPU 显存释放之后，启动 MXFP4 服务：

```bash
cd /home/shashen/vllm/vllm/benchmark/gpqa
./serve.sh mxfp4 > server_mxfp4.log 2>&1 &
until curl -sf localhost:8333/health >/dev/null; do sleep 10; done
grep -m1 "indexer cache for Lightning" server_mxfp4.log   # 应为 Using MXFP4 indexer cache ...
grep -c "unknown asm constraint" server_mxfp4.log          # 应为 0

cd /home/shashen/vllm/vllm
benchmark/gpqa/run_gpqa.sh mxfp4 > benchmark/gpqa/gpqa_mxfp4_gen131072.log 2>&1
```

NIAH 部分，把第 3 步的 `TAG` 改成 `mxfp4`，原样再跑一遍。

### 第 5 步：对比结果

| 数据集 | 看哪个指标 |
|---|---|
| GPQA | `gpqa_diamond_cot_zeroshot` 的 `exact_match`（flexible-extract） |
| NIAH | 每个任务在 8K / 32K / 128K 下的分数，以及 8 个任务的平均分 |

建议整理成下面这样的表，贴到 PR 里：

| Indexer | GPQA Diamond | NIAH 8K | NIAH 32K | NIAH 128K |
|---|---|---|---|---|
| Main (FP8 Indexer) | | | | |
| This PR (MXFP4 Indexer) | | | | |

## 踩坑记录：GPQA 有大量题目的回答为空

### 现象

第一次按 `max_gen_toks=32768` 跑 FP8，GPQA 只得了 **0.5202**，比这一级别模型的正常水平（70% 以上）低很多，strict-match 是 0。

### 排查过程

**① 检查样本，统计空回答的数量。** lm_eval 的样本文件里，每道题会按两种 filter 各记录一行。`resps` 只保存服务端返回的 `content`（正文），不包含推理过程。

```bash
cd /home/shashen/vllm/vllm/benchmark/gpqa
f=$(ls results/gpqa_fp8_gen32768/dsv4-pro/samples_*.jsonl)
/home/shashen/vllm/vllm/.venv/bin/python - "$f" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
fl = [r for r in rows if r.get("filter") == "flexible-extract"]
ne = [r for r in fl if r["resps"][0][0].strip()]
print("n", len(fl), "empty", len(fl) - len(ne),
      "acc", round(sum(r["exact_match"] for r in fl) / len(fl), 4),
      "acc_nonempty", round(sum(r["exact_match"] for r in ne) / len(ne), 4))
EOF
```

结果：198 题中有 **71 题的 `content` 是空字符串**；有回答的 127 题准确率是 **0.811**，这才是模型的正常水平。所以分数低是因为空回答，而不是答错。

**② 确认请求本身没有失败。** 服务端日志里的请求全部返回 200，lm_eval 日志里也没有重试或报错：

```bash
grep -oE '"POST /v1/chat/completions HTTP/1.1" [0-9]+' server_fp8.log | sort | uniq -c
```

**③ 挑一道回答为空的题原样重放，查看 `finish_reason` 和 token 用量：**

```bash
f=$(ls results/gpqa_fp8_gen32768/dsv4-pro/samples_*.jsonl)
/home/shashen/vllm/vllm/.venv/bin/python - "$f" <<'EOF'
import json, sys, requests
rows = [json.loads(l) for l in open(sys.argv[1])]
fl = [r for r in rows if r.get("filter") == "flexible-extract"]
r = [r for r in fl if not r["resps"][0][0].strip()][0]
a = r["arguments"]["gen_args_0"]
msgs = json.loads(a["arg_0"][0] if isinstance(a["arg_0"], list) else a["arg_0"])
resp = requests.post("http://127.0.0.1:8333/v1/chat/completions",
                     json={"model": "dsv4-pro", "messages": msgs,
                           "max_tokens": 32768, "temperature": 0}, timeout=3600).json()
c = resp["choices"][0]
print("finish:", c["finish_reason"], "usage:", resp["usage"])
print("content len:", len(c["message"].get("content") or ""))
EOF
```

结果：`finish_reason=length`，`completion_tokens=32768`，其中 `reasoning_tokens=32768`，`content` 长度为 0。

### 原因

V4-Pro 默认开启 thinking。服务端的 `--reasoning-parser deepseek_v4` 会把推理过程放进 `reasoning` 字段，正文放进 `content` 字段，而 lm_eval 只读取 `content`。GPQA 是研究生难度的题目，推理链经常超过 32K token。生成预算用完时模型还在推理，正文一个字都没输出，于是 lm_eval 拿到的是空字符串，答案提取失败，这道题就被判为错误。

### 解决方案

1. 把 `max_gen_toks` 从 32768 提高到 **131072**。`run_gpqa.sh` 默认就是这个值，第二个参数可以覆盖。
2. 同时把服务的 `--max-model-len` 设为 163840，要能装下 prompt 加上生成预算。
3. 把 lm_eval 的单请求 `timeout` 从 3600 提高到 10800。推理十几万 token 的请求可能要跑几十分钟。
4. 结果目录名带上生成预算（例如 `gpqa_fp8_gen131072`），避免不同预算的结果互相覆盖。

### 效果

| 生成预算 | GPQA（FP8） | 空回答 | 用时 |
|---|---|---|---|
| 32768 | 0.5202 | 71 / 198 | 约 12 分钟 |
| 131072 | 0.7121 | 13 / 198 | 约 65 分钟 |

提高到 131072 之后，仍有 11～13 题的回答为空。这些题的推理链超过了 13 万 token，很可能是贪心解码陷入了循环。FP8 和 MXFP4 两边的设置完全一致，所以不影响对比结论，只会压低绝对分数。如果想要更接近官方的绝对分数，可以改用 DeepSeek 推荐的采样参数，代价是两边之间的波动会变大。

### 经验

- **对开启 thinking 的模型，跑完评测后先检查空回答的比例，再看分数。** 分数异常低时，先区分是答错了，还是根本没输出答案。
- **lm_eval 默认只读 `content`。** 推理被截断不会报错，只会悄悄变成一道错题。
- **生成预算要按任务难度来设。** gsm8k 这类简单题用 4K～8K 就够了；GPQA 这类难题至少需要 128K。

## 实测结果（GPQA）

详细数据见 `dsv4_mxfp4_indexer_gpqa_results.md`。

| Indexer | GPQA Diamond | 空回答 | GPU KV 容量 |
|---|---|---|---|
| Main（FP8 Indexer） | 0.7121 ± 0.0323 | 13 / 198 | 1,878,431 tokens |
| This PR（MXFP4 Indexer） | 0.7121 ± 0.0323 | 11 / 198 | 2,051,155 tokens（+9.19%） |

## 注意事项

- **GPQA 的样本量很小。** 198 题对应的统计误差大约是 ±3.2～3.5 个百分点。FP8 和 MXFP4 差 2～3 个百分点算正常波动。可以考虑每种配置跑 2 遍，或者在 PR 里写明这一点。
- **128K 那一档最耗时。** NIAH 8 个任务各 100 条，每条约 128K token，在 TP8 上大概需要几十分钟。如果时间紧，128K 可以把 `--limit` 降到 50。
- **NIAH 的 lm_eval 参数还没有在本机实际跑过。** GPQA 的参数已经实测过。NIAH 的参数是根据上游源码核对的，例如 RULER 会按 `tokenizer` → `pretrained` → `model` 的顺序查找 tokenizer，`max_seq_lengths` 通过 `--metadata` 传入。如果 `local-completions` 加载 tokenizer 时报 `trust_remote_code` 相关的错误，就在 `--model_args` 里加上 `trust_remote_code=True`。另外，NIAH 命令里的 `timeout=3600` 也建议改成 10800。
- **两边的配置要完全一致。** 服务参数、lm_eval 版本、`--limit`、`temperature` 和 `max_gen_toks` 都要保持相同，唯一的区别只能是 `--attention-config` 这一项。
