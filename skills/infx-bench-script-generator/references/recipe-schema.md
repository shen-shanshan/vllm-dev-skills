# InferenceX srt-slurm Recipe 结构（本 skill 需要解析的部分）

来源：`inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/**/agentic.yaml`。本 skill 只支持单机 vllm recipe。

## 结构：base + override_*

一个 recipe 文件 = 一个 `base` 段 + 若干 `override_*` 段。每个 override 是 base 的**深合并增量**：
dict 递归合并（override 的叶子覆盖 base 的叶子），非 dict 值整体替换。一个 matrix 点
（TP/EP/DP、CONC、KV offload 组合）对应恰好一个 override variant。

## 关键字段

| 字段 | 含义 | 生成脚本中的去向 |
| --- | --- | --- |
| `model.path` | `hf:<repo-id>`，模型权重来源 | server 脚本下载/挂载路径 |
| `model.container` | 推理引擎容器镜像 | docker run 的 IMAGE |
| `model.precision` | 精度（fp4/fp8/...） | client 的 PRECISION |
| `resources.gpu_type` / `gpus_per_node` | 硬件 SKU、单机 GPU 数 | 仅用于命名/注释 |
| `engine` | `{type: vllm, ...}` 或字符串 | client 的 FRAMEWORK |
| `frontend` | `{type: vllm}` 或 `{type: vllm-router, args, env}` | dep8 变体需单独起 vllm-router |
| `roles.agg` | 聚合角色（单机即服务端）：`args` 是 vllm serve 参数，`env` 是服务端环境变量 | 02 脚本的 server 参数与 env |
| `benchmark` | `{type: custom, command: bash /infmax-workspace/benchmarks/srt_agentic.sh, env}` | client 脚本的 env passthrough |
| `setup_script` / env 里的 `SETUP_PIP_PACKAGES` | 镜像缺的 python 包，srt-slurm 会在启动前安装 | 02 脚本的 `pip install` 行 |
| `health_check` | 引擎就绪探测参数 | 参考值，02 用 /health + VLLM_ENGINE_READY_TIMEOUT_S |

## args 语义注意点

- `roles.agg.args` 的 key 是 kebab-case，与 `vllm serve` CLI 一一对应：`tensor-parallel-size: 8` → `--tensor-parallel-size 8`，`true` → 裸 flag（如 `--trust-remote-code`），`false`/`null` → 省略。
- `revision` / `tokenizer-revision` 只在 HF 下载时使用；权重已本地挂载时从 serve 参数中删除。
- `compilation-config` / `speculative-config` 是 JSON 字符串，原样加引号传给 CLI。
- `speculative-config` 的 `method: dspark` 等投机参数在 CI 里会被 golden acceptance 覆盖（合成验证），本地跑是**真实验证**，吞吐与 CI 发布数据不可直接比较。
- dep8 变体（如 `override_dep8_c64`）：`roles.agg.args` 改为 `tensor-parallel-size: 1, data-parallel-size: 8`，`frontend.type: vllm-router`，`frontend.env.SETUP_PIP_PACKAGES: 'vllm-router==0.1.14'`，且 `benchmark.env` 增加 `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID: '1'`（Router 按 X-Session-ID 哈希路由）。

## Worked example：dsv4/vllm/mi355x-fp4-mtp/agentic.yaml

- base：TP8、fp4、dspark-6 投机、`--kv-cache-dtype fp8`、`--compilation-config {"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}`、`SETUP_PIP_PACKAGES: Pillow fastapi uvicorn`、`VLLM_ENGINE_READY_TIMEOUT_S: '10800'`。
- `override_tp8_c56`：合并后 `max-num-seqs: 112`（=2×CONC）、`benchmark.env.CONC: '56'`，无 frontend 变化。
- `override_dep8_c64`：合并后 `tensor-parallel-size: 1, data-parallel-size: 8, prefill-schedule-interval: 8, long-prefill-token-threshold: 16384, max-num-seqs: 64, CONC: '64'`，frontend 变为 vllm-router（policy consistent_hash, request-timeout-secs 14400, disable-retries）。
