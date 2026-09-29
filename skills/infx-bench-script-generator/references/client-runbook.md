# Client 脚本领域知识（03_run_client.sh）

客户端 = InferenceX 仓库的 `benchmarks/srt_agentic.sh`（AgentX trace replay，AIPerf 驱动）。
容器内布局：repo 挂载在 `/infmax-workspace`（无 host 路径时 clone 进 named volume），
HF 缓存/token 在 `/hf-cache`，结果在 `/logs`。

## 环境变量清单（缺一不可）

`srt_agentic.sh` 及其调用的 `benchmark_lib.sh` 有多个 `check_env_vars` 门。03 脚本的导出顺序
（`inferencex-e2e/benchmarks/benchmark_lib.sh` 行号对应调用点）：

| 变量 | 值 | 来源 |
| --- | --- | --- |
| `IS_AGENTIC` / `SCENARIO_TYPE` / `KV_OFFLOADING` | `1` / `agentic-coding` / `none` | **必须最先导出**：否则 benchmark_lib 不清理 MAX_MODEL_LEN，可能把它变成 `--max-context-length`（:383-401, :3334） |
| `INFMAX_CONTAINER_WORKSPACE` | `/infmax-workspace` | 定位 repo 与 aiperf 源码（:3078） |
| `MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME DURATION` | 见 variant.env | srt_agentic.sh:39-41 |
| `RESULT_DIR EVAL_ONLY` | `/logs/agentic` / `false` | srt_agentic.sh:18 |
| `PORT` / `AIPERF_SERVER_URL` | 8000；dep8 时 URL 指向 router 30000 | build_replay_cmd:3237 |
| `AIPERF_SERVER_METRICS_URLS` | `http://localhost:8000/metrics` | 有 `AIPERF_REQUIRED_SERVER_METRIC_PREFIX` 的 recipe 必须有 metrics（:3385-3409 会校验前缀） |
| `AIPERF_FAILED_REQUEST_THRESHOLD` 等 AIPERF_* / AGENTIC_* | 默认值 | `source benchmarks/runtime_settings.sh`（:18-28） |
| `AIPERF_EXPERIMENTAL_FAST` | `0` | runtime_settings 没有默认，必须显式导出 |
| `ENABLE_AGENTX_POWER REQUIRE_POWER` | `0` / `0` | 本地关功耗监控 = 合法运行，且免去 TP/PP_SIZE/PCP_SIZE 与 amd-smi 依赖（:3413, :3521） |
| `IS_MULTINODE` | `false` | :3413 |
| `HF_HUB_CACHE` | `/hf-cache` | trace 数据集下载位置 |
| `SERVED_MODEL_NAME` | recipe 的 `served-model-name` | 请求 wire name（:3278） |
| recipe `benchmark.env` 其余项 | 原样 | 如 `AIPERF_HTTP_TCP_USER_TIMEOUT`、`AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:'`、dep8 的 `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID='1'` |

注意 `srt_agentic.sh` 自己**不 source** runtime_settings.sh（CI 的 workflow 才 source），所以 03 脚本必须显式 source，再覆盖 `ENABLE_AGENTX_POWER` 等。

## AIPerf 环境自举（install_agentic_deps, :3116-3158）

- 需要：`curl`（无 uv 时拉 astral.sh 安装 uv）、PyPI/HF 网络、`AIPERF_PYTHON_VERSION`（runtime_settings 给 3.11）。
- 不需要：git（`uv pip install -e $AIPERF_DIR` 直接装 repo 里的 `utils/aiperf` 子模块 checkout，所以 **submodule 必须已拉取**）。
- venv 建在 `/tmp`（AIPERF_RUNTIME_DIR），容器重启后重建，属正常。

## Trace 数据集（resolve_trace_source, :3165-3234）

`MODEL_PREFIX` 决定 HF 数据集 loader，client 用 `hf download --repo-type dataset` 下载：

| MODEL_PREFIX | HF 数据集 |
| --- | --- |
| dsv4* / glm5.2* / glm5.3* / minimaxm3* / kimik3* | `semianalysisai/cc-traces-weka-062126` |
| 其它（qwen3.5 等） | `semianalysisai/cc-traces-weka-062126-256k` |

下载进 `HF_HUB_CACHE`（/hf-cache），重复跑不重复下载。`WEKA_LOADER_OVERRIDE` 可强制换 loader。

## 产物（每个 CONC 点，写在 RESULT_DIR=/logs/agentic）

- `benchmark_command.txt` / `benchmark.log` — replay 命令与完整客户端日志
- `aiperf_artifacts/` — raw：`profile_export.jsonl`（逐请求 replay 记录）、`profile_export_aiperf.json`（含 `metadata.submission_valid`）、`server_metrics_export.json/.csv`（每秒服务端指标切片）
- `/logs/<RESULT_FILENAME>.json` — 聚合结果（`process_agentic_result` 生成，成功门）
- profiler 请求时：`/logs/traces/`（vllm worker trace，见 server-runbook.md）

## 其它注意

- 本地跑的是**真实验证**的投机解码；CI 的 AgentX 吞吐点用 golden acceptance length 合成验证（`infx/golden_al_distribution/`），两者数值不可直接比较。
- `DURATION` 默认 3600s；`AIPERF_EXPERIMENTAL_FAST=1` 可缩短到 20 分钟 profile（结果标 `submission_valid: false`，仅供调试）。
- 客户端脚本用 `set -e` 即可，**不要加 `-u`**：source 的 InferenceX 脚本不保证 nounset 安全（AGENTS.md 明确禁用）。
