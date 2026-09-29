# Server 脚本领域知识（02_start_server.sh）

## ROCm 容器起法（docker run，host 侧）

```bash
docker run -d --name <name> \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --entrypoint /bin/sleep \
  <mounts...> <image> infinity
```

- `--network host`：客户端在同一容器内访问 localhost，且 RDMA/指标抓取不受 NAT 影响。
- `--entrypoint /bin/sleep`：vllm 镜像可能带 ENTRYPOINT，用它保证容器只是"睡着的壳"，server 由 02 脚本经 `docker exec` 显式启动。
- env 一律不进 docker run（用户要求），在 02/03 脚本里 export。
- HF token：默认挂载 host `~/.cache/huggingface` → `/hf-cache`，02 脚本从 `/hf-cache/token` 读 token。gated 模型（如 DeepSeek-V4-Pro）没有 token 会下载失败。

## vllm serve 启动

- 模型路径：本地挂载/下载后 `vllm serve /model ...`（`model.path` 去掉 `hf:` 前缀仅用于下载）。
- args 来自合并后 `roles.agg.args`，kebab-case → CLI flag（规则见 recipe-schema.md）。
- env 来自 `roles.agg.env`，逐条 export；`SETUP_PIP_PACKAGES` 单独转成 `python3 -m pip install`。
- 就绪探测：轮询 `http://localhost:8000/health`，超时用 `VLLM_ENGINE_READY_TIMEOUT_S`（dsv4 冷加载可达 3h，recipe 设 10800s）。
- 日志写 `/logs/server.log`（nohup 后台 + 2>&1）。

## vllm-router（dep8 变体）

engine ready 后启动（`pip install vllm-router==0.1.14` 来自 frontend.env.SETUP_PIP_PACKAGES）：

```bash
nohup vllm-router \
  --worker-urls http://localhost:8000 \
  --intra-node-data-parallel-size 8 \
  --host 0.0.0.0 --port 30000 \
  <frontend.args 展开：--policy consistent_hash --request-timeout-secs 14400 --disable-retries> \
  > /logs/router.log 2>&1 &
```

- 单节点 DP8 = 一个 worker 基础 URL + `--intra-node-data-parallel-size 8`（srt-slurm 文档的 DP 例子语义）。
- Router 不转发引擎 metrics：client 必须直连 `http://localhost:8000/metrics` 抓指标。
- **未在真实集群外验证**：router 如何从基础 URL 派生 8 个 rank 端点、flags 是否有出入。生成脚本失败时先 `vllm-router --help` 核对；首次真机运行需验证 `http://localhost:30000/health` 与请求路由。

## torch profiler（--profile prefill|decode）

vLLM（nightly e975732… 之后，含本 skill 目标镜像）的采集链路：

1. **`vllm serve --profiler-config '<json>'`**（JSON 字符串，字段见 `vllm/config/profiler.py` 的 ProfilerConfig）。只有启动时带了 `--profiler-config`，`/start_profile`、`/stop_profile` 路由才会挂载。
2. **`POST /start_profile`**（无 body，旧版 num_steps/profile_by_stage 参数已废弃）→ 引擎开始按配置的 iteration schedule 采集；**`POST /stop_profile`** → 停止并导出 trace 到 `torch_profiler_dir`。

单步选择靠 schedule（引擎每个 prefill 步、每个 decode token 各算一个 iteration）：

| 想采集 | delay_iterations | max_iterations | 原理 |
| --- | --- | --- | --- |
| 一个完整 prefill step | 0 | 1 | start_profile 后第一个 iteration = 首个真实请求的 prefill |
| 一个完整 decode step | 1 | 1 | 跳过 prefill iteration，采下一个 = 首个 decode iteration |

生成的 profiler JSON（`torch_profiler_use_gzip` 默认 true → `.trace.json.gz`）：

```json
{"profiler":"torch","torch_profiler_dir":"/logs/traces",
 "torch_profiler_activities":["CPU","CUDA"],
 "delay_iterations":0,"max_iterations":1,"ignore_frontend":true}
```

- `ignore_frontend: true`：关闭 AsyncLLM 前端 profiling，只留 worker 的 CPU/GPU trace（delay/limit 组合时 vllm 自己也会警告前端开销）。
- **首次真机运行需验证**：① 每 rank trace 的文件命名（TP8 = 8 个 worker 文件）；② trace 是在 max_iterations 到达时自动落盘还是等到 /stop_profile（client 脚本两者都做了，stop_profile 在 replay 结束后必发）。
- **回退方案**（若单步窗口采不到东西）：去掉 `delay_iterations`/`max_iterations`，让 profiler 覆盖整个 replay，再用 `execute_context_*` step 注解后处理切出想要的步（可配合 vllm-vs-atom-decode-trace-comparison skill 的 extract_trace.py）。
