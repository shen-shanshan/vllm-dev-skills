---
name: infx-bench-script-generator
description: Generate three local bash scripts (docker run / server start / AgentX client replay) to run an InferenceX srt-slurm recipe YAML benchmark manually on a local GPU server, without Slurm or srt-slurm. Given a recipe like dsv4/vllm/mi355x-fp4-mtp/agentic.yaml and an override variant (e.g. override_tp8_c56 or override_dep8_c64), optionally capture one complete torch profiler prefill or decode step via vllm serve --profiler-config, with configurable host paths for model weights, the InferenceX checkout, the trace dataset, and results. Scripts are written to the skill's outputs/<recipe>_<override>/ directory. Use when the user wants to run an InferenceX agentic benchmark locally, asks for local benchmark scripts for a recipe + override, or requests torch profiler prefill/decode step capture for a vllm AgentX run. Triggered by requests like: 我想在本地服务器上跑 dsv4/vllm/mi355x-fp4-mtp/agentic.yaml 的 override_dep8_c64 配置、帮我生成本地跑这个 benchmark 的脚本、用 torch profiler 采集一个完整的 prefill step、generate local benchmark scripts for an InferenceX recipe.
---

# InferenceX 本地 Benchmark 脚本生成器

根据一个 InferenceX srt-slurm 单机 recipe（`xxx.yaml`）+ 一个 override variant，生成三份在
本地 GPU 服务器上手动跑 AgentX benchmark 的脚本（不依赖 Slurm / srt-slurm / GitHub Actions）。

## 输入（缺失项向用户确认，路径可留空走默认）

1. recipe 路径（如 `dsv4/vllm/mi355x-fp4-mtp/agentic.yaml`，相对于 InferenceX 仓库的
   `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/`，或绝对路径）
2. override 名（如 `override_tp8_c56`、`override_dep8_c64`；也接受 `base`）
3. 是否采集 torch profiler trace：`off`（默认）/ `prefill`（一个完整 prefill step）/
   `decode`（一个完整 decode step）
4. 模型权重 host 路径（留空 = server 脚本在容器内下载，存 named volume）
5. InferenceX 仓库 host 路径（留空 = client 脚本在容器内 clone）
6. trace 数据集 host 路径（留空 = 挂载 host `~/.cache/huggingface`，复用缓存与 HF token）
7. 测试结果 host 目录（留空 = named volume 挂到 `/logs`，即 InferenceX 默认结果位置）

## 工作流程

1. **读 recipe 并自检**：读目标 yaml，确认 override 存在；需要 schema/merge 规则细节时读
   [`references/recipe-schema.md`](references/recipe-schema.md)（含 dsv4 mi355x 的 worked example）。
2. **运行生成器**（产物由脚本确定性生成，不要手写这三个文件）：

   ```bash
   python3 <skill_dir>/scripts/gen_bench_scripts.py \
     --recipe <recipe 绝对路径> --override <override 名> \
     [--model-path <host 路径>] [--repo-path <host 路径>] \
     [--trace-path <host 路径>] [--results-dir <host 路径>] \
     [--profile off|prefill|decode] \
     --out-root <skill_dir>/outputs
   ```

   PyYAML 缺失时报错，先 `python3 -m pip install pyyaml`。
3. **复核生成产物**（`outputs/<recipe路径以_分隔>_<override>/`）：
   - `variant.json` / `variant.env` 与三份脚本一致（server args 与 `roles.agg.args`、
     client env 与 `benchmark.env`、CONC 与 variant 一致）；
   - `01_docker_run.sh` 中没有任何 `-e` 环境变量（env 全部在 02/03 脚本内）；
   - `03_run_client.sh` 里 `IS_AGENTIC=1` 等必须先于 `source runtime_settings.sh`，
     功耗开关（`ENABLE_AGENTX_POWER=0` 等）必须在 source 之后。
   核对领域细节时读 [`references/server-runbook.md`](references/server-runbook.md) 与
   [`references/client-runbook.md`](references/client-runbook.md)。
4. **汇报**：三份脚本的路径、容器名，以及执行顺序：

   ```text
   1. bash outputs/<文件夹>/01_docker_run.sh            # host 上起容器
   2. docker exec -it <容器名> bash /infx-scripts/02_start_server.sh
   3. docker exec -it <容器名> bash /infx-scripts/03_run_client.sh
   ```

   结果位置：聚合结果 `/logs/<RESULT_FILENAME>.json`、raw replay `/logs/agentic/`、
   profiler trace `/logs/traces/`（映射到用户给的 host 结果目录或 named volume）。

## 注意事项（写入回答的提醒项）

- 只支持单机 vllm recipe；sglang/atom/multi-node 不在本 skill 范围。
- 本地跑的是**真实验证**投机解码；CI 发布数据用 golden acceptance length（合成验证），数值不可直接比较。
- dep8 变体需要 vllm-router 前端（02 脚本自动起）；router flags 未经真实集群外验证，
  首次运行若 router 报错先 `vllm-router --help` 核对。
- profiler 采集机制：`vllm serve --profiler-config '<json>'` + client 侧
  `POST /start_profile` / `/stop_profile` 界定窗口（AIPerf 本身无 --profile 支持）。
  prefill = `delay_iterations: 0, max_iterations: 1`；decode = `delay_iterations: 1, max_iterations: 1`。
  每 rank trace 文件命名与落盘时机需首次真机运行验证（回退方案见 server-runbook.md）。
- gated 模型/数据集需要 HF token：默认挂载 host `~/.cache/huggingface`，02 脚本从
  `/hf-cache/token` 读取并导出 `HF_TOKEN`。
- 冷加载 805 GiB 级 checkpoint 可达数小时（`VLLM_ENGINE_READY_TIMEOUT_S`），02 脚本会持续探测 `/health`。
