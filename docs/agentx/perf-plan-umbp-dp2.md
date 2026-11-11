# 性能测试方案：DeepSeek-V4.1-Flash MI355X DP2，UMBP 与 #4992 基线对比（c128 / c256 / c384）

状态：**已完成**（2026-11-11）。6 个 fast 点全部跑完，结果见 `results/agentic_dp2_fast_by_concurrency.md`。用户决定：去掉 c128 点，不再做 3600 s 复测。

用户决定（2026-11-11）：
1. extensible KV 按 CI 的实际行为处理：两组都不加 `--no-enable-extensible-kv-cache`。
2. 先用 agentx-fast 模式（1200 s）跑完 6 个点并汇报，再决定是否补跑完整的 3600 s。
3. 同意第 7 节的脚本改动。
4. 冒烟缩短为约 15–20 分钟：只用 `smoke_umbp.sh` 起一次 S1，检查两个 DP rank 都就绪、恢复路径有 external hit；不再单独跑 300 s 的 agentic，这部分由第一个 fast 点（B0-c256）覆盖。

## 1. 目标

按当前 InferenceX 配置（分支 `dsv41-flash-umbp` 上的 `c656ff100`），用本地最新的 vLLM 代码，测量 UMBP standalone 在 c128、c256、c384 三个点上的收益，并与 [#4992](https://github.com/SemiAnalysisAI/InferenceX/pull/4992) 的 baseline 对比。

## 2. #4992 现有的 baseline 数据

#4992 只给出了**本地测的 c256**：8× MI355X，镜像 `nightly-083fbac02`，每个点跑 1 小时。**c128 和 c384 没有测，也还没有 CI sweep 结果。**

| Prefill 调度方式 | tok/s/GPU | p90 interactivity | TTFT p50 / p90 |
|---|---|---|---|
| cadence interval 11（#4992 选定的配置） | 310,739 | 16.6 | 1.44 s / 4.02 s |

因此，三个点都需要**在本地用同一份代码补跑不开 offloading 的 B0**，才能做到同条件对比。#4992 的 c256 数据只用来校验本地 B0 是否可信。

## 3. 测试环境

| 项 | 取值 | 说明 |
|---|---|---|
| 容器 | 当前容器，即镜像 `nightly-96710245`，和我们 recipe 用的是同一个镜像 | |
| vLLM 代码 | 本地 main `595b1514e8`（2026-11-10），比镜像新 59 个 commit | 这 59 个 commit 没有改 ROCm C++ kernel；native 部分（`*.so` 和 `vllm-rs`）仍是镜像 `96710245` 里的版本，Rust 前端的 4 个新 commit 不生效 |
| 相关改动 | #66605：connector 改为自己声明 KV 内存访问方式 | UMBP 声明为 `REGISTERS`，ROCm 上仍会自动关闭 extensible KV，无需手动处理 |
| GPU | 0、1 | 目前 8 张卡都空闲 |
| 主机内存 | 2.75 TB，可用约 2.6 TB | 449 GB 的池子足够 |
| 前端 | vllm-router 0.1.14，`consistent_hash`，`--intra-node-data-parallel-size 2`；client 带 `X-Session-ID` | |
| client | InferenceX `python3 -m infx.bench agentic`，`DURATION=3600`（InferenceX 默认值） | |

## 4. 两组配置（除 UMBP 外完全一致，与 recipe 对齐）

两组共用的引擎参数取自 recipe 的 base 加上 `override_dp2_umbp_c*`：
- `--tensor-parallel-size 1 --data-parallel-size 2`，不开 EP，不开 `--monokernel`；
- `--prefill-schedule-interval 11`；
- engram `cpu_offload: true`；
- `--max-cudagraph-capture-size 1024`，`--max-num-batched-tokens 16384`；
- block size 128，mxfp4 indexer，DSpark 5 个 draft token，synthetic AL 3.51；
- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE`，rust 前端。

| ID | 配置 |
|---|---|
| **B0** | 不开 offloading，参数等同于 #4992 的 `override_dp2_c*` |
| **S1** | B0 加 UMBP standalone：`UMBPStoreConnector`、`kv_load_failure_policy: recompute`，池子 449 GB（等于 recipe 里的 `TOTAL_CPU_DRAM_GB`），4 KiB 页 |

**extensible KV cache**：两组都按 recipe 的默认行为，不加 `--no-enable-extensible-kv-cache`，和 CI 保持一致。B0 会开启 extensible KV，S1 会因为 UMBP 被自动关闭。两组的 GPU KV 容量（token 数）会从引擎日志里取出来一并报告，如果差别明显，会在结论里单独说明。

## 5. 测试点与顺序（共 6 次完整运行，每次约 1.4 h）

| # | ID | 并发 |
|---|---|---|
| 1 | B0-c256 | 256 |
| 2 | S1-c256 | 256 |
| 3 | B0-c384 | 384 |
| 4 | S1-c384 | 384 |
| 5 | B0-c128 | 128 |
| 6 | S1-c128 | 128 |

- 先跑 c256：B0-c256 跑完后与 #4992 的 310,739 tok/s/GPU 对比，**偏差在约 5% 以内才继续**。偏差过大说明本地环境和他的不可比，先汇报再决定。
- 每个点 B0 和 S1 相邻跑，减少时间漂移带来的影响。
- 串行执行：CI 上单机作业独占一个节点，UMBP server 也是全节点共享的，并行跑会互相干扰。

**冒烟（约 1 h，不计入上面 6 次）**：用 S1 配置在最新代码上起服务，检查以下几项：
- 两个 DP rank 都就绪；
- 日志里能看到 cadence 生效；
- 恢复检查：先发一段长对话，然后 `/reset_prefix_cache`，再发一次，external hit > 0；
- 用 c256 跑 300 s，确认 router、结果 JSON 整个流程能跑通。

**总耗时**：冒烟约 1 h，加上 6 个点约 8.5 h，**合计约 9.5 h**。

可选的省时方案：先用 InferenceX 的 agentx-fast 模式（1200 s，每个点约 40 min）把 6 个点筛一遍，约 4 h；再只对有收益的点用 3600 s 复测。按约定，最终结论一律以 3600 s 的结果为准。

## 6. 指标与判断规则

- **主要指标**：每 GPU 总吞吐（tok/s/GPU）、p90 interactivity、TTFT p50 / p90。这三项和 #4992 的表格口径一致。
- **辅助指标**：
  - TPOT mean / p90、E2E；
  - 请求数、失败数、preemption；
  - GPU prefix 命中率、从 UMBP 恢复的 prompt 比例、实际计算的 prompt 比例；
  - 两个 DP rank 的负载是否均衡；
  - UMBP 池子是否写满或发生淘汰。
- **判断规则**（单次运行的噪声约 3%）：
  - 吞吐提升 ≥ 3%，且 TTFT p90 没有变差：算有收益；
  - 吞吐变化在 ±3% 以内：算持平；看 TTFT 有没有改善，再决定这个点要不要保留 UMBP；
  - 吞吐下降超过 3%：算负收益，要分析原因，并考虑在 PR 里去掉这个点。
- **c384 的额外检查**：cudagraph 上限是 1024，每个 rank 最多约 192 × 6 = 1,152 个 token，统计 decode step 超过 1024 token（走 eager 模式）的比例。如果这个比例高，在结论里给出是否调高上限的建议。

## 7. 需要修改的脚本（`docs/mori/umbp-dsv41-flash/`）

| 脚本 | 改动 |
|---|---|
| `run_vllm_dsv41_flash.sh` | 新增 `PREFILL_SCHEDULE_INTERVAL` 变量（默认为空，即不加该参数）；本次设为 11。运行时设 `QR_QUANT=NONE`、`EXTENSIBLE_KV_CACHE=1`（即不加 `--no-enable-extensible-kv-cache`） |
| `run_config.sh` | `TOTAL_CPU_DRAM_GB` 改为按 10^9 字节换算，与 InferenceX 的口径一致（449） |
| 其他 | 无改动。`DP=2` 时启动 router 和 client 走 30000 端口的逻辑，上次 c256 已经验证过 |

第 0 步：容器重启后 uv 安装的 Python 没了，AgentX client 的 venv 已失效，需要先执行 `uv python install 3.11` 修复。

运行命令（确认后执行）：

```bash
cd /home/shashen/vllm/vllm-dev-skills/docs/mori/umbp-dsv41-flash
export GPUS=0,1 DP=2 SYNTHETIC_AL=3.51 DURATION=3600 RUNS_DIR=/tmp/umbp_runs/agentic_dp2_cadence
export PREFILL_SCHEDULE_INTERVAL=11 QR_QUANT=NONE EXTENSIBLE_KV_CACHE=1
export UMBP_DRAM_CAPACITY=449000000000
for c in 256 384 128; do
  CONC=$c TAG=b0_dp2_c$c UMBP=0 bash run_config.sh
  CONC=$c TAG=s1_dp2_c$c UMBP=1 bash run_config.sh
done
```

## 8. 产出

- 结果汇总：`docs/agentx/results/agentic_dp2_umbp_vs_4992.md`，包含三个点的 B0 / S1 对比表，以及 c256 与 #4992 的交叉校验。
- 原始 JSON 和引擎指标快照放在同一目录。完整产物在 `/tmp/umbp_runs/agentic_dp2_cadence/`，不长期保留。
- 测完后把结论补进 PR 描述的「Why」部分，明确标注为本地运行结果。

## 9. 风险

- 本地的 native 部分（`*.so`、`vllm-rs`）是 `96710245` 的版本，CI 用的是同一个镜像，但 Python 代码比 CI 新 59 个 commit，结果可能与之后的 CI 有小幅差异。
- 我们的 B0 和 #4992 用的镜像不同（`96710245` 对 `083fbac02`）。所以 UMBP 的收益只用本地 B0 / S1 的差值来衡量，#4992 的数据只作为参考。
- B0 和 S1 的 extensible KV 设置不同（见第 4 节），这一点和 CI 实际运行的情况一致，但 KV 容量会作为一个变量单独报告。
