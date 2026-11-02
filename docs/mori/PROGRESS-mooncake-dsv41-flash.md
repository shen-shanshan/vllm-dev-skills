# 进度跟踪：vLLM DeepSeek-V4.1-Flash 接入 MooncakeStoreConnector（ROCm）

> 用途：换机器或开新会话时，先读这份文档恢复上下文，再接着做。
> 最后更新：2026-11-02。方案 B 已完成；GSM8K 精度测试已完成，结论是 Mooncake 不影响精度、ROCm 快路径保持开启。
> 每次推进后，同步更新这几节：「代码与分支状态」「当前结论」「测试进展」「未完成事项」。

## 1. 任务概述

- **最终目的**：在 MI355X 上，用 DeepSeek-V4.1-Flash 对比 UMBP 和 Mooncake 做 KV cache offloading 的性能。
- **本阶段**：先确认 Mooncake 在 AMD DeepSeek-V4.1-Flash 上能用、精度正常，暂不测性能。
  - 范围由用户限定：单机、TP2、只用 DRAM、只测 embedded 模式。
  - vLLM 启动参数沿用 AgentX recipe `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml` 的 `override_tp2_c64`，和 UMBP 的测试完全一致。
- **姊妹任务**：UMBP standalone，见 [`PROGRESS-umbp-dsv41-flash-standalone.md`](PROGRESS-umbp-dsv41-flash-standalone.md)。
- **背景**：
  - main 上的 `deepseek_v41/amd/rocm.py` 只要检测到 `kv_transfer_config`，就会关掉 aiter sparse MLA 和 `extra_cache_nan_free` 两条快路径。
  - 这样一来，接 Mooncake 时模型跑的是慢路径，而 UMBP 分支已经通过 `only_restores_canonical_kv` 保留了快路径，两者没法公平对比。
  - 方案 B 的做法：让 Mooncake 也能通过 opt-in 开关声明这个 hook。

## 2. 代码与分支状态

| 项 | 值 |
|---|---|
| 仓库 | `/home/shashen/vllm/vllm` |
| 分支 | `rocm-dsv41-flash-mooncake`。改动尚未 commit，也没有 push |
| base | main `410ab341a8`。这和 `rocm-dsv41-flash-umbp` 的 base 是同一个，便于公平对比；本地 `main` 分支比它旧 |

改动文件（不含测试）：

| 文件 | 改动 |
|---|---|
| `kv_connector/v1/base.py` | 新增 classmethod `only_restores_canonical_kv(kv_transfer_config) -> False`。从 UMBP 分支原样复制 |
| `kv_connector/v1/multi_connector.py` | 覆写该 hook：所有子 connector 都返回 True 时才返回 True。从 UMBP 分支原样复制 |
| `models/deepseek_v41/amd/rocm.py` | 新增 `_has_foreign_kv_writer()`，替换原来 4 处 `kv_transfer_config is not None` 判断。从 UMBP 分支复制，docstring 去掉了 "e.g. UMBP" |
| `kv_connector/v1/mooncake/store/connector.py` | **方案 B**：`only_restores_canonical_kv` 返回 `kv_connector_extra_config.only_restores_canonical_kv` 的值，默认 False |
| `docs/features/mooncake_store_connector_usage.md` | 补充该开关的说明和安全前提 |

## 3. 当前结论

1. **Mooncake 在 AMD DeepSeek-V4.1-Flash 上能跑通，精度无损。** 第二遍有 91.9% 的 prompt token 从 Mooncake 恢复，GSM8K 和 baseline 持平（见 5.2）。
2. **方案 B 生效。** 打开开关后，日志里不再出现 "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA is ignored"，aiter sparse MLA 和 NaN-free 快路径都保持开启。
3. **为什么必须是 opt-in。**
   - Mooncake 的 key 由 `模型 basename@tp_rank@…@group@hash` 组成，不包含 KV cache layout 或 dtype 的指纹。
   - 所以只有两种情况下打开开关是安全的：
     - 单个 embedded 实例，池随进程一起销毁；
     - 每种 layout 配一个不同的 `cache_prefix`。
   - standalone-store、SSD，或者多个配置不同的实例共用一个 master 时，不能打开。
4. **容器里没有 RDMA，只能用 tcp。**
   - 容器里没有 `/dev/infiniband`，`ulimit -l` 只有 8MB，也没有 `CAP_IPC_LOCK`。
   - 用探针测 GPU→DRAM→GPU 的 put/get：`tcp` 数据一致；`hip` 在 Store 侧连 transfer engine 都起不来（`Failed to initialize transfer engine`）。
   - 精度测试不受影响。但做性能对比时，tcp 对 Mooncake 明显不利，需要先处理（见第 6 节）。

## 4. 设计要点

- 为什么不直接返回 True：这个 hook 是 classmethod，只拿得到 `kv_transfer_config`，判断不出部署形态（embedded 还是 standalone、单实例还是多实例、单机还是跨机）。无条件返回 True，就等于替所有 Mooncake 用户做了保证。
- 另一个方案 A（暂不做）：在 key 里加上每层 spec 的指纹，再无条件返回 True。这样会改变所有平台的 key 格式，需要 Mooncake 维护者同意。
- 对其他后端的影响：
  - hook 默认返回 False，开关默认关闭；唯一的调用方是 ROCm 上的 `deepseek_v41/amd/rocm.py`。
  - CUDA 和现有 Mooncake 用户的行为不变。

## 5. 测试进展

### 5.1 环境探针（2026-11-02）

脚本是 `/tmp/umbp_runs/mooncake/mc_probe.py`，在 `/tmp` 下，不会长期保留。做法：起 `mooncake_master`，`setup(..., protocol)` 后注册一块 GPU buffer，执行 `batch_put_from` 再 `batch_get_into`，最后比较数据。

| protocol | 结果 |
|---|---|
| `tcp` | ✅ 数据一致 |
| `hip` | ❌ `Failed to initialize transfer engine`，进程 core dump |
| `rdma` | 未测，容器里没有 `/dev/infiniband` |

### 5.2 E2E 精度：GSM8K 5-shot，TP2，GPU 0/1（2026-11-02，✅ 无损）

- 方法和 UMBP 一致：
  - baseline（不带 connector）跑一遍；
  - Mooncake 跑两遍：第 1 遍把 KV 写进池里，然后 `/reset_prefix_cache`，第 2 遍从 Mooncake 恢复。
- Mooncake 的配置（`mooncake_config.json`）：
  - embedded 模式，`protocol=tcp`，`metadata_server=P2PHANDSHAKE`；
  - 每个 rank 设 `global_segment_size=64GB`，池子总共 128GB，和 UMBP 测试的 128 GiB 对齐；
  - `local_buffer_size=4GB`，`kv_load_failure_policy=recompute`，`only_restores_canonical_kv=true`。
- lm_eval：`local-completions`、`num_concurrent=64`、`max_gen_toks=2048`，task 是本地的 `gsm8k_local`（复用 `umbp-dsv41-flash/`）。

| Run | strict-match | flexible-extract | prompt token 来自 Mooncake | GPU prefix 命中 | draft 接受率 / 平均接受长度 |
|---|---|---|---|---|---|
| baseline | 0.9204 ± 0.0075 | 0.9196 ± 0.0075 | 0 | 0.2% | 49.43% / 3.47 |
| Mooncake 第 1 遍 | 0.9212 ± 0.0074 | 0.9204 ± 0.0075 | 0 | 0.2% | 48.67% / 3.43 |
| Mooncake 第 2 遍 | 0.9333 ± 0.0069 | 0.9333 ± 0.0069 | **1,057,408 / 1,150,678 = 91.9%** | 0.8% | 48.90% / 3.44 |

- 三组之间的差异都在约 1.5 个 stderr 以内，属于 DSpark 概率采样带来的正常波动。第 2 遍 KV 大部分来自 Mooncake，精度没有下降。
- 和 UMBP 对照（旧 base `5cae6fd0c9`）：baseline 0.9212；UMBP 第 1 遍 0.9333，第 2 遍 0.9295，第 2 遍有 92.7% 的 prompt token 来自 UMBP。两者表现一致。
- 引擎日志：
  - 没有 Traceback，也没有 NaN（`grep nan` 搜到的 2 处是 `tenant_id` 的子串）；
  - 没有 "is ignored" 警告；
  - 所有 `*failed_keys` 都为 0，没有触发 recompute。
- mooncake_master：最终使用 1.60 GB / 128 GB，共 8258 个 key。
- 结果文件在 `mooncake-dsv41-flash/results/`：
  - `gsm8k_{baseline,mooncake_pass1,mooncake_pass2}.json`
  - `metrics_baseline.txt`、`metrics_mooncake_pass1.txt`、`metrics_mooncake_after_pass2_cumulative.txt`
- 完整日志在 `/tmp/umbp_runs/mooncake/gsm8k/{baseline,mooncake}/`，`/tmp` 下不会长期保留。

### 5.3 脚本索引（`mooncake-dsv41-flash/`）

| 脚本 | 用途 |
|---|---|
| `mooncake_config.json` | `MOONCAKE_CONFIG_PATH` 指向的配置：embedded、tcp、每个 rank 64GB |
| `run_mooncake_master.sh` | 启动 `mooncake_master --port 50051` |
| `run_vllm_dsv41_flash.sh` | recipe 参数加上 `override_tp2_c64`。`MOONCAKE=1` 时挂 MooncakeStoreConnector；`CANONICAL=true/false` 控制方案 B 的开关；`GPUS` 默认是 `0,1` |
| `run_gsm8k_config.sh` | 一次完整部署：起 master、起引擎、跑 lm_eval（`MOONCAKE=1` 时跑两遍）、最后清理。用法：`TAG=mooncake MOONCAKE=1 bash run_gsm8k_config.sh` |

## 6. 未完成事项（按优先级）

1. 是否 commit 或 push `rocm-dsv41-flash-mooncake`，由用户决定。
2. 性能对比前，先给容器挂上 RDMA。需要重建容器，加 `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1`。之后：
   - 用 `ibv_devinfo -d ionic_0` 确认设备可见；
   - 用探针测 `rdma` 协议，重点看 ionic 网卡上能不能注册 GPU 显存（dma-buf）；
   - 把 `mooncake_config.json` 改成 `protocol=rdma`。
   - 如果做不到，性能报告里要写明 Mooncake 走的是 tcp。
3. UMBP 和 Mooncake 的 AgentX 性能对比：同一个 base、同一份 recipe（`override_tp2_c64`）、池子都是 128GB、`DURATION=3600`。
4. 可选：在 `CANONICAL=false` 下跑一组性能，量化快路径带来的收益。

## 7. 已知限制

- 方案 B 的安全性依赖部署约束：key 不含 layout 指纹，见第 3 节第 3 条。
- 单机 RDMA 实际是网卡回环，带宽受网卡限制。UMBP 是 server 进程用 HIP IPC 映射 GPU 显存，再用 gather kernel 或 hipMemcpy 直接读写锁页 DRAM 池。两者的数据通路不同，性能报告里需要说明。
