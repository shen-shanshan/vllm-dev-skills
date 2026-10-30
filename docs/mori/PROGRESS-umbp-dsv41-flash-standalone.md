# 进度跟踪：vLLM DeepSeek-V4.1-Flash 接入 UMBP standalone 模式（ROCm）

> 用途：换机器或开新会话时，先读这份文档恢复上下文，再接着做。
> 最后更新：2026-10-30（rebase 到 main `5cae6fd0c9`；GSM8K 精度测试完成，UMBP 无损）
> 每次推进后，同步更新以下几节：「代码与分支状态」「当前结论」「测试进展」「未完成事项」。

## 1. 任务概述

- **目标**：让 MI355X 上 vLLM 的 DeepSeek-V4.1-Flash，能把 prefix KV 卸载到节点本地的 MORI-UMBP `umbp_standalone_server`，并能恢复回来。
- **范围**：只做单机 standalone，不做 RDMA 跨机的 distributed 模式。
  - 用户要求：先保证能跑通。之后用户要求补测 GSM8K 精度（见 5.2）；性能暂不测。
  - 用户要求：不新增测试。等 `[Test][UMBP]` 系列 PR 合入后，自然会有对应用例。
- **RFC**：https://github.com/vllm-project/vllm/issues/57187
- **PR 描述正文**：[`pr-umbp-dsv41-flash-standalone.md`](pr-umbp-dsv41-flash-standalone.md)，标题为 `[ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4.1-Flash`。尚未开 PR。
- **设计方案**（用户已批准）：`/root/.claude/plans/lucky-popping-fog.md`
- **姊妹任务**：DeepSeek-V4-Pro 对应 PR #63002，进度见 [`PROGRESS-umbp-dsv4-standalone.md`](PROGRESS-umbp-dsv4-standalone.md)。本任务复用了其中的通用 connector 改动。
- **背景资料**：[`umbp-background.md`](umbp-background.md)、[`umbp-install.md`](umbp-install.md)、[`umbp-perf-tuning.md`](umbp-perf-tuning.md)

## 2. 代码与分支状态

| 项 | 值 |
|---|---|
| 仓库 | `/home/shashen/vllm/vllm` |
| 分支 | `rocm-dsv41-flash-umbp`，尚未 push |
| base | upstream/main `5cae6fd0c9`（用户 2026-10-30 rebase），已包含 `[UMBP][1/10]`–`[6/10]` |
| cherry-pick：`[UMBP][7/10]` #61795 standalone | `fbd0e8af99` `6310717d5f` `5e9943a09c` `1dcf75df82` `376f06dbc2` |
| cherry-pick：`[UMBP][9/10]` #61797 lazy offload | `100deaa070` `66752a7a73` `3f3db5b35e` |
| 本任务改动 | 已由用户 commit：`aa3076f283 support umbp for dsv41-flash` |

cherry-pick 的说明：
- 用了 `-x`，所以每个 commit 都记录了原 hash。
- 只挑了非测试 commit，所有 `[Test]` 和 `[CI]` commit 都跳过了，#61794（6/10）、#61796（8/10）、#61798（10/10）整个没拿。这样做是为了减少后续 rebase 时的冲突。
- `79d3181336` 和 PR9 中的 `50e01f2a42` patch-id 相同，只取了一次。
- 拿完之后，UMBP 目录和 `docs/features/umbp_store_connector_usage.md` 都与 #61797 head 完全一致。

注意：
- 本地没有 `gh`。PR 用 `git fetch upstream pull/<N>/head:<branch>` 拉取，已拉取的有 `umbp-pr-6/7/8/9`。
- 等 7/10 和 9/10 合入 main 后，rebase 时这些 cherry-pick 应该会自动消掉；如有冲突，以 main 为准。

## 3. 当前结论

- **跑通了**：TP2 单机 standalone 端到端可用。restore 路径命中 24320 token，恢复后的回答与 GPU 直出完全一致，aiter sparse MLA 快路径也保留了。详见 5.1。
- **精度无损**：GSM8K 5-shot，baseline 0.9212，UMBP pass1 0.9333，pass2（92.7% prompt token 从 UMBP 恢复）0.9295（strict-match），差异都在噪声内；MTP 接受率不变。详见 5.2。
- **v4.1 比 v4-Pro 简单，原因有两个**：
  - compressor state 是 `CircularBufferSpec`，不参与 prefix caching。命中按 128 对齐，open group 正好为空，和 GPU prefix cache 命中时的情况一样。
  - v4.1 在 ROCm 上本来就关掉了 prefix-cache SWA replay（`deepseek_v41/attention.py:553`），SWA 组照常做 prefix cache，并由 UMBP 一起存储。
  - 因此**不需要** #63002 里的 `requests_sliding_window_replay` 和 `rebuilt_by_replay`。
- **代价**：SWA 组也要存，对象数比 v4-Pro 多。若要在 ROCm 上打开 replay，需要先解决 prefill kernel 的 window clamp 问题，不在本任务范围内。

## 4. 代码设计（相对 main + 7/10 + 9/10）

### 4.1 保留 ROCm 快路径
- `base.py` 新增 hook `KVConnectorBase_V1.only_restores_canonical_kv`，默认返回 `False`。
- `MultiConnector`：所有子 connector 都返回 True 时，它才返回 True。
- `UMBPStoreConnector` 返回 True。
- `deepseek_v41/amd/rocm.py` 新增 `_has_foreign_kv_writer()`，替换掉 4 处 `kv_transfer_config is not None`：
  - metadata builder 和 attention 的 `_aiter_sparse_mla_enabled`；
  - `_trust_dsv4_extra_cache_nan_free`。

### 4.2 TP 复制 KV 去重（从 #63002 复制）
- standalone runtime 的 `shares_pool_across_ranks = True`。
- worker 用 `tp_replicated_group_ids()` 找出 MLA 和 SWA-MLA 组，通过 handshake 上报给 scheduler。
- 这些组用 `tp*` 作 key：rank 0 负责存，所有 rank 都能读。
- residency 计算改用 `keys_per_object()`。
- 如需关闭，设 `dedup_tp_replicated_kv: false`。PCP/DCP 下不会去重。

### 4.3 对其他后端的影响
- **非 ROCm**：hook 唯一的调用方是 v4.1 的 ROCm 文件，所以 CUDA、XPU、CPU 不受影响。
- **其他 connector**：hook 仍返回 False，快路径照旧关闭。
- **embedded UMBP**：不开启去重，key 格式不变。

## 5. 测试进展

### 5.1 单机 smoke：TP2，GPU 6/7（2026-10-30，✅ 通过）
- **配置**：InferenceX `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml` 的 base + `override_tp2_c64`。
  - DSpark MTP 5、monokernel、mxfp4 indexer、block 128，engram `cpu_offload=true`。
  - UMBP 池 128 GiB DRAM，不用 hugepages，不开 SSD。
  - MORI 为 v1.2.3.post1，没有 `lookup_timeout_ms`，日志会给出 warning，属于预期行为。
- **用户决定**：只测 TP2，不测 TP4。
- **流程**：
  1. 发送约 20K token 的两轮对话。
  2. 调用 `/reset_prefix_cache`。
  3. 重发第二轮。

| 检查项 | 结果 |
|---|---|
| `external_prefix_cache_hits_total` 增量 | 24320 token（190 个 128 块） |
| 恢复后回答 vs GPU 直出（greedy） | 一致 |
| `AITER_TRITON_SPARSE_MLA is ignored` warning | 无 |
| NaN / Traceback / ERROR | 无 |

- 结果在 `/tmp/umbp_runs/smoke/dsv41_tp2_smoke/`，包括 `smoke.json`、`engine.log`、`umbp_server.log`。注意 `/tmp` 不持久。
- 引擎启动约 40 分钟（权重加载、warmup、cudagraph capture）。
- **未验证**：TP 去重是否实际生效。server 的 info 日志不打印 key，日志里看不到 `tp*`。

### 5.2 E2E 精度：GSM8K 5-shot，TP2，GPU 0/1（2026-10-30，✅ 无损）

- **代码**：main `5cae6fd0c9` + 7/10 + 9/10 cherry-pick + `aa3076f283`。
- **配置**：与 5.1 相同的 recipe（`override_tp2_c64`），两组服务 flag 完全一致（都设 `VLLM_SERVER_DEV_MODE=1`），UMBP 组只多 `--kv-transfer-config`；UMBP 池 128 GiB。
- **测法**（同 v4-Pro）：baseline 部署跑一遍；UMBP 部署跑两遍，pass1 存 KV，然后 `/reset_prefix_cache` 清掉 GPU 前缀缓存，pass2 重发相同 prompt，从 UMBP 恢复。
- **lm_eval**：0.4.13，`local-completions`，`num_concurrent=64`，`max_gen_toks=2048`，全部 1319 题。

| Run | strict-match | flexible-extract | 接受率 | 平均接受长度 | UMBP / GPU 前缀命中 |
|---|---|---|---|---|---|
| Baseline | 0.9212 ± 0.0074 | 0.9204 ± 0.0075 | 48.69% | 3.43 | 0 / 0.04% |
| UMBP pass1（存） | 0.9333 ± 0.0069 | 0.9325 ± 0.0069 | 48.35% | 3.42 | 0 / 0.04% |
| UMBP pass2（恢复） | 0.9295 ± 0.0071 | 0.9287 ± 0.0071 | 48.87% | 3.44 | **92.7%** / 0 |

- pass2 中 1,150,678 个 prompt token 有 1,066,752 个从 UMBP 恢复。
- 最大差异是 baseline vs pass1 的 +1.2 个点，约 1.2 个合并标准误，属于噪声；UMBP 两遍都不低于 baseline。
- 接受率三组都在 48.5% 左右，说明恢复的 KV 没有损坏。
- engine 日志无 NaN / Traceback / `AITER_TRITON_SPARSE_MLA is ignored`。
- 原始结果在 [`umbp-dsv41-flash/results/`](umbp-dsv41-flash/results/)；完整产物在 `/tmp/umbp_runs/gsm8k/{baseline,umbp}`（不持久）。

踩过的坑：
- `/shared/datasets--openai--gsm8k` 是 HF hub cache 布局，离线时 `datasets` 无法按 `openai/gsm8k` 解析。改用自定义 task `gsm8k_local`（`umbp-dsv41-flash/gsm8k_task/`），`dataset_path` 直接指向 snapshot 目录。
- `--load-format fastsafetensors` 在 TP2 下会一次性在每张卡上申请 91.55 GiB，直接 OOM，引擎起不来，已从脚本中去掉。权重在 page cache 里时，默认加载约 6 分钟即可起服务。

### 5.3 脚本索引（`umbp-dsv41-flash/`）
| 脚本 | 作用 |
|---|---|
| `run_vllm_dsv41_flash.sh` | 启动引擎。`UMBP=0/1` 切换开关；`GPUS` 默认 `6,7`；`UMBP_EXTRA` 追加 connector 选项 |
| `smoke_umbp.sh` | 拉起 server 和引擎，跑 restore 检查。用法：`TAG=<name> bash smoke_umbp.sh` |
| `run_gsm8k_config.sh` | 一次完整部署跑 GSM8K。`TAG=baseline UMBP=0` 跑一遍；`TAG=umbp UMBP=1` 跑 pass1 + reset + pass2，并在每遍前后快照引擎计数器 |
| `run_gsm8k.sh`、`gsm8k_task/gsm8k_local.yaml` | lm_eval 命令和读本地 snapshot 的 gsm8k task |
| `../umbp-dsv4/run_umbp_server.sh` | 复用 v4-Pro 的 UMBP server 脚本，`UMBP_DRAM_CAPACITY` 的单位是字节 |

踩过的坑：
- 容器里已经导出了 `HIP_VISIBLE_DEVICES=0,...,7`，脚本里写 `${HIP_VISIBLE_DEVICES:-...}` 不会生效。已改为使用独立变量 `GPUS`。
- 输出目录 `/tmp/umbp_runs` 不存在时，重定向会直接失败。先执行 `mkdir -p`。
- GPU 0–5 常被容器外的进程占用。启动前先确认空闲显存，可以用 `rocm-smi --showmemuse`，也可以在 `.venv` 里用 `torch.cuda.mem_get_info(i)`。

## 6. 未完成事项（按优先级）

1. 是否 push、何时开 PR，由用户决定。
2. （可选）确认 TP 去重生效：开 server 的 debug 日志，或者临时统计 key 数量。
3. ~~精度：GSM8K~~ 已完成（5.2）。
4. 性能：AgentX agentic，按 InferenceX 默认 `DURATION=3600`，UMBP 开/关对比。用户说"先不测"。
5. 跟进上游：
   - 7/10 和 9/10 合入后 rebase。
   - 与 #63002 共享的改动，谁后合入谁 rebase，以 main 为准。
   - `[Test][UMBP]` 合入后补跑相关单测。

## 7. 已知限制 / 后续工作

- 只支持单机。
- ROCm 上没有 SWA replay，会额外存 SWA 组。
- 本任务与 #63002 的通用改动重复，合入顺序会影响谁需要 rebase。
