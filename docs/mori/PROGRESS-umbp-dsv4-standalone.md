# 进度跟踪：vLLM DeepSeek-V4-Pro 接入 UMBP standalone 模式（ROCm）

> 用途：换机器或开新会话时，先读这份文档恢复上下文，再接着做。
> 最后更新：2026-10-28（agentic 性能对比完成）
> 每次推进后请同步更新「当前状态」「测试进展」「未完成事项」三节。

## 1. 任务概述

- **目标**：在 vLLM 里让 AMD（MI355X）上的 DeepSeek-V4-Pro 把 prefix KV 卸载到节点本地的 MORI-UMBP `umbp_standalone_server`，之后再恢复回来。
- **范围**：只做单机 standalone 模式，不做 RDMA 跨机的 distributed 模式。
- **RFC**：https://github.com/vllm-project/vllm/issues/57187
- **我们的 PR**：https://github.com/vllm-project/vllm/pull/63002，PR 描述正文见 [`pr-umbp-dsv4-standalone.md`](pr-umbp-dsv4-standalone.md)。
- **依赖**：只依赖 base PR #58568（`[UMBP][1/10]`–`[10/10]`）。**不依赖** Kimi-K3 相关的 #59276、#59281（同事确认："It depends on the base PRs only, not on Kimi-K3 support PR"）。
  - 已合入 main：`[1/10]` #61789 至 `[5/10]` #61793（`[5/10]` 是 embedded DRAM runtime，2026-10-28 合入）。
  - 还在 review：`[4/10]`–`[10/10]`。standalone 模式在 `[7/10]` #61795。
- **参考实现**：
  - SGLang 的 DSv4 + UMBP：`/home/shashen/vllm/sglang`。只存压缩池，命中后重算最后一个窗口；TP 复制的 KV 用同一个 key。
  - MORI 源码：`/home/shashen/vllm/mori`。这是个人 fork 的 main，没有 tag。编译用的是 tag worktree，见第 6 节。
  - 背景知识：[`umbp-background.md`](umbp-background.md)。

## 2. 代码与分支状态

| 项 | 值 |
|---|---|
| 仓库 | `/home/shashen/vllm/vllm` |
| 分支 | `rocm-dsv4-umbp`（`origin` = fork `shen-shanshan/vllm`，`upstream` = `vllm-project/vllm`） |
| 本地 base | `fa3b9d1377`（2026-10-29 用户 rebase 到的最新 main，包含 `[5/10]`）。提交 1 = `c6e94502f1`，提交 2 = `387c1975ac` |
| origin 上的 base | `803bc13cf9`（旧 base；两个提交 `git range-diff` 完全一致，只是 base 不同） |
| 提交 1 | `a4d93efad4 support umbp for dsv4`：#58568 的 `[4/10]`–`[10/10]` stack 和 DSv4 增量**压在一个提交里** |
| 提交 2 | `2f2b03baf5 update`：已发布 MORI 没有 `lookup_timeout_ms` 时的兼容处理，以及 DSpark MTP 的测试 |
| 备份分支 | `pre-align-backup`，由 `dev/init.sh` 在改写历史前自动创建 |

注意：
- `dev/init.sh`（未跟踪）会把分支 rebase 到 Docker 镜像的构建 commit，所以本地和 origin 会显示 "diverged"。推送时需要 `git push -f origin rocm-dsv4-umbp`，**推送前先确认**。
- 本地 `gh` CLI 不可用，PR 信息用 `git fetch upstream pull/<N>/head:refs/remotes/pr/<N>` 拉取。

### 2026-10-29 rebase 冲突的处理（以 main 上已合入的 UMBP 代码为准）
- `runtime/embedded.py`：用 main 的版本，再加回 standalone 需要的 `serve_lookups` 参数。结果和 #58568 head 完全一致。
- `runtime/__init__.py`：在 main 的基础上补注册 standalone，去掉重复的 embedded 注册。结果和 #58568 head 一致。
- `test_umbp_embedded.py`：main 的 4 个测试和 2 个 fixture 原样保留（包括 `socket_dir`）；我们旧版的 `test_lookup_server_refuses_...` 换成 main 的写法；其余 stack 和 DSv4 的测试保留。
- 文档：删掉和 main 的 `umbp_store_connector_usage.md` 重复的 `umbp_embedded_dram.md`；把它独有的 `lazy_offload*` 两个选项按 #58568 的写法补进 usage 文档；README 保留 main 的条目，再加上 standalone 的链接。
- 语义冲突：main 的 embedded 给 load 加了超时（`_load_deadlines`）。为此带入了 #58568 的 `c174da7a0f`（standalone 的 poll 也检查 load 超时）和配套测试 `5625615f59`。
- 验证：ruff 0.14.0 通过；单元测试和回归测试 530 passed，另外 2 个失败是环境原因（gated Llama、缺 ray）。

## 3. 当前结论

1. 方案成立。DP8 + DSpark MTP 下跑 GSM8K，开 UMBP 后精度无损（数据见第 5 节）。
2. DSv4 不能直接套用通用 UMBP connector，原因有三个：
   - SWA 和 compressor state 组的 block 很小（4/8 token），会碎成海量小对象；
   - MLA latent 在每个 TP rank 上都一样，TP8 时会存 8 份；
   - 只要配置了任何 KV connector，ROCm DSv4 就会关掉 aiter sparse MLA 和 NaN-free 快速路径。
   这三点都已在 PR 中解决，见第 4 节。
3. 已发布的 MORI `v1.2.3.post1`（ROCm 镜像固定的版本）以及 ROCm/mori main 都**没有** `UMBPStandaloneProcessConfig.lookup_timeout_ms`。我们的分支在这种情况下会打 warning 后继续启动，lookup 不设超时。**不要**在 `kv_connector_extra_config` 里配 `lookup_timeout_ms`，配了会报错。
4. MTP 方面：只验证了 DSpark（`"method": "dspark"`）。EAGLE 风格的 `"method": "mtp"` 在开启 replay 时会在启动阶段被拒绝，需要设 `rebuild_sliding_window: false`。

## 4. 代码设计（DSv4 增量，相对 #58568）

### 4.1 只存压缩 KV，滑动窗口靠 replay 重建
- `KVConnectorBase_V1.requests_sliding_window_replay()`（`v1/base.py:225`）默认返回 False。`MultiConnector` 要求所有子 connector 都为 True。`UMBPStoreConnector` 返回 True，可通过 `rebuild_sliding_window: false` 关闭。
- `dsv4_swa_replay_enabled()`（`models/deepseek_v4/compressor.py:42`）只有同时满足以下条件才返回 True：ROCm、V2 model runner、`swa_bounded_replay` 打开、没有 PCP、connector 请求了 replay。
- 满足时：SWA cache 设 `bounded_replay=True`；C4/C128 compressor state 设 `rebuilt_by_replay=True`。这是新加的 `SlidingWindowMLASpec.rebuilt_by_replay` 字段（`v1/kv_cache_interface.py:1019`），作用是让该组不参与 prefix cache，同时不引入第二个 replay 窗口。
- `models/deepseek_v4/amd/model.py`：开启 replay 时，复用 `DeepseekV41ModelState` 来传递 `replay_start`。ROCm SWA builder 本来就会在 `replay_start` 处截断，**kernel 不用改**。
- 最终效果：UMBP 只存 C4/C128 latent 和 C4 indexer（每个对象 256 token）。命中后按 block 对齐，再重算最后 128 token 的窗口。

### 4.2 TP 复制的 KV 去重（`umbp/{data,connector,scheduler,worker}.py`、`runtime/{base,standalone}.py`）
- `tp_replicated_group_ids()`（`umbp/data.py:45`）：组内全是 MLA 风格的 spec、`tp_size > 1`、且没有 DCP/PCP 时，判定该组为 TP 复制组。
- `BlockIdentityCodec` 给复制组用 `tp*` key，每个对象只生成一个 key。lookup、residency 检查、lazy offload、KV event 都按 `keys_per_object` 计数。
- 写入只在 TP rank 0 做。其他 rank 丢掉自己的 store plan，但照常完成 store event，并加载共享对象。
- 只有 runtime 跨 rank 共享同一个池时才去重（`IUMBPRuntime.shares_pool_across_ranks`，standalone 为 True）。embedded 模式保持不变。
- 复制组信息通过 `UMBPHandshakeMetadata` 由 worker 传给 scheduler。
- 新增开关 `dedup_tp_replicated_kv`，默认 True。

### 4.3 保留 ROCm DSv4 快速路径
- `KVConnectorBase_V1.only_restores_canonical_kv()`（`v1/base.py:215`）默认返回 False，UMBP 返回 True，`MultiConnector` 要求所有子 connector 都为 True。
- `amd/rocm.py:128` 新增 `_has_foreign_kv_writer()`，替换原来判断 aiter sparse MLA（decode/prefill）和 `_trust_dsv4_extra_cache_nan_free` 时用的 `kv_transfer_config is not None`。DSv4.1 后端保留原来的检查。

### 4.4 DSpark MTP 与 MORI 兼容
- DSpark 的 3 层 draft 都是 `compress_ratio=0`，只有 SWA。replay 重建 target 时会一并重建 draft 的 KV，UMBP 不存 draft KV。EAGLE 风格的 draft 组在启动时被拒绝，报错信息里提示用 `rebuild_sliding_window: false`（`umbp/connector.py:150`）。
- `runtime/standalone.py` `_build_client()` 中 `lookup_timeout_ms` 的兼容逻辑，见第 3 节第 3 条。

## 5. 测试进展

### 5.1 单元测试

| 时间 / 环境 | 范围 | 结果 |
|---|---|---|
| 2026-10-28，新容器，MORI v1.2.3.post1 + UMBP | `tests/v1/kv_connector/unit/test_umbp_*.py` + `tests/v1/core/test_prefix_replay.py` | **305 passed**（102s） |
| 之前（PR 描述中记录） | 上面的范围 + `test_rocm_aiter_triton_sparse_mla.py`（19 passed）+ 回归集 | 回归集中 2 个失败是环境问题：HF 401 和缺少 ray |

PR 新增的测试列表见 PR 描述的 "Test Plan" 一节。

### 5.2 E2E 精度：DSV4-Pro-0813，DP8，DSpark MTP，GSM8K 5-shot

| Run | strict-match | flexible-extract | 接受率 | 平均接受长度 | UMBP / GPU 前缀命中 |
|---|---|---|---|---|---|
| Baseline | 0.9515 | 0.9507 | 57.97% | 4.48 | 0 / 0 |
| UMBP pass1（存） | 0.9538 | 0.9530 | 58.04% | 4.48 | 0.2% / 0 |
| UMBP pass2（恢复） | 0.9522 | 0.9515 | 58.36% | 4.50 | **72.6%** / 12.5% |

原始结果在 [`umbp-dsv4-mtp/results/`](umbp-dsv4-mtp/results/)。

### 5.3 E2E 性能：AgentX `override_dep8_c128`，DURATION=600（2026-10-28）

| 指标 | Baseline | UMBP | Δ |
|---|---|---|---|
| 输出吞吐（tok/s） | 1,157 | 1,324 | **+14.4%** |
| TTFT mean / p90（s） | 4.50 / 11.54 | 3.65 / 7.88 | −18.9% / −31.7% |
| TPOT mean（ms） | 78.8 | 70.2 | −10.9% |
| 每个请求实际计算的 prefill token | 13,937 | 11,203 | −19.6% |
| UMBP 命中 / GPU 前缀命中 | 0 / 92.4% | 0.8% / 93.0% | |

- 收益来源：prefill 少算的部分大约一半来自 UMBP 恢复，一半来自 replay 打开后 GPU 前缀命中不再受 32K SWA checkpoint 的限制。GPU KV 使用率只有约 1/3，UMBP 恢复的空间还没充分发挥。
- 每组只跑了一次，没有测量方差。详细数据和测试命令已写进 PR 描述。
- 结果文件：
  - `umbp-dsv4-mtp/results/agentic_c128_*`、`dsv4_vllm_dep8_c128_*.json`；
  - 原始产物在 `/home/shashen/vllm/umbp_runs/agentic/{baseline,umbp_replay}`，其中 baseline 的原始 server metrics JSON 只保存在本地 `/tmp/umbp_runs`。

### 5.4 E2E 精度：GPQA-Diamond，`override_dep8_c128`（2026-10-28，**未完成**）

配置：lm_eval 0.4.13，`gpqa_diamond_cot_zeroshot`，chat completions，开 thinking，`temperature=0`，`max_gen_toks=131072`，`timeout=10800`，并发 64。两组服务都设了 `VLLM_SERVER_DEV_MODE=1`，用于调 `/reset_prefix_cache`；UMBP 池 512 GiB。脚本：`umbp-dsv4-mtp/run_gpqa_config.sh`、`run_gpqa.sh`。

| Run | exact_match（flexible） | 空回答 | 非空回答的准确率 |
|---|---|---|---|
| Baseline | 0.7424 ± 0.0312 | 10 / 198 | 0.7819 |
| UMBP pass1（存储） | 0.7172 ± 0.0321 | 9 / 198 | 0.7513 |
| UMBP pass2（先 reset GPU 前缀缓存，再从 UMBP 恢复） | **用户中途终止，跑到 152/198，无结果** | | |

- 逐题对比（baseline vs UMBP pass1）：两边都对 122 题，只有 baseline 对 25 题，只有 UMBP 对 20 题，McNemar p = 0.55，没有显著差异。提取出的答案相同的有 145 / 198 题。
- pass1 期间 `external_prefix_cache_hits_total = 0`：GPQA 的 prompt 很短（约 300 token），各题之间也不共享前缀，所以 pass1 只验证了"挂上 connector、开启 SWA replay 之后精度不降"，没有覆盖恢复路径。
- 结果存在 `/home/shashen/vllm/umbp_runs/gpqa/{baseline,umbp}`。
- 用户已说明：baseline 跑一遍就够，不用重复跑。
- **下次继续**：只补跑 UMBP，`TAG=umbp UMBP=1 PASSES=2 bash umbp-dsv4-mtp/run_gpqa_config.sh`，约 1.5 小时。跑完后统计空回答、做逐题对比；如果精度差别不大，再更新 PR 描述。
- 环境：GPQA 已缓存在 `HF_HOME=/home/shashen/vllm/hf_cache`，评测时离线读取；lm_eval 装在 `/home/shashen/vllm/lm_eval_venv`；HF 已登录，token 在 `~/.cache/huggingface/token`。

### 5.5 脚本索引

| 脚本 | 用途 |
|---|---|
| `umbp-dsv4/build_mori_umbp.sh` | 一键编译并安装带 UMBP 的 MORI，详见 [`umbp-install.md`](umbp-install.md) |
| `umbp-dsv4/run_umbp_server.sh` | 启动 standalone server（默认 128 GiB DRAM 池） |
| `umbp-dsv4/run_vllm_dsv4_umbp.sh` | TP8 启动服务，`UMBP=0/1` 切换有无 connector |
| `umbp-dsv4/check_restore.py` | 恢复验证：冷请求 → reset prefix cache → 同一请求，对比外部命中和 TTFT |
| `umbp-dsv4/run_gsm8k.sh`、`umbp-dsv4-mtp/run_gsm8k.sh` | GSM8K 测试（lm_eval 装在 `/home/shashen/vllm/lm_eval_venv`） |
| `umbp-dsv4-mtp/run_vllm_dsv4_mtp.sh` | InferenceX `override_dep8_c128` + DSpark |
| `umbp-dsv4-mtp/run_config.sh`、`run_agentic.sh`、`run_router.sh`、`snapshot_metrics.sh`、`compare_agentic.py` | AgentX agentic 跑分，对比 baseline / umbp_replay / umbp_store_all，输出到 `/home/shashen/vllm/umbp_runs/agentic/<tag>` |

## 6. 环境恢复步骤（新机器 / 新容器）

> agentic 测试的环境：
> - AgentX runtime：`/home/shashen/vllm/agentx_runtime/venv`（`uv venv --python 3.11` 后执行 `uv pip install -r InferenceX/inferencex-e2e/infx/bench/agentic/requirements.txt`）。
> - router：`/home/shashen/vllm/vllm_router_venv`（`vllm-router==0.1.14`）。
> - trace 数据集：`HF_HOME=/home/shashen/vllm/hf_cache`（1.8G）。`/shared` 在这台机器上是只读 NFS，写不进去。
> - 模型：`/shared/models--deepseek-ai--DeepSeek-V4-Pro-DSpark/snapshots/7c09739…`。
> - 运行：`RUNS_DIR=/tmp/umbp_runs TAG=<tag> UMBP=0|1 DURATION=600 [UMBP_DRAM_CAPACITY=$((512*1024**3))] bash umbp-dsv4-mtp/run_config.sh`。
> - `/home/shashen` 所在的 NFS 卷和别人共享，而且基本满了。曾经因为 quota 耗尽导致 aiperf 的记录被截断，所以 `RUNS_DIR` 要放在本地盘上。

1. 对齐 vLLM 开发环境：`cd /home/shashen/vllm/vllm && bash dev/init.sh`。它会重建 `.venv` 并 rebase 到镜像的构建 commit。
2. 安装带 UMBP 的 MORI：`bash /home/shashen/vllm/vllm-dev-skills/docs/mori/umbp-dsv4/build_mori_umbp.sh`。
   - 会通过 apt 安装 gRPC/Protobuf，在 `/home/shashen/vllm/mori-v1.2.3.post1` 建 worktree，编译耗时约 3 分钟。
   - 装完要确认 import 到的是 `.venv/.../site-packages/mori`，并且 `umbp_standalone_server` 存在。踩坑点见 `umbp-install.md`。
3. 补装测试依赖：`uv pip install --python .venv/bin/python tblib`。新建的 venv 缺这个包，conftest 会 import 失败。
4. 跑单元测试：`.venv/bin/python -m pytest -q tests/v1/kv_connector/unit/test_umbp_*.py tests/v1/core/test_prefix_replay.py`，预期 305 passed。

2026-10-28 已在当前机器上按以上步骤完成安装，305 passed。

## 7. 未完成事项（按优先级）

- [x] ~~#58568 force-push 后 stack 内容有差异~~：2026-10-29 rebase 到 `fa3b9d1377` 时已对齐。除了本 PR 自己的增量，stack 和 #58568 head `e6a479fe63` 一致。
- [ ] **GPQA 精度对比收尾**：补跑 UMBP 的 pass1 和 pass2，然后更新 PR 描述，见 5.4 节。
- [ ] TP8 恢复验证（`check_restore.py`）：冷启动和命中后的 TTFT、外部命中 token 数、输出对比。
- [ ] 统计 UMBP server 上的 key 数和每个请求的字节数，对比 `dedup_tp_replicated_kv` 开/关（预期开启后约为 1/8）、`rebuild_sliding_window` 开/关。
- [ ] 开启 `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` 跑一次，确认没有 NaN。
- [ ] 长上下文（32K/128K 前缀）恢复后的 replay 精度。
- [x] AgentX agentic 性能对比（`override_dep8_c128`，DURATION=600）：UMBP 输出吞吐 +14.4%，见 5.3 节。
- [ ] agentic 对比再各跑一次以确认方差；跑一次 900s 以上或完整 trace；加一个 `rebuild_sliding_window: false` 的点，把 replay 和 restore 两部分的收益分开。
- [ ] 以上结果出来后，补进 PR 描述里的 "TODO (not run yet)" 部分，然后 force-push 更新 #63002。

## 8. 已知限制 / 后续工作

- 已发布的 MORI 没有 lookup 超时，server 卡住时 lookup 会一直阻塞。
- `UMBP_DRAM_CAPACITY=1TiB` 且不开 hugepages 时，server 启动超过 10 分钟。建议按负载设置池大小，或者打开 `UMBP_DRAM_USE_HUGEPAGES=1`。
- replay 依赖 V2 model runner。`VLLM_USE_V2_MODEL_RUNNER=0` 或开启 DBO 时，会退回到存所有组。
- 配置了 connector 后，GPU prefix cache 命中时也会 replay 最后一个窗口，因为 SWA 组已经退出了 prefix cache。
- distributed（RDMA）模式没有覆盖。以后可以复用 `tp*` key。
