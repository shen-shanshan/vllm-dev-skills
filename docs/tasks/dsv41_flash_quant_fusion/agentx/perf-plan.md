# 端到端测试计划：MXFP8 输入量化融合（#2/#3/#4）在 AgentX c256 上的收益

更新时间：2026-11-16。状态：**第 1 轮（带 UMBP）作废，第 2 轮（不带 UMBP）待执行**。

## 1. 目标

用 DSv4.1-Flash DEP2 配置跑 AgentX c256，对比融合前后的端到端表现，重点关注：
- **每 GPU 总吞吐**（tok/s/GPU，输入 + 输出）；
- **P90 interactivity**（tok/s/user）。

结果整理成表格，写进 PR 描述（`../mega_mhc_mxfp8_pr.md`）。

## 2. 两组代码

| 组 | vLLM 代码 |
|---|---|
| base | `60bd6dd2ed`（不做三处融合） |
| fused | 分支 `rocm-dsv41-flash-kernel-fusion`：`98de3d8aff`（#2/#3）+ `2532459f72`（#4） |

`run_ab.sh` 会原地切换 vLLM 的 5 个文件（不使用 worktree）：先备份 fused 版本，跑完 base 后恢复，并校验 `git diff HEAD` 与切换前完全一致。

## 3. 配置（两组完全相同）

- **硬件：** 2× MI355X，GPU 0,1（使用前先用 `rocm-smi` 确认空闲；11-16 GPU 5 曾被其他任务占用）。
- **并行：** DEP2，即 `--data-parallel-size 2 --enable-expert-parallel --moe-backend aiter_mega_moe --all2all-backend mori`，TP1。前端是 vllm-router（`consistent_hash`），client 带 session 亲和。
- **用户指定的额外参数：**
  - `--prefill-schedule-interval 11`；
  - `--kv-cache-dtype fp8_e4m3`（per-tensor fp8 KV，vllm-project/vllm#66036）。
- **其余参数沿用 InferenceX #5036 / recipe：**
  - engram `cpu_offload: true`，block size 128，mxfp4 indexer；
  - DSpark 5 个 draft token，synthetic AL 3.51；
  - `--max-num-batched-tokens 16384`，`--max-cudagraph-capture-size 1024`；
  - `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE`，rust 前端，`--max-model-len 1048576`。
- **KV offloading：第 2 轮不使用 UMBP**（用户 11-16 决定，原因见第 5 节）。
- **client：** InferenceX `infx.bench agentic`，agentx-fast 模式（`AIPERF_EXPERIMENTAL_FAST=1`，1200 s），CONC=256。

## 4. 执行（第 2 轮）

```bash
rocm-smi --showmemuse --showuse   # 确认 GPU 0,1 空闲
cd /home/shashen/vllm/vllm-dev-skills/docs/tasks/dsv41_flash_quant_fusion/agentx
nohup bash run_ab.sh > /tmp/quant_fusion_runs/driver_noumbp.log 2>&1 &   # UMBP 默认关闭
```

- 输出目录：`/tmp/quant_fusion_runs/fast_{base,fused}_dep2_c256_noumbp/`。每个点约 50 分钟，两个点共约 1.7 h。
- **跑完后的检查（结果可信的前提）：**
  - `python3 timeline.py <run_dir>...`：确认没有整分钟无请求完成的停顿；
  - engine 日志里没有异常警告；
  - 两组失败请求都为 0。
- **汇总：** `python3 compare_ab.py <base_dir> <fused_dir>`。表格写入 PR 描述的 Validation 部分，原始 JSON 和 engine 指标复制到 `results/`。
- **判断规则：** 单次 fast 运行的噪声按约 3% × 1.5 估计。差异在噪声以内时，结论写"持平"；按 kernel 层面的估算，预期收益约 0.2–0.3%，大概率会是持平。

## 5. 第 1 轮记录（带 UMBP，11-16，作废）

配置和第 3 节相同，另外开启了 UMBP standalone（449e9 B 池子，4 KiB 页，`kv_load_failure_policy: recompute`）。结果在 `results/compare_umbp_run1.md` 和 `results/timeline_umbp_run1.txt`。

| 指标 | base | fused |
|---|---:|---:|
| 总吞吐（tok/s/GPU） | 106,509 | 187,864 |
| P90 interactivity | 6.58 | 7.62 |
| TTFT p90 | 272.7 s | 8.6 s |

**作废原因：** 两组都出现了整段停顿，没有请求完成。base 约 12 分钟（14:12–14:21、14:21–14:25），fused 约 4 分钟（15:14–15:18）。差异主要来自停顿长短，不是融合。不停顿的时段里，两组吞吐都在约 315–375k 输入 tok/s/GPU，基本一样。

**停顿原因：UMBP 失去响应。**
- 开跑后几分钟，UMBP server 的 `batch_exists` 查询 1 s 内收不到回复；数据传输 30 s 超时后一直卡在 MORI 里（"transfers are refused while N timed-out transfer(s) are still blocked in MORI"）。
- server 日志在两个 worker 注册完成后就没有任何输出。
- 从 UMBP 恢复的 prompt 只有 0.22% / 0.05%。

之前 expert TP2（DP2，`fp8_ds_mla` KV）的 c256 测试中 UMBP 是正常工作的。这次多了两个变量：MegaMoE + EP（MORI shmem）和 `fp8_e4m3` KV。**这可能也会影响 InferenceX #5036**（它也启用了 `fp8_e4m3`，没有做过本地测试），需要另开任务定位。

## 6. 第 2 轮记录（不带 UMBP，11-17，结论：分辨不出收益）

配置同第 3 节，不带 UMBP，GPU 0,1。两组都正常跑完，失败请求为 0 和 1/5931。结果在 `results/compare_noumbp_run2.md`、`results/timeline_noumbp_run2.txt`、`results/jit_during_inference_*_noumbp_run2.txt`。

| 指标 | base | fused | 差异 |
|---|---:|---:|---:|
| 总吞吐（tok/s/GPU） | 211,939 | 270,450 | +27.6% |
| P90 interactivity（tok/s/user） | 9.70 | 15.10 | +55.6% |
| TTFT p90 | 10.54 s | 4.56 s | −56.7% |

**不能作为融合收益的原因：** base 在 profiling 窗口里有 4 段停顿，期间没有请求完成，共约 7 分钟：02:49–02:52（197 s），以及 02:57、03:01、03:09 附近各约 80 s；fused 没有停顿。

停顿与推理过程中的 FlyDSL JIT 编译相吻合：`megamoe_prepare_compact_m64`、`megamoe_stage2_compact`、`radix_topk_one_block`，都是 MegaMoE 或 topk 运行中才遇到的新 shape。两个 DP rank 要做集合通信，一个 rank 在编译时另一个只能等，于是整个 engine 停住。FlyDSL 默认的磁盘缓存目录 `~/.flydsl/cache` 不存在，因此是否触发编译取决于每次运行碰到的 shape，有随机性。fused 这次只现场编译了 Triton 的 `_silu_and_mul_mxfp8_kernel`，没有造成停顿。

按 kernel 层面估算，三处融合每个 prefill step 约省 1 ms（0.2–0.3%），远小于这种停顿带来的波动。所以单次 fast A/B 分辨不出这个量级的收益。

**更可靠的测法（待用户选择）：**
- 用 `vllm bench forward` 在 DEP2 上直接测 prefill / decode step 的耗时，几乎没有服务端噪声；
- 或者先预热 JIT（例如设置 `FLYDSL_RUNTIME_CACHE_DIR` 并先跑一次预热），再交替多次运行 base / fused，报告均值和波动范围。

## 7. 第 3 轮：只重跑 base（不带 UMBP，11-17），对比第 2 轮的 fused

- 用户决定：fused 直接复用第 2 轮的数据（没有停顿），只重跑 base（`STEPS=base bash run_ab.sh`）。第 2 轮有停顿的 base 已改名为 `*_run2_stalled` 保留。
- **可信度检查：** base 运行中没有触发 FlyDSL JIT 编译（只有开头两个 Triton kernel 的编译，fused 也一样），失败请求为 0/5965。两次运行里，相邻两个请求完成之间的最大间隔都只有 7 s，没有停顿。
- **结果**（`results/compare_noumbp_run3_base_vs_run2_fused.md`）：

| 指标 | base | fused | 差异 |
|---|---:|---:|---:|
| 总吞吐（tok/s/GPU） | 272,272 | 270,450 | −0.7% |
| P90 interactivity（tok/s/user） | 15.72 | 15.10 | −4.0% |
| TTFT p90 | 4.55 s | 4.56 s | +0.2% |
| TPOT p90 | 63.6 ms | 66.2 ms | +4.1% |

- **结论：** 差异在 fast 模式单次运行的噪声范围内（约 ±3–5%），端到端测不出收益。这与 step 级的测量一致：融合每步只省约 0.2–0.4%（见 `../forward/results/README.md`）。表格已写入 PR 描述。
