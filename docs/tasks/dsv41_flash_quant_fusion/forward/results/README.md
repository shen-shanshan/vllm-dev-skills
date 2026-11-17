# `vllm bench forward`：#2/#3/#4 全开 vs 全不开（2026-11-17）

## 测试设置

- **base：** vLLM `60bd6dd2ed`，不做三处融合。
- **fused：** 分支 `rocm-dsv41-flash-kernel-fusion`，`98de3d8aff`（#2/#3）+ `2532459f72`（#4），三处融合全开。
- **模型与并行：** DeepSeek-V4.1-Flash，dummy 权重，2× MI355X（GPU 0,1）。DEP2（attention DP2 + EP2，AITER MegaMoE + MoRI），`--kv-cache-dtype fp8_e4m3`，DSpark 5 个 draft token，`--routing balanced`。
- **脚本：** 启动参数见 `../run_forward.sh`，交替运行见 `../run_ab_forward.sh`。
- **计时方式：** 每个 case 的耗时是一次完整 forward step 的 graph 回放中位数，包含 target sampler 和 drafter，取两个 rank 中较慢的那个。每次运行内部的采样间差异 ≤0.6%。

## 结果（step 耗时，单位 ms，取各次运行中位数的中位数）

| case（每个 DP rank） | base | fused | 差异 |
|---|---:|---:|---:|
| `prefill16k`：一个 16384 token 的 prefill | 393.36（393.35–393.37，n=2） | 392.46（392.34–392.57，n=2） | **−0.23%（−0.91 ms）** |
| `prefill8k`：一个 8192 token 的 prefill | 214.41（213.67–215.15，n=2） | 213.62（213.38–213.67，n=6） | −0.37%（−0.79 ms），与 base 波动范围重叠 |
| `decode128`：128 个请求 × 6 token（verify，5 个 draft），KV 约 16k | 71.81（71.73–71.90，n=2） | 71.54（71.46–71.56，n=4） | **−0.38%（−0.27 ms）** |
| `decode16`：16 个请求 × 6 token（每步 96 token），只跑这一个 case，交替测量 | 38.22（38.20–38.25，n=2） | 38.21（38.20–38.23，n=2） | −0.02%（约 −10 µs），在噪声范围内 |
| `mix`：一个 8192 token 的 prefill + 64 个 verify 请求 | 236.97（n=2） | — | fused 未测到，见下文 |

`decode16` 每步只有 96 token，走 mHC overlap 路径，所以 #2/#3 不生效，只有 #4 生效。而且 96 token 低于 shared expert 走 aux stream 的阈值（256），shared expert 与 MegaMoE 并行执行，#4 省下的时间不在关键路径上，step 耗时也就看不出变化。（`d16_*`：base 与 fused 交替各跑 2 次，只跑 `decode16`，都没有卡住。早先完整测试中 base 的 `decode16` 为 38.33 / 38.41 ms。）

这与 kernel 层面的估算一致：prefill 每步约省 1 ms（约 0.2–0.4%）。`decode128` 每步 768 token，超过 mHC overlap 的 128 token 上限，所以三处融合都会生效，每步约省 0.3 ms（约 0.4%）。

## 各数据来自哪次运行

| 文件 | 代码 | 内容 |
|---|---|---|
| `base_r1`、`base_r2` | base | 5 个 case 全部完成 |
| `fused_r1`、`fused_r2` | fused | prefill16k、prefill8k 完成后，在 `decode128` 卡住，20 分钟超时被杀 |
| `repro_fused_decode128` | fused | 只跑 `decode128` 一个 case，正常完成 |
| `repro2` | fused | prefill8k 完成后，在 `decode128` 卡住 |
| `all_a`、`all_b`、`hang_catch_try1` | fused | prefill8k → `decode128`，正常完成 |

## 未解决：fused 偶发卡死（需要继续定位）

- **现象：** fused 代码在同一个进程里先跑 prefill case、再跑 `decode128` 时，有 3/7 次卡死。两个 rank 都停在 `CUDAGraph.replay`（`decode128` 的 warmup 回放），GPU 利用率 100%，说明有 kernel 在 GPU 上空转。base 跑了 2 次，都没有卡住。
- **已排除：**
  - AITER `mega_mhc` 单独测试，按 bench 的顺序（大 M → M=768，eager 与 graph 交替），bf16 和 mxfp8 模式都不卡（`../mhc_hang*.py`）；
  - 只开 #2/#3、只开 #4 各跑 1 次，都不卡。不过样本太少，排除不了概率性。
- **AgentX serving 下没有出现：** 两次 DEP2 c256 serving（FULL CUDA graph，大量每步超过 128 token 的 decode step）各跑约 20 分钟，都没有卡住。
- **下一步：** 用 `../hang_catch.sh` 复现，卡住时用 rocgdb 记录 `info dispatches`，确认是哪个 kernel 在空转（MegaMoE、MoRI 还是 Mega-MHC）。同时增加 base 的重复次数，判断这是融合引入的问题，还是 MegaMoE / MoRI 在 EP 下原本就有的跨 rank 竞争。
- `mix` 在 fused 的运行里排在卡住的 `decode128` 之后，所以一直没测到。`decode16` 后来单独补测了。
