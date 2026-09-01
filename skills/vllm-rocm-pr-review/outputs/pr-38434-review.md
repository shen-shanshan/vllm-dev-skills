# PR #38434: [Fix] Improve ROCm detection in WSL environments

> **Author**: @yiz-liu | **State**: OPEN (2026-03-28 创建，2026-08-14 rebase) | **Labels**: rocm
> **Branch**: `fix-rocm` → `main` | **Changes**: +15 -1 across 1 file | **ROCm 相关性**: 完全相关（平台检测逻辑，ROCm 专属分支）

## 1. 动机 (Motivation)

在 WSL 环境中，`amd-smi` / `rocm-smi` 目前不可达（`import amdsmi` 报 `ModuleNotFoundError`），导致 vLLM 无法识别 ROCm 平台、后续无法走 ROCm 专属路径（GCN arch 检测、ROCm attention backend 等）。本 PR 为 `rocm_platform_plugin()` 增加一个 WSL 专属的 fallback：当 amdsmi 检测失败且处于 WSL 时，用 PyTorch 侧信号（HIP 构建 + accelerator 可用 + vLLM 非 CPU-only 构建）确认 ROCm 平台。原有的 amdsmi 路径保持为主检测器，fallback 仅在 `is_rocm == False` 时生效。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/platforms/__init__.py` (L13) | 从 `.interface` 额外导入 `in_wsl` |
| `vllm/platforms/__init__.py` (`rocm_platform_plugin()`, L128–141) | 新增 WSL fallback 块：`in_wsl()` 且 amdsmi 未检出时，检查 `not vllm_version_matches_substr("cpu")`、`torch.version.hip` 非空、`torch.accelerator.is_available()` 三者全真才置 `is_rocm = True`；整体包在 `try/except Exception` 中，失败仅打 debug 日志 |

检测逻辑与同文件 `cuda_platform_plugin()` 的 CPU-build 检查模式一致（`vllm_version_matches_substr("cpu")`），三条件门控正确：HIP 构建排除 CUDA torch、`is_available()` 排除无可见 GPU 的机器、非 CPU 构建排除 CPU-only wheel。

## 3. Review 意见 (Findings)◊

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 测试 | — | — | 1 |
| 设计 | — | — | 2 |
| 一致性 | — | — | 1 |
| CI | — | — | 1 |

**📝【测试】平台检测逻辑无自动化测试，三条件矩阵仅靠一次真机手工验证** `[已验证]`

- **问题**: PR 未附带任何测试。fallback 的激活条件是三个条件的逻辑与（非 CPU 构建 × HIP torch × accelerator 可见），其否定组合（例如 WSL + HIP torch 但设备不可见时应**不**激活）没有任何回归保护；平台解析是全 vLLM 的入口，误报会立即影响所有用户。
- **影响**: 未来任何一处条件被改宽/改窄（例如去掉 `is_available()` 检查回到 CPU-only false-positive 风险）不会被 CI 拦截。
- **行动**: 建议作者补一个 mock 驱动的单测（monkeypatch `in_wsl`、`torch.version.hip`、`torch.accelerator.is_available`），覆盖激活与不激活两个方向。评论区 @Soluchann 已表示愿意在本 PR 合并后补测试，也可以直接收进本 PR。

**📝【设计】fallback 失败路径只打 debug 日志，用户遇到静默落到 `UnspecifiedPlatform` 时难以诊断** `[已验证]`

- **问题**: 三个条件中任一不满足（最常见：WSL 里装了 HIP torch 但 GPU 未通过 `/dev/kfd`、`/dev/dri` 透传，`is_available()` 为 False），`except` 分支和条件不满足分支都只输出 `logger.debug`，默认日志级别下用户完全无感知。
- **影响**: 平台解析最终落到 `UnspecifiedPlatform`，后续报错（设备相关 API 缺失等）与根因相距甚远，用户排查成本高。
- **行动**: 建议作者在"WSL + HIP torch 但 accelerator 不可见"这条路径上打 `logger.warning`，给出"ROCm fallback 未激活：torch 报告无可见加速器"之类的一次性提示。

**📝【一致性】主 amdsmi 检测路径仍缺 CPU-only 构建检查，与 fallback 不对称** `[已验证]`（触发场景 `[推测]`）

- **问题**: fallback 加了 `not vllm_version_matches_substr("cpu")` 检查，但主路径 `amdsmi.amdsmi_get_processor_handles() > 0` 直接置 `is_rocm = True`，无 CPU-build 检查（gemini-code-assist 在早期 review 中已指出，建议另开 PR 修）。
- **影响**: CPU-only wheel + 原生 Linux ROCm 机器（amdsmi 可用）时，rocm 与 cpu 两个内建插件同时激活，`resolve_current_platform_cls_qualname()` 直接抛 `RuntimeError("Only one platform plugin can be activated")`。此为既有问题、非本 PR 引入，WSL fallback 因带检查反而没有此问题。
- **行动**: 建议作者（或后续 PR）顺手把 CPU-build 检查提到主路径，保持两条路径对称。

**📝【设计】`torch.accelerator.is_available()` 在平台解析阶段提前初始化 HIP context** `[已验证]`

- **问题**: fallback 在 `current_platform` 首次访问时执行，`torch.accelerator.is_available()` 会查询设备并初始化 CUDA/HIP context——早于 `rocm.py` 中 GCN arch fallback 已有的警告场景（"This will initialize CUDA and may cause issues if CUDA_VISIBLE_DEVICES is not set yet"）。PR 自带测试日志里也可以看到后续 "CUDA is initialized → 强制 spawn" 的连锁反应。
- **影响**: 实际影响有限：vLLM 的 env 解析先于平台解析，且 WSL 下本就走 spawn 多进程路径（日志可见），单 GPU WSL 场景已被真机验证可用；多 GPU + 依赖 `CUDA_VISIBLE_DEVICES` 编排的场景理论上有初始化时机风险，但未证实。
- **行动**: 建议 review 时确认多 GPU WSL 场景下 `CUDA_VISIBLE_DEVICES` 的生效时机是否受提前初始化影响；如无问题可在注释中说明。

**📝【CI】AMD 硬件 CI 无法覆盖此路径（WSL-gated），真机日志是唯一验证证据** `[已验证]`

- **问题**: head commit 的 check-runs/status 只有 pre-commit、DCO、meta check、readthedocs 等快速检查，无 AMD 硬件 CI（Buildkite rocm-build）。且该 fallback 被 `in_wsl()` 门控，AMD CI 的原生 Linux MI300/MI355 机器上此分支根本不会触发，硬件 CI 即使跑了也测不到新代码。
- **影响**: 新逻辑的验证完全依赖作者在 WSL 上的一次手工 e2e（Qwen3.5-9B-AWQ-4bit，TRITON_ATTN，日志完整可信），无任何自动化回归网。
- **行动**: 建议作者在 PR 描述中保留该 e2e 日志并注明 CI 无法覆盖的原因；测试缺口由上述单测建议弥补。

## 4. 现有讨论 (Existing Discussion)

- **@gemini-code-assist**（2026-03-28）：建议加 CPU-only 构建检查，与 `cuda_platform_plugin` 保持一致——**已在 rebase 后的版本中实现**。
- **@Soluchann**（2026-08-12）：指出仅 `torch.version.hip` 检查不能确认设备可见（CPU-only false-positive 风险）、PR 与 main 冲突需 rebase、原 `warning_once → warning` 改动因 #46516 已 moot；并提供了扩展版参考实现（device-visibility 检查、`VLLM_ROCM_GCN_ARCH` env override、`rocminfo` fallback、测试）。
- **@yiz-liu**（作者，2026-08-14）：已 rebase、删除过时的 warning 改动、按建议收紧为三条件门控。
- **@Soluchann**（2026-08-15）：愿意在合并后补测试与 GCN-arch fallback，或直接开 PR 到本分支。
- **@MengqingCao**（2026-06-27）：**APPROVED**（LGTM），并 @tjtanaa 请求二查。

## 5. 结论 (Verdict)

✅ **LGTM** — 三条件门控正确且与 `cuda_platform_plugin` 的 CPU 检查模式一致，WSL 分支对 CUDA 构建（`torch.version.hip` 为 None）零影响；作者提供了完整的 WSL 真机 e2e 日志。剩余均为 📝 级建议（补单测、失败路径打 warning、主 amdsmi 路径的 CPU 检查对称性），不阻塞合入；建议等待 @tjtanaa 的 ROCm maintainer 二查后合并。
