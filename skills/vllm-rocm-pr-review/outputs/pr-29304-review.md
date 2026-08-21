# PR #29304: [ROCm][PD] add moriio kv connector.

> **Author**: @inkcherry | **State**: MERGED (2026-01-09) | **Labels**: documentation, rocm, frontend, ready, ci/build, v1, kv-connector
> **Branch**: `upstream_mori_` → `main` | **Changes**: +3369 -3 lines across 10 files
> **ROCm 相关性**: 完全相关（moriio KV connector，AMD ROCm 专属）

## 1. 动机 (Motivation)

vLLM 的 PD（prefill-decode）分离推理此前在 ROCm 上没有高性能的 KV 传输后端。本 PR 基于 AMD 的通信库 [MORI](https://github.com/ROCm/mori)（MORI-IO）引入 `MoRIIOConnector`，支持两种传输模式：PULL（READ，串行交互：prefill 完成后 decode 拉取）与 PUSH（WRITE，并行交互：producer 逐层非阻塞推送），并提供一个统一的 proxy 示例（`moriio_toy_proxy_server.py`）与 xPyD 并行策略支持（TP→TP、DP→DP）。RDMA 直接读写 GPU 显存（GPUDirect），目标是消除 PD 分离中 KV 传输的开销瓶颈。

## 2. 代码改动总结 (Change Summary)

| 模块 | 改动 |
|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/moriio/`（新增，~2450 行） | 核心实现：`moriio_connector.py`（Connector/Scheduler/Worker 三件套）、`moriio_engine.py`（MoRIIOWriter/MoRIIOWrapper 封装 mori.io API）、`moriio_common.py`（配置、元数据、ZMQ 工具） |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` (+6) | 注册 `"MoRIIOConnector"` |
| `vllm/envs.py` (+18) | 新增 4 个 env var：`VLLM_MORIIO_CONNECTOR_READ_MODE`、`VLLM_MORIIO_QP_PER_TRANSFER`、`VLLM_MORIIO_POST_BATCH_SIZE`、`VLLM_MORIIO_NUM_WORKERS` |
| `docker/Dockerfile.rocm_base` (+18) | 从源码构建并安装 mori |
| `examples/.../moriio_toy_proxy_server.py`（新增 320 行） | PUSH/PULL 统一的路由 proxy 示例 |
| `tests/v1/kv_connector/unit/test_moriio_connector.py`（新增 545 行） | 单测（mock 驱动） |
| `docs/getting_started/installation/gpu.rocm.inc.md` (+17) | 安装说明 |

核心流程（PUSH 模式）：proxy 将请求路由到 producer → prefill 完成后 scheduler 侧 `build_connector_meta` 组装传输元数据 → worker 侧经 ZMQ 握手交换 KV 布局与 mori engine 元数据 → `MoRIIOWriter` 后台线程逐层执行 RDMA 写 → 完成后 ZMQ notify 通知 consumer 释放/使用 block。

## 3. Review 意见 (Findings)

| 类型 | 🔴 | ⚠️ | 📝 |
|------|----|----|----|
| 正确性 | 1 | 4 | — |
| 设计 | — | 2 | 1 |
| 并发 | — | 1 | — |
| 测试 | — | 1 | — |
| 注释/文档 | — | — | 1 |
| 可维护性 | — | — | 3 |
| 依赖管理 | — | 1 | — |

---

**🔴【正确性】端口偏移计算的 `tp_size` 参数从未生效，同机多 DP + TP>1 时端口冲突** `[已验证]`

- **问题**: `moriio_common.py` 中 `get_port_offset(dp_rank, tp_rank, tp_size=1)` 默认值使公式退化为 `dp_rank + tp_rank`，而全部 4 个调用点（`MoRIIOConfig.from_vllm_config`、`MoRIIOConnectorWorker.__init__` 的 `side_channel_port`、scheduler 的 `update_state_after_alloc`、`_moriio_handshake`）都不传 `tp_size`。
- **影响**: 同一台机器上部署 DP≥2 且 TP≥2 时，(dp=1, tp=0) 与 (dp=0, tp=1) 计算出的 handshake/notify 端口完全相同 → 握手消息或 block 通知被路由到错误的 engine，或 ZMQ bind 冲突。多机部署时因 IP 不同被掩盖，同机部署（常见于测试或小规模集群）必现。
- **行动**: 作者应当让所有调用点显式传入 `tp_size`（公式本意是 `dp_rank * tp_size + tp_rank`），或删除死参数并补注释说明偏移语义。注：main 分支后续仅修复了一处调用点（传入 `remote_tp_size`），其余调用点至今仍省略，此问题仍部分存在。

**⚠️【设计】`save_kv_layer` 以无 sleep 忙等自旋等待异步握手完成** `[已验证]`（死循环触发条件 `[推测]`）

- **问题**: `MoRIIOConnectorWorker.save_kv_layer` 末尾 `while True:` 循环在 `_ready_requests` 为空且 `write_ready_flags` 未置位时 `continue`，无任何 sleep/Event 等待；且当 `metadata.reqs_to_save` 为空时 `remote_engine_id` 保持 `None`，`None not in write_ready_flags` 恒为真——若 forward 路径在无传输请求的 step 也调用此函数，则永久自旋。
- **影响**: 握手期间 forward 线程 100% 单核忙等（正常路径必现，仅浪费 CPU）；若 runner 对空 metadata 也调用 `save_kv_layer`，则 producer 直接死循环挂死——与 rasmith 在评论区报告的 decode 服务 hang 现象可能相关。
- **行动**: 作者应当为空 `reqs_to_save` 提前返回，并用 `threading.Event`/条件变量替代忙等；建议 review 时追问 runner 侧对 `save_kv_layer` 的调用条件。

**⚠️【正确性】`register_kv_caches` 使用泄漏的循环变量记录 KV 尺寸** `[已验证]`

- **问题**: `moriio_connector.py` 中 `self.local_kv_cache_size.append(cache.nelement() * cache.element_size())` 引用的是前一个 `for cache in cache_list` 循环遗留的变量 `cache`（最后一个 k/v tensor），而非当前层的 `kv_cache`。
- **影响**: 各层 shape 相同时数值恰好正确（因此测试无法暴露）；一旦存在跨层 shape 差异（如混合 attention、不同 head dim 的层），记录的全是最后一层的尺寸 → 传输长度计算错误 → RDMA 读写越界或数据错位。
- **行动**: 作者应当改为 `kv_cache.nelement() * kv_cache.element_size()`。

**⚠️【正确性】`_pop_done_transfers` 只检查最后一层传输的状态** `[已验证]`（影响 `[推测]`）

- **问题**: `status_list[-1].Succeeded()` 仅看 `_recving_transfers` 列表中最后一个传输；前面任一层的失败会被忽略，请求仍被标记为完成并通知 producer 释放 block。
- **影响**: 中间层 RDMA 失败时，consumer 以不完整的 KV cache 继续 decode → 静默输出错误；producer 侧 block 被提前释放后无法重传。
- **行动**: 作者应当检查 `all(s.Succeeded() for s in status_list)`，并确认 moriio 失败时的重传路径。

**⚠️【正确性】`_background_moriio_handshake` 存在不可达死分支 + 锁检查键不一致** `[已验证]`

- **问题**: ① `fut = self._handshake_futures.get(remote_engine_id)` 查询用无后缀键，而字典键总是带 `_dp<N>` 后缀——`fut` 恒为 `None`，`host/port/tp_size/remote_dp_size` 仅在 `fut is None` 分支赋值，若某天键语义改变，`range(remote_dp_size)` 将抛 `UnboundLocalError`；② 外层检查用 `dp0_remote_engine_id`（带后缀）、内层重查用 `remote_engine_id`（无后缀），双重检查锁失效。
- **影响**: 当前不可达故无运行时错误，但这是潜伏的地雷（D1b 型）；锁不一致可能导致并发重复握手。
- **行动**: 作者应当删除死分支，或统一键语义；统一内层与外层的检查键。

**⚠️【设计】`kv_transfer_params` 依赖 `SamplingParams.extra_args` 内部结构传递** `[已验证]`

- **问题**: consumer 侧 `build_connector_meta` 用 `assert hasattr(new_req.sampling_params, "extra_args")` 断言内部属性，再从 `extra_args.get("kv_transfer_params", {})` 取值；`add_new_req` 无条件下标访问 `kv_transfer_params["remote_block_ids"]` 等键。
- **影响**: 直接打到 decode engine、未携带 `kv_transfer_params` 的请求（无 proxy 注入）在 metadata 构建时直接 KeyError/AssertionError 崩溃，而非给出可诊断的错误；与 SamplingParams 内部结构强耦合，上游字段调整即破坏。
- **行动**: 作者应当为缺失参数的请求提供明确的报错/跳过路径，并通过正式的 KVTransferParams 通道（而非 extra_args 旁路）传递参数；建议 review 时确认与后续 `kv_transfer_params` 正规划 PR 的衔接。

**⚠️【并发】moriio 声明非线程安全，但握手线程与写线程并发调用** `[推测]`

- **问题**: 代码注释明确 "MoRIIO is not guaranteed to be thread-safe"，因此 `_handshake_initiation_executor` 限制为 1 worker；但 `MoRIIOWriter._write_worker_loop` 是另一个独立线程，同样调用 `moriio_wrapper` 的方法（`write_remote_data`、`build_session` 等），与握手线程之间无共享锁。
- **影响**: 并发调用 mori.io API 可能触发内部状态竞争（session/engine 状态），表现为偶发传输失败或 hang，难以复现与定位。
- **行动**: 建议作者将所有 moriio 调用收敛到单一线程（如统一由写线程执行，或加全局锁），或向 mori 上游确认 API 线程安全边界。

**⚠️【测试】单测全部基于 Fake wrapper，且真实集成路径无 CI 覆盖** `[已验证]`

- **问题**: `test_moriio_connector.py` 用 `FakeMoRIIOWrapper`/`MagicMock` 替换全部 moriio 调用，`skipif` 要求 ROCm + mori 才运行——vLLM 主 CI（CUDA runner）上这些测试恒被 skip；作者承认 ray executor 后端未测试（"shouldn't be a problem, further testing will be conducted subsequently"），随后 rasmith 实测报告 decode 服务 hang。
- **影响**: mock 测试无法捕获 RDMA 布局、线程同步、端口路由等真实集成问题（本报告前几条 finding 全部漏网）；合并后 AMD 用户成为实际测试者。
- **行动**: 建议作者在 AMD hardware CI 队列补一条最小的 e2e 用例（真实 mori，单机双 engine），并补充 ray executor 验证。

**⚠️【依赖管理】mori 依赖无版本 pin，API 漂移风险** `[已验证]`

- **问题**: 本 PR 未改动任何 `requirements` 文件——mori 仅通过 `Dockerfile.rocm_base` 从源码安装（默认 main），无版本约束；`requirements/kv_connectors_rocm.txt` 是后续才补上的。
- **影响**: `from mori.io import EngineDesc, PollCqMode, RdmaBackendConfig...` 依赖的 API 随 mori main 演化，用户侧构建时点不同即可能 ImportError 或行为漂移；且 CUDA 平台安装 vLLM 时该 connector 的 import guard 依赖 mori 可选性。
- **行动**: 作者应当 pin mori 版本（或 commit SHA）到 requirements，并在 PR 描述中链接 mori 侧对应版本。

**📝【注释】WRITE 模式的 `get_num_new_matched_tokens` 注释与行为矛盾** `[已验证]`

- **问题**: 注释写 "MoriiO in write mode, no remote prefill"，但代码返回 `len(token_ids) - num_computed_tokens, True`——实际上 WRITE 模式下 consumer 的全部 prompt KV 都来自远程推送，这正是"remote prefill"。
- **行动**: 建议作者修正注释为 "in write mode all prompt tokens are loaded from remote push"。

**📝【可维护性】`logging.getLogger("aiter").disabled = True` 全局副作用** `[已验证]`

- **问题**: Worker 初始化时直接禁用 aiter 库的全局 logger，影响同进程中其他使用 aiter 的组件（MoE、attention 后端）的诊断能力。
- **行动**: 建议作者改为只对 moriio 模块内的 aiter 调用降噪（局部 filter），或至少加注释说明原因。

**📝【可维护性】接口与实现脱节的三处参数** `[已验证]`

- **问题**: ① `MoRIIOConnector.get_finished(finished_req_ids)` 参数完全未使用（基类签名要求，属结构性丢弃）；② `update_state_after_alloc` 的 `num_external_tokens` 与 `connector_worker` 参数未使用，且调用方显式传入 `self.connector_worker`（scheduler 侧恒为 `None`），误导读者；③ `MoRIIOConnectorMetadata.__repr__` 的返回值被后续赋值覆盖，`reqs_to_recv` 信息在 repr 中丢失。
- **行动**: 建议作者为结构性丢弃的参数添加简短注释（"required by base class signature"），修正 `__repr__` 的拼串逻辑。

**📝【可维护性】`_ping` 重试计数逻辑不一致** `[已验证]`

- **问题**: `ConnectionRefusedError`/`OSError` 分支只递增 `retry_count` 但不检查上限，只有兜底 `Exception` 分支在达到 `MAX_PING_RETRIES` 后抛 `RuntimeError`；且 ZMQ DEALER 的 `connect` 在 peer 不存在时通常不抛 `ConnectionRefusedError`，该分支可能永远不触发。
- **行动**: 建议作者统一重试与退出逻辑（如用单调时间窗口判断），并说明 proxy 不可达时的预期行为。

## 4. 现有讨论 (Existing Discussion)

- **@tjtanaa**（maintainer）：要求 env var 全部注册进 `envs.py`（已落实）；要求补充单测（已落实）；要求清理 `use_flashinfer` 死代码（已清理并加 TODO）。
- **@KuntaiDu**：connector 单文件过长（1515 行），要求拆分（作者回应已拆为 Connector/Scheduler/Worker 三类，后续 main 又拆出 `moriio_layout.py`）。
- **@kouroshHakha**：询问是否测过 ray executor 后端——作者承认未测试。
- **@rasmith**：实测报告 decode 服务 hang（与本文 ⚠️ 忙等 finding 可能相关，建议作者对照排查）。
- **@ChuanLi1101**：指出 ZMQ socket 操作缺 timeout 可能 hang（作者回应为异步通知线程，不会阻塞主线程，但承认存在优化空间）。
- **@gshtras / @tjtanaa**：`Dockerfile.rocm_base` 的改动需要传播到 `rocm/vllm-dev:base` 镜像，并影响 CI 测试队列。
- gemini bot 提出的 `__del__` 清理、`assert 0` 兜底、broad `except: pass` 等问题作者均已修复。

## 5. 结论 (Verdict)

**⚠️ NEEDS WORK**（追溯性 review，PR 已合并）

作为 vLLM ROCm PD 分离的首个生产级 KV 传输后端，本 PR 的架构设计（PUSH/PULL 双模式、xPyD、分层非阻塞传输）合理且价值明确，与 reviewer 的互动质量高、反馈落实快。但代码存在一个明确的端口路由 bug（🔴）、一处忙等设计缺陷（⚠️）以及多处潜伏正确性隐患，且测试完全基于 mock、真实集成路径无 CI 兜底，合并后 main 分支仍持续修复相关问题（`get_port_offset` 仅部分修复）。若在合并前 review，建议阻塞至端口偏移修复、忙等改为事件驱动、并补一条真实 mori 的最小 e2e 用例。
