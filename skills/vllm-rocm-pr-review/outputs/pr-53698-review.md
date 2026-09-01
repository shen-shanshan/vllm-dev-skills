# PR #53698: [Bugfix][ROCm][Disagg] Fix MoRIIO shared KV memory region registration

> **Author**: @avininjamay8 | **State**: OPEN | **Date**: 2026-08-25
> **Branch**: `fix/moriio-shared-kv-mr-registration` → `main` | **Labels**: bug, rocm, kv-connector, verified
> **Changes**: +97 -31 across 5 files | **ROCm 相关性**: 完全相关（MoRIIO PD 分离 KV 传输 + ROCm 专属路径）

## 1. 动机 (Motivation)

PR #51718 让 MLA 模型把所有 KV 层放进同一个共享 GPU 大 buffer，每层只是其中的一个 view。MoRIIO 没有跟随这个改动：仍然逐层注册每个 view 做 RDMA，而这些 view 的地址不是页对齐的（`RegisterRdmaMemoryRegion errno:22`），并且 KV 被拷到错误地址，导致 MoRIIO 分离式 prefill/decode 在 nightly 上完全坏掉（GSM8K 0/1319）。本 PR 的修复思路是：把共享 backing 补到 4KB 边界、对整块 backing 注册一个页对齐的 MR、为每层记录其在 MR 内的字节偏移、并在 WRITE 路径按层应用偏移和缓存传输计划。作者在 MI300 上用 DeepSeek-V3 TP8 1P1D 实测，GSM8K 从 0/1319 恢复到 0.968，并声称 Kimi 2.5 也验证通过——证据充分，主路径修复有效。

## 2. 代码改动总结 (Change Summary)

| 文件 | 改动 |
|------|------|
| `vllm/v1/worker/utils.py` | KV cache backing 分配尺寸按 4KB 向上取整（页填充） |
| `moriio_connector.py` | 新增 `_build_shared_kv_mr`（页对齐整块 MR + 每层偏移）、`_remote_layer_mr_offset`（远端层偏移）；`register_kv_caches` 检测共享 backing 后改为整块注册；`_compute_block_transfer_offsets` 对 local/remote 偏移统一加层内 MR 偏移 |
| `moriio_engine.py` | WRITE 传输计划缓存 key 加入 `layer_name`（同 geometry 不同层的偏移不再互相复用）；传入 `remote_engine_id` |
| `moriio_common.py` | `TransferOffsetsKey` 类型别名（含层名） |
| `test_moriio_kv_layout.py` | 更新 write-plan 缓存单测以匹配新 key |

核心数据流：producer 与 consumer 各自注册一个覆盖整块共享 KV backing 的页对齐 MR；`kv_caches_base_addr` 仍按层发送每层 `data_ptr()`，对端用 `层地址 - MR 基址` 反推每层在 MR 内的偏移；`compute_block_transfer_offsets` 的偏移本身仍是"整张层 tensor 基址相对"（split K/V 的 V 偏移含 `local_kv_stride`），因此统一加层偏移在算术上自洽（已验证 `moriio_layout.py:325-339`）。

## 3. Review 意见 (Findings)

**意见类型统计**：⚠️ 建议修复 ×4 | 📝 备注 ×2

### ⚠️【正确性】页填充对"总尺寸恰为 4KB 整数倍"的情形不生效，slack>0 时启动即崩 `[已验证]`

- **问题**：`vllm/v1/worker/utils.py:396-401` 的填充是 `ceil(raw_size/4096)*4096`——当 `raw_size` 恰好是 4096 的整数倍时**一个字节都不补**。而 `_build_shared_kv_mr`（`moriio_connector.py:1700-1725`）需要覆盖 `[page_down(ptr), page_up(ptr+end))`，即 `reg_nbytes = ceil((slack + end)/4096)*4096`。当分配基址非 4K 对齐（`slack = ptr % 4096 > 0`）且 `end == raw_size` 时，`reg_nbytes = raw_size + 4096 > nbytes`，命中作者自己写的 `ValueError("shared KV backing too small...")`，**服务启动即失败**。
  - 触发输入（具体）：KV cache 总字节数为 4096 整数倍 + 基址来自非 4K 对齐的分配块（PyTorch 缓存分配器复用 segment 时块偏移按 512B 对齐；<1MB 的小分配也按 512B 对齐）。生产级大 KV cache 通常 2MB 对齐（slack=0）所以作者的 MI300 实测没踩到。
  - 反例验证：`raw_size = 8192, slack = 512` → `reg_nbytes = 12288 > nbytes = 8192` → 必崩；`raw_size = 9000, slack = 512` → `reg_nbytes = 12288 == nbytes` → 正常。
- **影响**：小模型/小 KV cache 的 MoRIIO disagg 配置无法启动（崩溃是响的，不会静默错数据，但属于本可避免的回归）。
- **行动**：作者应当把填充改为固定加一整页（`raw_size + 4096`，每 worker 成本 4KB），这样 slack>0 的所有情形都能覆盖；或者按 depthfirst bot 的建议在 `slack != 0` 时显式报错。二者选一，当前"半覆盖"状态最差。

### ⚠️【正确性】`_build_shared_kv_mr` 每层只取第一个注册区域，split-K/V 层的 V 区域不在 MR 覆盖内 `[已验证代码 / 实际可达性为推测]`

- **问题**：`moriio_connector.py:1710-1714` 用 `next(iter(self._iter_layer_registration_regions(layer_name)))` 只取每层第一个区域来算 `end`。而 `iter_layer_registration_regions`（`moriio_layout.py:258-259`）对 `split_kv_regions=True` 的层（5D `(2, num_blocks, …)` 布局）会返回 K、V 两个区域，V 区域从 `K 区域末尾` 延伸到 `2×K 区域`。对最后一层，`end` 只覆盖到 K 的末尾，V 的整段地址都落在注册 MR 之外。
- **影响**：这类层的 V 传输（偏移是整 tensor 相对、含 K→V stride，见 `moriio_layout.py:330-339`）会读写超出 MR 的地址 → RDMA 传输失败（或取决于 mori 是否做边界检查）。当前 vLLM main 的 `create_kv_cache_views` 只产标准化 4D `[B, H, N, C]` 视图（`kv_cache_interface.py:253`），5D 分支在 `test_moriio_kv_layout.py` 中仍被积极单测覆盖，属于"布局模块还在支持、共享 backing 新路径没覆盖"的潜伏缺口。
- **行动**：建议作者在 shared-backing 模式下对所有区域求和计算 `end`（并校验各区域偏移），或显式拒绝 `split_kv_regions=True` 的层进入 shared-MR 路径，避免未来布局回归时变成静默错地址。

### ⚠️【正确性】`_remote_layer_mr_offset` 的静默 0 回退用本地共享状态门控远端偏移 `[推测]`

- **问题**：`moriio_connector.py:1733` 的守卫 `if not self.kv_layer_mr_offset or not remote_engine_id: return 0` 用**本地**是否 shared-backing 来决定是否施加**远端** MR 偏移；`metas` 为空或 `idx` 越界（1739 行）也静默返回 0。若本地为非 shared（旧版本/旧注册路径）而远端是 shared，远端偏移被跳过，读写落到远端 MR 起始处——即错误层的 KV 数据，无任何报错。
- **影响**：混合版本 disagg（两侧 vLLM 不一致）或异构注册场景下静默数据损坏。同版本同模型部署（当前唯一被支持的方式）不受影响；非 shared 远端时 `层地址 - MR 基址` 自然为 0，所以无条件计算远端偏移本身是安全的。
- **行动**：建议作者去掉对本地状态的依赖，无条件计算远端偏移，并在 `metas` 为空 / `idx` 越界的回退路径加 warning 日志而不是静默返回 0。

### ⚠️【测试】新地址数学无单测、slack>0 分支未在真实硬件覆盖、CI 尚未全绿

- **问题**：`_build_shared_kv_mr`（slack、负 storage_offset、`end` 边界）和 `_remote_layer_mr_offset` 均无新增单测，只有 write-plan 缓存 key 的测试被更新。作者在 MI300 上的实测（2MB 对齐的大分配）只会走 `slack == 0` 分支，`-slack` 视图分支从未在硬件上执行过。CI 状态：Buildkite #85600、#85621 failed，最新 #85640 blocked（job 明细公共 API 不可见，无法归因）；mergify 两次提示 pre-commit 失败（8-26 01:39 与 06:04，需确认最终 commit 已修复）。
- **影响**：边界算术（尤其 Finding 1 的 4K 倍数情形）回归风险无人拦截；ROCm CI 未确认绿。
- **行动**：建议作者补充 `_build_shared_kv_mr` 的单元测试（slack=0 / slack>0 / raw_size 为 4K 倍数三种情形，可用构造的 misaligned fake storage），并在最终 commit 上拿到绿色的 AMD CI 与 pre-commit。

### 📝 关于 depthfirst bot 的"注册越界内存"评论：同页内存，非未拥有内存

- bot 认为 `-slack` 视图把"分配起始前最多 4095 字节的未拥有 GPU 内存"暴露给 RDMA 对端。实际上 `slack < 4096` 意味着 `[ptr - slack, ptr)` 与分配的第一页同属一个 4KB 页，而 GPU 内存按整页分配，这段字节属于本进程（注册合法、对端也不会被寻址到）。真正值得确认的是：mori 侧 `register_torch_tensor` 若按 `untyped_storage().nbytes()` 而非 tensor `numel` 注册，会把范围延伸到分配尾页之外导致 `ibv_reg_mr` 失败——建议 review 时向作者/mori 侧确认注册语义（本 PR 的 slack>0 分支未被硬件验证过，此风险未被排除）。

### 📝 `_remote_layer_mr_offset` 依赖"两侧层序与区域数一致"的隐含不变式

- `idx = self.layer_base_addr_index[layer_name]` 是本地逐层索引，却用于索引远端的扁平 `kv_caches_base_addr` 列表（每区域一项）。同模型同版本 + 每层单区域时成立（两侧 dict 插入顺序一致），但代码里没有注释或断言说明这个不变式。建议作者加一行注释或轻量校验，避免后续有人改动 `register_kv_caches` 的区域计数逻辑时破坏对齐。

## 4. 现有讨论 (Existing Discussion)

- **depthfirst-app[bot]**（8-25）：对 `-slack` 负 storage_offset 提出 LOW 级安全评论，建议 `slack != 0` 时直接报错（见上文 📝 的回应分析）。
- **CI 互动**：@shen-shanshan 三次触发 `/ci run`（#85600 / #85621 / #85640），前两次 failed、最新 blocked；mergify 两次提示 pre-commit 失败。作者 @avininjamay8 邀请 @lcskrishna review。
- 尚无 maintainer 的实质性 review 意见。

## 5. 结论 (Verdict)

⚠️ **NEEDS WORK** — 主路径修复方向正确且有真实硬件验证（GSM8K 0.968），WRITE 缓存 key 加层名和 read/write 双侧偏移应用都做对了；但页填充对"4KB 整数倍总尺寸"的边界漏洞（Finding 1）建议用一行修复消除，split-K/V 区域覆盖与静默 0 回退应至少加防护或注释，测试与 CI 需要补齐后才能合入。
