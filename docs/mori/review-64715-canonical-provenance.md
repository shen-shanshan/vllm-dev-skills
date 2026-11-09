# PR #64715 review 意见分析：UMBP 的 canonical KV 来源无法保证

> - **Review 链接**：https://github.com/vllm-project/vllm/pull/64715#discussion_r4490119163
> - **Reviewer**：charxwu（2026-11-06）
> - **位置**：`vllm/distributed/kv_transfer/kv_connector/v1/umbp/connector.py:109`（commit `cf59c081a4`），对应 `UMBPStoreConnector.only_restores_canonical_kv`
> - **状态**：用户已确认方案 B（2026-11-09）。代码已改，单测和冒烟测试都通过，已 commit 为 `c1050779b7`（未 push）；回复草稿见第五节

## 原文

> `key_namespace="auto"` fingerprints the cache layout, but does not guarantee canonical provenance. Consider this sequence:
>
> 1. Another child in `MultiConnector` restores non-canonical KV.
> 2. UMBP receives [the same blocks](https://github.com/vllm-project/vllm/blob/dd228c3746d828fa0775b67167756a3cdb2a3829/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L513-L523) and can [save them](https://github.com/vllm-project/vllm/blob/dd228c3746d828fa0775b67167756a3cdb2a3829/vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py#L651-L686) under its auto namespace.
> 3. A later UMBP-only engine restores those blocks and returns `True` here, allowing the AITER path even though the KV may contain NaNs.
>
> Since AITER does not scrub NaNs, could we keep this capability `False` until canonical provenance is guaranteed?

被评论的代码：

```python
@classmethod
def only_restores_canonical_kv(cls, kv_transfer_config: KVTransferConfig) -> bool:
    # Objects are whole blocks this cache layout stored, and the derived
    # namespace fingerprints every per-layer spec. An explicit
    # key_namespace skips that fingerprint, so it may hold foreign bytes.
    extra = kv_transfer_config.kv_connector_extra_config
    return extra.get("key_namespace", "auto") == "auto"
```

## 一、这条意见在说什么

### 背景：为什么要区分 KV 是不是 vLLM 自己写的

在 ROCm 上，DSv4.1 有两条快路径：

- aiter sparse MLA；
- `fp8_ds_mla` extra cache 的 NaN-free 优化。

见 `deepseek_v41/amd/rocm.py` 中的 `_aiter_sparse_mla_enabled` 和 `_trust_dsv4_extra_cache_nan_free`。

这两条快路径都假设 GPU KV cache 中的每一个字节都由 vLLM 自己按当前 layout 写入。本文把这种 KV 称为 **canonical KV**，它不会在未写入的位置留下 NaN 之类的垃圾值。aiter kernel 本身**不会清洗 NaN**。如果它读到外部写入的字节（非 canonical），NaN 会一直传播到输出。

为此，本 PR 增加了 hook `only_restores_canonical_kv`：

- connector 声明"只会恢复 canonical KV"时返回 True，ROCm 保留快路径；
- 否则返回 False，关闭快路径。

UMBP 当前的判断依据是 `key_namespace` 是否为 `auto`，是则返回 True。理由是 auto namespace 对 layout 做了指纹，所以池中的对象一定由使用同一 layout 的 vLLM 写入。

### reviewer 指出的漏洞

reviewer 的意思是：**namespace 只能证明这些字节的 layout 对得上，不能证明这些字节是谁写的。** layout 对得上，数据仍可能是从别处搬进来的。

他举的例子分三步：

```
① Engine A：MultiConnector = [NIXL（或其他会写入外来字节的 connector）, UMBP]

   远端 ──NIXL 写入──▶ GPU block #7 (可能含 NaN 的外来字节)
                           │
                           │ MultiConnector 把同一个请求的真实 block 也交给 UMBP
                           │ (multi_connector.py:513-523，num_external_tokens=0)
                           ▼
   UMBP scheduler：这个新请求不是由我加载的，那就把它已算好的 block 都存一份
   (scheduler.py:651-686，_store_plans 从第 0 个 block 开始存)
                           │
                           ▼
   standalone 池：key = umbp:vllm:v1:<auto namespace>:...:g0:<hash>
                  内容 = NIXL 写入的外来字节   ◀── 被"洗白"成了普通对象

② 池在整台机器上共享，并且跨 engine 持久存在

③ Engine B：只配了 UMBP（同模型、同 layout → 同一个 auto namespace）

   UMBP.only_restores_canonical_kv() == True   → 打开 aiter sparse MLA
   命中 <hash> → 把 ① 中的外来字节恢复进 GPU → aiter 读到 NaN ✗
```

关键点：

1. **Engine A 本身没有问题。** MultiConnector 的 hook 要求所有子 connector 都返回 True。NIXL 返回 False，所以 Engine A 整体为 False，快路径本来就是关闭的。
2. **出问题的是另一个 engine。** standalone 池跨 engine 共享，而 namespace 只包含模型和 layout，不包含"写入方配置了哪些 connector"。因此 A 存进去的外来字节，在 B 中被当作 canonical KV，B 就打开了快路径。

被存进池的也不只是"本步刚加载的 block"：

- NIXL 加载的 block 会留在 A 的 GPU prefix cache 中；
- 之后别的请求在本地命中这些 block 时，UMBP 同样会把它们存进池。

所以无法只靠"跳过兄弟 connector 刚加载的 token"来修复。

reviewer 建议：在能保证数据来源之前，UMBP 的这个 hook 一律返回 False。

## 二、修复方案

### 方案 A（reviewer 的建议）：UMBP 不再覆写 hook，一律返回 False

- **优点**：只改一行，绝对安全。
- **缺点**：
  - ROCm 上 DSv4.1 接入 UMBP 后，aiter sparse MLA 和 NaN-free 快路径都会被关闭，而保留快路径正是本 PR 的主要目的之一。
  - 之前的冒烟测试、GSM8K 精度和性能数据都是在快路径打开时测得的，改动后需要重测，性能损失目前未知。

### 方案 B（推荐）：把数据来源写进 namespace，让被污染的数据和干净的数据落在不同的 key 空间

思路：在 `UMBPNamespace.from_vllm_config` 的指纹中增加一个字段。如果本 engine 配置的 connector 整体上可能写入外来字节，就在 namespace 中加入 `"foreign_kv": true`。判断直接复用现有的 `kv_connector_may_load_foreign_kv(vllm_config)`。它读取的是顶层 config，所以能看到 MultiConnector 中的其他子 connector。

```
Engine A：[NIXL, UMBP] → may_load_foreign_kv = True  → namespace = H(... , foreign_kv=true)  = N_dirty
Engine B：[UMBP]       → may_load_foreign_kv = False → namespace = H(...)                    = N_clean

池中：N_dirty:*:<hash> 与 N_clean:*:<hash> 是两个不同的 key
→ B 永远查不到 A 存入的对象。B 的 hook 返回 True 时，它能读到的只有纯 UMBP engine 写入的数据
```

**为什么这样就安全了**：

- 在只配了 UMBP 的 engine 中，GPU KV 只有两个来源：模型自己计算，或者从 `N_clean` 恢复。
- `N_clean` 中的对象只可能由同样只配了 UMBP 的 engine 写入。
- 这两点构成闭环，所以 hook 返回 True 是成立的。
- Engine A 的 hook 本来就是 False，它读写 `N_dirty` 不会影响快路径的判断。

**实现要点**：

- 只在 `foreign_kv=True` 时才把这个字段加入指纹，做法与现有的 `speculative` 字段相同。这样所有现有配置（只配 UMBP 的 engine）的 namespace 保持不变，已经存入池中的对象不会失效。
- `UMBPStoreConnector.only_restores_canonical_kv` 的逻辑不变，仍是 `key_namespace == "auto"`；只修改注释，说明 auto namespace 同时编码了 layout 和数据来源。
- 显式设置 `key_namespace` 时，hook 仍返回 False。用户自定义的 namespace 无法保证数据来源，这一点保持原样。

**需要接受的代价**：

- 配置为 `[NIXL, UMBP]` 等组合的 engine，不再与只配 UMBP 的 engine 共享池中的对象。从安全角度看这正是需要的，但跨 engine 的复用会减少一些。
- 判断偏保守。例如 `[OffloadingConnector, UMBP]` 实际上也只写 canonical KV，但 OffloadingConnector 没有覆写 hook，也会被划入 `N_dirty`。这只会损失一些复用，不影响正确性。

**改动范围**：

1. `umbp/data.py`：在 `UMBPNamespace.from_vllm_config` 中增加上述条件字段，用延迟 import 引入 `kv_connector_may_load_foreign_kv`，避免循环导入。
2. `umbp/connector.py`：只修改 hook 的注释。
3. `test_umbp_core.py`：新增测试，验证两点：
   - `[UMBP]` 与 `MultiConnector[外来 connector, UMBP]` 得到的 namespace 不同；
   - 只配 UMBP 时，namespace 与改动前一致。

   上一轮 review 修复时，用户已同意在该文件中新增测试，这次待确认。
4. 同步更新以下文档：
   - PR 描述 `pr-umbp-dsv41-flash-standalone.md`；
   - `docs/features/umbp_store_connector_usage.md` 中关于 namespace 的说明，增加 connector 组合一项；
   - 进度文档 `PROGRESS-umbp-dsv41-flash-standalone.md`。
5. 回复 reviewer，可以由我起草英文回复。大意是：已采纳，但不是直接返回 False，而是把数据来源纳入 namespace 来保证来源，并附上上面的闭环推理。

**验证计划**：方案 B 只修改 namespace 的计算和单测，不涉及推理路径。除单测外，只跑一次 TP2 冒烟，确认三点：

- 只配 UMBP 时 namespace 不变；
- 恢复仍然命中；
- 快路径仍然打开（日志中没有 "is ignored"）。

## 三、用户决定（2026-11-09）

1. 采用方案 B。
2. 同意在 `test_umbp_core.py` 中新增测试。
3. 起草给 reviewer 的英文回复，并重点说明：我们不想损失 aiter 快路径的性能，因为使用 UMBP 本来就是为了在 AgentX 上获得更好的性能。

## 四、实现情况（未 commit）

**关键实现细节**：MultiConnector 创建子 connector 时，会浅拷贝 `vllm_config`，并把其中的 `kv_transfer_config` 换成该子 connector 自己的配置（`multi_connector.py` 的 `_get_connector_classes_and_configs`）。因此 UMBP 从 `vllm_config.kv_transfer_config` 看不到兄弟 connector。

解决办法：namespace 推导时改读当前 engine 的 config（`get_current_vllm_config_or_none()`）。依据如下：

- worker 创建 connector 的两条路径，即 `WorkerWrapperBase.initialize_from_config` 和 `extend_kv_cache`，都包在 `set_current_vllm_config(self.vllm_config)` 中。
- `extend_kv_cache` 的注释也写明 "The KV connector created here reads the current config"。

如果读不到当前 config，就退回使用传入的 config。在只配 UMBP 的场景下，两者都会得到 False，结果一致。

**改动**：

| 文件 | 改动 |
|---|---|
| `umbp/data.py` | 新增 `_engine_may_load_foreign_kv()`。`UMBPNamespace.from_vllm_config` 在该函数返回 True 时，向指纹加入 `"foreign_kv": true`（仅在为 True 时才加，只配 UMBP 时 namespace 不变） |
| `umbp/connector.py` | 更新 `only_restores_canonical_kv` 的注释，说明 derived namespace 同时区分了数据来源 |
| `docs/features/umbp_store_connector_usage.md` | 在 Sharing 一节说明：与"可能加载外来 KV 的 connector"组合使用的 engine 会使用独立的 key |
| `tests/v1/kv_connector/umbp_test_utils.py` | 测试用的假 config 补上 `kv_connector="UMBPStoreConnector"`，因为 namespace 推导现在要查询 connector 类 |
| `tests/v1/kv_connector/unit/test_umbp_core.py` | 新增 `test_namespace_separates_engines_whose_siblings_load_foreign_kv`，覆盖 3 个用例：无当前 config、`Multi[UMBP]`、`Multi[ExampleConnector, UMBP]`。同时验证显式 `key_namespace` 不受影响 |

另外，新 main 上的 pre-commit hook `check-docstring-style` 自动把本 PR 早先加入的几处 docstring 中的 ` ``x`` ` 改成了 `` `x` ``。

**验证**：

- 新测试在没有修复的代码上会失败（`Multi[ExampleConnector, UMBP]` 用例），修复后通过。
- `tests/v1/kv_connector/unit/test_umbp_*.py` 共 341 个测试通过。
- pre-commit 全部通过。
- TP2 冒烟（2026-11-09，base main `9ff2963a79` + 本 PR + 本修改，GPU 2,3，Python 前端）：
  - 从 UMBP 恢复 24,320 token，恢复后的回答与 GPU 直出一致（`match: true`）。
  - store 379 个对象，load 380 个，失败 0。
  - engine 日志中没有 Traceback，没有 NaN，也没有 "is ignored"（快路径仍然开启）。
  - 以上与改动前的冒烟结果完全相同。
  - 第一次运行没有跑起来：新 main 的 #59478 在 INT4 QuickReduce 下会调用 aiter `QuickAllReduceInt4.compile()`，而容器里的 aiter `0.1.28.post1` 没有这个方法，worker 在初始化 TP 通信时就崩了，与本修改无关。运行脚本的 `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` 现在可以用 `QR_QUANT` 覆盖，这次冒烟用的是 `QR_QUANT=INT8`。

## 五、给 reviewer 的回复草稿（英文）

> Thanks, good catch. `key_namespace="auto"` pins the layout, not who wrote the bytes, and your sequence does let a MultiConnector sibling's KV reach a UMBP-only engine through the shared standalone pool.
>
> We'd rather not return `False` here, though. On ROCm that turns off the aiter sparse MLA and NaN-free extra-cache paths for every UMBP deployment of DeepSeek-V4.1-Flash, and that combination is why we use UMBP at all. Every AgentX number in the description was measured with those paths on: +46% throughput and −78% mean TTFT at TP2 c192, and +3.8% throughput at DP2 c256. Falling back to the Triton sparse attention for all UMBP users, to guard against a MultiConnector setup none of them run, gives back part of what the connector is for.
>
> So this PR now guarantees the provenance instead of dropping the capability:
>
> - The derived namespace also records whether any connector of the engine may load foreign KV (`kv_connector_may_load_foreign_kv` on the engine config). If one may, `"foreign_kv": true` joins the fingerprint.
> - The child UMBP connector only sees its own `kv_transfer_config` inside a MultiConnector, so it reads the engine's current config. Workers set that config while creating their connector, in both `initialize_from_config` and `extend_kv_cache`.
> - In your example, engine A (`[NIXL, UMBP]`) now stores under a namespace of its own, and engine B (UMBP-only) restores from the plain one. B never sees A's objects.
> - The argument for B: its KV comes only from its own forward passes or from the plain namespace, and only UMBP-only engines write that namespace. So `only_restores_canonical_kv() == True` holds. An explicit `key_namespace` still returns `False`.
> - The field is added only when set, so UMBP-only namespaces and pools are unchanged.
>
> The cost is that mixed engines no longer share objects with UMBP-only ones, and the check is conservative: a sibling that never overrides the hook, such as `OffloadingConnector`, also gets the separate namespace. That costs some reuse but never correctness.
>
> `test_namespace_separates_engines_whose_siblings_load_foreign_kv` covers the mixed case; it fails without the change.
