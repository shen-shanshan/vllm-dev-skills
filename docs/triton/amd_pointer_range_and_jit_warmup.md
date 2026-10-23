# AMD 后端的 2 GB 指针特化与 vLLM JIT warmup：`ptr_range_bytes` 的原理

> 背景：PR #60834（ROCm DeepSeek-V4 MXFP4 indexer）review 时，reviewer 问 `_compress_pool_kernel` 是否需要注册 JIT warmup。排查过程中发现，ROCm 上 warmup 编译出来的 kernel 和运行时实际用到的 kernel 不是同一个版本。为了修复这个问题，给 `TritonWarmupTensor` 新增了 `ptr_range_bytes` 字段。本文解释这个字段背后的原理。
>
> 涉及代码：
> - `vllm/model_executor/warmup/jit_warmup_triton_helper.py`：`TritonWarmupTensor.ptr_range_bytes`
> - `vllm/models/deepseek_v4/amd/rocm.py`：`_compress_pool_warmup_inputs`、`compress_pool`
> - Triton：`triton/backends/amd/compiler.py`（`is_within_2gb`、`get_tensor_specialization`）

一句话概括：**`ptr_range_bytes` 的作用是在 JIT warmup 时告诉 Triton，这块 tensor 背后的存储有多大。Triton 的 AMD 后端会根据存储是否超过 2 GB 编出两种不同的代码。**

---

## 1. Triton 会按参数的特征分别编译

Triton kernel 不是只编译一份。每次调用时，Triton 先从实际传入的参数里提取一组"特化特征"，和参数类型一起组成编译缓存的 key。key 一样就复用已编译的版本，不一样就重新编译一次。

通用的特化特征有：

| 参数类型 | 特征 | 对生成代码的影响 |
|---|---|---|
| 整数 | 值是否为 1；是否能被 16 整除 | 常量折叠；按对齐方式生成访存指令 |
| 指针 | 地址是否按 16 字节对齐 | 能否使用向量化访存 |

AMD 后端在此基础上**多加了一个指针特征**，代码在 `triton/backends/amd/compiler.py`：

```python
def is_within_2gb(arg):
    MAX_INT_32 = 2**31 - 1
    if hasattr(arg, "ptr_range"):                 # 占位对象：问它自己
        return arg.ptr_range() <= MAX_INT_32
    if isinstance(arg, torch.Tensor):             # 真实 tensor：看底层存储大小
        return arg.untyped_storage().size() <= MAX_INT_32
    return False

def get_tensor_specialization(arg, **kwargs):
    ret = BaseBackend.get_tensor_specialization(arg, **kwargs)
    if knobs.amd.use_buffer_ops and HIPBackend.is_within_2gb(arg):   # 默认开启（AMDGCN_USE_BUFFER_OPS=1）
        ret += "S"            # 编译时变成参数属性 tt.pointer_range = 32
    return ret
```

也就是说，**一个指针参数指向的底层存储如果不超过 2 GB，就会被打上 `S` 标记。** 在生成的 IR 里，这个标记体现为参数上的 `tt.pointer_range = 32` 属性。

## 2. AMD 为什么关心 2 GB：buffer 指令

AMD GPU（CDNA）访问显存有两类指令：

```
global_load / global_store                buffer_load / buffer_store
────────────────────────────              ─────────────────────────────────────
每个线程自己算出 64 位完整地址             基址 + 大小放在一个"资源描述符"里
  addr = base + offset（64 位加法）          （所有线程共享，放在标量寄存器）
  每个地址占 2 个向量寄存器                 每个线程只需要一个 32 位偏移量
没有越界保护                               硬件做越界检查：越界读返回 0、越界写丢弃
```

buffer 指令更省寄存器，地址计算更便宜，还能免费拿到越界保护，所以 Triton 的 AMD 后端会尽量把 `tl.load` 和 `tl.store` 改写成 buffer 指令。但它有一个前提：**偏移量必须能放进 32 位有符号整数**，也就是整个可寻址范围不能超过 2³¹−1 字节，约 2 GB。

编译器本身无法知道运行时的 tensor 有多大，所以要由调用方在编译之前给出这个承诺，这就是 `tt.pointer_range = 32` 的作用。

- **有这个承诺**：可以对这个指针使用 buffer 指令。
- **没有这个承诺**：只能使用 64 位的 global 指令。

这两种情况生成的机器码不同，所以它必须作为编译 key 的一部分：同一个 kernel，存储小于等于 2 GB 和大于 2 GB 会编译出两个不同的二进制。

有一个关键细节：Triton 判断时看的是 `untyped_storage().size()`，也就是**整块底层存储的大小，而不是这个视图本身的大小**。一个 tensor 即使只是某块大 tensor 的一小段切片，只要底层存储超过 2 GB，就会被判定为"大"。

## 3. vLLM 的 JIT warmup 怎么让 Triton 提前编译

JIT warmup 发生在服务启动阶段，这时还没有真实的输入数据，有些 buffer 甚至还没分配。所以 vLLM 用 `TritonWarmupTensor` 这种占位对象来"假装"是 tensor。它只描述编译需要的元信息：dtype、shape、stride、是否对齐，然后让 Triton 按这些信息算出 key 并提前编译。

问题出在占位对象原来是这样写的：

```python
def ptr_range(self) -> int:
    return 0          # 固定返回 0，相当于"我的存储永远不超过 2 GB"
```

Triton 自己的占位对象也是这么写的，注释原话是 "optimistically assumes 32 bit pointer range"。对大多数参数来说这没问题，比如 positions、slot_mapping、block_table、pooled_k 这块 scratch，它们本来就很小，运行时也确实不超过 2 GB，两边一致。

## 4. 为什么在这里会出问题

`_compress_pool_kernel` 的 `state_cache` 参数是 compressor 的状态缓存。它是 vLLM 统一分配的 **KV cache 大池中的一个视图**。这个池子会把 GPU 剩余的大部分显存都占满，每张卡上有几十 GB。所以：

```
                       state_cache 指针                   编译 key
warmup 占位对象      ptr_range() = 0      → 判定 ≤2GB → 带 S（tt.pointer_range=32）  ┐
运行时真实 tensor    untyped_storage()       → 判定 >2GB → 不带 S                       ┘ 两个 key 不一样
                     = 整个 KV 池（几十 GB）
```

warmup 编好的那个版本，运行时永远用不上；运行时真正需要的版本，只好等到第一次被调用时才编译。

### 实验证据

用全新的 `TRITON_CACHE_DIR` 启动 DeepSeek-V4-Pro（TP8、MXFP4 indexer）服务，同一个 kernel 出现了两个编译产物，它们的 TTIR 只有一处不同：

```
warmup 编的（TNX6XA2L…）: %state_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16, tt.pointer_range = 32}
运行时用的（KNQGK5DX…）:  %state_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16}
```

而且后一个是在 JIT warmup 结束之后才编译出来的。

之前没被 jit monitor 抓到，只是因为运气好：CUDA graph capture 的预热会先调用一次这个 kernel，把运行时版本编译好了，而 jit monitor 是在这之后才开启的。但这一点完全依赖于 capture 流程恰好覆盖到这个 kernel。

## 5. `ptr_range_bytes` 做了什么

做法很简单：让占位对象可以声明"我背后的存储有多大"。

```python
ptr_range_bytes: int = 0          # 默认 0：行为和原来完全一样

def ptr_range(self) -> int:
    return self.ptr_range_bytes   # Triton 调这个来判断是否 ≤2GB
```

然后在 `compress_pool` 的 warmup 里，只给 `state_cache` 生成两种情况：

```python
for pool_bytes in (0, 1 << 32):     # ≤2GB 和 >2GB 各编译一份
    state_cache=TritonWarmupTensor(..., ptr_range_bytes=pool_bytes)
```

两种情况都编译，是因为 KV 池的实际大小取决于部署方式，比如 `gpu_memory_utilization` 的设置、显卡的显存大小，或者测试时的小显存配置。这里没法提前确定会落在哪一边，多编译一份只多花大约 1 秒，就能保证无论哪种情况都能命中。

### 修复后的验证

再用全新的 Triton cache 启动服务：

| 编译版本 | 编译时间 | 说明 |
|---|---|---|
| 运行时实际使用的版本（不带 `pointer_range`，`KNQGK5DX…`） | 04:00:27 | 落在 JIT warmup 阶段（04:00:25–04:00:57）之内 |
| 小池版本 | 04:00:14 | 在 warmup 之前就编好了 |

warmup 结束之后没有任何新的编译，推理期间 jit monitor 也没有报这个 kernel。

## 6. 影响范围

- **对 CUDA 没有影响。** NVIDIA 后端计算特化时只看地址对齐（Triton `backends/compiler.py` 的 `get_tensor_specialization`），根本不会调用 `ptr_range()`。
- **对 ROCm 上已有的 warmup 没有影响。** 默认值是 0，和原来的返回值一样；只有显式设置了这个字段的地方才会变化。
- **它不改变运行时的代码选择。** 存储超过 2 GB 时，那个指针本来就只能走 global 指令。这个改动只是让 warmup 提前编译出运行时真正会用到的那个版本。

## 7. 其他 kernel 可能也有同样的问题（未核实）

凡是在 ROCm 上接收"KV 池视图"参数、又注册了 JIT warmup 的 kernel，都可能踩到同一个坑。jit monitor 在 MXFP4 和 FP8 两种 indexer 路径下都报过下面这些已注册 warmup 的 kernel 在推理期间编译：

- `DequantizeAndGatherKCacheKernel`
- `_compress_gather_split_sparse_attn`
- `_combine_topk_swa_indices_kernel`
- …

它们很可能是同一个原因，但没有逐个核实。适合另开 issue 或 PR，用 `ptr_range_bytes` 修复。

### 排查方法

1. 设置一个全新的 `TRITON_CACHE_DIR` 启动服务，避免旧缓存掩盖问题。
2. 对比 `<cache>/<hash>/<kernel>.hsaco` 的写入时间和日志里 `JIT kernel warmup starting/finished` 的时间：如果有编译产物晚于 warmup 结束时间，说明 warmup 没覆盖到运行时用的版本。
3. 对比同一个 kernel 多个编译产物的 `.ttir` 中 `tt.func` 的参数属性，例如是否带 `tt.pointer_range = 32`、`tt.divisibility = 16`，就能定位是哪个参数的特化不一致。
