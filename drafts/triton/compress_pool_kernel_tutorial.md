# 以 `_compress_pool_kernel` 为例学习 Triton 算子开发

> 学习对象：vLLM PR [#60834](https://github.com/vllm-project/vllm/pull/60834) 中的 `_compress_pool_kernel`
> 本地代码：`vllm/vllm/models/deepseek_v4/amd/rocm.py:408`（kernel 定义）、`rocm.py:1771`（调用处）
> 相关代码：`vllm/vllm/models/deepseek_v4/common/ops/save_partial_states.py`（负责写入 `state_cache`），`vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:713`（原版融合 kernel）
> 配套脚本：同目录下的 [`triton_learn_compress_pool.py`](./triton_learn_compress_pool.py)。它把 kernel 原样拷出来运行，并用纯 PyTorch 参考实现对比结果。已在 AMD Instinct MI350X（torch 2.12.0+rocm10.0.0，triton 3.8.0）上验证两者一致。

---

## 第一部分：Triton 入门的核心概念

### 1.1 为什么要写算子？Triton 是什么？

PyTorch 里一行 `(kv * score.softmax(0)).sum(0)` 背后其实是好几个 GPU kernel：softmax 一个、乘法一个、求和一个。每个 kernel 都要把数据从显存（HBM）读进来、算完再写回去。显存读写很慢，所以把多步计算**融合（fuse）**成一个 kernel，让数据只读一次、只写一次，通常能快很多。这就是写算子的主要动机。

写 GPU kernel 传统上用 CUDA/HIP C++，要自己管理每一个线程。**Triton 是一个用 Python 写 GPU kernel 的 DSL**，它的抽象层次更高：

| | CUDA / HIP | Triton |
|---|---|---|
| 编程粒度 | 单个**线程**：「我这个线程处理第几个元素」 | 一个**程序块（program）**：「我这个程序处理一整块数据」 |
| 线程怎么分工 | 自己写 | 编译器自动分配 |
| 共享内存、访存合并 | 自己写 | 编译器自动处理 |
| 跨平台 | CUDA 只能用于 NVIDIA | 同一份代码能编译到 NVIDIA 和 AMD |

一句话：**在 Triton 里，你操作的是"一小块张量"，不是"一个标量"。**

### 1.2 必须掌握的 10 个概念

#### ① `@triton.jit`：声明一个 kernel
```python
@triton.jit
def my_kernel(x_ptr, n, BLOCK: tl.constexpr):
    ...
```
加了这个装饰器，函数就不再由 Python 解释执行，而是在第一次调用时编译成 GPU 机器码（JIT，即时编译）。函数体里只能用 Triton 支持的语法（`tl.*`、基本运算、if/for），不能调用 torch 或 numpy。

#### ② Grid 和 `tl.program_id`：并行是怎么来的
启动 kernel 的写法很特别，方括号里填的是 **grid**：
```python
my_kernel[(num_programs,)](args...)
```
意思是同时启动 `num_programs` 个程序实例，**每个实例运行同一份代码**（这叫 SPMD）。实例之间靠 `tl.program_id(0)` 区分自己是第几个，跟"多个工人拿同一份说明书，各自查看工号再决定处理哪一份数据"一样。grid 最多可以有 3 维，用 `program_id(0/1/2)` 取。

#### ③ 指针（ptr）与指针算术
kernel 拿不到 tensor 对象，只能拿到**首元素的地址**。传 torch.Tensor 进去时，Triton 会自动把它转换成指针，类型就是元素的 dtype。访问第 i 个元素的地址是 `ptr + i`，单位是**元素**，不是字节。

#### ④ Stride：多维张量在内存里的排布
显存是一维的。对形状 `[A, B, C]` 的连续张量来说，`x[a, b, c]` 的地址是
```
ptr + a * stride0 + b * stride1 + c * stride2      # 连续张量时 stride = (B*C, C, 1)
```
所以 kernel 一般会接收 `tensor.stride(i)` 作为参数，自己算地址。这比假定"一定是连续张量"更通用，比如 tensor 是一个切片时 stride 就不规则。

#### ⑤ `tl.arange`：生成一个"块"
```python
offs = tl.arange(0, 8)     # [0,1,2,3,4,5,6,7]，一个长度为 8 的块
```
这是 Triton 里最重要的函数：后面所有向量化计算都是从它开始的。**限制：长度必须是编译期常量，并且是 2 的幂**（8、16、128……）。

#### ⑥ `tl.load` / `tl.store` 和 mask
```python
x = tl.load(ptr + offs, mask=offs < n, other=0.0)   # 批量读
tl.store(out_ptr + offs, x, mask=offs < n)          # 批量写
```
- `ptr + offs` 是一个**指针块**，一次读回一个数据块。
- `mask`：为 False 的位置**不访问内存**（防止越界），对应的值用 `other` 填充。
- **`other` 填什么值很讲究**：后面求和就填 0，取 max 或 softmax 就填 `-inf`。这正是本例的一个关键点。

#### ⑦ 广播（broadcast）与二维块
和 NumPy 一样：
```python
rows = tl.arange(0, 8)     # 形状 [8]
cols = tl.arange(0, 128)   # 形状 [128]
rows[:, None]              # 形状 [8, 1]
cols[None, :]              # 形状 [1, 128]
rows[:, None] * 128 + cols[None, :]   # 形状 [8, 128]，一个二维偏移网格
```
用二维偏移网格加上指针，就能**一次 load 出一个二维 tile**。

#### ⑧ 块级运算与归约
`tl.sum(x, axis=0)`、`tl.max`、`tl.softmax(x, dim=0)`、`tl.where(cond, a, b)`、`x.to(tl.bfloat16)` 都作用在整个块上。

#### ⑨ `tl.constexpr`：编译期常量
```python
HEAD_SIZE: tl.constexpr
```
标了 `constexpr` 的参数会**写死进编译出的代码里**，所以：
- 可以用在 `tl.arange` 的长度这种必须是常量的地方；
- 编译器可以针对它做优化，比如循环展开、整除变成移位；
- 代价是**每出现一个新取值就重新编译一次**（之后会缓存）。所以只把"形状类"的、取值很少的参数设成 constexpr。

没标 constexpr 的整数参数（例如 `block_size`）是运行时值，换值不需要重新编译。

#### ⑩ `num_warps`：一个 program 用多少硬件线程
`kernel[grid](..., num_warps=4)` 决定每个 program 分到几个 warp。在 AMD 上一个 warp（wavefront）是 64 个线程，NVIDIA 上是 32 个。数据块小就用小值，避免线程闲置。

### 1.3 最小示例：向量加法
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                      # 我是第几个程序
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # 我负责的那一段下标
    mask = offs < n                             # 最后一段可能越界
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
```
**几乎所有 Triton kernel 都是这几步：program_id 确定负责哪块 → arange 生成偏移 → 指针算术 → 带 mask 的 load → 计算 → store。** 带着这个模板去看下面的 kernel。

---

## 第二部分：先弄懂这个 kernel 在算什么

不懂业务就看不懂代码，所以先讲背景。

### 2.1 DeepSeek V4 的 KV 压缩（Compressor）

DeepSeek V4 会把每 `COMPRESS_RATIO` 个 token 的 KV **压缩成 1 个**。对 indexer 来说 ratio=4，也就是每 4 个 token 合成 1 个压缩 token。压缩方法是 **softmax 门控的加权池化**：每个 token 除了产生 `kv` 向量，还产生一个同样维度的 `score` 向量。对窗口内的 token，在每个通道上分别对 score 做 softmax，再用得到的权重对 kv 加权求和：

```
compressed[c] = Σ_i  softmax_i(score[i, c]) * kv[i, c]      # c 是通道，i 是窗口内的 token
```

**Overlap（重叠）**：ratio=4 时 `OVERLAP=1`，窗口长度是 `(1+1)*4 = 8` 个 token，也就是当前 4 个加上前一组的 4 个。每个 token 投影出**两份** head（`coff=2`）：前一份在它作为"前一组"时使用，后一份在它作为"当前组"时使用。

### 2.2 数据在显存里怎么摆

前一个 kernel `SavePartialStatesKernel`（`common/ops/save_partial_states.py`）在**每个 token** 到来时，把它的 kv 和 score 写进 `state_cache`：

```
state_cache 形状: [num_blocks, block_size, 2 * STATE_WIDTH]   (fp32)
                    ↑分页的块    ↑块内槽位   ↑一行 = 一个 token 的状态

一行（STATE_WIDTH = (1+OVERLAP)*HEAD_SIZE = 256）:
┌──────────┬──────────┬────────────┬────────────┐
│ kv_a 128 │ kv_b 128 │ score_a 128│ score_b 128│
└──────────┴──────────┴────────────┴────────────┘
 0          128        256          384        512
 └── kv 状态 (STATE_WIDTH) ──┘└── score 状态 (STATE_WIDTH) ──┘
```

`state_cache` 是 vLLM 的 **PagedAttention 式分页存储**：一个请求中第 `pos` 个 token 存在
```
物理块号 = block_table[req_idx, pos // block_size]
块内偏移 = pos % block_size
```

### 2.3 这个 kernel 的任务

对 batch 中的每个 token：**如果它刚好是一组的最后一个**（position = 3, 7, 11, …），就从 `state_cache` 中取出它前面 8 个 token 的 kv/score，做 softmax 池化，输出一行 128 维的 bf16 向量。其他 token 什么都不做。

调用处（`rocm.py:1771`）之后紧接着调用 aiter 的 `rocm_mxfp4_indexer_k_store`，完成 RMSNorm、RoPE、MXFP4 量化和写入 KV cache。docstring 里的 "stopping before the norm" 指的就是：**原本 CUDA 路径上是一个大融合 kernel，ROCm 路径把它拆成"Triton 做池化"和"aiter 做 norm/rope/量化/写 cache"两步**，因为 aiter 那种预先 shuffle 过的 MXFP4 页面布局只有 aiter 自己的算子能正确写入。

---

## 第三部分：逐行精读

### 完整代码

```python
@triton.jit
def _compress_pool_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    kv_slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    out_ptr,
    out_stride,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    """The softmax-gated pooling of `_fused_kv_compress_norm_rope_insert_*`,
    stopping before the norm: one bf16 row per boundary token."""
    token_idx = tl.program_id(0)
    # Pool exactly the rows aiter's writer stores; it skips the same tokens.
    if tl.load(kv_slot_mapping_ptr + token_idx) < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + pos // block_size,
        mask=mask_pos,
        other=0,
    )
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + (pos % block_size) * state_cache_stride1
        + head_offset
    )

    block = tl.arange(0, HEAD_SIZE)
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=mask_pos[:, None],
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base[:, None] + block[None, :], mask=mask_pos[:, None], other=0.0)
    compressed_kv = tl.sum(kv * score, axis=0)
    tl.store(
        out_ptr + token_idx.to(tl.int64) * out_stride + block,
        compressed_kv.to(tl.bfloat16),
    )
```

```python
@triton.jit
def _compress_pool_kernel(
```
声明这是一个 Triton kernel。名字带下划线，表示它是模块内部函数。

### 参数列表

```python
    state_cache_ptr,          # 指针：state_cache [num_blocks, block_size, 512] 的首地址
    state_cache_stride0,      # 跨一个 block 要跳多少个元素 (= block_size * 512)
    state_cache_stride1,      # 跨一个槽位(token)要跳多少个元素 (= 512)
    token_to_req_indices_ptr, # [num_tokens]：第 t 个 token 属于哪个请求
    positions_ptr,            # [num_tokens]：第 t 个 token 在它所属序列中的位置
    kv_slot_mapping_ptr,      # [num_tokens]：它在压缩 KV cache 里的槽位，-1 = 无效/padding
    block_table_ptr,          # [num_reqs, max_blocks]：分页表
    block_table_stride,       # block_table 一行的长度
    block_size,               # 每页多少个 token（运行时值）
    out_ptr,                  # 输出 [num_tokens, 128] bf16
    out_stride,               # 输出一行的长度
    HEAD_SIZE: tl.constexpr,      # 128
    STATE_WIDTH: tl.constexpr,    # 256
    COMPRESS_RATIO: tl.constexpr, # 4
    OVERLAP: tl.constexpr,        # 1
):
```
**为什么这样分**：
- 张量只传指针和 stride（概念③④）。
- `HEAD_SIZE` / `COMPRESS_RATIO` / `OVERLAP` 要用作 `tl.arange` 的长度，**必须**是 constexpr。它们是模型结构常量，取值固定，所以只会编译一次。`COMPRESS_RATIO` 做成 constexpr 还有一个好处：`% 4` 可以优化成位运算。
- `block_size` 不需要是常量，保留为运行时参数，换配置时不必重新编译。

### 第 1 步：确定"我是谁"，不该干活就提前退出

```python
    token_idx = tl.program_id(0)
```
启动时 grid 是 `(num_tokens,)`，即**每个 token 一个 program**。`token_idx` 就是"我负责 batch 中的第几个 token"。

```python
    if tl.load(kv_slot_mapping_ptr + token_idx) < 0:
        return
```
读一个标量。slot 为 -1 表示 padding token：做 CUDA Graph / HIP Graph 时 batch 会补齐到固定大小，补出来的就是这种无效 token。原代码的注释是 "Pool exactly the rows aiter's writer stores"，意思是后面那个 aiter 写 cache 的算子也会跳过这些 token，两边保持一致，就不会算无用功。

```python
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
```
只有每组最后一个 token（position = 3, 7, 11, …）才需要输出压缩结果。在配套脚本的测试中，请求 0 是 0..12 共 13 个 token，只有下标 3、7、11 会写输出，其余 10 个 program 在这里直接退出。

> **设计要点：early exit**。每个 program 先用最便宜的标量读取判断自己要不要干活。ratio=4 时 3/4 的 program 在这里就结束了，几乎不占资源。这种"按 token 启动，大多数提前返回"的写法很常见，因为 grid 大小只依赖 `num_tokens`，**不依赖数据内容**，所以可以被 CUDA Graph 捕获（graph 里的 grid 是固定的）。

```python
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
```
查出这个 token 属于哪个请求，后面查分页表要用。

### 第 2 步：确定窗口里的 8 个 token

```python
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
```
用 position=11 举例：
```
start  = 11 - 8 + 1 = 4
tokens = [0, 1, 2, 3, 4, 5, 6, 7]        ← 窗口内的相对序号，块形状 [8]
pos    = [4, 5, 6, 7, 8, 9,10,11]        ← 序列中的绝对位置
         └ 前一组(第1组) ┘└ 当前组(第2组) ┘
```
`(1+OVERLAP)*COMPRESS_RATIO = 8` 由 constexpr 算出，是 2 的幂，满足 `tl.arange` 的要求（概念⑤）。

`mask_pos`：当 position=3 时 `pos = [-4..3]`，前 4 个是负数，表示**序列开头之前没有"前一组"**。这些位置不能访问内存，用 mask 标出来。

> 注意，从这里开始，变量就从**标量**变成了**形状为 [8] 的块**。8 个 token 的地址计算是并行完成的，没有写 for 循环。这就是 Triton 的思维方式。

### 第 3 步：查分页表，算出每个 token 那一行的起始地址

```python
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + pos // block_size,
        mask=mask_pos,
        other=0,
    )
```
- `block_table_ptr + req_idx * block_table_stride` 是该请求那一行分页表的起始位置（概念④）。
- `+ pos // block_size` 是一个 [8] 的偏移块，所以整体是一个 **[8] 的指针块**，一次 load 出 8 个物理块号（**gather**）。
- `mask=mask_pos, other=0`：负 pos 不读内存，填 0。这里填 0 只是为了让后面的地址计算不出问题，真正读数据时还会再 mask 一次，所以不会读到错误数据。

```python
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
```
这是 **overlap 的关键一行**：
```
tokens       = [0, 1, 2, 3, 4,   5,   6,   7]
tokens >= 4  = [F, F, F, F, T,   T,   T,   T]
head_offset  = [0, 0, 0, 0, 128, 128, 128, 128]
```
前一组的 token 读每行的**前半个 head**（`kv_a` / `score_a`），当前组的 token 读**后半个 head**（`kv_b` / `score_b`）。对照 2.2 的内存图，一个 token 产生的两份 head 正好分别服务于"作为前一组"和"作为当前组"两种情况。

> 这里用 **算术代替分支**：布尔值转成 0/1 再乘以 128，没有写 if/else，因为块级代码里每个元素走的分支不同，用 if 不好写。也可以写成 `tl.where(tokens >= COMPRESS_RATIO, HEAD_SIZE, 0)`，效果一样。

```python
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + (pos % block_size) * state_cache_stride1
        + head_offset
    )
```
这就是概念④的三维寻址：`state_cache[物理块号, 块内偏移, head_offset]`，结果是 **8 个指针组成的块**，每个指向对应 token 那一行（再加上 head 偏移）的开头。

**为什么要 `.to(tl.int64)`？** 这是写 kernel 时最常见的坑之一：**int32 溢出**。`block_numbers` 是 int32，`stride0 = block_size * 512`。物理块号一大，乘积就会超过 2³¹ ≈ 21 亿，变成负数，导致越界访问。所以**凡是"大索引 × stride"的地方，都先转成 int64 再乘**。其它几项（块内偏移 × 512）数值很小，不需要转。

### 第 4 步：读出一个 [8, 128] 的 score tile，做 softmax

```python
    block = tl.arange(0, HEAD_SIZE)           # [0..127]，通道下标
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=mask_pos[:, None],
        other=float("-inf"),
    )
```
形状推导（概念⑦）：
```
row_base[:, None]   : [8, 1]     每个 token 一行的起点
block[None, :]      : [1, 128]   通道偏移
相加                : [8, 128]   二维指针网格
+ STATE_WIDTH(256)  :            跳过 kv 部分，指向 score 部分
```
一次 load 就读出 8 个 token × 128 个通道的 score。每一行的 128 个元素在内存中连续，所以编译器可以生成高效的向量化加载。

`mask=mask_pos[:, None]` 也是广播：形状 [8,1] 自动扩展成 [8,128]，即"整行要么都读，要么都不读"。

**为什么 `other=float("-inf")`？** 因为下一步是 softmax：`exp(-inf) = 0`，所以无效 token 的权重**正好是 0**，相当于不参与池化。如果填 0，`exp(0)=1`，就会把一个不存在的 token 算进去，结果就错了。这就是概念⑥里说的"other 的值要和后面的运算匹配"。
（会不会整列都是 -inf，从而出现 0/0 = NaN？不会：一个 token 能走到这里时 position ≥ 3，所以当前组的 4 个 token 一定有效。）

```python
    score = tl.softmax(score, dim=0)
```
沿 `dim=0`（8 个 token 的方向）做 softmax，**每个通道独立**归一化。结果形状仍是 [8,128]，每一列的和为 1。`tl.softmax` 内部会先减去最大值，保证数值稳定。

### 第 5 步：读 kv，加权求和

```python
    kv = tl.load(row_base[:, None] + block[None, :], mask=mask_pos[:, None], other=0.0)
```
和读 score 一样，只是不加 `STATE_WIDTH`，读的是每行前面的 kv 部分。这里 `other=0.0`，因为后面是乘法和求和，填 0 最安全。（其实这些位置的权重已经是 0 了，但如果 other 是 NaN 或 inf，`0 * inf = NaN` 仍会出问题，所以填 0 更稳妥。）

```python
    compressed_kv = tl.sum(kv * score, axis=0)
```
`kv * score` 是 [8,128] 的逐元素乘法，`sum(axis=0)` 把 8 个 token 加起来，得到 [128]。这就是 2.1 节的公式。

> **这里体现了融合的价值**：softmax、乘法、求和全部在寄存器里完成，中间结果从不写回显存。整个计算只读了 8×512 个 fp32、写了 128 个 bf16。

### 第 6 步：写回

```python
    tl.store(
        out_ptr + token_idx.to(tl.int64) * out_stride + block,
        compressed_kv.to(tl.bfloat16),
    )
```
- 地址是 `out[token_idx, 0:128]`。这里同样转成 int64 防溢出，属于稳妥写法，养成习惯就好。
- `.to(tl.bfloat16)`：前面用 fp32 计算保证精度，存的时候转成 bf16 以节省带宽，后面 aiter 的算子要求输入是 bf16。
- **没有 mask**：`HEAD_SIZE=128` 正好是 2 的幂，`block` 不会越界，所以不需要 mask。

### 调用侧（`rocm.py:1771`）

```python
_compress_pool_kernel[(num_tokens,)](
    state_cache, state_cache.stride(0), state_cache.stride(1),
    state_metadata.token_to_req_indices, positions, kv_slot_mapping,
    state_metadata.block_table, state_metadata.block_table.stride(0),
    state_metadata.block_size,
    pooled_k, pooled_k.stride(0),
    HEAD_SIZE=self.head_dim, STATE_WIDTH=state_width,
    COMPRESS_RATIO=self.compress_ratio, OVERLAP=self.overlap,
    num_warps=1,
)
```
- `[(num_tokens,)]`：一维 grid，每个 token 一个 program。
- `num_warps=1`：每个 program 只处理 8×128 = 1024 个元素，一个 64 线程的 wavefront 每个线程负责 16 个元素就够了。warp 多了反而增加跨 warp 归约（softmax 和 sum 都是归约）的同步开销，而且 3/4 的 program 一开始就会退出，给太多线程是浪费。
- 输出 buffer `_pooled_k_scratch` 用了 `functools.cache`，所有 indexer 层共用一块显存，避免每层都重新分配。

---

## 第四部分：从这个 kernel 学到的设计原则

**和原版融合 kernel 对比**（`fused_compress_quant_cache.py:713`）：原版用的是 `TRITON_BLOCK_SIZE = next_power_of_2(head_size)`，再加 `mask = block < HEAD_SIZE`，以支持非 2 的幂的 head_dim。ROCm 版在 Python 侧 `assert self.head_dim == 128`，所以直接用 `tl.arange(0, HEAD_SIZE)`，把这层 mask 去掉了。这是**用前置条件换取代码简洁**的做法。

总结成一张清单，以后写 kernel 可以对照：

| 原则 | 本例中的体现 |
|---|---|
| 先想好"一个 program 负责什么" | 一个 token 产出一行输出 |
| 便宜的判断放最前面，尽早返回 | slot < 0、非组边界就 return |
| grid 不依赖数据，方便 Graph 捕获 | grid 恒为 `(num_tokens,)` |
| 用块代替循环 | 8 个 token 的地址一次算完，一次 gather |
| 用 `[:, None]` 和 `[None, :]` 构造二维 tile | `[8,128]` 的 score/kv |
| mask 的 `other` 要匹配后续运算 | softmax 用 `-inf`，求和用 `0` |
| 大索引 × stride 转 int64 | `block_numbers.to(tl.int64)` |
| 用 fp32 计算、用低精度存储 | 最后 `.to(tl.bfloat16)` |
| `num_warps` 和数据量匹配 | 只有 1024 个元素，用 1 个 warp |
| 形状参数用 constexpr，可变参数用运行时值 | `HEAD_SIZE` 与 `block_size` |

---

## 第五部分：动手练习

先运行配套脚本：
```bash
python /home/shashen/vllm/vllm-dev-skills/triton/triton_learn_compress_pool.py
# 输出：被写入输出的 token 下标: [3, 7, 11, 13]
#       与 PyTorch 参考实现一致 ✅
```
其中下标 13 是请求 1 那个 position=23 的 decode token，下标 14 是 padding（slot=-1），被跳过了。脚本里的 `reference()` 用 Python for 循环实现了相同的逻辑，可以和 kernel 逐行对照，理解"块化"到底省掉了哪些循环。

然后试着改一改，每改一处就预测会发生什么，再跑一遍看是否和预测一致：
1. 把 score 的 `other=float("-inf")` 改成 `0.0`，看 token 3（position=3）的结果是否和参考实现对不上，并想清楚原因。
2. 去掉 `head_offset`，看结果怎么变，加深对 overlap 布局的理解。
3. 把 `num_warps=1` 改成 4 或 8，用 `triton.testing.do_bench` 测一下耗时。
4. 进阶：设 `HEAD=96`（不是 2 的幂），按原版 kernel 的写法加上 `TRITON_BLOCK_SIZE` 和 `mask`，让它也能运行。
5. 进阶：在 kernel 里用 `tl.device_print("pos", pos)` 打印块的内容，或者设置环境变量 `TRITON_INTERPRET=1` 用 CPU 解释模式执行，这样可以在 kernel 中打断点、打印中间值，调试很方便。

如果想继续深入，推荐按这个顺序读 Triton 官方教程：01-vector-add → 02-fused-softmax → 03-matrix-multiplication。第 2 篇讲的正是本例用到的"按行 softmax"，第 3 篇会引入 `tl.dot` 和分块循环，那是写 GEMM 和 attention 类算子的基础。
