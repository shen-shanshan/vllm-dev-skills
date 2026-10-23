# 学习用：独立运行 _compress_pool_kernel，并用纯 PyTorch 参考实现校验结果。
# 运行：python triton_learn_compress_pool.py
import torch
import triton
import triton.language as tl


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
    token_idx = tl.program_id(0)
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


def reference(state_cache, token_to_req, positions, slot_mapping, block_table,
              block_size, head, state_width, ratio, overlap, out):
    """逐 token 的 Python 版本，语义与 kernel 完全一致。"""
    win = (1 + overlap) * ratio
    for t in range(positions.numel()):
        p = int(positions[t])
        if int(slot_mapping[t]) < 0 or (p + 1) % ratio != 0:
            continue
        r = int(token_to_req[t])
        kvs, scores = [], []
        for i in range(win):
            q = p - win + 1 + i
            if q < 0:
                continue
            blk = int(block_table[r, q // block_size])
            row = state_cache[blk, q % block_size]
            h = head if i >= ratio else 0
            kvs.append(row[h:h + head])
            scores.append(row[state_width + h:state_width + h + head])
        kv, sc = torch.stack(kvs), torch.stack(scores)
        out[t] = (kv * sc.softmax(dim=0)).sum(0).to(torch.bfloat16)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    HEAD, RATIO, OVERLAP = 128, 4, 1           # DeepSeek V4 indexer 的配置
    STATE_WIDTH = (1 + OVERLAP) * HEAD          # 256
    block_size, num_blocks, num_reqs = 16, 32, 2

    # state_cache: [num_blocks, block_size, 2*STATE_WIDTH] = [kv_a|kv_b|score_a|score_b]
    state_cache = torch.randn(num_blocks, block_size, 2 * STATE_WIDTH,
                              device=dev, dtype=torch.float32)
    block_table = torch.randperm(num_blocks, device=dev, dtype=torch.int32).view(num_reqs, -1)

    # 一个 batch：请求 0 是 prefill 了 13 个 token，请求 1 是一个 decode token(pos=23)，再加一个 padding
    positions = torch.tensor(list(range(13)) + [23, 0], device=dev, dtype=torch.int64)
    token_to_req = torch.tensor([0] * 13 + [1, 0], device=dev, dtype=torch.int32)
    slot_mapping = torch.tensor([0] * 14 + [-1], device=dev, dtype=torch.int64)
    num_tokens = positions.numel()

    out = torch.zeros(num_tokens, HEAD, device=dev, dtype=torch.bfloat16)
    _compress_pool_kernel[(num_tokens,)](
        state_cache, state_cache.stride(0), state_cache.stride(1),
        token_to_req, positions, slot_mapping,
        block_table, block_table.stride(0), block_size,
        out, out.stride(0),
        HEAD_SIZE=HEAD, STATE_WIDTH=STATE_WIDTH,
        COMPRESS_RATIO=RATIO, OVERLAP=OVERLAP,
        num_warps=1,
    )

    ref = torch.zeros_like(out)
    reference(state_cache, token_to_req, positions, slot_mapping, block_table,
              block_size, HEAD, STATE_WIDTH, RATIO, OVERLAP, ref)

    written = (out != 0).any(dim=1).nonzero().flatten().tolist()
    print("被写入输出的 token 下标:", written)  # 期望 [3, 7, 11, 13]
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("与 PyTorch 参考实现一致 ✅")


if __name__ == "__main__":
    main()
