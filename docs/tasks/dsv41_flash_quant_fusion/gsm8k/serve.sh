#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 2x MI355X, DEP2 (attention DP2 + EP2, AITER MegaMoE).
# Same serving args as the prefill-trace run
# (vllm-profile-traces/dsv41_flash/dep2_8k1k_c256_prefill/serve.sh), without
# the profiler.
set -euo pipefail

MODEL=${MODEL:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
VLLM_BIN=${VLLM_BIN:-/home/shashen/vllm/vllm/.venv/bin/vllm}
PORT=${PORT:-8000}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5}
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --port "$PORT" \
  --tokenizer-mode deepseek_v41 \
  --language-model-only \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --moe-backend aiter_mega_moe \
  --all2all-backend mori \
  --engram-config '{"cpu_offload":true}' \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-cudagraph-capture-size 1024 \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
