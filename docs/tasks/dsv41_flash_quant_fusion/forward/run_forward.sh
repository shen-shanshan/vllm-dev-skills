#!/usr/bin/env bash
# `vllm bench forward` for DeepSeek-V4.1-Flash DEP2 (attention DP2 + EP2, AITER
# MegaMoE) with the AgentX serving flags, dummy weights. Times one forward step
# per case as a graph replay (no scheduler, router or runtime JIT stalls).
#   OUT=/tmp/x.json bash run_forward.sh --case prefill=16384 ...
set -euo pipefail
VLLM=/home/shashen/vllm/vllm
MODEL=${MODEL:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
OUT=${OUT:?output json}

export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
unset PYTHONPATH VIRTUAL_ENV

cd /tmp
exec "${VLLM}/.venv/bin/python" -m torch.distributed.run --nproc-per-node=2 -m vllm.entrypoints.cli.main \
  bench forward \
  --model "${MODEL}" \
  --tokenizer-mode deepseek_v41 \
  --language-model-only \
  -dp 2 --enable-expert-parallel \
  --moe-backend aiter_mega_moe --all2all-backend mori \
  --engram-config '{"cpu_offload":true}' \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len ${MAX_MODEL_LEN:-65536} \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}' \
  --output-json "${OUT}" \
  "$@"
