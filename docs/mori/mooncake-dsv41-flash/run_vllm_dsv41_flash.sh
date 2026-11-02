#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 2x MI355X with the InferenceX recipe
# dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml (base + override_tp2_c64): TP2,
# DSpark MTP (5 tokens), monokernel, mxfp4 indexer, engram tables in host memory.
# MOONCAKE=1 adds MooncakeStoreConnector (embedded mode, mooncake_config.json);
# MOONCAKE=0 is the baseline with otherwise identical flags.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM_BIN=/home/shashen/vllm/vllm/.venv/bin/vllm
MODEL=${MODEL:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
PORT=${PORT:-8333}
MOONCAKE=${MOONCAKE:-1}
# Opt in to keeping the ROCm aiter sparse MLA / NaN-free fast paths.
CANONICAL=${CANONICAL:-true}
LOG_NAME=${LOG_NAME:-vllm_dsv41_mooncake${MOONCAKE}}
LOG_DIR=${LOG_DIR:-/tmp/umbp_runs/mooncake}
mkdir -p "${LOG_DIR}"

# roles.agg.env
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
# The container exports HIP_VISIBLE_DEVICES for all 8 GPUs; pick ours explicitly.
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
export MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-${HERE}/mooncake_config.json}
unset PYTHONPATH VIRTUAL_ENV
cd /tmp

EXTRA=()
if [[ "${MOONCAKE}" == "1" ]]; then
  EXTRA+=(--kv-transfer-config "{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"only_restores_canonical_kv\":${CANONICAL}}}")
fi

exec "${VLLM_BIN}" serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --moe-backend aiter \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --max-model-len 1048576 \
  --monokernel \
  --disable-uvicorn-access-log \
  --tensor-parallel-size 2 \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 1024 \
  --max-num-batched-tokens 16384 \
  "${EXTRA[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_NAME}.log"
