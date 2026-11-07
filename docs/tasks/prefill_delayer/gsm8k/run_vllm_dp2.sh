#!/usr/bin/env bash
# DeepSeek-V4.1-Flash, attention DP2 (MoE TP-sharded across both ranks), DSpark
# MTP, on 2x MI355X. POLICY=cadence (default flags) or POLICY=delayer.
#   POLICY=delayer INTERVAL=10 MAX_WAIT_MS=5000 bash run_vllm_dp2.sh
set -euo pipefail
VLLM_BIN=/home/shashen/vllm/vllm/.venv/bin/vllm
MODEL=${MODEL:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
PORT=${PORT:-8333}
POLICY=${POLICY:-cadence}
INTERVAL=${INTERVAL:-1}
MAX_WAIT_MS=${MAX_WAIT_MS:-}
# DSpark runs only on Model Runner V2, and V4.1 ships no classic-MTP draft, so
# with VLLM_USE_V2_MODEL_RUNNER=0 set SPEC_CONFIG=none (no speculative decoding).
SPEC_CONFIG=${SPEC_CONFIG:-'{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'}
LOG_NAME=${LOG_NAME:-vllm_dp2_${POLICY}}
LOG_DIR=${LOG_DIR:-/tmp/prefill_delayer_runs}
mkdir -p "${LOG_DIR}"

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
# The delayer's decision counters are logged at debug level.
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-DEBUG}
# The container exports HIP_VISIBLE_DEVICES for all 8 GPUs; pick ours explicitly.
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
unset PYTHONPATH VIRTUAL_ENV
cd /tmp

EXTRA=(--prefill-schedule-policy "${POLICY}" --prefill-schedule-interval "${INTERVAL}")
if [[ "${SPEC_CONFIG}" != "none" ]]; then
  EXTRA+=(--speculative-config "${SPEC_CONFIG}")
fi
if [[ -n "${MAX_WAIT_MS}" ]]; then
  EXTRA+=(--prefill-max-wait-ms "${MAX_WAIT_MS}")
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
  --max-model-len 1048576 \
  --disable-uvicorn-access-log \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 1024 \
  --max-num-batched-tokens 16384 \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  "${EXTRA[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_NAME}.log"
