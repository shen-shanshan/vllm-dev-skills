#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 2x MI355X: attention DP2, MoE TP-sharded across both
# ranks (no EP, no --monokernel: mono decode rejects DP), recipe base args
# (dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml), DSpark 5 with the golden
# acceptance length, rust frontend. Only the prefill scheduling flags vary.
#   POLICY=delayer INTERVAL=4 MAX_WAIT_MS=3000 bash run_vllm_dp2.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM_BIN=/home/shashen/vllm/vllm/.venv/bin/vllm
MODEL=${MODEL:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
PORT=${PORT:-8333}
POLICY=${POLICY:-cadence}
INTERVAL=${INTERVAL:-1}
MAX_WAIT_MS=${MAX_WAIT_MS:-}
CHUNK=${CHUNK:-16384}
SYNTHETIC_AL=${SYNTHETIC_AL:-3.51}
LOG_NAME=${LOG_NAME:-vllm_dp2_${POLICY}${INTERVAL}}
LOG_DIR=${LOG_DIR:-/tmp/prefill_delayer_perf}
mkdir -p "${LOG_DIR}"

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1}
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
# INFO everywhere, DEBUG only for the delayer's decision counters.
export VLLM_LOGGING_CONFIG_PATH=${HERE}/logging_delayer_debug.json
# The container exports HIP_VISIBLE_DEVICES for all 8 GPUs; pick ours explicitly.
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
unset PYTHONPATH VIRTUAL_ENV VLLM_LOGGING_LEVEL
cd /tmp

EXTRA=(--prefill-schedule-policy "${POLICY}" --prefill-schedule-interval "${INTERVAL}")
# Extra engine flags, e.g. EXTRA_ARGS="--no-enable-extensible-kv-cache".
read -r -a EXTRA_ARR <<< "${EXTRA_ARGS:-}"
EXTRA+=("${EXTRA_ARR[@]}")
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
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${SYNTHETIC_AL},\"enable_adaptive_verification\":false}" \
  --max-model-len 1048576 \
  --disable-uvicorn-access-log \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 1024 \
  --max-num-batched-tokens "${CHUNK}" \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  "${EXTRA[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_NAME}.log"
