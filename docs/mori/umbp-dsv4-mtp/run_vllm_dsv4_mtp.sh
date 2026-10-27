#!/usr/bin/env bash
# DeepSeek-V4-Pro-0813 on 8x MI355X with the InferenceX recipe
# dsv4/vllm/mi355x-fp4-mtp/agentic.yaml (base + override_dep8_c128): DP8 x TP1,
# DSpark MTP (6 tokens), prefix caching. UMBP=1 adds the standalone UMBP
# connector; UMBP=0 is the baseline with otherwise identical flags.
# The recipe's vllm-router frontend (consistent_hash on X-Session-ID) runs
# separately: run_router.sh. Accuracy runs may skip it and use the engine's
# built-in DP load balancer.
set -euo pipefail
VLLM_BIN=/home/shashen/vllm/vllm/.venv/bin/vllm
MODEL=${MODEL:-/shared/DeepSeek-V4-Pro-0813}
PORT=${PORT:-8333}
UMBP=${UMBP:-1}
ENDPOINT=${ENDPOINT:-unix:///tmp/umbp/standalone.grpc.sock}
# Extra kv_connector_extra_config entries, e.g. UMBP_EXTRA=',"rebuild_sliding_window":false'
UMBP_EXTRA=${UMBP_EXTRA:-}
LOG_NAME=${LOG_NAME:-vllm_mtp_umbp${UMBP}}
LOG_DIR=${LOG_DIR:-/home/shashen/vllm/umbp_runs}
mkdir -p "${LOG_DIR}"

# roles.agg.env
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
unset PYTHONPATH VIRTUAL_ENV
cd /tmp

EXTRA=()
if [[ "${UMBP}" == "1" ]]; then
  EXTRA+=(--kv-transfer-config "{\"kv_connector\":\"UMBPStoreConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"mode\":\"standalone\",\"endpoint\":\"${ENDPOINT}\"${UMBP_EXTRA}}}")
fi

exec "${VLLM_BIN}" serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro-0813 \
  --load-format "${LOAD_FORMAT:-fastsafetensors}" \
  --trust-remote-code \
  --async-scheduling \
  --distributed-executor-backend mp \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --gpu-memory-utilization 0.86 \
  --moe-backend aiter \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 6, "draft_sample_method": "probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --prefill-schedule-interval 8 \
  --long-prefill-token-threshold 16384 \
  --max-num-seqs 128 \
  "${EXTRA[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_NAME}.log"
