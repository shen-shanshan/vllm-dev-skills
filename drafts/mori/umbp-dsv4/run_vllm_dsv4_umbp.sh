#!/usr/bin/env bash
# DSV4-Pro TP8 on MI355X with prefix caching; UMBP=1 attaches the standalone
# connector, UMBP=0 is the baseline (same flags, no connector).
set -euo pipefail
VLLM_BIN=/home/shashen/vllm/vllm/.venv/bin/vllm
MODEL=${MODEL:-/shared/models/deepseek-ai/DeepSeek-V4-Pro}
PORT=${PORT:-8333}
UMBP=${UMBP:-1}
ENDPOINT=${ENDPOINT:-/tmp/umbp/standalone.grpc.sock}
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
unset PYTHONPATH VIRTUAL_ENV
cd /tmp
EXTRA=()
if [[ "${UMBP}" == "1" ]]; then
  EXTRA+=(--kv-transfer-config "{\"kv_connector\":\"UMBPStoreConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"mode\":\"standalone\",\"endpoint\":\"${ENDPOINT}\",\"dedup_tp_replicated_kv\":${DEDUP:-true}}}")
fi
exec "${VLLM_BIN}" serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --distributed-executor-backend mp \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --moe-backend aiter \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  "${EXTRA[@]}" 2>&1 | tee "/tmp/umbp/vllm_umbp${UMBP}.log"
