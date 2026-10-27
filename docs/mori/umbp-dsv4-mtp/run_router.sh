#!/usr/bin/env bash
# The recipe's frontend for DEP8: vllm-router==0.1.14 (own venv) in front of the
# DP8 engine, routing each session to one DP rank by X-Session-ID.
set -euo pipefail
ENGINE_PORT=${ENGINE_PORT:-8333}
ROUTER_PORT=${ROUTER_PORT:-30000}
LOG_DIR=${LOG_DIR:-/home/shashen/vllm/umbp_runs}
mkdir -p "${LOG_DIR}"
exec /home/shashen/vllm/vllm_router_venv/bin/vllm-router \
  --worker-urls "http://localhost:${ENGINE_PORT}" \
  --intra-node-data-parallel-size 8 \
  --host 0.0.0.0 --port "${ROUTER_PORT}" \
  --policy consistent_hash \
  --request-timeout-secs 14400 \
  --disable-retries 2>&1 | tee "${LOG_DIR}/router.log"
