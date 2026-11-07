#!/usr/bin/env bash
# GSM8K 5-shot on a fresh DP2 deployment: engine -> lm_eval -> teardown.
#   TAG=cadence POLICY=cadence bash run_gsm8k_config.sh
#   TAG=delayer POLICY=delayer INTERVAL=10 MAX_WAIT_MS=5000 bash run_gsm8k_config.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
export LOG_DIR=${LOG_DIR:-/tmp/prefill_delayer_runs}
OUT=${LOG_DIR}/gsm8k/${TAG}
mkdir -p "${OUT}"
LOG_NAME=gsm8k_${TAG} setsid nohup bash "${HERE}/run_vllm_dp2.sh" >/dev/null 2>&1 &
ENGINE_PGID=$!
# Kill the engine's whole process group (setsid), never by cmdline pattern.
trap 'kill -- -"${ENGINE_PGID}" 2>/dev/null || true; sleep 30' EXIT
until curl -sf localhost:8333/health >/dev/null; do
  kill -0 "${ENGINE_PGID}" 2>/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done
OUT="${OUT}" bash "${HERE}/../../../mori/umbp-dsv41-flash/run_gsm8k.sh" >/dev/null
cp "${LOG_DIR}/gsm8k_${TAG}.log" "${OUT}/engine.log"
echo "done ${TAG}"
