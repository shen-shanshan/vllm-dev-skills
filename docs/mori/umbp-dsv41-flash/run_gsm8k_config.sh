#!/usr/bin/env bash
# GSM8K on a fresh deployment: [UMBP server] -> engine -> lm_eval -> teardown.
# UMBP=0 runs one pass (baseline). UMBP=1 runs two passes: pass 1 stores the
# KV; then the GPU prefix cache is dropped, so pass 2 restores the same prompts
# from the UMBP pool. Engine counters are snapshotted around each pass.
#   TAG=baseline UMBP=0 bash run_gsm8k_config.sh
#   TAG=umbp UMBP=1 bash run_gsm8k_config.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
UMBP=${UMBP:-0}
export RUNS_DIR=${RUNS_DIR:-/tmp/umbp_runs}
export LOG_DIR=${RUNS_DIR}
export UMBP_DRAM_CAPACITY=${UMBP_DRAM_CAPACITY:-$((128 * 1024**3))}
export GPUS=${GPUS:-0,1}
OUT=${RUNS_DIR}/gsm8k/${TAG}
mkdir -p "${OUT}"
cleanup() {
  pkill -f "bin/vllm serve /shared" || true
  pkill -f "mori/umbp_standalone_server" || true
  sleep 30
}
cleanup
trap cleanup EXIT
if [[ "${UMBP}" == "1" ]]; then
  rm -f /tmp/umbp/standalone.grpc.sock /tmp/umbp/standalone.fd.sock
  nohup bash "${HERE}/../umbp-dsv4/run_umbp_server.sh" >/dev/null 2>&1 &
  until [ -S /tmp/umbp/standalone.grpc.sock ]; do sleep 5; done
fi
# /reset_prefix_cache is a dev endpoint; set for both runs to keep flags equal.
export VLLM_SERVER_DEV_MODE=1
UMBP=${UMBP} LOG_NAME=gsm8k_${TAG} nohup bash "${HERE}/run_vllm_dsv41_flash.sh" >/dev/null 2>&1 &
ENGINE_PID=$!
until curl -sf localhost:8333/health >/dev/null; do
  kill -0 "${ENGINE_PID}" 2>/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done
SNAP="${HERE}/../umbp-dsv4-mtp/snapshot_metrics.sh"
PASSES=1
[[ "${UMBP}" == "1" ]] && PASSES=2
for pass in $(seq 1 "${PASSES}"); do
  if [[ "${pass}" -gt 1 ]]; then
    # Reset fails while blocks are still referenced; retry until it lands.
    until curl -sf -X POST localhost:8333/reset_prefix_cache | grep -q '"success":true'; do
      sleep 5
    done
  fi
  bash "${SNAP}" > "${OUT}/metrics_pass${pass}_before.txt"
  OUT="${OUT}/pass${pass}" bash "${HERE}/run_gsm8k.sh" >/dev/null
  bash "${SNAP}" > "${OUT}/metrics_pass${pass}_after.txt"
done
cp "${RUNS_DIR}/gsm8k_${TAG}.log" "${OUT}/engine.log" 2>/dev/null || true
[[ "${UMBP}" == "1" ]] && cp "${RUNS_DIR}/umbp_server.log" "${OUT}/umbp_server.log" || true
echo "done ${TAG}"
