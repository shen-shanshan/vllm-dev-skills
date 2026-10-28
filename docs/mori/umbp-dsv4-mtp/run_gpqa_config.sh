#!/usr/bin/env bash
# GPQA accuracy on a fresh deployment of the override_dep8_c128 engine:
# [UMBP server] -> engine -> PASSES x GPQA -> engine metric snapshots -> teardown.
# Pass 2+ first drops the GPU prefix cache (UMBP keeps its pool), so repeated
# prompts can only hit the UMBP pool and exercise restore + SWA replay.
#   TAG=baseline UMBP=0 bash run_gpqa_config.sh
#   TAG=umbp UMBP=1 PASSES=2 bash run_gpqa_config.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
UMBP=${UMBP:-0}
PASSES=${PASSES:-1}
export RUNS_DIR=${RUNS_DIR:-/tmp/umbp_runs}
export LOG_DIR=${RUNS_DIR}
export UMBP_DRAM_CAPACITY=${UMBP_DRAM_CAPACITY:-$((512 * 1024**3))}
OUT=${RUNS_DIR}/gpqa/${TAG}
mkdir -p "${OUT}"
cleanup() {
  pkill -f "bin/vllm serve /shared" || true
  pkill -f "mori/umbp_standalone_server" || true
  sleep 30
}
cleanup
# Tear down on any exit, including a failed startup.
trap cleanup EXIT
if [[ "${UMBP}" == "1" ]]; then
  rm -f /tmp/umbp/standalone.grpc.sock /tmp/umbp/standalone.fd.sock
  nohup bash "${HERE}/../umbp-dsv4/run_umbp_server.sh" >/dev/null 2>&1 &
  until [ -S /tmp/umbp/standalone.grpc.sock ]; do sleep 5; done
fi
# /reset_prefix_cache is a dev endpoint; set for both runs to keep flags equal.
export VLLM_SERVER_DEV_MODE=1
UMBP=${UMBP} LOG_NAME=gpqa_${TAG} nohup bash "${HERE}/run_vllm_dsv4_mtp.sh" >/dev/null 2>&1 &
ENGINE_PID=$!
until curl -sf localhost:8333/health >/dev/null; do
  kill -0 "${ENGINE_PID}" 2>/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done
for pass in $(seq 1 "${PASSES}"); do
  if (( pass > 1 )); then
    until curl -sf -X POST localhost:8333/reset_prefix_cache | grep -q '"success":true'; do
      sleep 10
    done
  fi
  OUT="${OUT}/pass${pass}" bash "${HERE}/run_gpqa.sh" >/dev/null 2>&1 || echo "pass${pass} rc=$?" >&2
  bash "${HERE}/snapshot_metrics.sh" > "${OUT}/engine_metrics_after_pass${pass}.txt"
done
cp "${RUNS_DIR}/gpqa_${TAG}.log" "${OUT}/engine.log" 2>/dev/null || true
[[ "${UMBP}" == "1" ]] && cp "${RUNS_DIR}/umbp_server.log" "${OUT}/umbp_server.log" || true
echo "done ${TAG}"
