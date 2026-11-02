#!/usr/bin/env bash
# One agentic point end to end on a fresh deployment: [UMBP server] -> engine ->
# AgentX replay -> engine metric snapshot -> teardown.
#   TAG=b0_c128 UMBP=0 bash run_config.sh
#   TAG=s2_decode_c128 UMBP=1 UMBP_EXTRA=',"save_decode_cache":true' bash run_config.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
UMBP=${UMBP:-0}
# Keep logs off the nearly full NFS home: a quota hit truncates AIPerf records.
export RUNS_DIR=${RUNS_DIR:-/tmp/umbp_runs/agentic_dsv41}
export LOG_DIR=${RUNS_DIR}
export UMBP_DRAM_CAPACITY=${UMBP_DRAM_CAPACITY:-$((128 * 1024**3))}
export GPUS=${GPUS:-0,1}
OUT=${RUNS_DIR}/${TAG}
mkdir -p "${OUT}"
cleanup() {
  pkill -f "bin/vllm serve /shared" || true
  pkill -x umbp_standalone || true
  pkill -f "mori/umbp_standalone_server" || true
  sleep 30
}
cleanup
# Tear down on any exit, including a failed startup.
trap cleanup EXIT
if [[ "${UMBP}" == "1" ]]; then
  rm -f /tmp/umbp/standalone.grpc.sock /tmp/umbp/standalone.fd.sock
  nohup bash "${HERE}/../umbp-dsv4/run_umbp_server.sh" >/dev/null 2>&1 &
  # The socket opens once the pool is faulted in and registered.
  until grep -q "host memory registered for GPU access" "${RUNS_DIR}/umbp_server.log" 2>/dev/null \
      && [ -S /tmp/umbp/standalone.grpc.sock ]; do
    sleep 5
  done
  cp "${RUNS_DIR}/umbp_server.log" "${OUT}/umbp_server_start.log" || true
fi
UMBP=${UMBP} LOG_NAME=agentic_${TAG} nohup bash "${HERE}/run_vllm_dsv41_flash.sh" >/dev/null 2>&1 &
# Watch the launcher's PID: pgrep on the vllm cmdline races its exec.
ENGINE_PID=$!
until curl -sf localhost:8333/health >/dev/null; do
  kill -0 "${ENGINE_PID}" 2>/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done
bash "${HERE}/../umbp-dsv4-mtp/snapshot_metrics.sh" > "${OUT}/engine_metrics_start.txt"
if [[ "${UMBP}" == "1" ]]; then
  # InferenceX aggregation requires backend metadata whenever KV offload is on.
  export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=umbp
  export KV_OFFLOAD_BACKEND_METADATA='{"name":"umbp","version":"mori-1.2.3.post1"}'
  export TOTAL_CPU_DRAM_GB=$(( UMBP_DRAM_CAPACITY / 1024**3 ))
fi
TAG=${TAG} OUT=${OUT} bash "${HERE}/run_agentic.sh" >/dev/null 2>&1 || echo "client rc=$?" >&2
bash "${HERE}/../umbp-dsv4-mtp/snapshot_metrics.sh" > "${OUT}/engine_metrics.txt"
cp "${RUNS_DIR}/agentic_${TAG}.log" "${OUT}/engine.log" 2>/dev/null || true
[[ "${UMBP}" == "1" ]] && cp "${RUNS_DIR}/umbp_server.log" "${OUT}/umbp_server.log" || true
echo "done ${TAG}"
