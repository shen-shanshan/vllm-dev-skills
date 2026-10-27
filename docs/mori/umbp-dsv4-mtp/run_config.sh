#!/usr/bin/env bash
# One agentic point end to end on a fresh deployment: [UMBP server] -> engine ->
# router -> AgentX replay -> engine metric snapshot -> teardown.
#   TAG=umbp_replay UMBP=1 bash run_config.sh
#   TAG=umbp_store_all UMBP=1 UMBP_EXTRA=',"rebuild_sliding_window":false' bash run_config.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
UMBP=${UMBP:-0}
OUT=/home/shashen/vllm/umbp_runs/agentic/${TAG}
mkdir -p "${OUT}"
cleanup() {
  pkill -f "vllm_router_venv/bin/vllm-router" || true
  pkill -f "bin/vllm serve /shared" || true
  pkill -f "mori/umbp_standalone_server" || true
  sleep 30
}
cleanup
if [[ "${UMBP}" == "1" ]]; then
  rm -f /tmp/umbp/standalone.grpc.sock /tmp/umbp/standalone.fd.sock
  nohup bash "${HERE}/../umbp-dsv4/run_umbp_server.sh" >/dev/null 2>&1 &
  until [ -S /tmp/umbp/standalone.grpc.sock ]; do sleep 5; done
  cp /home/shashen/vllm/umbp_runs/umbp_server.log "${OUT}/umbp_server_start.log" || true
fi
UMBP=${UMBP} LOG_NAME=agentic_${TAG} nohup bash "${HERE}/run_vllm_dsv4_mtp.sh" >/dev/null 2>&1 &
until curl -sf localhost:8333/health >/dev/null; do
  pgrep -f "bin/vllm serve /shared" >/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done
nohup bash "${HERE}/run_router.sh" >/dev/null 2>&1 &
until curl -sf localhost:30000/health >/dev/null; do sleep 5; done
if [[ "${UMBP}" == "1" ]]; then
  export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=umbp
  export TOTAL_CPU_DRAM_GB=$(( ${UMBP_DRAM_CAPACITY:-$((128 * 1024**3))} / 1024**3 ))
fi
TAG=${TAG} bash "${HERE}/run_agentic.sh" >/dev/null 2>&1 || echo "client rc=$?" >&2
bash "${HERE}/snapshot_metrics.sh" > "${OUT}/engine_metrics.txt"
cp "/home/shashen/vllm/umbp_runs/agentic_${TAG}.log" "${OUT}/engine.log" 2>/dev/null || true
[[ "${UMBP}" == "1" ]] && cp /home/shashen/vllm/umbp_runs/umbp_server.log "${OUT}/umbp_server.log" || true
cleanup
echo "done ${TAG}"
