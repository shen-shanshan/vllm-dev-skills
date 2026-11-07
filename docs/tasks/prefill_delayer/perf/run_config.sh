#!/usr/bin/env bash
# One AgentX point end to end on a fresh deployment: engine -> router ->
# AgentX replay (DURATION=3600) -> engine metric snapshot -> teardown.
#   TAG=base POLICY=cadence INTERVAL=1 bash run_config.sh
#   TAG=del4_wait POLICY=delayer INTERVAL=4 MAX_WAIT_MS=3000 bash run_config.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MORI=${HERE}/../../../mori
TAG=${TAG:?}
export LOG_DIR=${LOG_DIR:-/tmp/prefill_delayer_perf}
export RUNS_DIR=${LOG_DIR}
OUT=${LOG_DIR}/${TAG}
mkdir -p "${OUT}"
ENGINE_PGID= ROUTER_PGID=
kill_group() {
  # A whole process group (setsid), never by cmdline pattern. Hung workers
  # ignore SIGTERM, so SIGKILL whatever is left after a grace period.
  local pgid=$1
  [[ -z "${pgid}" ]] && return
  kill -TERM -- -"${pgid}" 2>/dev/null || true
  for _ in $(seq 30); do pgrep -g "${pgid}" >/dev/null || return 0; sleep 2; done
  kill -KILL -- -"${pgid}" 2>/dev/null || true
  sleep 5
}
cleanup() {
  kill_group "${ROUTER_PGID}"
  kill_group "${ENGINE_PGID}"
  sleep 10
}
trap cleanup EXIT
# Refuse to start on GPUs still holding memory (a leftover engine would OOM us).
for g in $(echo "${GPUS:-0,1}" | tr ',' ' '); do
  used=$(rocm-smi -d "${g}" --showmemuse 2>/dev/null | grep -o "VRAM%): [0-9]*" | grep -o "[0-9]*$")
  [[ "${used:-0}" -gt 2 ]] && { echo "GPU ${g} busy (${used}% VRAM)" >&2; exit 2; }
done
LOG_NAME=engine_${TAG} setsid nohup bash "${HERE}/run_vllm_dp2.sh" >/dev/null 2>&1 &
ENGINE_PGID=$!
until curl -sf localhost:8333/health >/dev/null; do
  kill -0 "${ENGINE_PGID}" 2>/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done
ROUTER_DP_SIZE=2 setsid nohup bash "${MORI}/umbp-dsv4-mtp/run_router.sh" >/dev/null 2>&1 &
ROUTER_PGID=$!
until curl -sf localhost:30000/health >/dev/null; do sleep 5; done
export DP=2 DP_LABEL=dp2
bash "${MORI}/umbp-dsv4-mtp/snapshot_metrics.sh" > "${OUT}/engine_metrics_start.txt"
CLIENT_RC=0
TAG=${TAG} OUT=${OUT} bash "${MORI}/umbp-dsv41-flash/run_agentic.sh" >/dev/null 2>&1 || CLIENT_RC=$?
bash "${MORI}/umbp-dsv4-mtp/snapshot_metrics.sh" > "${OUT}/engine_metrics.txt"
cp "${LOG_DIR}/engine_${TAG}.log" "${OUT}/engine.log" 2>/dev/null || true
cp "${LOG_DIR}/router.log" "${OUT}/router.log" 2>/dev/null || true
if grep -q "EngineCore encountered a fatal error" "${OUT}/engine.log"; then
  echo "engine crashed during ${TAG}" >&2
  exit 3
fi
echo "done ${TAG} client_rc=${CLIENT_RC}"
