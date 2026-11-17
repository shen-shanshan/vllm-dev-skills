#!/usr/bin/env bash
# Repeat prefill8k -> decode128 on the current vLLM checkout until the
# decode128 graph replay hangs (seen 3/5 on the fused code), then record the
# in-flight GPU dispatches of both ranks with rocgdb and stop.
#   TRIES=4 bash hang_catch.sh <tag>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${1:?tag}
TRIES=${TRIES:-4}
OUT=/tmp/fwd/hang_${TAG}
ROCGDB=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin/rocgdb
mkdir -p "${OUT}"

rank_pids() {
  ps -eo pid,args | grep "cli.main bench forward" | grep -v "grep\|distributed.run" | awk '{print $1}'
}

for i in $(seq 1 "${TRIES}"); do
  log=${OUT}/try${i}.log
  OUT=${OUT}/try${i}.json bash "${HERE}/run_forward.sh" --case prefill8k=8192 \
    --case "decode128=128*6@16384:5" --routing balanced > "${log}" 2>&1 &
  launcher=$!
  until grep -q "\[forward\] prefill8k" "${log}" 2>/dev/null; do
    kill -0 ${launcher} 2>/dev/null || break
    sleep 10
  done
  waited=0
  while kill -0 ${launcher} 2>/dev/null && ! grep -q "\[forward\] decode128" "${log}"; do
    sleep 10; waited=$((waited + 10))
    [[ ${waited} -ge 150 ]] && break
  done
  if grep -q "\[forward\] decode128" "${log}"; then
    echo "try${i}: ok"; wait ${launcher}; continue
  fi
  echo "try${i}: hang suspected after ${waited}s"
  rocm-smi --showuse > "${OUT}/gpu_use.txt" 2>&1
  for p in $(rank_pids); do
    timeout 300 "${ROCGDB}" -p "${p}" -batch \
      -ex "info agents" -ex "info queues" -ex "info dispatches" \
      > "${OUT}/rocgdb_${p}.txt" 2>&1
    ~/.local/bin/py-spy dump --pid "${p}" > "${OUT}/pyspy_${p}.txt" 2>&1
  done
  for p in $(rank_pids); do kill "${p}"; done
  kill ${launcher} 2>/dev/null; wait ${launcher} 2>/dev/null
  exit 0
done
echo "no hang in ${TRIES} tries"
