#!/usr/bin/env bash
# The four points of perf_plan.md, serially on GPUs 0,1. P3's max wait is P0's
# TTFT p90 rounded to 500 ms.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=${LOG_DIR:-/tmp/prefill_delayer_perf}
# Stop at the first failed point instead of cascading into the next.
run() { echo "$(date +%F' '%T) start $1"; TAG=$1 "${@:2}" bash "${HERE}/run_config.sh"; echo "$(date +%F' '%T) end $1"; }
run base env POLICY=cadence INTERVAL=1
run cad4 env POLICY=cadence INTERVAL=4
run del4 env POLICY=delayer INTERVAL=4
WAIT_MS=$(python3 -c "
import json, glob
a = json.load(open(glob.glob('${LOG_DIR}/base/dsv41flash_vllm_*_base.json')[0]))
print(int(round(a['request_metrics']['latency']['ttft']['p90'] * 1000 / 500) * 500))")
echo "P3 max wait = ${WAIT_MS} ms"
run del4_wait env POLICY=delayer INTERVAL=4 MAX_WAIT_MS="${WAIT_MS}"
