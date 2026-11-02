#!/usr/bin/env bash
# Mooncake master for single-node embedded mode: the vLLM ranks contribute
# the DRAM pool (global_segment_size each), the master only keeps metadata.
set -euo pipefail
LOG_DIR=${LOG_DIR:-/tmp/umbp_runs/mooncake}
mkdir -p "${LOG_DIR}"
exec /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master \
  --port 50051 > "${LOG_DIR}/mooncake_master.log" 2>&1
