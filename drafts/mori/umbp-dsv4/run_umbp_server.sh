#!/usr/bin/env bash
# Node-local UMBP standalone server for the DSV4-Pro e2e check.
# Needs a mori build with UMBP (umbp_standalone_server + libmori_umbp_pybinds).
set -euo pipefail
SOCK_DIR=${SOCK_DIR:-/tmp/umbp}
mkdir -p "${SOCK_DIR}"
MORI_PKG=$(/home/shashen/vllm/vllm/.venv/bin/python -c 'import mori, os; print(os.path.dirname(mori.__file__))')
SERVER_BIN=${UMBP_STANDALONE_BIN:-${MORI_PKG}/umbp_standalone_server}
export LD_LIBRARY_PATH=${MORI_PKG}:${LD_LIBRARY_PATH:-}
export UMBP_STANDALONE_ADDRESS=unix://${SOCK_DIR}/standalone.grpc.sock
export UMBP_DRAM_CAPACITY=${UMBP_DRAM_CAPACITY:-$((1024 * 1024**3))}
export UMBP_DRAM_USE_HUGEPAGES=${UMBP_DRAM_USE_HUGEPAGES:-0}
export UMBP_SSD_ENABLED=0
export MORI_UMBP_LOG_LEVEL=${MORI_UMBP_LOG_LEVEL:-info}
exec "${SERVER_BIN}" "${UMBP_STANDALONE_ADDRESS}" 2>&1 | tee "${SOCK_DIR}/server.log"
