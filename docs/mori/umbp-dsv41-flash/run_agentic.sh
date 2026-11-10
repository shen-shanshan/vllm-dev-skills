#!/usr/bin/env bash
# AgentX trace replay (InferenceX `python3 -m infx.bench agentic`) against the
# DeepSeek-V4.1-Flash TP2 engine: the override_tp2_c64 engine args driven at
# CONC=128, straight to the engine (the recipe's frontend is the engine itself).
#   TAG=b0_c128 bash run_agentic.sh
set -eo pipefail
INFX=/home/shashen/vllm/InferenceX/inferencex-e2e
RUNTIME=/home/shashen/vllm/agentx_runtime
TAG=${TAG:?set TAG, e.g. b0_c128}
OUT=${OUT:-${RUNS_DIR:-/tmp/umbp_runs/agentic_dsv41}/${TAG}}
mkdir -p "${OUT}"

# Matrix-point markers first (InferenceX AGENTS rule), then workflow defaults.
export IS_AGENTIC=1 SCENARIO_TYPE=agentic-coding
export KV_OFFLOADING=${KV_OFFLOADING:-none}
source "${INFX}/benchmarks/runtime_settings.sh"
# Local run: no power telemetry, single node, full (non-fast) replay.
export ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 IS_MULTINODE=false EVAL_ONLY=false
# FAST=1 runs the InferenceX agentx-fast replay (1200 s, one warmup request per lane).
export AIPERF_EXPERIMENTAL_FAST=${AIPERF_EXPERIMENTAL_FAST:-0}

# Recipe base benchmark.env.
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:'
export CONC=${CONC:-128}
# Matrix shape (amd-master.yaml dsv41flash-fp4-mi355x-vllm-agentic-dspark:
# tp 2, kv-offloading none, spec-decoding mtp).
export TP=2 EP_SIZE=1 DP_ATTENTION=false SPEC_DECODING=mtp
CLIENT_PORT=${ENGINE_PORT:-8333}
if [[ "${DP:-1}" -gt 1 ]]; then
  # Attention DP behind the session-affine router: keep each session on one rank.
  export DP_ATTENTION=true AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
  CLIENT_PORT=${ROUTER_PORT:-30000}
fi
# InferenceX matrix default (DEFAULT_AGENTIC_DURATION_SECONDS).
export DURATION=${DURATION:-3600}

export MODEL=deepseek-ai/DeepSeek-V4.1-Flash
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash
export TOKENIZER=${TOKENIZER:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
export MODEL_PREFIX=dsv41flash FRAMEWORK=vllm PRECISION=fp4
export AIPERF_SERVER_URL=http://localhost:${CLIENT_PORT}
export AIPERF_SERVER_METRICS_URLS=http://localhost:${ENGINE_PORT:-8333}/metrics
export RESULT_DIR="${OUT}" AGENTIC_OUTPUT_DIR="${OUT}"
export RESULT_FILENAME=dsv41flash_vllm_${DP_LABEL:-tp2}_c${CONC}_${TAG}
export HF_HOME=${HF_HOME:-/home/shashen/vllm/hf_cache}
export AIPERF_RUNTIME_DIR="${RUNTIME}"

cd "${INFX}"
PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${INFX}" \
  "${RUNTIME}/venv/bin/python" -m infx.bench agentic 2>&1 | tee "${OUT}/client.log"
