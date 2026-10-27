#!/usr/bin/env bash
# AgentX trace replay (InferenceX `python3 -m infx.bench agentic`) at the
# recipe's override_dep8_c128 point, through the router, scraping engine metrics.
#   TAG=baseline DURATION=600 bash run_agentic.sh
# The AgentX runtime (agentx-harness==1.0.12) lives in a persistent venv so it
# is not re-bootstrapped per run; the trace corpus is cached under /shared.
set -eo pipefail
INFX=/home/shashen/vllm/InferenceX/inferencex-e2e
RUNTIME=/home/shashen/vllm/agentx_runtime
TAG=${TAG:?set TAG, e.g. baseline / umbp_replay}
OUT=${OUT:-/home/shashen/vllm/umbp_runs/agentic/${TAG}}
mkdir -p "${OUT}"

# Matrix-point markers first (InferenceX AGENTS rule), then workflow defaults.
export IS_AGENTIC=1 SCENARIO_TYPE=agentic-coding
export KV_OFFLOADING=${KV_OFFLOADING:-none}
source "${INFX}/benchmarks/runtime_settings.sh"
# Local run: no power telemetry, single node, full (non-fast) replay.
export ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 IS_MULTINODE=false EVAL_ONLY=false
export AIPERF_EXPERIMENTAL_FAST=0

# Recipe: base + override_dep8_c128 benchmark.env
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:'
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
export CONC=128
export DURATION=${DURATION:-600}

export MODEL=deepseek-ai/DeepSeek-V4-Pro-0813
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro-0813
export TOKENIZER=/shared/DeepSeek-V4-Pro-0813
export MODEL_PREFIX=dsv4 FRAMEWORK=vllm PRECISION=fp4
export AIPERF_SERVER_URL=http://localhost:${ROUTER_PORT:-30000}
export AIPERF_SERVER_METRICS_URLS=http://localhost:${ENGINE_PORT:-8333}/metrics
export RESULT_DIR="${OUT}" AGENTIC_OUTPUT_DIR="${OUT}"
export RESULT_FILENAME=dsv4_vllm_dep8_c${CONC}_${TAG}
export HF_HOME=/shared/huggingface HF_HUB_CACHE=/shared/huggingface/hub
export AIPERF_RUNTIME_DIR="${RUNTIME}"

cd "${INFX}"
PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${INFX}" \
  "${RUNTIME}/venv/bin/python" -m infx.bench agentic 2>&1 | tee "${OUT}/client.log"
