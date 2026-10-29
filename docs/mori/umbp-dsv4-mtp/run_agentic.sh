#!/usr/bin/env bash
# AgentX trace replay (InferenceX `python3 -m infx.bench agentic`) at the
# recipe's override_dep8_c128 point, through the router, scraping engine metrics.
#   TAG=baseline bash run_agentic.sh
# The AgentX runtime (agentx-harness==1.0.12) lives in a persistent venv so it
# is not re-bootstrapped per run. The trace corpus (1.8 GB) is cached in a
# persistent HF_HOME; /shared is a read-only mount on some hosts.
set -eo pipefail
INFX=/home/shashen/vllm/InferenceX/inferencex-e2e
RUNTIME=/home/shashen/vllm/agentx_runtime
TAG=${TAG:?set TAG, e.g. baseline / umbp_replay}
OUT=${OUT:-${RUNS_DIR:-/tmp/umbp_runs}/agentic/${TAG}}
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
# Matrix shape of this point (amd-master.yaml: tp 8, ep 1, dp-attn true,
# spec-decoding draft_model); the aggregation derives per-GPU numbers from it.
export TP=8 EP_SIZE=1 DP_ATTENTION=true SPEC_DECODING=draft_model
# InferenceX matrix default (infx/matrix/validation.py:
# DEFAULT_AGENTIC_DURATION_SECONDS); infx.bench itself requires DURATION.
export DURATION=${DURATION:-3600}

export MODEL=deepseek-ai/DeepSeek-V4-Pro-0813
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro-0813
export TOKENIZER=${TOKENIZER:-/shared/models--deepseek-ai--DeepSeek-V4-Pro-DSpark/snapshots/7c09739fd136abfb70a49ec334157f65f45b52cd}
export MODEL_PREFIX=dsv4 FRAMEWORK=vllm PRECISION=fp4
export AIPERF_SERVER_URL=http://localhost:${ROUTER_PORT:-30000}
export AIPERF_SERVER_METRICS_URLS=http://localhost:${ENGINE_PORT:-8333}/metrics
export RESULT_DIR="${OUT}" AGENTIC_OUTPUT_DIR="${OUT}"
export RESULT_FILENAME=dsv4_vllm_dep8_c${CONC}_${TAG}
export HF_HOME=${HF_HOME:-/home/shashen/vllm/hf_cache}
export AIPERF_RUNTIME_DIR="${RUNTIME}"

cd "${INFX}"
PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${INFX}" \
  "${RUNTIME}/venv/bin/python" -m infx.bench agentic 2>&1 | tee "${OUT}/client.log"
