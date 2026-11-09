#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 2x MI355X with the InferenceX recipe
# dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml (base + override_tp2_c64): TP2,
# DSpark MTP (5 tokens), monokernel, mxfp4 indexer, engram tables in host memory.
# UMBP=1 adds the standalone UMBP connector; UMBP=0 is the baseline with
# otherwise identical flags. SYNTHETIC_AL=3.51 swaps block rejection for the
# golden acceptance length, as InferenceX throughput runs do
# (infx/golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, 5 drafts).
set -euo pipefail
VLLM_BIN=/home/shashen/vllm/vllm/.venv/bin/vllm
MODEL=${MODEL:-/shared/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6}
PORT=${PORT:-8333}
UMBP=${UMBP:-1}
ENDPOINT=${ENDPOINT:-unix:///tmp/umbp/standalone.grpc.sock}
# Extra kv_connector_extra_config entries, e.g. UMBP_EXTRA=',"lazy_offload":true'
UMBP_EXTRA=${UMBP_EXTRA:-}
# DP=2: attention DP2 with the MoE TP-sharded across both ranks (no expert
# parallelism). The mono decode kernels need TP 2/4/8 without DP and fail on
# an explicit opt-in, so DP>1 drops --monokernel.
DP=${DP:-1}
SYNTHETIC_AL=${SYNTHETIC_AL:-}
# main's extensible KV cache (#56492, on by default on ROCm) backs the KV cache
# with VMM pages, which hipIpcGetMemHandle rejects, so the standalone UMBP
# server cannot map it. Off for every arm, baseline included, to keep KV sizing
# equal; EXTENSIBLE_KV_CACHE=1 restores the default.
EXTENSIBLE_KV_CACHE=${EXTENSIBLE_KV_CACHE:-0}
LOG_NAME=${LOG_NAME:-vllm_dsv41_umbp${UMBP}}
LOG_DIR=${LOG_DIR:-/tmp/umbp_runs}
mkdir -p "${LOG_DIR}"

# roles.agg.env
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# INT4 needs an aiter with QuickAllReduceInt4.compile() on main since #59478;
# override (e.g. INT8) on older aiter builds.
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${QR_QUANT:-INT4}
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1}
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
# The container exports HIP_VISIBLE_DEVICES for all 8 GPUs; pick ours explicitly.
export HIP_VISIBLE_DEVICES=${GPUS:-6,7}
unset PYTHONPATH VIRTUAL_ENV
cd /tmp

REJECTION='"rejection_sample_method":"block"'
if [[ -n "${SYNTHETIC_AL}" ]]; then
  REJECTION="\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${SYNTHETIC_AL}"
fi

EXTRA=()
if [[ "${DP}" -gt 1 ]]; then
  EXTRA+=(--tensor-parallel-size 1 --data-parallel-size "${DP}")
else
  EXTRA+=(--tensor-parallel-size 2 --monokernel)
fi
if [[ "${EXTENSIBLE_KV_CACHE}" != "1" ]]; then
  EXTRA+=(--no-enable-extensible-kv-cache)
fi
if [[ "${UMBP}" == "1" ]]; then
  EXTRA+=(--kv-transfer-config "{\"kv_connector\":\"UMBPStoreConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"mode\":\"standalone\",\"endpoint\":\"${ENDPOINT}\"${UMBP_EXTRA}}}")
fi

exec "${VLLM_BIN}" serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --moe-backend aiter \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",${REJECTION},\"enable_adaptive_verification\":false}" \
  --max-model-len 1048576 \
  --disable-uvicorn-access-log \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 1024 \
  --max-num-batched-tokens 16384 \
  "${EXTRA[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_NAME}.log"
