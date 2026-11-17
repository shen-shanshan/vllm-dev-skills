#!/usr/bin/env bash
# AgentX c256 A/B for the MXFP8 input-quant fusions (#2, #3, #4) on DSv4.1-Flash
# DEP2 (attention DP2 + EP2, AITER MegaMoE) with prefill cadence 11 and
# per-tensor fp8 KV, in InferenceX agentx-fast mode (1200 s). No KV offloading:
# UMBP standalone stalled the engine for minutes in this config (run 1, see
# perf-plan.md), so UMBP=1 is opt-in only.
#
#   base : vLLM 60bd6dd2ed (no fusion)
#   fused: rocm-dsv41-flash-kernel-fusion HEAD (#2/#3 98de3d8aff, #4 2532459f72)
#
# The vLLM checkout is switched in place (no worktree): the fused files are
# backed up first and restored, and the restore is verified against the
# original `git diff HEAD`.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM=/home/shashen/vllm/vllm
UMBP_SCRIPTS=/home/shashen/vllm/vllm-dev-skills/docs/mori/umbp-dsv41-flash
BASE=60bd6dd2ed
FILES=(
  vllm/_aiter_ops.py
  vllm/models/deepseek_v4/amd/model.py
  vllm/models/deepseek_v41/amd/mhc.py
  vllm/models/deepseek_v41/amd/model.py
  vllm/models/deepseek_v41/amd/rocm.py
)
BACKUP=${HERE}/code_backup
STEPS=${STEPS:-base fused}

export RUNS_DIR=${RUNS_DIR:-/tmp/quant_fusion_runs}
export GPUS=${GPUS:-0,1} DP=2 CONC=256
export SYNTHETIC_AL=3.51 AIPERF_EXPERIMENTAL_FAST=1
export PREFILL_SCHEDULE_INTERVAL=11 QR_QUANT=NONE EXTENSIBLE_KV_CACHE=1
export KV_CACHE_DTYPE=fp8_e4m3
export MOE_ARGS="--enable-expert-parallel --moe-backend aiter_mega_moe --all2all-backend mori"
export UMBP=${UMBP:-0} UMBP_DRAM_CAPACITY=449000000000
SUFFIX=$([[ "${UMBP}" == "1" ]] && echo "" || echo "_noumbp")
mkdir -p "${RUNS_DIR}" "${BACKUP}"

cd "${VLLM}"
git diff HEAD > "${BACKUP}/fused_vs_head.diff"
for f in "${FILES[@]}"; do
  mkdir -p "${BACKUP}/$(dirname "$f")"
  cp "$f" "${BACKUP}/$f"
done

restore_fused() {
  cd "${VLLM}"
  git checkout -q HEAD -- "${FILES[@]}"
  for f in "${FILES[@]}"; do cp "${BACKUP}/$f" "$f"; done
  git diff HEAD | cmp -s - "${BACKUP}/fused_vs_head.diff" \
    || { echo "restore mismatch" >&2; exit 1; }
}
trap restore_fused EXIT

run_point() {
  local tag=$1
  echo "$(date) start ${tag} ($(cd "${VLLM}" && git diff ${BASE} --stat -- "${FILES[@]}" | tail -1))"
  TAG=${tag} bash "${UMBP_SCRIPTS}/run_config.sh"
  echo "$(date) done ${tag}"
}

for step in ${STEPS}; do
  case ${step} in
    base)
      git checkout -q "${BASE}" -- "${FILES[@]}"
      run_point fast_base_dep2_c256${SUFFIX}
      restore_fused
      ;;
    fused)
      restore_fused
      run_point fast_fused_dep2_c256${SUFFIX}
      ;;
  esac
done
