#!/usr/bin/env bash
# `vllm bench forward` A/B for the MXFP8 input-quant fusions (#2, #3, #4) on
# DSv4.1-Flash DEP2, interleaved base/fused ROUNDS times,
# timing only (`--profile-dir` fails on ROCm: "GPU activities don't split into
# 3 replays", then the other rank waits ~30 min). Each run is capped at
# RUN_TIMEOUT. Switches the vLLM checkout in place like ../agentx/run_ab.sh.
#   base : vLLM 60bd6dd2ed (no fusion)
#   fused: rocm-dsv41-flash-kernel-fusion HEAD (#2/#3 98de3d8aff, #4 2532459f72)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM=/home/shashen/vllm/vllm
BASE=60bd6dd2ed
FILES=(
  vllm/_aiter_ops.py
  vllm/models/deepseek_v4/amd/model.py
  vllm/models/deepseek_v41/amd/mhc.py
  vllm/models/deepseek_v41/amd/model.py
  vllm/models/deepseek_v41/amd/rocm.py
)
ROUNDS=${ROUNDS:-2}
OUT=${OUT_DIR:-/tmp/fwd}
CASES=(
  --case prefill16k=16384
  --case prefill8k=8192
  --case "decode128=128*6@16384:5"
  --case "decode16=16*6@16384:5"
  --case "mix=8192,64*6@16384:5"
)
# CASES_ARGS overrides the case list, e.g. CASES_ARGS='--case decode16=16*6@16384:5'.
if [[ -n "${CASES_ARGS:-}" ]]; then read -r -a CASES <<< "${CASES_ARGS}"; fi
mkdir -p "${OUT}"

cd "${VLLM}"
[[ -z "$(git status --short -- "${FILES[@]}")" ]] || { echo "dirty fused files" >&2; exit 1; }
restore() { cd "${VLLM}" && git checkout -q HEAD -- "${FILES[@]}"; }
trap restore EXIT

for round in $(seq 1 "${ROUNDS}"); do
  for arm in base fused; do
    if [[ ${arm} == base ]]; then git checkout -q "${BASE}" -- "${FILES[@]}"; else restore; fi
    tag=${TAG_PREFIX:-}${arm}_r${round}
    echo "$(date) start ${tag} ($(git diff ${BASE} --stat -- "${FILES[@]}" | tail -1))"
    OUT=${OUT}/${tag}.json timeout --kill-after=60 "${RUN_TIMEOUT:-1200}" \
      bash "${HERE}/run_forward.sh" "${CASES[@]}" --routing balanced \
      > "${OUT}/${tag}.log" 2>&1 \
      || echo "${tag} rc=$?"
    grep "^\[forward\]" "${OUT}/${tag}.log" || true
  done
done
restore
echo "$(date) done"
