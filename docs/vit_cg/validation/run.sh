#!/usr/bin/env bash
# run.sh <qwen3_vl|dsv41|step3_vl|gemma4|deepseek_ocr> <tag> [eager_vit]
# Writes out/<model>_<tag>.{json,log} next to this script.
set -u
which=$1; tag=$2; mode=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM_PY=${VLLM_PY:-/home/shashen/vllm/vllm/.venv/bin/python}
mkdir -p "$HERE/out"
cd /tmp
if [[ "$which" == dsv41 ]]; then
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
    AITER_TRITON_LOG_LEVEL=ERROR VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True OMP_NUM_THREADS=1
  export HIP_VISIBLE_DEVICES=${GPUS:-6,7}
else
  export HIP_VISIBLE_DEVICES=${GPUS:-0}
fi
unset PYTHONPATH VIRTUAL_ENV
# collective_rpc ships the harness's callables to the workers.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
timeout 5400 "$VLLM_PY" -I "$HERE/harness.py" "$which" \
  "$HERE/out/${which}_${tag}.json" $mode > "$HERE/out/${which}_${tag}.log" 2>&1
echo "exit=$?"
