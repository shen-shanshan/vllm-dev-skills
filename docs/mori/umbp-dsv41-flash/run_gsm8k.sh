#!/usr/bin/env bash
# GSM8K 5-shot (lm_eval local-completions) against the running server.
# lm_eval lives in its own venv so vLLM's .venv is not touched.
#   OUT=/path/to/result_dir bash run_gsm8k.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/umbp_runs/hf_datasets}
PORT=${PORT:-8333}
OUT=${OUT:-/tmp/umbp_runs/gsm8k}
mkdir -p "${OUT}"
cd "${OUT}"
/home/shashen/vllm/lm_eval_venv/bin/lm_eval \
  --model local-completions \
  --model_args "model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://0.0.0.0:${PORT}/v1/completions,num_concurrent=64,max_retries=10,max_gen_toks=2048,timeout=60000,tokenized_requests=False,tokenizer_backend=None" \
  --batch_size auto \
  --include_path "${HERE}/gsm8k_task" \
  --tasks gsm8k_local \
  --num_fewshot 5 \
  --output_path . 2>&1 | tee -a eval.log
