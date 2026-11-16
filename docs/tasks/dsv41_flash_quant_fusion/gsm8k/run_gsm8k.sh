#!/usr/bin/env bash
# GSM8K with the settings of vLLM's lm-eval config for DeepSeek-V4-Flash
# (.buildkite/lm-eval-harness/configs/DeepSeek-V4-Flash-MXFP4.yaml):
# 8-shot, all 1319 questions, strict-match and flexible-extract.
# Usage: run_gsm8k.sh <output_dir>
set -euo pipefail

OUT=${1:?output dir}
PORT=${PORT:-8000}
LM_EVAL=${LM_EVAL:-/home/shashen/vllm/lm_eval_venv/bin/lm_eval}

"$LM_EVAL" \
  --model local-completions \
  --model_args "model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://localhost:$PORT/v1/completions,tokenizer_backend=None,num_concurrent=64,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k \
  --num_fewshot 8 \
  --limit 1319 \
  --batch_size 1 \
  --output_path "$OUT"
