#!/usr/bin/env bash
# GSM8K 5-shot (lm_eval local-completions) against the running server.
# lm_eval lives in its own venv so vLLM's .venv is not touched; the dataset is
# cached under /shared/huggingface for everyone.
#   OUT=/path/to/result_dir bash run_gsm8k.sh
set -euo pipefail
export HF_HOME=/shared/huggingface
export HF_DATASETS_CACHE=/shared/huggingface/datasets
export HF_DATASETS_OFFLINE=1
PORT=${PORT:-8333}
OUT=${OUT:-/home/shashen/vllm/umbp_runs/gsm8k}
mkdir -p "${OUT}"
cd "${OUT}"
/home/shashen/vllm/lm_eval_venv/bin/lm_eval \
  --model local-completions \
  --model_args "model=deepseek-ai/DeepSeek-V4-Pro-0813,base_url=http://0.0.0.0:${PORT}/v1/completions,num_concurrent=64,max_retries=10,max_gen_toks=2048,timeout=60000,tokenized_requests=False,tokenizer_backend=None" \
  --batch_size auto \
  --tasks gsm8k \
  --num_fewshot 5 \
  --output_path . 2>&1 | tee -a eval.log
