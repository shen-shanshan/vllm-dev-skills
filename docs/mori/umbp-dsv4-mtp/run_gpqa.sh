#!/usr/bin/env bash
# GPQA Diamond CoT zero-shot (lm_eval 0.4.13, chat completions, thinking on,
# greedy) against the running server. lm_eval lives in its own venv so vLLM's
# .venv is not touched. The gated dataset is cached once in HF_HOME (needs an
# HF login that accepted the GPQA terms), then read offline.
#   OUT=/path/to/result_dir bash run_gpqa.sh
set -euo pipefail
export HF_HOME=${HF_HOME:-/home/shashen/vllm/hf_cache}
export HF_DATASETS_OFFLINE=1
PORT=${PORT:-8333}
# 32K truncates many reasoning traces before the answer; see
# docs/tasks/dsv4_mxfp4_indexer_gpqa_results.md.
MAX_GEN=${MAX_GEN:-131072}
CONC=${CONC:-64}
OUT=${OUT:-/tmp/umbp_runs/gpqa/manual}
mkdir -p "${OUT}"
cd "${OUT}"
/home/shashen/vllm/lm_eval_venv/bin/lm_eval \
  --model local-chat-completions \
  --apply_chat_template \
  --model_args "model=deepseek-ai/DeepSeek-V4-Pro-0813,base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=${CONC},max_retries=3,timeout=10800,tokenized_requests=False" \
  --tasks gpqa_diamond_cot_zeroshot \
  --gen_kwargs "max_gen_toks=${MAX_GEN},temperature=0" \
  --output_path . \
  --log_samples 2>&1 | tee -a eval.log
