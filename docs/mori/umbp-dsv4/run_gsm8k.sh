#!/usr/bin/env bash
# GSM8K against the running server (reuses dev/acc.sh). Run once with UMBP=1 and
# once against a UMBP=0 server; scores should match within noise.
set -euo pipefail
export HF_DATASETS_CACHE=/shared/models/huggingface/datasets
export HF_DATASETS_OFFLINE=1
LIMIT=${LIMIT:-200} bash /home/shashen/vllm/vllm/dev/acc.sh
