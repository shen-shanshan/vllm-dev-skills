#!/usr/bin/env bash
# Sum the engine's cumulative counters across DP ranks (includes warmup).
curl -s "http://localhost:${ENGINE_PORT:-8333}/metrics" | grep -E "^vllm:(spec_decode_num_(accepted|draft)_tokens_total|spec_decode_num_drafts_total|prefix_cache_(hits|queries)_total|external_prefix_cache_(hits|queries)_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" \
  | awk '{split($1,a,"{"); s[a[1]]+=$2} END{for(k in s) printf "%s %.0f\n", k, s[k]}' | sort
