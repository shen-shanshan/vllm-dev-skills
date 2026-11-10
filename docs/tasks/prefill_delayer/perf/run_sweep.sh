#!/usr/bin/env bash
# Tuning points of dp2_delayer_tuning_plan.md, serially. Each argument is one
# point: "TAG VAR=value ...", e.g.
#   bash run_sweep.sh "f0_base FAST=1 POLICY=cadence INTERVAL=1" \
#                     "f1_n11_w5000 FAST=1 POLICY=delayer INTERVAL=11 MAX_WAIT_MS=5000"
# Stops at the first failed point. Results are copied next to this script after
# every point, since /tmp does not survive a container restart.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export LOG_DIR=${LOG_DIR:-/tmp/prefill_delayer_tuning}
BACKUP=${HERE}/tuning/results
mkdir -p "${BACKUP}"
for point in "$@"; do
  read -r -a words <<< "${point}"
  tag=${words[0]}
  echo "$(date +%F' '%T) start ${tag}: ${words[*]:1}"
  env TAG="${tag}" "${words[@]:1}" bash "${HERE}/run_config.sh"
  echo "$(date +%F' '%T) end ${tag}"
  mkdir -p "${BACKUP}/${tag}"
  (cd "${LOG_DIR}/${tag}" && cp ./*.json engine_metrics*.txt client.log "${BACKUP}/${tag}/" 2>/dev/null) || true
  # Summary JSONs as is, the (large) per-request records compressed.
  mkdir -p "${BACKUP}/${tag}/aiperf_artifacts"
  for f in "${LOG_DIR}/${tag}"/aiperf_artifacts/*; do
    [[ -f "${f}" ]] || continue
    case "${f}" in
      # Skip the per-scrape server metrics and timeslices (~200 MB a point).
      */server_metrics_export.json|*_timeslices.json) ;;
      *.json) cp "${f}" "${BACKUP}/${tag}/aiperf_artifacts/" ;;
      *.jsonl) gzip -c "${f}" > "${BACKUP}/${tag}/aiperf_artifacts/$(basename "${f}").gz" ;;
    esac
  done
  gzip -c "${LOG_DIR}/${tag}/engine.log" > "${BACKUP}/${tag}/engine.log.gz" || true
done
