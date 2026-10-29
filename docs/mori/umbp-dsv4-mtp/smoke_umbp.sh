#!/usr/bin/env bash
# Minutes-long smoke test of a UMBP connector config before a full AgentX run:
# [UMBP server] -> engine -> a two-turn chat whose second turn reuses the first
# turn's prompt and answer -> GPU prefix cache dropped -> second turn again,
# now restored from the UMBP pool. Checks the restore path ran (external hits
# grew) and compares the restored answer with the GPU-served one (greedy;
# a mismatch is reported, not failed, since MTP numerics may differ).
#   TAG=u3_smoke UMBP_EXTRA=',"save_decode_cache":true' bash smoke_umbp.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
export RUNS_DIR=${RUNS_DIR:-/tmp/umbp_runs}
export LOG_DIR=${RUNS_DIR}
export UMBP_DRAM_CAPACITY=${UMBP_DRAM_CAPACITY:-$((128 * 1024**3))}
OUT=${RUNS_DIR}/smoke/${TAG}
mkdir -p "${OUT}"
cleanup() {
  pkill -f "bin/vllm serve /shared" || true
  pkill -f "mori/umbp_standalone_server" || true
  sleep 30
}
cleanup
trap cleanup EXIT
rm -f /tmp/umbp/standalone.grpc.sock /tmp/umbp/standalone.fd.sock
nohup bash "${HERE}/../umbp-dsv4/run_umbp_server.sh" >/dev/null 2>&1 &
until [ -S /tmp/umbp/standalone.grpc.sock ]; do sleep 5; done
# /reset_prefix_cache is a dev endpoint.
export VLLM_SERVER_DEV_MODE=1
UMBP=1 LOG_NAME=smoke_${TAG} nohup bash "${HERE}/run_vllm_dsv4_mtp.sh" >/dev/null 2>&1 &
ENGINE_PID=$!
until curl -sf localhost:8333/health >/dev/null; do
  kill -0 "${ENGINE_PID}" 2>/dev/null || { echo "engine died" >&2; exit 1; }
  sleep 10
done

python3 - "${OUT}" <<'EOF'
import json, sys, time, urllib.request

out = sys.argv[1]
base = "http://localhost:8333"
model = "deepseek-ai/DeepSeek-V4-Pro-0813"


def post(path, body=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read() or b"{}")


def chat(messages):
    # Greedy and no thinking keep the answer deterministic and short. DP ranks
    # share one UMBP key namespace, so any rank can restore another's blocks.
    r = post("/v1/chat/completions", {
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"thinking": False},
    })
    return r["choices"][0]["message"]["content"]


def external_hits():
    with urllib.request.urlopen(base + "/metrics") as r:
        return sum(
            float(line.split()[-1])
            for line in r.read().decode().splitlines()
            if line.startswith("vllm:external_prefix_cache_hits_total")
        )


# ~20K-token prompt: many 256-token blocks, and past one SWA window.
doc = "\n".join(
    f"Record {i}: sensor {i % 97} reported value {(i * 7919) % 10007} "
    f"at step {i}; status {'ok' if i % 13 else 'fault'}."
    for i in range(1200)
)
turn1 = [{"role": "user", "content": doc + "\n\nList the first five records "
          "whose status is fault, with their values."}]
a1 = chat(turn1)
turn2 = turn1 + [
    {"role": "assistant", "content": a1},
    {"role": "user", "content": "Now give the sensor ids of those records."},
]
gpu_served = chat(turn2)
time.sleep(5)  # let the stores finish
# Reset fails while blocks are still referenced; retry until it lands.
for _ in range(60):
    if post("/reset_prefix_cache").get("success", True) is not False:
        break
    time.sleep(5)
else:
    sys.exit("FAIL: /reset_prefix_cache never succeeded")
hits_before = external_hits()
restored = chat(turn2)
hits = external_hits() - hits_before
result = {
    "turn1": a1, "turn2_gpu_served": gpu_served, "turn2_restored": restored,
    "external_hit_tokens": hits, "match": gpu_served == restored,
}
json.dump(result, open(f"{out}/smoke.json", "w"), indent=2)
print(json.dumps({k: result[k] for k in ("external_hit_tokens", "match")}))
print("restored:", restored[:300])
if hits <= 0:
    sys.exit("FAIL: no external prefix hits; the restore path did not run")
EOF
cp "${RUNS_DIR}/smoke_${TAG}.log" "${OUT}/engine.log" 2>/dev/null || true
cp "${RUNS_DIR}/umbp_server.log" "${OUT}/umbp_server.log" 2>/dev/null || true
grep -i -E "nan|Traceback|ERROR" "${OUT}/engine.log" | head -5 || true
echo "done ${TAG}"
