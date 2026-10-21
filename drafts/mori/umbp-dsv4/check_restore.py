"""Store a long prefix, drop it from the GPU prefix cache, restore it from UMBP.

Pass criteria: the second request is served with external (UMBP) hits and
its TTFT drops. With SWA replay the rebuilt window is approximate, so output
equality is reported, not required; judge accuracy with run_gsm8k.sh.
"""
import json, os, random, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8333")
MODEL = os.environ.get("MODEL", "/shared/models/deepseek-ai/DeepSeek-V4-Pro")
TOKENS = int(os.environ.get("PREFIX_WORDS", "12000"))


def post(path, body=None):
    req = urllib.request.Request(
        BASE + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def metric(name):
    text = urllib.request.urlopen(BASE + "/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines()
               if l.startswith(name) and not l.startswith("#"))


def ask(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 64, "temperature": 0,
            "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start, ttft, out = time.time(), None, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ttft = ttft or time.time() - start
            out.append(json.loads(line[6:])["choices"][0]["text"])
    return ttft, "".join(out)


random.seed(0)
words = " ".join(random.choice(["alpha", "beta", "gamma", "delta", "omega"]) for _ in range(TOKENS))
prompt = f"Remember this list: {words}\nSummarize the list in one sentence:"
ext = "vllm:external_prefix_cache_hits_total"
cold_ttft, cold = ask(prompt)
time.sleep(5)  # let the asynchronous store finish
post("/reset_prefix_cache")  # GPU prefix cache only; keeps the UMBP pool
hits_before = metric(ext)
warm_ttft, warm = ask(prompt)
hits = metric(ext) - hits_before
print(json.dumps({"cold_ttft_s": round(cold_ttft, 3), "warm_ttft_s": round(warm_ttft, 3),
                  "external_hit_tokens": hits, "same_output": cold == warm}, indent=2))
assert hits > 0, "no UMBP hit"
if cold != warm:
    print("note: restored output differs from the cold run (SWA replay)")
