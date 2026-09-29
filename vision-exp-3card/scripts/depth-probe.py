#!/usr/bin/env python3
"""Deep-context probe + per-card VRAM sampler. Used for both 0731 (fill baseline) and
Vision-Exp (fill campaign) so the numbers are directly comparable.

Builds a real-text prompt (source files, not repeated boilerplate) of ~TARGET tokens,
measured with the server's own /tokenize endpoint, with a random tag at the very start
so prefix caching can't short-circuit the prefill. Sends it with `concurrency` parallel
copies (each with its own tag), samples nvidia-smi every 0.5 s the whole time, and prints
per-card peak memory.used plus prefill/decode timing.

usage: depth-probe.py --port 8098 --model dsv4s --tokens 600000 [--concurrency 1]
                      [--gpus 0,1,2] [--out results/xxx.json] [--corpus DIR]
"""
import argparse, concurrent.futures as cf, json, os, random, subprocess, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=200)
ap.add_argument("--gpus", default="0,1,2")
ap.add_argument("--out")
ap.add_argument("--corpus", default=os.path.expanduser("~/CMP-170HX-PROJECT/vllm-dsv4/vllm/vllm"))
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(text):
    return post("/tokenize", {"model": a.model, "prompt": text})["count"]


# Corpus: real source files, deterministic order (read-only access to the tree).
files = []
for root, _, names in os.walk(a.corpus):
    files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".py")]
files.sort()
chunks, est = [], 0
for f in files:
    try:
        t = open(f, errors="ignore").read()
    except OSError:
        continue
    chunks.append(f"\n### FILE {os.path.relpath(f, a.corpus)}\n{t}")
    est += len(t) / 3.2
    if est > a.tokens * 1.3:
        break
body = "".join(chunks)
# Trim to target by bisection on character length using the real tokenizer.
lo, hi = 0, len(body)
while hi - lo > 2000:
    mid = (lo + hi) // 2
    if ntok(body[:mid]) < a.tokens:
        lo = mid
    else:
        hi = mid
body = body[:lo]
question = "\n\nQuestion: list the first three FILE headers above, then summarize what the last file does in two sentences."


def one(i):
    tag = f"[probe {random.getrandbits(64):016x}-{i}] "
    msgs = [{"role": "user", "content": tag + body + question}]
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens,
                                       "temperature": 0})
    dt = time.time() - t0
    u = r["usage"]
    return {"i": i, "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
            "wall_s": round(dt, 2), "finish": r["choices"][0]["finish_reason"],
            "answer_head": (r["choices"][0]["message"].get("content") or "")[:200]}


# VRAM sampler.
samples, stop = [], threading.Event()
def sampler():
    while not stop.is_set():
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        ts = time.time()
        for line in o.strip().splitlines():
            idx, mem, pw = [x.strip() for x in line.split(",")]
            samples.append((ts, int(idx), int(mem), float(pw)))
        stop.wait(0.5)

th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(2)
idle = {}
for ts, idx, mem, pw in samples:
    idle[idx] = mem
print(f"prompt ~{a.tokens} tokens x{a.concurrency}; idle MiB {idle}", flush=True)

results, error = [], None
try:
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for r in ex.map(one, range(a.concurrency)):
            results.append(r); print(r, flush=True)
except Exception as e:  # server crash / HTTP 500 is a result, not a script bug
    error = repr(e); print("REQUEST FAILED:", error, flush=True)
time.sleep(2); stop.set(); th.join()

gpus = [int(g) for g in a.gpus.split(",")]
peak = {g: max((m for _, i, m, _ in samples if i == g), default=None) for g in range(4)}
out = {"port": a.port, "model": a.model, "target_tokens": a.tokens, "concurrency": a.concurrency,
       "idle_mib": idle, "peak_mib": peak, "results": results, "error": error,
       "health_after": None, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
try:
    urllib.request.urlopen(BASE + "/health", timeout=10); out["health_after"] = "ok"
except Exception as e:
    out["health_after"] = repr(e)
print(json.dumps({k: out[k] for k in ("idle_mib", "peak_mib", "error", "health_after")}), flush=True)
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
