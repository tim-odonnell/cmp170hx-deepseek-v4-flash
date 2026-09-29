#!/usr/bin/env python3
"""Decode speed at depth, same method as 0731's 2026-08-15 Phase 5 entry: prefill a real-text
prefix once (timed = prefill rate), then use vLLM's automatic prefix caching to measure decode
at that depth cheaply. Identical prefix text for every model, so results are comparable.

usage: depth-decode.py --port 8099 --model dsv4v --depths 100000,300000 [--runs 3] [--out f.json]
"""
import argparse, json, os, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--depths", default="100000,300000")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--out")
ap.add_argument("--corpus", default=os.path.expanduser("~/CMP-170HX-PROJECT/vllm-dsv4/vllm/vllm"))
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


files = []
for root, _, names in os.walk(a.corpus):
    files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".py")]
files.sort()
corpus = "".join(f"\n### FILE {os.path.relpath(f, a.corpus)}\n{open(f, errors='ignore').read()}" for f in files)


def prefix_for(n):
    lo, hi = 0, min(len(corpus), int(n * 4.5))
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        if post("/tokenize", {"model": a.model, "prompt": corpus[:mid]})["count"] < n:
            lo = mid
        else:
            hi = mid
    return corpus[:lo]


QUESTIONS = ["Summarize the purpose of the first file above in detail.",
             "Explain in detail what the second file above is for.",
             "Describe in detail how the files above are organized."]


def stream(content):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tf = tl = None; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time(); tf = tf or now; tl = now
    n = usage["completion_tokens"]
    return {"ttft_s": round(tf - t0, 2), "decode_tok_s": round((n - 1) / (tl - tf), 2),
            "prompt_tokens": usage["prompt_tokens"]}


out = {"model": a.model, "port": a.port, "depths": {}}
for d in [int(x) for x in a.depths.split(",")]:
    pre = prefix_for(d)
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": a.model, "max_tokens": 1, "temperature": 0,
                                      "messages": [{"role": "user", "content": pre + "\n\n" + QUESTIONS[0]}],
                                      "chat_template_kwargs": {"thinking": False}})
    pf = time.time() - t0; pt = r["usage"]["prompt_tokens"]
    runs = [stream(pre + "\n\n" + QUESTIONS[i % len(QUESTIONS)]) for i in range(a.runs)]
    dec = [x["decode_tok_s"] for x in runs]
    out["depths"][d] = {"prompt_tokens": pt, "prefill_s": round(pf, 1), "prefill_tok_s": round(pt / pf),
                        "decode_mean": round(statistics.mean(dec), 2), "decode_median": round(statistics.median(dec), 2),
                        "runs": runs}
    print(json.dumps({"depth": d, **{k: v for k, v in out["depths"][d].items() if k != "runs"}}), flush=True)
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
