#!/usr/bin/env python3
"""Decode-speed benchmark, identical for 0731 and Vision-Exp.
Rules (from kaka86mm/leinasi2014, each learned from a real misdiagnosis):
  - streaming; decode rate = (completion_tokens - 1) / (t_last - t_first_token), TTFT separate
  - token counts from the final `usage` object, never from chunk counts (DSpark emits
    several tokens per chunk)
  - first request after launch is a discarded warmup (Triton JIT)
  - random tag at the start of every prompt so prefix caching can't help
  - fixed length: ignore_eos + max_tokens, so every run decodes the same number of tokens

usage: speed-bench.py --port 8098 --model dsv4s [--runs 5] [--tokens 300] [--image] [--out f.json]
"""
import argparse, base64, io, json, random, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--image", action="store_true", help="attach a 448x448 test image to each request")
ap.add_argument("--out")
a = ap.parse_args()

PROMPTS = ["Write a detailed technical explanation of how B-spline basis functions are evaluated with the Cox-de Boor recursion, with an example.",
           "Explain, step by step, how a pipeline-parallel inference server splits a transformer across GPUs and what limits its throughput.",
           "Write a Python function that parses a CSV of part dimensions and reports outliers, then explain the design choices.",
           "Describe the history and working principle of the Stirling engine in depth.",
           "Explain how a web browser turns HTML and CSS into pixels on screen, covering layout and painting."]

img_url = None
if a.image:
    from PIL import Image
    im = Image.new("RGB", (448, 448))
    for x in range(448):
        for y in range(448):
            im.putpixel((x, y), (x // 2, y // 2, 128))
    buf = io.BytesIO(); im.save(buf, "PNG")
    img_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(prompt):
    text = f"[{random.getrandbits(48):012x}] {prompt}"
    content = text if not img_url else [{"type": "text", "text": text + " (Ignore the attached image.)"},
                                        {"type": "image_url", "image_url": {"url": img_url}}]
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time(); t_first = t_first or now; t_last = now
    n = usage["completion_tokens"]
    return {"ttft_s": round(t_first - t0, 3), "decode_tok_s": round((n - 1) / (t_last - t_first), 2),
            "completion_tokens": n, "prompt_tokens": usage["prompt_tokens"]}


run(PROMPTS[0])  # warmup, discarded
rs = []
for i in range(a.runs):
    r = run(PROMPTS[i % len(PROMPTS)]); rs.append(r); print(r, flush=True)
d = [r["decode_tok_s"] for r in rs]
summ = {"mean": round(statistics.mean(d), 2), "median": round(statistics.median(d), 2), "min": min(d), "max": max(d),
        "ttft_median_s": statistics.median(r["ttft_s"] for r in rs), "runs": a.runs, "tokens": a.tokens, "image": a.image}
print("SUMMARY", json.dumps(summ))
if a.out:
    json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "port": a.port, "model": a.model, "summary": summ, "runs": rs},
              open(a.out, "w"), indent=1)
