#!/usr/bin/env python3
"""Quick decode/prefill benchmark against a local vLLM OpenAI-compatible server.
Usage: bench.py <prompt_file_or_'short'> <max_tokens> [label]
"""
import sys
import time
import json
import urllib.request

URL = "http://127.0.0.1:8098/v1/completions"

def run(prompt, max_tokens, label):
    payload = {
        "model": "dsv4s",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    usage = body.get("usage", {})
    ptoks = usage.get("prompt_tokens", 0)
    ctoks = usage.get("completion_tokens", 0)
    print(f"[{label}] prompt_tokens={ptoks} completion_tokens={ctoks} "
          f"elapsed={elapsed:.2f}s decode_tok/s={ctoks/elapsed:.2f} "
          f"(includes prefill; rough)" )
    return body

if __name__ == "__main__":
    src = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    label = sys.argv[3] if len(sys.argv) > 3 else src

    if src == "short":
        prompt = "The quick brown fox jumps over the lazy dog. Explain why this sentence is famous."
    else:
        with open(src) as f:
            prompt = f.read()

    run(prompt, max_tokens, label)
