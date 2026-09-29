#!/usr/bin/env python3
"""Tool-calling, long-output, and web-research tests (run after the Phase 6 soak).
Server flags under test: --enable-auto-tool-choice --tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4. Chat-template defaults (thinking as the server sets it), i.e.
the way opencode/Open WebUI actually call it.

  T1 single tool call        right function, valid JSON args, finish_reason tool_calls
  T2 uses the tool result    final answer quotes the value the tool returned
  T3 no tool when not needed plain arithmetic with tools offered: answers directly
  T4 two lookups             both part IDs get looked up (parallel or sequential)
  T5 vision -> tool          reads a table image, calls the tool for the right row
  T6 long output             ~3,500-token generation stays coherent (no loops/garbage)
  T7 web research agent      real-world corruption case (1MDB) via SearXNG web_search +
                             fetch_url, multi-step; graded on key facts AND on citations:
                             every URL it cites must be one its tools actually returned
                             (no invented sources)

usage: tool-tests.py --port 8099 --model dsv4v [--out f.json] [--only T1,T7]
"""
import argparse, base64, html, io, json, re, time, urllib.parse, urllib.request
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--model", default="dsv4v")
ap.add_argument("--searx", default="http://127.0.0.1:8888")
ap.add_argument("--out")
ap.add_argument("--only")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}/v1/chat/completions"
PRICES = {"P-466": 261.28, "P-744": 324.47, "P-220": 832.37, "P-299": 640.88}

PRICE_TOOL = {"type": "function", "function": {
    "name": "get_part_price", "description": "Look up the current unit price of a part by its ID (format P-123).",
    "parameters": {"type": "object", "properties": {"part_id": {"type": "string"}}, "required": ["part_id"]}}}
SEARCH_TOOLS = [
    {"type": "function", "function": {"name": "web_search",
     "description": "Search the web. Returns a list of results with title, url and snippet.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_url",
     "description": "Fetch a web page and return its readable text (truncated).",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}}]


def chat(messages, tools=None, max_tokens=2048, **kw):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, **kw}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    c = j["choices"][0]
    return c["message"], c["finish_reason"], j["usage"], round(time.time() - t0, 1)


def calls(msg):
    out = []
    for tc in msg.get("tool_calls") or []:
        args = tc["function"]["arguments"]
        try:
            parsed = json.loads(args)
        except Exception:
            parsed = None
        out.append({"id": tc.get("id"), "name": tc["function"]["name"], "raw": args, "args": parsed})
    return out


def asst(msg):  # assistant message to feed back (content + tool_calls, no reasoning)
    m = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        m["tool_calls"] = msg["tool_calls"]
    return m


def T1():
    m, fr, u, s = chat([{"role": "user", "content": "What's the current price of part P-466?"}], [PRICE_TOOL])
    cs = calls(m)
    ok = fr == "tool_calls" and len(cs) >= 1 and cs[0]["name"] == "get_part_price" and cs[0]["args"] == {"part_id": "P-466"}
    return {"pass": ok, "finish": fr, "calls": cs, "s": s}


def T2():
    msgs = [{"role": "user", "content": "What's the current price of part P-466?"}]
    m, fr, u, s = chat(msgs, [PRICE_TOOL])
    cs = calls(m)
    if not cs:
        return {"pass": False, "why": "no tool call", "content": m.get("content")}
    msgs += [asst(m), {"role": "tool", "tool_call_id": cs[0]["id"], "content": json.dumps({"part_id": "P-466", "price": 261.28, "currency": "USD"})}]
    m2, fr2, u2, s2 = chat(msgs, [PRICE_TOOL])
    t = m2.get("content") or ""
    return {"pass": "261.28" in t and fr2 == "stop", "finish": fr2, "answer": t[:300], "s": s + s2}


def T3():
    m, fr, u, s = chat([{"role": "user", "content": "What is 17 times 23? Just the number."}], [PRICE_TOOL])
    return {"pass": not m.get("tool_calls") and "391" in (m.get("content") or ""), "answer": (m.get("content") or "")[:100], "finish": fr}


def run_price_loop(msgs, tools, max_turns=5):
    asked = []
    for _ in range(max_turns):
        m, fr, u, s = chat(msgs, tools)
        cs = calls(m)
        if not cs:
            return asked, m.get("content") or ""
        msgs.append(asst(m))
        for c in cs:
            pid = (c["args"] or {}).get("part_id", "")
            asked.append(pid)
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": json.dumps({"part_id": pid, "price": PRICES.get(pid), "currency": "USD"})})
    return asked, "(turn limit)"


def T4():
    asked, final = run_price_loop([{"role": "user", "content": "Get me the prices of parts P-466 and P-744, then tell me which is cheaper."}], [PRICE_TOOL])
    return {"pass": {"P-466", "P-744"} <= set(asked) and "466" in final, "asked": asked, "answer": final[:300]}


def T5():
    cols = ["Part", "Qty", "Width", "Height"]
    rows = [["P-466", "53", "62.5", "37.2"], ["P-744", "36", "29.9", "60.2"], ["P-220", "26", "76.4", "89.8"], ["P-299", "55", "49.0", "49.0"]]
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22); cw, rh = 150, 44
    im = Image.new("RGB", (cw * 4 + 20, rh * 5 + 20), "white"); d = ImageDraw.Draw(im)
    for ri, row in enumerate([cols] + rows):
        for ci, cell in enumerate(row):
            x, y = 10 + ci * cw, 10 + ri * rh
            d.rectangle([x, y, x + cw, y + rh], outline="black"); d.text((x + 10, y + 10), cell, fill="black", font=f)
    buf = io.BytesIO(); im.save(buf, "PNG"); url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Find the part with the highest Qty in this table and look up its current price with the tool."},
                                         {"type": "image_url", "image_url": {"url": url}}]}]
    asked, final = run_price_loop(msgs, [PRICE_TOOL])
    return {"pass": "P-299" in asked and "640.88" in final, "asked": asked, "answer": final[:300]}


def T6():
    m, fr, u, s = chat([{"role": "user", "content": "Write a detailed, well-structured technical guide (about 2,500 words) on designing a NURBS surface modeling kernel: data structures, knot vectors, evaluation, trimming, tessellation, and robustness. Use headings."}],
                       max_tokens=6000)
    t = m.get("content") or ""
    w = re.findall(r"\w+", t.lower()); grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    uniq = len(set(grams)) / max(1, len(grams))
    return {"pass": u["completion_tokens"] >= 2000 and uniq >= 0.6 and fr == "stop", "completion_tokens": u["completion_tokens"],
            "unique_4gram_ratio": round(uniq, 3), "finish": fr, "tail": t[-200:], "s": s,
            "tok_s": round(u["completion_tokens"] / s, 1)}


def searx(q):
    u = f"{a.searx}/search?" + urllib.parse.urlencode({"q": q, "format": "json"})
    with urllib.request.urlopen(u, timeout=30) as r:
        res = json.loads(r.read()).get("results", [])[:8]
    return [{"title": x.get("title", ""), "url": x["url"], "snippet": (x.get("content") or "")[:300]} for x in res]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-test"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read(3_000_000).decode(errors="ignore")
    raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    txt = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", txt).strip()[:8000]


def T7():
    q = ("Research the 1MDB (1Malaysia Development Berhad) corruption scandal using web search. "
         "Who were the central figures, roughly how much money was misappropriated, what role did Goldman Sachs play, "
         "and what were the main legal outcomes? Use at least two searches and read at least one source page. "
         "Finish with a concise report and a 'Sources' list of the URLs you actually used.")
    msgs = [{"role": "user", "content": q}]
    seen, trace = set(), []
    final, fr = None, None
    for step in range(14):
        m, fr, u, s = chat(msgs, SEARCH_TOOLS, max_tokens=4096)
        cs = calls(m)
        if not cs:
            final = m.get("content") or ""; break
        msgs.append(asst(m))
        for c in cs:
            args = c["args"] or {}
            try:
                if c["name"] == "web_search":
                    res = searx(args.get("query", "")); seen.update(r["url"] for r in res); out = json.dumps(res)
                elif c["name"] == "fetch_url":
                    url = args.get("url", ""); out = fetch(url); seen.add(url)
                else:
                    out = f"unknown tool {c['name']}"
            except Exception as e:
                out = f"ERROR: {e!r}"
            trace.append({"step": step, "tool": c["name"], "args": args, "bad_json": c["args"] is None, "result_chars": len(out)})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    final = final or ""
    cited = set(re.findall(r"https?://[^\s)\]>\"']+", final))
    cited = {x.rstrip(".,;") for x in cited}
    invented = sorted(x for x in cited if x not in seen)
    low = final.lower()
    facts = {"najib": "najib" in low, "jho_low": "jho low" in low or "low taek jho" in low, "goldman": "goldman" in low,
             "billions": bool(re.search(r"\b(4\.5|2\.7|3\.5|4\.[0-9]|[0-9]+(\.[0-9])?) ?(billion|bn)", low)),
             "legal_outcome": any(w in low for w in ["convicted", "sentenced", "guilty", "prison", "settle", "plea"])}
    n_search = sum(t["tool"] == "web_search" for t in trace); n_fetch = sum(t["tool"] == "fetch_url" for t in trace)
    return {"pass": all(facts.values()) and n_search >= 2 and n_fetch >= 1 and cited and not invented
                    and not any(t["bad_json"] for t in trace),
            "searches": n_search, "fetches": n_fetch, "facts": facts, "cited": sorted(cited), "invented_urls": invented,
            "steps": len(trace), "finish": fr, "trace": trace, "report": final}


res = {}
only = set(a.only.split(",")) if a.only else None
for t in [T1, T2, T3, T4, T5, T6, T7]:
    if only and t.__name__ not in only:
        continue
    try:
        res[t.__name__] = t()
    except Exception as e:
        res[t.__name__] = {"pass": False, "error": repr(e)}
    r = res[t.__name__]
    show = {k: v for k, v in r.items() if k not in ("trace", "report")}
    print(f"{t.__name__}: {'PASS' if r.get('pass') else 'FAIL'}  {json.dumps(show)[:600]}", flush=True)
print("SUMMARY", json.dumps({k: v.get("pass") for k, v in res.items()}))
if a.out:
    json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": res}, open(a.out, "w"), indent=1)
