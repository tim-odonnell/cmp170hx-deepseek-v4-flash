#!/usr/bin/env python3
"""Phase 3 vision gates for DeepSeek-V4-Flash-Vision-Exp. Every test has a known
ground truth so it grades itself; raw answers are saved for human review.

  G1 gradient        green->orange gradient: names both colours, green first
  G2 no-image ctl    same question, no image: must NOT produce G1's answer
  G3 wrong-image ctl yellow->purple gradient: must name those, not green/orange
  G4 table           rendered 4x5 table with random numbers: cell accuracy
  G5 UI screenshot   real Firefox-rendered page: heading text, card count, button colour,
                     planted visual bug (overlapping text) -- the dev-agent use case
  G6 two images      which of two images contains yellow
  G7 contamination   image + text requests interleaved concurrently: text answers must be
                     correct and must not leak other requests' content (CUDA-graph bug
                     reported by kaka86mm)
  G8 invalid image   garbage image -> clean 4xx, server stays healthy
  G9 concurrency     c=1,2,4 image requests all succeed, server stays healthy

usage: vision-gates.py --port 8099 --model dsv4v [--out results/vision-gates-<ts>.json]
"""
import argparse, base64, concurrent.futures as cf, io, json, os, random, re, shutil, subprocess, tempfile, time, urllib.error, urllib.request
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--model", default="dsv4v")
ap.add_argument("--out")
ap.add_argument("--only", help="comma list, e.g. G1,G4")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
rng = random.Random(1101)


def b64png(img):
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(content, max_tokens=400, timeout=900):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
        return {"status": 200, "text": j["choices"][0]["message"].get("content") or "",
                "s": round(time.time() - t0, 2), "usage": j.get("usage")}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "text": e.read().decode()[:300], "s": round(time.time() - t0, 2)}


def healthy():
    try:
        urllib.request.urlopen(BASE + "/health", timeout=10); return True
    except Exception:
        return False


def img_msg(text, *imgs):
    return [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]


def gradient(c1, c2, size=448):
    im = Image.new("RGB", (size, size))
    px = im.load()
    for x in range(size):
        t = x / (size - 1)
        col = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        for y in range(size):
            px[x, y] = col
    return b64png(im)


GRN_ORG = gradient((20, 160, 60), (245, 140, 20))  # not red/blue: 0731 guessed "BLUE, RED" blind
YEL_PUR = gradient((240, 220, 20), (130, 20, 170))
Q_GRAD = "This image is a horizontal gradient between two colours. Name the two colours, left colour first. Answer in the form: LEFT, RIGHT"
res = {}


def first_pos(t, words):
    ps = [t.find(w) for w in words if w in t]
    return min(ps) if ps else -1


def G1():
    r = chat(img_msg(Q_GRAD, GRN_ORG), 60); t = r["text"].lower()
    r["pass"] = 0 <= first_pos(t, ["green"]) < first_pos(t, ["orange"])
    return r

def G2():
    r = chat([{"type": "text", "text": Q_GRAD}], 80); t = r["text"].lower()
    r["pass"] = not ("green" in t and "orange" in t)  # blind guess must not hit G1's answer
    return r

def G3():
    r = chat(img_msg(Q_GRAD, YEL_PUR), 60); t = r["text"].lower()
    r["pass"] = ("yellow" in t or "gold" in t) and ("purple" in t or "violet" in t or "magenta" in t) and "green" not in t and "orange" not in t
    return r

def G4():
    cols = ["Part", "Qty", "Width", "Height", "Cost"]
    rows = [[f"P-{rng.randint(100,999)}", str(rng.randint(1, 60)), f"{rng.randint(10,99)}.{rng.randint(0,9)}",
             f"{rng.randint(10,99)}.{rng.randint(0,9)}", f"{rng.randint(100,999)}.{rng.randint(10,99)}"] for _ in range(4)]
    f = ImageFont.truetype(FONT, 22); cw, rh = 150, 44
    im = Image.new("RGB", (cw * 5 + 20, rh * 5 + 20), "white"); d = ImageDraw.Draw(im)
    for ri, row in enumerate([cols] + rows):
        for ci, cell in enumerate(row):
            x, y = 10 + ci * cw, 10 + ri * rh
            d.rectangle([x, y, x + cw, y + rh], outline="black", fill="#dde6f0" if ri == 0 else "white")
            d.text((x + 10, y + 10), cell, fill="black", font=f)
    r = chat(img_msg("Transcribe this table exactly as CSV (header row first). Output only the CSV.", b64png(im)), 400)
    truth = [cols] + rows
    got = [[c.strip() for c in ln.split(",")] for ln in r["text"].strip().strip("`").replace("csv\n", "").splitlines() if "," in ln]
    ok = sum(1 for i in range(5) for j in range(5) if i < len(got) and j < len(got[i]) and got[i][j] == truth[i][j])
    r.update(truth=truth, cells_correct=f"{ok}/25", pass_=ok == 25); r["pass"] = ok >= 24
    return r

def G5():
    heading = "Aurora Parts Dashboard"
    html = f"""<!doctype html><html><body style="margin:0;font-family:DejaVu Sans,sans-serif;background:#f4f1ea">
<header style="background:#1d3557;color:white;padding:24px 32px;font-size:34px">{heading}</header>
<main style="display:flex;gap:24px;padding:32px">
 <div style="background:white;width:260px;height:180px;padding:16px;border-radius:10px">Orders<br><b style="font-size:40px">128</b></div>
 <div style="background:white;width:260px;height:180px;padding:16px;border-radius:10px">Returns<br><b style="font-size:40px">7</b></div>
 <div style="background:white;width:260px;height:180px;padding:16px;border-radius:10px;position:relative">Revenue
   <b style="font-size:40px;position:absolute;top:18px;left:14px;color:#888">$41,200</b></div>
</main>
<div style="padding:0 32px"><button style="background:#e4572e;color:white;border:0;padding:16px 40px;font-size:22px;border-radius:8px">Export report</button></div>
</body></html>"""
    # Snap Firefox can't read/write hidden dirs, so use a plain dir next to this script's
    # results; throwaway profile so it never touches a running user Firefox.
    tmp = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"), prefix="vg-")
    hp, pp = os.path.join(tmp, "page.html"), os.path.join(tmp, "shot.png")
    open(hp, "w").write(html); os.makedirs(os.path.join(tmp, "prof"))
    subprocess.run(["firefox", "--headless", "--no-remote", "--profile", os.path.join(tmp, "prof"),
                    "--window-size=1100,560", f"--screenshot={pp}", "file://" + hp],
                   capture_output=True, timeout=120)
    if not os.path.exists(pp):
        shutil.rmtree(tmp, ignore_errors=True)
        return {"pass": None, "text": "SKIPPED: firefox headless screenshot failed"}
    img = b64png(Image.open(pp).convert("RGB")); shutil.copy(pp, os.path.join(os.path.dirname(a.out or "."), "G5-screenshot.png")) if a.out else None
    shutil.rmtree(tmp, ignore_errors=True)
    q = ("This is a screenshot of a web page I built. Answer as a numbered list: 1) the exact heading text; "
         "2) how many white cards are shown; 3) the colour of the button and its label; "
         "4) any visual/layout bug you can see (be specific about which card).")
    r = chat(img_msg(q, img), 400); t = r["text"].lower()
    r["checks"] = {"heading": heading.lower() in t, "three_cards": bool(re.search(r"\b(3|three)\b", t)),
                   "button_orange_red": any(w in t for w in ["orange", "red", "coral", "vermilion"]),
                   "button_label": "export report" in t,
                   "bug_revenue_overlap": "revenue" in t and any(w in t for w in ["overlap", "overlapp", "collid", "on top", "obscur", "cover"])}
    r["pass"] = all(r["checks"].values())
    return r

def G6():
    r = chat(img_msg("Two images follow. Which one contains yellow: the first or the second? Answer with one word.", GRN_ORG, YEL_PUR), 20)
    r["pass"] = "second" in r["text"].lower() and "first" not in r["text"].lower()
    return r

def G7():
    facts = [("What is the capital of France? One word.", "paris"), ("What is the capital of Japan? One word.", "tokyo"),
             ("What is the capital of Italy? One word.", "rome"), ("What is the capital of Canada? One word.", "ottawa")]
    jobs = []
    for i in range(4):
        jobs.append(("img", img_msg(Q_GRAD, GRN_ORG if i % 2 == 0 else YEL_PUR), None))
        jobs.append(("txt", [{"type": "text", "text": facts[i][0]}], facts[i][1]))
    with cf.ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda j: chat(j[1], 40), jobs))
    bad = []
    for (kind, _, want), o in zip(jobs, outs):
        t = o["text"].lower()
        if kind == "txt" and (want not in t or any(c in t for c in ["green", "orange", "yellow", "purple", "gradient"])):
            bad.append({"want": want, "got": o["text"][:120]})
    return {"pass": not bad and all(o["status"] == 200 for o in outs), "bad": bad,
            "answers": [o["text"][:80] for o in outs]}

def G8():
    r = chat(img_msg("Describe this image.", "data:image/png;base64,bm90IGFuIGltYWdl"), 40)
    r["healthy_after"] = healthy()
    r["pass"] = 400 <= r["status"] < 500 and r["healthy_after"]
    return r

def G9():
    out = {}
    for c in (1, 2, 4):
        with cf.ThreadPoolExecutor(c) as ex:
            rs = list(ex.map(lambda i: chat(img_msg(Q_GRAD, GRN_ORG), 40), range(c)))
        out[f"c{c}"] = {"ok": sum(r["status"] == 200 for r in rs), "max_s": max(r["s"] for r in rs)}
    out["healthy_after"] = healthy()
    out["pass"] = all(v["ok"] == int(k[1:]) for k, v in out.items() if k.startswith("c")) and out["healthy_after"]
    return out


tests = [G1, G2, G3, G4, G5, G6, G7, G8, G9]
only = set(a.only.split(",")) if a.only else None
for t in tests:
    if only and t.__name__ not in only:
        continue
    if not healthy():
        res[t.__name__] = {"pass": False, "text": "server not healthy before test"}; break
    try:
        res[t.__name__] = t()
    except Exception as e:
        res[t.__name__] = {"pass": False, "error": repr(e)}
    r = res[t.__name__]
    print(f"{t.__name__}: {'PASS' if r.get('pass') else ('SKIP' if r.get('pass') is None else 'FAIL')}  "
          f"{json.dumps({k: v for k, v in r.items() if k not in ('usage', 'truth')})[:400]}", flush=True)

summary = {k: v.get("pass") for k, v in res.items()}
print("SUMMARY", json.dumps(summary))
if a.out:
    json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "port": a.port, "model": a.model,
               "summary": summary, "results": res}, open(a.out, "w"), indent=1)
