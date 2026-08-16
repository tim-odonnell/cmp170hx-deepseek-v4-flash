#!/usr/bin/env bash
# Short proof that the unlocked 64 GB is genuinely usable — not just reported, and not just
# allocatable, but correct under a real model doing real compute.
#
# A pattern sweep proves the memory stores and returns bytes. It does NOT prove the card can
# hold 63 GB of model weights and produce correct arithmetic from them. Corrupted or unstable
# VRAM on this card shows up as GIBBERISH OUTPUT, not as an error — there is no ECC and no
# error telemetry. So the pass criteria here are:
#
#   1. --fit actually places >60 GiB of weights in VRAM
#   2. the model generates COHERENT text (printed in full, judge it yourself)
#   3. zero Xid faults during the run
#
# Runtime ~4 minutes, most of it model load.
set -uo pipefail
MODEL=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
PORT=8080
SER=$(sudo nvidia-smi -q 2>/dev/null | awk -F: '/Serial Number/{gsub(/ /,"",$2);print $2;exit}')
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
LOG="$OUT/verify64_${SER}_$(date +%Y%m%d_%H%M%S).log"

[[ -f "$MODEL" && -x "$SERVER" ]] || { echo "model or sm_80 server missing"; exit 1; }

XID_BEFORE=$(sudo dmesg 2>/dev/null | grep -ci xid)
SWAP=0; SRV=""
cleanup(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 5; kill -9 "$SRV" 2>/dev/null; }
           (( SWAP )) && sudo systemctl start llama-swap 2>/dev/null; }
trap cleanup EXIT INT TERM
systemctl is-active --quiet llama-swap && { SWAP=1; sudo systemctl stop llama-swap; sleep 3; }

echo "=================================================================="
echo " 64 GB usability check — card serial ${SER}"
echo "=================================================================="
echo "  before: $(nvidia-smi --query-gpu=memory.total,memory.used,temperature.gpu --format=csv,noheader)"
echo ">>> loading DeepSeek-V4-Flash-0731 UD-Q8_K_XL (151 GiB model, ~90 s)"

"$SERVER" --model "$MODEL" --alias verify64 --fit on --ctx-size 8192 --ctx-checkpoints 0 \
  -fa on -ctk f16 -ctv f16 --parallel 1 --temp 0.6 --top-p 0.95 --min-p 0.0 \
  --threads 11 --threads-batch 11 --no-mmap \
  --reasoning-format deepseek --reasoning on --reasoning-budget 2000 \
  --host 127.0.0.1 --port "$PORT" --jinja > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 30); do
    kill -0 "$SRV" 2>/dev/null || { echo "  SERVER DIED:"; tail -20 "$LOG"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "  ready in $((i*10))s"; break; }
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "  never became ready"; tail -20 "$LOG"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo ""
echo "  === CHECK 1: weights actually resident in VRAM ==="
printf "  VRAM in use: %s MiB of %s MiB\n" "$VRAM" "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"
if (( VRAM > 61440 )); then echo "  PASS — over 60 GiB of the model is on the card"
else echo "  FAIL — only ${VRAM} MiB resident"; fi
free -h | awk '/^Mem:/{print "  host RAM: "$3" of "$2}'

echo ""
echo "  === CHECK 2: coherent generation (the real test) ==="
# Use the CHAT endpoint, not /completion. /completion feeds the prompt as raw text with no
# chat template, so the model does dataset-style continuation instead of answering -- coherent,
# but useless as a demonstration. /v1/chat/completions applies the jinja template.
RESP=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In exactly three sentences, explain why a linked list has O(1) insertion but O(n) lookup."}],"max_tokens":400,"temperature":0.6}')
printf '%s' "$RESP" > /tmp/verify64_resp.json
python3 - /tmp/verify64_resp.json <<'PYEOF'
import json, re, sys
r = json.load(open(sys.argv[1]))
# chat endpoint: text is under choices[0].message.content; reasoning models may put the
# visible answer there and the chain-of-thought in reasoning_content.
ch = (r.get("choices") or [{}])[0].get("message", {})
txt = ch.get("content") or ""
if not txt.strip():
    txt = ch.get("reasoning_content") or ""
t = r.get("timings", {})
# NB: pull values into locals first. Backslash-escaped quotes inside an f-string
# expression are a SyntaxError, and this block lives inside a shell heredoc.
tps = t.get("predicted_per_second", 0)
ntok = t.get("predicted_n", 0)
print("  ---------------- model output ----------------")
for line in (txt.strip() or "(EMPTY)").splitlines():
    print("   ", line)
print("  ----------------------------------------------")
print(f"  decode {tps:.2f} tok/s, {ntok} tokens")
words = re.findall(r"[A-Za-z]{2,}", txt)
uniq = len(set(w.lower() for w in words))
print(f"  sanity: {len(words)} words, {uniq} unique")
print("  PASS -- output is real text" if len(words) > 25 and uniq > 15
      else "  FAIL -- output looks degenerate")
PYEOF
rm -f /tmp/verify64_resp.json

echo ""
echo "  === CHECK 3: no hardware faults ==="
XID_AFTER=$(sudo dmesg 2>/dev/null | grep -ci xid)
echo "  Xid before: ${XID_BEFORE}   after: ${XID_AFTER}"
(( XID_AFTER == XID_BEFORE )) && echo "  PASS — no new Xid faults" || echo "  FAIL — new Xid faults appeared"
echo "  peak temp: $(nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader)"
echo "=================================================================="
