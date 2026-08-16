#!/usr/bin/env bash
# Fine power-cap sweep (125-175W, 5W steps) on card #3 while Qwen3.6-35B-A3B-MTP fullctx
# server is live on :8080. Sweeps UP then DOWN through the same levels to catch thermal-drift
# bias (a single-direction sweep can look monotonic purely from residual heat -- see project
# memory, +5.8%/-4% effects seen this way before). Same fixed-length prompt as the earlier
# 100/125/150/200W tests so token count is pinned at 250 and decode tok/s is directly
# comparable -- variable-length reasoning output at temp=1.0 previously produced misleading
# short-generation numbers.
set -uo pipefail

PORT=8080
PROMPT='Write a 250-word essay about the history of Tokyo, covering its origins as Edo and its transformation into the modern capital of Japan.'
LEVELS_UP=(100 105 110 115 120 125 130)
LEVELS_DOWN=(130 125 120 115 110 105 100)
OUT="$(dirname "$0")/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="${OUT}/power-sweep-qwen35b-fine_${STAMP}.csv"
echo "direction,watt_cap,tokens,decode_tok_s,prefill_tok_s,avg_power_w,peak_power_w,avg_temp_c,max_temp_c,tok_per_watt" > "$CSV"

run_one() {
    local DIR="$1" CAP="$2"
    sudo nvidia-smi -i 0 -pl "$CAP" >/dev/null
    sleep 1.5
    local LOG; LOG=$(mktemp)
    local RESPFILE; RESPFILE=$(mktemp)
    echo "timestamp,temp_c,power_w" > "$LOG"
    ( while true; do nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$LOG"; sleep 0.25; done ) &
    local SAMPLER=$!
    curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT"),\"n_predict\":250,\"cache_prompt\":false}" \
        > "$RESPFILE"
    kill "$SAMPLER" 2>/dev/null

    python3 <<PY
import json, csv
resp = json.load(open("$RESPFILE"))
t = resp["timings"]
tokens = t.get("predicted_n", 0)
decode = t.get("predicted_per_second", 0.0)
prefill = t.get("prompt_per_second", 0.0)
rows = list(csv.DictReader(open("$LOG")))
temps = [float(r["temp_c"]) for r in rows] if rows else [0]
pwrs = [float(r["power_w"]) for r in rows] if rows else [0]
avg_p = sum(pwrs)/len(pwrs)
peak_p = max(pwrs)
avg_t = sum(temps)/len(temps)
max_t = max(temps)
tpw = decode/avg_p if avg_p else 0
print(f"[{'$DIR':4s} {int('$CAP'):3d}W] tokens={tokens:3d}  decode={decode:6.2f} tok/s  prefill={prefill:6.2f} tok/s  "
      f"avgP={avg_p:6.1f}W peakP={peak_p:6.1f}W  avgT={avg_t:4.1f}C maxT={max_t:4.1f}C  tok/W={tpw:.3f}")
with open("$CSV", "a", newline="") as f:
    w = csv.writer(f)
    w.writerow(["$DIR", "$CAP", tokens, f"{decode:.2f}", f"{prefill:.2f}", f"{avg_p:.1f}", f"{peak_p:.1f}", f"{avg_t:.1f}", f"{max_t:.1f}", f"{tpw:.4f}"])
PY
    rm -f "$LOG" "$RESPFILE"
}

echo "=================================================================="
echo " Fine power sweep: 125-175W, 5W steps, UP then DOWN"
echo "=================================================================="
echo "--- UP sweep ---"
for W in "${LEVELS_UP[@]}"; do run_one "UP" "$W"; done
echo "--- DOWN sweep ---"
for W in "${LEVELS_DOWN[@]}"; do run_one "DOWN" "$W"; done

echo ""
echo "=================================================================="
echo "results: $CSV"
echo "=================================================================="
python3 <<PY
import csv
from collections import defaultdict
rows = list(csv.DictReader(open("$CSV")))
by_watt = defaultdict(list)
for r in rows:
    by_watt[int(r["watt_cap"])].append(r)

print(f"{'watt':>5} {'decode(UP)':>11} {'decode(DOWN)':>13} {'avg_decode':>11} {'avg_tok/W':>10}")
for w in sorted(by_watt):
    entries = by_watt[w]
    up = next((float(e["decode_tok_s"]) for e in entries if e["direction"]=="UP"), None)
    down = next((float(e["decode_tok_s"]) for e in entries if e["direction"]=="DOWN"), None)
    vals = [float(e["decode_tok_s"]) for e in entries]
    tpws = [float(e["tok_per_watt"]) for e in entries]
    avg_d = sum(vals)/len(vals)
    avg_tpw = sum(tpws)/len(tpws)
    print(f"{w:5d} {up if up else 0:11.2f} {down if down else 0:13.2f} {avg_d:11.2f} {avg_tpw:10.3f}")
PY
