#!/usr/bin/env bash
# Repeated-samples power-cap sweep on card #3, Qwen3.6-35B-A3B-MTP fullctx server (:8080).
#
# The fine 5W-step sweep (power-sweep-qwen35b-fine.sh, 100-175W) showed run-to-run noise
# (spread up to +/-22 tok/s on n=1-2 samples) comparable to or larger than the point-to-point
# differences -- no knee could be trusted at that resolution. This script trades step
# resolution for repeat count: fewer, coarser power levels, REPEATS samples at each, so the
# mean actually beats the noise instead of chasing it.
#
# Round-robin direction: odd rounds sweep the LEVELS list ascending, even rounds descending.
# This spreads thermal drift roughly evenly across all levels (a single up-then-down pass
# still biases the two directions differently), rather than trying to eliminate it.
set -uo pipefail

PORT=8080
PROMPT='Write a 250-word essay about the history of Tokyo, covering its origins as Edo and its transformation into the modern capital of Japan.'
LEVELS=(100 120 140 160 180 200)
REPEATS="${REPEATS:-5}"
OUT="$(dirname "$0")/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="${OUT}/power-sweep-qwen35b-repeated_${STAMP}.csv"
echo "round,direction,watt_cap,tokens,decode_tok_s,prefill_tok_s,avg_power_w,peak_power_w,avg_temp_c,max_temp_c,tok_per_watt" > "$CSV"

run_one() {
    local ROUND="$1" DIR="$2" CAP="$3"
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
print(f"[round {$ROUND} {'$DIR':4s} {int('$CAP'):3d}W] tokens={tokens:3d}  decode={decode:6.2f} tok/s  "
      f"avgP={avg_p:6.1f}W peakP={peak_p:6.1f}W  avgT={avg_t:4.1f}C  tok/W={tpw:.3f}")
with open("$CSV", "a", newline="") as f:
    w = csv.writer(f)
    w.writerow(["$ROUND", "$DIR", "$CAP", tokens, f"{decode:.2f}", f"{prefill:.2f}", f"{avg_p:.1f}", f"{peak_p:.1f}", f"{avg_t:.1f}", f"{max_t:.1f}", f"{tpw:.4f}"])
PY
    rm -f "$LOG" "$RESPFILE"
}

echo "=================================================================="
echo " Repeated-samples power sweep: ${LEVELS[*]} W, ${REPEATS} rounds, alternating direction"
echo "=================================================================="
for ((r=1; r<=REPEATS; r++)); do
    if (( r % 2 == 1 )); then
        DIR="ASC"; SEQ=("${LEVELS[@]}")
    else
        DIR="DESC"
        SEQ=()
        for ((i=${#LEVELS[@]}-1; i>=0; i--)); do SEQ+=("${LEVELS[i]}"); done
    fi
    for W in "${SEQ[@]}"; do run_one "$r" "$DIR" "$W"; done
done

echo ""
echo "=================================================================="
echo "results: $CSV"
echo "=================================================================="
python3 <<PY
import csv, statistics
from collections import defaultdict
rows = list(csv.DictReader(open("$CSV")))
by_watt = defaultdict(list)
for r in rows:
    by_watt[int(r["watt_cap"])].append(r)

print(f"{'watt':>5} {'n':>3} {'mean_decode':>11} {'std_decode':>10} {'mean_avgP':>10} {'mean_tok/W':>11} {'std_tok/W':>10}")
for w in sorted(by_watt):
    entries = by_watt[w]
    dec = [float(e["decode_tok_s"]) for e in entries]
    avgp = [float(e["avg_power_w"]) for e in entries]
    tpw = [float(e["tok_per_watt"]) for e in entries]
    print(f"{w:5d} {len(entries):3d} {statistics.mean(dec):11.2f} {statistics.pstdev(dec):10.2f} "
          f"{statistics.mean(avgp):10.1f} {statistics.mean(tpw):11.3f} {statistics.pstdev(tpw):10.3f}")
PY
