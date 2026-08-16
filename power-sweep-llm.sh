#!/usr/bin/env bash
# Find the lowest power limit that keeps real LLM performance high.
#
# WHY A SYNTHETIC GEMM TEST CANNOT ANSWER THIS
# --------------------------------------------
# LLM decode on this box is DDR5-bandwidth-bound, not GPU-bound: during the 85-minute soak
# the GPU sat at ~69 W for 78% of samples while util read high -- it was waiting on memory.
# Prefill is the opposite: GPU-bound, with bursts to 165.9 W against a 100 W cap. So the two
# phases have completely different power sensitivity and must be measured separately, on the
# real model, not on a matmul benchmark.
#
# METHOD
#   * one model load, then the power limit is changed live with `nvidia-smi -pl` between
#     measurements -- no reload, so the only variable is the cap
#   * UP then DOWN through the same values, results averaged per level. A single-direction
#     sweep on this box previously gave wrong answers because of ~8.6% thermal drift; going
#     both ways cancels the drift instead of baking it in
#   * prefill is measured with a long prompt (short prompts are dominated by fixed overhead)
set -uo pipefail

MODEL_DIR=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731
MODEL="$MODEL_DIR/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
PORT=8080
CTX=65536
RUNS="${RUNS:-2}"
NPREDICT=200
LEVELS="${LEVELS:-100 150 200 250}"
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUT/power_sweep_${STAMP}.csv"
ORIG_PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)

[[ -f "$MODEL" && -x "$SERVER" ]] || { echo "model or server missing"; exit 1; }

SWAP_WAS_ACTIVE=0; SRV_PID=""
cleanup() {
    echo ""; echo ">>> restoring power limit to ${ORIG_PL} W"
    sudo nvidia-smi -pl "$ORIG_PL" >/dev/null 2>&1
    [[ -n "$SRV_PID" ]] && { kill "$SRV_PID" 2>/dev/null; sleep 5; kill -9 "$SRV_PID" 2>/dev/null; }
    (( SWAP_WAS_ACTIVE )) && sudo systemctl start llama-swap 2>/dev/null
}
trap cleanup EXIT INT TERM

systemctl is-active --quiet llama-swap && { SWAP_WAS_ACTIVE=1; sudo systemctl stop llama-swap; sleep 3; }
for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${u:-9999}" -le 2000 ] && break; sleep 5; done

echo ">>> loading model once (power limit changes live afterwards)"
"$SERVER" --model "$MODEL" --alias pwr --fit on --ctx-size "$CTX" --ctx-checkpoints 0 \
  --reasoning-format deepseek --reasoning on --reasoning-budget 10000 \
  -fa on -ctk f16 -ctv f16 --parallel 1 --temp 1.0 --top-p 1.0 --min-p 0.0 \
  --threads 11 --threads-batch 11 --no-mmap \
  --host 127.0.0.1 --port "$PORT" --jinja > "$OUT/pwr_server_${STAMP}.log" 2>&1 &
SRV_PID=$!
for i in $(seq 1 180); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "server died"; tail -20 "$OUT/pwr_server_${STAMP}.log"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo ">>> ready after $((i*10))s"; break; }
    sleep 10
done

# ~1800-token prompt so prefill is actually exercised
LONG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog while considering the '
'implications of distributed systems design, consensus protocols, and the CAP theorem. ')*90)")

echo "phase,watt_limit,direction,prefill_tps,decode_tps,avg_draw_w,gpu_c" > "$CSV"

measure_at() {  # $1 = watt limit, $2 = direction
    local W=$1 DIR=$2
    sudo nvidia-smi -pl "$W" >/dev/null 2>&1
    sleep 8   # let the cap settle
    for r in $(seq 1 $RUNS); do
        ( for _ in $(seq 1 40); do nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits; sleep 1; done ) > /tmp/pw.$$ &
        MON=$!
        RESP=$(curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
          -d "$(python3 -c "
import json,sys
print(json.dumps({'prompt':open('/dev/stdin').read(),'n_predict':${NPREDICT},'cache_prompt':False}))" <<< "$LONG")")
        kill $MON 2>/dev/null
        read -r pp tg <<< "$(printf '%s' "$RESP" | python3 -c '
import json,sys
t=json.load(sys.stdin)["timings"]
print(t.get("prompt_per_second",0), t.get("predicted_per_second",0))')"
        read -r dw tc <<< "$(awk -F, '{p+=$1;t+=$2;n++} END{if(n)printf "%.1f %.0f",p/n,t/n}' /tmp/pw.$$ 2>/dev/null)"
        rm -f /tmp/pw.$$
        printf "  %4sW %-4s run%d  prefill %7.2f  decode %6.2f  draw %6s W  %s C\n" \
               "$W" "$DIR" "$r" "$pp" "$tg" "${dw:-?}" "${tc:-?}"
        echo "run,$W,$DIR,$pp,$tg,${dw:-0},${tc:-0}" >> "$CSV"
    done
}

echo ""; echo ">>> sweep UP"
for W in $LEVELS; do measure_at "$W" up; done
echo ""; echo ">>> sweep DOWN (cancels thermal drift)"
for W in $(echo $LEVELS | tr ' ' '\n' | tac | tr '\n' ' '); do measure_at "$W" down; done

echo ""
echo "=================================================================="
python3 - "$CSV" <<'PYEOF'
import csv,sys,collections,statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]))]
by=collections.defaultdict(lambda: {"pp":[],"tg":[],"w":[],"c":[]})
for r in rows:
    k=int(r["watt_limit"])
    by[k]["pp"].append(float(r["prefill_tps"])); by[k]["tg"].append(float(r["decode_tps"]))
    by[k]["w"].append(float(r["avg_draw_w"]));   by[k]["c"].append(float(r["gpu_c"]))
print(f"  {'cap':>5} {'prefill':>9} {'decode':>8} {'draw':>7} {'temp':>6} {'dec tok/W':>10} {'pre tok/W':>10}")
res={}
for k in sorted(by):
    d=by[k]; pp=st.mean(d["pp"]); tg=st.mean(d["tg"]); w=max(st.mean(d["w"]),1); c=st.mean(d["c"])
    res[k]=(pp,tg,w,c,tg/w,pp/w)
    print(f"  {k:>4}W {pp:9.2f} {tg:8.2f} {w:6.1f}W {c:5.0f}C {tg/w:10.4f} {pp/w:10.4f}")
print()
bd=max(res,key=lambda k:res[k][4]); bp=max(res,key=lambda k:res[k][5])
print(f"  BEST decode tokens/watt : {bd}W  ({res[bd][4]:.4f} tok/W, {res[bd][1]:.2f} tok/s)")
print(f"  BEST prefill tokens/watt: {bp}W  ({res[bp][5]:.4f} tok/W, {res[bp][0]:.2f} tok/s)")
print()
# agentic mix measured on the cowboy-clue soak: 34,607 prefill + 17,082 decode tokens
P,D=34607,17082
print("  === real agentic mix (34,607 prefill + 17,082 decode tokens) ===")
print(f"  {'cap':>5} {'job time':>10} {'energy':>9} {'tok/W overall':>14}")
best=None
for k in sorted(res):
    pp,tg,w,c,_,_=res[k]
    t=P/pp + D/tg; e=t*w/3600.0
    tw=(P+D)/(t*w)
    print(f"  {k:>4}W {t:9.0f}s {e:8.2f}Wh {tw:14.4f}")
    if best is None or tw>best[1]: best=(k,tw,t,e)
print()
print(f"  >>> BEST OVERALL tokens/watt: {best[0]}W  ({best[1]:.4f} tok/W, job {best[2]:.0f}s, {best[3]:.2f} Wh)")
lo=min(res); print(f"  >>> vs {lo}W: job time {100*(best[2]-(P/res[lo][0]+D/res[lo][1]))/(P/res[lo][0]+D/res[lo][1]):+.1f}%")
PYEOF
echo "=================================================================="
echo "csv: $CSV"
