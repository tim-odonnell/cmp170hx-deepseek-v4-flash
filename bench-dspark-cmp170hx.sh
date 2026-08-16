#!/usr/bin/env bash
# DSpark speculative-decoding benchmark on the unlocked CMP 170HX.
#
# Compares against the non-speculative baseline measured on identical hardware/settings:
#   decode 13.65 tok/s | prefill 36.59 tok/s   (ctx 65536, fp16 KV, 3x 300-tok runs)
#
# Everything except the speculative flags is held identical to that run, so spec decoding
# is the only variable.
#
# ADAPTATIONS from ai-models/.../start-deepseek-v4-flash-dspark.sh, which cannot run here:
#   * that script uses build-cuda/ = sm_120 (Blackwell). This card is sm_80 -> build-cuda-sm80/
#   * that script has --override-tensor "...=CUDA1,exps=CPU". There is no CUDA1 any more
#     (single GPU). Dropped; --fit bin-packs the one 64GB card instead.
#
# TUNING NOTES (from prior DSpark work on this box):
#   * --spec-draft-n-max is NOT a fixed optimum: 5 is fastest at ~0 context, 2 wins at depth,
#     and 5 is actually SLOWER than no-spec by 100k. This benchmark uses a short prompt, i.e.
#     ~0 depth, so both are swept rather than assumed.
#   * --spec-draft-p-min is deliberately NOT set: it means something different for dspark than
#     for MTP and is a net loss at --parallel 1.
#   * --top-k 0 matters and is easy to miss. The non-spec baseline did not set it (llama.cpp
#     defaults to 40), so this is the one sampling difference vs that run -- noted, not hidden.
#   * acceptance rate is not a speed proxy; judge on tok/s.
#   * --fit-target 12288 IS LOAD-BEARING. --fit cannot measure the dspark draft model
#     ("[spec] failed to measure draft model memory ... dflash requires ctx_other to be set"),
#     so it packs the main model into all 64GB and leaves only --fit-target's default 1024 MiB.
#     The drafter then needs 10,386 MiB and dies with cudaMalloc OOM. Reserving 12 GiB fixes it,
#     at the cost of ~11 GB less of the main model being GPU-resident.
set -uo pipefail

MODEL_DIR=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731
MODEL="$MODEL_DIR/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
DRAFT="$MODEL_DIR/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
PORT=8080
CTX=65536
RUNS="${RUNS:-3}"
NPREDICT=300
SWEEP="${SWEEP:-2 5}"
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)

[[ -f "$MODEL"  ]] || { echo "ERROR: model missing"; exit 1; }
[[ -f "$DRAFT"  ]] || { echo "ERROR: draft model missing: $DRAFT"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build missing"; exit 1; }

SWAP_WAS_ACTIVE=0
SRV_PID=""
cleanup() {
    [[ -n "$SRV_PID" ]] && { kill "$SRV_PID" 2>/dev/null; sleep 5; kill -9 "$SRV_PID" 2>/dev/null; }
    (( SWAP_WAS_ACTIVE )) && { echo ">>> restarting llama-swap"; sudo systemctl start llama-swap 2>/dev/null; }
}
trap cleanup EXIT INT TERM

if systemctl is-active --quiet llama-swap 2>/dev/null; then
    SWAP_WAS_ACTIVE=1; echo ">>> stopping llama-swap"; sudo systemctl stop llama-swap; sleep 3
fi

run_one() {
    local NMAX=$1
    local SLOG="${OUT}/dspark_n${NMAX}_${STAMP}.log"
    echo ""
    echo "=================================================================="
    echo " DSpark  --spec-draft-n-max ${NMAX}"
    echo "=================================================================="
    for i in $(seq 1 30); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${u:-9999}" -le 2000 ] && break; sleep 5
    done

    "$SERVER" \
        --model "$MODEL" \
        --model-draft "$DRAFT" \
        --spec-type draft-dspark,ngram-mod \
        --spec-draft-n-max "$NMAX" \
        --spec-ngram-mod-n-match 60 \
        --spec-ngram-mod-n-min 12 \
        --spec-ngram-mod-n-max 24 \
        --alias "deepseek-v4-flash-0731-dspark" \
        --fit on \
        --fit-target 12288 \
        --n-gpu-layers-draft 999 \
        --ctx-size "$CTX" --ctx-checkpoints 0 \
        --reasoning-format deepseek --reasoning on --reasoning-budget 10000 \
        -fa on -ctk f16 -ctv f16 --parallel 1 \
        --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
        --threads 11 --threads-batch 11 \
        --no-mmap --host 127.0.0.1 --port "$PORT" --jinja > "$SLOG" 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 120); do
        kill -0 "$SRV_PID" 2>/dev/null || { echo "  SERVER DIED. tail:"; tail -20 "$SLOG"; SRV_PID=""; return 1; }
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "  ready after $((i*10))s"; break; }
        sleep 10
    done
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "  never ready"; tail -20 "$SLOG"; return 1; }

    nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader | sed 's/^/  VRAM: /'
    free -h | awk '/^Mem:/{print "  host RAM: "$3" of "$2}'

    local P="Write a detailed technical explanation of how a singly linked list differs from a doubly linked list, including insertion and deletion complexity."
    local tgs=()
    for r in $(seq 1 $RUNS); do
        RESP=$(curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
          -d "{\"prompt\":$(printf '%s' "$P" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"n_predict\":${NPREDICT},\"cache_prompt\":false}")
        read -r pp tg <<< "$(printf '%s' "$RESP" | python3 -c '
import json,sys
t=json.load(sys.stdin)["timings"]
print(t.get("prompt_per_second",0), t.get("predicted_per_second",0))')"
        printf "  run %d: prefill %8.2f  decode %8.2f tok/s\n" "$r" "$pp" "$tg"
        tgs+=("$tg"); echo "$NMAX,$r,$pp,$tg" >> "${OUT}/dspark_sweep_${STAMP}.csv"
    done
    printf '%s\n' "${tgs[@]}" | python3 -c "
import sys
v=[float(x) for x in sys.stdin if x.strip()]
if v:
    a=sum(v)/len(v)
    print(f'  --> n_max=${NMAX}: decode avg {a:6.2f} tok/s   vs 13.65 no-spec = {100*(a-13.65)/13.65:+5.1f}%')
"
    # acceptance stats, if the build reports them
    grep -oiE 'accept[^,]*|n_drafted[^,]*|draft accept[^,]*' "$SLOG" | tail -3 | sed 's/^/  /'
    kill "$SRV_PID" 2>/dev/null; sleep 6; kill -9 "$SRV_PID" 2>/dev/null; SRV_PID=""
}

for n in $SWEEP; do run_one "$n" || echo "  (n_max=$n failed, continuing)"; done

echo ""
echo "=================================================================="
echo " SUMMARY   (baseline, no speculation: 13.65 tok/s decode)"
python3 - "${OUT}/dspark_sweep_${STAMP}.csv" <<'PY'
import csv,sys,collections
try: rows=list(csv.reader(open(sys.argv[1])))
except FileNotFoundError: print("  no results"); raise SystemExit
d=collections.defaultdict(list)
for r in rows:
    if len(r)>=4: d[r[0]].append(float(r[3]))
for k in sorted(d,key=int):
    v=d[k]; a=sum(v)/len(v)
    print(f"  n_max={k:>2}  decode {a:6.2f} tok/s  ({100*(a-13.65)/13.65:+5.1f}% vs no-spec)  runs={len(v)}")
PY
echo "=================================================================="
