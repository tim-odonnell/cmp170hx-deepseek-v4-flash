#!/usr/bin/env bash
# Benchmark Qwen3.6-35B-A3B UD-Q4_K_XL (unsloth) on the unlocked CMP 170HX, WITH MTP enabled,
# using Unsloth's own suggested MTP settings from their docs (Qwen3.6 - How to Run Locally,
# "MTP Benchmarks" section, p.28-29 of the PDF in this model's folder):
#   - draft tokens = 2 (their measured sweet spot for this exact model: ~1.15-1.25x speedup,
#     peaking at ~220 tok/s in their own testing). They explicitly warn against going higher:
#     acceptance rate drops from 83% to 50% at draft=4, making MTP less beneficial past 2.
#   - sampling: --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 (their general-tasks preset
#     for Qwen3.6-35B-A3B, thinking mode default-enabled).
#
# Engine: mainline llama.cpp-portable build-cuda-sm80 (commit 6ea215d, built 2026-08-05/06).
# This is NOT the engine used for the original cowboy-clue baseline (ik_llama.cpp) -- both
# ik_llama.cpp trees on this box core-dump on this card with "ggml was not compiled with any
# CUDA arch <= 800" (missing -DCMAKE_CUDA_ARCHITECTURES=80 at their build time). This mainline
# build already proved sm_80-clean earlier today (98.57 tok/s decode, no MTP, 0 crashes), so
# it's the one used here. Flag names differ from ik_llama.cpp's older API:
# --spec-type draft-mtp (not "mtp"), --spec-draft-n-max (not "--draft-max").
#
# Direct comparison target: this box's own no-MTP baseline from earlier today (same model,
# same engine, same full GPU offload): decode 98.57 tok/s avg, prefill 222.26 tok/s avg,
# GPU 32-38C.
#
# GPU temp is sampled once per second for the whole run (load + inference).
set -uo pipefail

MODEL="/home/user/ai-models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
PORT=8080
CTX=32768
RUNS=3
NPREDICT=300
OUT="$(dirname "$0")/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${OUT}/qwen36-35b-mtp-server_${STAMP}.log"
TEMPLOG="${OUT}/qwen36-35b-mtp-temp_${STAMP}.csv"

[[ -f "$MODEL"  ]] || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build not found: $SERVER"; exit 1; }

SWAP_WAS_ACTIVE=0
TEMP_PID=""
cleanup() {
    echo ""
    echo ">>> cleaning up"
    [[ -n "${TEMP_PID}" ]] && kill "$TEMP_PID" 2>/dev/null
    [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null && sleep 5
    kill -9 "${SRV_PID:-}" 2>/dev/null
    if (( SWAP_WAS_ACTIVE )); then
        echo ">>> restarting llama-swap"
        sudo systemctl start llama-swap 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

if systemctl is-active --quiet llama-swap 2>/dev/null; then
    SWAP_WAS_ACTIVE=1
    echo ">>> stopping llama-swap for the duration of the benchmark"
    sudo systemctl stop llama-swap
    sleep 3
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo ">>> GPU currently using ${USED} MiB"
(( USED > 2000 )) && { echo "ERROR: GPU not free -- something else is holding VRAM"; exit 1; }

echo "timestamp,temp_c,power_w,mem_used_mib" > "$TEMPLOG"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used \
            --format=csv,noheader,nounits >> "$TEMPLOG"
        sleep 1
    done
) &
TEMP_PID=$!
echo ">>> temp sampler running (PID $TEMP_PID) -> $TEMPLOG"

echo ">>> launching llama-server (full GPU offload, MTP draft=2, Unsloth sampling preset)"
"$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL-mtp" \
    -ngl 999 \
    --ctx-size "$CTX" \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    -fa on -ctk q8_0 -ctv q8_0 \
    --parallel 1 \
    --threads 12 --threads-batch 12 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --reasoning-format deepseek \
    --reasoning-budget 12000 \
    --no-mmap \
    --host 127.0.0.1 --port "$PORT" \
    --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

echo ">>> waiting for server (PID $SRV_PID)"
for i in $(seq 1 90); do    # up to 15 min
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR: server died during load. Tail of log:"; tail -25 "$SLOG"; exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo ">>> server ready after $((i*10))s"; break
    fi
    CUR_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    printf "\r    loading... %ds  GPU temp: %sC" "$((i*10))" "$CUR_TEMP"
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never became ready"; tail -25 "$SLOG"; exit 1; }

LPID=$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo ">>> listener on :${PORT} owned by PID ${LPID:-?} (launched ${SRV_PID})"

echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
free -h | awk '/^Mem:/{print "host RAM used: "$3" of "$2}'
echo ""

PROMPT="Write a detailed technical explanation of how a singly linked list differs from a doubly linked list, including insertion and deletion complexity."
declare -a TG PP
for r in $(seq 1 $RUNS); do
    echo ">>> run ${r}/${RUNS}"
    RESP=$(curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\":$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"n_predict\":${NPREDICT},\"cache_prompt\":false}")
    read -r pp tg <<< "$(printf '%s' "$RESP" | python3 -c '
import json,sys
t=json.load(sys.stdin)["timings"]
print(t.get("prompt_per_second",0), t.get("predicted_per_second",0))')"
    CUR_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    CUR_PWR=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    printf "    prefill %8.2f tok/s   decode %8.2f tok/s   GPU %sC %sW\n" "$pp" "$tg" "$CUR_TEMP" "$CUR_PWR"
    PP+=("$pp"); TG+=("$tg")
    echo "$r,$pp,$tg,$CUR_TEMP,$CUR_PWR" >> "${OUT}/qwen36-35b-mtp-runs_${STAMP}.csv"
done

echo ""
echo "=================================================================="
printf '%s\n' "${TG[@]}" | python3 -c '
import sys
v=[float(x) for x in sys.stdin if x.strip()]
if v:
    print(f"decode  avg {sum(v)/len(v):6.2f} tok/s   min {min(v):6.2f}   max {max(v):6.2f}")
'
printf '%s\n' "${PP[@]}" | python3 -c '
import sys
v=[float(x) for x in sys.stdin if x.strip()]
if v:
    print(f"prefill avg {sum(v)/len(v):6.2f} tok/s")
'
echo ""
echo "temp/power summary over whole run:"
python3 -c "
import csv
rows=list(csv.DictReader(open('$TEMPLOG')))
if rows:
    temps=[float(r['temp_c']) for r in rows]
    pwrs=[float(r['power_w']) for r in rows]
    print(f'  temp  min {min(temps):5.1f}C  max {max(temps):5.1f}C  avg {sum(temps)/len(temps):5.1f}C  (n={len(temps)} samples)')
    print(f'  power min {min(pwrs):5.1f}W  max {max(pwrs):5.1f}W  avg {sum(pwrs)/len(pwrs):5.1f}W')
"
echo ""
echo "NO-MTP BASELINE (this box, same engine, same model, earlier today): decode 98.57 tok/s avg, prefill 222.26 tok/s avg, GPU 32-38C"
echo "=================================================================="
echo "server log: $SLOG"
echo "temp log:   $TEMPLOG"
