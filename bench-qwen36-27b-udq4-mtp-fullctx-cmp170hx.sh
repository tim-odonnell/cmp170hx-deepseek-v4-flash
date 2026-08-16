#!/usr/bin/env bash
# Benchmark Qwen3.6-27B UD-Q4_K_XL (unsloth) MTP, FULL NATIVE CONTEXT (262144), on the
# unlocked CMP 170HX (64GB, sm_80). Never tested at full context before on any hardware on
# this box -- the old RTX PRO 4000 24GB scripts capped this model at 57344 (mtp profile) or
# 147456 (maxctx profile) because full context didn't fit in 24GB alongside the model.
#
# Why it fits now: Qwen3.6-27B is a HYBRID architecture -- 65 blocks total (64 real + 1 nextn
# MTP head), but only 16 of the 64 are quadratic-attention layers that grow a KV cache; the
# rest use linear "Gated Delta Net" attention with fixed-size state (per this project's own
# architecture notes in cmp-170x-performance-benchmarks.md). Confirmed via GGUF metadata:
# head_count_kv=4, key/value_length=256 each. KV math for the 16 growing layers only, q8_0:
#   16 layers * 2(K+V) * 4 kv_heads * 256 dim * 262144 ctx (~1 byte/elem at q8_0) ~= 8.6 GiB
# vs ~17GB model weights -- both fit in 64GB with ~38GB to spare. No --fit/hybrid needed.
#
# Engine: mainline llama.cpp-portable build-cuda-sm80 (commit 6ea215d) -- ik_llama.cpp is
# blocked on this card for anything past a token GPU-layer count (see project memory /
# sibling Qwen3.6-35B-A3B scripts for the "ggml was not compiled with any CUDA arch <= 800"
# detail). MTP via mainline's own flags: --spec-type draft-mtp --spec-draft-n-max 2 --
# Unsloth's own recommended MTP setting (draft=2 is their measured sweet spot for this model;
# they warn against going higher). Sampling: --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00,
# matching both Unsloth's docs and this GGUF's own baked-in sampling metadata.
#
# GPU temp/power sampled once per second for the whole run (load + inference).
set -uo pipefail

MODEL="/home/user/ai-models/unsloth-qwen3.6-27b-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
PORT=8080
CTX=262144
RUNS=3
NPREDICT=300
OUT="$(dirname "$0")/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${OUT}/qwen36-27b-mtp-fullctx-server_${STAMP}.log"
TEMPLOG="${OUT}/qwen36-27b-mtp-fullctx-temp_${STAMP}.csv"

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

echo ">>> launching llama-server (full GPU offload, ctx ${CTX} = full native, MTP draft=2)"
"$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/Qwen3.6-27B-UD-Q4_K_XL-mtp-fullctx" \
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

echo ">>> waiting for server (PID $SRV_PID) -- full-ctx KV alloc may take longer than usual"
for i in $(seq 1 90); do    # up to 15 min
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR: server died during load. Tail of log:"; tail -40 "$SLOG"; exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo ">>> server ready after $((i*10))s"; break
    fi
    CUR_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    printf "\r    loading... %ds  GPU temp: %sC" "$((i*10))" "$CUR_TEMP"
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never became ready"; tail -40 "$SLOG"; exit 1; }

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
    echo "$r,$pp,$tg,$CUR_TEMP,$CUR_PWR" >> "${OUT}/qwen36-27b-mtp-fullctx-runs_${STAMP}.csv"
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
echo "OLD HARDWARE (RTX PRO 4000 24GB, ik_llama MTP, capped ctx -- never ran full 262144): 57344-ctx profile ~35 tok/s, 147456-ctx profile ~26 tok/s"
echo "=================================================================="
echo "server log: $SLOG"
echo "temp log:   $TEMPLOG"
