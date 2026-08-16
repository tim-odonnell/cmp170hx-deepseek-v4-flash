#!/usr/bin/env bash
# Benchmark DeepSeek-V4-Flash-0731 UD-Q8_K_XL hybrid CPU+GPU on the unlocked CMP 170HX.
#
# WHAT THIS IS TESTING
# --------------------
# Two things changed since the 9.37 tok/s baseline:
#   1. GPU: 1x RTX PRO 4000 Blackwell 24GB  ->  1x CMP 170HX 64GB (unlocked)
#   2. RAM: 4x64GB DDR5 @ 3600 MT/s (derated) -> 2x64GB DDR5 @ 5600 MT/s
#
# (2) is the one predicted to matter. This workload is DDR5-bandwidth-bound: the model is
# ~151 GiB and only ~55 GiB of it fits in VRAM, so most weights stay in system RAM and the
# memory controller sets the pace. Measured read bandwidth went 37 GB/s (inferred) -> 58 GB/s.
# Prediction on record: 9.4 -> 12-14 tok/s.
#
# Settings are matched to the baseline run EXACTLY (fp16 KV, ctx 65536, threads 11,
# --fit on, same sampling) so the hardware is the only variable.
#
# NOTE: needs the sm_80 build. The production build-cuda/ tree is sm_120 (Blackwell) and
# physically cannot run on this Ampere-based card.
set -uo pipefail

MODEL="/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
PORT=8080
CTX=65536
RUNS=3
NPREDICT=300
OUT="$(dirname "$0")/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${OUT}/server_${STAMP}.log"

[[ -f "$MODEL"  ]] || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build not found: $SERVER"; exit 1; }

SWAP_WAS_ACTIVE=0
cleanup() {
    echo ""
    echo ">>> cleaning up"
    [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null && sleep 5
    kill -9 "${SRV_PID:-}" 2>/dev/null
    if (( SWAP_WAS_ACTIVE )); then
        echo ">>> restarting llama-swap"
        sudo systemctl start llama-swap 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# llama-swap owns the GPU on :18888 and will happily swap a model in underneath a running
# standalone benchmark -- a known way to silently corrupt results on this box. Stop it.
if systemctl is-active --quiet llama-swap 2>/dev/null; then
    SWAP_WAS_ACTIVE=1
    echo ">>> stopping llama-swap for the duration of the benchmark"
    sudo systemctl stop llama-swap
    sleep 3
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo ">>> GPU currently using ${USED} MiB"
(( USED > 2000 )) && { echo "ERROR: GPU not free -- something else is holding VRAM"; exit 1; }

echo ">>> launching llama-server (model is ~151 GiB, load takes several minutes)"
"$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/DeepSeek-V4-Flash-0731-UD-Q8_K_XL" \
    --fit on \
    --ctx-size "$CTX" \
    --ctx-checkpoints 0 \
    --reasoning-format deepseek \
    --reasoning on \
    --reasoning-budget 10000 \
    -fa on -ctk f16 -ctv f16 \
    --parallel 1 \
    --temp 1.0 --top-p 1.0 --min-p 0.0 \
    --threads 11 --threads-batch 11 \
    --no-mmap \
    --host 127.0.0.1 --port "$PORT" \
    --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

echo ">>> waiting for server (PID $SRV_PID)"
for i in $(seq 1 180); do    # up to 30 min
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR: server died during load. Tail of log:"; tail -25 "$SLOG"; exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo ">>> server ready after $((i*10))s"; break
    fi
    printf "\r    loading... %ds  RAM used: %s" "$((i*10))" "$(free -g | awk '/^Mem:/{print $3"Gi"}')"
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never became ready"; tail -25 "$SLOG"; exit 1; }

# Confirm the PID we launched actually owns the listener -- a stale server on this port
# has faked a whole benchmark on this box before.
LPID=$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo ">>> listener on :${PORT} owned by PID ${LPID:-?} (launched ${SRV_PID})"

echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total,power.draw --format=csv,noheader
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
    printf "    prefill %8.2f tok/s   decode %8.2f tok/s\n" "$pp" "$tg"
    PP+=("$pp"); TG+=("$tg")
    echo "$r,$pp,$tg" >> "${OUT}/runs_${STAMP}.csv"
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
echo "BASELINE (1x RTX PRO 4000 24GB, 4x64GB DDR5-3600, fp16 KV @ 65536): 9.37 tok/s"
echo "PREDICTION on record for the RAM upgrade                          : 12-14 tok/s"
echo "=================================================================="
echo "server log: $SLOG"
