#!/usr/bin/env bash
# Persistent server: Qwen3.6-35B-A3B UD-Q4_K_XL (unsloth) + MTP, FULL NATIVE CONTEXT (262144),
# on the unlocked CMP 170HX #3 (65536 MiB, sm_80). Unlike bench-qwen36-35b-a3b-udq4-mtp-cmp170hx.sh
# (CTX=32768, runs 3 prompts then exits), this stays up for interactive use.
#
# Fits fully offloaded with room to spare: this GGUF has quadratic self-attention on ALL 41
# blocks (not hybrid like Qwen3.6-27B) -- confirmed via GGUF tensor dump, every blk.N has its
# own attn_qkv tensor alongside the ssm_* (Gated Delta Net) tensors, head_count_kv=2,
# key/value_length=256 each. KV math at q8_0/q8_0, full 262144 ctx:
#   41 layers * 2(K+V) * 2 kv_heads * 256 dim * 262144 ctx * ~1 byte/elem ~= 10.3 GiB
# vs ~22.9GB model weights -- both fit in 65.5GB with ~30GB to spare. No --fit/hybrid needed.
#
# Engine: mainline llama.cpp-portable build-cuda-sm80 (ik_llama.cpp core-dumps on this card,
# see project memory). MTP: --spec-type draft-mtp --spec-draft-n-max 2, matching this card's
# own measured winner (132.47 tok/s @ CTX=32768, +34.4% vs no MTP) and Unsloth's own
# recommended setting for this model. Sampling matches Unsloth's docs + this GGUF's baked-in
# sampling metadata.
set -uo pipefail

MODEL="/home/user/ai-models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
PORT="${PORT:-8080}"
CTX=262144
OUT="$(dirname "$0")/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${OUT}/qwen36-35b-mtp-fullctx-server_${STAMP}.log"

[[ -f "$MODEL"  ]] || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build not found: $SERVER"; exit 1; }

if systemctl is-active --quiet llama-swap 2>/dev/null; then
    echo ">>> stopping llama-swap (would otherwise fight this standalone server for the GPU)"
    sudo systemctl stop llama-swap
    sleep 3
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo ">>> GPU currently using ${USED} MiB"
(( USED > 2000 )) && { echo "ERROR: GPU not free -- something else is holding VRAM"; exit 1; }

echo ">>> launching llama-server (full GPU offload, MTP draft=2, ctx=${CTX})"
nohup "$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL-mtp-fullctx" \
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
disown

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
if [[ "$LPID" != "$SRV_PID" ]]; then
    echo "ERROR: listener on :${PORT} is PID ${LPID:-?}, not the server we launched (${SRV_PID}). Refusing to claim success."
    exit 1
fi
echo ">>> listener on :${PORT} confirmed owned by PID ${SRV_PID}"

echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
free -h | awk '/^Mem:/{print "host RAM used: "$3" of "$2}'

echo ""
echo ">>> sanity generation:"
curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","n_predict":16,"cache_prompt":false}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))'

echo ""
echo "=================================================================="
echo "Server is UP and staying up: http://127.0.0.1:${PORT}  (PID ${SRV_PID})"
echo "  ctx-size: ${CTX} (full native context)"
echo "  log: $SLOG"
echo "Stop it with:  kill ${SRV_PID}"
echo "=================================================================="
