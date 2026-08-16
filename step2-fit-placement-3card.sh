#!/usr/bin/env bash
# Step 2 of the 3rd-CMP-170HX brainstorm: no-spec, --fit on, small ctx, 3-card auto placement.
# Goal: see how many (if any) expert layers --fit still leaves on CPU now that VRAM is 192GB
# (up from 128GB with 2 cards). No DSpark yet -- that has a known-broken drafter (see
# BENCHMARK-RESULTS.md "2026-08-14" entry) and is a separate step.
set -uo pipefail

MODEL="/home/user/models/models/unsloth-DeepSeek-V4-Flash-0731/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
SERVER="/home/user/CMP-170HX-PROJECT/llama-cpp-dsv4-fresh/build/bin/llama-server"
PORT=8080
CTX=8192
OUT="/home/user/CMP-170HX-PROJECT/bench-results"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${OUT}/deepseek-v4-flash-3card-nospec-fit_${STAMP}.log"

[[ -f "$MODEL"  ]] || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: server not found: $SERVER"; exit 1; }

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
echo ">>> GPU memory currently in use (MiB, per card): $USED"

echo ">>> launching llama-server (172GB across 5 shards; load may take a while on these cards' PCIe link)"
"$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/DeepSeek-V4-Flash-0731-UD-Q8_K_XL" \
    --fit on \
    --ctx-size "$CTX" \
    -fa on -ctk f16 -ctv f16 \
    --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0 \
    --threads 24 --threads-batch 24 \
    --no-mmap \
    --host 127.0.0.1 --port "$PORT" \
    --jinja > "$SLOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > /tmp/deepseek-v4-step2.pid
echo ">>> server PID $SRV_PID, logging to $SLOG"
