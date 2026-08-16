#!/usr/bin/env bash
# llama-swap-managed variant of start-qwen36-35b-a3b-mtp-fullctx-cmp170hx.sh.
#
# 2026-08-07: the original script is a standalone-only daemon launcher -- it (a) calls
# `sudo systemctl stop llama-swap` on itself as its first action, and (b) backgrounds the real
# server via nohup+disown then lets the wrapper script exit after a health check. Both are fine
# for manual/standalone use but fatal under llama-swap, which expects its `cmd:` process to BE
# the server and stay in the foreground: (a) killed llama-swap's own parent process the instant
# it ran, and (b) even without that, the wrapper exiting after backgrounding looks to llama-swap
# exactly like "upstream command exited prematurely" (confirmed in llama-swap.log). This variant
# fixes both: no self-stop, and `exec` replaces the shell with llama-server directly, matching
# the pattern every other working llama-swap entry in this project uses (see
# start-deepseek-v4-flash.sh). Same model/serving flags as the original, otherwise unchanged.
set -uo pipefail

MODEL="/home/user/ai-models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"

[[ -f "$MODEL"  ]] || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build not found: $SERVER"; exit 1; }

# ── Kill any standalone model holding the lock ───────────────
LOCK_PID=$(lsof -t /tmp/llama-server.lock 2>/dev/null | head -1)
if [ -n "$LOCK_PID" ]; then
    echo "Stopping running model (PID $LOCK_PID) to free VRAM..."
    kill "$LOCK_PID"
    sleep 5
fi

/home/user/ai-models/gpu-guard.sh || exit 1

LOCK=/tmp/llama-server.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "ERROR: Another model is already running: $(cat $LOCK 2>/dev/null)"
    echo "Stop it first:  pkill -f llama-server"
    exit 1
fi
echo "Qwen3.6-35B-A3B-MTP UD-Q4_K_XL fullctx (port ${PORT:-8080})" >&9

exec "$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL-mtp-fullctx" \
    -ngl 999 \
    --ctx-size 262144 \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    -fa on -ctk q8_0 -ctv q8_0 \
    --parallel 1 \
    --threads 12 --threads-batch 12 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --reasoning-format deepseek \
    --reasoning-budget 12000 \
    --no-mmap \
    --host 0.0.0.0 --port "${PORT:-8080}" \
    --jinja
