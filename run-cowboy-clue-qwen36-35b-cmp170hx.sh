#!/usr/bin/env bash
# Cowboy-clue agentic build task: Qwen3.6-35B-A3B-MTP (unsloth) UD-Q4_K_XL, re-run on the
# unlocked CMP 170HX, for comparison against the existing row in
# /home/user/Downloads/COWBOY-CLUE-BENCHMARK-REPORT.md (row 6):
#
#   Old hardware (RTX PRO 4000 24GB, hybrid CPU+GPU via --fit, ik_llama.cpp, MTP n_max=1):
#     build 4.1m, 1 debug round ~1.5m, total ~5.6m, FIXED
#
# NOT a pure hardware-only comparison: both ik_llama.cpp builds on this box (build/ and
# -latest/) crash on this card with "ggml was not compiled with any CUDA arch <= 800" as soon
# as more than a token GPU-layer count is requested -- a real incompatibility, not a config
# error (confirmed via a minimal -ngl 1 load that worked, then -ngl 999 that core-dumped on
# both builds). So this run uses the mainline llama.cpp-portable build-cuda-sm80 (confirmed
# working earlier today: 98.57 tok/s plain decode, full offload) with NO MTP -- engine and
# speculative-decoding are both different from the old-hardware row. Same prompt.txt (copied
# byte-for-byte from that test folder). The ~22.85GB model fits entirely in the 64GB card
# (-ngl 999, no CPU spill).
#
# Continuous 2s-resolution thermal log for the whole run (load + build + any debug rounds).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNDIR="/home/user/cowboy-clue benchmark/runs/cowboy-clue-qwen36-35b-mtp-udq4-cmp170hx-test"
MODEL="/home/user/ai-models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
OCMODEL="direct-qwen36-35b-cmp170hx/qwen36-35b-a3b-mtp-cmp170hx"
PORT=8080
CTX=131072

mkdir -p "$RUNDIR"
[[ -f "$RUNDIR/prompt.txt" ]] || cp -f "/home/user/Downloads/cowboy-clue-qwen36-35b-mtp-udq4-test/prompt.txt" "$RUNDIR/prompt.txt"
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${RUNDIR}/server_${STAMP}.log"
TLOG="${RUNDIR}/thermal_${STAMP}.csv"
OLOG="${RUNDIR}/opencode_${STAMP}.log"

[[ -f "$MODEL"  ]] || { echo "ERROR: model missing"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build missing"; exit 1; }
command -v opencode >/dev/null || { echo "ERROR: opencode not on PATH"; exit 1; }

SWAP_WAS_ACTIVE=0
cleanup() {
    echo ""; echo ">>> cleanup"
    [[ -n "${MON_PID:-}" ]] && kill "$MON_PID" 2>/dev/null
    [[ -n "${SRV_PID:-}" ]] && { kill "$SRV_PID" 2>/dev/null; sleep 5; kill -9 "$SRV_PID" 2>/dev/null; }
    (( SWAP_WAS_ACTIVE )) && { echo ">>> restarting llama-swap"; sudo systemctl start llama-swap 2>/dev/null; }
}
trap cleanup EXIT INT TERM

if systemctl is-active --quiet llama-swap 2>/dev/null; then
    SWAP_WAS_ACTIVE=1; echo ">>> stopping llama-swap"; sudo systemctl stop llama-swap; sleep 3
fi

echo ">>> fan daemon: $(systemctl is-active gpu-fan-daemon.service 2>/dev/null)"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
(( USED > 2000 )) && { echo "ERROR: GPU busy (${USED} MiB)"; exit 1; }

echo ">>> starting llama-server (ctx ${CTX}, full GPU offload, no MTP -- ik_llama MTP is blocked on this card)"
"$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/Qwen3.6-35B-A3B-cmp170hx" \
    -ngl 999 \
    --ctx-size "$CTX" \
    -fa on -ctk q8_0 -ctv q8_0 \
    --parallel 1 \
    --threads 12 --threads-batch 12 \
    --reasoning-format deepseek --reasoning-budget 12000 \
    --host 127.0.0.1 --port "$PORT" \
    --no-mmap --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 90); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "ERROR: server died"; tail -25 "$SLOG"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo ">>> ready after $((i*10))s"; break; }
    printf "\r    loading %ds" "$((i*10))"; sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never ready"; tail -25 "$SLOG"; exit 1; }

LPID=$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo ">>> listener PID ${LPID:-?} (launched ${SRV_PID})"
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader

# ---- continuous sensor log, 2s resolution, for the entire run ----
echo "elapsed_s,gpu_c,power_w,util_pct,vram_mib" > "$TLOG"
(
  T0=$(date +%s)
  while true; do
    G=$(nvidia-smi -i 0 --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$(( $(date +%s) - T0 )),${G}" >> "$TLOG"
    sleep 2
  done
) & MON_PID=$!

echo ""
echo ">>> starting opencode build task (old-hardware baseline: build 4.1m + debug ~1.5m = ~5.6m)"
echo ">>> thermal abort: GPU>=80C"
START=$(date +%s)

PROMPT="$(cat "$RUNDIR/prompt.txt")"
ABORT_GPU=80 ABORT_MEM=88 RISE_DELTA=12 POLL=5 \
  "${HERE}/thermal-guard.sh" \
  opencode run --dir "$RUNDIR" --model "$OCMODEL" "$PROMPT" 2>&1 | tee "$OLOG"
RC=${PIPESTATUS[0]}
END=$(date +%s); ELAPSED=$((END-START))

kill "$MON_PID" 2>/dev/null; MON_PID=""

echo ""
echo "=================================================================="
printf "wall-clock : %ds (%dm %ds)\n" "$ELAPSED" $((ELAPSED/60)) $((ELAPSED%60))
echo "guard exit : $RC  (1 = THERMAL ABORT)"
echo ""
python3 - "$TLOG" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
def col(n):
    return [float(r[n]) for r in rows if r.get(n) not in (None,'')]
for name,label in [('gpu_c','GPU core'),('power_w','power W'),('util_pct','GPU util%')]:
    v=col(name)
    if v: print(f"  {label:10s} min {min(v):6.1f}   max {max(v):6.1f}   avg {sum(v)/len(v):6.1f}")
print(f"\n  samples: {len(rows)}  (2s resolution)")
PY
echo ""
ls -la "$RUNDIR"/*.html 2>/dev/null
echo ""
echo "OLD HARDWARE (RTX PRO 4000 24GB, hybrid, ik_llama.cpp MTP n_max=1): build 4.1m + debug ~1.5m = ~5.6m, FIXED"
echo "=================================================================="
echo "thermal csv: $TLOG"
echo "server log : $SLOG"
echo "opencode log: $OLOG"
