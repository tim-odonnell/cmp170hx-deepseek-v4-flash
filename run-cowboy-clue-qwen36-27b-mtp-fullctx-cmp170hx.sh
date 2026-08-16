#!/usr/bin/env bash
# Cowboy-clue agentic build task: Qwen3.6-27B UD-Q4_K_XL, MTP draft=2, FULL NATIVE CONTEXT
# (262144), on the unlocked CMP 170HX. Same canonical prompt.txt as every other row in
# COWBOY-CLUE-BENCHMARK-REPORT.md.
#
# Server config matches bench-qwen36-27b-udq4-mtp-fullctx-cmp170hx.sh exactly (same model,
# same full 262144 ctx, same MTP draft=2, same sampling) -- see that script's header for the
# KV-cache sizing math (only 16 of 65 blocks grow a KV cache; hybrid linear+quadratic
# attention architecture) and the ik_llama.cpp-is-blocked-on-this-card rationale for using
# mainline llama.cpp-portable build-cuda-sm80 instead.
#
# This is the first cowboy-clue run for Qwen3.6-27B on the CMP 170HX (rows 13/14 in the
# report were Qwen3.6-35B-A3B). Old-hardware analog: row 7 (Qwen3.6-27B MTP UD-Q4 on RTX PRO
# 4000) -- but that row ran at a much smaller context (24GB card couldn't fit 262144), so this
# is not a clean hardware-only comparison either.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNDIR="/home/user/cowboy-clue benchmark/runs/cowboy-clue-qwen36-27b-mtp-udq4-fullctx-cmp170hx-test"
MODEL="/home/user/ai-models/unsloth-qwen3.6-27b-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
OCMODEL="direct-qwen36-27b-cmp170hx/qwen36-27b-mtp-fullctx-cmp170hx"
PORT=8080
CTX=262144

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

echo ">>> starting llama-server (ctx ${CTX} = full native, full GPU offload, MTP draft=2)"
"$SERVER" \
    --model "$MODEL" \
    --alias "unsloth/Qwen3.6-27B-UD-Q4_K_XL-mtp-fullctx-cmp170hx" \
    -ngl 999 \
    --ctx-size "$CTX" \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    -fa on -ctk q8_0 -ctv q8_0 \
    --parallel 1 \
    --threads 12 --threads-batch 12 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --reasoning-format deepseek --reasoning-budget 12000 \
    --host 127.0.0.1 --port "$PORT" \
    --no-mmap --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 90); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "ERROR: server died"; tail -40 "$SLOG"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo ">>> ready after $((i*10))s"; break; }
    printf "\r    loading %ds" "$((i*10))"; sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never ready"; tail -40 "$SLOG"; exit 1; }

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
echo ">>> starting opencode build task"
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
echo "OLD HARDWARE ANALOG (row 7, RTX PRO 4000, MTP, much smaller ctx -- 24GB couldn't fit 262144): 62.6m build, 1 debug round, FIXED"
echo "=================================================================="
echo "thermal csv: $TLOG"
echo "server log : $SLOG"
echo "opencode log: $OLOG"
