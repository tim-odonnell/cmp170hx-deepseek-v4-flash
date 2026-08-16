#!/usr/bin/env bash
# Long-duration soak test: the "cowboy clue" agentic build task on the unlocked CMP 170HX,
# with continuous thermal logging and an automatic abort if the card ever gets hot.
#
# Direct comparison target -- the same prompt, same model, same harness, on the OLD hardware:
#   2026-07-31, 2x RTX PRO 4000 Blackwell 24GB, 249GB DDR5-3600
#   wall-clock 2965 s (49m 25s), decode 9.26 tok/s, output 44,092 bytes / 1,211 lines
#
# This run: 1x CMP 170HX 64GB (unlocked), 128GB DDR5-5600.
#
# Thermal safety: opencode runs under thermal-guard.sh (abort at 80C core / 88C HBM), the
# gpu-fan-daemon is expected to be live, and a 2s-resolution CSV of every sensor is written
# for the whole run so the thermal behaviour can be reviewed afterwards rather than trusted.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNDIR="/home/user/cowboy-clue benchmark/runs/cowboy-clue-cmp170hx"
PROMPT_SRC="/home/user/cowboy-clue benchmark/runs/cowboy-clue-deepseek-v4-flash-2gpu-test/prompt.txt"
MODEL="/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
OCMODEL="direct-deepseek-2gpu/deepseek-v4-flash-2gpu"
PORT=8080
CTX=720896

mkdir -p "$RUNDIR"
cp -f "$PROMPT_SRC" "$RUNDIR/prompt.txt"
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

echo ">>> starting llama-server (ctx ${CTX}, fp16 KV)"
"$SERVER" --model "$MODEL" \
    --alias "deepseek-v4-flash-2gpu" \
    --fit on --ctx-size "$CTX" --ctx-checkpoints 0 \
    --reasoning-format deepseek --reasoning on --reasoning-budget 10000 \
    -fa on -ctk f16 -ctv f16 --parallel 1 \
    --temp 1.0 --top-p 1.0 --min-p 0.0 \
    --threads 11 --threads-batch 11 \
    --no-mmap \
    --host 127.0.0.1 --port "$PORT" --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 180); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "ERROR: server died"; tail -25 "$SLOG"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo ">>> ready after $((i*10))s"; break; }
    printf "\r    loading %ds" "$((i*10))"; sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never ready"; tail -25 "$SLOG"; exit 1; }

LPID=$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo ">>> listener PID ${LPID:-?} (launched ${SRV_PID})"
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,temperature.memory,power.draw --format=csv,noheader

# ---- continuous sensor log, 2s resolution, for the entire run ----
echo "elapsed_s,gpu_c,mem_c,power_w,util_pct,vram_mib,cpu_c,fan_duty_pct,fan1_rpm,fan2_rpm" > "$TLOG"
(
  T0=$(date +%s)
  while true; do
    G=$(nvidia-smi -i 0 --query-gpu=temperature.gpu,temperature.memory,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    C=$(sudo ipmitool sdr get TEMP_CPU 2>/dev/null | awk -F: '/Sensor Reading/{split($2,a," ");print a[1]+0;exit}')
    D=$(sudo ipmitool raw 0x3a 0xd0 0x0f 2>/dev/null | awk '{print ("0x"$1)+0}')
    R=$(sudo ipmitool sdr type Fan 2>/dev/null | head -2 | awk -F'|' '{gsub(/[^0-9]/,"",$5); printf "%s,", $5}')
    echo "$(( $(date +%s) - T0 )),${G},${C:-0},${D:-0},${R}" | sed 's/,$//' >> "$TLOG"
    sleep 2
  done
) & MON_PID=$!

echo ""
echo ">>> starting opencode build task (previous hardware took 49m25s)"
echo ">>> thermal abort: GPU>=80C, HBM>=88C"
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
    v=[float(r[n]) for r in rows if r.get(n) not in (None,'','0')]
    return v
for name,label in [('gpu_c','GPU core'),('mem_c','HBM mem'),('cpu_c','CPU'),
                   ('power_w','power W'),('util_pct','GPU util%'),('fan_duty_pct','fan duty%')]:
    v=col(name)
    if v: print(f"  {label:10s} min {min(v):6.1f}   max {max(v):6.1f}   avg {sum(v)/len(v):6.1f}")
print(f"\n  samples: {len(rows)}  (2s resolution)")
PY
echo ""
ls -la "$RUNDIR"/*.html 2>/dev/null && \
  for f in "$RUNDIR"/*.html; do echo "  $(basename "$f"): $(wc -c < "$f") bytes, $(wc -l < "$f") lines"; \
  node --check "$f" 2>/dev/null && echo "    (node --check on raw html is not meaningful; see extracted JS)"; done
echo ""
echo "COMPARISON (2x RTX PRO 4000 24GB, DDR5-3600, 2026-07-31): 2965s / 44,092 bytes / 1,211 lines"
echo "=================================================================="
echo "thermal csv: $TLOG"
echo "server log : $SLOG"
