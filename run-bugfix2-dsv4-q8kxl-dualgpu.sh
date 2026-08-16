#!/usr/bin/env bash
# Bug-fix round #2: DeepSeek-V4-Flash-0731 UD-Q8_K_XL + DSpark, same dual-GPU full-ctx config as the
# original cowboy-clue benchmark, pointed at the SAME rundir (so it finds the existing
# cowpoke-mystery.html, already carrying round-1's white-screen fix) with a NEW prompt describing
# only the observed shading symptom -- no solution given, per the established
# "self-repair, not spoon-fed" methodology this whole benchmark used.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNDIR="/home/user/cowboy-clue benchmark/runs/cowboy-clue-dsv4-q8kxl-dualgpu"
PROMPT_SRC="${RUNDIR}/bugfix2-prompt.txt"
MD=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731
MODEL="$MD/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
DRAFT="$MD/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
OCMODEL="direct-dsv4-q8kxl-dualgpu/dsv4-q8kxl-dualgpu-dspark"
PORT=8080
CTX=1048576

[[ -f "$PROMPT_SRC" ]] || { echo "ERROR: bugfix2 prompt missing"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="${RUNDIR}/bugfix2-server_${STAMP}.log"
TLOG="${RUNDIR}/bugfix2-thermal_${STAMP}.csv"
OLOG="${RUNDIR}/bugfix2-opencode_${STAMP}.log"

[[ -f "$MODEL" ]] || { echo "ERROR: model missing"; exit 1; }
[[ -f "$DRAFT" ]] || { echo "ERROR: dspark drafter missing"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build missing"; exit 1; }
command -v opencode >/dev/null || { echo "ERROR: opencode not on PATH"; exit 1; }

SWAP_WAS_ACTIVE=0
ABORT_FLAG="${RUNDIR}/.abort_${STAMP}"
cleanup() {
    echo ""; echo ">>> cleanup"
    [[ -n "${MON_PID:-}" ]] && kill "$MON_PID" 2>/dev/null
    [[ -n "${SRV_PID:-}" ]] && { kill "$SRV_PID" 2>/dev/null; sleep 5; kill -9 "$SRV_PID" 2>/dev/null; }
    rm -f "$ABORT_FLAG"
    (( SWAP_WAS_ACTIVE )) && { echo ">>> restarting llama-swap"; sudo systemctl start llama-swap 2>/dev/null; }
}
trap cleanup EXIT INT TERM

if systemctl is-active --quiet llama-swap 2>/dev/null; then
    SWAP_WAS_ACTIVE=1; echo ">>> stopping llama-swap"; sudo systemctl stop llama-swap; sleep 3
fi

echo ">>> fan daemon: $(systemctl is-active gpu-fan-daemon.service 2>/dev/null)"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
(( USED > 3000 )) && { echo "ERROR: GPU busy (${USED} MiB combined)"; exit 1; }

echo ">>> starting llama-server (Q8_K_XL + DSpark n_max=3, full ctx ${CTX}, dual GPU, cache-reuse fix)"
"$SERVER" --model "$MODEL" --model-draft "$DRAFT" \
    --alias "dsv4-q8kxl-dualgpu-dspark" \
    --spec-type draft-dspark --spec-draft-n-max 3 --n-gpu-layers-draft 999 \
    --fit on --fit-target 12288 \
    --ctx-size "$CTX" \
    --cache-reuse 256 --cache-ram -1 \
    -fa on -ctk f16 -ctv f16 --parallel 1 \
    --temp 1.0 --top-p 0.95 --min-p 0.01 \
    --threads 11 --threads-batch 11 --no-mmap \
    --reasoning-format deepseek --reasoning on --reasoning-budget 2000 \
    --host 127.0.0.1 --port "$PORT" --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 90); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "ERROR: server died"; tail -40 "$SLOG"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo ">>> ready after $((i*10))s"; break; }
    printf "\r    loading %ds" "$((i*10))"; sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "ERROR: never ready"; tail -40 "$SLOG"; exit 1; }

LPID=$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [[ "$LPID" != "$SRV_PID" ]]; then
    echo "ERROR: listener on :${PORT} is PID ${LPID:-?}, not the server we launched (${SRV_PID})."
    exit 1
fi
echo ">>> listener PID ${LPID} confirmed owned by launched server"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw --format=csv,noheader

echo "elapsed_s,gpu0_c,mem0_c,power0_w,util0_pct,vram0_mib,gpu1_c,mem1_c,power1_w,util1_pct,vram1_mib,cpu_c,fan_duty_pct,fan1_rpm,fan2_rpm" > "$TLOG"
(
  T0=$(date +%s)
  while true; do
    G=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ',' | sed 's/,$//')
    G0=$(echo "$G" | cut -d, -f1-5)
    G1=$(echo "$G" | cut -d, -f6-10)
    C=$(sudo ipmitool sdr get TEMP_CPU 2>/dev/null | awk -F: '/Sensor Reading/{split($2,a," ");print a[1]+0;exit}')
    D=$(sudo ipmitool raw 0x3a 0xd0 0x0f 2>/dev/null | awk '{print ("0x"$1)+0}')
    R=$(sudo ipmitool sdr type Fan 2>/dev/null | head -2 | awk -F'|' '{gsub(/[^0-9]/,"",$5); printf "%s,", $5}')
    echo "$(( $(date +%s) - T0 )),${G0},${G1},${C:-0},${D:-0},${R}" | sed 's/,$//' >> "$TLOG"
    G1_CORE=$(echo "$G1" | cut -d, -f1); G1_MEM=$(echo "$G1" | cut -d, -f2)
    if [[ -n "$G1_CORE" ]] && { (( $(echo "$G1_CORE" | cut -d. -f1) >= 80 )) || (( $(echo "${G1_MEM:-0}" | cut -d. -f1) >= 88 )); }; then
        echo "!!! GPU1 THERMAL ABORT: core=${G1_CORE}C mem=${G1_MEM}C !!!" >> "$TLOG"
        touch "$ABORT_FLAG"
    fi
    sleep 2
  done
) & MON_PID=$!

(
  while [[ ! -f "$ABORT_FLAG" ]]; do sleep 3; [[ -d "/proc/$MON_PID" ]] || exit; done
  pkill -f "opencode run --dir ${RUNDIR}" 2>/dev/null
) & WATCHDOG_PID=$!

echo ""
echo ">>> starting opencode bug-fix task"
START=$(date +%s)

PROMPT="$(cat "$PROMPT_SRC")"
ABORT_GPU=80 ABORT_MEM=88 RISE_DELTA=12 POLL=5 \
  "${HERE}/thermal-guard.sh" \
  opencode run --dir "$RUNDIR" --model "$OCMODEL" "$PROMPT" 2>&1 | tee "$OLOG"
RC=${PIPESTATUS[0]}
END=$(date +%s); ELAPSED=$((END-START))

kill "$MON_PID" "$WATCHDOG_PID" 2>/dev/null; MON_PID=""

echo ""
echo "=================================================================="
printf "wall-clock : %ds (%dm %ds)\n" "$ELAPSED" $((ELAPSED/60)) $((ELAPSED%60))
echo "guard exit : $RC"
[[ -f "$ABORT_FLAG" ]] && echo "GPU1 independent abort: TRIGGERED"
echo "=================================================================="
echo "thermal csv: $TLOG"
echo "server log : $SLOG"
echo "opencode log: $OLOG"
