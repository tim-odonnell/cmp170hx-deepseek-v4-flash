#!/usr/bin/env bash
# Standalone thermal safety monitor. Polls given GPU indices, logs to CSV,
# docker-stops CONTAINER and exits 1 if any watched GPU crosses the abort thresholds.
set -uo pipefail
GPU_INDICES=${GPU_INDICES:-0}
ABORT_CORE=${ABORT_CORE:-83}
ABORT_MEM=${ABORT_MEM:-90}
CONTAINER=${CONTAINER:-}
KILL_PATTERN=${KILL_PATTERN:-}
LOG=${LOG:-/home/user/CMP-170HX-PROJECT/bench-results/thermal_watchdog.csv}
STOP_FILE=${STOP_FILE:-}
ABORT_FILE=${ABORT_FILE:-/home/user/CMP-170HX-PROJECT/bench-results/ABORTED_${CONTAINER}}

echo "timestamp,gpu_idx,temp_core_c,temp_mem_c,power_w,util_pct" > "$LOG"
rm -f "$ABORT_FILE"
echo "watchdog: watching gpu(s) [$GPU_INDICES], abort core>=${ABORT_CORE}C mem>=${ABORT_MEM}C, container=$CONTAINER"

while true; do
  if [[ -n "$STOP_FILE" && -f "$STOP_FILE" ]]; then
    echo "watchdog: stop file seen, exiting cleanly."
    exit 0
  fi
  IFS=',' read -ra IDXS <<< "$GPU_INDICES"
  for i in "${IDXS[@]}"; do
    raw=$(nvidia-smi -i "$i" --query-gpu=temperature.gpu,temperature.memory,power.draw,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    [[ -z "$raw" ]] && continue
    IFS=',' read -r tcore tmem pwr util <<< "${raw// /}"
    echo "$(date '+%F %T'),$i,$tcore,$tmem,$pwr,$util" >> "$LOG"
    if [[ "$tcore" -ge "$ABORT_CORE" || "$tmem" -ge "$ABORT_MEM" ]]; then
      echo "watchdog: *** THERMAL ABORT *** gpu$i core=${tcore}C mem=${tmem}C" | tee "$ABORT_FILE"
      [[ -n "$CONTAINER" ]] && docker stop $CONTAINER >/dev/null 2>&1
      [[ -n "$KILL_PATTERN" ]] && pkill -f "$KILL_PATTERN" 2>/dev/null
      exit 1
    fi
  done
  sleep 3
done
