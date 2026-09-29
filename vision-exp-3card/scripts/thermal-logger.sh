#!/bin/bash
# Thermal logger for the Vision-Exp campaign: every 5 s, all 4 CMP 170HX cards
# (core temp, HBM memory temp, power, thermal-throttle flags) + chassis fan RPMs from the BMC
# + the gpu-fan-daemon's current duty. Writes a CSV; prints ALERT lines and a 10-minute summary.
# Read-only: never changes fans, clocks, or power (gpu-fan-daemon owns the fans).
#
# Alert thresholds (card specs from nvidia-smi -q: core max operating 85C, slowdown 95C,
# HBM max operating 95C):  core >= 80C, HBM >= 85C, any HW/SW thermal slowdown active,
# or a fan that was spinning dropping below 300 RPM.
#
# usage: nohup ./thermal-logger.sh > ../logs/thermal-<ts>.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
CSV="logs/thermal-$(date +%Y%m%d-%H%M%S).csv"
CORE_ALERT=80 MEM_ALERT=85 FAN_MIN=300
echo "time,gpu,core_c,mem_c,power_w,hw_thermal,sw_thermal,fans_rpm,daemon_duty" > "$CSV"
echo "thermal logger started $(date '+%F %T'), csv=$CSV"
declare -A maxc maxm
last_summary=$SECONDS
while true; do
  ts=$(date '+%F %T')
  fans=$(sudo ipmitool sdr type Fan 2>/dev/null | awk -F'|' '$5 ~ /RPM/ {gsub(/ /,"",$1); gsub(/ RPM| /,"",$5); printf "%s=%s ", $1, $5}')
  duty=$(journalctl -u gpu-fan-daemon -n 1 --no-pager -o cat 2>/dev/null | grep -o 'duty [0-9]*%' | tail -1)
  while IFS=, read -r idx core mem pw hw sw; do
    idx=${idx// /}; core=${core// /}; mem=${mem// /}; hw=${hw# }; sw=${sw# }
    echo "$ts,$idx,$core,$mem,${pw// /},$hw,$sw,\"$fans\",$duty" >> "$CSV"
    [[ $core =~ ^[0-9]+$ ]] || continue
    (( core > ${maxc[$idx]:-0} )) && maxc[$idx]=$core
    [[ $mem =~ ^[0-9]+$ ]] && (( mem > ${maxm[$idx]:-0} )) && maxm[$idx]=$mem
    (( core >= CORE_ALERT )) && echo "ALERT $ts GPU$idx core ${core}C (>= ${CORE_ALERT})"
    [[ $mem =~ ^[0-9]+$ ]] && (( mem >= MEM_ALERT )) && echo "ALERT $ts GPU$idx HBM ${mem}C (>= ${MEM_ALERT})"
    [[ $hw == Active || $sw == Active ]] && echo "ALERT $ts GPU$idx THERMAL SLOWDOWN hw=$hw sw=$sw"
  done < <(nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,power.draw,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits 2>/dev/null)
  for f in $fans; do
    rpm=${f#*=}; [[ $rpm =~ ^[0-9]+$ ]] && (( rpm < FAN_MIN )) && echo "ALERT $ts fan ${f%%=*} at ${rpm} RPM (< ${FAN_MIN})"
  done
  if (( SECONDS - last_summary >= 600 )); then
    s="SUMMARY $ts max-last-10min:"; for g in 0 1 2 3; do s+=" GPU$g core=${maxc[$g]:-?}C hbm=${maxm[$g]:-?}C;"; done
    echo "$s fans: $fans| $duty"; maxc=(); maxm=(); last_summary=$SECONDS
  fi
  sleep 5
done
