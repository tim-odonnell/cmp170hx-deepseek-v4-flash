#!/bin/bash
# Run this FIRST after any Vision-Exp (or 0731) crash, BEFORE restarting or rebooting.
# It collects everything needed to diagnose the crash into one timestamped folder.
# Read-only: changes nothing, restarts nothing, reloads no drivers.
# usage: ./capture-crash.sh [container=dsv4-vision-3card]
set -uo pipefail
cd "$(dirname "$0")/.."
C=${1:-dsv4-vision-3card}
D="results/crash-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$D"
docker logs "$C" > "$D/container.log" 2>&1
docker inspect "$C" > "$D/container-inspect.json" 2>&1          # exact env + flags it ran with
docker inspect "$C" --format '{{range .Config.Env}}{{println .}}{{end}}' > "$D/env.txt" 2>&1
sudo dmesg -T | grep -i -E "xid|nvrm" > "$D/dmesg-nvrm.txt" 2>&1
nvidia-smi > "$D/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,temperature.memory,power.draw,pstate --format=csv > "$D/gpus.csv" 2>&1
T=$(ls -t logs/thermal-*.csv 2>/dev/null | head -1); [ -n "$T" ] && tail -n 400 "$T" > "$D/thermal-last-100s-x4gpus.csv"
cp logs/soak.out "$D/" 2>/dev/null
# Quick triage, printed to screen and saved:
{
  echo "== crash capture $(date '+%F %T') container=$C"
  echo "-- DSV4_LOGITS_ROW_CHUNK in container env: $(grep -h DSV4_LOGITS_ROW_CHUNK "$D/env.txt" || echo 'MISSING  <-- the leinasi Xid-31 trigger')"
  echo "-- Xid lines: $(grep -ci xid "$D/dmesg-nvrm.txt")  (Xid 154 = driver wants a REBOOT)"
  grep -i xid "$D/dmesg-nvrm.txt" | tail -5
  echo "-- key errors in container log:"
  grep -E "illegal memory|AcceleratorError|unspecified launch|device-side assert|out of memory|ValueError|shm_broadcast|cancelled|EngineDeadError|Worker_PP[0-9].*Error" "$D/container.log" \
    | grep -v -E "Failed to import from (vllm\._C|vllm\._rocm_C|amdsmi)|No available shared memory broadcast block found in 60 seconds" \
    | sed 's/^.*\] //' | sort | uniq -c | sort -rn | head -12   # filtered: known-harmless startup warnings
  echo "-- throughput lines just before the crash (vLLM doesn't log per-request prompt size;"
  echo "   our test scripts do: see logs/soak.out, logs/*depth*.log, results/*/deep*.json):"
  grep -E "Avg prompt throughput" "$D/container.log" | tail -3 | sed 's/^.*\] //' 
  echo "-- error lines per rank (PP2 = GPU 2 = last rank, drafter + lm_head): $(grep -E 'ERROR|Error' "$D/container.log" | grep -v -E "Failed to import|matmul_ogs" | grep -o 'Worker_PP[0-9]' | sort | uniq -c | tr '\n' ' ')"
} | tee "$D/TRIAGE.txt"
echo "saved to $D -- now read TROUBLESHOOTING-XID31.md. Do NOT rmmod/modprobe nvidia modules; if dmesg shows Xid, reboot."
