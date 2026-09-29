#!/bin/bash
# One Phase 4 boot attempt: launch with the given settings, wait until healthy or exited,
# save the full log, print the lines that matter (KV sizing, fit errors, crashes, Xid).
# Usage: try-boot.sh <util> <maxlen> [seqs] [partition] [batch]
set -uo pipefail
cd "$(dirname "$0")/.."
UTIL=$1 MAXLEN=$2 SEQS=${3:-8} PART=${4:-15,15,13} BATCH=${5:-2048}
TAG="u${UTIL}-L${MAXLEN}-s${SEQS}-p${PART//,/_}-b${BATCH}"
XID0=$(sudo dmesg | grep -ci xid)
./scripts/launch-vision-3card.sh "$UTIL" "$MAXLEN" "$SEQS" "$PART" "$BATCH" on || exit $?
while docker ps --format '{{.Names}}' | grep -qx dsv4-vision-3card; do
  curl -sf -m3 http://127.0.0.1:8099/health >/dev/null && { echo "RESULT $TAG: HEALTHY $(date +%T)"; break; }
  sleep 10
done
docker logs dsv4-vision-3card > "logs/phase4-boot-$TAG.log" 2>&1
grep -v -E "matmul_ogs|rocm.py" "logs/phase4-boot-$TAG.log" | grep -E \
  "Available KV cache|GPU KV cache size|Maximum concurrency|KV cache is needed|estimated maximum|ValueError|AcceleratorError|illegal|out of memory" \
  | sed 's/^.*\] //' | sort -u | cut -c1-300
docker ps --format '{{.Names}}' | grep -qx dsv4-vision-3card || echo "RESULT $TAG: EXITED"
XID1=$(sudo dmesg | grep -ci xid)
[ "$XID1" -gt "$XID0" ] && echo "!!! NEW XID EVENTS: $((XID1-XID0)) -- check dmesg, reboot before next launch"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
