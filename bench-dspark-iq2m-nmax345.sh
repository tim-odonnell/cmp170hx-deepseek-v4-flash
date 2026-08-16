#!/usr/bin/env bash
# Follow-up to bench-dspark-iq2m-dualgpu.sh: that sweep covered no-spec + n_max 2/3/5 and found
# n_max=2 best (47.70 tok/s). This fills in n_max=4 and re-confirms 3/5 with the same methodology
# so all four DSpark depths (2,3,4,5) are directly comparable.
set -uo pipefail

MD=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731
MODEL="$MD/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf"
DRAFT="$MD/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
PORT=8080; CTX=65536; RUNS="${RUNS:-3}"; NPREDICT=300
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUT/dspark_iq2m_nmax345_${STAMP}.csv"

[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL"; exit 1; }
[[ -f "$DRAFT" ]] || { echo "ERROR: drafter missing: $DRAFT"; exit 1; }

SWAP=0; SRV=""
cleanup(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 5; kill -9 "$SRV" 2>/dev/null; }
           (( SWAP )) && sudo systemctl start llama-swap 2>/dev/null; }
trap cleanup EXIT INT TERM
systemctl is-active --quiet llama-swap && { SWAP=1; sudo systemctl stop llama-swap; sleep 3; }

wait_free(){ for i in $(seq 1 30); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [ "${u:-9999}" -le 3000 ] && return; sleep 5; done; }

start(){
  local NM=$1
  local L="$OUT/dspark_iq2m_nmax345_n${NM}_${STAMP}.log"
  LASTLOG="$L"
  wait_free
  "$SERVER" --model "$MODEL" --model-draft "$DRAFT" \
    --spec-type draft-dspark --spec-draft-n-max "$NM" --n-gpu-layers-draft 999 \
    --alias dsc-iq2m-nmax345 \
    --fit on --fit-target 12288 \
    --ctx-size "$CTX" --ctx-checkpoints 0 \
    -fa on -ctk f16 -ctv f16 --parallel 1 \
    --temp 1.0 --top-p 0.95 --min-p 0.01 \
    --threads 11 --threads-batch 11 --no-mmap \
    --reasoning-format deepseek --reasoning on --reasoning-budget 2000 \
    --host 127.0.0.1 --port "$PORT" --jinja > "$L" 2>&1 &
  SRV=$!
  for i in $(seq 1 24); do
    kill -0 "$SRV" 2>/dev/null || return 1
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}
stop(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 6; kill -9 "$SRV" 2>/dev/null; SRV=""; }; }

echo "=================================================================="
echo " DeepSeek-V4-Flash-0731 UD-IQ2_M + DSpark n_max sweep: 3, 4, 5"
echo "=================================================================="
echo "n_max,run,prefill,decode,vram0_mib,vram1_mib,host_ram_gib" > "$CSV"

bench(){
  local NM=$1
  echo ""; echo ">>> DSpark n_max=${NM}"
  start "$NM" || { echo "  failed to start"; tail -40 "$LASTLOG"; stop; return 1; }
  local V0 V1 RAM
  read -r V0 V1 <<< "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  RAM=$(free -g | awk '/^Mem:/{print $3}')
  echo "  GPU0 ${V0} MiB   GPU1 ${V1} MiB   hostRAM ${RAM}Gi"
  local P="Write a detailed technical explanation of how a singly linked list differs from a doubly linked list, including insertion and deletion complexity."
  for r in $(seq 1 $RUNS); do
    RESP=$(curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(printf '%s' "$P" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"n_predict\":${NPREDICT},\"cache_prompt\":false}")
    read -r pp tg <<< "$(printf '%s' "$RESP" | python3 -c '
import json,sys
t=json.load(sys.stdin)["timings"]
print(t.get("prompt_per_second",0), t.get("predicted_per_second",0))')"
    printf "  run%d  prefill %7.2f  decode %6.2f tok/s\n" "$r" "$pp" "$tg"
    echo "$NM,$r,$pp,$tg,$V0,$V1,$RAM" >> "$CSV"
  done
  stop
}

bench 3
bench 4
bench 5

echo ""
echo "=================================================================="
python3 - "$CSV" <<'PYEOF'
import csv,sys,collections,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
g=collections.defaultdict(lambda:{"pp":[],"tg":[]})
for r in rows:
    g[r["n_max"]]["pp"].append(float(r["prefill"])); g[r["n_max"]]["tg"].append(float(r["decode"]))
best=None; bestv=-1
for k in g:
    tg=st.mean(g[k]["tg"]); pp=st.mean(g[k]["pp"])
    print(f"  n_max={k:3s} decode {tg:6.2f} tok/s  prefill {pp:7.2f} tok/s")
    if tg > bestv: bestv=tg; best=k
print(f"\n  BEST OF 3/4/5: n_max={best} at {bestv:.2f} tok/s")
print(f"  (for reference, n_max=2 from the earlier sweep was 47.70 tok/s)")
PYEOF
echo "=================================================================="
echo "csv: $CSV"
