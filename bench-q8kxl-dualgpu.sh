#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 UD-Q8_K_XL split across dual CMP-170HX (128GB combined).
#
# 150.75 GiB model does NOT fit fully in 128GB (unlike IQ3_XXS) -- --fit should still place a
# much bigger fraction GPU-resident than the single-card 39% (~74% estimated). This is a real,
# same-quant data point on the residency-vs-speed curve, to properly ground a 3-card estimate
# instead of the cross-quant IQ3_XXS extrapolation.
set -uo pipefail

MD=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731
MODEL="$MD/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
DRAFT="$MD/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
PORT=8080; CTX=65536; RUNS="${RUNS:-3}"; NPREDICT=300
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUT/q8kxl_dualgpu_${STAMP}.csv"

[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL"; exit 1; }

SWAP=0; SRV=""
cleanup(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 5; kill -9 "$SRV" 2>/dev/null; }
           (( SWAP )) && sudo systemctl start llama-swap 2>/dev/null; }
trap cleanup EXIT INT TERM
systemctl is-active --quiet llama-swap && { SWAP=1; sudo systemctl stop llama-swap; sleep 3; }

wait_free(){ for i in $(seq 1 30); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [ "${u:-9999}" -le 3000 ] && return; sleep 5; done; }

start(){
  local SPEC=$1 NM=$2
  local L="$OUT/q8kxl_dualgpu_spec${SPEC}_n${NM}_${STAMP}.log"
  LASTLOG="$L"
  wait_free
  local specargs=()
  (( SPEC )) && specargs=( --model-draft "$DRAFT" --spec-type draft-dspark --spec-draft-n-max "$NM" --n-gpu-layers-draft 999 )
  "$SERVER" --model "$MODEL" "${specargs[@]}" --alias q8kxl-dual \
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
echo " DeepSeek-V4-Flash-0731 UD-Q8_K_XL, dual CMP-170HX (128GB combined)"
echo "=================================================================="
echo "spec,n_max,run,prefill,decode,vram0_mib,vram1_mib,host_ram_gib" > "$CSV"

bench(){
  local LBL=$1 SPEC=$2 NM=$3
  echo ""; echo ">>> ${LBL}"
  start "$SPEC" "$NM" || { echo "  failed to start"; tail -40 "$LASTLOG"; stop; return 1; }
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
    echo "$SPEC,$NM,$r,$pp,$tg,$V0,$V1,$RAM" >> "$CSV"
  done
  stop
}

bench "no-spec baseline" 0 0
bench "DSpark n_max=3" 1 3

echo ""
echo "=================================================================="
python3 - "$CSV" <<'PYEOF'
import csv,sys,collections,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
g=collections.defaultdict(lambda:{"pp":[],"tg":[]})
for r in rows:
    k = "no-spec" if r["spec"]=="0" else f"n_max={r['n_max']}"
    g[k]["pp"].append(float(r["prefill"])); g[k]["tg"].append(float(r["decode"]))
for k in g:
    tg=st.mean(g[k]["tg"]); pp=st.mean(g[k]["pp"])
    print(f"  {k:12s} decode {tg:6.2f} tok/s  prefill {pp:7.2f} tok/s")
PYEOF
echo "=================================================================="
echo "csv: $CSV"
