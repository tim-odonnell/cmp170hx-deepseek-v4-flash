#!/usr/bin/env bash
# Qwen3.6-35B-A3B UD-Q4_K_XL: single-card baseline vs true tensor-parallel split
# (-sm tensor, mainline llama.cpp) across both unlocked CMP-170HX cards (128GB combined).
#
# Unlike the DeepSeek-V4-Flash dual-GPU scripts on this box, this model (22.85 GiB)
# already fits entirely on ONE 64GB card -- so this is NOT a --fit capacity-driven
# split. It's a genuine "does splitting weights+KV across 2 GPUs over PCIe Gen2 x4
# (no NVLink, no P2P, PHB topology) speed up or slow down decode" test.
#
# Test A: CUDA_VISIBLE_DEVICES=0 only, -ngl 999, no split (single-card baseline).
# Test B: both cards visible, -sm tensor (splits weights+KV, not just offload),
#         -ts 1,1 (equal split, cards are identical).
set -uo pipefail

MODEL="/home/user/ai-models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
PORT=8080; CTX=32768; RUNS="${RUNS:-3}"; NPREDICT=300
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUT/qwen36-35b-tensorparallel_${STAMP}.csv"

[[ -f "$MODEL"  ]] || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: sm_80 build not found: $SERVER"; exit 1; }

SWAP=0; SRV=""
cleanup(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 5; kill -9 "$SRV" 2>/dev/null; }
           (( SWAP )) && sudo systemctl start llama-swap 2>/dev/null; }
trap cleanup EXIT INT TERM
systemctl is-active --quiet llama-swap && { SWAP=1; echo ">>> stopping llama-swap"; sudo systemctl stop llama-swap; sleep 3; }

wait_free(){ for i in $(seq 1 30); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [ "${u:-9999}" -le 3000 ] && return; sleep 5; done; }

# $1 = "single" | "dual"
start(){
  local MODE=$1
  local L="$OUT/qwen36-35b-tp_${MODE}_${STAMP}.log"
  LASTLOG="$L"
  wait_free
  local envprefix=() splitargs=()
  if [[ "$MODE" == "single" ]]; then
    envprefix=(env CUDA_VISIBLE_DEVICES=0)
  else
    splitargs=( -sm tensor -ts 1,1 )
  fi
  "${envprefix[@]}" "$SERVER" --model "$MODEL" "${splitargs[@]}" \
    --alias "qwen36-35b-${MODE}" \
    -ngl 999 --ctx-size "$CTX" \
    --reasoning-format deepseek --reasoning-budget 12000 \
    -fa on -ctk q8_0 -ctv q8_0 --parallel 1 \
    --threads 12 --threads-batch 12 --no-mmap \
    --host 127.0.0.1 --port "$PORT" --jinja > "$L" 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do
    kill -0 "$SRV" 2>/dev/null || return 1
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    CUR_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    printf "\r    loading... %ds  GPU0 temp: %sC" "$((i*10))" "$CUR_TEMP"
    sleep 10
  done
  return 1
}
stop(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 6; kill -9 "$SRV" 2>/dev/null; SRV=""; }; }

echo "=================================================================="
echo " Qwen3.6-35B-A3B UD-Q4_K_XL: single-card vs -sm tensor dual-card"
echo " (CMP-170HX x2, PCIe Gen2 x4, no NVLink/P2P, PHB topology)"
echo "=================================================================="
echo "mode,run,prefill,decode,vram0_mib,vram1_mib,temp0_c,temp1_c,pwr0_w,pwr1_w" > "$CSV"

bench(){
  local LBL=$1 MODE=$2
  echo ""; echo ">>> ${LBL}"
  start "$MODE" || { echo "  failed to start"; tail -40 "$LASTLOG"; stop; return 1; }
  echo ""
  LPID=$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  echo "  listener on :${PORT} owned by PID ${LPID:-?} (launched ${SRV})"
  local V0 V1 T0 T1 P0 P1
  read -r V0 V1 <<< "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  read -r T0 T1 <<< "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr '\n' ' ')"
  read -r P0 P1 <<< "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | tr '\n' ' ')"
  echo "  GPU0 ${V0} MiB / ${T0}C / ${P0}W   GPU1 ${V1} MiB / ${T1}C / ${P1}W"
  local P="Write a detailed technical explanation of how a singly linked list differs from a doubly linked list, including insertion and deletion complexity."
  for r in $(seq 1 $RUNS); do
    RESP=$(curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(printf '%s' "$P" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"n_predict\":${NPREDICT},\"cache_prompt\":false}")
    read -r pp tg <<< "$(printf '%s' "$RESP" | python3 -c '
import json,sys
t=json.load(sys.stdin)["timings"]
print(t.get("prompt_per_second",0), t.get("predicted_per_second",0))')"
    read -r V0 V1 <<< "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    read -r T0 T1 <<< "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr '\n' ' ')"
    read -r P0 P1 <<< "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | tr '\n' ' ')"
    printf "  run%d  prefill %7.2f  decode %6.2f tok/s   GPU0 %sC/%sW  GPU1 %sC/%sW\n" "$r" "$pp" "$tg" "$T0" "$P0" "$T1" "$P1"
    echo "$MODE,$r,$pp,$tg,$V0,$V1,$T0,$T1,$P0,$P1" >> "$CSV"
  done
  stop
}

bench "Test A: single card baseline (GPU0 only)" "single"
bench "Test B: dual card -sm tensor (both GPUs)" "dual"

echo ""
echo "=================================================================="
python3 - "$CSV" <<'PYEOF'
import csv,sys,collections,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
g=collections.defaultdict(lambda:{"pp":[],"tg":[]})
for r in rows:
    g[r["mode"]]["pp"].append(float(r["prefill"])); g[r["mode"]]["tg"].append(float(r["decode"]))
res={}
for k in g:
    tg=st.mean(g[k]["tg"]); pp=st.mean(g[k]["pp"])
    res[k]=tg
    print(f"  {k:8s} decode {tg:6.2f} tok/s  prefill {pp:7.2f} tok/s")
if "single" in res and "dual" in res:
    delta = (res["dual"]/res["single"] - 1) * 100
    print(f"\n  dual vs single: {delta:+.1f}%")
PYEOF
echo "=================================================================="
echo "csv: $CSV"
