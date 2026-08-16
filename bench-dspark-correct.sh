#!/usr/bin/env bash
# DSpark benchmark using unsloth's DOCUMENTED configuration, corrected from my earlier run.
#
# WHAT WAS WRONG BEFORE (bench-dspark-cmp170hx.sh):
#   * used `--fit on --fit-target 12288`. The DSpark README states plainly:
#     "--fit off is required when using a DSpark drafter." My --fit-target workaround was
#     invented to dodge an OOM that only existed because --fit should not have been on.
#     Side effect: it displaced ~12 GB of weights into system RAM for no reason.
#   * used `--spec-type draft-dspark,ngram-mod`. The documented minimum is `draft-dspark`.
#
# WHAT WAS ACTUALLY FINE:
#   * `--spec-draft-n-max 2`. The README's "5" is the CLAMP (the trained block size), not the
#     optimum. Community reports on r/LocalLLaMA converge on 2-3, including one user in our
#     exact situation ("16 t/s -> 18 t/s with n-max of 2, but i have to offload a bunch to ram
#     anyway"). Swept here anyway.
#   * no `-devd`/`--spec-draft-device` (README: must not be passed; the drafter borrows the
#     target's embeddings and output head so it must span the same devices).
#   * stock build. The -DGGML_SCHED_MAX_SPLIT_INPUTS=48 rebuild is MULTI-GPU only; the README
#     says single-GPU setups are unaffected.
#
# THE PART THE README DOES NOT COVER: it assumes the whole model fits in VRAM (`-ngl 99` on
# 4x B200). Ours is 151 GiB on a 64 GiB card, so `--fit off` requires manual expert offload.
# 43 layers, 256 experts, 6 active. `-ncmoe N` keeps the first N layers' MoE on CPU; this
# script finds the LOWEST N that still loads (i.e. the most experts on the GPU) and uses it.
set -uo pipefail

MD=/home/user/ai-models/unsloth-DeepSeek-V4-Flash-0731
MODEL="$MD/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
DRAFT="$MD/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
SERVER=/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server
PORT=8080; CTX=65536; RUNS="${RUNS:-3}"; NPREDICT=300
OUT="$(dirname "$0")/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUT/dspark_correct_${STAMP}.csv"

SWAP=0; SRV=""
cleanup(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 5; kill -9 "$SRV" 2>/dev/null; }
           (( SWAP )) && sudo systemctl start llama-swap 2>/dev/null; }
trap cleanup EXIT INT TERM
systemctl is-active --quiet llama-swap && { SWAP=1; sudo systemctl stop llama-swap; sleep 3; }

wait_free(){ for i in $(seq 1 30); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${u:-9999}" -le 2000 ] && return; sleep 5; done; }

# $1=ncmoe $2=nmax $3=spec(1|0) -> starts server, returns 0 if healthy
start(){
  local NC=$1 NM=$2 SPEC=$3
  local L="$OUT/dsc_ncmoe${NC}_n${NM}_spec${SPEC}_${STAMP}.log"
  LASTLOG="$L"
  wait_free
  local specargs=()
  (( SPEC )) && specargs=( --model-draft "$DRAFT" --spec-type draft-dspark
                           --spec-draft-n-max "$NM" -ngld 999 )
  "$SERVER" --model "$MODEL" --alias dsc \
    --fit off -ngl 999 -ncmoe "$NC" \
    "${specargs[@]}" \
    --ctx-size "$CTX" --ctx-checkpoints 0 \
    -fa on -ctk f16 -ctv f16 --parallel 1 \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
    --threads 11 --threads-batch 11 --no-mmap \
    --reasoning-format deepseek --reasoning on --reasoning-budget 10000 \
    --host 127.0.0.1 --port "$PORT" --jinja > "$L" 2>&1 &
  SRV=$!
  for i in $(seq 1 30); do
    kill -0 "$SRV" 2>/dev/null || return 1
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}
stop(){ [[ -n "$SRV" ]] && { kill "$SRV" 2>/dev/null; sleep 6; kill -9 "$SRV" 2>/dev/null; SRV=""; }; }

echo "=================================================================="
echo " DSpark with the DOCUMENTED config (--fit off, --spec-type draft-dspark)"
echo "=================================================================="
echo ">>> step 1: find the lowest -ncmoe that loads (most experts on GPU)"
BEST_NC=""
for NC in 32 34 36 38 40; do
  printf "  -ncmoe %-3s ... " "$NC"
  if start "$NC" 2 1; then
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    R=$(free -g | awk '/^Mem:/{print $3}')
    echo "LOADED   VRAM ${V} MiB   hostRAM ${R}Gi"
    BEST_NC="$NC"; stop; break
  else
    grep -qi 'out of memory' "$LASTLOG" && echo "OOM" || echo "failed"
    stop
  fi
done
[[ -n "$BEST_NC" ]] || { echo "  no working -ncmoe found in range"; exit 1; }
echo "  -> using -ncmoe ${BEST_NC}"

echo "spec,n_max,run,prefill,decode" > "$CSV"
bench(){  # $1=label $2=nmax $3=spec
  local LBL=$1 NM=$2 SPEC=$3
  echo ""; echo ">>> ${LBL}"
  start "$BEST_NC" "$NM" "$SPEC" || { echo "  failed to start"; tail -12 "$LASTLOG"; stop; return 1; }
  nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu --format=csv,noheader | sed 's/^/  VRAM /'
  free -h | awk '/^Mem:/{print "  hostRAM "$3" of "$2}'
  local P="Write a detailed technical explanation of how a singly linked list differs from a doubly linked list, including insertion and deletion complexity."
  for r in $(seq 1 $RUNS); do
    RESP=$(curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(printf '%s' "$P" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"n_predict\":${NPREDICT},\"cache_prompt\":false}")
    read -r pp tg <<< "$(printf '%s' "$RESP" | python3 -c '
import json,sys
t=json.load(sys.stdin)["timings"]
print(t.get("prompt_per_second",0), t.get("predicted_per_second",0))')"
    printf "  run%d  prefill %7.2f  decode %6.2f tok/s\n" "$r" "$pp" "$tg"
    echo "$SPEC,$NM,$r,$pp,$tg" >> "$CSV"
  done
  stop
}

bench "BASELINE — no speculation" 0 0
bench "DSpark n_max=2" 2 1
bench "DSpark n_max=5 (README's clamp value)" 5 1

echo ""
echo "=================================================================="
python3 - "$CSV" <<'PYEOF'
import csv,sys,collections,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
g=collections.defaultdict(lambda:{"pp":[],"tg":[]})
for r in rows:
    k = "no-spec" if r["spec"]=="0" else f"dspark n_max={r['n_max']}"
    g[k]["pp"].append(float(r["prefill"])); g[k]["tg"].append(float(r["decode"]))
base=None
for k in ["no-spec","dspark n_max=2","dspark n_max=5"]:
    if k not in g: continue
    tg=st.mean(g[k]["tg"]); pp=st.mean(g[k]["pp"])
    if base is None: base=tg
    print(f"  {k:22s} decode {tg:6.2f} tok/s  prefill {pp:7.2f}  {100*(tg-base)/base:+6.1f}%")
print()
print("  earlier MISCONFIGURED run (--fit on --fit-target 12288, +ngram-mod):")
print("    no-spec 13.65 | n_max=2 14.72 (+7.8%) | n_max=5 11.82 (-13.4%)")
print("  unsloth reference (4x B200, full VRAM): 1.2x-1.3x")
PYEOF
echo "=================================================================="
