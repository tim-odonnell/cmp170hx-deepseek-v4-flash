#!/usr/bin/env bash
# Cowboy-clue agentic build benchmark: GLM-5.2 UD-Q2_K_XL on the unlocked CMP 170HX (64GB VRAM).
#
# Engine: mainline llama.cpp-portable build-cuda-sm80 (NOT ik_llama — both ik_llama.cpp builds on
# this box are compiled for CUDA arch 120/Blackwell, not this card's sm_80, and core-dump here; see
# project memory). Mainline now has native glm-dsa architecture support including the DSA lightning
# indexer and NextN/MTP speculative decoding (commits 88bfee1, 7be2c65 in the checked-out tree,
# confirmed present via `strings` on libllama.so: llama_model_glm_dsa, GLM_DSA MTP asserts,
# gated_delta_net.cu / lightning-indexer.cu present in libggml-cuda.so). This replaces the
# ik_llama.cpp-latest "-mla 3 -amb 512 --merge-qkv -muge" flag set entirely: mainline reads MLA
# dims (key/value_length_mla=256) and the indexer top_k (2048) straight out of GGUF metadata.
#
# Prior benchmark (2026-07-22, ik_llama.cpp-latest fbcc743, RTX Pro 4000 24GB, hybrid --fit,
# MTP n_max=2): 5.96 tok/s decode, n_max sweep 1=5.20/2=5.96(best)/3=5.89 vs 4.86 stock.
# This run: same model/quant, CMP 170HX 64GB VRAM (2.7x the old card's VRAM) -- expect --fit to
# push substantially more experts onto GPU, so decode speed is not assumed to hold at 5.96 tok/s.
# n_max=2 kept as the informed starting point (re-sweeping costs a full 237GB reload per point).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNDIR="/home/user/cowboy-clue benchmark/runs/cowboy-clue-glm52-cmp170hx"
PROMPT_SRC="/home/user/cowboy-clue benchmark/runs/cowboy-clue-cmp170hx/prompt.txt"
MODEL="/home/user/ai-models/unsloth-GLM-5.2-UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf"
SERVER="/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"
OCMODEL="direct-glm52-cmp170hx/glm-5.2-mtp-cmp170hx"
PORT=8080
CTX=81920

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
for svc in aas loracula loracula-ears; do
    if systemctl --user is-active --quiet "$svc" 2>/dev/null; then
        echo ">>> stopping $svc"; systemctl --user stop "$svc"; sleep 3
    fi
done

echo ">>> fan daemon: $(systemctl is-active gpu-fan-daemon.service 2>/dev/null)"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
(( USED > 2000 )) && { echo "ERROR: GPU busy (${USED} MiB)"; exit 1; }

echo ">>> starting llama-server (GLM-5.2, hybrid --fit, ctx ${CTX}, MTP n_max=2, q8_0 KV)"
echo ">>> 237GB across 7 shards -- first load can take a while, be patient"
export GGML_CUDA_NO_PINNED=1
"$SERVER" --model "$MODEL" \
    --alias "glm-5.2-mtp-cmp170hx" \
    --fit on \
    --load-mode none \
    --ctx-size "$CTX" \
    -ctk q8_0 -ctv q8_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --parallel 1 \
    --threads 11 --threads-batch 11 \
    --reasoning-format deepseek \
    --reasoning-budget 10000 \
    --host 127.0.0.1 --port "$PORT" --jinja > "$SLOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 240); do    # up to 40 min for the 237GB hybrid load
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
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw --format=csv,noheader
free -h | awk '/^Mem:/{print "host RAM used: "$3" of "$2}'

echo ""
echo ">>> sanity generation:"
curl -sf "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","n_predict":16,"cache_prompt":false}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))'

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
echo ">>> starting opencode build task (comparison: GLM-5.2 on RTX Pro 4000 24GB, 2026-07-22: 5.96 tok/s decode)"
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
echo ">>> decode speed from server log (n_predict/predicted_per_second on each completion):"
grep -oE '"predicted_per_second":[0-9.]+' "$SLOG" | tail -5
echo ""
ls -la "$RUNDIR"/*.html 2>/dev/null && \
  for f in "$RUNDIR"/*.html; do echo "  $(basename "$f"): $(wc -c < "$f") bytes, $(wc -l < "$f") lines"; done
echo ""
echo "COMPARISON (GLM-5.2, RTX Pro 4000 24GB, ik_llama MTP n_max=2, 2026-07-22): 5.96 tok/s decode"
echo "=================================================================="
echo "thermal csv: $TLOG"
echo "server log : $SLOG"
echo "opencode log: $OLOG"
