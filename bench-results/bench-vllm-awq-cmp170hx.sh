#!/usr/bin/env bash
# Reproduces the dual-3090 151.80 tok/s vLLM+AWQ+MTP benchmark on CMP-170HX.
# Same model, same flags (minus tensor-parallel-size which is set per phase), same
# CAD prompt, same max_tokens/temperature, same TG = completion_tokens / wall-clock method.
set -uo pipefail

NAME=${NAME:-vllm-cmp170hx}
DEVICES=${DEVICES:-0}                 # e.g. "0" or "1" or "0,1"
TP=${TP:-1}
PORT=${PORT:-10004}
MAXLEN=${MAXLEN:-262144}
GPU_INDICES=${GPU_INDICES:-0}         # nvidia-smi indices to watch for thermal safety, comma-sep
ABORT_CORE=${ABORT_CORE:-83}
ABORT_MEM=${ABORT_MEM:-90}
RESULTS_DIR=/home/user/CMP-170HX-PROJECT/bench-results
PROMPT_FILE="$RESULTS_DIR/vllm-awq-cad-prompt.txt"
MODEL=/models/cyankiwi/cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit
IMG=vllm/vllm-openai:v0.24.0-cu129-ubuntu2404

mkdir -p "$RESULTS_DIR"
TEMP_LOG="$RESULTS_DIR/thermal_${NAME}_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,gpu_idx,temp_core_c,temp_mem_c,power_w,util_pct" > "$TEMP_LOG"

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[$NAME] starting container on device(s) $DEVICES, TP=$TP, port $PORT ..."
docker run -d --name "$NAME" \
    --gpus "\"device=${DEVICES}\"" \
    -v /home/user/models:/models \
    --shm-size=32g \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -p "${PORT}:8000" \
    "$IMG" \
        --model "$MODEL" \
        --served-model-name cyankiwi-35b-a3b-awq \
        --trust-remote-code \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-util 0.95 \
        --max-num-seqs 1 \
        --block-size 64 \
        --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
        --host 0.0.0.0 --port 8000 >/dev/null

# --- thermal watchdog: runs until it sees $DONE_FILE, or aborts the container on overheat ---
DONE_FILE=$(mktemp)
ABORT_FILE="$RESULTS_DIR/ABORTED_${NAME}"
rm -f "$ABORT_FILE"
(
  while [[ ! -f "$DONE_FILE" ]]; do
    IFS=',' read -ra IDXS <<< "$GPU_INDICES"
    for i in "${IDXS[@]}"; do
      raw=$(nvidia-smi -i "$i" --query-gpu=temperature.gpu,temperature.memory,power.draw,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
      [[ -z "$raw" ]] && continue
      IFS=',' read -r tcore tmem pwr util <<< "${raw// /}"
      echo "$(date '+%F %T'),$i,$tcore,$tmem,$pwr,$util" >> "$TEMP_LOG"
      if [[ "$tcore" -ge "$ABORT_CORE" || "$tmem" -ge "$ABORT_MEM" ]]; then
        echo "[$NAME] *** THERMAL ABORT *** gpu$i core=${tcore}C mem=${tmem}C (limits: core>=${ABORT_CORE} mem>=${ABORT_MEM})" | tee "$ABORT_FILE"
        docker stop "$NAME" >/dev/null 2>&1
        touch "$DONE_FILE"
        exit 1
      fi
    done
    sleep 3
  done
) &
WATCHDOG_PID=$!

cleanup() { touch "$DONE_FILE"; wait "$WATCHDOG_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[$NAME] waiting for vLLM ready (thermal watchdog active, abort at core>=${ABORT_CORE}C / mem>=${ABORT_MEM}C) ..."
READY=0
READY_ITERS=${READY_ITERS:-180}
for i in $(seq 1 "$READY_ITERS"); do
  if [[ -f "$ABORT_FILE" ]]; then echo "[$NAME] aborted during startup."; exit 1; fi
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
if [[ "$READY" -ne 1 ]]; then
  echo "[$NAME] FAILED to become ready in 900s"; docker logs "$NAME" --tail 100; exit 1
fi
echo "[$NAME] ready."

# warm-up
curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"cyankiwi-35b-a3b-awq\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 10}" >/dev/null

PROMPT_JSON=$(python3 -c "import json,sys; print(json.dumps(open('$PROMPT_FILE').read()))")
BODY=$(python3 -c "
import json
print(json.dumps({
    'model': 'cyankiwi-35b-a3b-awq',
    'messages': [{'role': 'user', 'content': open('$PROMPT_FILE').read()}],
    'max_tokens': 24000,
    'temperature': 0.2
}))
")

if [[ -f "$ABORT_FILE" ]]; then echo "[$NAME] aborted before timed request."; exit 1; fi

echo "[$NAME] running timed CAD-prompt request (max_tokens=24000, temperature=0.2) ..."
T0=$(date +%s.%N)
echo "$BODY" | curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" -d @- > "$RESULTS_DIR/result_${NAME}.json"
T1=$(date +%s.%N)

if [[ -f "$ABORT_FILE" ]]; then
  echo "[$NAME] *** RESULT INVALID -- THERMAL ABORT OCCURRED DURING RUN ***"
  cat "$ABORT_FILE"
  exit 1
fi

TOK=$(python3 -c "import json; print(json.load(open('$RESULTS_DIR/result_${NAME}.json'))['usage']['completion_tokens'])" 2>/dev/null || echo "?")
TIME=$(echo "$T1 - $T0" | bc)
if [[ "$TOK" != "?" ]]; then
  SPEED=$(echo "scale=2; $TOK / $TIME" | bc)
else
  SPEED="?"
fi

echo "=================================================================="
echo "[$NAME] RESULT: tokens=$TOK  wall_time=${TIME}s  TG=${SPEED} tok/s"
echo "[$NAME] thermal log: $TEMP_LOG"
tail -5 "$TEMP_LOG"
echo "=================================================================="

docker stop "$NAME" >/dev/null 2>&1
