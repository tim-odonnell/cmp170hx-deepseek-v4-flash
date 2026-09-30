#!/bin/bash
# DeepSeek-V4-Flash-Vision-Exp on 3x CMP 170HX (GPUs 0,1,2; GPU 3 never touched).
# Engine: leinasi2014/deepseek-v4-vision-cmp170hx @ 4fe10bf (see results/build-record.txt).
#
# Shares GPUs 0-2 with the 0731 setup, so only one can run at a time. This script
# stops the 0731 container (dsv4-a100-3card) if it is running. It does NOT modify
# anything belonging to 0731 (image, scripts, source); switch back any time with
# ./switch-to-0731.sh.
#
# Defaults = config A chosen 2026-09-29 after Phase 4: util 0.965 / 691,968 / seqs 8 (same as
# 0731). Passed ceiling + seqs-8 630k stress w/ images + 3 cold restarts. See CAMPAIGN-LOG.md.
#
# Usage: ./launch-vision-3card.sh [util] [maxlen] [max_num_seqs] [partition] [max_batched_tokens] [spec:on|off]
set -euo pipefail

IMG="${IMG:-dsv4-vision:sm80-leinasi-4fe10bf-p1}"  # p1 = + patch 0001 (vision tower on PP0 only)
MODEL="${MODEL:-$HOME/models/models/deepseek-ai-DeepSeek-V4-Flash-Vision-Exp}"  # override only for A/B experiments
UTIL="${1:-0.965}"
MAXLEN="${2:-691968}"
MAXSEQS="${3:-8}"
PARTITION="${4:-15,15,13}"
MAXBATCH="${5:-2048}"
SPECMODE="${6:-on}"
NAME="dsv4-vision-3card"
PORT="${PORT:-8099}"
# Max images per CONVERSATION (vLLM counts every image in the request history, not per message).
# Was 2; raised to 8 on 2026-09-29 after an opencode agent reviewing its own screenshots hit the cap.
IMAGES="${IMAGES:-8}"

# Hard ceiling from the project brief: fill must not exceed the 0731 recipe (0.965).
awk -v u="$UTIL" 'BEGIN{exit !(u <= 0.965)}' || { echo "ERROR: util $UTIL > 0.965 (0731 ceiling), refusing"; exit 1; }
[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR: image $IMG not built yet"; exit 1; }

if docker ps --format '{{.Names}}' | grep -qx dsv4-a100-3card; then
  echo "stopping 0731 (dsv4-a100-3card) to free GPUs 0-2..."
  docker stop -t 60 dsv4-a100-3card >/dev/null
fi
docker stop -t 60 "$NAME" >/dev/null 2>&1 || true
docker rm "$NAME" >/dev/null 2>&1 || true

# Wedge probe: FAIL FAST, never auto-recover. On 2026-09-29 the 0731-style auto-recovery
# (rmmod/modprobe nvidia_uvm) after an Xid 31 caused a UVM global fatal error and Xid 154
# ("OS reboot required") on ALL 4 cards, GPU 3 included. On these unlocked cards the only
# reliable recovery is a reboot (CARD-REGISTRY 2026-08-15 said the same).
if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
        --entrypoint python3 "$IMG" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(3)]' \
        >/dev/null 2>&1; then
  echo "ERROR: CUDA probe on GPUs 0-2 failed (GPU wedged or driver faulted)."
  echo "Check: sudo dmesg -T | grep -i xid | tail   -- if Xid 31/154 appear, REBOOT."
  echo "Do NOT rmmod/modprobe nvidia_uvm: on these cards it escalates to Xid 154 on every GPU."
  exit 2
fi

SPEC=()
if [[ "$SPECMODE" == on ]]; then
  # n=5 exactly: 6/7 hang at startup on this fork (leinasi2014 red line).
  SPEC=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic"}')
fi

# Env: leinasi2014 serve-pp4.sh set, minus NCCL_P2P_LEVEL=SYS / NCCL_CUMEM_ENABLE=1
# (no P2P on this box; Wiziechen no-P2P profile). DSV4_LOGITS_ROW_CHUNK=64 is MANDATORY:
# without it the unchunked indexer path illegal-writes at ~174k context (Xid 31).
# VLLM_MARLIN_FP8_DEQUANT_BF16 defaults OFF here (leinasi has it on): it dequantizes FP8
# dense weights to BF16 at load (+~2.4 GiB/rank measured 2026-09-29). On 3 cards that left
# PP2 no room for the DSpark drafter's Marlin repack -> illegal memory access + Xid 31 at
# boot. 0731 never used it. Opt in with MARLIN_DEQUANT=1 only to test the speed trade.
docker run -d --name "$NAME" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
  --ipc=host --shm-size=32g -p "$PORT":8000 \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e VLLM_MARLIN_FP8_DEQUANT_BF16="${MARLIN_DEQUANT:-0}" \
  -e VLLM_MARLIN_FP8_DEQUANT_EXCLUDE="${MARLIN_DEQUANT_EXCLUDE:-}" \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -e VLLM_DSPARK_FUSED_MARKOV=1 \
  -e VLLM_PREFILL_BLOCK_H=8 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_SPARSE_DENSE_QUERY_BLOCK="${SDQB:-4}" \
  -e VLLM_PP_COMM_PRIME=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_IB_DISABLE=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "$MODEL":/model:ro \
  "$IMG" \
  /opt/venv/bin/vllm serve /model --served-model-name dsv4v \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --pipeline-parallel-size 3 --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAXBATCH" \
    --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" \
    --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
    --override-generation-config '{"top_p": 0.95}' \
    --limit-mm-per-prompt "{\"image\": ${IMAGES}}" \
    "${SPEC[@]}" >/dev/null

echo "launched $NAME on :$PORT (util=$UTIL, maxlen=$MAXLEN, seqs=$MAXSEQS, partition=$PARTITION, batch=$MAXBATCH, dspark=$SPECMODE)"
echo "watch load: docker logs -f $NAME"
echo "health: curl -s http://127.0.0.1:$PORT/health"
