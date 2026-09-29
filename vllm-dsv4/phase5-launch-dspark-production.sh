#!/bin/bash
# Phase 5: PRODUCTION DEFAULT launch script for DeepSeek-V4-Flash-0731 + DSpark
# on 3x CMP-170HX. This is the settled, fully-verified config as of 2026-08-15
# after a systematic binary-search campaign between the known-good baseline
# (0.95 util / 262144 ctx) and the known-bad crash point (0.98 util / 1048576
# ctx / full context). See CARD-REGISTRY.md and the deepseek_v4_vllm_pp_dspark
# memory file for the full campaign writeup.
#
# Verified results from that campaign (all tested by deliberately reproducing
# the crash: --max-num-seqs 8 at ~95% of the context ceiling, real deep
# prefill, not just a load test):
#   util=0.95   -> max ctx 338,432   PASSED stress test
#   util=0.96   -> max ctx 574,464   PASSED stress test
#   util=0.965  -> max ctx 691,968   PASSED stress test  <- this script's default
#   util=0.97   -> max ctx 799,488   CRASHED (Worker_PP2, illegal memory access)
#   util=0.98   -> max ctx 1,048,576 CRASHED (twice, in earlier sessions)
# There is a hard cliff between 0.965 (safe) and 0.97 (crashes) -- do not push
# utilization above 0.965 for this workload on this hardware without re-running
# the full stress-test methodology above.
#
# VLLM_PP_LAYER_PARTITION rebalancing was tested and DISPROVEN as a lever for
# this crash -- vLLM's own gpu_worker.py memory log shows all 3 PP ranks
# always land at an identical total fill % regardless of partition (it only
# reshuffles weights vs. KV cache within an already-fixed total budget).
# Don't waste time changing VLLM_PP_LAYER_PARTITION as a stability fix.
#
# Usage: ./phase5-launch-dspark-production.sh [util] [maxlen] [max_num_seqs] [partition] [max_batched_tokens]
#   defaults: util=0.965, maxlen=691968, max_num_seqs=8, partition=15,15,13, max_batched_tokens=2048
set -euo pipefail

IMG="dsv4-a100:devel"
MODEL="$HOME/models/models/deepseek-ai-DeepSeek-V4-Flash-0731"
UTIL="${1:-0.965}"
MAXLEN="${2:-691968}"
MAXSEQS="${3:-8}"
PARTITION="${4:-15,15,13}"
MAXBATCH="${5:-2048}"
NAME="dsv4-a100-3card"
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":5}'
# top_p=0.95 (not the checkpoint's default 1.0) per DeepSeek's official recommendation
# for agentic workloads -- this box's only opencode use case. Verified via A/B
# (2026-08-20): no speed cost, no coherence loss, slightly sharper tool/command
# choices in a sample coding-debug prompt. temperature stays at the checkpoint
# default (1.0) via generation_config.json; only top_p is overridden.

[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR: image $IMG not built yet"; exit 1; }

docker stop -t 60 "$NAME" >/dev/null 2>&1 || true
docker rm "$NAME" >/dev/null 2>&1 || true

if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
        --entrypoint python3 "$IMG" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(3)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2 >/dev/null 2>&1 || true
  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
  for g in 0 1 2; do sudo nvidia-smi -i "$g" -pl 100 >/dev/null; done
fi

# shellcheck disable=SC2086
docker run -d --name "$NAME" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e VLLM_SPARSE_DENSE_QUERY_BLOCK=0 \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -v "$MODEL":/model \
  --shm-size=16g -p 8098:8000 \
  "$IMG" vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 3 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAXBATCH" --trust-remote-code \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAXSEQS" \
  --override-generation-config '{"top_p": 0.95}' \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  $SPEC >/dev/null

echo "launched $NAME on :8098 (util=$UTIL, maxlen=$MAXLEN, max_num_seqs=$MAXSEQS, partition=$PARTITION, DSpark) -- PRODUCTION DEFAULT"
echo "watch load: docker logs -f $NAME"
echo "health check once loaded: curl -s http://127.0.0.1:8098/health"
