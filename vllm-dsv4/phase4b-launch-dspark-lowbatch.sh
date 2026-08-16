#!/bin/bash
# Phase 4b: same as phase4-launch-dspark.sh but with reduced
# --max-num-batched-tokens / --max-num-seqs, to test whether freeing more of
# PP2's transient VRAM margin lets DSpark survive a real long-context prefill
# (phase4 crashed with a CUDA illegal memory access in the sparse attention
# indexer on a ~97k-token prefill at max-model-len=1048576, 0.98 util).
#
# Usage: ./phase4b-launch-dspark-lowbatch.sh [maxlen] [max_batched_tokens] [max_num_seqs]
#   defaults: maxlen=1048576, max_batched_tokens=512, max_num_seqs=2
set -euo pipefail

IMG="dsv4-a100:devel"
MODEL="$HOME/models/models/deepseek-ai-DeepSeek-V4-Flash-0731"
MAXLEN="${1:-1048576}"
MAXBATCH="${2:-512}"
MAXSEQS="${3:-2}"
NAME="dsv4-a100-3card"
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":5}'

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
  -e VLLM_PP_LAYER_PARTITION=15,15,13 \
  -v "$MODEL":/model \
  --shm-size=16g -p 8098:8000 \
  "$IMG" vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 3 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAXBATCH" --trust-remote-code \
  --gpu-memory-utilization 0.98 --max-num-seqs "$MAXSEQS" \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  $SPEC >/dev/null

echo "launched $NAME on :8098 (maxlen=$MAXLEN, max_batched_tokens=$MAXBATCH, max_num_seqs=$MAXSEQS, DSpark)"
echo "watch load: docker logs -f $NAME"
echo "health check once loaded: curl -s http://127.0.0.1:8098/health"
