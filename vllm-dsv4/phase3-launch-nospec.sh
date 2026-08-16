#!/bin/bash
# Phase 3: DeepSeek-V4-Flash-0731 on 3x CMP 170HX, pipeline-parallel=3, NO DSpark.
# Adapted from allover326/deepseek-v4-cmp170hx launch/run-pp-dspark.sh (fetched verbatim
# 2026-08-14) for our 3-card box. Patches are baked into the dsv4-a100:devel image at build
# time (COPY+pip install -e . picked up the already-patched source), so unlike the upstream
# script we do NOT need its bind-mount step -- DSV4_NO_MOUNT behavior is implicit here.
#
# Usage: ./phase3-launch-nospec.sh [maxlen]   (default 8192, per the project plan's Phase 3)
set -euo pipefail

IMG="dsv4-a100:devel"
MODEL="$HOME/models/models/deepseek-ai-DeepSeek-V4-Flash-0731"
MAXLEN="${1:-8192}"
NAME="dsv4-a100-3card"

[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR: image $IMG not built yet"; exit 1; }

docker stop -t 60 "$NAME" >/dev/null 2>&1 || true
docker rm "$NAME" >/dev/null 2>&1 || true

# GPU wedge check (a prior illegal-memory-access can leave cards unable to create CUDA
# contexts; rmmod/modprobe nvidia_uvm clears it without a host reboot). Our cards are
# power-capped to 100W via a systemd service (see [[rtx3090_power_limit]] pattern /
# CMP-170HX-PROJECT cmp-powerlimit.service), NOT the repo's 180W -- reapply 100W, not 180W,
# if a reset is needed.
if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
        --entrypoint python3 "$IMG" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(3)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2 >/dev/null 2>&1 || true
  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
  for g in 0 1 2; do sudo nvidia-smi -i "$g" -pl 100 >/dev/null; done
fi

docker run -d --name "$NAME" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e VLLM_SPARSE_DENSE_QUERY_BLOCK=0 \
  -e VLLM_PP_LAYER_PARTITION=15,15,13 \
  -v "$MODEL":/model \
  --shm-size=16g -p 8098:8000 \
  "$IMG" vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 3 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 --trust-remote-code \
  --gpu-memory-utilization 0.93 --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  >/dev/null

echo "launched $NAME on :8098 (maxlen=$MAXLEN, no speculative decoding)"
echo "watch load: docker logs -f $NAME"
echo "health check once loaded: curl -s http://127.0.0.1:8098/health"
