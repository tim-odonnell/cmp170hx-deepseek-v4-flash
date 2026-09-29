#!/bin/bash
# Phase 1: build the DeepSeek-V4-Flash-Vision-Exp engine image for 3x CMP 170HX.
#
# Source: leinasi2014/deepseek-v4-vision-cmp170hx (vLLM fork, same fast-SM80 lineage as
# our 0731 engine), built with that fork's own docker/Dockerfile.cmp170hx-sm80.
# Full native build (13 csrc/ files differ from our 0731 base, so the
# VLLM_USE_PRECOMPILED=1 shortcut used for dsv4-a100:devel cannot be used here).
#
# Does NOT touch the 0731 setup: separate source dir, separate build context,
# new image tag. dsv4-a100:devel and ~/CMP-170HX-PROJECT/vllm-dsv4/vllm are never read
# or written by this script.
#
# Usage: nohup ./build-image.sh > ../logs/build-<ts>.log 2>&1 &
set -euo pipefail

BASE=${BASE:-$HOME/CMP-170HX-PROJECT/vllm-dsv4}   # any working dir with ~60 GB free
SRC=$BASE/vllm-vision-leinasi          # git clone of the fork
CTX=$BASE/build-ctx-vision-leinasi     # docker build context (source copy + dsv4-srcs/)
PIN=4fe10bf                            # fork commit (includes the post-validation DSpark fixes)
CUTLASS_TAG=v4.4.2                     # = CUTLASS_REVISION pinned in the fork's CMakeLists.txt
TRITON_TAG=v3.5.1                      # = TRITON_KERNELS_TAG pinned in cmake/external_projects/triton_kernels.cmake
TAG="dsv4-vision:sm80-leinasi-$PIN"
OUT=$(cd "$(dirname "$0")/.." && pwd)/results

echo "=== build start $(date -Is)"
mkdir -p "$BASE"
[ -d "$SRC/.git" ] || git clone -q https://github.com/leinasi2014/deepseek-v4-vision-cmp170hx "$SRC"
git -C "$SRC" checkout -q "$PIN"
echo "fork:    $(git -C "$SRC" rev-parse HEAD)"

# Pinned CUTLASS/Triton bundle the Dockerfile expects at dsv4-srcs/.
mkdir -p "$CTX/dsv4-srcs"
[ -d "$CTX/dsv4-srcs/cutlass" ] || git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass.git "$CTX/dsv4-srcs/cutlass"
[ -d "$CTX/dsv4-srcs/triton" ]  || git clone -q --depth 1 --branch "$TRITON_TAG"  https://github.com/triton-lang/triton.git "$CTX/dsv4-srcs/triton"
echo "cutlass: $(git -C "$CTX/dsv4-srcs/cutlass" rev-parse HEAD) ($CUTLASS_TAG)"
echo "triton:  $(git -C "$CTX/dsv4-srcs/triton" rev-parse HEAD) ($TRITON_TAG)"

# Source copy into the context (fork's own recipe: rsync without .git).
rsync -a --delete --exclude .git --exclude dsv4-srcs "$SRC"/ "$CTX"/

DOCKER_BUILDKIT=1 docker build \
  -f "$CTX/docker/Dockerfile.cmp170hx-sm80" \
  -t "$TAG" \
  "$CTX"

echo "=== post-build gates $(date -Is)"
docker run --rm --entrypoint python3 "$TAG" -c "
import torch, vllm, vllm._custom_ops
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)
from vllm.models.deepseek_v4.nvidia import vl_model, model, dspark
print('deepseek_v4 vision/model/dspark imports OK')
"

{
  echo "image:   $TAG"
  echo "id:      $(docker image inspect "$TAG" --format '{{.Id}}')"
  echo "built:   $(date -Is)"
  echo "fork:    leinasi2014/deepseek-v4-vision-cmp170hx @ $(git -C "$SRC" rev-parse HEAD)"
  echo "cutlass: NVIDIA/cutlass @ $(git -C "$CTX/dsv4-srcs/cutlass" rev-parse HEAD) ($CUTLASS_TAG)"
  echo "triton:  triton-lang/triton @ $(git -C "$CTX/dsv4-srcs/triton" rev-parse HEAD) ($TRITON_TAG)"
  echo "dockerfile: docker/Dockerfile.cmp170hx-sm80 (unmodified)"
} | tee "$OUT/build-record.txt"
# Patch 0001 (vision tower on the first PP rank only; frees ~0.87 GiB on GPUs 1-2, which is what
# lets 3 cards reach 691,968 context). Pure-Python overlay on the native image.
docker build -q -t "$TAG-p1" "$(dirname "$0")/../overlay"
echo "overlay:  $TAG-p1 id=$(docker image inspect "$TAG-p1" --format '{{.Id}}')" | tee -a "$OUT/build-record.txt"
echo "=== BUILD_OK $(date -Is) -- use image $TAG-p1"
