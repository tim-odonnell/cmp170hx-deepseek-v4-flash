# Project plan: DeepSeek-V4-Flash-0731, vLLM + pipeline-parallel + DSpark, 3x CMP-170HX

Goal: replicate `allover326/deepseek-v4-cmp170hx` (98 tok/s decode / ~5300 tok/s prefill on
4x CMP-170HX) on our 3-card box. This is a different engine (vLLM, not llama.cpp) and a
different model format (native MXFP4+FP8 safetensors, not GGUF) — a parallel track alongside
the existing llama.cpp work, not a replacement for it.

Source: https://github.com/allover326/deepseek-v4-cmp170hx (patches/docker/launch scripts/docs)
Base engine: https://github.com/haosdent/vllm branch `dsv4-flash-a100`

## FINAL RESULTS (2026-08-15) — all 6 phases complete

**Goal met: 2.3-3.4x faster than the best llama.cpp result on this box (23.8 tok/s).**
Full numbers, configs, and every gotcha hit are in `CARD-REGISTRY.md` (search `2026-08-15`) and
[[deepseek_v4_vllm_pp_dspark]] (persistent memory note). Short version:

| Config | Decode | Notes |
|---|---|---|
| No-spec, ctx 8192 | 55.8 tok/s | 2.3x llama.cpp best |
| DSpark, ctx 8192 | 80.95 tok/s avg (peak 99.93) | 3.4x peak; matches the source repo's 4-card flagship number on 1 fewer card |
| No-spec, max-model-len 1M, ~338k real depth | 39.7 tok/s decode / 1,785 tok/s prefill | VRAM flat vs ctx 8192 — this arch's KV cache really is nearly free |
| DSpark, max-model-len 1M, near-zero depth | 90.7 tok/s avg | works fine short-context |
| **DSpark, max-model-len 1M, ~97k real depth** | **CRASHES** | CUDA illegal memory access in the sparse indexer kernel on the VRAM-starved PP2 rank (only ~1.4 GiB free at the 0.98 utilization needed just to load) |

**Practical takeaway:** no-spec is the safe choice for long-context work (verified clean to
338k); DSpark is a genuine win for short/interactive sessions only, on this 3-card box.
Untried mitigation for the crash: lower `--max-num-batched-tokens`/`--max-num-seqs` to free
more of PP2's margin, at a real throughput cost — not attempted.

Two real engine bugs were hit and fixed along the way (not just flag tuning): a 3-card-specific
OOM from the repo's 4-card `--gpu-memory-utilization` defaults not transferring, and a Triton
shared-memory kernel crash (`VLLM_SPARSE_DENSE_QUERY_BLOCK=0` workaround). A fabricated commit
hash from an earlier research pass was caught and discarded before it could waste a build.

## Pre-flight results (checked 2026-08-14, before committing to this)

- **PWRBRK# hardware power brake: NOT active on any of our 3 cards.** This is the repo's
  headline gotcha (4x slowdown on boards that assert it on x16 slots, e.g. their ASUS
  WRX80E-SAGE) — we don't have it on the ROMED8-2T. Good sign.
- **PCIe: confirmed Gen2 x4 (downgraded from x16), no P2P** — same topology the repo optimized
  around. Confirms pipeline-parallel (not tensor-parallel) is the right call here too: TP would
  hit the same 86-all-reduce PCIe bottleneck they measured (~800 tok/s flat regardless of
  context) that PP avoids.
- **Driver already sufficient** — 610.43.02 reports CUDA UMD 13.3, comfortably covers the
  CUDA 13.0.2 base image the build needs. No host CUDA/driver upgrade required. This container
  is isolated from the host CUDA 12.0 toolkit and doesn't touch the KTransformers CUDA
  12.6-12.9 requirement — [[feedback_cuda]] doesn't apply here since nothing on the host changes.
- **Layer count sanity check:** the 3-card partition the README gives (`15,15,13`) sums to
  43 — exactly this model's layer count (40 MoE + 3 dense). Not a guess, it's an exact fit.
- **Blocker: disk space.** 53 GB free, need ~155-167 GiB for the checkpoint alone, plus Docker
  build cache. Must free space before Phase 3. See Phase 0.

## Known risks/unknowns going in

- **3-card DSpark+PP is unverified.** The repo's 98 tok/s / 1.93x DSpark figures are all
  measured on 4 cards. The `15,15,13` partition is documented as working for base PP, but no
  measured DSpark-on-3-card number exists in their docs — we'd be the first data point.
- **VRAM is tight at 3 cards:** their own estimate is ~52 GiB weights + ~10-12 GiB activations
  per card = ~62-64 GiB against a 64 GiB ceiling. Little headroom for KV cache at extreme
  context depths — may need to cap `--max-model-len` below the full 1M until proven safe, unlike
  our llama.cpp `--fit` result which had ~40 GB of slack.
  - Also worth checking (not yet confirmed): the 5th weight-shard's worth of extra headroom our
    llama.cpp run showed unused VRAM per card, but that's a different quant/format entirely —
    don't assume it transfers.
- Known upstream issues from their docs: accumulating chats crash near 718-733k tokens unless
  `DSV4_LOGITS_ROW_CHUNK=64`; retrieval accuracy degrades from 100%@150k to 30%@900k regardless;
  DSpark output isn't deterministic at temp=0.

## Phase 0 — Free disk space (blocking, do first)

Current usage under `~/models/`, largest first:
| Dir | Size | Related to this project? |
|---|---|---|
| `models/models` (incl. unsloth-DeepSeek-V4-Flash-0731 GGUF+DSpark) | 554G | Partially — the GGUF/DSpark files here are for the *llama.cpp* track, not needed for vLLM/native |
| `mradermacher` | 377G | Unrelated (different model) |
| `Kimi-K2.5-GGUF` | 351G | Unrelated |
| `Qwen-Qwen3.6-27B-bf16-dequant` | 52G | Unrelated |
| `cyankiwi` | 43G | Unrelated |
| `Qwen-Qwen3.6-27B-FP8` | 29G | Unrelated |
| `unsloth-qwen3.6-35B-A3B-GGUF` | 21G | Unrelated |
| `intel-autoround` | 18G | Unrelated |

Need to free ~200 GiB+ (167 GB checkpoint + Docker image/build cache + working margin).
**This is your call, not mine** — I don't know which of these you still need. Candidates worth
you deciding on: `mradermacher` (377G) and/or `Kimi-K2.5-GGUF` (351G) are the biggest and look
least related to this project, but confirm before I touch anything.

## Phase 1 — Build the engine (Docker, isolated from host CUDA)

```bash
git clone --branch dsv4-flash-a100 --single-branch https://github.com/haosdent/vllm.git
cd vllm
git fetch origin '+refs/*:refs/remotes/all/*'
git checkout f8ea5bb        # verified to exist; a later-cited "c3046d1, +7% decode" commit was
                             # checked directly and does NOT exist anywhere in this repo's
                             # history (incl. pre-force-push archive branches) -- that claim was
                             # a fabrication from an earlier page-summarization pass, discarded

git clone https://github.com/allover326/deepseek-v4-cmp170hx.git ../deepseek-v4-cmp170hx
for p in ../deepseek-v4-cmp170hx/patches/*.patch; do patch -p1 < "$p"; done

cp ../deepseek-v4-cmp170hx/docker/Dockerfile.devel .
cp ../deepseek-v4-cmp170hx/docker/dockerignore.txt .dockerignore
docker build -f Dockerfile.devel -t dsv4-a100:devel .   # ~10 min, VLLM_USE_PRECOMPILED=1
```
Must build from inside the vLLM checkout (not a parent dir with weights) — a stray weights
folder in the build context ships hundreds of GB to the Docker daemon.

## Phase 2 — Download the real checkpoint

- Source: `deepseek-ai/DeepSeek-V4-Flash-0731` (native MXFP4+FP8 safetensors, ~155.4 GiB)
- This is a **different file set** from the unsloth GGUF quants already on disk — not reusable.
- Use the same single-connection aria2c approach as before ([[hf_aria2_xet]] — HF's Xet CDN
  needs `-x1 -s1`, presigned URLs 403 on multi-connection). Need to confirm whether this repo
  ships FP8/MXFP4 through the Xet CDN or plain LFS; check before assuming the same gotcha applies.
- Also need to locate the DSpark drafter in **native format** for vLLM — the GGUF dspark files
  we already have (`~/models/models/unsloth-DeepSeek-V4-Flash-0731/dspark/`) are llama.cpp-only
  and won't load in vLLM. Check the deepseek-ai repo / haosdent's docs for where the native
  drafter weights live before assuming they're bundled in the main checkpoint.

## Phase 3 — First launch: base PP, no DSpark, small context

Confirmed exact flags from the repo's `SETTINGS.md`/`launch/run-pp-dspark.sh` (verified against
primary source, not just search summaries):

```bash
export VLLM_PP_LAYER_PARTITION=15,15,13   # 3-card split; naive 15,14,14 fails Marlin MXFP4 repack
                                            # (last rank also carries lm_head + DSpark drafter)
export NVIDIA_VISIBLE_DEVICES=0,1,2
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DSV4_LOGITS_ROW_CHUNK=64

python -m vllm.entrypoints.openai.api_server \
  --model <path-to-downloaded-checkpoint> \
  --pipeline-parallel-size 3 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 8 \
  --tokenizer-mode deepseek_v4 \
  --block-size 256 \
  --no-enable-flashinfer-autotune \
  --trust-remote-code
# do NOT pass --enforce-eager -- repo measured 8-10 tok/s with it vs full speed without
```
Actual runnable scripts (adapted from the repo's real `launch/run-pp-dspark.sh`, fetched
verbatim, not reconstructed): `~/CMP-170HX-PROJECT/vllm-dsv4/phase3-launch-nospec.sh` and
`phase4-launch-dspark.sh` — use `docker run` (matching how the repo actually launches it,
with `--shm-size=16g`), not a bare `python -m vllm...` invocation as originally sketched
above. Both include a GPU-wedge recovery check (illegal-memory-access can leave cards unable
to create CUDA contexts) adapted to reapply our 100W power cap, not the repo's 180W.

Also confirmed straight from the real script's own comments: patch 0001 is explicitly called
"precautionary, the report was retracted" by the repo author too — independent confirmation
(beyond the git-history evidence already found) that skipping it was correct. And
`DSV4_LOGITS_ROW_CHUNK` **defaults to 0/off** — patch 0006 is installed but inert unless this
is explicitly set, so it's baked into both launch scripts as `-e DSV4_LOGITS_ROW_CHUNK=64`,
never left to default.
Note: `--kv-cache-dtype fp8` is REQUIRED here (model asserts `fp8_ds_mla layout only supports
fp8 kv-cache`) — this is the opposite of the llama.cpp track, where fp8/q8_0 KV silently
corrupts output and fp16 is mandatory. Different engine, different KV precision rule — don't
carry the fp16-KV assumption over by habit.

Compare against our existing llama.cpp `--fit` baseline (23.8 tok/s at ctx 8192, zero CPU
offload, no DSpark) — that's the number this needs to beat to justify the engine swap.

## Phase 4 — Enable DSpark under PP

The drafter is embedded in the main checkpoint (`mtp.{0,1,2}` + `markov_head` +
`confidence_head` layers) — no separate download needed. Add:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```
`num_speculative_tokens` must be >= the checkpoint's DSpark block size (5) — vLLM enforces this.
Repo measured 98.1 tok/s at 5 vs 60.3 tok/s at 7, so don't raise it past 5 expecting more gain.

This is the unverified-at-3-cards step — the repo's 98 tok/s number is 4-card only. Watch for
the draft-broadcast patches (0003/0004/0005) behaving correctly across only 3 ranks before
trusting any speedup number.

## Phase 5 — Ramp context

Stage `--max-model-len` up (8k → 100k → 500k → 1M), set `DSV4_LOGITS_ROW_CHUNK=64` for
multi-turn safety (128 only for one-shot prefill), watch per-card VRAM headroom at each step
given the tight budget noted above. Back off before OOM rather than after.

## Phase 6 — Record results

Append to `CARD-REGISTRY.md` and `bench-results/`, matching this project's existing
append-only benchmark convention ([[feedback_benchmark_doc]]).
