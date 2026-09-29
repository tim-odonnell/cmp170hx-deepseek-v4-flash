# Project plan: DeepSeek-V4-Flash-Vision-Exp on 3x CMP-170HX (vLLM, PP3, DSpark, vision)

**Goal:** get `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` working on 3 of the 4 CMP 170HX
cards (GPUs 0,1,2; GPU 3 stays completely free for other work), as well as or better than
our published 0731 recipe, with **vision verified working**. The hard constraint is GPU fill:
each card filled **equal to or slightly less than** the 0731 recipe. Context length is allowed
to come out lower than 0731's 691,968 if that's what the fill limit gives.
Publish to GitHub **only after** function and token speeds are confirmed.

Everything we learn goes in this folder (`vision-exp-3card/`). It stays uncommitted until
we're ready to publish.

- Research round: [`PRIOR-ART.md`](PRIOR-ART.md) (read this first; it changed the plan).
- Superseded first draft: [`history/PLAN-v1-2026-09-29.md`](history/PLAN-v1-2026-09-29.md).
- 0731 recipe this is measured against: `../PLAN-vllm-pp-dspark-3card.md`,
  `../PERFORMANCE-SUMMARY.md`, `../CARD-REGISTRY.md` (2026-08-15 entries),
  `../vllm-dsv4/phase5-launch-dspark-production.sh`.

## FINAL RESULTS (2026-09-29): all phases complete, published

User guide: [`../HOWTO-RUN-DEEPSEEK-V4-FLASH-VISION-EXP.md`](../HOWTO-RUN-DEEPSEEK-V4-FLASH-VISION-EXP.md).
Full record: [`CAMPAIGN-LOG.md`](CAMPAIGN-LOG.md). Crash guide: [`TROUBLESHOOTING-XID31.md`](TROUBLESHOOTING-XID31.md).

| goal | result |
|---|---|
| GPUs 0-2 only, GPU 3 free | ✅ GPU 3 at 0 MiB throughout |
| fill ≤ 0731 | ✅ config A (user's pick): util 0.965 like 0731; GPU2 peak +134 MiB (+0.2%), GPU0 +1.27 GB = the vision tower. Config B (0.96) also validated: GPUs 1-2 ~700 MiB under 0731 |
| context | ✅ **691,968 = 0731**, after patch 0001 (vision tower only on PP0; the fork put a dead 0.87 GiB copy on every rank) |
| vision verified | ✅ 9/9 gates (incl. real webpage screenshot, planted-bug detection, 25/25 table cells, cross-request leak check), repeated through stress + soak |
| tools | ✅ 7/7 incl. vision→tool and multi-step 1MDB web research with 0 invented sources |
| stability | ✅ seqs-8 630k stress with images, 3 cold restarts ×2 configs, 38-min soak 20/20 (user stopped early; crossed the ~174k leinasi trigger 9×), 0 Xid |
| speed | ⚠️ ~86 vs ~100 tok/s short-prompt decode; per-step speed equal, gap = the checkpoint's DSpark drafter acceptance (2.9 vs 3.5). Near parity at 100k; Vision-Exp prefills faster |
| temps | ✅ core ≤ 67 °C, HBM ≤ 77 °C under load |

Deviations from the plan below: Phase 1 used a pinned-tag CUTLASS/Triton bundle; the BF16 dequant
from leinasi's profile had to be turned off (it crashed boot on 3 cards); the fill campaign
needed patch 0001; the soak was cut short by user decision.

## What "as well or better" means, concretely

| | 0731 recipe (published) | Vision-Exp target |
|---|---|---|
| Cards | GPUs 0,1,2, PP=3 | same, GPU 3 untouched |
| **GPU fill** | `--gpu-memory-utilization 0.965` (hard cliff: 0.97 crashed) | **≤ 0.965**, and per-card measured usage ≤ the Phase 0 baseline |
| Power | 100 W/card cap | 100 W/card cap (hard rule, never higher) |
| Context | 691,968 | whatever fits at that fill. Lower is acceptable, higher is a bonus |
| Decode, DSpark | ~81 tok/s avg at 8k (peak ~100); ~93 tok/s quoted production | match or beat |
| Decode, no-spec | 55.8 tok/s | reference point only |
| Stability | passed seqs=8 at ~95% of the context ceiling, 3 cold restarts | same test, **plus image traffic mixed in**, plus a long soak |
| Vision | n/a | all Phase 3 gates pass |

## How the 0731 fill was set (the method we're copying)

From `CARD-REGISTRY.md` 2026-08-15 (final):
- `--gpu-memory-utilization` is the one lever. `VLLM_PP_LAYER_PARTITION` was tested and
  **disproven** as a fill lever: vLLM normalizes every rank to the same total budget and
  backfills with KV cache.
- For each util level: binary-search the largest `--max-model-len` that loads (from vLLM's
  "X GiB KV cache needed, Y available" error; its estimate runs slightly low), then relaunch at
  that ceiling with `--max-num-seqs 8` and push a real deep prefill to ~91-98% of it.

  | util | max ctx | stress @ ~95% depth |
  |---|---|---|
  | 0.95 | 338,432 | PASSED |
  | 0.96 | 574,464 | PASSED |
  | **0.965** | **691,968** | **PASSED** ← production |
  | 0.97 | 799,488 | CRASHED (PP2, illegal memory access) |
- Trust vLLM's own `gpu_worker.py` memory log over `nvidia-smi` for per-rank analysis on this
  unlocked hardware. Record both anyway.

## Pre-flight facts (all checked 2026-09-29)

- **Checkpoint** downloaded and SHA256-verified: `~/models/models/deepseek-ai-DeepSeek-V4-Flash-Vision-Exp/`,
  167,831,846,872 bytes. Identical weights to the revision other CMP groups validated (see
  PRIOR-ART).
- **Same backbone as 0731**: 43 layers, same MoE, same DSpark drafter (3 layers, block size
  5), same `max_position_embeddings`. **Extra:** a 32-layer ViT (dim 1024) + aligner, ≤384
  tokens per image. That's roughly 0.3-0.4B params, <1 GiB, all on PP0. Expect the fill budget
  to be about 0731's, with context landing a bit lower because PP0 carries the vision tower and
  vLLM also reserves memory to profile the image encoder.
- **DSpark IS supported** (4,708 drafter tensors in our checkpoint). My earlier "not
  supported" note was wrong; see PRIOR-ART. That's good news for the speed target.
- **No P2P** between our cards (same as 0731). Use the no-P2P env set (`Wiziechen` profile):
  no `NCCL_P2P_LEVEL=SYS`, no `NCCL_CUMEM_ENABLE=1`.
- **Resources**: 24 cores, 503 GB RAM (plenty for streaming weight load), 290 GB free disk
  (enough for a ~30 GB image + build cache, but watch it).
- **Nobody has run this on 3 cards.** All published data is 4-card. On 4 cards at util
  0.85-0.93 the cards sit at ~50-61 GiB each. On 3 cards the weights alone are ~52+ GiB/card,
  the same tight regime as 0731.

## Engine choice

**Primary: `leinasi2014/deepseek-v4-vision-cmp170hx`**, a full vLLM fork on the same fast SM80
base as our 0731 engine, with vision + DSpark + 1M-context validated on CMP 170HX and the most
mature bug-fixing record. Cost: a **full native CUDA build** (13 native files differ from our
base, so the precompiled shortcut can't be used).

Fallbacks, in order:
1. PixelML prebuilt image `ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902`
   (built on our exact base commit). No compile, but weaker stability (c=4 crash, 131k crash).
2. `kaka86mm` recipe (wtdcode `dsv4-vision-exp` + PR #54566 cherry-picks + 4 sm80 patches,
   DSpark n=3).
3. Our own hand-merge (`vllm-vision-exp-experiment/`). Shelved: missing DSpark and known fixes.

**Isolation rules (all phases):** never modify `~/CMP-170HX-PROJECT/vllm-dsv4/vllm/` or the
`dsv4-a100:devel` image. 0731 must stay one command away. New image tag and container name,
test port **8099** (0731 keeps 8098). Only one of the two can run at a time, since they share
GPUs 0-2.

---

## Phase 0: Measure the 0731 fill baseline (≈20 min, uses the existing image)

The published recipe gives the *setting* (0.965) but we never recorded per-card *measured*
usage at that setting. Without it, "equal or slightly less fill" can't be checked. So:
1. `curl /health` first, per habit. Launch `phase5-launch-dspark-production.sh` unchanged.
2. Record from the startup log, per rank: model weights GiB, "Actual usage … KV cache …"
   lines, KV cache tokens / GiB, the free margin after profiling. Record `nvidia-smi`
   memory.used per card at idle.
3. Run one deep request (~600k tokens, the same depth as the original stress test) and sample
   `nvidia-smi` every 0.5 s for peak per-card usage.
4. Write it to `FILL-BASELINE.md`. **That table is the ceiling Vision-Exp must meet or come
   under.** Stop 0731.

## Phase 1: Build the engine (background, ~1-3 h, estimate unverified)

1. Clone `leinasi2014/deepseek-v4-vision-cmp170hx` to
   `~/CMP-170HX-PROJECT/vllm-dsv4/vllm-vision-leinasi/`. Pin the commit (current head
   `4fe10bf51`; last code change `f00f0eecc`). Diff head vs `f00f0eecc` to confirm
   later commits are docs-only.
2. Prepare the `dsv4-srcs/` bundle (CUTLASS + Triton). They didn't record revisions, so derive
   them from the fork's own cmake pins / torch 2.13's Triton, and **record what we used**.
3. Build with their `docker/Dockerfile.cmp170hx-sm80` (torch 2.13.0, CUDA 13.0.2,
   `TORCH_CUDA_ARCH_LIST=8.0`, `MAX_JOBS` sized to 24 cores). Log to this folder
   (`logs/build-*.log`), run in the background. Tag `dsv4-vision:sm80-leinasi-<sha>`.
4. Gates before any GPU use: `import vllm`, `import vllm._custom_ops`, the DSV4 vision model
   classes import, image ID/size recorded.
5. If the build fails in a way that isn't quickly fixable, pull the PixelML image (fallback 1)
   so Phases 2-3 can proceed while the build is debugged.

## Phase 2: First boot on 3 cards (conservative)

Start from their `serve-pp4.sh` + Wiziechen's no-P2P changes, adapted to 3 cards:
- `NVIDIA_VISIBLE_DEVICES=0,1,2`, `--pipeline-parallel-size 3`,
  `VLLM_PP_LAYER_PARTITION=15,15,13` (0731's split; exact 43-layer fit). If it fails to boot,
  try `16,15,12` / `15,16,12`. PixelML found only one partition boots on 4 cards, and the
  last rank carries drafter + lm_head (+ embedding for DSpark), so fewer layers there.
- **Mandatory env**: `DSV4_LOGITS_ROW_CHUNK=64`. Plus their set: `VLLM_MARLIN_FP8_DEQUANT_BF16=1`,
  `VLLM_DSPARK_FUSED_MARKOV=1`, `VLLM_PREFILL_BLOCK_H=8`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
  `VLLM_SPARSE_DENSE_QUERY_BLOCK=4` (0731 needed `0`, so fall back to 0 on a Triton
  `OutOfResources`), `VLLM_PP_COMM_PRIME=0`, `NCCL_IB_DISABLE=1`, `HF_HUB_OFFLINE=1`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- DSpark `{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic"}`.
- `--gpu-memory-utilization 0.93`, `--max-model-len 16384`, `--max-num-seqs 2`,
  `--max-num-batched-tokens 2048`, fp8 KV, block 256, tokenizer/tool/reasoning parser
  `deepseek_v4`, `--limit-mm-per-prompt '{"image":2}'`.
- Launch script generated as a complete file (`scripts/launch-vision-3card.sh`), never
  hand-retyped (leinasi's Xid-31 lesson).
- Gates: health, `/v1/models`, greedy "capital of France" → Paris ×3, then one image request.
  If DSpark won't boot, boot no-spec once to isolate whether the drafter is the problem.

## Phase 3: Vision verification (must pass before any tuning)

Saved as a rerunnable script `scripts/vision-gates.py`, results in `results/`:
1. **Pixel-reading proof**: 64x64 two-colour gradient → names both colours in order; also a
   **no-image control** (same question, no image) and a **wrong-image control** (different
   gradient). Correct answers only with the right image prove it's seeing, not guessing.
2. **Our real use case**: screenshots of a real web page / UI (from Will's site or a local
   page) → describe layout, colours, visible text; spot a deliberately introduced visual bug.
   Aesthetic-reference image → describe style.
3. **Structured content** (the ik_llama #2464 concern): a table/spreadsheet screenshot →
   exact cell values and headers.
4. **Cross-request contamination check** (kaka86mm's CUDA-graph bug): interleave image
   requests with plain-text probes, and confirm text answers never leak content from other
   requests.
5. **Robustness**: invalid image → clean HTTP 400, engine stays up. Two images in one
   request. Image at depth (after a long text context).
6. Image requests at c=1, 2, 4 (PixelML's build crashed at c=4).

## Phase 4: Fill campaign (the core of the project)

Same method as 0731, **capped at util 0.965**:
1. For util ∈ {0.95, 0.96, 0.965}: binary-search max `--max-model-len` that loads, with
   DSpark on, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`.
2. Stress each at its ceiling: seqs=8, deep prefill to ~95% of ceiling, **with image
   requests interleaved**, then 3 cold restarts. Record per-rank vLLM memory log +
   `nvidia-smi` peaks.
3. Compare every rank against `FILL-BASELINE.md`. **The winner is the highest util ≤ 0.965
   that passes stress and whose measured per-card fill is ≤ the 0731 baseline.** If
   0.965's fill measures higher than 0731's (vision encoder overhead), step down, as the
   brief says ("equal or slightly less").
4. If the tightest rank crashes like 0731's PP2 did: first lever `--max-num-seqs` (0731's
   proven fix), not partition.
5. Log every attempt to `CAMPAIGN-LOG.md` (append-only), including failures.

## Phase 5: Speed benchmarks

Using the 0731 `bench.py` method, plus kaka86mm's rules (TTFT separated from decode, tokens
counted from `usage`, first request after restart discarded, randomized prompt prefix):
- DSpark decode at shallow / ~100k / ~300k / ~90% of ceiling; no-spec at shallow for
  reference; prefill tok/s at each depth; image-request latency (cold/hot).
- Power per card during decode (100 W cap), tok/Wh, same as `PERFORMANCE-SUMMARY.md`.
- Side-by-side table vs 0731.

## Phase 6: Soak

≥2 h of mixed text + image + deep-context traffic at the chosen config (leinasi's crash
appeared only after 67 min of live traffic). Clean dmesg (no Xid) at the end.

## Phase 7: Publish (only after you confirm Phases 3-6 results)

- `HOWTO-RUN-DEEPSEEK-V4-FLASH-VISION-EXP.md` in the style of the 0731 HOWTO; final launch
  script; `CARD-REGISTRY.md` entries (append-only); README/PERFORMANCE-SUMMARY sections;
  build recipe with pinned revisions; attribution to leinasi2014, Wiziechen, kaka86mm,
  PixelML, wtdcode, allover326.
- Also fix the already-committed-but-unpushed `deepseek-v4-flash-vision-agent/PLAN.md`
  (contains the wrong "DSpark unsupported" claim) before pushing commit `3b4aa0f`.

## Known risks

- **Fill may not allow useful context.** On 3 cards the ~1 GiB vision tower + encoder profiling
  reserve competes with KV on PP0. Likely outcome: context somewhat below 691,968. Could be
  much lower if the image-encoder profiling budget is large. Mitigation: check whether
  `--limit-mm-per-prompt` / encoder-cache settings shrink that reserve.
- **The native build is the biggest unknown** (unrecorded CUTLASS/Triton revisions, long compile).
- **100 W vs their 180-250 W**: their absolute speeds won't transfer. Our own 0731 at 100 W is
  the fair comparison.
- **Partition boot sensitivity** (PixelML: only one 4-card partition booted).
- A hard GPU wedge on these unlocked cards needs a **reboot** (0731 experience). Budget for it
  during stress testing.
