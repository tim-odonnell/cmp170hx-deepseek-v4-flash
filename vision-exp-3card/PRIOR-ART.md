# Prior art: DeepSeek-V4-Flash-Vision-Exp on CMP 170HX / SM80 (research round 2026-09-29)

GitHub search for anyone who has already run Vision-Exp on CMP 170HX or other SM80 hardware.
Every repo below was cloned and its docs and scripts read directly, not summarized from
search snippets. Clones live in the session scratchpad; re-clone from the URLs to re-check.

**Headline: three independent groups run Vision-Exp on 4x CMP 170HX with vision and DSpark
working. Nobody has published a 3-card result.** `PixelML/club-170hx`'s model page lists
3-card as "untested" and points at *our* 0731 repo for 3-card numbers. `leinasi2014` and
`Wiziechen` both credit `tim-odonnell/cmp170hx-deepseek-v4-flash` as prior art. We'd be the
first 3-card data point, same position we were in for 0731.

## Correction this research forced

**Vision-Exp DOES ship a DSpark drafter.** Earlier notes (old PLAN v1, the
`deepseek-v4-flash-vision-agent/PLAN.md` files, memory) said "DSpark not supported for
Vision-Exp." That was wrong. It generalized one fork's limitation into a model limitation:
`wtdcode/vllm-backport` PR #58's loader drops the drafter weights ("not supported for the
vision variant" was *that code's* comment). The checkpoint itself has the drafter, checked
directly in our downloaded `model.safetensors.index.json`:

| | 0731 | Vision-Exp |
|---|---|---|
| `mtp.*` tensors | 4,705 | 4,708 (incl. `mtp.2.markov_head`, `mtp.2.confidence_head`) |
| `num_nextn_predict_layers` | 3 | 3 |
| `dspark_block_size` | 5 | 5 |
| `vision.*` / `aligner.*` tensors | 0 | 259 / 4 |

Three forks run DSpark on it on CMP 170HX (below). So the fair comparison against 0731's
DSpark production recipe is **Vision-Exp with DSpark**, not no-spec.

## Our checkpoint = the validated one

HF `main` is `6821d6ad` (2026-09-01). PixelML pinned `86f746b3` (2026-08-31). Every commit
between them is README/model-card only (`Update README`, `Add community evaluation results`,
`Update Model Card for SGLang support`, `Updata vLLM recipe`). Weights are identical. Total
167,831,846,872 bytes, matching our download exactly.

## The repos, ranked by usefulness to us

### 1. `leinasi2014/deepseek-v4-vision-cmp170hx`: strongest starting point

https://github.com/leinasi2014/deepseek-v4-vision-cmp170hx (full vLLM fork, Apache-2.0)

- **Same lineage as our 0731 engine.** History starts at `e48d0e0` "DSv4 SM80:
  DeepSeek-V4-Flash on A100: sparse MLA enablement, kernel tuning" (the fast SM80 base our
  `f8ea5bb16` comes from). Then a production snapshot, then "port: add DeepSeek V4 Vision to
  old CMP170HX fast base". They ported the new vision semantics *into* the proven fast base,
  which is the same idea as our own merge attempt, done much more thoroughly (10 adaptation
  points in `docs/cmp170hx/PORTING.md`, plus later correctness hardening).
- Validated: PP4, Vision + DSpark n=5 (probabilistic), fp8 KV, `max_model_len=1,048,576`
  service contract, 13h+ production uptime with zero Xid after the root-cause fix.
- Measured (4 cards, P2P on, their power cap ~230-250W): 77.3 tok/s mean decode (B8 arm),
  99.1 tok/s with `NCCL_P2P_LEVEL=SYS`, 4,225 tok/s exact-10K prefill.
- Per-GPU memory under load at util 0.85 on 4 cards: 51.0 / 49.9 / 54.7 / 56.1 GiB
  (`benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/gpu-summary.csv`). Peak 60.9 GiB on
  util-0.93 arms.
- **Native build required**: 13 files under `csrc/` differ from our `f8ea5bb16` tree (topk,
  sampler, marlin, moe bindings, etc.), so our `VLLM_USE_PRECOMPILED=1` Dockerfile.devel
  trick won't work. Recipe is `docker/Dockerfile.cmp170hx-sm80`: torch 2.13.0 / CUDA 13.0.2,
  `TORCH_CUDA_ARCH_LIST=8.0`, plus a CUTLASS + Triton source bundle whose exact revisions
  they did **not** record. We'll have to pin our own and record them.
- **Key lessons from their docs (apply directly to us):**
  - `DSV4_LOGITS_ROW_CHUNK=64` is **mandatory**. Their multi-day Xid-31 crash hunt ended at
    a hand-written relaunch script that dropped it. Unset, the unchunked indexer path
    illegal-writes at ~174k context every time. Our 0731 scripts already set it; never lose it.
  - Generate relaunch commands from `docker inspect`, never hand-transcribe env vars.
  - `num_speculative_tokens` exactly 5 (6/7 hang at startup on this fork).
  - Never `--enforce-eager`, never `--attention-backend flashinfer`, never
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (GA100 VMM corruption).
  - Env set: `VLLM_MARLIN_FP8_DEQUANT_BF16=1`, `VLLM_DSPARK_FUSED_MARKOV=1`,
    `VLLM_PREFILL_BLOCK_H=8`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
    `VLLM_SPARSE_DENSE_QUERY_BLOCK=4` (ours is 0, which was needed for a Triton
    shared-memory crash on our older build; re-test), `VLLM_PP_COMM_PRIME=0`.
  - Repeated Xid 31 escalates to Xid 154 (reboot required). Matches our own 0731 wedge
    experience.

### 2. `Wiziechen/cmp170hx-dsv4v-conda`: the no-P2P profile of #1

https://github.com/Wiziechen/cmp170hx-dsv4v-conda

- Runs leinasi2014's fork on 4x CMP 170HX **without P2P**, which is our situation. Drops
  `NCCL_P2P_LEVEL=SYS` and `NCCL_CUMEM_ENABLE=1` (no-ops without peer access), keeps
  everything else.
- Measured: plain ~28-32 tok/s, DSpark n=5 without SYS 77.7 tok/s. That 77.7 is the most
  comparable 4-card number to what we can expect.
- Default util 0.85 (0.9/0.93 tested).

### 3. `kaka86mm/dsv4-vision-170hx`: alternative recipe on a different base

https://github.com/kaka86mm/dsv4-vision-170hx

- Base: `wtdcode/vllm-backport` branch `dsv4-vision-exp` + cherry-picks from upstream vLLM
  PR #54566 + 4 sm80 patches (`scripts/sm80-patches.py`, readable and small).
- PP4, 512k context, DSpark **n=3** (their engine requires n divisible by 3), util 0.93,
  partition `12,11,11,9`. 90-96 tok/s decode flat across 85k-178k depth; vision QA 0.4 s hot.
- Important bugs they documented, all plausible for any Vision-Exp build:
  - **CUDA-graph corruption with multimodal**: text queries return content from *unrelated
    requests*. Fixed by upstream PR #54566 commit `5ab628dd1` ("fix breakable cg").
    **We must test for this explicitly** (a "capital of France" style correctness probe after
    image traffic).
  - Vision MoE routing needs `input_ids` on every PP rank. Relay via `IntermediateTensors`
    (our hand-merge has this from wtdcode).
  - DSpark drafter needs the embedding table on the last PP rank (+~1 GB there). Our hand-merge
    **lacks** this.
  - Don't use PR #54566's latest head (merges upstream main, hard DeepGEMM gate, dead on sm80).
- leinasi's sweep also measured this "plan-2" engine and preferred their own.

### 4. `PixelML/club-170hx` + `PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX`: evidence + prebuilt images

https://github.com/PixelML/club-170hx (already cloned at `~/deepseek-v4.1-sm80/reference-repos/club-170hx`),
https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX

- Built on **our exact base commit** `f8ea5bb163c1…` + an 8-file SM80/DSpark patch set + a
  5-file vision port. **Prebuilt GHCR image** `ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902`
  (no compile needed).
- PP4, partition `11,11,11,10` (only partition that booted), DSpark k=6, util 0.90, 16k ctx.
  119 tok/s median text-only c=1 at 180W; 45 / 78 tok/s aggregate with an image at c=1 / c=2.
- **Weakest stability of the three**: crashes at c=4 (`shm_broadcast` EngineCore), and a
  DSpark speculator Triton illegal-access at a 131k prompt. Not our base candidate, but useful
  as (a) a quick no-build smoke test and (b) a source of 5 concrete vision boot bugs, listed in
  their Troubleshooting: missing `_plan_prompt_updates`, processor never returning
  `input_ids`, `process_weights_after_loading()` skipped, CUDA-graph capture nulling
  `input_ids` on non-first ranks, eager safetensors load exhausting host RAM.
- **Vision test method worth copying**: 64x64 two-colour gradient ("Name the two colors, left
  first"), plus no-image and wrong-image controls, which prove the model is actually reading
  the pixels rather than guessing. 10/10 image golden rows.

### Also seen, not directly useful

- `r0b0tlab/dsv4-flash-vision-exp-vllm-sm121`, `shige0501/dsv4-vision-spark`,
  `tonyd2wild/...-2x-DGX-Spark`: DGX Spark (Blackwell SM121). Different kernels. Their NIAH
  and vision-canary scripts (`scripts/vision_canary.py`) may be worth borrowing for tests.
- `AlexKGwyn/ds4-vllm`: AMD/ROCm cluster stack.
- `liangdabiao/*vision-rag`, `karminski/...video-input`: applications (PDF RAG, video
  frames). Usage patterns, not serving.

## What this means for our own hand-merged tree

`~/CMP-170HX-PROJECT/vllm-dsv4/vllm-vision-exp-experiment/` (wtdcode PR #58 merged onto our
0731 tree, 9 conflicts resolved, syntax-clean, never built). Compared with the forks above, it
is missing at least: DSpark drafter support (PR #58 drops it), the last-rank embedding for
DSpark, the multimodal CUDA-graph corruption fix, and leinasi's later correctness hardening.
**Shelved as a reference, not the build path.** Don't delete it; it's a useful diff when
debugging.
