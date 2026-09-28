# DeepSeek-V4.1-Flash on SM80 (Ampere) — research tracker

Goal: find or build a real way to run DeepSeek-V4.1-Flash on Ampere GPUs (our 4x CMP 170HX,
GA100 sm80), via vLLM. We do NOT use llama.cpp for DeepSeek — vLLM only.

**Status as of 2026-09-28: NOT SOLVED YET.** No verified working solution exists. This folder
tracks the one authoritative open issue, the specific technical blockers, and adjacent
community projects worth watching (but none of them currently claim V4.1 support).

Contrast: our current model, DeepSeek-V4-Flash-0731, already runs great on this hardware via
vLLM (see main memory: `deepseek_v4_vllm_pp_dspark.md`, `~/CMP-170HX-PROJECT/vllm-dsv4/`).
V4.1-Flash is a *different architecture* (Causal Encoder-Decoder, not decoder-only MoE), so
those existing patches don't carry over — this is genuinely new porting work, not a rerun.

---

## Authoritative source — the real status

- **[vLLM Issue #57144 — SM8x (Ampere A100/A800) support for DeepSeek-V4.1-Flash](https://github.com/vllm-project/vllm/issues/57144)**
  Filed 2026-09-16 by thomaslwang. As of our last check: **open, zero comments, no
  activity since filing.** The issue author explicitly states there is currently **no
  Ampere (sm_80) path** for V4.1-Flash, and lays out the blockers themselves (see below).
  **This is the one to watch/subscribe to.** No ETA posted anywhere.

### The three specific technical blockers (from the issue itself)
1. **Weight quantization block size changed**: V4.1 uses `weight_block_size: [32,32]`
   vs. V4-Flash-0731's `[128,128]`. Our existing Marlin-based MXFP4 repack patches
   (`~/CMP-170HX-PROJECT/vllm-dsv4/deepseek-v4-cmp170hx/patches/`) were built around the
   128x128 block size — they don't apply to 32x32. Needs new kernel work via
   `ModelOptLinearMethod` variants.
2. **Engram tables** — a new 203GB n-gram lookup component (384M rows x 256) that has
   *no equivalent at all* in V4-Flash-0731. Hashing kernels are "plain-integer Triton"
   (should be architecture-agnostic-ish) but the embedding gather layer needs FP8 support
   that hasn't been ported to sm80.
3. **Vision tower + DeepSelect pre-filter** gating (`candidate_topk_blocks: 2048,
   candidate_block_size: 8`) riding on the indexer — another new subsystem, no V4 equivalent.

---

## Unverified / do not trust yet

- **[StellarVoyager/DeepSeek-V4.1-Flash-A100-A800-SM80 (Hugging Face)](https://huggingface.co/StellarVoyager/DeepSeek-V4.1-Flash-A100-A800-SM80)**
  Claims a full working solution on 8x A800: benchmarks (3,040 tok/s short-context,
  1,945 tok/s sustained long-context, 87/500 SWE-bench Verified effort75), built on
  `wtdcode/vllm-backport`. **Red flags:** contradicts the official vLLM issue (which says
  no Ampere path exists, filed *after* this repo would need to have been proven), cites
  "redacted evidence" instead of reproducible logs, zero external corroboration (not
  referenced on the vLLM issue, no HF discussion, no reddit mentions found). Does not
  address any of the three specific blockers above (quant block size, Engram, vision
  tower) — just serving/launch config. **Treat as unverified marketing-shaped content
  until someone independently reproduces it. Do not build on this without re-checking.**

---

## Adjacent real projects (proven for V4-Flash, NOT yet confirmed for V4.1)

These are legitimate, working, community-maintained — worth watching since they're the
most likely place V4.1 support would eventually land, given they already solved the
equivalent V4-Flash problem.

- **[PixelML/sm80vllm](https://github.com/PixelML/sm80vllm)** — "vLLM for NVIDIA Ampere /
  SM80 (CMP 170HX, A100, RTX 3090): SM80 fallbacks for DeepSeek-V4 and GLM-5.3-Flash,"
  maintained by the "club-170hx community." Forked directly from `vllm-project/vllm`
  (not related to our `haosdent/vllm` base). Actively pushed (as of 2026-09-28). Currently
  scoped to V4/V4-Flash-Vision-Exp and GLM-5.3-Flash — **no V4.1 mention found yet.**
  Worth checking back periodically.
- **[wtdcode/vllm-backport PR #58](https://github.com/wtdcode/vllm-backport/pull/58)** —
  merged, real, validated on our exact hardware (4x CMP 170HX, GA100 sm80, PP rank 4).
  Adds DeepSeek-V4-Flash-**Vision-Exp** (old architecture) support with 3 sm80 fixes:
  PP input relay, embedding table placement, SM80 attention routing. This is the
  legitimate ancestor StellarVoyager's repo claims to build on — but Vision-Exp is the
  old (V4) architecture, not V4.1's CED/CSA2. Does not by itself imply V4.1 support.
- **[Lasimeri/vllm-dsv4-ampere](https://github.com/Lasimeri/vllm-dsv4-ampere)** —
  "DeepSeek-V4-Flash on Ampere SM 8.6 via vLLM (pyref kernel replacements)." Same
  situation — real, but scoped to V4-Flash, not V4.1.
- **[vLLM Issue #50576](https://github.com/vllm-project/vllm/issues/50576)** — the
  original (already-resolved-in-practice-by-community-forks) Ampere-support request for
  V4-Flash-0731/DSpark. Useful as a template for how #57144 might eventually get resolved
  the same way (community fork first, official merge later/maybe never).

---

## Local reference clones

Shallow, code-only clones (no weights, no LFS) kept in `reference-repos/` for offline lookup
during experiments:

- **`reference-repos/sm80vllm/`** — `PixelML/sm80vllm`, `sm80` branch, depth 1 (154M).
- **`reference-repos/club-170hx/`** — `PixelML/club-170hx` (the real upstream — confirmed via
  GitHub API it's not a fork; `mcx/club-170hx` is a fork *of* this one, so this is the one to
  track), depth 1 (69M). Community recipes/notebooks/diagnostics specifically for our exact
  card (CMP 170HX), independent of the V4.1 question — useful general reference too.

**Full-text search of both, done locally (2026-09-28) — confirms the web search wasn't
missing anything:**
- No real V4.1/CSA2/Engram/Causal-Encoder-Decoder content in either repo. The apparent
  hits in club-170hx notebooks were false positives (base64 plot-image blobs inside the
  `.ipynb` JSON coincidentally matching the regex, not real text).
- One tangentially interesting file: `sm80vllm/csrc/libtorch_stable/ngram_embedding_kernels.cu`
  — a hashed n-gram embedding lookup kernel, but it's adapted from SGLang for **LongCat-Flash's**
  n-gram embedding variant, a different model. Same general *shape* of problem as V4.1's Engram
  (hash-based lookup table, GPU kernel) but not a solution to it, and not DeepSeek-specific.
  Worth a look if/when someone starts implementing Engram support, not proof of progress.

Skipped cloning: `StellarVoyager/...` (flagged unverified above — not worth the disk until
independently corroborated) and `wtdcode/vllm-backport` (its relevant fixes are already folded
into `sm80vllm`'s history, so it's redundant to clone separately).

---

## llama.cpp/ik_llama runtime progress (not our engine, but the leading indicator)

We don't use llama.cpp for DeepSeek — vLLM only. But this is the fastest-moving real
progress on V4.1 anywhere, and it's the direct gate on both "when does Unsloth ship a
GGUF" and "is the CSA2/Engram architecture understanding proven correct yet" (which
matters for a future vLLM sm80 port too, since a proven-correct reference implementation
lowers the risk of anyone attempting that).

- **[ggml-org/llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)** —
  mainline. **Conversion-only as of last check** ("these files do not run on upstream
  llama.cpp yet"). This is what actually gates an official Unsloth GGUF release.
- **[ikawrakow/ik_llama.cpp PR #2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455)**
  — **MERGED 2026-09-21** (11 days after V4.1's Sept 10 release). Real, working `deepseek41`
  runtime: reuses the V4 loader/cache, adds the 4 real deltas (shared compressed streams,
  lagged hyper-connections, query-norm changes, Engram lookup). **PPL-validated**: 2.2355 ±
  0.0626 on wikitext-2, matching the reference. This is the credible one.
- **[ikawrakow/ik_llama.cpp PR #2512](https://github.com/ikawrakow/ik_llama.cpp/pull/2512)**
  — **MERGED 2026-09-26.** Adds an opt-in *alternate* graph path (`V41_SEPARATE`) with a
  hierarchical sparse indexer. **Real caveat: no hash-layer expert routing** — falls back
  to generic routing, which the actual trained model doesn't use, so output fidelity on
  this path specifically is suspect. ~5-10% slower than the #2455 path at large context.
  Don't treat this one as equivalent to #2455.
- **[ikawrakow/ik_llama.cpp PR #2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522)**
  — **MERGED 2026-09-25** (day before #2512, "second half" continuation of #2455).
  **Real, working DSpark speculative decoding for V4.1** — handles V4.1's lagged
  hyper-connections and the draft model's missing `output_hc_*` heads via a new
  `dflash_dsv41` flag. Measured: 60% acceptance at n_max=3 → 1.41x speedup (Sept 13),
  refined to 41.6%@n_max=5 / 62.8%@n_max=3 (Sept 25). Real draft checkpoint exists on HF:
  `DeepSeek-V4.1-Flash-DSpark` (7.97GB). **This is the opposite of what happened on the
  mainline-llama.cpp/Vision-Exp track, where DSpark was reported broken (no speedup) —
  ik_llama built V4.1 DSpark support in working from the start.**
  **Caveat worth watching**: acceptance rate was reported "below break-even" on "certain
  hardware" initially, fixed by switching to bf16 target embeddings. Unclear if "certain
  hardware" means Ampere specifically or something else — bf16-vs-fp8 embedding precision
  is exactly the kind of thing that could hit sm80 (no native FP8 tensor cores)
  differently than newer cards. Worth re-checking if we ever get to testing this directly.

- **[JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF)**
  — the actual draft-model checkpoint (14B params, 7.97GB, same size as the one #2522
  tested — likely the same weights). **Important: this is a SEPARATE, THIRD fork** —
  requires the `dsv41-porte` branch of `JigSawPT/llama.cpp`, not ik_llama.cpp and not
  upstream. So there are now at least two independent community runtimes with working
  V4.1+DSpark (ik_llama.cpp's #2522, and JigSawPT's `dsv41-porte`) — still fragmented,
  not consolidated anywhere. **Real benchmark numbers here are more sobering than #2522's
  headline 1.41x**: acceptance rate 51% (prose) / 79% (code) / 97% (repetition), and
  measured real speed on RTX 5090 was roughly **neutral to slightly worse** (4.92 vs 5.13
  tok/s) on mixed content — the speedup story is content-dependent, not a clean universal
  win. Low adoption so far (3,550 dl/mo, 4 likes) — early/niche.
- **Unsloth**: no `unsloth/DeepSeek-V4.1-Flash-GGUF` repo exists yet (checked their HF org
  listing directly). Their historical pattern (same-architecture checkpoint = hours,
  new-architecture = ~9 days after open weights once mainline runtime lands) suggests
  they're waiting on #28696's runtime half, not doing independent R&D.

---

## Ideas / angles not yet explored

- Check whether `ModelOptLinearMethod`'s 32x32 block kernels have any prior art from
  *other* models using the same NVIDIA ModelOpt quantization scheme — this blocker might
  not be DeepSeek-specific, worth searching NVIDIA ModelOpt's own repo/issues for sm80
  32x32 block support independent of DeepSeek.
- The Engram FP8 embedding-gather blocker might be solvable by falling back to a
  non-FP8 (bf16/fp16) embedding gather path at a VRAM cost — worth checking if this is
  configurable rather than hard-coded, since our 256GB VRAM pool has more headroom than
  most sm80 setups being discussed.
- Watch `PixelML/sm80vllm` commit history directly (not just search results) for any
  V4.1 branch/PR activity — it's the most likely candidate to get there first given it
  already has 2/3 blockers' problem domain experience (sm80 MoE + vision).

---

## Update log
- 2026-09-28: Folder created. Confirmed #57144 open/unsolved, flagged StellarVoyager repo
  as unverified, catalogued adjacent real projects. No working solution found yet.
