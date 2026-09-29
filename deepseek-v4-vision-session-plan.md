> **CORRECTION (2026-09-29):** this planning doc says DSpark is "not supported" for Vision-Exp.
> **That's wrong.** The checkpoint ships the full DSpark drafter; only `wtdcode/vllm-backport` PR #58's
> code dropped it. Vision-Exp now runs with DSpark on 3 cards. See
> [`HOWTO-RUN-DEEPSEEK-V4-FLASH-VISION-EXP.md`](HOWTO-RUN-DEEPSEEK-V4-FLASH-VISION-EXP.md) and
> [`vision-exp-3card/`](vision-exp-3card/). Kept as the historical planning record.

# DeepSeek V4 / vision research session — plan (2026-09-28)

One research session, two threads that ended up connected: started by checking whether a
new DeepSeek release could replace the working [[deepseek_v4_vllm_pp_dspark]] setup, ended
with a concrete 3-card porting project for vision. This file is the narrative/plan tying
both together; the two project folders below have the full working detail.

- **Thread 1 detail**: `~/deepseek-v4.1-sm80/README.md`
- **Thread 2 detail**: `~/deepseek-v4-flash-vision-agent/README.md` + `PLAN.md`
- **Memory**: `deepseek_v41_sm80_tracker.md`, `deepseek_v4_flash_vision_exp_project.md`
- **Weekly automated check** (covers both threads): routine `trig_01JFxfSzgMLEWSsJorM7ULjb`,
  "DeepSeek-V4 SM80/Ampere weekly check (V4.1 + Vision-Exp)", Sundays 22:00 UTC

---

## Thread 1: is DeepSeek-V4.1-Flash a replacement for dsv4-flash-0731?

**Question**: a newer DeepSeek release (V4.1-Flash, 2026-09-10) existed — was it worth
switching to, or waiting for?

**Answer, arrived at in stages**:
1. V4.1-Flash is not a checkpoint refresh — a genuinely new architecture (40-layer Causal
   Encoder-Decoder, CSA2 sparse attention replacing DSA, new 203GB Engram lookup tables,
   different weight-quant block size). 510.31GB native (confirmed via HF API after an
   initial unit-confusion correction — GB vs GiB, both numbers were actually right, just
   different conventions) — ~2x the entire 256GB 4-card VRAM pool.
2. No working vLLM Ampere path exists (issue #57144, open, untouched). Mainline llama.cpp
   is conversion-only. The only working implementation anywhere is `ik_llama.cpp` — a
   different engine than the vLLM-only setup this rig runs.
3. Considered whether "wait for the non-experimental version" made sense (the user's
   framing, since Vision-Exp for the base V4 line is labeled "exp"). Turned out backwards:
   DeepSeek already *did* ship the non-experimental successor — it's V4.1-Flash itself,
   and it's currently *harder* to run on this hardware than the "experimental" Vision-Exp,
   not easier. There's nothing to wait for that isn't already shipped and still blocked.
4. Found and ruled out one bad lead: `StellarVoyager/DeepSeek-V4.1-Flash-A100-A800-SM80`
   claimed a full working Ampere solution with benchmarks, but contradicted the official
   open issue and had zero external corroboration — flagged as unverified, not used.
5. Set up a folder + weekly automated check to watch for real progress (vLLM issue,
   `sm80vllm`, mainline llama.cpp PR #28696, `ik_llama.cpp` follow-on work, an official
   Unsloth GGUF appearing) without relying on manually re-asking.

**Conclusion**: not viable now, don't wait for it. dsv4-flash-0731 stays the daily driver.
This thread is now a passive watch, not active work.

---

## Thread 2: giving dsv4-flash-0731 a vision capability

**Question, arrived at via Thread 1's dead end**: since V4.1 wasn't the path to vision,
what is? Goal stated by the user: DeepSeek-V4-Flash acting as a developer that reviews its
own work via screenshots and takes aesthetic direction from reference images.

**Two paths considered, real prior art found for both** (not hypothetical — searched for
and verified actual people/projects doing this):

- **Path A** — pair a separate small VLM (Qwen3-VL-8B, leaning on its strong OCR/document
  reading) via an opencode tool, leaving 0731 untouched. Found `oyale/opencode-screenshot-
  vision`, a real plugin, confirmed vLLM-compatible; read its full source rather than
  trusting the README summary, which surfaced both good news (auto-discovers vision-capable
  opencode providers, real prompt-injection defenses, path containment) and a real risk
  (silent cloud fallback unless `OPENCODE_VISION_BACKENDS=local` is set explicitly — same
  shape of risk as the logged [[feedback_local_model_auto_risk]] incident).
- **Path B** — swap to DeepSeek-V4-Flash-Vision-Exp, same lineage as 0731. Initially
  flagged real caveats (OCR/table hallucination reports, only 4-card-validated Ampere
  support). Then the user reframed around a hardware constraint — 3 cards only, 4th
  reserved for ComfyUI/Blender — which turned this from "maybe risky" into a concrete,
  scoped porting project once the actual numbers were checked:
  - Checkpoint size (156.31GB) and layer count (43, identical to 0731) confirmed directly
    via HF API/config — capacity was never really the blocker.
  - Read the actual `wtdcode/vllm-backport#58` diff (not just its description) — all three
    sm80 fixes are rank-count-generic, not hardcoded to 4 cards. Also caught and corrected
    a wrong secondhand claim in the process: DSpark is NOT supported for Vision-Exp in this
    fork (the earlier "88-92 tok/s DSpark" figure was from an AI-summarized description,
    not the code).
  - Verified by SHA256 hash (not just file size) that the two checkpoints are **not**
    swappable — every shard differs, because Vision-Exp was made via continued training on
    top of 0731, not a frozen base plus a bolted-on module.
  - Found Vision-Exp actually **wins all 6 directly shared benchmarks** against 0731,
    including text-only agent tasks — better than DeepSeek's own "comparable on text"
    framing suggested. One apparent trade-off (a smaller max-output-token cap) turned out
    to be an aggregator-site artifact, not real — checked both models' actual
    `max_position_embeddings` directly and they're identical (1,048,576).
  - Pulled the user's own real production config for 0731 on 3 cards from
    `github.com/tim-odonnell/cmp170hx-deepseek-v4-flash`
    (`vllm-dsv4/phase5-launch-dspark-production.sh`) as the literal starting baseline,
    rather than reconstructing it from memory.

**Project goal set**: port Vision-Exp to the same 3-card topology using the *same GPU-fill
methodology* 0731 used (util=0.965 starting point, the same real stress-test procedure —
not a fixed context-length target), so the outcome is honest either way: if Vision-Exp
lands on less usable context than 0731 (vision tower eating headroom) or more (if it turns
out more efficient), either result is a real answer, not a forced comparison.

**Status**: planning complete, not yet tested — explicitly deferred by the user until
ready. `PLAN.md` in the vision-agent folder has the phased breakdown (read the sm80 fixes
→ done; stand up the 3-card config next; re-run the utilization binary search; smoke-test
the hallucination concern early and cheap; only then decide on wiring into opencode).

---

## Corrections made along the way (worth keeping visible, not just fixed silently)

- 510GB vs 475GB checkpoint size: both right, GB vs GiB — verified via HF API rather than
  re-asserting either number.
- The "88-92 tok/s DSpark for Vision-Exp" claim: wrong, sourced from an AI-summarized PR
  description; the actual diff shows DSpark is unsupported for this variant. Caught by
  reading the real patch, not the summary.
- The "393,216 max output tokens" trade-off for Vision-Exp: wrong, an aggregator-site
  artifact; both models' real configs show identical 1,048,576 context.
- `VLLM_PP_LAYER_PARTITION` as a stability lever: the user's own 0731 production script
  already disproved this for the crash ceiling — surfaced while reading it for this plan,
  corrected the assumption in `PLAN.md` before it became a wasted testing cycle.

## Open items / next steps
1. Stand up Vision-Exp on the 3-card config (Phase 2 in the vision-agent `PLAN.md`).
2. Re-run the `gpu-memory-utilization` binary search from first principles.
3. Smoke-test the OCR/table hallucination concern before investing further.
4. Decide Path A vs. B only after Path B's real numbers exist — no decision forced yet.
5. Weekly routine keeps watching both threads passively; check in on it or let it surface
   findings on its own schedule.
