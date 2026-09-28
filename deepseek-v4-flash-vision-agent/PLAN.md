# PLAN — fitting DSv4-Flash + vision on 3 cards (card 4 reserved for ComfyUI/Blender)

Hard constraint: whatever we build must run entirely on 3x CMP 170HX (192GB pool), leaving
the 4th card completely free and undisturbed for ComfyUI/Blender rendering. Not started —
this is the planning pass before any testing (per 2026-09-28 conversation: not ready to
test yet, want a real plan first).

## Project goal (Path B): match the 0731 GPU-fill methodology, not a fixed context number

Source repo: **`https://github.com/tim-odonnell/cmp170hx-deepseek-v4-flash`** (local
checkout: `~/cmp170hx-deepseek-v4-flash/`) — this is the actual project, not just the
`~/CMP-170HX-PROJECT/vllm-dsv4/` working copy. The authoritative production config is
`vllm-dsv4/phase5-launch-dspark-production.sh`, settled 2026-08-15 after a full
binary-search stress-test campaign (`PLAN-vllm-pp-dspark-3card.md` has the writeup):

```
IMG="dsv4-a100:devel"
--pipeline-parallel-size 3 --kv-cache-dtype fp8 --block-size 256
--gpu-memory-utilization 0.965 --max-model-len 691968 --max-num-seqs 8
--max-num-batched-tokens 2048 --no-enable-flashinfer-autotune
--tokenizer-mode deepseek_v4
VLLM_PP_LAYER_PARTITION=15,15,13
DSV4_LOGITS_ROW_CHUNK=64
VLLM_SPARSE_DENSE_QUERY_BLOCK=0
--speculative-config {"method":"dspark","num_speculative_tokens":5}   # DROP for Vision-Exp — confirmed unsupported, see below
```
Verified crash ceiling from that campaign: 0.965 util → 691,968 ctx passed the real
stress test (max-num-seqs 8, deep prefill at ~95% of ceiling); 0.97 → crashed.

**The goal, stated the way it was asked for**: run Vision-Exp through the *same* fill
methodology (same util starting point 0.965, same stress-test procedure — max-num-seqs 8,
deep prefill near whatever ceiling results, multiple cold restarts) rather than targeting
691,968 specifically. Record whatever context ceiling comes out the other side, whether
it's lower (vision tower eating headroom), the same, or higher (if Vision-Exp turns out
more memory-efficient some other way) than 0731's number. The metric is "same fill
discipline," not "same number."

**One correction this surfaces to the earlier plan**: the production script's own comments
say `VLLM_PP_LAYER_PARTITION` rebalancing was tested and **disproven** as a lever for the
crash ceiling — vLLM's memory log showed all 3 PP ranks always land at identical total
fill % regardless of partition; it only reshuffles weights vs. KV cache within an
already-fixed total budget. So `15,15,13` is still the right starting partition (same
43-layer backbone), but if Vision-Exp's ceiling comes out different from 0731's, don't
waste time trying to fix it via partition — the lever that actually matters is
`--gpu-memory-utilization` and `--max-model-len`, per 0731's own campaign.

**Also**: the `$SPEC` (DSpark) flag must be dropped entirely for this test — confirmed
earlier in this file that Vision-Exp's weight loader strips MTP/DSpark weights, so passing
`--speculative-config` for it would just fail to load rather than silently do nothing.

---

## Path A already satisfies this — no plan needed

Qwen3-VL-8B was always slated for the **WX9100**, a separate GPU entirely outside the CMP
cluster. 0731 keeps its existing, working 3-card vLLM config untouched
(`~/CMP-170HX-PROJECT/vllm-dsv4/phase5-launch-dspark-production.sh`). Card 4 stays free
automatically — this constraint doesn't add any work to Path A. If Path A ends up being
the choice, this PLAN.md is only relevant to Path B.

---

## Path B — real capacity findings (2026-09-28)

Checked directly against HF, not assumed:

| | 0731 (current, working on 3 cards) | Vision-Exp |
|---|---|---|
| Native checkpoint | 155.4 GiB (48 shards) | **156.31 GiB (48 shards)** — barely bigger |
| `num_hidden_layers` | 43 | **43 — identical** |
| Architecture class | `DeepseekV4ForCausalLM` | `DeepseekV4ForCausalLM` — identical |
| model_type | `deepseek_v4` | `deepseek_v4` — identical |

**Takeaway: this was never a raw-capacity problem.** Vision-Exp is 0731's exact same
43-layer backbone with an additive vision tower (separate config fields:
`vision_n_layers`, `vision_dim`, `vision_n_heads`, etc.) bolted on — not a bigger or
restructured decoder. If 0731 fits comfortably on 3 cards at `util=0.965`, Vision-Exp's
backbone should fit with very similar headroom. The "4-card only" limitation in
`wtdcode/vllm-backport#58` is about **where the sm80 porting work was tested**, not about
whether the weights fit on 3 cards.

**Working hypothesis, not yet verified**: our existing `VLLM_PP_LAYER_PARTITION=15,15,13`
is a strong starting point for a 3-rank Vision-Exp split too, since it's partitioning the
same 43 layers. Untested — the vision tower and its KV-cache-for-image-tokens overhead
might shift the right split slightly, and this needs to be checked, not assumed.

### Is Vision-Exp just 0731 + a swappable vision file? No — verified by hash (2026-09-28)

Compared every shard between the two repos directly (HF API, sha256 from the LFS blob
metadata, not just file sizes):

- Shard 1 differs in **size** (1.06GB in 0731 vs 1.99GB in Vision-Exp — where the vision
  tower/aligner/image-token embeddings live). Shards 2-48 are byte-*size*-identical.
- But **all 48 shards differ by SHA256 hash**, including the size-identical ones 2-48.
  Tensor *shapes* didn't change outside shard 1, but the actual *weight values* did,
  everywhere.
- **Why**: Vision-Exp was made via continued training on top of 0731, not "freeze the base,
  bolt on a vision tower." Continued training nudges every weight, even ones structurally
  unrelated to vision.
- **Implication**: not swappable in either direction. Can't drop Vision-Exp's shard 1 into
  an 0731 folder for a cheap hybrid, and can't strip vision back out by swapping shard 1
  back to 0731's — the rest of the shards are trained differently too. Two full separate
  150+GB downloads, not a shared base + add-on file. Any plan that assumed partial reuse
  between the two checkpoints on disk is wrong; budget full storage for whichever one (or
  both) we actually use.

### Is Vision-Exp actually better, not just "comparable"? Yes, on every shared benchmark

Real finding, not the DeepSeek's own "comparable on text" framing (checked llm-stats.com's
side-by-side, 2026-09-28): **Vision-Exp wins 6 of 6 directly shared benchmarks against
0731** — not just the vision-specific ones (ApexBench 36.5 vs 26.2, Agents' Last Exam
27.3 vs 25.2), but the **text-only agent ones too**: Terminal-Bench 2.1 (83.9 vs 82.7),
DeepSWE (59.3, edges past Opus-4.8's 58.0), DSBench-Hard (63.6 vs 59.6), NL2Repo,
AutomationBench. Zero benchmarks found where 0731 wins. Plausible explanation: continued
training itself often improves general capability somewhat, even when the target skill
being trained is unrelated (vision, here) — not a guaranteed outcome, but a known effect.

**Correction (2026-09-28)**: llm-stats.com's comparison claimed a max-output-token gap
(1,048,576 for 0731 vs 393,216 for Vision-Exp). Checked directly against both models'
actual `config.json` instead of trusting the aggregator: **`max_position_embeddings =
1048576` on both, identical**, same `rope_scaling` config too. The 393,216 figure was
almost certainly a hosted-API-provider's product policy (some providers cap max output
per call below the model's real context window as a serving decision, not a model limit)
— **not an architectural constraint, and irrelevant to us anyway since we self-host via
vLLM and control `--max-model-len`/generation limits directly.** No real output-length
trade-off found. (API pricing still shows Vision-Exp at ~3.7x cost per token on hosted
APIs — a genuine signal the vision tower adds real compute overhead, relevant to the
Phase 3 gpu-memory-utilization re-tuning question above, but that's a compute-cost
signal, not a capability limit.)

**Updated read**: with the output-token concern cleared, Vision-Exp looks like a
plausible across-the-board upgrade for text/agentic work too, not just "0731 plus a
vision tax" — no real downside found yet against 0731 on capability, only the unresolved
3-card porting/tuning work above.

---

## What's genuinely unresearched still (the real next steps)

1. ~~Read wtdcode's 3 actual sm80 fixes~~ **DONE (2026-09-28)**, via the real PR #58 diff
   (`api.github.com/repos/wtdcode/vllm-backport/pulls/58/files`), not just its description:
   - **Attention routing** (`nvidia/model.py::_select_dsv4_attn_cls`): keys off
     `device_capability.major == 8` — a per-GPU check, not rank-count-dependent.
   - **PP relay**: threads `dsv4_img_ids` through vLLM's standard `IntermediateTensors`
     dict, gated on `get_pp_group().is_first_rank`/`is_last_rank` — vLLM's own rank
     abstraction, identical at any PP size.
   - **Vision attention math** (bidirectional in-image visibility, `max_image_tokens`
     workspace sizing, `sparse_swa.py`): all per-token/per-request, orthogonal to topology.
   - **Verdict: none of the three fixes hardcode rank count.** This should be a config
     change (adapting `VLLM_PP_LAYER_PARTITION` etc. to 3 ranks), not a code change.
     Confirms the working hypothesis above.
   - **⚠️ Correction to earlier research**: the diff explicitly **removes** the old
     "does a speculative drafter need the target embedding on the last rank" logic
     (`_drafter_needs_target_embed`), because the vision wrapper's weight loader strips
     MTP/DSpark weights outright — source comment: *"The MTP/DSpark draft heads are not
     supported for the vision variant; drop their weights."* **The "88-92 tok/s DSpark"
     figure reported earlier (from the PR description, not the code) was wrong/
     misattributed — DSpark is not wired up for Vision-Exp in this fork at all.** Drop
     any plan that assumed DSpark would carry over from 0731's vision port.
2. **Vision tower's own memory budget is a new variable 0731 never had.** Image tokens need
   their own KV cache headroom on top of the existing text budget. Given 0731's own
   binary search already found a hard cliff (0.965 safe / 0.97 crashes) with zero vision
   overhead, adding vision-token KV cache could tighten that further — **budget for
   re-running the full binary-search campaign, don't assume 0.965 carries over.** (DSpark
   is moot here now — confirmed above it's not supported for the vision variant at all, so
   it's not an extra headroom variable to plan around.)
3. **PCIe Gen2 x4 bandwidth check for image token transfer** — untested whether feeding
   image patches through a 3-card pipeline-parallel split (image likely processed on
   whichever rank hosts the vision tower, then relayed) hits our already-known PCIe
   bandwidth ceiling harder than pure text tokens do. No data on this yet either way.
4. Confirm whether `sm80vllm` (still V4/GLM-5.3-Flash only as of last check) or
   `wtdcode/vllm-backport` is the better base to fork from for a 3-card attempt — the
   backport PR is closed/merged upstream into the fork, `sm80vllm` is the actively
   maintained one going forward. Lean `sm80vllm` for anything new, but confirm the
   Vision-Exp fixes actually landed there (last check: they hadn't — only V4/GLM-5.3-Flash).

---

## Proposed phases

1. ~~**Phase 1**: read the 3 sm80 fixes' source, confirm rank-count assumptions.~~
   **DONE 2026-09-28 — confirmed generic, see above.**
2. **Phase 2 (next actionable step)**: stand up Vision-Exp on the existing 3-card
   `vllm-dsv4` container/image, forking from `wtdcode/vllm-backport`'s patch (or
   `sm80vllm` once/if it picks up the same fixes — check again before starting, last
   check it hadn't). Start from `VLLM_PP_LAYER_PARTITION=15,15,13` as the initial guess,
   not from scratch — confirmed same 43-layer backbone as 0731.
3. **Phase 3**: re-run the `gpu-memory-utilization` binary search from first principles
   (don't inherit 0.965) — same methodology as 0731's port
   (`~/CMP-170HX-PROJECT/vllm-dsv4/PLAN-vllm-pp-dspark-3card.md` is the template to mirror).
4. **Phase 4**: smoke-test the actual hallucination/OCR concern (ik_llama issue #2464 —
   spreadsheet/table misreading) directly on our own quant/config before trusting it for
   real work — cheap to check early, expensive to discover after building a whole pipeline
   around it.
5. **Phase 5**: only after 1-4, wire into opencode (either as the main model directly, or
   alongside 0731 if we decide to keep both — TBD, not decided yet).

## Update log
- 2026-09-28: Plan created. Confirmed Vision-Exp's capacity math is a non-issue (near-
  identical checkpoint size, identical 43-layer backbone to 0731). Real unresearched item
  identified: whether wtdcode's sm80 fixes are rank-count-generic or 4-card-hardcoded —
  that's the actual gate on this plan, not capacity.
