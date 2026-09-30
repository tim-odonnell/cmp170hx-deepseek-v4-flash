# Vision-Exp 3-card campaign log (append-only, every attempt incl. failures)

## 2026-09-29 09:06-09:42 — Phase 1 build: OK

`dsv4-vision:sm80-leinasi-4fe10bf`, ~36 min (≈20 min compile + ≈8 min layer export).
Details and pins: `results/build-record.txt`. Gate must run WITH GPUs (`--runtime=nvidia`); a
GPU-less gate fails on a Triton-disabled import, which is not an image fault.

## 2026-09-29 09:43 — Phase 2 boot 1: FAILED at load (PP2 illegal memory access, Xid 31)

Config: util 0.93, maxlen 16384, seqs 2, partition 15,15,13, batch 2048, DSpark n=5, leinasi env
set incl. `VLLM_MARLIN_FP8_DEQUANT_BF16=1`.

- PP0 loaded 54.04 GiB, PP1 52.81 GiB. That's **+2.37 / +2.36 GiB vs 0731** (51.67 / 50.45).
  The identical growth on both ranks means a setting, not the vision tower.
- PP2 died inside the **DSpark drafter's Marlin MXFP4 repack**
  (`load_dspark_model` → `prepare_moe_mxfp4_layer_for_marlin` → `marlin_permute_scales`):
  `CUDA error: an illegal memory access`. dmesg: Xid 31 MMU fault (REGION_VIOLATION
  VIRT_WRITE) on `c1:00`. Same step 0731's notes flag as memory-sensitive on the last rank.
- Cause: `VLLM_MARLIN_FP8_DEQUANT_BF16=1` (from leinasi's 4-card profile) dequantizes the
  block-FP8 dense weights to BF16 at load for faster cuBLAS GEMMs, costing the FP8→BF16 byte
  difference in VRAM (`vllm/model_executor/kernels/linear/scaled_mm/marlin.py`). On 4 cards
  there's room. On 3 cards it pushes PP2 over the edge during the repack.
- ~~Cards recovered on their own~~ **Wrong, corrected below:** `nvidia-smi` showed 0 MiB / P0,
  but the relaunch's wedge probe actually FAILED (output hidden by my `tail -3`). See boot 2.
- **Fix: default it OFF** in `launch-vision-3card.sh` (0731 never used it). Kept as an opt-in
  (`MARLIN_DEQUANT=1`) to test the speed trade later, if the fill budget ever allows.

## 2026-09-29 09:47 — Phase 2 boot 2 (BF16 dequant off): FAILED before load. Driver needs reboot

Never reached the model. `cuInit` segfaulted in vLLM's registry-inspection subprocess ("CUDA
driver initialization failed").

**Root cause: my launcher's auto-recovery, not the model.** Boot 1's Xid 31 left the driver
unhealthy, so the relaunch's wedge probe failed and the launcher silently ran the recovery
block copied from the 0731 scripts (`nvidia-smi -r` + `rmmod`/`modprobe nvidia_uvm`). I missed
the "GPUs wedged" line because I only displayed the last 3 lines. dmesg 09:48:03: Xid 31 from
`modprobe`, then `uvm encountered global fatal error 0x60, requiring os reboot`, then **Xid 154
(recovery action: OS Reboot) on all 4 cards**, GPU 3 included. CUDA init now fails on every
card. `nvidia-smi` still looks normal (0 MiB, P0), so it can't be trusted as a health check.

- 0731 artifacts re-verified unchanged vs `results/0731-protection-snapshot-*.txt` (image id,
  source tree, production script, model size). 0731 is unharmed on disk; it just needs a
  rebooted driver like everything else.
- **Fix:** launcher now FAILS FAST on a bad probe with "reboot" instructions, and never
  reloads modules. CARD-REGISTRY 2026-08-15 had already warned that module reloads can break
  CUDA on all cards. Lesson re-learned the hard way.
- Rule for the rest of the campaign: after any Xid 31, check `dmesg` and reboot before the
  next launch. Expect roughly one reboot per crash while tuning near the fill edge.

## 2026-09-29 10:55 — reboot clean
0 Xid in dmesg, CUDA init OK on GPUs 0-2, 100 W caps reapplied by the systemd service.

## 2026-09-29 10:56 — Phase 2 boot 3 (util 0.93, BF16 dequant off): clean refusal, no KV room
All ranks loaded (no crash, no Xid), then vLLM's pre-check: `ValueError: No available memory for
the cache blocks`. Same story as 0731 (DSpark needed ≥0.96 there too).
Weights vs 0731: PP0 52.59 (+0.92), PP1 51.36 (+0.91), PP2 57.15 (+0.91) GiB. Uniform ~+0.91 GiB
on EVERY rank, so it's something this fork/engine allocates everywhere, not the vision tower
(which only lives on PP0). **Open question: find it.** It costs KV headroom on every rank.

## 2026-09-29 11:00 — Phase 2 boot 4 (util 0.965 = 0731 value, maxlen 16384, seqs 2): HEALTHY
- Healthy in ~3.5 min. No Xid. GPU 3 at 0 MiB throughout.
- vLLM per rank (consumed / peak act / graphs, budget 61.3 GiB): PP0 53.97/0.82/0.16,
  PP1 52.74/0.80/0.16, PP2 58.57/0.72/0.18. KV available: PP0 6.21, PP1 7.45, **PP2 1.69 GiB**
  (0731 PP2 ≈ 1.82). KV pool reported 31,080 tokens (0731: 804,497 at maxlen 691,968; not
  comparable yet at maxlen 16k, since this arch's SWA cost is per request. Phase 4 decides).
- nvidia-smi idle: **58,040 / 56,828 / 62,786 MiB**, vs 0731 idle 58,160 / 57,024 / 63,092.
  Already at or slightly under 0731's fill.
- **Phase 3 vision gates: 9/9 PASS** (`results/phase3-vision-gates-boot4.json`,
  `results/G5-screenshot.png`): gradient GREEN, ORANGE; no-image control didn't guess;
  wrong-image YELLOW, PURPLE; table 25/25 cells exact (ik_llama #2464 hallucination NOT
  reproduced); real Firefox screenshot: heading, 3 cards, orange "Export report" button all
  right, and it found the planted bug ("Revenue label on top of the value"); two-image pick
  right; cross-request leak check clean (Paris/Tokyo/Rome/Ottawa, no colour words); garbage
  image → HTTP 400, server fine; c=1/2/4 OK. **Caveat:** seqs=2, so c=4 partly queued.
  Re-test at seqs=8.
- **Speed (same script as 0731):** text mean **83.1** / median 80.8 tok/s (0731: 99.7 / 95.5);
  with an image attached mean 90.0 / median 90.5. **About 15-17% behind 0731 on text so far**
  (5 runs, DSpark variance is large, so needs more runs).
- DSpark acceptance is healthy: 32-43%, mean accepted length 2.6-3.2 (0731 Aug: 2.0-2.7). So
  the gap is per-step forward cost, not the drafter. Prime suspect: BF16 dequant off (fork
  comment: cuBLAS-on-BF16 beats Marlin at every M on A100), which we can't afford on 3 cards
  at full size. Others to test in Phase 5: `VLLM_SPARSE_DENSE_QUERY_BLOCK` 4 vs 0, attention
  backend path, and `VLLM_MARLIN_FP8_DEQUANT_EXCLUDE` (dequant only SOME layers: partial
  speed for partial VRAM).

## 2026-09-29 11:10 — Phase 4 attempt 1 (unpatched image, 0.965 / 691,968 / seqs 8): clean refusal
PP2 needed 2.16 GiB KV for 691,968, had 1.26 → short 0.90 GiB, ≈ the unexplained +0.91 GiB/rank.

## 2026-09-29 11:12 — ROOT CAUSE of the +0.91 GiB/rank: vision tower built on EVERY PP rank
- Checkpoint headers compared tensor-by-tensor (`ckpt_sizes.py`): all 43 layers, embed, head,
  mtp **byte-identical in size** to 0731. The only extra is `vision` 0.767 + `aligner` 0.102 =
  **0.869 GiB** (BF16).
- The fork's `DeepseekV4ForConditionalGeneration.__init__` builds ViT + aligner whenever images
  are enabled, with no PP-rank check. The v1 GPU model runner only runs the encoder, its cache and
  its profiling on the first PP rank (`model_runner.py` lines 209/722/1374 gate on
  `is_first_pp_rank`), and the DSpark (dflash) speculator declares `supports_mm_inputs=False`.
  So PP1/PP2 each carried a dead 0.87 GiB copy.
- **Patch 0001** (`patches/0001-*.patch`, branch `cmp170hx-3card` in the fork clone): build
  ViT/aligner only when `get_pp_group().is_first_rank`; drop `vision.`/`aligner.` weights on
  other ranks via the fork's existing mapper-to-None mechanism; keep the tiny `image_*`
  sentinel vectors on all ranks (used by `embed_input_ids`). Worth upstreaming to leinasi2014:
  it also frees 0.87 GiB on 3 of their 4 PP4 ranks.
- Image `dsv4-vision:sm80-leinasi-4fe10bf-p1` = thin overlay (`overlay/Dockerfile`) copying
  the patched `vl_model.py` into the editable install. sha256 verified in-image. Native code
  untouched.

## 2026-09-29 11:13 — Phase 4 attempt 2 (p1, 0.965 / 691,968 / seqs 8): HEALTHY
- Loaded: PP0 52.59 (0731 +0.92 = the tower, correct), **PP1 50.45, PP2 56.24: identical to
  0731.** KV pool **797,299 tokens** (0731: 804,497 → 99.1%). Max concurrency at 691,968: 1.15x.
- Vision gates 9/9 PASS at seqs 8 (G9 c=4 now truly concurrent).
- **Stress: 629,705-token real-text request + vision gates running concurrently: SURVIVED.**
  471 s wall (0731: 480 s). Healthy after, 0 Xid. Answered correctly (listed the first three
  FILE headers from ~630k tokens back; 0731 echoed the question in its baseline run).
  Mid-prefill gate round: G1/G4/G5/G7/G9 all PASS (only 1 round fit; gates queue behind the
  prefill chunks).
- **Fill vs 0731 peaks (MiB): GPU0 60,722 vs 59,456 (+1,266, vision tower + encoder work
  memory); GPU1 58,410 vs 58,338 (+72); GPU2 64,504 vs 64,370 (+134, +0.2%).** Slightly OVER the
  brief's "equal or slightly less" on every card, so not yet the pick. Next: cold restarts
  at 0.965, then the same search + stress at 0.96 for a strictly-under option.

## 2026-09-29 11:35 — my error: 3 "cold restarts" ran the UNPATCHED image (forgot IMG=…-p1)
All three cleanly refused (KV check, no Xid). Fixed at the source: launcher default IMG is now
`dsv4-vision:sm80-leinasi-4fe10bf-p1`.

## 2026-09-29 11:39-11:49 — 0.965 / 691,968 / seqs 8: 3 cold restarts, all HEALTHY
KV pool 797,299 tokens each time; G1 + G7 PASS after each; 0 Xid.
**0.965 passes the full 0731 method** (ceiling = 0731's 691,968, seqs-8 deep stress with
images, 3 cold restarts). Fill: GPU2 +134 MiB vs 0731 peak.

## 2026-09-29 11:52-12:07 — util 0.96 search + stress
- 691,968: refused (2.16 needed / 1.85 avail; estimate 468,224, which ran low again).
- 574,464 (0731's 0.96 ceiling): refused by a hair (1.99 / 1.99); estimate 572,160.
- **572,160: HEALTHY.** KV pool 658,562 tokens. 0731 at 0.96: 574,464 (−0.4%).
- Stress: 534,953-token request (0731's 0.96 test depth: 534,950) + concurrent gates:
  **SURVIVED**, 383 s, correct answer, healthy, 0 Xid. Mid-prefill gates G1/G4/G5/G7/G9 PASS.
- **Peaks (MiB) vs 0731@0.965: GPU0 60,388 (+932), GPU1 57,638 (−700), GPU2 63,680 (−690).**
  GPUs 1-2 strictly under 0731. GPU0 above only because it hosts the vision tower (0.87 GiB
  weights + encoder working memory) that 0731 doesn't have; it's the roomiest card (92%).

## 2026-09-29 12:11-12:21 — 0.96 / 572,160 / seqs 8: 3 cold restarts, all HEALTHY
KV pool 658,562 each time; G1 + G7 PASS after each; 0 Xid. **0.96 also passes the full method.**

## Phase 4 outcome: two validated configs (decision pending with user)
| | util | max ctx | KV pool | peak MiB GPU0/1/2 vs 0731 (59,456/58,338/64,370) |
|---|---|---|---|---|
| A | 0.965 | 691,968 (= 0731) | 797,299 | +1,266 / +72 / +134 |
| B | 0.96 | 572,160 (−17%) | 658,562 | +932 / −700 / −690 |
GPU0's excess in both is the vision tower 0731 lacks; it can't go below 0731 on GPU0 without a
large context cut (~0.945 util).

**User decision 2026-09-29 12:30: config A (0.965 / 691,968 / seqs 8) is the default.** Now the launcher default.

## 2026-09-29 13:00-13:30 — Phase 5: where the speed gap comes from
Same script as the 0731 baseline (`speed-bench.py`, 300 tok, 10 runs, warmup discarded).
| run | mean | median | DSpark mean accepted len |
|---|---|---|---|
| 0731 on its own engine (Phase 0, 5 runs) | 99.7 | 95.5 | n/a |
| **Vision-Exp config A** (run 1 / run 2) | **86.5 / 86.3** | 86.5 / 86.0 | ~2.92 (2.81-3.03) |
| Vision-Exp A + BF16 dequant on PP0/PP1 layers only | 81.3 | 82.3 | n/a |
| **0731 checkpoint on the new engine** (same launcher, same util/ctx) | **107.6** | 98.6 | ~3.53 (2.91-4.02) |
- **Partial BF16 dequant = no gain** (81.3 vs 86.5, within noise) while costing ~1.6 GB idle on
  GPU0/GPU1. Stays off. Mechanics, for the record: `VLLM_MARLIN_FP8_DEQUANT_EXCLUDE` is a
  substring match on layer prefixes; the DSpark drafter's layers are named `layers.43-45` +
  `main_proj` + `markov_head` (dspark.py:94), so excluding PP2 needs `layers.30.`…`layers.45.`,
  `mtp`, `main_proj`, `markov_head`. Missing 43-45 put +0.30 GiB on PP2 and lost the 691,968 fit.
- **The new engine is not slower**: the 0731 checkpoint on it matches or beats 0731's own engine.
- **Per-step rate is the same**: tok/s ÷ accepted length = 29.5 steps/s (Vision-Exp) vs 30.5
  (0731 ckpt). **~95% of the gap is DSpark draft acceptance**, a property of the Vision-Exp
  checkpoint's drafter on these prompts (2.9 vs 3.5 tokens accepted per step), not the engine,
  the patch, or config. `num_speculative_tokens` is pinned at 5 (checkpoint block size 5 is the
  floor; 6/7 hang on this fork), so there's no config lever for it.
- With an image attached, decode measured 90.0 mean (Phase 2), no slower than text.

## 2026-09-29 13:50-14:15 — Phase 5b: decode + prefill at depth (`depth-decode.py`, 3 runs/depth, prefix cache)
| depth | Vision-Exp decode | 0731 decode | Vision-Exp prefill | 0731 prefill |
|---|---|---|---|---|
| short (speed-bench) | 86.3 | 99.7 | — | — |
| ~100k (99,724) | 78.1 | 80.3 (−3%) | **2,154 tok/s** | 1,935 (Vision +11%) |
| ~300k (299,806) | 68.4 | 79.1 (−14%) | **2,384 tok/s** | 2,282 (Vision +4%) |
Near parity on decode at 100k; Vision-Exp prefills faster at both depths. 0731 was run via
`scripts/switch-to-0731.sh` and Vision-Exp relaunched after, so switching was exercised both ways.

## 2026-09-29 14:02 — thermal logging added (user request)
`scripts/thermal-logger.sh` (read-only): every 5 s all 4 cards' core + HBM temp, power, HW/SW
thermal-slowdown flags, 6 chassis fan RPMs (BMC via ipmitool) and gpu-fan-daemon duty → CSV in
`logs/thermal-*.csv`. Alerts: core ≥80 °C, HBM ≥85 °C, any thermal slowdown, fan <300 RPM.
Card limits (nvidia-smi -q): core max operating 85 °C / slowdown 95 °C; HBM max operating 95 °C.
The cards have no fans of their own; gpu-fan-daemon.service drives the chassis fans from the
hottest card's core/HBM temp. First 10 min (0731 depth runs): max core 66 °C, HBM 75 °C, 0 alerts.

## 2026-09-29 14:16 — Phase 6 soak started (`scripts/soak.sh 9000`, config A, 2.5 h)
Soak result: **20/20 loops passed in 38 min** (2× deep630k, 6× deep250k, 20× full gate suite +
6-request bursts), 0 Xid, 0 thermal alerts. **Stopped early at the user's call**, 1h52m short
of the planned 2.5 h. Server stayed healthy after the mid-prefill abort.

## 2026-09-29 14:56 — Phase 6b: tool calling + long output + web research: 7/7 PASS
`scripts/tool-tests.py` → `results/phase6b-tool-tests.json`. Chat-template defaults (as
opencode/Open WebUI call it), `deepseek_v4` tool + reasoning parsers.
- T1 single call (right fn, valid JSON, finish=tool_calls) · T2 uses the tool's value ·
  T3 no tool for 17×23 · T4 two lookups + comparison · **T5 vision→tool** (read table image,
  called tool for P-299, the max-Qty row) · T6 4,371-token guide, finish=stop, 96.5% unique 4-grams,
  75.8 tok/s.
- **T7 real-world corruption research (1MDB) via SearXNG**: 6 searches + 9 page fetches
  (Wikipedia, Reuters ×2, BBC, Guardian, PBS, Britannica), all 5 fact checks (Najib, Jho Low,
  Goldman, $ billions, legal outcomes), **0 invented sources** (every cited URL came from its
  own tool results), 0 malformed tool-call JSON. Ran while Google/Startpage were CAPTCHA-
  suspended in SearXNG and results were noisy (Bruno Mars/Verizon pages); it filtered them.
- SearXNG had been stopped since ~2026-09-26; started for T7, user said leave it running.

## 2026-09-29 15:05 — TROUBLESHOOTING-XID31.md + scripts/capture-crash.sh (user request)
Debug starting point for future crashes. Key insight from re-reading leinasi's write-up + the
indexer source: their "Xid 31 after 67 min" is triggered by **context length (~174k, first long
request), not uptime**. The unchunked indexer logits buffer is 2048 × ctx × 4 B ≈ 1.33 GiB at
174k, about all of the last rank's slack. Our tests crossed 174k nine times with ROW_CHUNK=64, clean.

## 2026-09-29 21:00 — image cap raised 2 → 8 per conversation (`IMAGES` in the launcher, default 8)
Why: an opencode agent reviewing its own screenshots (snowboard step 2) hit the 2-image cap. vLLM counts
every image in the conversation history, so the session could never make another request.
- Boot: identical to config A. KV pool 797,299, per-rank consumed 54.63/52.5/58.43 GiB, idle VRAM
  59,138/57,006/63,090 MiB. Encoder profiling budget is set by max-num-batched-tokens (2048 tokens),
  not by the per-conversation cap.
- Stress: 630k request with 6-image requests + gates G1/G6/G7 running concurrently (3 rounds).
  6 solid colours named correctly in order every time, gates PASS, 0 Xid.
  **Peak 60,292 / 58,214 / 64,264 MiB**: slightly lower than config A's first stress, and GPUs 1-2 under
  0731's peaks (58,338 / 64,370).
