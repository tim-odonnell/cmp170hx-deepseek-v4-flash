# Troubleshooting: the leinasi Xid-31 crash (and its look-alikes)

**Start here if Vision-Exp (or 0731) crashes.** This covers the crash the leinasi2014 fork's
authors spent days chasing, why it happens, how to check we're protected, how to re-test for
it, and how to tell it apart from the other crashes seen on these cards.

Written 2026-09-29, from leinasi2014's incident write-ups
(`docs/cmp170hx/XID31-ROOT-CAUSE-2026-09-03.md` and `OPTIMIZATION-2026-09-03.md` in the fork,
cloned at `~/CMP-170HX-PROJECT/vllm-dsv4/vllm-vision-leinasi/`), the engine source, and our own
testing today.

---

## 1. If it crashed just now: do this first

1. **Don't restart anything yet.** Capture the evidence first:
   ```bash
   cd ~/cmp170hx-deepseek-v4-flash/vision-exp-3card
   ./scripts/capture-crash.sh            # read-only; saves results/crash-<time>/ + prints TRIAGE
   ```
   It saves the container log, the exact env/flags the container ran with (`docker inspect`),
   the kernel's NVIDIA/Xid messages, GPU state, and the last ~100 s of the thermal log.
2. **Never run `rmmod`/`modprobe nvidia_uvm` on these cards.** On 2026-09-29 that turned one
   Xid 31 into "UVM global fatal error" + **Xid 154 (OS reboot required) on all 4 cards**.
   `nvidia-smi` still looked healthy afterwards; it lies in this state.
3. **If `sudo dmesg -T | grep -i xid` shows anything new: reboot** (`sudo systemctl reboot`).
   That's the only reliable recovery on these unlocked cards. The 100 W caps and SearXNG come
   back on their own.
4. Then use the triage table in §4 to work out which crash it was.

---

## 2. The leinasi issue: what it is

**Symptom:** the server runs fine, then dies mid-request. `dmesg` shows
`NVRM: Xid (...): 31 ... MMU Fault ... FAULT_INFO_TYPE_REGION_VIOLATION ACCESS_TYPE_VIRT_WRITE`
on the **last pipeline rank** (the card holding the DSpark drafter). The container log shows a
CUDA illegal memory access / `AcceleratorError` in the sparse-attention indexer. Repeated
Xid 31s escalate to **Xid 154** ("OS reboot required").

**When it happens:** the **first time a single request's context goes past roughly 174,000
tokens**, if `DSV4_LOGITS_ROW_CHUNK` is not set. It is **not** time-based. leinasi's production
box crashed "67 minutes" after a relaunch only because that's when live traffic first sent a
request that long. They reproduced it at ~174k in every one of 5+ controlled rounds, with
CUDA graphs on or off, batch 2048 or 4096, patched or unpatched image.

**Root cause (verified in our engine source,
`vllm/model_executor/layers/sparse_attn_indexer.py` ~line 650):** during prefill, the DSV4 sparse
indexer computes a float32 logits buffer of size **rows × context length**, where rows = the
prefill chunk (`--max-num-batched-tokens`, 2048 for us). With `DSV4_LOGITS_ROW_CHUNK` unset
(the code default is `0` = off), it allocates that whole buffer in one shot:

| context | buffer, unchunked (2048 rows) | buffer with ROW_CHUNK=64 |
|---|---|---|
| 100k | 0.76 GiB | 0.02 GiB |
| **~174k** | **~1.33 GiB** ← about all the slack the last rank has | 0.04 GiB |
| 691,968 | 5.3 GiB | 0.16 GiB |

The last PP rank has only ~1-2 GiB free after weights + KV cache (see `FILL-BASELINE.md`), so
somewhere around 174k the Triton kernel writes past its buffer, which is an MMU fault (Xid 31)
rather than a clean out-of-memory error. `DSV4_LOGITS_ROW_CHUNK=64` processes 64 rows at a time
and gives exactly the same results (each row's top-k only reads its own row), just with a tiny
buffer.

**How leinasi got it wrong at first** (worth knowing so we don't repeat it): they had just
changed `--max-num-batched-tokens` from 4096 to 2048, so they blamed batch size and wrote an
incident note saying "bat2048 causes Xid 31". The real cause was that their hand-written
relaunch script had **silently dropped** `DSV4_LOGITS_ROW_CHUNK=64`. Lesson: when a crash
follows a config change, diff the **full** container env (`docker inspect`), not just the
setting you meant to change.

**Same family, seen by us on 0731 (2026-08-15, `CARD-REGISTRY.md`):** a DSpark + 1M-context
config crashed at ~97k depth in `fp8_mqa_logits_triton` on PP2 with only ~1.4 GiB free. Same
kernel, same rank, same "transient buffer vs. thin last-rank margin" story.

---

## 3. Are we protected? (check any time)

```bash
docker inspect dsv4-vision-3card --format '{{range .Config.Env}}{{println .}}{{end}}' | grep ROW_CHUNK
# must print: DSV4_LOGITS_ROW_CHUNK=64
docker inspect dsv4-a100-3card   --format '{{range .Config.Env}}{{println .}}{{end}}' | grep ROW_CHUNK   # 0731, same
```
Both launchers hard-code it (`scripts/launch-vision-3card.sh`, and 0731's
`phase5-launch-dspark-production.sh`). **If you ever write a new launch script, copy the whole
env block. Never retype it from memory.** That's exactly how leinasi lost it.

**Tested today (2026-09-29, config A: 0.965 / 691,968 / seqs 8), all with ROW_CHUNK=64, all
past the ~174k trigger:**
- Phase 4 stress: one 629,705-token request with image traffic alongside. Passed.
- Soak: 6× fresh 250k-token prefills and 2× ~630k requests with concurrent image traffic.
  All passed, 0 Xid.
- Soak ran 38 min (20/20 loops) before we stopped it. It never reached leinasi's "67 minutes",
  but as explained above, that number was incidental. The trigger is context length, and we
  crossed it 9 times.

---

## 4. Which crash is it? Triage table

Everything below has actually happened on this hardware or on other people's CMP 170HX builds
of these models. Match your `TRIAGE.txt` against it.

| # | What you see | When | Cause | Fix / lever |
|---|---|---|---|---|
| 1 | **leinasi crash**: Xid 31 on the last rank (GPU 2), illegal access in `sparse_attn_indexer` / `fp8_mqa_logits_triton` | first request past ~174k tokens | `DSV4_LOGITS_ROW_CHUNK` missing | set it to 64 (check §3) |
| 2 | Same signature (PP2, indexer, illegal access / "unspecified launch failure") **with ROW_CHUNK=64 set** | deep prompt on a config filled right to the edge | last-rank margin too thin for other transient buffers | lower `--max-num-seqs` first (0731's proven fix: 8→2), then util (0.965→0.96, config B). **Partition changes don't help** (disproven 2026-08-15) |
| 3 | Crash at **boot**, in `load_dspark_model` → `prepare_moe_mxfp4_layer_for_marlin` / `marlin_permute_scales`, illegal access + Xid 31 | during model load, before healthy | too little VRAM on GPU 2 for the drafter's weight repack. Seen when `VLLM_MARLIN_FP8_DEQUANT_BF16=1` (+2.4 GiB/rank) | keep `MARLIN_DEQUANT` off (launcher default) |
| 4 | Clean `ValueError: No available memory for the cache blocks` or "X GiB KV cache is needed … estimated maximum model length is N" | boot, no Xid | config doesn't fit (util too low / context too high / wrong image) | expected and harmless. Check `IMG` is `…-p1` (the unpatched image can't fit 691,968), or use a smaller `--max-model-len` |
| 5 | `cuInit` segfault / "CUDA driver initialization failed" / "CUDA unknown error" on every card, `nvidia-smi` looks fine | after an Xid 31, especially after a module reload | driver poisoned (Xid 154) | reboot. Never rmmod/modprobe |
| 6 | Text answers contain content from **other people's requests** | graph-mode + multimodal traffic | CUDA-graph corruption bug reported by kaka86mm (fixed upstream PR #54566 `5ab628dd1`) | our gate G7 tests exactly this; passed every run so far |
| 7 | `RuntimeError: cancelled` in `shm_broadcast.acquire_read`, all in-flight requests HTTP 500, **no Xid** | ~4+ concurrent requests on PixelML's build | engine IPC bug in a different fork | not seen on ours (G9 c=4 and soak bursts of 6 passed); if seen, lower `--max-num-seqs` |
| 8 | Triton illegal access in `prepare_dflash_inputs` (DSpark speculator) on the last rank | ~131k prompt, PixelML's build | different fork's speculator | not seen on ours (passed 250k/630k); if seen, try no-spec: launcher arg 6 = `off` |
| 9 | Xid 31 **without** a vLLM error, during non-LLM GPU work | e.g. video gen (2026-08-18) | card-level MMU faults, see memory note `cmp170hx_xid_faults` | reboot; note which card serial |

---

## 5. How to test for it again (e.g. after any change to image, launcher, driver, or checkpoint)

Everything runs from `~/cmp170hx-deepseek-v4-flash/vision-exp-3card`. Start the thermal logger
first so temps are on record:
```bash
nohup setsid scripts/thermal-logger.sh > logs/thermal-logger.out 2>&1 < /dev/null &
```

**Step 1: basic health and env check** (1 min)
```bash
curl -s http://127.0.0.1:8099/health && echo UP
docker inspect dsv4-vision-3card --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "ROW_CHUNK|DEQUANT|PARTITION"
sudo dmesg | grep -ci xid          # note the number; it must not go up during testing
```

**Step 2: cross the trigger** (~3 min). A single request past 174k is the actual test for this bug:
```bash
python3 scripts/depth-probe.py --port 8099 --model dsv4v --tokens 250000 --out results/retest-250k.json
```
Pass = prints `"error": null, "health_after": "ok"` and the Xid count is unchanged.

**Step 3: full-depth stress with images** (~9 min), the same test config A passed on 2026-09-29:
```bash
python3 scripts/depth-probe.py --port 8099 --model dsv4v --tokens 630000 --out results/retest-630k.json &
sleep 60; python3 scripts/vision-gates.py --port 8099 --model dsv4v --only G1,G4,G5,G7,G9; wait
```

**Step 4: soak** (as long as you can spare; stops itself at the first failure):
```bash
nohup setsid scripts/soak.sh 9000 > logs/soak.out 2>&1 < /dev/null &
tail -f logs/soak.out
```

**Optional, only if you need proof the protection is what's saving you:** relaunch with
`DSV4_LOGITS_ROW_CHUNK` removed and run Step 2. By leinasi's evidence it **will** crash around
174k with Xid 31, and **you will need to reboot afterwards**. Don't do this casually, and never
on a box someone's using.

---

## 6. Numbers to compare against (known-good, 2026-09-29)

| | value |
|---|---|
| Config A | util 0.965, max-model-len 691,968, seqs 8, partition 15,15,13, batch 2048, DSpark n=5 |
| KV pool at boot | 797,299 tokens (every boot identical) |
| Idle VRAM GPU 0/1/2 | ~59,100 / 57,000 / 63,100 MiB |
| Peak VRAM under 630k stress | 60,722 / 58,410 / 64,504 MiB |
| 630k request wall time | 471-517 s |
| Temps under sustained load | core ≤ 67 °C, HBM ≤ 77 °C, fans 44-62% duty |
| Decode, short / 100k / 300k | ~86 / 78 / 68 tok/s |

If a future test lands far from these (e.g. a different KV pool size at boot, or a much higher
idle VRAM), something in the image/launcher/driver changed. Diff `docker inspect` against a known-
good run before chasing the model.

## 7. Where everything is

| what | where |
|---|---|
| This project's full history | `CAMPAIGN-LOG.md` (every boot, crash, and fix, in order) |
| 0731 comparison numbers | `FILL-BASELINE.md` |
| Launcher (known-good env block) | `scripts/launch-vision-3card.sh` |
| Evidence collector | `scripts/capture-crash.sh` |
| Test tools | `scripts/depth-probe.py`, `vision-gates.py`, `soak.sh`, `thermal-logger.sh`, `tool-tests.py` |
| leinasi's own write-ups | `~/CMP-170HX-PROJECT/vllm-dsv4/vllm-vision-leinasi/docs/cmp170hx/` |
| The row-chunk code | `.../vllm-vision-leinasi/vllm/model_executor/layers/sparse_attn_indexer.py` (search `DSV4_LOGITS_ROW_CHUNK`) |
| Our 0731 crash history | `../CARD-REGISTRY.md` (2026-08-15 entries) |
