# CMP 170HX card registry

Per-card identity and test results, so the four cards stay distinguishable after they are
installed, swapped, or sold on.

## How to tell the cards apart

**Only two fields are actually unique per card.** Everything else is identical across the whole
product line — the community cohort of 4 cards across 2 hosts all reported the same Board PN,
GPU PN and subsystem ID.

| Field | Unique? | Card #1 value |
|---|---|---|
| **Serial Number** | ✅ **unique** | `1322421002704` |
| **GPU UUID** | ✅ **unique** | `GPU-9629793f-1a75-b287-7619-2a9ba84efafc` |
| Board Part Number | ❌ shared | `900-11001-0108-000` |
| GPU Part Number | ❌ shared | `20C2-105-A1` |
| PCI device / subsystem | ❌ shared | `10de:20c2` / `10de:1585` |
| VBIOS | ⚠️ varies by batch | `92.00.6D.00.0A` (300 W OC) |
| Inforom image | ⚠️ varies by batch | `1001.0108.01.02` |

Read them with:

```bash
nvidia-smi -q | grep -E 'Serial Number|GPU UUID|Board Part|GPU Part|VBIOS'
```

**Physically label each card with the last 6 digits of its serial before installing it.** Once a
card is out of the machine there is no way to read the serial without powering it up again.

## Why per-card records matter here

Two things are genuinely per-die and must not be assumed to transfer between cards:

- **Clock/voltage offset headroom.** The wiki's sweep is explicit that the highest safe offset is a
  silicon property. Card #1 verified clean at +300 / 1400 MHz ceiling; that says nothing about #2–4.
- **Memory stability.** No ECC and no error telemetry, so a full-VRAM pattern sweep is the only
  evidence a card is sound. Must be repeated per card.

VBIOS also varies in the field — `92.00.67.00.01` (250 W) and `92.00.6D.00.0A` (300 W OC) are both
shipped on 8 GB cards. Unlock behaviour is identical either way, but the power ceiling and memory
clock differ.

---

# Card #1 — serial `1322421002704`

**Status: UNLOCKED, tested, in service.** Installed 2026-08-06 in `user-AM5D4ID2`, PCIe slot
`0000:01:00.0`.

## Identity

| | |
|---|---|
| Serial Number | **1322421002704** |
| GPU UUID | `GPU-9629793f-1a75-b287-7619-2a9ba84efafc` |
| Board Part Number | `900-11001-0108-000` |
| GPU Part Number | `20C2-105-A1` |
| PCI ID / subsystem | `10de:20c2` / `10de:1585` (8 GB SKU → 64 GB) |
| **VBIOS** | **`92.00.6D.00.0A`** — the **300 W OC** image (not the 250 W stock `92.00.67`) |
| Inforom image | `1001.0108.01.02`, OEM object 2.0 |
| Compute capability | 8.0 (GA100) |

## Unlock result

| | |
|---|---|
| Memory | 8192 → **65536 MiB** ✅ |
| Registers written | `SS0=0x88888888 SS1=0x00000008 CFG1=0x02779000 LMR=0x0000020b` — all match the authoritative table |
| PCIe | Gen1 → **Gen2 x4** ✅ (retrain succeeded at iteration 24) |
| ECC | N/A — fused off (expected on this SKU) |
| Driver used | nvidia-open **610.43.02**, cmpunlocker `360acd7` |

## Silicon characteristics

| | Card #1 | Note |
|---|---|---|
| Max memory clock | **1728 MHz** | higher than the ~1590 MHz quoted for stock 8 GB cards — the 300 W OC VBIOS raises it |
| Max SM clock | 1695 MHz | |
| Power limits | min **100 W** / default 250 W / **max 300 W** | 300 W ceiling comes from the OC VBIOS; stock cards cap at 250 W |

## Test results

**Memory integrity** — 56 GiB allocated in 14×4 GiB chunks, distinct per-chunk patterns, verified at
start/middle/end of every chunk: **0 errors, no fold, no aliasing, zero Xid.** Repeated clean during
the clock-offset test (3 further full-VRAM sweeps, patterns `0xA5A5A5A5`/`0x5A5A5A5A`/`0xFFFFFFFF`).

**Compute** (PyTorch GEMM, 100 W cap):

| | Measured | vs locked 0.395 TFLOPS |
|---|---|---|
| FP32 (TF32 off) | **7.42 TFLOPS** | **18.8×** |
| TF32 tensor | **30.47 TFLOPS** | tensor paths restored |
| FP16 tensor | **51.71 TFLOPS** | tensor paths restored |
| BF16 @ 1400 MHz ceiling | 58.13 TFLOPS / 98.8 W / 588 GFLOP/W | |

**Clock offset** — `+300` at a 1400 MHz ceiling: **memory clean, 3 sweeps, 0 errors.** Matches
upstream's validated maximum for that ceiling *on this die*. ⚠️ Throughput effect untested — the
100 W cap binds before 1400 MHz is reachable (card ran 636–699 MHz).

**Thermals** (2 fans, GPU-temp-driven BMC fan daemon):

| | Idle | Under sustained load |
|---|---|---|
| GPU core | 39–40 C | **47 C peak** (85 C spec) |
| HBM | 52–54 C | **60 C peak** (95 C spec) |
| Power draw | 38–41 W | ~69 W avg, 165.9 W spikes in prefill |

85-minute continuous agentic run: GPU core avg 41.2 C, max 47 C. Never thermally limited.

**LLM performance** (DeepSeek-V4-Flash-0731 UD-Q8_K_XL, ctx 65536, fp16 KV, hybrid CPU+GPU):

| | |
|---|---|
| Decode | **13.65 tok/s** (sd 0.03) |
| Decode + DSpark n_max=2 | **14.72 tok/s** (sd 1.11, +7.8%) |
| Prefill (long prompt) | 16.35 tok/s |

> 2026-08-14, both cards (`--n-cpu-moe 30`, unbalanced GPU0/GPU1 residency): only 7.33-8.02 tok/s
> — *slower* than this card's own single-card 13.65 tok/s baseline above, and DSpark measured as
> effectively inert (0.69% draft acceptance) due to a local SWA-disable patch on the drafter.
> Not a hardware regression — a placement/config issue. Full writeup, root cause, and re-tuning
> plan: `unsloth-DeepSeek-V4-Flash-0731/docs/BENCHMARK-RESULTS.md` ("2026-08-14 — first run on
> `user-ROMED8-2T`").
| VRAM used by `--fit` | 63,412–64,178 MiB of 65,536 |
| Model load | 77–90 s (`--no-mmap`) |

**Power behaviour** — draws ~78 W under LLM load regardless of whether the cap is 100 W or 125 W.
Never approaches the limit; the workload is DDR5-bound, not GPU-bound.

**LLM performance — vLLM + AWQ, single card** (cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit, MTP n=3,
262144 ctx, `--tensor-parallel-size 1`, 100 W, same CAD prompt/methodology as the dual-3090
151.80 tok/s benchmark in `~/Desktop/benchmark-of-3090-v620-b50-b580-mi60.md`):

| | |
|---|---|
| Decode (solo run, GPU0) | **122.70 tok/s** (8,952 tokens, 72.96 s, finish=stop) |
| Decode (independent-replica run, GPU0 vs. card #4 on GPU1 concurrently) | **127.44 tok/s** (9,808 tokens, 76.96 s, finish=stop) |
| Peak temps | 63–64 C core / 74 C HBM |
| vs. dual-3090 @195W (151.80 tok/s, 373.9 W) | 80.8–84.0% of the speed at ~27% of the power |

Marlin AWQ kernel confirmed working cleanly on this card's sm_80/GA100 silicon, no
compatibility issues. Consistently the slower of the two cards benchmarked under this vLLM
config — see card #4 below, which beat the dual-3090 baseline outright on the same test.
2026-08-13, `user-ROMED8-2T`, fan floor 55%.

**Currently running at:** `-pl 100`, clock offset 0, clocks unlocked.

## Incidents

- **2026-08-13, on `user-ROMED8-2T`:** `Xid 79 "GPU has fallen off the bus"` during an
  unrelated file transfer. Resolved by full power-off, reseating the EPS 8-pin connector,
  power back on -- no recurrence across a driver reinstall, unlock, and two further cold
  power cycles. See `card-records/card_1322421002704_20260806.md` for detail. Treat a
  repeat as a connector/seating check first on this specific card.

---

# Card #2 — serial `1322321009409`

**Status: UNLOCKED, baseline captured.** Installed 2026-08-06 in `user-AM5D4ID2`, slot
`0000:01:00.0`, **replacing card #1** (only one card in the machine at a time).

## 🔑 It unlocked with NO install.sh re-run

Card #2 came up at **65536 MiB / Gen2 x4 on its first boot**, with the same
`POST-WRITE SS0=0x88888888 SS1=0x00000008 CFG1=0x02779000 LMR=0x0000020b` line as card #1.

The patched `nvidia.ko` carries both geometries and selects one at GSP boot from
`pGpu->idInfo.PCIDeviceID`. **It is not bound to a particular card** — any `10de:20c2` that
appears gets unlocked automatically. **Cards #3 and #4 need no install step either: swap and
boot.** `install.sh` only needs re-running after a kernel upgrade (DKMS is removed).

## Identity

| | |
|---|---|
| Serial Number | **1322321009409** |
| GPU UUID | `GPU-57324c55-e2f9-1900-7300-a9eaa7034fd4` |
| Board / GPU Part Number | `900-11001-0108-000` / `20C2-105-A1` (shared, not unique) |
| PCI ID / subsystem | `10de:20c2` / `10de:1585` |
| **VBIOS** | **`92.00.6D.00.0A`** — 300 W OC, **same as card #1** |
| Inforom image | `1001.0108.01.02` |

## Unlock result

| | |
|---|---|
| Memory | 8192 → **65536 MiB** ✅ (automatic) |
| PCIe | **Gen2 x4** ✅ |
| Xid faults | 0 |
| Max memory clock | 1728 MHz (same as #1) |
| Power limits | min 100 / default 250 / **max 300 W** (same as #1) |

## Test results

**Memory integrity** — 60 GiB allocated, pattern `0xA5A5A5A5`: **0 errors, CLEAN.**
(Quick mode. Run `--full` for 3 sweeps before trusting it under load.)

**Compute** (`capture-card.sh`, 12 s windows, 100 W cap):

| | Card #2 | Card #1 (same script) | Δ |
|---|---|---|---|
| FP32 (TF32 off) | **7.83 TFLOPS** | 7.16 | +9.4% |
| TF32 tensor | **31.81 TFLOPS** | 28.95 | +9.9% |
| FP16 tensor | **53.33 TFLOPS** | 47.94 | +11.2% |

> ⚠️ **Do NOT read that as card #2 being faster silicon.** Card #2 started at **31 C** and card #1
> at **46 C** — a 15 C difference in starting temperature, on a 12-second benchmark, at a 100 W
> cap where the card is power-limited. A fair comparison needs both cards benchmarked from the
> same thermal state with the same window length. Card #1's own two runs of the *longer* 25 s
> benchmark gave 7.42 / 30.47 / 51.71, which brackets card #2's numbers.

**Thermals** — idle **31 C core / 44 C HBM**, 33.8 W. Peak 38 C during the compute test.

**Clock offset probe — `+300` at a 1400 MHz ceiling: MEMORY CLEAN, 4 sweeps, 0 errors.**
Patterns `0xA5A5A5A5` / `0x5A5A5A5A` / `0xFFFFFFFF` / `0x00000000` over 56 GiB. Matches upstream's
validated maximum for that ceiling and matches card #1. `nvmlDeviceSetGpcClkVfOffset` returned
rc=0, readback 300 MHz.

| | BF16 | Draw | SM clock | GFLOP/W |
|---|---|---|---|---|
| offset +0 | 65.33 TFLOPS | 100.3 W | 803 MHz | 651 |
| offset +300 | 65.31 TFLOPS | 100.3 W | 759 MHz | 651 |

**The offset changed throughput by 0.03% — i.e. nothing.** Same reason as card #1: at `-pl 100`
the **power cap binds long before the 1400 MHz ceiling** (card ran 759–803 MHz), so neither the
ceiling nor the VF offset is the active constraint. Confirming the offset is *safe* on this die is
the useful result; measuring its *benefit* still requires `-pl` ≥150 W.

Note card #2 reached **803 MHz vs card #1's 699 MHz** at the same 100 W cap and same ceiling,
giving 65.33 vs 58.13 TFLOPS (+12.4%). Card #2 was ~15 C cooler throughout, which at a
power-limited operating point directly buys clock — so this is still not established as silicon
quality. A matched-temperature rerun would settle it.

**Not yet run on this card:** LLM throughput benchmark, sustained thermal soak.

---

# Card #3 — serial `1322621127916`

**Status: UNLOCKED, baseline captured.** Installed 2026-08-07 in `user-AM5D4ID2`, slot
`0000:01:00.0`, **replacing card #2** (only one card in the machine at a time).

## Unlocked with NO install.sh re-run (3rd card in a row)

Came up at **65536 MiB / Gen2 x4** on first boot, same as card #2 — further confirms the
patched `nvidia.ko` unlocks any `10de:20c2` card automatically at GSP boot, not just the one
present during `install.sh`.

## Identity

| | |
|---|---|
| Serial Number | **1322621127916** |
| GPU UUID | `GPU-45d6cec7-e12e-82bd-2526-6959125c58be` |
| Board / GPU Part Number | `900-11001-0108-000` / `20C2-105-A1` (shared, not unique) |
| PCI ID / subsystem | `10de:20c2` / `10de:1585` |
| **VBIOS** | **`92.00.6D.00.0A`** — 300 W OC, **same as cards #1 and #2** |
| Inforom image | `1001.0108.01.02` |

## Unlock result

| | |
|---|---|
| Memory | 8192 → **65536 MiB** ✅ (automatic) |
| PCIe | **Gen2 x4** ✅ |
| Max memory clock | 1728 MHz (same as #1/#2) |
| Max SM clock | 1695 MHz |
| Power limits | min 100 / default 250 / **max 300 W** (same as #1/#2) |

## Test results

**Memory integrity** — two independent sweeps, both clean: `capture-card.sh --full` (60 GiB,
3 patterns, 0 errors) and `clock-offset-test.sh`'s own pre/post sweep (56 GiB, 3 patterns,
0 errors at +300 offset). **0 errors total across both.**

**Compute** (`capture-card.sh`, 12 s windows, 100 W cap, idle start 44 C):

| | Card #3 | Card #1 | Card #2 |
|---|---|---|---|
| FP32 (TF32 off) | **7.80 TFLOPS** | 7.16 | 7.83 |
| TF32 tensor | **31.93 TFLOPS** | 28.95 | 31.81 |
| FP16 tensor | **53.54 TFLOPS** | 47.94 | 53.33 |

Near-identical to card #2 (within ~1%) — plausible since both started from a similar idle
temp (44 C vs card #2's 31 C, card #1's 46 C). Per the standing caveat below, don't read small
deltas here as silicon quality without matched start temps.

**Thermals** — idle **44 C core / 55 C HBM**, 34.5 W. Peak 49 C during the compute test.

**Clock offset — `+300` at a 1400 MHz ceiling: MEMORY CLEAN under load, 3 sweeps, 0 errors**
(via `clock-offset-test.sh`, the dedicated pre/post-sweep script — not just an NVML
set/readback). Matches upstream's validated maximum and matches cards #1 and #2.

| | BF16 | Draw | SM clock | GFLOP/W |
|---|---|---|---|---|
| offset +0 | 65.11 TFLOPS | 100.6 W | 794 MHz | 647 |
| offset +300 | 65.66 TFLOPS | 101.9 W | 782 MHz | 644 |

Offset changed throughput by +0.8% — within noise, same power-cap-binds-first story as #1/#2
(card ran 782–794 MHz vs the 1400 MHz ceiling).

**Not yet run on this card:** sustained thermal soak, cowboy-clue agentic build.

## LLM throughput + power sweep (2026-08-07)

**Qwen3.6-35B-A3B UD-Q4_K_XL + MTP (draft=2), full native 262144 ctx, single-card full GPU offload:**
repeated-samples power sweep (5 rounds, alternating direction, 250-token fixed-length essay
prompt to control for variable reasoning-length confounds — see
`bench-results/power-sweep-qwen35b-repeated_*.csv`):

| power cap | mean decode tok/s | mean tok/W |
|---|---|---|
| 100 W | 121.95 (sd 6.38) | 1.261 |
| 120 W | 138.52 (sd 8.32) | 1.220 |
| 140 W | 141.54 (sd 6.08) | 1.109 |
| 160 W | 148.86 (sd 9.31) | 1.040 |
| 180 W | 149.77 (sd 7.20) | 0.988 |
| 200 W | 146.58 (sd 5.61) | 0.988 |

No hard efficiency cliff found (an earlier fine 5W-step sweep at 100-175W was noise-dominated,
spread larger than the point-to-point deltas — repeated sampling was needed to see a trend at
all). Decode keeps climbing to ~180W then flattens; tok/W falls off steadily from 100W. **Card
left at 100W as the standing config** (best tok/W, thermal headroom, diminishing decode gains
above it) — not a "found the knee" result, an efficiency-over-peak-speed choice. Temps stayed
≤55C across the whole sweep (100-200W).

**DeepSeek-V4-Flash-0731 UD-Q8_K_XL + DSpark** (`--spec-type draft-dspark --spec-draft-n-max 3`,
Unsloth's documented default, hybrid `--fit on --fit-target 12288`): single-sample decode
**18.29 tok/s** (draft_n=25, draft_n_accepted=20, 80% acceptance) vs **13.65 tok/s** no-spec
baseline on this same card — **+34%**. Not a controlled multi-run sweep (see
[[project_deepseek_v4_flash_dspark]] for the rigorous 10-run n_max=2 result on this card, 14.72
tok/s — this single n_max=3 sample beat it, but hasn't been validated the same way; don't treat
18.29 as settled over the sweep-backed 14.72 without re-testing n_max=3 properly).

⚠️ **DeepSeek-V4-Flash-0731 (with or without DSpark) is NOT viable as an opencode-driving model
on this card** — multi-turn tool-calling conversations stall for 8+ minutes on turn 2+ regardless
of speculative decoding or a `--reasoning-budget` cap, because DeepSeek's own documented
`drop_thinking` spec keeps every prior turn's full reasoning trace in every subsequent prompt
once tools are present (opencode always sends tools). This is a model/spec-level limitation, not
fixable by tuning this card's config. opencode's default model was switched to
Qwen3.6-35B-A3B-MTP as a result. Full writeup: [[project_deepseek_v4_flash_dspark]] and
`unsloth-DeepSeek-V4-Flash-0731/docs/OFFICIAL-VS-ACTUAL-DIVERGENCE.md` in the model folder.

---

# Card #4 — serial `1322421041391`

**Status: UNLOCKED, baseline captured.** Installed 2026-08-07 in `user-AM5D4ID2`, slot
`0000:01:00.0`, **replacing card #3** (only one card in the machine at a time).

## Unlocked with NO install.sh re-run (4th card in a row)

Came up at **65536 MiB / Gen2 x4** on first boot, same as cards #2 and #3 — further confirms the
patched `nvidia.ko` unlocks any `10de:20c2` card automatically at GSP boot, not just the one
present during `install.sh`.

## Identity

| | |
|---|---|
| Serial Number | **1322421041391** |
| GPU UUID | `GPU-2775e649-c961-519f-2160-53fce9dbf5e8` |
| Board / GPU Part Number | `900-11001-0108-000` / `20C2-105-A1` (shared, not unique) |
| PCI ID / subsystem | `10de:20c2` / `10de:1585` |
| **VBIOS** | **`92.00.6D.00.0A`** — 300 W OC, **same as cards #1–#3** |
| Inforom image | `1001.0108.01.02` |

## Unlock result

| | |
|---|---|
| Memory | 8192 → **65536 MiB** ✅ (automatic) |
| PCIe | **Gen2 x4** ✅ |
| Max memory clock | 1728 MHz (same as #1–#3) |
| Max SM clock | 1695 MHz |
| Power limits | min 100 / default 250 / **max 300 W** (same as #1–#3) |

## Test results

**Memory integrity** — two independent sweeps, both clean: `capture-card.sh --full` (60 GiB,
3 patterns, 0 errors) and `clock-offset-test.sh`'s own pre/post sweep (56 GiB, 3 patterns,
0 errors at +300 offset). **0 errors total across both (6 sweeps, 2 scripts).**

**Compute** (`capture-card.sh`, 12 s windows, 100 W cap, idle start 41 C):

| | Card #4 | Card #1 | Card #2 | Card #3 |
|---|---|---|---|---|
| FP32 (TF32 off) | **7.91 TFLOPS** | 7.16 | 7.83 | 7.80 |
| TF32 tensor | **32.34 TFLOPS** | 28.95 | 31.81 | 31.93 |
| FP16 tensor | **54.56 TFLOPS** | 47.94 | 53.33 | 53.54 |

Slightly ahead of #2/#3 (within ~2%) — plausible given a similar idle start temp (41 C vs #3's
44 C, #2's 31 C, #1's 46 C). Per the standing caveat below, don't read small deltas here as
silicon quality without matched start temps.

**Thermals** — idle **41 C core / 58 C HBM**, 33.59 W. Peak 46 C during the compute test.

**Clock offset — `+300` at a 1400 MHz ceiling: MEMORY CLEAN under load, 3 sweeps, 0 errors**
(via `clock-offset-test.sh`). Matches upstream's validated maximum and matches cards #1–#3.

| | BF16 | Draw | SM clock | GFLOP/W |
|---|---|---|---|---|
| offset +0 | 65.31 TFLOPS | 98.5 W | 798 MHz | 663 |
| offset +300 | 66.15 TFLOPS | 104.3 W | 779 MHz | 634 |

Offset changed throughput by +1.3% — within noise, same power-cap-binds-first story as #1–#3
(card ran 779–798 MHz vs the 1400 MHz ceiling).

## LLM throughput at full native context (2026-08-07)

**DeepSeek-V4-Flash-0731 UD-Q8_K_XL + DSpark (n_max=3), `--ctx-size 1048576` (full native max,
up from the prior 720896 fp16-KV cap)** — required rebuilding `build-cuda-sm80/llama-server`
with `-DGGML_SCHED_MAX_SPLIT_INPUTS=48` (previously only the sm120 `build-cuda/` tree had this;
never carried into the sm80 tree this card actually runs, until now). Booted clean: `n_ctx_slot
= 1048576`, load time 80s, VRAM flat at **63,426 MiB** (matches the flat-regardless-of-context
behavior already established for this model's MLA cache — see
[[project_deepseek_v4_flash_kv_precision]]), host RAM **116 GiB of 249 GiB**.

Same day, host RAM changed **2x64GB DDR5-5600 (128GB) -> 4x64GB DDR5-3600 (256GB)** — 2
DIMMs/channel derates Configured Memory Speed on this AM5 board. This workload is
DDR5-bandwidth-bound (established in `bench-deepseek-v4-flash-cmp170hx.sh`'s own commentary), so
the derating shows up directly in decode speed:

| | decode tok/s |
|---|---|
| run 1 | 12.56 |
| run 2 | 12.80 |
| run 3 | 9.46 |
| run 4 | 11.39 |
| run 5 | 11.19 |
| **mean** | **11.48** |

**vs. the same DSpark n_max=3 config at 720896 ctx on the old 2x64GB@5600 RAM: 18.29 tok/s.**
**-37.2%.** Also below the no-spec 13.65 tok/s baseline measured on that same old RAM config.
⚠️ **Net effect of this RAM change, for this specific bandwidth-bound model: capacity roughly
doubled, decode throughput dropped ~37%.** Not a card #4 silicon issue -- same regression this
project's own bench script predicted in reverse when RAM went 4x64GB@3600 -> 2x64GB@5600 the
first time (9.4 -> 12-14 tok/s). Full context now boots and works; whether the capacity is worth
the bandwidth cost is a standing open question, not resolved here.

## LLM performance — vLLM + AWQ (2026-08-13, `user-ROMED8-2T`)

**cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit, MTP n=3, 262144 ctx, `--tensor-parallel-size 1`, 100 W,**
independent-replica run (concurrent with card #1 on GPU0, same config, see card #1's entry
above), same CAD prompt/methodology as the dual-3090 151.80 tok/s benchmark:

| | |
|---|---|
| Decode | **154.02 tok/s** (10,833 tokens, 70.33 s, finish=stop) |
| Peak temps | 59 C core / 71 C HBM |
| vs. dual-3090 @195W (151.80 tok/s, 373.9 W) | **101.5% — beats it**, at ~27% of the power (~3.7x tok/s per watt) |

**This is the best single-stream LLM result recorded anywhere in this project.** Beats card #1
under the identical concurrent test (127.44 tok/s) by +20.8% and beats card #1's solo run
(122.70 tok/s) by +25.6% — wider than the 10.5% FP32 TFLOPS gap the original 2026-08-06/07
compute benchmarks showed (7.16 vs 7.91 TFLOPS, different host, sequential single-card
testing). Root cause isolated 2026-08-13 — see below.

**Not yet run on this card:** sustained thermal soak, cowboy-clue agentic build.

### 2026-08-13 — root-caused: card #1 vs card #4 clock gap is a silicon binning difference

Isolated test, this host (`user-ROMED8-2T`), both cards concurrently: 30 s of sustained bf16
`8192×8192` matmuls per card (`bench-results/stress_gemm.py`), both at the standing 100 W cap,
matched starting temps (45 C / 41 C core), 1 s `nvidia-smi` clock/power/temp sampling
(`bench-results/clockcmp.csv`). No vLLM, no decode, no PCIe traffic between cards — isolates
pure per-die compute behavior from everything else in the stack.

| | Card #1 (GPU0) | Card #4 (GPU1) |
|---|---|---|
| Matmuls in 30 s | 1,700 | **1,950 (+14.7%)** |
| Steady-state SM clock | ~450–465 MHz | **~510–540 MHz** |
| Steady-state power draw | ~99–101 W | ~99–101 W (essentially identical) |
| Steady-state temp | 50–53 C core | 48–50 C core (cooler, not hotter) |
| Memory clock | 1728 MHz | 1728 MHz (unchanged, both) |

**At identical power draw and similar-to-cooler temps, card #4 sustains ~14% higher SM clock
than card #1.** PCIe link (both Gen2 x4, confirmed via `lspci -vv`), memory clock, and thermal
throttling are all ruled out — `nvidia-smi -q -d PERFORMANCE` showed no active throttle reason
on either card, and card #1 ran *cooler* than card #4 during the test, not hotter. This is a
silicon voltage/frequency-curve difference: card #4's die needs less voltage to hit a given
clock, so at a fixed power budget it clocks higher. Matches the original 2026-08-07 finding
that card #1 ran 699 MHz vs card #2's 803 MHz at the same 100 W cap (+14.9%, a different card
pairing, same magnitude) — **card #1 is consistently the weakest-binned die of the four**,
confirmed independently on two different hosts and now two different card pairings.

The 14.7% GEMM-throughput gap is smaller than the 20.8–25.6% gap seen in the real LLM decode
benchmark above. Expected: single-token autoregressive decode (batch=1, latency-bound) is more
clock-sensitive than a throughput-bound batched GEMM stress test, so it amplifies the same
underlying clock disadvantage more than raw matmul throughput does. The clock gap is the root
cause; decode just feels it more acutely.

### 2026-08-13 — follow-up: raising the power cap to 150 W nearly closes the gap

Same-day follow-up test: reran both the isolated GEMM stress and the vLLM AWQ+MTP decode
benchmark (same cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit / MTP n=3 / 262144 ctx config as the 100 W
runs above) at `-pl 150` on both cards, to test whether the card #1/#4 clock gap is purely a
function of the power cap binding harder on the weaker die.

| | Card #1 (GPU0) | Card #4 (GPU1) | Gap |
|---|---|---|---|
| LLM decode @ 100 W | 122.70–127.44 tok/s | 154.02 tok/s | 20.8–25.6% |
| **LLM decode @ 150 W** | **177.43 tok/s** (8,793 tokens, 49.56 s) | **183.19 tok/s** (10,078 tokens, 55.01 s) | **3.25%** |
| GEMM (isolated) @ 100 W | 1,700 matmuls/30s | 1,950 matmuls/30s | 14.7% |
| GEMM (isolated) @ 150 W | 3,050 matmuls/30s | 3,300 matmuls/30s | 8.2% |

Peak temps at 150 W: **70 C / 78 C (GPU0), 67 C / 76 C (GPU1)** — still 10 C+ of margin under
the 80 C/88 C safety thresholds with the 55% fan floor; no thermal concern raising the cap this
far. Neither card actually reached the 150 W ceiling under the GEMM stress (~131 W plateau on
both) — something other than the power cap itself becomes the limit above ~130 W for this
workload.

**Confirms the hypothesis: the gap is power-cap-driven, not fixed.** At 100 W both dies are
hard against the cap, so card #1's weaker voltage/frequency curve costs it real clock. At
150 W there's enough headroom that card #1 nearly catches up. One open question: the LLM
decode gap closed *further* than the raw GEMM gap (25.6%→3.25% vs. 14.7%→8.2%) — the opposite
of what "decode is more clock-sensitive" alone would predict. Possibly MTP draft-acceptance
run-to-run variance (different generated content → different acceptance rate → tok/s noise
independent of silicon); worth a repeat run or two at 150 W before treating 177 vs. 183 as a
stable number rather than partly noise.

**Power reverted to the standing 100 W after this test** (`sudo nvidia-smi -pl 100`, matches
`cmp-powerlimit.service`'s default) — this was an experiment, not a config change. If 150 W
becomes the new standing power limit, `set-power-limit.sh` / `cmp-powerlimit.service` need
updating (currently hardcoded to 100 W default) and the fan floor needs re-validating over a
longer sustained run than this ~50 s benchmark.

---

## Comparison table (fill in as cards are tested)

| | #1 `…002704` | #2 `…009409` | #3 `…127916` | #4 `…041391` |
|---|---|---|---|---|
| Serial | 1322421002704 | 1322321009409 | 1322621127916 | 1322421041391 |
| VBIOS | `92.00.6D.00.0A` (300 W) | `92.00.6D.00.0A` (300 W) | `92.00.6D.00.0A` (300 W) | `92.00.6D.00.0A` (300 W) |
| Unlocked MiB | 65536 | 65536 | 65536 | 65536 |
| Needed install.sh? | yes (first card) | **no — automatic** | **no — automatic** | **no — automatic** |
| PCIe after unlock | Gen2 x4 | Gen2 x4 | Gen2 x4 | Gen2 x4 |
| Max mem clock | 1728 MHz | 1728 MHz | 1728 MHz | 1728 MHz |
| Max power limit | 300 W | 300 W | 300 W | 300 W |
| FP32 TFLOPS (12s @100 W) | 7.16 | 7.83 | 7.80 | 7.91 |
| TF32 TFLOPS (12s @100 W) | 28.95 | 31.81 | 31.93 | 32.34 |
| FP16 TFLOPS (12s @100 W) | 47.94 | 53.33 | 53.54 | 54.56 |
| ⚠️ start temp for that run | 46 C | 31 C | 44 C | 41 C |
| Memory sweep | clean (4 sweeps) | clean (4 sweeps) | clean (6 sweeps, 2 scripts) | **clean (6 sweeps, 2 scripts)** |
| Max safe offset @1400 | +300 (clean) | +300 (clean, 4 sweeps) | +300 (clean, 3 sweeps) | +300 (clean, 3 sweeps) |
| Idle temp | 39-40 C | 31 C | 44 C | 41 C |
| Decode tok/s (V4-Flash Q8) | 13.65 | not tested | 13.65 (no-spec) / 18.29 (+DSpark n_max=3) | 11.48 (DSpark n_max=3, full 1M ctx, RAM now 3600MT/s) |
| Qwen3.6-35B-A3B MTP decode @100W (ik-llama, GGUF UD-Q4_K_XL) | not tested | not tested | 121.95 tok/s (sd 6.38, 5-run mean) | not tested |
| Qwen3.6-35B-A3B-AWQ MTP decode @100W (vLLM, AWQ-4bit) | 122.70 solo / 127.44 concurrent | not tested | not tested | **154.02** (best result in project) |

⚠️ **Compute numbers above are NOT comparable between cards** unless the start temperature and
benchmark window match — see card #2's note. Treat differences under ~10% as measurement noise.
⚠️ **The two Qwen3.6-35B-A3B MTP rows are not comparable to each other** — different engine
(ik-llama/GGUF vs. vLLM/AWQ), different quant maker, different card (#3 vs. #1/#4). Both are
correct for their own config; see [[2026-08-13 dual-card vLLM+AWQ benchmark]] below for the
apples-to-apples comparison against the dual-3090 baseline.

---

## 2026-08-13 dual-card vLLM+AWQ benchmark (both cards vs. dual-3090 baseline)

Reproduced the dual-3090 151.80 tok/s benchmark (cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit, vLLM,
MTP n=3, 262144 ctx, same CAD prompt/methodology) on this system's two CMP-170HX cards, at
100 W each, fan floor 55%. Full detail in card #1's and card #4's sections above; this section
covers the cross-card and tensor-parallel results only.

| Mode | tok/s | Power | Peak temps | Notes |
|---|---|---|---|---|
| Single card, GPU0 (card #1) | 122.70 | 100 W | 63 C / 74 C | `--tensor-parallel-size 1` |
| Independent replicas, GPU0 + GPU1 concurrently | 127.44 + **154.02** = 281.46 combined | 200 W | 64 C / 74 C (GPU0), 59 C / 71 C (GPU1) | Two separate vLLM servers, no cross-GPU traffic |
| **Tensor-parallel, both cards (`--tensor-parallel-size 2`)** | **146.29** | ~200 W | 65 C / 74 C | 24,000 completion tokens (hit max_tokens cap, finish=length — longer/more demanding run than the other rows) |

**TP=2 works — settles the open question in `P2P-ENABLEMENT-PROJECT-PLAN.md` Phase 2 step 3.**
No NCCL hang. `vLLM`'s own `shm_broadcast.py:705` warning ("No available shared memory
broadcast block found in 60 seconds") fired mid-startup while the two workers' independent
`torch.compile` passes were still out of sync with each other — a red herring, not the hang
the P2P plan's one existing community report described. The run completed cleanly end to end,
with MTP speculative decoding also working correctly under TP (81–100% draft acceptance,
in line with the single-card numbers).

**Verdict for this workload:** TP=2 (146.29 tok/s) is not worth it over independent replicas
(281.46 tok/s combined, or 154.02 tok/s for the single best stream) — matches the standing
`-sm layer`/PP-over-TP conclusion from the ik-llama testing, now confirmed for vLLM too. Keep
TP=2 as a known-working fallback for the rare case a single request needs both cards' VRAM,
not as a throughput strategy.

**Efficiency:** card #4 solo (154.02 tok/s / 100 W = 1.54 tok/s/W) is ~3.7x more efficient than
the dual-3090 baseline (151.80 / 373.9 W = 0.406 tok/s/W) for this exact model/quant/config.

## 2026-08-15 — vLLM + pipeline-parallel=3, DeepSeek-V4-Flash-0731 native FP8, 3x CMP-170HX

Engine: patched haosdent/vllm@dsv4-flash-a100 (base f8ea5bb, 6 of 7 repo patches applied --
0001 skipped, confirmed obsolete via git history AND the repo's own script comments
"precautionary, the report was retracted"), Docker image dsv4-a100:devel, CUDA 13.0.2 container
(isolated, no host CUDA change). Model: native MXFP4+FP8 checkpoint, 155.4 GiB, downloaded
directly from deepseek-ai/DeepSeek-V4-Flash-0731 (NOT the unsloth GGUF quant used by the
llama.cpp track).

**Phase 3 (no DSpark), ctx=8192, --gpu-memory-utilization 0.93, VLLM_PP_LAYER_PARTITION=15,15,13:**
- Decode: **55.8 tok/s** avg (55.48/56.05/55.98, 300-token completions, stable)
- Prefill: 24-token prompt, negligible in these runs -- not yet isolated separately
- VRAM: ~59.6/58.6/53.1 GiB used of 65.5 GiB per card (tight, as predicted)
- 2.3x the best llama.cpp result on this box (23.8 tok/s, --fit auto-placement, ctx 8192, no
  CPU offload -- see step2-fit-placement-3card.sh)

**Gotchas hit getting here (both required fixes, not just flag tuning):**
1. `--gpu-memory-utilization 0.85` (the repo's 4-card default) OOM'd on "No available memory
   for the cache blocks" -- 3-card splits weights into ~52GB/card vs 4-card's ~39GB/card, much
   less headroom in the same 64GB cards. Fixed: 0.93.
2. Real crash, not a memory issue: Triton kernel `_sparse_attn_prefill_blocked_kernel`
   requested 204,800 B shared memory against A100's 166,912 B hardware limit during warmup
   (`triton.runtime.errors.OutOfResources`). The code's own default (BLOCK_M=8) should avoid
   this per its own docstring, but our model's head config tripped it anyway. Fixed:
   `VLLM_SPARSE_DENSE_QUERY_BLOCK=0` (disables the blocked/tiled prefill kernel, falls back to
   the safe per-query path -- costs some prefill speed, not yet quantified; worth revisiting).

Next: Phase 4 (DSpark, `num_speculative_tokens=5`) -- unverified at 3 cards, repo's 98 tok/s
figure is 4-card only.

## 2026-08-15 — vLLM + pipeline-parallel=3 + DSpark, DeepSeek-V4-Flash-0731 native FP8, 3x CMP-170HX

**First-ever measurement of DSpark+PP at 3 cards** -- the repo's 98 tok/s figure is 4-card only,
this config was unverified going in. `--gpu-memory-utilization 0.96` (bumped again from Phase
3's 0.93 -- DSpark's extra drafter weights ate the remaining KV headroom: needed 1.2 GiB, only
0.69 GiB was free at 0.93).

- Decode: **80.95 tok/s avg** (4 runs: 79.89 / 99.93 / 52.59 / 91.38, 300-token completions) --
  **+45% over the Phase 3 no-spec baseline (55.8 tok/s)**. High run-to-run variance, tracks
  directly with draft acceptance rate swings (see below), not noise.
- Peak single run (99.93 tok/s) essentially **matches the repo's flagship 4-card number (98.1
  tok/s) on one fewer card.**
- SpecDecoding metrics from the server log across 3 sampled windows: mean acceptance length
  2.00-2.73, avg draft acceptance rate 20.0%-34.5%, per-position acceptance decaying with
  position within the 5-token block (e.g. 0.645/0.403/0.266/0.200/0.148) -- same shape the repo
  documented, confirms this is genuine DSpark behavior, not a fluke.
- VRAM: 57.7/56.6/62.7 GiB of 65.5 GiB per card -- tighter than Phase 3, as expected.

Both phases combined: going from llama.cpp's best (23.8 tok/s, no-spec, `--fit` auto-placement)
to vLLM+PP+DSpark on the same 3 cards is a **3.4x** improvement at peak, ~2.4x on average.

Next: Phase 5, ramp `--max-model-len` toward the full 1M (8k tested so far), watch the already-
tight per-card VRAM budget.

## 2026-08-15 — Phase 5: context ramp toward full 1M, no-spec, PP=3, DeepSeek-V4-Flash-0731 native

Launched directly at `--max-model-len 1048576` (full native max) rather than creeping up --
worked on the first try. **VRAM at full 1M max-model-len config is essentially identical to
the 8k config** (59.9/58.7/53.3 GiB vs 59.6/58.6/53.1 GiB) -- confirms the same "KV cache is
nearly free" property already documented for this architecture on the llama.cpp track, now
verified on vLLM/native FP8 too.

Real-text depth test (source files from this vllm checkout, not repeated boilerplate), using
vLLM's automatic prefix caching to isolate decode-at-depth cheaply:

| Depth | Prefill | Prefill rate | Decode (300 tok) | Decode rate |
|---|---|---|---|---|
| ~0 | -- | -- | -- | ~50.2 tok/s avg (3 runs) |
| 96,667 | 44.53s | **2,170 tok/s** | 6.61s | ~45.4 tok/s |
| 338,195 | 189.44s | **1,785 tok/s** | 7.55s | ~39.7 tok/s |

Decode degrades only ~21% from near-zero to 338k depth (vs. llama.cpp track's own finding of
flat-ish decode with depth for this arch -- consistent). Prefill decays gradually with depth,
same shape as the repo's own documented curve, but at roughly **30-40x the prefill speed**
llama.cpp achieved on this box (best prior: ~60 tok/s). VRAM headroom at 338k depth: 60.8/59.7/
54.0 of 65.5 GiB per card -- still 5-11 GiB free, comfortable margin remaining.

**Not yet tested:** actual generation at the full 1,048,576 depth (prefill alone would take
~10 min at the observed decay rate) and DSpark combined with large max-model-len (DSpark's
extra VRAM tax was already tight at ctx=8192 -- 0.96 utilization needed -- unclear if it fits
at all with max-model-len set to 1M, independent of how much context is actually used).
Trend so far strongly suggests the full 1M is usable at good speed given flat VRAM and gradual
(not cliff) decode/prefill decay, but this is inference from the trend, not a direct measurement.

## 2026-08-15 — DSpark + full 1M max-model-len: loads, but crashes on real-depth prefill

`--gpu-memory-utilization 0.98` was needed just to get DSpark to load with `--max-model-len
1048576` (0.96 wasn't enough: needed 2.65 GiB KV cache, only 1.43 GiB available). At 0.98,
GPU2 (the PP2 rank -- 13 layers + lm_head + DSpark drafter) loaded with only **~1.4 GiB free
VRAM** of 65.5 GiB.

**Near-zero depth: works well.** 4 runs, 88.64/86.30/99.32/88.60 tok/s, avg **90.7 tok/s** --
actually more consistent than the ctx=8192 DSpark result (which had a low outlier at 52.59).

**Real depth (~97k tokens): CUDA illegal memory access, worker dies, whole engine crashes.**
Root cause traced to `fp8_mqa_logits_triton` (the sparse attention indexer kernel) on
Worker_PP2 specifically -- the rank with almost no VRAM margin. Most likely explanation: the
indexer's transient logits buffer for a 2048-token prefill chunk at real depth needs more
scratch space than the ~1.4 GiB sliver left on that rank, and a Triton kernel writing past an
under-provisioned workspace buffer manifests as an illegal-access crash rather than a clean
OOM. Container exited cleanly afterward (VRAM fully released, no GPU wedge).

**Conclusion: DSpark + max-model-len=1M is not safely usable on 3 cards at these settings.**
Loads and serves short requests fine, but a real long-context prefill will crash it. This is
distinct from (but rhymes with) the source repo's own documented finding of an illegal-memory-
access crash in accumulating chats past 718-733k on 4 cards -- ours happens far earlier (~97k),
almost certainly because our 3-card VRAM margin is far tighter to begin with, not the same root
cause.

**Not yet tried:** reducing `--max-num-batched-tokens` (2048) or `--max-num-seqs` (8) to free
more of PP2's margin -- both are worth trying before writing this off entirely, but at real
cost in prefill/concurrency throughput. For now: no-spec is the only combination verified safe
at real long-context depth (confirmed clean to 338k -- see the Phase 5 entry above); DSpark
stays a short/interactive-context-only feature on this hardware, matching the same conclusion
the llama.cpp track independently reached for the same reason (indexer cost scaling with depth
eating the speculative-decoding gain, though the failure mode there was a slowdown, not a crash).

## 2026-08-15 (post-reboot) — DSpark + full 1M max-model-len FIXED via lower batch/seq limits

Tried the untried mitigation from the entry above. `phase4b-launch-dspark-lowbatch.sh` --
identical to phase4 (0.98 util, `VLLM_PP_LAYER_PARTITION=15,15,13`, all the same env vars)
except `--max-num-batched-tokens 2048→512` and `--max-num-seqs 8→2`.

**Result: no crash.** Same GPU2/PP2 margin as before (~1.1-1.4 GiB free, still razor-thin) but
survived real-depth prefill cleanly at both test depths:
- ~116.6k tokens (`prompt_tokens=116604`): survived, 3x in a row (1 cold + 2 prefix-cache-hit
  repeats, confirming the KV cache itself stayed intact/uncorrupted across requests, not just
  "didn't crash once").
- ~365.9k tokens (`prompt_tokens=365896`): survived, container stayed healthy afterward,
  decode speed back to 131 tok/s on a fresh short request immediately after.

Root cause confirmed: the crash was VRAM-scratch-space pressure in the sparse attention
indexer's per-chunk transient buffer, exactly as diagnosed yesterday -- shrinking the
prefill chunk size (`max-num-batched-tokens`) and concurrent-sequence ceiling
(`max-num-seqs`) was sufficient to fit the indexer's transient buffer in PP2's ~1.2 GiB
sliver. Cost: not yet quantified against phase4's throughput at ctx=8192 (short-context
DSpark decode was still ~131 tok/s post-deep-prefill here, comparable to the earlier 80.95
tok/s avg / 99.93 peak at maxlen=8192, batch=2048 -- but that comparison is at *shallow*
depth on both, not an apples-to-apples throughput-vs-batch-size tradeoff at depth. Worth a
follow-up: sweep `max-num-batched-tokens` between 512 and 2048 to find the largest value that
still survives real depth, since a bigger prefill chunk should mean faster prefill.

**Conclusion revised: DSpark + max-model-len=1M IS viable on 3 cards**, at the cost of
`--max-num-batched-tokens 512 --max-num-seqs 2` instead of the repo's defaults. Script:
`~/CMP-170HX-PROJECT/vllm-dsv4/phase4b-launch-dspark-lowbatch.sh [maxlen] [max_batched_tokens]
[max_num_seqs]`.

## 2026-08-15 (later) — isolated the actual lever: max-num-seqs, not max-num-batched-tokens

Swept batch size and seq count independently to find the real boundary (rather than assuming
both knobs mattered equally):
- `batch=1024, seqs=2`: survived 100k and 366k depth.
- `batch=2048 (repo default), seqs=2`: survived 100k **and** 366k depth. So batch size doesn't
  need to be lowered at all -- **`max-num-seqs` alone was the fix.**
- `batch=2048, seqs=4`: survived 100k depth, but **crashed at 366k depth** -- same failure
  signature as the original bug (`torch.AcceleratorError` in `sparse_attn_indexer.py` ->
  `_top_k_per_row_prefill_torch` -> `logits.topk`, this time reported as `CUDA error:
  unspecified launch failure` rather than "illegal memory access", but same root cause: PP2's
  indexer transient buffer overrunning its ~1 GiB margin). Confirms the crash is real-depth-
  dependent, not just a load-time check -- `seqs=4` looked fine at shallow depth and only broke
  under an actual long prefill.

**Final recommended DSpark+1M config: `--max-num-batched-tokens 2048 --max-num-seqs 2`**
(repo's original batch size, only seqs lowered from 8->2). No prefill-throughput cost from
batch size, since that never needed to move. Re-verified clean at ~367k tokens after the GPU2
incident below (this exact config, this exact depth, survived again).

**GPU2 got hard-wedged during the seqs=4/366k crash** -- `nvidia-smi` reported `[GPU requires
reset]`, and recovery required a **full machine reboot**: `nvidia-smi -r` reported "Not
Supported" on this unlocked-firmware card, `rmmod nvidia_uvm`/`modprobe` reload wasn't enough,
and even a manual PCIe secondary-bus reset via sysfs
(`/sys/bus/pci/devices/0000:c2:00.0/reset`) hard-hung in kernel D-state (uninterruptible) after
successfully unbinding the driver. **Takeaway: on these unlocked CMP-170HX cards, a crash in
this specific failure mode (indexer OOB write under real depth) is not softly recoverable --
budget for a reboot, not just a container restart, if this class of crash happens.** The
existing recovery block in the phase3/phase4 launch scripts (probe -> rmmod/modprobe -> repower)
does NOT reliably clear this; worth updating those scripts to detect `[GPU requires reset]`
specifically via `nvidia-smi --query-gpu=pstate` and fail fast with a "reboot needed" message
instead of looping on a recovery path that won't work.

**Bonus fix found while debugging the wedge:** Xorg was auto-binding to all 3 CMP-170HX cards
(visible as open `/dev/nvidia0/1/2` handles from the `Xorg` process) via the NVIDIA driver
package's stock `OutputClass "nvidia"` auto-probe rule, *despite* the AMD Lexa Pro (E9260) at
PCI `01:00.0` being the actual configured display GPU. This is why `nvidia-smi -r` and the
`rmmod`-based recovery couldn't get exclusive access to the wedged card. **Fixed** by adding
`/etc/X11/xorg.conf`:
```
Section "ServerFlags"
    Option "AutoAddGPU" "false"
EndSection
```
Verified post-fix and post-reboot: Xorg no longer opens handles on nvidia0/1/2. **Residual,
not yet fixed:** `sunshine` (game-streaming daemon, both the user session `sunshine.service`
and the login-screen `sunshine-gdm.service`) independently opens handles on all 3 CMP cards at
every startup, regardless of the Xorg fix -- it enumerates all CUDA/NVENC-capable devices
itself. This did NOT cause the wedge and does NOT block normal AI workloads (containers ran
fine through it all session), but it does block crash-recovery reset attempts and doesn't match
the "keep CMP-170HX clear of anything OS/desktop-related" intent. Fix would mean finding
sunshine's adapter/encoder-selection config (`~/.config/sunshine/sunshine.conf` is currently
empty/default) and pinning it to the AMD card only -- not yet attempted.

## 2026-08-15 (later) -- layer-rebalance disproven; util=0.95 confirmed as the real fix (at a context cost)

**Layer rebalancing does NOT change per-rank crash risk -- confirmed via vLLM's own memory log,
not just nvidia-smi.** Tested `VLLM_PP_LAYER_PARTITION` at 15,15,13 (original), 16,16,11, and
15,16,12. `nvidia-smi` showed wildly different "fill %" per rank across these (PP2 ranged
88-97.5% depending on partition), which looked like a real, fixable imbalance -- but vLLM's own
startup log (`gpu_worker.py`'s "Actual usage is X GiB for consumed memory... Current kv cache
memory in use is Y GiB" line) shows **all three ranks landing at an identical ~98.2-98.3% of
budget in every partition tested**, because `--gpu-memory-utilization` auto-normalizes each
rank's *total* usage independently -- moving layers off PP2 just makes vLLM backfill the freed
space with more KV cache to hit the same target. `nvidia-smi`'s raw MiB readings on this
unlocked-firmware hardware do not reliably reflect the real per-rank allocator state; trust the
vLLM log breakdown over `nvidia-smi` for this kind of analysis. **Do not spend more time on
`VLLM_PP_LAYER_PARTITION` as a crash-risk lever** -- it only reshuffles weights vs. KV cache
inside an already-fixed total budget.

**Root cause, verified directly in our own checked-out source:** `vllm/v1/worker/gpu_worker.py`
sets `redundancy_buffer_memory = 150 * (1 << 20)` -- a **fixed 150 MiB** OOM safety margin,
completely independent of the `--gpu-memory-utilization` value chosen. This means 0.95, 0.96,
0.97, and 0.98 all carry the identical *nominal* protection from vLLM's own bookkeeping; picking
a value in that band is not meaningfully safer, it just trades away KV cache room. (Checked and
ruled out a second candidate cause: historical vLLM bug where speculative-decoding draft-model
weights weren't counted during memory profiling, PR #14067 -- traced `load_model()` in our
`model_runner.py` and confirmed the DSpark drafter's weights DO load inside the same
`DeviceMemoryProfiler` block that measures `model_memory_usage`, so that specific bug does not
apply to our fork. Not fully ruled out: whether DSpark's *transient/activation* memory during a
real draft-and-verify forward pass is fully captured by the profiling run, vs. just its static
weights.)

**`--gpu-memory-utilization 0.95` does NOT fit DSpark + max-model-len=1,048,576** (needs 2.65
GiB KV cache on the tightest rank, only 0.79 GiB available -- estimated max fittable length at
this util is ~5,424 tokens). **0.95 does not fit at half context (524,288)** either (needs 1.92
GiB, only 1.44 GiB available; estimated max ~175,616). **0.95 DOES fit at quarter context
(262,144)** -- loads cleanly, and by vLLM's own log all three ranks land at ~3.0 GiB free each
(vs. ~1.1-1.6 GiB at util=0.98), roughly double the real margin, still evenly distributed across
ranks regardless of partition (used the original 15,15,13 split for this testing since partition
doesn't matter, per the finding above).

**Deliberately tried to reproduce the crash at util=0.95/ctx=262144 -- survived every known
trigger:**
- `max-num-seqs=4` (the exact setting that crashed at real depth previously) at ~256,920 tokens
  depth (98% of the 262,144 ceiling): **survived**, decode continued normally afterward.
- `max-num-seqs=8` (repo's original default, the most aggressive setting, never before tested at
  real depth): same ~256,920-token depth: **survived**.
- 3x cold restart cycles (testing for the unexplained near-startup crash seen earlier today on a
  since-abandoned rebuild): **all 3 healthy**, no recurrence.

**Conclusion (superseded below): `--gpu-memory-utilization 0.95` is a working fix -- at a real
cost.** The tradeoff is explicit and unavoidable given how tightly DSpark's extra weights fill
these cards: full 1M context (`0.98`, crashed twice) vs. reliable 262k context (`0.95`, survived
every stress test thrown at it including the two known crash triggers and repeated cold
starts). No-spec remains unaffected by any of this -- it was never the problem, and still
supports the full 1M context cleanly on its own (see the 2026-08-15 Phase 5 entry above).

## 2026-08-15 (final) -- full binary-search campaign between 0.95 and 0.98, production default set

Systematically mapped the boundary between the known-good `0.95`/262k and known-bad `0.98`/1M
points, testing 0.96, 0.965, and 0.97. For each utilization level: binary-searched the exact max
`max-model-len` that fits (via vLLM's own "ValueError: X GiB KV cache needed, Y GiB available"
messages and iterative retries -- the error's own "estimated maximum model length" is a
mild *underestimate*, since it's computed from the failed larger request's block-table overhead;
the real ceiling is always a bit higher than the first estimate suggests), then relaunched at
that exact ceiling with `--max-num-seqs 8` (the most aggressive setting, not just the `2` used
for the 0.95 baseline) and ran a real deep-prefill stress test at ~91-98% of that ceiling --
the same methodology that reproduced the original crash.

| util | max ctx that fits | seqs=8 stress test at ~95% depth |
|---|---|---|
| 0.95 | 338,432 | PASSED (328,164 tokens depth) |
| 0.96 | 574,464 | PASSED (534,950 tokens depth) |
| **0.965** | **691,968** | **PASSED (629,809 tokens depth)** |
| 0.97 | 799,488 | **CRASHED** -- Worker_PP2, `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`, same signature as every prior crash. GPU state stayed clean afterward (no reset flag), no reboot needed this time. |
| 0.98 | 1,048,576 (full) | CRASHED (established earlier, twice) |

**Hard cliff between 0.965 (safe) and 0.97 (crashes) -- not a gradient.** 0.965 passed the exact
same aggressive test (`seqs=8`, ~95% depth) that broke 0.97, with a comfortable enough margin
that it's the recommended ceiling, not a knife's-edge pick.

**New production default: `util=0.965`, `max-model-len=691968`, `max-num-seqs=8`.** Chosen over
`seqs=2` because the box only ever has ~2 real users -- `seqs=8` gives headroom for that without
any proven downside, since it already passed the stress test at that setting. 2.6x more context
than the earlier `0.95`/262k config, while still passing every crash-reproduction test thrown at
it. Script: `~/CMP-170HX-PROJECT/vllm-dsv4/phase5-launch-dspark-production.sh [util] [maxlen]
[max_num_seqs] [partition] [max_batched_tokens]` (defaults 0.965/691968/8/15,15,13/2048 -- this
is now THE default launch script, superseding phase4b/phase4c/phase4d for normal use; those
remain useful for future re-testing if the config needs revisiting).

**Incidental finding during this campaign:** a module-reload recovery attempt (after a routine,
non-hard-wedged "device busy" state) this time broke CUDA init across **all 3** GPUs at once
(worse than the usual GPU2-only pattern), and restarting `nvidia-persistenced` did not fix it --
only a full reboot did. Recovery approach going forward: if `rmmod nvidia_uvm`/`modprobe` doesn't
immediately restore all 3 GPUs, don't keep trying more module-level fixes -- just reboot.
