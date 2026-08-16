# CMP-170HX Dual-Card P2P Enablement — Project Plan

Status: DRAFT — Phase 0/1 (raw P2P capability + the `aikitoria` patch) still not started. Written
2026-08-08 after the `-sm tensor` benchmark showed dual-card tensor-parallel decode 7.4%
*slower* than single-card (89.72 → 83.04 tok/s) on Qwen3.6-35B-A3B, and after reading the
community's dedicated P2P investigation. See `bench-qwen36-35b-a3b-udq4-tensorparallel-cmp170hx.sh`
and its results in `bench-results/`.

**2026-08-13 update:** a second, independent TP baseline now exists — vLLM
`--tensor-parallel-size 2` (unpatched, no P2P), tested directly against a real inference
workload rather than a synthetic collective. See §1 and the new §8 below. Headline: **it
completed cleanly, no hang** — this answers part of Phase 2 step 3's question but does *not*
substitute for Phase 0/1, since no P2P patch was involved. The hang risk this plan flags
remains specifically about the *patched* P2P path, not vLLM TP in general on this hardware.

---

## 0. Reality check — read this before starting

This is not a "run a known patch, get a known win" task. It's genuinely unsolved. The
community's own P2P research page states it plainly: **peer-to-peer is absent on the CMP 170HX
by default, no shipping code enables it, and the one third-party patch that might change that
has never been validated on this card's GPU family (GA100).** [1] Three separate structural
facts push the expected value of this project toward low:

1. **PCIe generation is hardware-fused, not a driver limitation.** Gen1→Gen2 was cracked in
   software (already applied on this box — see §1). Gen3/Gen4 is a confirmed hardware fuse
   with no known software path; the community maintainer's own assessment as of 2026-07-27 is
   "Gen 3 doesn't work whatsoever, it's going to require a GSP patch... I haven't seen anybody
   at all get a working GSP patch." [2] This directly matters here because:
2. **The community's own stated precondition for P2P being worth attempting at all is "at
   least PCIe Gen3, otherwise it's kind of a waste on these cards."** [1] That precondition is
   currently unreachable on this hardware. Gen2 x16 (reachable via a physical mod, see §5) is
   *below* that stated threshold.
3. **BAR1 is a fixed, non-resizable 64 MiB window on this card**, even though the unlocked
   framebuffer is 64 GiB. [1] The `aikitoria` P2P patch maps peer memory through BAR1. No
   source has established whether BAR1 P2P can function through a 64 MiB aperture at all — this
   may be why the one negative field report exists.

None of that makes the project pointless — open question #1 on the community's own tracker
("does the layered patch enable P2P between two 170HX cards?") is explicitly described as
"nothing else in this domain is as cheap or as decisive," [1] and nobody has actually run that
test on a 170HX-only pair, patched or unpatched. This plan is scoped around answering that
question cheaply *first*, before committing to the riskier driver-patching work, and it
recommends a clear stop condition if the cheap test comes back negative.

**Recommendation: run Phase 0 (a few hours, non-destructive, fully reversible) before deciding
whether Phases 1–3 are worth the time.** If Phase 0 confirms no P2P and no realistic path to
one, the existing `-sm layer` / `--fit` pipeline-parallel approach already in use on this box
remains the right answer, and that's a legitimate outcome of this plan, not a failure of it.

---

## 1. Current measured state (this system, 2026-08-08)

- 2× CMP 170HX, unlocked via `cmpunlocker` (nvidia-open 610.43.0x), 65536 MiB each, confirmed
  via `CARD-REGISTRY.md`.
- `gen2.service` **enabled** — live check today: both cards report
  `LnkCap: Speed 5GT/s, Width x16` / `LnkSta: Speed 5GT/s, Width x4 (downgraded)`. Gen2 speed is
  active; width is still x4 (no capacitor mod done).
- Topology: PHB (no shared switch), no NVLink, no P2P — `torch`/`nvidia-smi` level checks not
  yet run on this box specifically, but this matches the community's own connectivity table for
  unlocked 170HX fleets (0 of 56 pairs P2P-capable, all pairs, including within a `PIX` group). [1]
- Measured today: `-sm tensor` (mainline llama.cpp, true tensor split, weights+KV across both
  cards) is **7.4% slower** on decode and **31.7% slower** on prefill than single-card, for
  Qwen3.6-35B-A3B UD-Q4_K_XL. This matches the community's own TP measurement on a *different*
  model (Qwen2.5-72B dense, vLLM, Gen1 x4): TP was 2.3–2.8x worse at prefill for a +23% decode
  gain — direction differs by model/engine, but the underlying message is the same: **TP is a
  measured net negative-to-marginal on this link today, PP is not.** [1]
- `HOWTO-UNLOCK-CMP-170HX.md` (this project's own doc, line 763) already lists
  "Peer-to-peer GPU transfers | Attempted by the community, unresolved" in its own capability
  table — this project is picking up a problem this repo already flagged as open, not starting
  from zero.
- **2026-08-13, unpatched vLLM TP=2 baseline:** cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit, vLLM
  `--tensor-parallel-size 2`, MTP n=3, 262144 ctx, 100 W/card, no P2P patch applied — completed
  cleanly at **146.29 tok/s** (24,000 completion tokens, no hang, no NCCL error). For comparison,
  independent single-GPU replicas on the same two cards got 127.44 and **154.02** tok/s (the
  latter beating a dual-3090-at-195W baseline of 151.80 tok/s outright). TP=2 is ~3–13% slower
  than running each card standalone, consistent with the GGUF `-sm tensor` finding above and the
  community's own PP-over-TP conclusion — see §8 for full detail.

---

## 2. What "P2P" would actually change here, mechanically

`-sm tensor` in llama.cpp splits weights *and* KV cache across GPUs and requires multiple
cross-GPU reductions per layer, every token. [3] Without P2P, every one of those reductions
stages through host RAM (GPU0 → host RAM → GPU1, and back) instead of a direct DMA write into
the peer GPU's memory. [1] P2P wouldn't change *how much* data moves — it changes *the path* it
takes and removes the round-trip through host RAM.

Two independent things determine whether that helps:

- **Bandwidth** — set by PCIe generation × width. Already at Gen2 x4 on this box.
- **Whether the transfer is direct GPU↔GPU or staged through host RAM** — set by P2P
  availability, which is a **driver policy gate**, not a bandwidth number. NVIDIA disables true
  P2P DMA on consumer/GeForce-derived silicon by default, confirmed by NVIDIA itself, and this
  applies architecture-wide. [4] [5] The CMP 170HX's GA100 die is the same family as the A100,
  which *does* support P2P in its professional form — so the capability plausibly exists in
  silicon, gated by firmware/driver rather than physically fused off, but **nobody has
  confirmed this for the 170HX specifically.** [1]

---

## 3. Phase 0 — cheap, decisive, non-destructive (do this first)

Goal: answer the community's own open question #1 — "does raw P2P work between two 170HX
cards at all, patched or not?" — without touching the driver.

1. Run `torch.cuda.can_device_access_peer(0,1)` and the reverse pair on this box, both
   directions. Expected result, per the community's fleet-wide data: `False`. [1]
2. Run NVIDIA's `simpleP2P` and `p2pBandwidthLatencyTest` samples (from `cuda-samples`) on this
   exact two-card, 170HX-only pair. **Nobody in the community corpus has posted this matrix for
   a 170HX-only host** — this is genuinely new data, not a repeat of someone else's test. [1]
3. Check `nvidia-smi nvlink -s` and `nvidia-smi topo -m` for a topology sanity check against the
   community's own capture (`Device does not have or support Nvlink`, `PIX` grouping with P2P
   still reported unavailable). [1]
4. Record results in `bench-results/` alongside a short write-up, and (optional but
   encouraged, since this is genuinely unposted data) consider contributing the connectivity
   matrix back to the `Consensus-Protocol/cmp170hx` wiki's open-questions tracker — it's
   explicitly asked for. [1]

**Stop condition:** if step 1/2 confirm no P2P and nothing here suggests a config-only fix
(ACS, IOMMU mode), decide explicitly whether Phase 1's driver-patching risk is worth it before
proceeding — don't drift into it by default.

---

## 4. Phase 1 — layer the `aikitoria` P2P patch onto `cmpunlocker` (if Phase 0 doesn't rule it out)

This card is **not** on the patch's supported list (RTX 3090/4090/5090 only; GA100 is
explicitly untested) [4], and the modified code paths (`kern_bus_gp100.c`, `io_vaspace.c`,
`nv_gpu_ops.c`) are Pascal-and-later bus code with no confirmed GA100 branch inside them. [1]
This phase is genuine experimentation, not applying a known-good recipe.

**Recipe** (community-documented, three-commit diff, medium confidence — reported working by
one tester, not independently reproduced on a 170HX-only host): [1]

```bash
git clone https://github.com/aikitoria/open-gpu-kernel-modules open-gpu-kernel-modules-p2p
git -C open-gpu-kernel-modules-p2p diff --src-prefix=a/ HEAD~3 > ./cmpunlocker/driver/patches/0009-unlock-p2p.patch
cd ./cmpunlocker && sudo ./install.sh
```

**Before running this, apply these corrections to the community recipe:**

- **Cherry-pick only the P2P commit (`9fb650447c7b`), not the full `HEAD~3` range.** The range
  also includes an experimental hugepage `cudaHostRegister` change with no GA100 validation of
  any kind, described by its own author as skipping page-table bookkeeping "the stock driver
  handles correctly" in edge cases. Taking the full diff means it activates automatically with
  no way to isolate its effects from the P2P change. [1]
- **Number the patch `0009` or later**, not `0007` — this box's `cmpunlocker` checkout already
  uses `0007-pcie-gen2.patch` and `0008-pcie-gen2-probe-retrain.patch` (confirmed present in
  `cmpunlocker/driver/patches/` on this system). `build.sh` applies patches in lexicographic
  order; a colliding number either fails to apply or applies out of order. [1]
- **Check the device-ID gate before assuming a clean build means a working patch.** One
  unverified report claims `cmpunlocker`'s own patch stack gates P2P behind
  `devId == 0x20C2` read from a field (`pGpu->idInfo.PCIDeviceID`) that isn't populated yet at
  the point it's checked — meaning the gate silently never opens even on a successful build.
  Upstream `aikitoria` sets the equivalent flags unconditionally. Worth diffing before assuming
  a successful build implies an active patch. [1]
- **`build.sh` does no checksum/signature verification** on the upstream tarball it fetches, and
  this compounds when layering a second unverified diff on top. Build on a system you can
  reinstall, not the primary inference box, at least for the first attempt.
- **IOMMU passthrough is required** (`amd_iommu=on iommu=pt` for this box's AM5 platform) or BAR1
  DMA writes get translated and transfers fail. The `cmpunlocker` Gen2 branch already configures
  this correctly and this system is on Gen2, so this may already be satisfied — verify with
  `grep -qw iommu=pt /proc/cmdline`. **This weakens DMA isolation system-wide** — acceptable on
  a single-user workstation, not on anything multi-tenant. [1]
- **Check ACS separately from IOMMU.** If P2P negotiates but is slow, root-port ACS is forcing
  traffic through the CPU root complex anyway, which defeats the point. Options in order of
  preference: disable ACS in BIOS; `pcie_acs_override=downstream,multifunction` kernel param; or
  an ACS override patch (this further weakens IOMMU group isolation — same caveat as above). [1]
- **Secure Boot must be disabled** — the patched modules are unsigned (same requirement as the
  base `cmpunlocker` install already in use on this box).

**Verification after install:** compare `/sys/module/nvidia/srcversion` against
`modinfo -F srcversion` on the patched `nvidia.ko`. A mismatch means stock modules won the load
race and neither the base unlock nor the P2P patch is active — this is the same check
`cmpunlocker`'s own build script already does. [1]

---

## 5. Phase 2 — validate against the actual inference stack, not just the driver

A raw P2P capability check passing is **not** sufficient — there's a specific, credible
(unverified but mechanistically plausible) failure mode on record: peer *data movement* working
while peer *synchronisation* does not, which would let raw `cudaMemcpyPeer` succeed while NCCL
and every collective-based multi-GPU library (including llama.cpp's cross-GPU reductions and
vLLM's custom all-reduce) hangs indefinitely at transport connect. [1]

Test in this order, stopping at the first failure:

1. `can_device_access_peer` — already covered in Phase 0/1.
2. Timed `cudaMemcpyPeer` between the two cards — confirms raw DMA, independent of any
   collective library. Community reference (unverified, different rig): 6.25 GB/s peer copy vs.
   5.70 GB/s staged through host memory — a real but modest win if it reproduces here. [1]
3. A minimal NCCL collective (or llama.cpp/vLLM actually loading with `-sm tensor` /
   `--tensor-parallel-size 2` and completing one generation). **This is the one that reportedly
   hangs** on the only rig that's tried it. If it hangs: kill it, don't let it sit — the report
   describes both GPUs pinning at 100% with no timeout.
   - ✅ **Partially answered 2026-08-13, unpatched:** `vLLM --tensor-parallel-size 2` was run to
     completion on this exact two-card host (no P2P patch, no Phase 1 applied) — see §8. No
     hang, 146.29 tok/s. This confirms the *unpatched* NCCL/vLLM collective path is safe on this
     hardware. It does **not** confirm anything about the *patched* path Phase 1 would add —
     that's still an open, untested question, and the one community hang report may be specific
     to the patched driver, a different vLLM version, or a different rig. Don't skip Phase 1's
     own verification on the strength of this result alone.
4. Only if step 3 succeeds: rerun this project's own
   `bench-qwen36-35b-a3b-udq4-tensorparallel-cmp170hx.sh` and compare against today's baseline
   (single 89.72 tok/s, dual-no-P2P 83.04 tok/s decode) to get a real before/after number.

**Rollback plan:** `cmpunlocker/remove.sh --yes` plus reverting IOMMU/ACS kernel params, cold
reboot. Confirm this rollback path works *before* starting Phase 1, on a config you're not
already relying on for other benchmarks (this box currently runs several other active
benchmark projects — DeepSeek-V4-Flash, GLM, Kimi — that depend on the driver staying stable).

---

## 6. Phase 3 (optional, separate track) — physical x16 mod

Independent of P2P; this only affects link *width*, not whether P2P works. Included because the
community's own position is that Gen2 x16 is the practical ceiling and it's a prerequisite for
any TP attempt to be worth revisiting even without P2P:

- 24× 0402 220 nF X7R AC-coupling capacitors, confirmed working parts: Taiyo Yuden
  `MAASJ105SB7224KFCA01` or Samsung `CL05B224KO5NNNC`. [6]
- Reference designators C1100–C1350, ~20 minutes/card by hand, described as beginner-to-hobbyist
  rework. [6]
- Restores x16 *width* only — generation stays at whatever the driver has negotiated (Gen2 on
  this box already). Result: Gen2 x16, reproduced on two independent rigs at 5.97–6.67 GB/s. [1]
- This is still *below* the community's stated "Gen3 or don't bother" threshold for TP. [1] Treat
  this phase as raising the ceiling for Phase 1/2 results, not as a fix on its own — and it's the
  only phase in this plan that's physically irreversible (desoldering/soldering on the card).

---

## 7. Decision gates

| Gate | Proceed if | Stop / fall back to `-sm layer` if |
|---|---|---|
| After Phase 0 | `simpleP2P`/`p2pBandwidthLatencyTest` show *any* peer bandwidth above host-staged baseline | Confirms 0/2 pairs capable, matching fleet-wide community data, with no config knob (ACS/IOMMU) left untried |
| After Phase 1 build | `srcversion` matches, patch confirmed active | Build fails, or active-patch check fails — don't assume a clean build = working patch |
| After Phase 2 step 2 | Peer `cudaMemcpyPeer` beats staged copy | No measurable difference — matches the one existing negative report |
| After Phase 2 step 3 | A real collective/inference run completes | Hangs — kill it, record it, revert. This is the documented failure mode; don't debug a hang past a few minutes on faith. **Unpatched vLLM TP=2 already cleared this gate 2026-08-13 (§8) — Phase 1's patched variant still needs its own run of this gate, don't assume it inherits the unpatched result** |

At any gate that resolves negative, the answer "P2P doesn't work on this card yet, use
`-sm layer` / `--fit`" is a valid, complete, already-partially-validated conclusion — the
project doesn't need Phase 3 or a "yes" at every gate to be worth having run.

---

## 8. 2026-08-13 finding — unpatched vLLM TP=2, real inference workload, no hang

Run outside this plan's original phase order (no Phase 0/1 P2P work had been done — this was a
straight vLLM throughput benchmark that happened to also answer part of Phase 2 step 3), while
reproducing a dual-3090 vLLM benchmark on this system's two CMP-170HX cards for comparison.
Full detail and raw logs in `CARD-REGISTRY.md`'s "2026-08-13 dual-card vLLM+AWQ benchmark"
section and `bench-results/thermal_p3_live.csv` / `phase3_run.log`.

**Config:** cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit, vLLM v0.24.0 (official CUDA image), MTP n=3
speculative decoding, 262144 ctx, `--tensor-parallel-size 2`, both cards at 100 W, unpatched
driver (no `aikitoria` P2P patch, no IOMMU/ACS changes beyond what `cmpunlocker`'s Gen2 branch
already sets).

**Result:** completed cleanly end to end — 146.29 tok/s (24,000 completion tokens, hit
`max_tokens` cap rather than stopping early), no NCCL error, no indefinite hang, peak temps
65 C core / 74 C HBM. MTP speculative decoding worked correctly under TP too (81–100% draft
acceptance per-window, in line with single-card numbers).

**One transient warning, not a hang:** `vLLM`'s own `shm_broadcast.py:705` logged *"No
available shared memory broadcast block found in 60 seconds. This typically happens when some
processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache
quantization)."* This fired while Worker_TP0 had finished its ~50s `torch.compile` pass and
Worker_TP1 was still mid-compile on its own timeline — i.e. exactly the benign cause the
warning itself names. It cleared on its own once TP1's compile finished; no manual
intervention was needed. Worth knowing this log line will appear on a normal boot and is not
by itself evidence of the failure mode Phase 2 step 3 warns about.

**How this relates to the rest of the plan:**

- This does **not** test P2P. No patch was applied, so the collective almost certainly used the
  same host-RAM-staged path as the GGUF `-sm tensor` baseline in §1 — this run shows that path
  is *reliable* (no hang) on this exact hardware/software combination, which the plan's original
  wording treated as an open risk.
- It does **not** resolve whether Phase 1's patch would work, be faster, or introduce a new
  hang mode — that patch changes the code path this run exercised. Phase 0/1 remain genuinely
  unstarted.
- It **does** update the practical recommendation: TP=2 is a safe fallback to reach for (e.g. a
  single request needing more VRAM than one card offers) without fear of the hang this plan
  flagged as Phase 2's biggest risk — at least for vLLM 0.24 on this driver/hardware. It is
  still not a throughput win: two independent single-GPU replicas (127.44 + 154.02 = 281.46
  tok/s combined) comfortably beat TP=2's 146.29 tok/s for aggregate throughput, matching this
  plan's standing PP-over-TP conclusion.

---

## Sources

1. [Consensus-Protocol/cmp170hx — `docs/frontier/p2p.md`](https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/frontier/p2p.md) — the primary source for this plan: measured connectivity data, the three-commit patch recipe, BAR1 constraint, unverified peer-DMA-vs-sync report, IOMMU/ACS requirements, and the open-questions list.
2. [Consensus-Protocol/cmp170hx — `docs/frontier/pcie-gen3-gen4.md`](https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/frontier/pcie-gen3-gen4.md) — confirms Gen3/Gen4 is fuse-blocked with no known software path.
3. [llama.cpp `docs/multi-gpu.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) — split-mode mechanics (`layer` vs `row` vs `tensor`), cross-GPU reduction behavior.
4. [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules) — the P2P driver patch itself; supported-GPU list explicitly excludes GA100/CMP.
5. [NVIDIA GeForce GPUs Do Not Incorporate Peer-To-Peer Support — Wccftech](https://wccftech.com/nvidia-geforce-gpus-do-not-incorporate-peer-to-peer-support-redirecting-consumers-to-purchase-expensive-gpus/) / [Tom's Hardware confirmation](https://www.tomshardware.com/news/nvidia-confirms-geforce-cards-lack-p2p-support) — NVIDIA's driver-level P2P restriction on consumer silicon, the policy this project is trying to work around.
6. [Consensus-Protocol/cmp170hx — `docs/operations/physical-mods.md`](https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/operations/physical-mods.md) — capacitor mod bill of materials, procedure, and the width-vs-speed distinction.

Also reviewed, not directly cited above: [cmpunlocker (amoghmunikote)](https://github.com/amoghmunikote/cmpunlocker) — the unlock tool already installed on this box, whose own capability table does not list P2P as solved; [engineered.at — patching NVIDIA's driver and vLLM for P2P](https://engineered.at/articles/patching-nvidia-s-driver-and-vllm-to-enable-p2p-on-consumer-gpus); [NVIDIA developer forum — cuSolverMG and CMP 170HX](https://forums.developer.nvidia.com/t/cusolvermg-and-cmp-170-hx/299423) (multi-GPU library hang, resolved by driver upgrade, no P2P-specific findings); [NVIDIA developer forum — cheaper way to 1T VRAM (8× 170HX + DGX Spark)](https://forums.developer.nvidia.com/t/cheaper-way-to-go-1t-vram-170hx-4-sparks/378348) (independent real-world confirmation of no-P2P and TP-underperforms-PP on a larger 170HX cluster); [HuggingFace — kostakoff, CMP 170HX inference notes](https://huggingface.co/posts/kostakoff/660728237282538) (general CUDA/inference viability, not P2P-specific); this project's own `HOWTO-UNLOCK-CMP-170HX.md` (line 763, lists P2P as community-attempted and unresolved) and `cmpunlocker/driver/patches/` (confirms `0007`/`0008` are already in use on this box, informing the patch-numbering correction in §4).
