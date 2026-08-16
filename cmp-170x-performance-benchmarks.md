# CMP 170HX Performance Benchmarks & Token-Speed Predictions

> **⚠️ FROZEN PREDICTION — DO NOT OVERWRITE.** This document is a pre-hardware-arrival
> estimate, written 2026-08-04 before your 4× CMP 170HX cards arrived. Per your
> instruction, these numbers must never be edited or replaced once written. When
> the cards arrive and you have real measurements, record them in a **separate
> new file** (e.g. `cmp-170hx-actual-results.md`) so the prediction and the
> reality can be compared side by side. If this file needs a correction for a
> factual research error, add a dated addendum at the bottom — do not alter the
> original numbers above it.

## 0. What this covers

Estimated single-user (batch=1) decode and prefill token speeds for:

1. **Qwen3.6-27B UD-Q4_K_XL, MTP enabled, full context**
2. **Qwen3.6-35B-A3B UD-Q4_K_XL, MTP enabled, full context**
3. **DeepSeek-V4-Flash-0731 UD-Q8_K_XL, DFlash speculative decoding, full context**

...on 4× NVIDIA CMP 170HX, pre-unlock and post-unlock, with explicit reasoning
about tensor vs. pipeline parallelism and PCIe bandwidth. Two hardware variants
are carried in parallel throughout because you haven't specified which you're
getting and it materially changes feasibility (see §4):

- **8GB-variant cards** → unlock to **64GB** each (`10de:20c2`)
- **10GB-variant cards** → unlock to **40GB** each (`10de:2082`)

Every number below is either (a) a directly cited real-world measurement from
someone else running LLM inference on this exact unlocked card, or (b) an
estimate I derived and clearly labeled as such, with the reasoning shown so you
can audit or correct it. Confidence is explicitly stated per number — nobody
has published a benchmark of these three specific model+quant combinations on
this hardware yet, so several figures are extrapolated from close analogs.

---

## 1. CMP 170HX hardware specs — pre-unlock vs. post-unlock

| Property | Pre-unlock (stock) | Post-unlock | Source / confidence |
|---|---|---|---|
| Die | GA100 (same silicon as A100) | same | high — consistent across all sources |
| SMs visible/schedulable | 70 (`smid` 0–69, no gaps) | 70 (unchanged) | high — direct `smid` probe, [Consensus-Protocol wiki, compute-throttle.md](https://github.com/Consensus-Protocol/cmp170hx) |
| CUDA cores | 8,960 (70 SM × 128) | unchanged | high |
| Tensor cores | 280 | unchanged | high |
| Memory capacity | **8GB or 10GB** (firmware/OTP-restricted) | **64GB (8GB card) / 40GB (10GB card)** | high — this is cmpunlocker's core function |
| HBM2e bus width | 4096-bit | unchanged | high |
| **Memory bandwidth** | **~1,306–1,600 GB/s measured** (theoretical 1,555 GB/s) | **~1,592–1,599 GB/s measured — essentially identical to locked state** | **high** — directly cited: "Stock vs. modded (same card): 1592-1599 GB/s" ([performance.md](https://github.com/Consensus-Protocol/cmp170hx)). **The unlock does not increase memory bandwidth — only capacity.** |
| FP32 (scalar/FMA, non-tensor) | 0.30–0.41 TFLOPS (deliberately throttled) | 12.2–12.9 TFLOPS | high — 26–32x uplift, multiple independent measurements |
| FP64 tensor (DMMA) | ~0.20 TFLOPS | 11.6–12.0 TFLOPS | high — ~59–62x uplift |
| FP16 tensor | reported anywhere from ~42 TFLOPS to ~48.7 TFLOPS pre-unlock depending on tool | 158.7–190 TFLOPS post-unlock depending on benchmark tool (OpenCL-Benchmark vs. clpeak disagree by ~4x) | **medium** — wide variance between benchmarking tools even within the same wiki; see §1a caveat |
| BF16 tensor | not separately reported pre-unlock | 164.4–202.1 TFLOPS | medium, same caveat |
| PCIe | **Gen1 x4, physically wired for x4 only** (extra lanes exist on the A100-shared PCB but their AC-coupling capacitors are unpopulated — de-soldering fix requires hand-soldering 24 capacitors and is a hardware mod, not part of cmpunlocker) | **Gen2 x4 by default** — cmpunlocker has shipped a software Gen2 retrain patch on `master` since 2026-07-29; lane **width stays x4** regardless | high |
| PCIe bandwidth (measured) | 0.80–0.85 GB/s | 1.68–1.71 GB/s (Gen2 x4) | high, directly measured |
| NVLink | absent, unpopulated | absent — not something cmpunlocker adds | high |
| GPU-to-GPU P2P | not supported | **still not supported** — the wiki's own "frontier/unsolved problems" section lists P2P as unsolved | high — this is the single most important fact for the parallelism decision in §5 |
| TDP | 250W | 250W (same power limit) | high |
| Power draw, real inference load | not separately reported | 170–287W observed during actual LLM decode (varies by model/tuning — see §6 footnotes) | medium |

### 1a. Why the FP16 TFLOPS numbers disagree with each other

Two different community benchmark passes on the *same* unlocked hardware
reported FP16 as low as 48.74 TFLOPS (OpenCL-Benchmark) and as high as 179.19
TFLOPS (clpeak) on different dates. This isn't a typo — it reflects real
variance in how different tools exercise the tensor cores (GEMM shape, kernel
choice, batch size), not a single ground-truth number. **I am not using raw
TFLOPS as the basis for the token-speed estimates below** — I'm using directly
measured LLM decode/prefill numbers instead (§3), which already bake in
whatever real-world kernel efficiency actually happens. TFLOPS is included
here for completeness and for the prefill reasoning in §6, not as a decode
predictor.

### 1b. The load-bearing conclusion from §1

Because **memory bandwidth doesn't change with the unlock** (only capacity
does), and single-user decode on a model that already fits in VRAM is
bandwidth-bound, not compute-bound — **the compute unlock (FP32/FP64/Tensor
gains) mostly doesn't matter for decode speed on models that fit**. What the
unlock actually buys you for decode is *capacity to fit the model on-GPU at
all* (see §4) rather than a faster decode engine underneath a model that was
already resident. It matters much more for **prefill** (§6), which is
genuinely compute-bound.

---

## 2. Target model specs (fetched from HF configs 2026-08-04)

| | Qwen3.6-27B | Qwen3.6-35B-A3B | DeepSeek-V4-Flash-0731 |
|---|---|---|---|
| Architecture | Dense, hybrid linear+quadratic attention | Sparse MoE, same hybrid attention family | Sparse MoE, MLA-style compressed attention |
| Total params | 27B (all active) | 35B total | 284B (per Baseten/MarkTechPost) or 304B (per HF model card) — **sources disagree by 20B; I could not resolve this from the config.json alone**, treat as ~290B ± 10% |
| Active params/token | 27B (dense) | 3B (top-8 routed + 1 shared expert) | ~13B (commonly cited figure; not independently verifiable from config.json) |
| Layers | 64 (16× [3× Gated DeltaNet + 1× Gated Attention]) | 40 (10× [3× (DeltaNet→MoE) + 1× (Attention→MoE)]) | 43 |
| Hidden size | 5120 | 2048 | 4096 |
| Attention heads (Q / KV) | 24 / 4 | 16 / 2 | 64 / 1 (MLA-style, compressed) |
| Routed experts | n/a | 256 total, 8 active/token | 256 total, 6 active/token + 1 shared |
| Native context | 262,144 (YaRN-extensible to 1,010,000) | 262,144 (YaRN-extensible to 1,010,000) | 1,048,576 native |
| Quant used | UD-Q4_K_XL | UD-Q4_K_XL | UD-Q8_K_XL |
| **Quant file size (verified via direct HF file listing)** | **17.6 GB** (single file) | **22.4 GB** (single file) | **167.16 GB** (5 shards: 5.26+49.2+49.7+49.5+13.5 GB) |
| Speculative decoding | MTP (native head, per your naming) | MTP (native head) | DFlash: 5-layer sliding-window (window 2048) draft model, `hc_mult=4`, drafts up to 7 tokens/step — architecturally distinct from generic MTP |

### 2a. Estimated KV cache size at full native context (fp16, my own derivation)

This determines how much headroom is left after weights for "full-context" use.

- **Qwen3.6-27B**: only the 16 quadratic-attention layers accumulate a growing
  KV cache (the 48 Gated DeltaNet layers use a fixed-size recurrent state, not
  a growing cache — a real advantage of this hybrid architecture for long
  context). Assuming head_dim 128: 2×4×128×2 bytes = 2,048 bytes/token/layer ×
  16 layers = 32 KB/token → **≈8.4 GB at 262,144 tokens**.
- **Qwen3.6-35B-A3B**: same hybrid pattern, 10 quadratic layers. head_dim 128:
  2×2×128×2 = 1,024 bytes/token/layer × 10 = 10 KB/token → **≈2.6 GB at
  262,144 tokens** (≈10.1 GB if extended to the 1.01M YaRN ceiling).
- **DeepSeek-V4-Flash-0731**: MLA-style compressed cache, not per-head KV
  (`num_key_value_heads: 1` in config confirms this). Inferring a
  DeepSeek-V3-style compressed rank (~512) + `qk_rope_head_dim` (64, confirmed
  in config) = 576 compressed dims/token/layer × 2 bytes × 43 layers ≈ 49.5
  KB/token → **≈48–52 GB at the full 1,048,576-token native context**. This is
  a large number and drives the capacity math in §4 — **I could not confirm
  `kv_lora_rank` directly in the config**, so treat this as a reasoned
  approximation, not a measured value.

---

## 3. Directly cited real-world LLM inference numbers on unlocked CMP 170HX

These are the best evidence available — actual tokens/sec figures other
people have reported running inference engines on unlocked (64GB or 40GB)
CMP 170HX cards, pulled from the [Consensus-Protocol/cmp170hx wiki's
operations/ pages](https://github.com/Consensus-Protocol/cmp170hx). None of
these are your exact three model+quant combos, but several are extremely
close analogs.

**Single 64GB card, llama.cpp/ik_llama, real-workload tuning tests:**
| Model | Config | Decode (tok/s) | Notes |
|---|---|---|---|
| Qwen 27B (quant not explicitly stated, but doc recommends "q4-class is the practical sweet spot on a 64GB card") | MTP on, clock-offset tuning | **55 → 59 t/s** (+7% from offset tuning) | Power draw 268W→287W during the test |
| Qwen 35B(-A3B implied) | MTP on, offset tuning | **130 → 144 t/s** (+11%) | Power draw 170W→185W |
| unspecified model | ik_llama, Q8, **no MTP**, 40GB card | tg128 = 26.36 | pp512=271.49, pp2048=584.31, pp8192=697.18 |
| unspecified model, same rig | ik_llama, **with MTP**, 40GB card | tg128 = **38.15 (Gen1 x4) → 41.52 (Gen2 x4)** | pp512 = 203.84 → 277.84 |

**Single 64GB card, vLLM (different engine, included for cross-check):**
| Model | Decode (tok/s) | Prefill (tok/s) |
|---|---|---|
| Qwen3.6-35B-A3B-AWQ | 113 | 1,700 |
| Qwen3.6-27B BF16 | 19.2 | 2,231 |
| Qwen3.6-27B-AWQ-INT4 | 58.5 | 2,044 |

**Multi-GPU, giant MoE, pipeline-parallel — closest available analog for
DeepSeek-V4-Flash-0731 at scale (GLM-5.2 is 744B total MoE, larger than
DeepSeek-V4-Flash's ~284–304B):**
| Setup | Decode | Prefill |
|---|---|---|
| GLM-5.2, 8×64GB, vLLM pipeline-parallel | 30.2 t/s | 2,675 t/s @ 131k ctx |
| GLM-5.2 UD-IQ2_M, 8×64GB, llama.cpp | 17.33 t/s | 113.0 t/s |
| (unconfirmed) single A100, GLM-5.2 | ~55 t/s | — |

**Explicit tensor-vs-pipeline test (Qwen2.5-72B-AWQ, directly answers §5):**
> "TP2 is 2.3–2.8x *worse* at prefill for +23% decode" — and the wiki's own
> tuning guidance states plainly: **"Parallelism across cards: pipeline,
> never tensor."**

---

## 4. Does each model actually fit? (capacity math)

Pool sizes: 4× 8GB-variant = **32GB pre-unlock / 256GB post-unlock**.
4× 10GB-variant = **40GB pre-unlock / 160GB post-unlock**.
Community-observed real llama.cpp VRAM residency on 64GB cards runs
53–60GB/64GB in practice (driver + context overhead), so treat nominal
capacity as ~90% usable, not 100%.

| Model | Weights + full-context KV | 8GB-var pre-unlock (32GB) | 8GB-var post-unlock (256GB) | 10GB-var pre-unlock (40GB) | 10GB-var post-unlock (160GB) |
|---|---|---|---|---|---|
| Qwen3.6-27B | 17.6 + 8.4 = **26.0 GB** | **Marginal fit** — 26GB of 32GB pool, ~6GB left across 4 GPUs for compute buffers; realistically needs some context trimming below full 262K to be safe | Fits on **one single card alone** with 38GB to spare — no multi-GPU needed | Fits comfortably, 14GB headroom | Fits on one card, 14GB to spare |
| Qwen3.6-35B-A3B | 22.4 + 2.6 = **25.0 GB** | **Marginal fit**, same caveat as above | Fits on **one single card alone**, 39GB to spare | Fits comfortably, 15GB headroom | Fits on one card, 15GB to spare |
| DeepSeek-V4-Flash-0731 | 167.16 + ~50 = **~217 GB** | **Does not fit — not close** (32GB vs 217GB needed) | **Fits across the 4-card pool** (217GB of 256GB, ~39GB headroom for compute buffers — workable but not generous) | **Does not fit** (40GB vs 217GB) | **Does not fit** — 217GB needed vs. 160GB pool. Even weights alone (167GB) exceed the 160GB pool. Would need to drop well below full native 1M context, or isn't viable on 4× 10GB-variant cards at all. |

**This is the single most decisive finding in this document**: DeepSeek-V4-Flash-0731 at Q8_K_XL with full native context is **only viable at all on 4× 8GB-variant (→64GB) cards, post-unlock**. On the 10GB-variant, it doesn't fit even with all 4 cards and the full unlock. If you know which variant is arriving, that determines whether the DeepSeek row below is a real plan or needs a smaller quant/shorter context.

**Second finding**: both Qwen models don't strictly *need* the unlock to fit
(pre-unlock capacity is marginally sufficient), and once unlocked they don't
need multi-GPU pooling at all — a single unlocked card holds either one with
huge headroom. The unlock's value for these two models is mostly about
*margin* (comfortable full-context headroom, ability to run several
instances concurrently across your 4 cards) rather than *feasibility*.

---

## 5. Tensor parallelism vs. pipeline parallelism — decision and why

**Decision: pipeline (layer-split) parallelism, never tensor parallelism, for
any scenario on this hardware that needs multiple cards.** This applies to
DeepSeek-V4-Flash-0731 in all fitting scenarios, and to both Qwen models only
in the pre-unlock (forced multi-card) scenario.

**Reasoning, in order of weight:**

1. **Direct empirical evidence (§3) trumps theory here**: the community
   already measured tensor-parallel vs. pipeline-parallel on this exact
   crippled interconnect (Qwen2.5-72B-AWQ, TP2) and found TP **2.3–2.8x worse
   at prefill** for only a 23% decode gain — and the community's own written
   guidance is unambiguous: "pipeline, never tensor." I'm not second-guessing
   that; it's a directly relevant, already-run test on the same hardware
   class.

2. **First-principles confirmation of why**: tensor parallelism requires an
   all-reduce (or at minimum an all-gather) synchronization **at every
   layer** — for Qwen3.6-27B's 64 layers or DeepSeek-V4-Flash's 43 layers,
   that's dozens of blocking cross-GPU synchronization events per token.
   Pipeline parallelism only needs to hand off the hidden-state activation
   vector **at pipeline-stage boundaries** — 3 hops total for a 4-way split,
   regardless of how many layers are inside each stage. That's roughly a
   15–20x reduction in cross-GPU synchronization events for TP vs PP on
   these layer counts, and each one of those events has to survive the
   0.85–1.7 GB/s, no-P2P, host-bounced PCIe link (§1) — every GPU-to-GPU
   transfer round-trips through system RAM because there's no confirmed P2P
   path, adding real latency on top of the raw bandwidth constraint.

3. **This reverses what worked on your existing rig.** Your RTX Pro 4000 dual-
   GPU setup benefits from tensor-split (`-sm tensor`, +38.6% decode, memory
   in [[project_dual_card_tp2_powersweep]]) — but that pair has a real PCIe
   link with working P2P behind it. The CMP 170HX cluster has neither. Don't
   carry the "-sm tensor wins" assumption over to this hardware; it's tuned
   for a fundamentally different interconnect.

4. **MoE models get a second, independent benefit from avoiding TP**: the
   wiki notes MoE architectures reduce cross-device activation traffic
   relative to dense models under constrained links — relevant if you ever
   consider expert-parallel or tensor-parallel MoE serving strategies (vLLM
   supports these), but not something that changes the pipeline-parallel
   recommendation for llama.cpp/ik_llama, which pipeline-splits by layer
   regardless of dense/MoE.

**Practical effect on the token-speed numbers below**: for the DeepSeek-
V4-Flash-0731 multi-card scenario, decode speed with pipeline-parallel
llama.cpp-family engines is **not** a multiple of single-card speed — single-
stream (batch=1) decode is inherently sequential across pipeline stages (GPU
N+1 can't start layer N+1 until GPU N finishes layer N for that same token),
so more cards mainly buys you *capacity to fit the model at all*, plus a
small amount of *speed* if it lets the whole thing avoid CPU/RAM overflow —
not near-linear multi-GPU scaling the way data-parallel serving would.

---

## 6. Token speed estimates

Legend: **[measured-analog]** = derived from a real number in §3 for a close
but non-identical config. **[derived]** = my own calculation from §1/§2 specs,
no direct real-world analog available. **[extrapolated]** = scaled from a
real number for a substantially different model/scale.

### 6.1 Qwen3.6-27B UD-Q4_K_XL, MTP, full context (262,144 native)

| Scenario | Decode estimate | Confidence | Basis |
|---|---|---|---|
| Pre-unlock, 4× 8GB-var (32GB pool, marginal fit, pipeline-split forced) | **~12–20 tok/s** | Low-medium **[derived]** | Bandwidth is unchanged by the unlock (§1b), so if the marginal pre-unlock fit holds, per-token weight-read cost is the same as post-unlock. But: pipeline-split across 4 cards adds host-bounce latency at 3 stage boundaries per token (small but real), the marginal VRAM headroom likely forces smaller compute/KV buffers than the tuned reference config used to get 55–59 t/s, and pre-unlock compute-path throttling (§1) may degrade the attention/MTP-verification compute portion of each token even though raw weight-streaming bandwidth doesn't change. I discount the 55–59 t/s single-card post-unlock anchor by roughly 60–70% to account for these compounding penalties. |
| Pre-unlock, 4× 10GB-var (40GB pool, comfortable fit) | **~18–28 tok/s** | Low-medium **[derived]** | Same reasoning as above but with more headroom (comfortable rather than marginal fit), so less buffer-starvation penalty. Still pipeline-split across 4 cards with the same PCIe-hop and pre-unlock compute caveats. |
| **Post-unlock, single card (either variant, no multi-GPU needed)** | **~50–60 tok/s** | **High [measured-analog]** | Directly cited: "Qwen 27B: +200 offset yields 55→59 t/s" with MTP on a single 64GB card, llama.cpp/ik_llama, quant class matching your q4-class choice. This is your best-supported number in the whole document. |

### 6.2 Qwen3.6-35B-A3B UD-Q4_K_XL, MTP, full context (262,144 native)

| Scenario | Decode estimate | Confidence | Basis |
|---|---|---|---|
| Pre-unlock, 4× 8GB-var (32GB pool, marginal fit) | **~35–55 tok/s** | Low-medium **[derived]** | Same discounting logic as §6.1 applied to the 130–144 t/s single-card anchor. MoE models are less exposed to the per-layer sync problem than dense models under TP, but this is pipeline-split, not TP, so that advantage doesn't directly apply — the discount here is mainly about buffer headroom and pre-unlock compute-path uncertainty. |
| Pre-unlock, 4× 10GB-var (40GB pool, comfortable fit) | **~50–75 tok/s** | Low-medium **[derived]** | More headroom than the 8GB-variant pre-unlock case. |
| **Post-unlock, single card (either variant, no multi-GPU needed)** | **~120–145 tok/s** | **High [measured-analog]** | Directly cited: "Qwen 35B: +200 offset yields 130→144 t/s" with MTP, single 64GB card. Cross-checked against vLLM's 113 t/s AWQ figure on the same card class (different engine/quant, same ballpark) — gives confidence this range is real, not a fluke. |

### 6.3 DeepSeek-V4-Flash-0731 UD-Q8_K_XL, DFlash, full context (1,048,576 native)

| Scenario | Decode estimate | Confidence | Basis |
|---|---|---|---|
| Pre-unlock (either variant) | **Not viable — do not attempt** | High | §4: needs ~217GB, pre-unlock pool is 32–40GB. Would require CPU/RAM offload exactly like your existing single-RTX-Pro-4000 rig, except the CMP cards' PCIe link (0.85 GB/s Gen1 x4) is far worse than a normal PCIe x16 slot for host↔GPU weight streaming, so this would very likely be *slower* than what you already get today ([[project_deepseek_v4_flash_kv_precision]] — your existing rig gets ~9.4 tok/s decode with heavy RAM offload on a properly-connected card). Don't burn setup time on this combination pre-unlock. |
| Post-unlock, 4× 8GB-var (256GB pool, ~217GB used, fits with ~39GB headroom, pipeline-split mandatory) | **~4–10 tok/s** | **Low [extrapolated]** | Scaled from the GLM-5.2 8-card pipeline-parallel analog (30.2 t/s vLLM / 17.33 t/s llama.cpp on 8×64GB for a 744B-total MoE). DeepSeek-V4-Flash-0731 has roughly half the pipeline-stage count opportunity (4 cards vs. 8) and a different active-param profile (~13B active vs. GLM-5.2's unknown-to-me active count), so this is a genuine extrapolation across both card-count and model-architecture, not a clean scaling. DFlash speculative decoding could meaningfully lift this *if* acceptance rates are good (DeepSeek's own materials claim DFlash beats plain MTP specifically by avoiding the accuracy falloff at later draft positions) — but I have no real acceptance-rate data for DFlash on this hardware to quantify that lift, so I'm presenting the range without a speculative-decoding multiplier applied. Treat this row as the least trustworthy number in the document. |
| Post-unlock, 4× 10GB-var (160GB pool) | **Not viable** | High | §4: 217GB needed exceeds the 160GB pool even with the full unlock. Would need a smaller quant (Q4_K_XL) or a heavily reduced context length to fit at all — not the config you specified. |

---

## 7. Practical notes that affect all three estimates

- **Prefill (time-to-first-token) is genuinely compute-bound**, unlike
  decode, so it *does* benefit substantially from the compute unlock — expect
  pre-unlock prefill to be dramatically worse than the decode-speed ratio
  alone would suggest, especially at the 262K–1M "full context" prompt sizes
  you're targeting. The one directly-measured single-card prefill datapoint
  (Qwen3.6-27B, pp2048: 328.81→449.41 t/s Gen1x4→Gen2x4) is *only* the
  Gen1→Gen2 PCIe effect, not the FP32/tensor compute-unlock effect — the
  pre-unlock-vs-post-unlock prefill gap from compute throttling alone could
  be much larger than that number implies. I don't have a clean pre/post
  prefill comparison for compute-throttled state and didn't want to guess a
  number without evidence.
- Community guidance: **"q4-class is the practical sweet spot on a 64GB
  card"** — directly validates your own choice of Q4_K_XL for both Qwen
  models.
- Community guidance: **MTP recommended ON for single-stream llama.cpp**
  (your use case) but **OFF for vLLM serving the 35B MoE** — if you ever
  switch engines for these models, don't carry the MTP-on assumption to
  vLLM without re-checking.
- Budget **~1GB of nominal capacity per card for driver/context overhead**
  before counting usable VRAM; real observed llama.cpp residency on 64GB
  cards clusters at 53–60GB/64GB, not the full 64.
- If using vLLM instead of llama.cpp/ik_llama for any of these: keep
  `gpu_memory_utilization` ≤ 0.90 — 0.95 reportedly crashed a card in
  community testing.
- An untested lead from the community wiki worth trying once hardware
  arrives: `n_ubatch` in the 48–80 range reportedly gave a 3.04x gain on a
  different card model, unverified on CMP 170HX specifically.

---

## 8. Sources

- [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) — the tool itself, local clone in this project folder
- [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) — 55-page community wiki; primary source for §1, §3, §5, §7 (pages: `docs/unlock/compute-throttle.md`, `docs/operations/performance.md`, `docs/operations/llm-inference.md`, `docs/operations/tuning.md`)
- [niconiconi.neocities.org CMP 170HX teardown review](https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/) — independent hardware teardown, cross-checked against the wiki
- Model specs fetched directly from HF `config.json`: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- GGUF file sizes fetched directly from HF file listings: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF), [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
- DFlash architecture: [RedHatAI/DeepSeek-V4-Flash-speculator.dflash](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash), [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/pdf/2602.06036)
- Not included: GPU spec-aggregator sites (technical.city, cputronic, gpurankings, topcpu) whose numbers were internally inconsistent with the primary teardown/wiki sources and weren't used for any figure above.

---

## Addendum — 2026-08-04 (later same day): variant confirmed

You confirmed the incoming cards are the **8GB SK Hynix variant** (`10de:20c2`,
unlocks to 64GB). Per the no-overwrite rule at the top of this document, I'm
not touching any number above — this addendum just tells you which rows are
the live scenario:

- **§4 capacity table**: use the "8GB-var pre-unlock (32GB)" and "8GB-var
  post-unlock (256GB)" columns. The 10GB-variant columns are moot for your
  actual hardware.
- **§4's decisive finding stands and is now settled, not conditional**:
  DeepSeek-V4-Flash-0731 at Q8_K_XL + full context (~217GB) **fits** on your
  4-card pool post-unlock (256GB, ~39GB headroom) — this is the scenario that
  will actually happen, not a hypothetical.
- **§6.1–6.3**: use the "4× 8GB-var" pre-unlock rows and the post-unlock rows
  (post-unlock numbers were already variant-agnostic for the two Qwen models
  since they fit on a single card either way).
- I checked whether "SK Hynix" specifically (as opposed to another HBM2e
  vendor) changes anything: it doesn't appear to — SK Hynix is the standard/
  only memory vendor reported for CMP 170HX cards in general, not a distinct
  sub-variant with different unlock behavior or stability characteristics.
  I found no vendor-specific stability or performance data to fold into the
  estimates above.
- One real, unresolved caution from the research (not new, but worth
  repeating now that this is your confirmed configuration): the community
  wiki source itself states the 64GB unlock mode "lacks long-term stability
  testing" across large samples/batches — worth budgeting time for a
  stability burn-in (e.g. `verify.sh` + a sustained memory-heavy workload,
  not just a quick `nvidia-smi` check) before trusting it for long unattended
  runs, per the thermal-runaway risk already noted in `TUTORIAL.md`.

---

## Addendum 2 — 2026-08-04 (later same day): correction to §6.3's DeepSeek-V4-Flash-0731 post-unlock estimate

**You caught a real methodological error.** The original §6.3 post-unlock row
(~4-10 tok/s) extrapolated from the GLM-5.2 8-card figures by loosely scaling
on total model size and card count. That was wrong: decode speed for a
sparse MoE model is governed by *active* bytes read per token, not total
resident weight size — a principle this document already establishes
correctly in §1b/§4, then fails to apply here. I hadn't actually looked up
GLM-5.2's active parameter count when I wrote the original estimate.

**Corrected comparison, using real specs (GLM-5.2 config.json fetched
2026-08-04):**

| | Active params/token | Quant in the cited test | Est. active bytes/token |
|---|---|---|---|
| GLM-5.2 | **40B** (confirmed: `num_experts_per_tok=8` × `moe_intermediate_size=2048`, `hidden_size=6144`, 78 layers) | W4A16 (vLLM) / UD-IQ2_M (llama.cpp) | ~22GB (vLLM) / ~13GB (llama.cpp), estimated from quant file size ÷ total params applied to the active subset |
| DeepSeek-V4-Flash-0731 | ~13B | Q8_K_XL | ~7-8GB, same estimation method |

GLM-5.2 activates roughly **3x more parameters per token** than DeepSeek-
V4-Flash-0731. Applying bandwidth ÷ active-bytes-per-token as a roofline, then
the real efficiency ratios the two GLM-5.2 tests actually achieved relative
to their own roofline (30.2 measured / ~61 roofline ≈ 49% for the vLLM/W4A16
run; 17.33 measured / ~105 roofline ≈ 16.5% for the llama.cpp/UD-IQ2_M run),
DeepSeek-V4-Flash's ~170-190 tok/s roofline scales to a **revised estimate of
roughly 25-70 tok/s** — well above the original 4-10 tok/s figure.

Two secondary factors both push in the same direction (faster, not slower):
Q8_K_XL is a simple block format, cheap to dequantize, unlike IQ2_M's
importance-matrix codebook lookups which likely explain GLM-5.2's much lower
16.5% llama.cpp efficiency; and your 4-card pipeline has only 3 inter-GPU
hops vs. GLM-5.2's 7 on 8 cards, i.e. less synchronization overhead per
token, not more.

**This revised range (~25-70 tok/s) supersedes the ~4-10 tok/s figure in the
§6.3 table for the post-unlock, 4× 8GB-var scenario.** Confidence is still
**low** — the underlying GLM-5.2 citation is itself a single unreproduced
test session (see the wiki author's own hedge, quoted in conversation), the
active-bytes-per-token figures for both models are estimated from blended
quant-file-size ratios rather than measured per-layer, and DFlash's actual
acceptance rate on this hardware is completely unknown, so no speculative-
decoding multiplier is included in this range either. Treat 25-70 tok/s as a
better-reasoned estimate, not a confident one.

---

## Addendum 3 — 2026-08-04 (later same day): findings from reading the full local wiki clone

You asked me to search the locally cloned `cmp170hx/` wiki for more detail.
Reading the full pages directly (rather than the summarized fetches used
while building the original document) surfaced one finding that changes
feasibility, not just speed, plus several numbers that refine existing
estimates. None of this alters the original tables — appended per the
no-overwrite rule.

### A. Engine/DFlash support gap for DeepSeek-V4-Flash-0731 (new, decisive)

This is not in the wiki — I found it by following a thread the wiki's GLM-5.2
section opened up (GLM-5.2 uses DeepSeek's own sparse attention family, DSA/
CSA) and checking DeepSeek-V4-Flash-0731's own support status directly. The
wiki documents that **llama.cpp has no DSA support and silently falls back to
dense attention**, which is the documented cause of GLM-5.2's prefill
collapsing to ~120-160 tok/s on llama.cpp while vLLM's prefill *improves*
with context using the real sparse kernel (llama.cpp issue #24730).
DeepSeek-V4-Flash-0731 uses the same attention family — its config's
`index_head_dim`/`index_n_heads` fields (noted in §2 above without being
connected to this) are the DSA/lightning-indexer signature.

Checking current engine support directly: **ik_llama.cpp does not support
DeepSeek-V4-Flash's architecture at all** (open feature request, quoting the
requester: "V4-flash is only in VLLM?"). **Mainline llama.cpp** added
DeepSeek V4 support via PR #24162, explicitly marked work-in-progress.
**DFlash for GGUF specifically** exists only via a separate PR (#22105) and
a third-party integration project, with vLLM's DFlash implementation being
the mature one.

**Practical consequence**: "DeepSeek-V4-Flash-0731 UD-Q8_K_XL with DFlash" as
a single coherent GGUF + llama.cpp-family stack may not be a fully working
combination yet. You may end up on mainline llama.cpp without functioning
DFlash (falling back to plain decode), or need vLLM for a proper DSA kernel
and mature DFlash support — which does not consume your Q8_K_XL GGUF file at
all. **This is a feasibility risk independent of every token-speed number in
this document, and worth resolving before setup, not after.**

### B. A more precise, more directly comparable single-card Qwen3.6-27B table

The wiki documents a specific controlled benchmark of
`unsloth/Qwen3.6-27B-MTP-GGUF` (UD-Q4_K_XL and UD-Q8_K_XL) on a real unlocked
10GB→40GB card — closer to your actual quant naming than anything used in
the original §6.1 anchor. Selected figures (llama.cpp, Q4, `tg128`):
**33.10 tok/s without MTP → 46.24 tok/s (peak 56.67) with MTP**
(`--spec-type draft-mtp --spec-draft-n-max 2`). This is somewhat below the
"55→59 tok/s" figure §6.1 was anchored to. The wiki itself flags this
territory as unresolved: published Qwen-27B-class single-card figures span
**97 / 90 / 75 / 58.5 / 36.87 tok/s** across different reports, "never
reconciled" because quantisation, MTP state, context length and engine
version were never held constant between them. **Read §6.1's "~50-60 tok/s"
post-unlock estimate as sitting inside a wider, genuinely unresolved
50-97 tok/s band in the underlying community data, not as a tight number.**

### C. MTP is backend-inconsistent for the 35B-class MoE — worth knowing before you pick an engine

For a 35B-A3B-class MoE, MTP measured **+21% decode on llama.cpp** (108.4 →
131.3 tok/s) but **-23% decode on vLLM** (147 → 113 tok/s) despite a
reported 75% token-acceptance rate — the standing (unproven) explanation is
a CPU-side bottleneck specific to Gen1 x4 hosts. This directly supports
§6.2's engine choice (llama.cpp/ik_llama with MTP on) but is worth stating
explicitly: **if you ever move the 35B model to vLLM, re-test with MTP off
before assuming it still helps.**

### D. A 4-card GLM-5.2 result — a better hop-count match for your DeepSeek estimate than the 8-card one

Addendum 2 revised the DeepSeek-V4-Flash-0731 estimate using an **8-card**
GLM-5.2 analog. The wiki separately documents a **4-card** result closer to
your actual configuration: GLM-5.2-UD-IQ2_XXS across 4× 64GB cards via a
PCIe bifurcation board (every card Gen1 x4, layer split, no MTP): **~15
tok/s decode, 24.07 tok/s prefill**. Working through the same active-bytes/
roofline method as Addendum 2 (GLM-5.2's 40B active params at an estimated
~0.25 bytes/param for this quant tier ≈ 10GB/token, roofline ≈135 tok/s,
15 tok/s measured ≈ 11% efficiency) lands in the same neighborhood as
Addendum 2's conclusion — this 4-card, hop-count-matched data point
**corroborates the ~25-70 tok/s revised range rather than overturning it**,
since DeepSeek's simpler Q8_K_XL block format should still clear this
particular quant's low efficiency (IQ2_XXS's codebook lookups are expensive
to dequantize).

### E. Explicit quantitative threshold for when tensor parallelism would become worth trying

The wiki states a specific stated threshold rather than just a direction:
tensor parallelism is judged worth attempting only above **PCIe Gen2 x16 or
Gen3 x4**. cmpunlocker's software unlock delivers **Gen2 x4** — below that
threshold on both axes (right generation, wrong width; reaching x16 needs
the 24-capacitor physical mod, a separate axis from the generation unlock).
This sharpens §5's "pipeline, never tensor" recommendation from directional
guidance into a stated numeric line you are currently on the wrong side of.

### F. P2P absence is now confirmed at high confidence, not just "not supported"

`torch.cuda.can_device_access_peer()` returned `False` for all 56 pairs
tested on an 8-card rig, including within a PCIe topology group; an
`nvidia-smi nvlink` capture reports no NVLink support; a source-tree grep
across shipping `master` and all 12 unreleased branches found zero P2P
enablement code. A third-party out-of-tree patch (`aikitoria/open-gpu-
kernel-modules`) builds and loads on top of cmpunlocker but is explicitly
unvalidated on GA100, and one unverified single-rig report claims it enables
raw peer memory copies (6.25 GB/s) while every collective library (NCCL,
vLLM's custom all-reduce) still hangs — meaning even if that patch works,
it would not help tensor-parallel inference frameworks, only hand-rolled
code that doesn't rely on collectives. **§5's "no confirmed P2P" framing
holds at high confidence; there is no practical path to changing it for
mainstream inference engines today.**

### G. A methodology caveat worth carrying forward

The wiki itself documents how it was built: a synthesis of a ~26,000-message
Discord archive, extracted and adjudicated by LLM agents, cross-checked
against the actual driver source code wherever a claim was code-settleable.
Firmware/register claims are consequently close to verifiable — but the wiki
is explicit that **performance, thermal and PCIe-bandwidth numbers are not
code-settleable and remain "a small pile of field reports,"** frequently a
single tester, single card, single session, never independently reproduced.
This matches and reinforces the confidence caveats already stated throughout
this document (§1a, §3, §6) — it isn't new information so much as
independent confirmation that the conservative confidence labels used
throughout this document are the right level of skepticism, not excessive.

---

## Addendum 4 — 2026-08-04 (later same day): DFlash speedup applied to the DeepSeek-V4-Flash-0731 estimate

You asked me to factor DFlash's own speedup into the §6.3 estimate. Addendum
2's revised 25-70 tok/s range was explicitly the *base* decode rate with **no**
speculative-decoding multiplier applied — this addendum adds one, using
DFlash's own reported numbers rather than a guess.

**What DFlash actually reports** (DeepSeek's own DFlash paper and community
coverage, not CMP 170HX-specific): average accepted length ≈ **2.14 tokens
per verification step** in general use (non-thinking, temperature 0) on
DeepSeek-V4, rising to **~6 accepted tokens** on code/math-heavy content
specifically. For context, DeepSeek's newer DSpark method beats DFlash by
16-18% on accepted length — worth knowing if DSpark becomes available on
your stack before DFlash does, but you asked about DFlash specifically, so
that's what this estimate uses.

**Translating accepted length into a speedup multiplier**: accepted length
is an upper bound on speedup, not identical to it — each verification pass
costs slightly more than one plain forward pass (processing multiple draft
positions), and the small 5-layer draft model itself has to run every step.
Typical real-world speculative-decoding speedups land somewhat below the raw
accepted-length number. For DFlash's ~2.14 accepted length, I'd estimate a
realistic decode multiplier of roughly **1.4-2.0x** for general use, and
plausibly higher (~3-5x, much lower confidence) on code/math-heavy sessions
given the ~6 accepted-tokens figure — I'm not extending that domain-specific
number with any real confidence, since it wasn't measured on this hardware
or this pipeline-parallel configuration.

**A specific, documented reason this could go the other way.** Addendum 3
already noted that native MTP on vLLM *regressed* decode speed by 23% on a
35B-class MoE, with the standing (unproven) explanation being a CPU-side
bottleneck specific to constrained PCIe-x4 hosts. DFlash's verification step
is architecturally different from MTP but shares the same basic shape — an
extra round of draft-model computation and verification layered onto the
base decode loop — on the same class of constrained host. **There is a real,
documented precedent on this exact hardware family for a speculative-
decoding method making things worse, not better, and no CMP 170HX-specific
DFlash measurement exists to rule that out here.** This is layered on top of
Addendum 3 Finding A: none of this matters until a working DFlash+GGUF+
llama.cpp-family stack actually exists to run in the first place.

**Combined estimate**: applying a 1.4-2.0x multiplier to Addendum 2's 25-70
tok/s base range gives a DFlash-adjusted estimate of roughly **35-140 tok/s**
for general use, with a plausible but low-confidence upside toward
code/math-heavy sessions, and a real, precedented chance the multiplier is
closer to 1x or even slightly below it if DFlash hits the same PCIe-x4
CPU-bottleneck pattern documented for vLLM's MTP. This is now the most
uncertain number in the entire document — treat the 35-140 tok/s band as
"plausible given the public DFlash numbers," not as a confident prediction,
until you have your own measurement on this hardware.

---

## Addendum 5 — 2026-08-04 (later same day): correction using the model-specific DFlash validation numbers

You asked me to research DFlash performance with DeepSeek-V4-Flash specifically
and confirm Addendum 4's assumptions. The 2.14 accepted-length figure Addendum 4
used came from general "DeepSeek-V4" search coverage — not necessarily the
Flash variant. I found a better source: the DFlash speculator's own Hugging
Face model card (`RedHatAI/DeepSeek-V4-Flash-speculator.dflash`) is a
validation run built and evaluated specifically **against your exact base
model**, `deepseek-ai/DeepSeek-V4-Flash`, not a generic DeepSeek-V4 figure.
This should be treated as the more authoritative number for this document.

**What that validation actually reports**: per-position greedy acceptance
rates declining from **78.8% at draft position 1 to 19.3% at position 7**,
an **average accepted length of ≈3.9 tokens** (notably higher than Addendum
4's 2.14 figure), and **41.6% full-sequence acceptance** (all 7 draft
positions accepted). Validated on **Nvidia H200** hardware — not CMP 170HX,
and not any PCIe-constrained multi-GPU setup. Critically, **this model card
reports acceptance rates only — no end-to-end throughput, latency or
speedup-multiplier number is given anywhere on it.** Every tok/s figure
attached to DFlash in this document, including this one, is still my own
derivation from an accepted-length number, not a directly measured speed.

**Revised multiplier**: a ≈3.9 average accepted length is meaningfully
higher than the 2.14 figure Addendum 4 discounted from, and the fixed
per-step overhead (small 5-layer draft-model forward pass, verification
kernel launch) amortizes better over a longer accepted run — so the
realized-to-raw ratio should if anything be somewhat *better* than Addendum
4's estimate, not worse. Revised realistic decode multiplier: roughly
**2.0-3.0x** for general use (up from 1.4-2.0x), still well short of the raw
3.9x accepted-length ceiling.

**Revised combined estimate**: applying 2.0-3.0x to Addendum 2's 25-70 tok/s
base range gives **roughly 50-210 tok/s**. I don't trust the top of that
band — it would exceed even the non-speculative roofline ceiling estimated
earlier in this document (~170-190 tok/s), and I haven't modeled every
per-verification-step overhead (kernel launch cost, and specifically whether
the pipeline-parallel activation handoff at each of the 3 inter-GPU hops has
to repeat once per *verification step* rather than once per generated token,
which would erode the speedup this whole calculation assumes). Treat
**~50-120 tok/s as the more defensible center of the band**, 50-210 tok/s as
the outer bound, and everything above that as unsupported.

**Every caveat from Addendum 4 still applies, unchanged**: the vLLM-MTP
regression precedent on PCIe-x4 hosts is a real, documented reason this
multiplier could land near 1x instead; the RedHatAI card's own "Serving
Requirements" section states it needs a vLLM build with DFlash support,
which reinforces (doesn't resolve) Addendum 3 Finding A — the GGUF/llama.cpp-
family path for DFlash is the less mature one, and this validation wasn't
even run on that path. Nothing here should be read as more certain than
Addendum 4 already was; it's a better-sourced number layered onto the same
amount of real uncertainty.

---

## Addendum 6 — 2026-08-04 (later same day): GitHub/Reddit search for ±5 tok/s precision — negative result

You asked me to search GitHub and Reddit specifically to narrow the estimates
to a ±5 tok/s range. I searched: the cmpunlocker repo's own GitHub issues
(three open issues, all feature requests — NVLink, ECC, PCIe Gen3/4 — none
contain benchmark data); a long-running llama.cpp "Performance on Nvidia
CUDA" megathread, checked across its full date range for every CMP 170HX
mention; targeted GitHub searches for Qwen3.6-27B/35B-A3B and DeepSeek-V4-
Flash combined with "170HX"; and multiple Reddit-scoped search variants,
including the tool's own "unlobotomize" tagline as a search term.

**Result: nothing new.** Reddit returned no relevant indexed content at all
for this specific hardware+model combination. The only additional GitHub
data found was two more CMP 170HX posts in the llama.cpp megathread, both
**Llama-2 7B Q4_0** — not remotely close to your target model sizes, and not
usable for narrowing anything here. Everything else either duplicated
material already mined into Addenda 1-5, or was unrelated noise (spec-
aggregator sites, marketplace listings).

**Why ±5 tok/s isn't achievable from public data right now, stated plainly**:
the wiki's own account of the *single best-documented* real analog —
Qwen-27B-class decode on one unlocked card — is an unreconciled spread of
roughly 37 to 97 tok/s across different testers who never held quant, MTP
state, or context length constant. That's the tightest real data point
available, and it's already more than 10x wider than the precision you
asked for. Nobody has published a controlled benchmark of your exact three
model+quant+context combinations on this card at all. Narrowing to ±5 tok/s
without new data would mean presenting a fabricated number as if it were
supported evidence, which I'm not willing to do.

**What would actually close this gap**: a real measurement on your own
hardware once it arrives, using `verify.sh` plus `llama-bench`/equivalent
with your exact quant and context settings — not further internet research.
The existing ranges in this document (Addenda 1-5) already represent the
tightest defensible bounds the current public evidence supports.

---

## Addendum 7 — 2026-08-04 (later same day): committed ~10-tok/s-wide estimates

You asked for a single decision-usable ~10 tok/s band per model instead of
the full uncertainty ranges above. These are my best-judgment central
estimates, not a re-derivation of new evidence — picked from within the
ranges already established in Addenda 1-6, for the single most likely
real-world scenario (post-unlock, full context, MTP/DFlash on):

| Model | Scenario | Estimate |
|---|---|---|
| Qwen3.6-27B UD-Q4_K_XL, MTP | Post-unlock, single card, full 262K ctx | **38-48 tok/s** |
| Qwen3.6-35B-A3B UD-Q4_K_XL, MTP | Post-unlock, single card, full 262K ctx | **100-110 tok/s** |
| DeepSeek-V4-Flash-0731 UD-Q8_K_XL, DFlash | Post-unlock, 4-card pipeline, full 1M ctx | **65-75 tok/s** |

**Method**: for the two Qwen models, took the real single-card llama.cpp+MTP
clusters already cited (Qwen-27B: 46-59 tok/s at short context across
several independent testers; Qwen-35B-A3B: 108-144 tok/s) and discounted
each ~15-20% for full-context KV-cache overhead, since the cited benchmarks
ran short `tg128`-style tests, not full 262K-token context. For DeepSeek,
took the lower-middle of Addendum 5's 50-120 tok/s central band rather than
its midpoint, weighted down for the documented risk that speculative
decoding regresses on this PCIe-constrained hardware class (the real vLLM-
MTP precedent).

**The DeepSeek row is meaningfully softer than the other two**, even in this
tighter format — it rests on an active-parameter/bandwidth calculation
corroborated only loosely by a different model on a different quant, times
an unmeasured DFlash multiplier. The two Qwen rows have several independent
real testers clustering in the same neighborhood; the DeepSeek row does not.
Treat all three as **committed estimates for planning purposes, not
confidence intervals** — the fuller uncertainty picture in Addenda 1-6
above still stands and is the more honest account of what's actually known.

---

## Addendum 8 — 2026-08-04 (later same day): a different scenario — single CMP 170HX (64GB) replacing this machine's RTX PRO 4000(s), hybrid CPU/GPU

This is a distinct question from everything above: not the 4-card, full-GPU-
resident scenario the rest of this document estimates, but swapping this
exact machine's GPU(s) for a single unlocked CMP 170HX and running DeepSeek-
V4-Flash-0731 UD-Q8_K_XL hybrid CPU+GPU, the same way it already runs today.

**Correction to the premise**: this machine's own scripts
(`/home/user/ai-models/start-deepseek-v4-flash.sh`,
`/home/user/ai-models/LLM-BENCHMARK-RESULTS.md`) show **ik_llama.cpp cannot
load this model at all** — both local ik_llama.cpp copies only recognize
the `deepseek2` architecture; DeepSeek-V4-Flash uses the newer `deepseek4`
arch (MLA + DSA sparse-attention indexer). Production runs on **llama.cpp
mainline** instead, the only engine on this box whose arch table includes
`deepseek4`. This estimate is for mainline llama.cpp, not ik_llama.

**Already-measured baseline on this exact machine** (9900X, 256GB DDR5 @
3600MT/s configured, same model/quant, 720,896 ctx):

| Config | VRAM | Host RAM resident (of 162GB) | Decode |
|---|---|---|---|
| 1× RTX PRO 4000 (24GB) | 24GB | ~140GB | 9.37-9.46 tok/s |
| 2× RTX PRO 4000 (48GB combined) | 48GB | ~119GB | 9.26 tok/s avg |

Doubling VRAM (24→48GB) produced **no measurable decode speedup** —
GPU utilization during decode averaged 17-27%, far from saturated. The
documented bottleneck is DDR5 bandwidth feeding the RAM-resident majority of
the model, independent of GPU capacity, since even 48GB is nowhere near the
162GB model size. The same DDR5-bound pattern is independently confirmed on
this box for DeepSeek-R1, Kimi-K2.6, and GLM-5.2.

**Estimate for 1× CMP 170HX (64GB) replacing both RTX PRO 4000s: ≈9.0-9.5
tok/s** — essentially unchanged from the existing measurement, not a
meaningful improvement. 64GB is only +16GB over the already-tested 48GB
config, which already showed a much larger jump (+24GB, doubling from 24GB)
move nothing. The fundamental constraint doesn't shift when the GPU holding
the smaller resident fraction changes.

**Two real unknowns not ruled out by the existing data, given honestly**:
- CMP 170HX is Ampere/SM80 (compute cap 8.0) vs. the RTX PRO 4000's Blackwell
  (compute cap 12.0). The fused `GGML_OP_LIGHTNING_INDEXER` CUDA kernel that
  fixed this model's original OOM problem has never been tested on Ampere on
  this box — an architecture-specific performance difference there isn't
  ruled out by anything measured so far.
- CMP 170HX's much worse PCIe (Gen1/Gen2 x4) shouldn't hurt **decode**,
  since llama.cpp's hybrid offload computes CPU-resident tensors on the CPU
  directly from RAM rather than streaming them through the GPU each token —
  but it could plausibly hurt **model load time** and **prefill**, neither
  of which this machine's existing data measures.

---

## Addendum 9 — 2026-08-04 (later same day): dropping to 2×64GB DDR5 @ 5600MT/s alongside the CMP 170HX

Follow-up to Addendum 8: instead of keeping all 4 DIMMs (256GB @ 3600MT/s
configured), drop to 2×64GB (128GB @ their full 5600MT/s rated speed — a
well-known AM5 platform behavior where 1 DIMM/channel reaches rated XMP/EXPO
speed but 2 DIMMs/channel forces a derate, which is almost certainly why
this machine currently runs 3600MT/s instead of the DIMMs' own 5600MT/s
rating). This directly targets the actual bottleneck Addendum 8 established
(DDR5 bandwidth), unlike the GPU swap alone.

**Bandwidth math**: dual-channel DDR5 bandwidth = MT/s × 8 bytes × 2
channels. Current: 3600 × 8 × 2 = **57.6 GB/s** theoretical. Proposed:
5600 × 8 × 2 = **89.6 GB/s** theoretical — **+55.6%**.

**Capacity check**: 128GB RAM + 64GB VRAM (CMP 170HX) = 192GB combined pool.
DeepSeek-V4-Flash UD-Q8_K_XL needs ~162GB weights + ~5-8GB post-fix compute-
buffer overhead (per the `GGML_OP_LIGHTNING_INDEXER` fix already documented
in this machine's own scripts) ≈ 168-170GB total. **Still fits**, ~22-24GB
headroom — tighter than the current 256GB-RAM pool, but workable for this
specific model.

**Estimate: ≈12-14 tok/s**, up from the established 9.37-9.46 tok/s
baseline. Applied a realistic ~1.35-1.5x multiplier rather than the full
theoretical 1.556x bandwidth ratio, since real-world DDR5 bandwidth scaling
for streaming MoE-expert reads rarely reaches 100% of theoretical peak, and
Addendum 8's open unknowns (untested-on-Ampere fused CUDA kernel) still
apply. This is a genuine prediction, not a measurement — nothing on this
box has tested 2-DIMM/5600MT/s operation yet.

**Separate practical tradeoff, not part of the tok/s estimate**: halving
system RAM to 128GB could push other hybrid CPU/GPU models off this box —
Kimi-K2.6 (~1T params) and GLM-5.2 (751B), both already running in CPU+GPU
hybrid mode on the current 256GB per this machine's own benchmark log, may
not fit in 128GB depending on quant. Worth checking before committing to
the 2-DIMM change if those are still in regular use.

---

## Addendum 10 — 2026-08-04 (later same day): image/video generation — LTX-2.3, Wan 2.2, HunyuanVideo 1.5 at highest quality fitting 64GB

A different topic from everything above (LLM decode) — this document's title
isn't LLM-specific, so recording it here rather than a separate file.

**Key architectural fact, decisive for LTX specifically**: NVFP4 (what this
machine's Blackwell workstation runs for LTX-2.3) requires dedicated FP4
tensor cores that exist **only on Blackwell**. The CMP 170HX is Ampere
(GA100) — same generation as this machine's RTX 3090s. A real-world Ampere
test found NVFP4 emulation buys only ~7-10% over plain BF16 there, not worth
the complexity. This machine's own `VIDEO-PRODUCTION-WORKFLOW.md` already
makes the identical argument for why the 3090s run Wan 2.2 in full BF16
rather than a quantized format — Ampere has no FP8 tensor cores either.
**Same reasoning applies directly to the 170HX**: BF16 is the natively-
supported highest-quality tier for LTX and Wan 2.2 here, not NVFP4/FP8.

**Highest quality that fits 64GB** (VRAM figures from this machine's own
`AI-SETUP-GUIDE.md` / `VIDEO-PRODUCTION-WORKFLOW.md`):

| Model | Format | VRAM | Headroom on 64GB |
|---|---|---|---|
| Wan 2.2 14B | BF16 | ~28GB | ~36GB free |
| HunyuanVideo 1.5 | FP16 (T2V) | ~13GB | ~51GB free |
| LTX-2.3 19B | BF16 (NVFP4 unavailable on Ampere) | not measured on Ampere anywhere | substantial, unmeasured |

**Speed estimates**, using the real 170HX-vs-RTX-3090 diffusion ratios
already established in this document's non-LLM data (SDXL 1.60x, LTX-Video
1.82x, Wan2.1 1.81x faster than a 3090 — same-precision, compute-bound,
VRAM-resident comparisons, the regime diffusion sits in generally):

- **Wan 2.2 BF16**: this machine's `AI-SETUP-GUIDE.md` implies a dual-3090
  Wan 2.2 baseline of roughly ~330s for a 65-frame/768×512 clip (backed out
  from a stated 18.4x-faster-per-frame comparison against another model
  elsewhere in that file). Applying the 1.81x same-family ratio:
  **≈180-200 seconds** on a single 170HX. Moderate confidence — real
  underlying data, but two translation steps (indirect baseline derivation,
  cross-model ratio application).
- **LTX-2.3 BF16**: one genuine Ampere-native LTX data point found via
  search — an RTX 3090 doing a 10s/4K clip at 30-36 steps in ~20-25 minutes.
  Applying the same 1.6-1.82x ratio: **≈11-16 minutes** for that same
  10s/4K config on a 170HX. Deliberately not scaled down to the shorter/
  lower-res draft setting this machine actually uses day to day — the
  pixel×frame scaling math attempted for that didn't converge to a number
  worth trusting, so it's omitted rather than forced. Also unclear whether
  the 3090 test used LTX-2.3 specifically or an earlier LTX-Video version —
  weakest-grounded of the three estimates.
- **HunyuanVideo 1.5 FP16**: no Ampere-class speed reference found anywhere,
  in this machine's own files or via search. VRAM fit is solid (~13GB, very
  comfortable); decode/generation speed is genuinely unknown rather than
  estimated — no number given rather than a fabricated one.
