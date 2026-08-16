# References — CMP 170HX / cmpunlocker

Links gathered while researching and cross-checking `TUTORIAL.md` (2026-08-04).

## Primary tool

- [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) — main repo, cloned into this folder. README, install.sh, remove.sh, verify.sh, docs/.
- [Discord community](https://discord.gg/CdHSakKSFv) — support server linked from the README; `#issue-support` channel for install problems.

## Community documentation

- [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) — 55-page technical wiki: silicon/firmware details, unlock mechanism, procedures, risks, frontier/unsolved problems. Deeper than the main repo's own docs; used to cross-check and fill gaps in the tutorial. **Cloned locally into this project folder as `cmp170hx/` (2026-08-04, commit `879454d`) for offline search.** Specific pages pulled so far:
  - `docs/start/risks.md` — bricking/thermal/warranty risk breakdown
  - `docs/start/quick-start.md` — condensed install walkthrough
  - `docs/procedures/install.md` — full prerequisites and install steps
  - `docs/procedures/troubleshooting.md` — symptom/cause/fix tables, Xid codes
  - `docs/procedures/multi-gpu.md` — mixed-GPU auto-detect bug details
  - `mkdocs.yml` — site nav, used to find correct page paths after a 404
  - `docs/unlock/compute-throttle.md` — measured SM count / FP32-FP64/tensor TFLOPS pre vs. post unlock
  - `docs/operations/performance.md` — memory bandwidth pre/post unlock, PCIe bandwidth by generation/width
  - `docs/operations/llm-inference.md` — real vLLM/llama.cpp/ik_llama tokens-per-second figures on unlocked cards, multi-GPU (8-card) GLM-5.2 pipeline-parallel numbers
  - `docs/operations/tuning.md` — "pipeline, never tensor" parallelism guidance, MTP on/off recommendations by engine, VRAM/quant sizing tips
  - `docs/appendix/methodology.md` — how the wiki was built (Discord-archive synthesis by LLM agents, cross-checked against source where code-settleable); read this before trusting any performance/thermal/PCIe number on the site
  - `docs/frontier/p2p.md` — full P2P absence evidence, the unvalidated third-party `aikitoria` patch, BAR1 64MB constraint, IOMMU/ACS interaction
- [amoghmunikote/170th-street](https://github.com/amoghmunikote/170th-street) — companion project resource: hardware teardown, PCIe capacitor mod, watercooling notes, AI/ML workload notes (LLM inference, FP16, FMA workaround). Not yet cross-referenced in depth.
- [niconiconi.neocities.org — CMP 170HX teardown review](https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/) — independent hardware teardown (SM/CUDA core counts, HBM2e bus width, PCIe lane wiring, throttled FP32/FP64 figures); used to cross-check the community wiki's hardware claims for `cmp-170x-performance-benchmarks.md`.

## Forks / alternative implementations

- [abobasixseven/unlock-cmp-170hx](https://github.com/abobasixseven/unlock-cmp-170hx) — alternative fork, not evaluated in detail. Worth a look if the main repo stalls or is taken down (note: a DMCA takedown against at least one fork is mentioned in the wiki's risks page).

## Model specs (for cmp-170x-performance-benchmarks.md)

- HF `config.json` fetched directly: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- GGUF file sizes fetched directly from HF file listings: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF), [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
- DFlash speculative decoding architecture: [RedHatAI/DeepSeek-V4-Flash-speculator.dflash](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash) — **the authoritative DFlash number for this project**: validated specifically against `deepseek-ai/DeepSeek-V4-Flash` (not generic DeepSeek-V4), reports per-position acceptance rates and ≈3.9 average accepted length on Nvidia H200, but no throughput/speedup multiplier; [DFlash: Block Diffusion for Flash Speculative Decoding (arXiv)](https://arxiv.org/pdf/2602.06036)
- DSpark (DeepSeek's newer speculative decoding method, beats DFlash 16-18% on accepted length): [MarkTechPost coverage](https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/), [DSpark paper (arXiv)](https://arxiv.org/pdf/2607.05147)

## Notes

- Not included: search-result noise not actually used as a source (eBay listings for the card, a Wikipedia disambiguation page, a Facebook group post, unrelated bookmark aggregator entries, GPU spec-aggregator sites like technical.city/cputronic/gpurankings/topcpu whose numbers were internally inconsistent with the primary teardown/wiki sources).
- The wiki describes an earlier state of the tool in a couple of places (no PCIe Gen2, multi-GPU unmerged) — this repo's actual code is ahead of that. See "Cross-check notes" at the bottom of `TUTORIAL.md` for the reconciliation.
