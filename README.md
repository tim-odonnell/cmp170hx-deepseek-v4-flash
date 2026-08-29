# CMP 170HX for local LLM inference

Scripts, benchmarks, and hard-won notes from repurposing NVIDIA CMP 170HX mining
cards (real GA100/A100 silicon, firmware-unlocked from 8GB to 64GB) as a cheap
multi-GPU inference box, running DeepSeek-V4-Flash-0731 and other large MoE
models across 3x CMP 170HX on an ASRockRack ROMED8-2T.

This is not a polished library — it's the working project folder for that
effort, cleaned of personal/local data before publishing. Expect shell
scripts, not a package.

**Quick summary: [PERFORMANCE-SUMMARY.md](PERFORMANCE-SUMMARY.md)** — one page,
current best settings and how we found them.

## What's actually the unlock tool

This repo does **not** include the unlock tool itself — that's
[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker), an
open-source patch for NVIDIA's open-gpu-kernel-modules that reconfigures the
card's SM count and reported VRAM at driver boot (no VBIOS flash, no fuse
blows, fully reversible by power-off). Start with `TUTORIAL.md` and
`HOWTO-UNLOCK-CMP-170HX.md` here, then go run that tool.

The community technical reference for this card is
[Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx).

## Layout

- `TUTORIAL.md`, `HOWTO-UNLOCK-CMP-170HX.md`, `TRANSFER-README.md` — unlock
  procedure, first-boot checklist, and gotchas moving cards between machines.
- `CARD-REGISTRY.md` — per-card identity, verification results, and
  benchmark history. Append-only by design (GPU sets get swapped between
  benchmark runs, so old rows stay valid).
- `card-records/` — one file per physical card (serial, UUID, VBIOS,
  first-boot verification).
- `cmp-170x-performance-benchmarks.md`, `bench-results/` — throughput numbers
  across models/quants/engines tested on this hardware.
- `ONLINE-SOURCES.md`, `REFERENCES.md` — sourcing notes for the hardware specs
  and community claims used throughout, including what was checked and
  rejected as unreliable.
- `vllm-dsv4/` — the newer, faster track: DeepSeek-V4-Flash-0731 on a patched
  vLLM fork with pipeline-parallelism + DSpark speculative decoding. See
  `vllm-dsv4/deepseek-v4-cmp170hx/` for the patch set and rationale, and
  `vllm-dsv4/phase5-launch-dspark-production.sh` for the current best-known
  launch config (verified via a deliberate crash-reproduction test campaign,
  not just "it loaded once").
- `HOWTO-RUN-DEEPSEEK-V4-FLASH.md` — beginner walkthrough for the `vllm-dsv4/`
  track: load, launch, chat with, and shut down the model step by step.
- `bench-*.sh`, `run-*.sh`, `label-artifact*.sh` — the llama.cpp/GGUF-based
  benchmark and inference launch scripts (the earlier, slower track before
  the vLLM work above).
- `gpu-fan-daemon.sh`/`.service`, `thermal-guard.sh`, `cpu-temp-guard.sh`,
  `set-power-limit.sh`, `cmp-powerlimit.service` — the persistent
  power/thermal management this hardware needs (these cards ship passively
  cooled for a server chassis' forced airflow; without active management,
  genuine thermal runaway is possible).
- `capture-card.sh`, `verify-64gb-usable.sh`, `clock-offset-test.sh` — per-card
  verification and characterization tooling.

## Hardware context

ASRockRack ROMED8-2T, AMD EPYC 7413, 3x CMP 170HX unlocked to 64GB each
(192GB total VRAM), PCIe Gen2 x4 per card (no P2P), 100W power cap per card.
Not NVLink, not a datacenter box — a homelab build using secondhand mining
cards. Numbers here reflect that topology; don't expect them to transfer
directly to NVSwitch/TP setups.

## Status

Actively used, not archived. `CARD-REGISTRY.md` and the `vllm-dsv4/` scripts
are the most current parts of this repo; other benchmark scripts reflect
whichever GPU set/model combination was being tested at the time and may be
superseded by newer results in `CARD-REGISTRY.md`.
