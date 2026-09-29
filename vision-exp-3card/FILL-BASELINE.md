# Phase 0: 0731 fill baseline (measured 2026-09-29)

**This is the ceiling Vision-Exp must meet or come in under** (project brief: "filled an
equal or slightly less amount than the dsv4f-0731 recipe").

Config measured: the live production launcher exactly as you run it
(`~/CMP-170HX-PROJECT/vllm-dsv4/phase5-launch-dspark-production.sh`, no args):
util 0.965, max-model-len 691,968, max-num-seqs 8, partition 15,15,13, batch 2048, DSpark
n=5, fp8 KV, 100 W/card. Image `dsv4-a100:devel` (`sha256:49516546cb70…`).

## vLLM's own per-rank accounting (startup log)

| Rank | GPU | Weights loaded | Consumed (weights + non-torch) | Peak activation | CUDA graphs | Budget (0.965) |
|---|---|---|---|---|---|---|
| PP0 | 0 | 51.67 GiB | 53.57 GiB | 0.81 GiB | 0.33 GiB | 61.3 GiB |
| PP1 | 1 | 50.45 GiB | 52.50 GiB | 0.80 GiB | 0.32 GiB | 61.3 GiB |
| PP2 | 2 | 56.24 GiB | 58.41 GiB | 0.71 GiB | 0.36 GiB | 61.3 GiB |

- Free on device at startup: 63.07 of 63.53 GiB per card.
- KV cache pool: **804,497 tokens** (6.92 GiB available on PP0). Max concurrency at 691,968
  tokens/request: 1.16x.
- PP2 is the tight rank (last stage: 13 layers + lm_head + DSpark drafter). Same as 2026-08-15.
- Load time: ~5 min to healthy (weights ~65-90 s per rank, init/profile/warmup 147 s).

## `nvidia-smi` memory.used per card

| | GPU 0 (PP0) | GPU 1 (PP1) | GPU 2 (PP2) | GPU 3 |
|---|---|---|---|---|
| Idle, loaded | 58,160 MiB | 57,024 MiB | 63,092 MiB | 0 |
| **Peak during 629,708-token request** | **59,456 MiB** | **58,338 MiB** | **64,370 MiB** | 0 |
| Peak as % of 65,536 MiB | 90.7% | 89.0% | **98.2%** | 0% |

Deep request: 629,708 prompt tokens of real source text (`scripts/depth-probe.py`, 0.5 s
sampling), 480 s wall for prefill + 20-token answer (~1,310 tok/s effective prefill). Survived
and stayed healthy afterwards. Same depth as the original 2026-08-15 stress pass (629,809).
Raw: `results/phase0-0731-depth630k.json`, `logs/phase0-0731-startup.log`.

## Shallow decode speed (same script Vision-Exp will run)

`scripts/speed-bench.py`: streaming, 300 tokens with ignore_eos, 1 warmup discarded, 5 runs,
TTFT separated, tokens from `usage`, random prefix per prompt, thinking off.

| mean | median | min | max | TTFT median |
|---|---|---|---|---|
| **99.73 tok/s** | 95.52 | 76.33 | 121.5 | 0.098 s |

Spread is DSpark draft acceptance varying by prompt, as documented on 2026-08-15. Raw:
`results/phase0-0731-speed-shallow.json`.

## Pass criteria this sets for Vision-Exp (Phase 4)

1. `--gpu-memory-utilization` ≤ 0.965.
2. Per-card `nvidia-smi` peak under the equivalent deep-context stress ≤ **59,456 / 58,338 /
   64,370 MiB** (GPU 0/1/2). Note especially GPU 0: Vision-Exp puts the vision tower there.
3. GPU 3 stays at 0 MiB throughout.
4. Speed target: shallow decode mean ≥ ~99.7 tok/s on the same script (or within noise of it,
   given DSpark variance; compare medians as well as means).

## Side findings

- **Harness self-check against 0731**: asked the gradient question *with no image*, 0731
  confidently guessed "BLUE, RED". Red/blue is the model's default blind guess, so a
  red→blue gradient can't prove vision works. The gate suite now uses green→orange, and the
  no-image control fails only if the blind answer matches.
- The repo copy `../vllm-dsv4/phase5-launch-dspark-production.sh` is **stale** vs the live
  one (missing `--override-generation-config '{"top_p": 0.95}'` and the tool/reasoning
  parser flags added 2026-08-20). Memory-neutral. Fix at publish time.
