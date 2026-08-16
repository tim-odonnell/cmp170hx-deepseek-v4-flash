# DeepSeek-V4-Flash-0731 on 3x CMP 170HX — Performance Summary

## TL;DR

We're running [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) —
a large, frontier-class open-weight model — entirely at home, with room for **691,968 tokens
of context** (that's roughly a 500,000-word conversation or document the model can hold in
memory at once). It runs on **3x NVIDIA CMP 170HX**: secondhand cryptocurrency-mining GPUs
(SK Hynix 8GB variant) that we firmware-unlocked to their true 64GB each, each capped at a
modest 100 watts. The serving stack is a patched build of vLLM using pipeline-parallelism
(splitting the model's layers across the three cards) and DSpark speculative decoding (a
"draft and verify" trick that lets the model guess several tokens ahead and check them in
one pass, instead of generating strictly one token at a time).

**Decode speed: ~93 tokens per second.**

If you're new to this space: "tokens per second" is the standard way to measure how fast a
language model writes — roughly 1 token ≈ ¾ of an English word, so ~93 tok/s is around
70 words a second, several times faster than anyone can read.

## The settings

```
--pipeline-parallel-size 3          # split the model across all 3 GPUs
--kv-cache-dtype fp8                # store conversation memory compressed, to save VRAM
--block-size 256
--max-model-len 691968              # the context ceiling described above
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.965      # how much of each GPU's VRAM vLLM is allowed to claim
--max-num-seqs 8                    # how many requests can run at once
--tokenizer-mode deepseek_v4
--no-enable-flashinfer-autotune
--speculative-config {"method":"dspark","num_speculative_tokens":5}

env:
VLLM_PP_LAYER_PARTITION=15,15,13    # how the model's layers are split across the 3 cards
DSV4_LOGITS_ROW_CHUNK=64
VLLM_SPARSE_DENSE_QUERY_BLOCK=0
```

Full launch script: [`vllm-dsv4/phase5-launch-dspark-production.sh`](vllm-dsv4/phase5-launch-dspark-production.sh).

## How we found these numbers

`--gpu-memory-utilization` is the one setting that decides everything else here: push it too
high and one of the three GPUs runs out of the tiny sliver of memory a background calculation
needs mid-request, which crashes the whole server. Push it too low and you leave usable
context on the table for no reason.

We didn't guess. We knew `0.95` utilization was safe (262k tokens of context) and `0.98` reliably
crashed (at the full 1,048,576-token ceiling). So we closed in on the real edge step by step —
testing `0.96`, `0.965`, and `0.97` — and at each step we found the largest context that would
actually load, then **deliberately tried to break it**: maxing out concurrent requests and
feeding it a real prompt near that context limit, the exact conditions that had caused crashes
before. `0.965` survived every attempt. `0.97` crashed on the very first one. That's a hard
cliff, not a gradual slope — which is why we settled on `0.965` rather than inching closer to
`0.97` for a little more context at real risk of instability.

## Build this yourself for cheap

You don't need a server-grade motherboard to run 3 GPUs like this — the model lives entirely
in GPU VRAM, so the rest of the PC barely matters:

- **CPU/board**: an AMD Ryzen 5000G-series APU (e.g. 5600G/5700G) on a small ITX AM4 board.
  These chips have built-in graphics, so you don't need a 4th GPU just to see a desktop. They
  offer 20 PCIe lanes total (16 from the CPU, 4 from the chipset) — plenty, since each mining
  card only needs 4 lanes.
- **RAM**: two cheap DDR4-3200 sticks in dual-channel. The model never touches system RAM
  during inference, so there's nothing to gain from spending more here.
- **Splitting one slot into three**: a PCIe x16→x4/x4/x4/x4 bifurcation card plugged into the
  board's one x16 slot, plus a riser cable per card to physically route them out of the case.
  **Check your motherboard's BIOS explicitly supports 4-way bifurcation before buying anything**
  — many boards don't.
- **Power supply**: budget for the GPUs' 300W combined cap plus roughly 65-100W for the rest of
  the machine. A 650-750W unit gives comfortable headroom rather than cutting it close.

## Tokens per watt-hour

While actively generating text, the three cards together draw about 245 watts (a bit under
their 300W combined cap, since pipeline-parallelism means the cards take turns doing the heavy
lifting rather than all working flat-out at once). At ~93 tok/s, that works out to **roughly
1,150 tokens generated per watt-hour of electricity.**

**Peak vs. sustained power.** The 100W-per-card limit is a hard cap the driver enforces, but
GPU power limiting works like cruise control, not a wall — it targets an average over a short
window, so brief spikes above 100W on an individual card are normal and don't mean the cap
failed. Measured during a real request (a ~115k-token prompt plus a 200-token reply, sampled
every quarter-second):

| | GPU 1 | GPU 2 | GPU 3 | All 3 combined |
|---|---|---|---|---|
| Sustained average | 98 W | 93 W | 95 W | **286 W** |
| Peak instant | 158 W | 235 W | 193 W | **442 W** |

So: budget for ~300W sustained (matches the 100W×3 cap almost exactly), but don't be alarmed
if a power meter briefly shows a higher number for a fraction of a second — that's expected,
not a fault.

We haven't run this exact model on an RTX 3090 (350W), RTX 4090 (450W), RTX 5090 (575W), or
RTX PRO 6000 Blackwell Max-Q (300W) ourselves, so we can't honestly claim a matched
tokens-per-watt comparison against them. What we can say plainly: this entire 3-card system
draws less power while generating text (~245W) than the rated power limit of even one of
those single cards. The real story here isn't "beats a modern GPU's efficiency" — it's
running a genuinely huge model on cheap, heavily power-capped secondhand silicon at all.

**What about a dual NVIDIA DGX Spark?** It's a natural comparison — a single DGX Spark draws
roughly 140W and can be paired with a second unit over its ConnectX-7 networking to pool
memory for big models, landing at a similar ~280W power envelope to our 3-card setup. We
haven't benchmarked one ourselves, so take this as general knowledge rather than a measured
result: DGX Spark's unified memory is fast to buy capacity in (128GB per unit) but relies on
LPDDR5x, which is far slower than the HBM2 on real datacenter-class silicon like these
unlocked A100 dies — and token-generation speed for large models is usually bottlenecked by
memory bandwidth, not raw compute. Community reports on large MoE models running on dual Spark
setups commonly describe decode speeds well below what we're seeing here, despite similar
power draw. If that matches your own testing, we'd genuinely like to know — open an issue.

## How does this compare to the big AI assistants?

For context, here's roughly how fast some well-known hosted assistants tend to feel in
practice, from general knowledge rather than a live benchmark run today — these vary a lot by
model tier, current load, and whether a response is streaming, so treat them as a feel for the
ballpark, not a precise leaderboard:

| Assistant | Typical response speed |
|---|---|
| Claude (Anthropic) | roughly 40-90 tok/s, faster on smaller models like Haiku, slower on the largest (Opus-class) models |
| Gemini (Google) | roughly 60-150 tok/s, with the lightweight "Flash" tier much faster than the top-end "Pro" tier |
| Codex / GPT-5-class (OpenAI) | roughly 40-80 tok/s for typical coding/chat responses |
| Grok (xAI) | roughly 50-90 tok/s in typical use |

Our **~93 tok/s** sits comfortably inside that same range — on hardware that cost a fraction of
what any of those companies spend per GPU, running entirely under your own roof.
