# How to run DeepSeek-V4-Flash-Vision-Exp on 3x CMP 170HX

The vision-capable sibling of the 0731 recipe
([HOWTO-RUN-DEEPSEEK-V4-FLASH.md](HOWTO-RUN-DEEPSEEK-V4-FLASH.md)): same 3 cards, same GPU fill,
same 691,968-token context, **plus image input**. It can read screenshots, tables and
reference images, and it calls tools. A 4th card, if you have one, stays completely free.

It runs in a Docker container with an OpenAI-compatible API on **port 8099** (0731 uses 8098).
Both use the same three GPUs, so you run one or the other and switch with one command.

The full engineering record (every boot, crash and fix) is in
[`vision-exp-3card/`](vision-exp-3card/): start with its
[CAMPAIGN-LOG.md](vision-exp-3card/CAMPAIGN-LOG.md). If it ever crashes, go to
[TROUBLESHOOTING-XID31.md](vision-exp-3card/TROUBLESHOOTING-XID31.md).

---

## Results at a glance (measured 2026-09-29, 100 W per card)

| | 0731 recipe | **Vision-Exp recipe** |
|---|---|---|
| Cards | 3 (GPUs 0-2) | 3 (GPUs 0-2), GPU 3 untouched |
| `--gpu-memory-utilization` | 0.965 | **0.965** |
| Context | 691,968 | **691,968** |
| Peak VRAM per card under a 630k-token request (MiB) | 59,456 / 58,338 / 64,370 | 60,722 / 58,410 / 64,504 |
| Decode, short prompt (DSpark) | ~99.7 tok/s | **~86 tok/s** |
| Decode at 100k / 300k context | 80 / 79 tok/s | 78 / 68 tok/s |
| Prefill at 100k / 300k | 1,935 / 2,282 tok/s | **2,154 / 2,384 tok/s** |
| Images | no | **yes**, 9/9 vision gates incl. a real webpage screenshot and a 25/25-cell table |
| Tool calling | yes | **yes**, 7/7 incl. image→tool and multi-step web research with 0 invented sources |
| Stress / stability | seqs 8 at ~630k, 3 cold restarts | same, **plus** image traffic, 38-min soak (20/20 loops), 0 Xid |
| Temps under load | — | core ≤ 67 °C, HBM ≤ 77 °C |

**Why it's ~13% slower on short prompts:** not the engine or the settings. The per-step
speed is identical (29.5 vs 30.5 steps/s). Vision-Exp's built-in DSpark draft head simply gets
fewer guesses accepted (~2.9 vs ~3.5 tokens per step). That's a property of DeepSeek's
checkpoint. At 100k context the two are nearly even, and Vision-Exp prefills faster.

**GPU 0 sits ~1.3 GB higher than 0731** because it carries the vision tower (0731 has none).
GPU 2, the card that crashes when overfilled, is within 0.2% of 0731. A more conservative
variant is also validated: `0.96` / `572,160` context puts GPUs 1-2 ~700 MiB *under* 0731
(pass `0.96 572160` to the launcher).

---

## 0. One-time setup

You need everything from the 0731 setup (cards unlocked to 64 GB, 100 W power cap, Docker +
NVIDIA Container Toolkit, driver with CUDA ≥ 13). Plus:

### 0a. Download the model (~168 GB)

```bash
# either: huggingface CLI
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ~/models/models/deepseek-ai-DeepSeek-V4-Flash-Vision-Exp
# or: aria2c, ONE connection per file (-x1 -s1); HF's Xet CDN rejects split downloads (403)
```
48 shards, 167,831,846,872 bytes. Any revision from `86f746b3` on has identical weights (later
commits only changed the README).

### 0b. Build the engine image (~40 min, mostly compiling)

This is the [leinasi2014/deepseek-v4-vision-cmp170hx](https://github.com/leinasi2014/deepseek-v4-vision-cmp170hx)
vLLM fork (a Vision-Exp port onto the same fast SM80 base as the 0731 engine), built unmodified,
plus our one small patch. One command does all of it (clones the fork at a pinned commit,
fetches pinned CUTLASS v4.4.2 / Triton v3.5.1, full native build for sm_80, then the patch
overlay):

```bash
cd ~/cmp170hx-deepseek-v4-flash/vision-exp-3card
nohup ./scripts/build-image.sh > logs/build.log 2>&1 &
tail -f logs/build.log            # ends with "BUILD_OK ... use image dsv4-vision:sm80-leinasi-4fe10bf-p1"
```
Set `BASE=/some/dir` first if you don't want the sources under `~/CMP-170HX-PROJECT/vllm-dsv4`.

**About the patch** (`vision-exp-3card/patches/0001-*.patch`, one file, ~20 lines): the fork builds a copy
of the 0.87 GiB vision encoder on *every* pipeline stage, but only the first stage ever uses
it. On 3 cards that dead copy was exactly the memory the last card was missing, so without the
patch the maximum context drops a lot. With it, GPUs 1 and 2 load byte-for-byte what 0731 loads.

**Check it built:** `docker image inspect dsv4-vision:sm80-leinasi-4fe10bf-p1` prints JSON.

---

## 1. Check what's running

```bash
curl -s http://127.0.0.1:8099/health && echo "  <-- Vision-Exp UP"
curl -s http://127.0.0.1:8098/health && echo "  <-- 0731 UP"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,temperature.memory --format=csv
```

## 2. Launch it

```bash
cd ~/cmp170hx-deepseek-v4-flash/vision-exp-3card
./scripts/launch-vision-3card.sh
```
No arguments needed; the defaults are the validated recipe (0.965 / 691,968 / 8 seqs / DSpark).
**If 0731 is running, this stops it first.** The script refuses any utilization above 0.965,
and if the GPUs are in a bad state it stops with a message telling you to reboot, rather than
trying risky driver tricks.

## 3. Wait for it (~3-5 minutes)

```bash
docker logs -f dsv4-vision-3card        # ready at "Application startup complete"; Ctrl+C just stops watching
curl -s http://127.0.0.1:8099/health && echo UP
```
Healthy startup shows `GPU KV cache size: 797,299 tokens`. A different number means something
changed (wrong image tag, different settings).

## 4. Use it

OpenAI-compatible API: base URL `http://localhost:8099/v1`, model **`dsv4v`**, any API key.

**Text:**
```bash
curl -s http://localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv4v", "messages": [{"role": "user", "content": "In one sentence, what is a NURBS surface?"}]
}' | python3 -m json.tool
```

**An image** (a local screenshot, sent as base64):
```bash
IMG=$(base64 -w0 screenshot.png)
curl -s http://localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv4v",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this page and point out any layout bugs."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}
  ]}]
}' | python3 -m json.tool
```
Up to **8 images per conversation** (vLLM counts every image in the chat history, not just the latest message; change with `IMAGES=N ./scripts/launch-vision-3card.sh`); each costs ≤ 384 tokens. Raising it from 2 to 8 was validated on 2026-09-29: identical memory/KV pool, and the 630k stress with 6-image requests alongside peaked slightly *lower* than before. Tool calling works with standard OpenAI
`tools`.

**Thinking is OFF by default** in this recipe (unlike the 0731 HOWTO, which thinks by default).
All of today's tests used that default. To turn it on for a request, add
`"chat_template_kwargs": {"thinking": true}`. The reasoning then comes back in
`reasoning` / `reasoning_content`, and the answer in `content`.

**In opencode / Open WebUI:** add an OpenAI-compatible provider at `http://localhost:8099/v1`
with model `dsv4v`, and mark it as image-capable so the client sends pictures.

**Check vision really works** (9 self-grading tests, ~1 min):
```bash
python3 scripts/vision-gates.py --port 8099 --model dsv4v     # expect SUMMARY ... all true
```

## 5. Switch back to 0731

```bash
./scripts/switch-to-0731.sh         # stops Vision-Exp, runs the unmodified 0731 launcher (port 8098)
```
And `./scripts/launch-vision-3card.sh` switches the other way. Each switch is a ~3-5 min load.

## 6. Stop it

```bash
docker stop dsv4-vision-3card       # frees GPUs 0-2 immediately
```

## 7. If something goes wrong

1. Run `./scripts/capture-crash.sh` **before** restarting anything. It's read-only and saves the logs.
2. **Don't** `rmmod`/`modprobe` the NVIDIA modules. On these cards that turns one fault into
   "reboot required" on every GPU. If `sudo dmesg | grep -i xid` shows anything, reboot.
3. Open [TROUBLESHOOTING-XID31.md](vision-exp-3card/TROUBLESHOOTING-XID31.md). It has a
   triage table of every crash type seen on this hardware and how to re-test.

## Cheat sheet

| Goal | Command (from `vision-exp-3card/`) |
|---|---|
| Is it up? | `curl -s http://127.0.0.1:8099/health && echo UP` |
| Start (stops 0731) | `./scripts/launch-vision-3card.sh` |
| Conservative variant | `./scripts/launch-vision-3card.sh 0.96 572160` |
| Back to 0731 | `./scripts/switch-to-0731.sh` |
| Vision self-test | `python3 scripts/vision-gates.py --port 8099 --model dsv4v` |
| Stop | `docker stop dsv4-vision-3card` |
| Crash evidence | `./scripts/capture-crash.sh` |
| Watch temps | `nohup setsid scripts/thermal-logger.sh > logs/thermal-logger.out 2>&1 &` |

**Never** set `--gpu-memory-utilization` above 0.965, raise the 100 W power cap, or remove
`DSV4_LOGITS_ROW_CHUNK=64` from the launcher (that last one causes a guaranteed crash on the
first request past ~174k tokens; see the troubleshooting doc).

## Credits

Engine: [leinasi2014/deepseek-v4-vision-cmp170hx](https://github.com/leinasi2014/deepseek-v4-vision-cmp170hx)
(Apache-2.0). No-P2P profile: [Wiziechen/cmp170hx-dsv4v-conda](https://github.com/Wiziechen/cmp170hx-dsv4v-conda).
Bug reports and test ideas: [kaka86mm/dsv4-vision-170hx](https://github.com/kaka86mm/dsv4-vision-170hx),
[PixelML/club-170hx](https://github.com/PixelML/club-170hx). Earlier SM80 work:
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport),
[allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx).
This is the first published **3-card** Vision-Exp config for the CMP 170HX that we know of;
all the others above are 4-card.
