# How to run DeepSeek-V4-Flash-0731 on the 3x CMP 170HX box

A start-to-finish walkthrough: load the model, talk to it, then shut it down
cleanly. Written for someone who has never used this setup before.

The model runs inside a Docker container that exposes an OpenAI-compatible API
on **port 8098**. You start that container, wait for it to load, send it chat
requests, then stop it. Stopping the container is what frees the GPUs.

---

## 0. One-time prerequisites (already done on this machine)

You only need to check these if something below fails.

- The 3 CMP 170HX cards are firmware-unlocked to 64 GB and power-capped to
  100 W (a systemd service does this at boot — leave it alone).
- The Docker image `dsv4-a100:devel` is built.
  Check: `docker image inspect dsv4-a100:devel` (should print JSON, not an error).
- The model files are at `~/models/models/deepseek-ai-DeepSeek-V4-Flash-0731`
  (~156 GB, 48 shards).
  Check: `ls ~/models/models/deepseek-ai-DeepSeek-V4-Flash-0731`

If any of those are missing, see `PERFORMANCE-SUMMARY.md` and
`PLAN-vllm-pp-dspark-3card.md` for how they were built.

---

## 1. Check nothing is already running

```bash
curl -s http://127.0.0.1:8098/health && echo "  <-- already UP" || echo "not running"
docker ps --filter name=dsv4-a100-3card
nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv
```

If the server is already up, skip to step 4. If `nvidia-smi` shows the cards
using several GB each but no `dsv4-a100-3card` container exists, something else
is using the GPUs — sort that out before continuing.

---

## 2. Launch it

```bash
cd ~/cmp170hx-deepseek-v4-flash        # or wherever this repo is checked out
./vllm-dsv4/phase5-launch-dspark-production.sh
```

That's the whole command. The script uses the settled production settings
(DSpark speculative decoding, ~691k token context, the verified-safe GPU
memory setting). You don't pass any arguments for normal use.

It prints something like:

```
launched dsv4-a100-3card on :8098 (util=0.965, maxlen=691968, ... DSpark) -- PRODUCTION DEFAULT
```

### Alternative: no speculative decoding

For very long-context work (documents past ~300k tokens), the plain version is
more stable:

```bash
./vllm-dsv4/phase3-launch-nospec.sh 691968
```

Same API, same port. Slower per token but no crash risk at extreme depth.

---

## 3. Wait for it to finish loading

Loading the 156 GB model across 3 cards takes roughly **3-8 minutes**. Watch it:

```bash
docker logs -f dsv4-a100-3card
```

It's ready when the log shows a line like `Application startup complete` /
`Starting vLLM API server`. Press `Ctrl+C` to stop watching the log (that does
**not** stop the server).

Confirm it's live:

```bash
curl -s http://127.0.0.1:8098/health          # prints nothing + exit 0 = healthy
curl -s http://127.0.0.1:8098/v1/models       # lists the model, served as "dsv4s"
```

---

## 4. Talk to it

It speaks the OpenAI chat API. Base URL `http://localhost:8098/v1`, model name
`dsv4s`, no API key needed.

### Quick test from the terminal

```bash
curl -s http://localhost:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4s",
    "messages": [{"role": "user", "content": "In one sentence, what is a NURBS surface?"}]
  }' | python3 -m json.tool
```

The reply text is in `choices[0].message.content`. This model also "thinks"
before answering — that reasoning is returned separately in
`choices[0].message.reasoning_content`, not mixed into the answer.

### From a real client

Point any OpenAI-compatible tool at it:

- Base URL: `http://localhost:8098/v1`
- API key: any non-empty string (e.g. `local`)
- Model: `dsv4s`

That includes `opencode`, Open WebUI, the `openai` Python package, etc.

---

## 5. Close it (stop serving, free the GPUs)

```bash
docker stop dsv4-a100-3card
```

Stopping the container immediately frees all GPU memory. Verify:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv
```

All three cards should drop back to ~0 MiB. The model files, the Docker image,
and your settings are untouched — starting again is just step 2.

### Fully remove the stopped container (optional tidy-up)

```bash
docker rm dsv4-a100-3card
```

Not required — the launch script deletes a leftover container automatically on
the next start.

---

## 6. If the GPUs get stuck

Rare, but a mid-request crash can leave a card wedged (`nvidia-smi` shows
`[GPU requires reset]` or errors). The launch scripts try to auto-recover with
a driver-module reload. If that doesn't work, **just reboot the machine** —
that's the only reliable fix for this hardware, and it's harmless:

```bash
sudo reboot
```

After it comes back, go to step 1.

---

## Cheat sheet

| Goal | Command |
|---|---|
| Is it running? | `curl -s http://127.0.0.1:8098/health && echo UP` |
| Start | `./vllm-dsv4/phase5-launch-dspark-production.sh` |
| Watch it load | `docker logs -f dsv4-a100-3card` |
| List model | `curl -s http://127.0.0.1:8098/v1/models` |
| Chat | POST to `http://localhost:8098/v1/chat/completions`, model `dsv4s` |
| Stop / free GPUs | `docker stop dsv4-a100-3card` |
| Check GPUs freed | `nvidia-smi` |
| Recover stuck GPUs | `sudo reboot` |

Do **not** raise the GPU power limit above 100 W or the
`--gpu-memory-utilization` above `0.965` — both are deliberate and tested. See
`PERFORMANCE-SUMMARY.md` for why.
