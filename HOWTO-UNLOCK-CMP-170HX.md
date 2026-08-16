# How to unlock an NVIDIA CMP 170HX — complete beginner's guide

**A step-by-step procedure to turn an 8 GB CMP 170HX into a 64 GB GPU.**

Written 2026-08-06, immediately after doing exactly this successfully. Every command here
was actually run, and every "what you should see" block is real output from a real card —
not an example. Written for someone who has used Ubuntu a little but does not consider
themselves an expert.

**Time required:** about 45 minutes, most of it waiting. Two restarts are required.

---

## Contents

1. [What this does](#1-what-this-does)
2. [Read this before you buy or power anything on](#2-read-this-before-you-buy-or-power-anything-on)
3. [What you need](#3-what-you-need)
4. [Terminal basics](#4-terminal-basics-skip-if-you-know-this)
5. [Step 1 — Identify your card](#step-1--identify-your-card)
6. [Step 2 — Check the four prerequisites](#step-2--check-the-four-prerequisites)
7. [Step 3 — Get the unlock tool](#step-3--get-the-unlock-tool)
8. [Step 4 — Install the correct driver](#step-4--install-the-correct-driver)
9. [Step 5 — Restart (do not skip)](#step-5--restart-do-not-skip)
10. [Step 6 — Run the unlock installer](#step-6--run-the-unlock-installer)
11. [Step 7 — Cold power cycle](#step-7--cold-power-cycle-not-a-restart)
12. [Step 8 — Verify](#step-8--verify)
13. [Step 9 — Prove the memory is real](#step-9--prove-the-memory-is-real-recommended)
14. [Step 10 — Set a power limit](#step-10--set-a-power-limit)
15. [Keeping it working](#keeping-it-working-important)
16. [Troubleshooting](#troubleshooting)
17. [Undoing everything](#undoing-everything)

---

## 1. What this does

The CMP 170HX is a mining card NVIDIA sold in 2021. Underneath, it is the **same GA100
silicon as the A100** — a data-centre GPU that cost around $10,000. NVIDIA crippled it in
firmware so it could not be used for general computing:

| | Locked (as sold) | Unlocked |
|---|---|---|
| Memory | 8 GB | **64 GB** |
| FP32 compute | 0.395 TFLOPS | **~12.7 TFLOPS** (about 32× faster) |
| PCIe link speed | Gen1 | **Gen2** |

The unlock is **software only**. It writes to volatile GPU registers during driver startup —
it does **not** flash your BIOS, does not blow fuses, and does not modify the card
permanently. Turning the computer off and on returns the card to stock. That makes the
software side very low-risk.

There is a 10 GB version of this card too. It unlocks to 40 GB, not 64 GB. Step 1 tells you
which one you have.

---

## 2. Read this before you buy or power anything on

Please actually read this section. The software is safe; the hardware is where people get
hurt.

### ⚠️ Cooling is the real danger

This is a **250-watt** card designed for a server chassis with screaming high-pressure fans.
It ships **with no fan of its own** — just a bare heatsink. If you run a heavy workload
without arranging airflow first, the chip can enter *thermal runaway*: it gets hot, which
increases power leakage, which makes it hotter, and so on.

- **One 40 mm fan is not enough.** Plan on two, in a duct or shroud that forces air through
  the heatsink fins.
- Get cooling sorted **before** you run any real workload.
- Ordinary benchmarks will not reveal the problem — they do not stress memory hard enough.
  A large AI model running for several minutes will.

Doing the unlock itself is safe without a fan (it is just register writes, and the card sits
near idle). *Using* the card is not.

### ⚠️ The power connector is not a PCIe connector

The card uses an **EPS 8-pin** connector — the kind that normally powers a CPU. It looks very
similar to a PCIe 8-pin GPU connector but **the pin assignments are different**. Forcing a
PCIe cable in **will damage the card**. Check the label on your cable; use the EPS one.

### ⚠️ No display output

The 170HX cannot drive a monitor. You need something else for display — integrated graphics,
another GPU, or a server management chip. Check you have that before you start.

### ⚠️ Legal / warranty

NVIDIA issued a DMCA takedown against at least one copy of this tool in 2026. The tool builds
NVIDIA's own open-source driver code with patches applied. Running unsigned kernel modules
voids any remaining seller warranty — usually irrelevant on a secondhand mining card, but
worth knowing.

---

## 3. What you need

| | |
|---|---|
| **Operating system** | Ubuntu 24.04 LTS (this guide's exact commands). Other Linux works but package names differ. |
| **The card** | Physically installed, EPS 8-pin power connected, cooling arranged |
| **A display** | Integrated graphics or a second GPU — the 170HX cannot drive a monitor |
| **Internet** | Needed once, to download the driver and tool |
| **Secure Boot** | Must be **disabled** in BIOS/UEFI |
| **Disk space** | About 2 GB |
| **Time** | ~45 minutes including two restarts |

---

## 4. Terminal basics (skip if you know this)

Everything here happens in the Terminal.

**Opening it:** press `Ctrl` + `Alt` + `T`.

**Pasting into it:** `Ctrl` + `Shift` + `V` (note the extra `Shift` — plain `Ctrl+V` does not
work in a terminal).

**About `sudo`:** commands starting with `sudo` run as administrator. The first time you use
it, it asks for your password. **As you type your password, nothing appears on screen — no
dots, no stars.** This is normal and not a bug. Type it and press Enter.

**Running a command:** paste it, press Enter, wait for the prompt to come back before running
the next one.

**If something looks wrong, stop.** Do not proceed to the next step hoping it resolves
itself. The troubleshooting section covers what actually goes wrong.

---

## Step 1 — Identify your card

This determines everything else. Open a terminal and run:

```bash
lspci -nn | grep -i nvidia
```

**What you should see** — the important part is the code in the second set of square
brackets:

```
01:00.0 3D controller [0302]: NVIDIA Corporation GA100 [CMP 170HX] [10de:20c2] (rev a1)
```

Find your code in this table:

| Code | Card | Unlocks to | Profile name |
|---|---|---|---|
| `10de:20c2` | 8 GB CMP 170HX | **64 GB** | `8gb` |
| `10de:2082` | 10 GB CMP 170HX | **40 GB** | `10gb` |
| `10de:20b0` | A100 engineering sample | **Does not unlock** — stop here | — |

**Write down your code and profile name.** You need the profile name in Step 6.

> **Note:** you may read that the 8 GB card has SK Hynix memory and the 10 GB card has
> Samsung. That is an educated guess the community has never confirmed, and it does not
> affect anything — the unlock is chosen purely by the code above.

If you get no output at all, the card is not seated properly or has no power. Shut down and
recheck the card and its EPS cable.

---

## Step 2 — Check the four prerequisites

Run these one at a time.

### 2a. Secure Boot must be OFF

```bash
mokutil --sb-state
```

**Want to see:** `SecureBoot disabled`

If it says *enabled*, you must turn it off in your BIOS/UEFI. Restart, press the setup key
during boot (usually `Del`, `F2`, or `F12` — your screen says which), find Secure Boot under
Security or Boot, disable it, save and exit. The patched driver files are unsigned and will
not load with Secure Boot on.

### 2b. Note your kernel version

```bash
uname -r
```

**Example:** `7.0.0-28-generic`. Write this down — you will see it again.

### 2c. Install the build tools

```bash
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r) python3 curl git patch
```

This takes a minute or two. `linux-headers-$(uname -r)` automatically fills in your kernel
version from 2b.

### 2d. Confirm the card is working *before* you change anything

```bash
nvidia-smi
```

If you have an NVIDIA driver installed already, you should see the card listed at its stock
size (8192 MiB or 10240 MiB). **This is a good thing** — it confirms the card and slot work
before you change anything.

If you get `command not found`, that is fine too — you have no NVIDIA driver yet, and Step 4
installs one.

---

## Step 3 — Get the unlock tool

```bash
mkdir -p ~/CMP-170HX-PROJECT
cd ~/CMP-170HX-PROJECT
git clone https://github.com/amoghmunikote/cmpunlocker.git
cd cmpunlocker
```

If that URL fails, the repository may have been taken down. See
[ONLINE-SOURCES.md](ONLINE-SOURCES.md) — this project keeps a complete offline copy that can
be restored with:

```bash
git clone /path/to/offline-archive/git-bundles/cmpunlocker-360acd7.bundle cmpunlocker
```

Now check which driver versions your copy of the tool accepts:

```bash
cat driver/VERSION
```

**What you should see** (yours may differ — trust your file, not this guide):

```
610.57.04
610.43.03
610.43.02
```

**This list is the single most important thing in this guide.** The tool works with *only*
these driver versions. Anything else fails immediately.

---

## Step 4 — Install the correct driver

> **This is where nearly everyone gets stuck.** The tool needs a very specific driver
> version. If you have a different one — even a newer one — it will refuse to run.

Check what you have now:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

If that number is already in your `driver/VERSION` list, skip to Step 6.

Otherwise, check whether Ubuntu has the right one:

```bash
apt-cache policy nvidia-driver-610-open
```

**What you should see:**

```
nvidia-driver-610-open:
  Installed: (none)
  Candidate: 610.43.02-0ubuntu0.24.04.1
     500 http://security.ubuntu.com/ubuntu noble-security/multiverse amd64 Packages
```

The `Candidate` version must appear in your `driver/VERSION` list. In this example
`610.43.02` does, so we install it:

```bash
sudo apt install -y nvidia-driver-610-open
```

This takes several minutes and prints a great deal of text. It will **remove your old NVIDIA
driver** and build the new one. That is expected.

> **Note the `-open` suffix.** The tool patches NVIDIA's *open source* driver. A regular
> (proprietary) driver will not work. Always install the package name ending in `-open`.

**If `apt-cache policy` showed no candidate**, your Ubuntu does not carry a supported
version, and you will need NVIDIA's `.run` installer instead. Ubuntu's `noble-security/multiverse`
only ever carried `610.43.02` as `.deb` packages — `610.43.03` and `610.57.04` never appeared
there. Both `.run` installers (and the matching `open-gpu-kernel-modules` source tarballs
`cmpunlocker/driver/build.sh` builds against) are pre-downloaded in
`offline-archive/nvidia-driver-610.43.03-run/`, `offline-archive/nvidia-driver-610.57.04-run/`,
and `offline-archive/source/` — checksums in `offline-archive/SHA256SUMS`. Install with
`sudo sh NVIDIA-Linux-x86_64-<version>.run -m=kernel-open` (the `-m=kernel-open` flag is required
— it's what makes the `.run` installer build the *open* kernel modules cmpunlocker needs, same
requirement as the `-open` package suffix above). ⚠️ Only `610.43.02` has actually been installed
and tested against this hardware in this project (see `CARD-REGISTRY.md`) — the other two are
staged for future use, not yet verified end-to-end.

---

## Step 5 — Restart (do not skip)

```bash
sudo reboot
```

**Why this is mandatory and not optional:** installing the driver puts new files on disk, but
your computer keeps running the *old* driver until it restarts. The unlock installer checks
which driver is *currently running*, not which is installed. Skip this restart and it sees
the old version and refuses to continue — a confusing failure that looks like the install
did not work.

After it comes back, confirm the new driver is actually running:

```bash
cat /proc/driver/nvidia/version
```

**Want to see** your target version:

```
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.43.02  Release Build
```

If it still shows the old version, the restart did not take effect. Do not continue.

---

## Step 6 — Run the unlock installer

Use the profile name from Step 1 (`8gb` or `10gb`):

```bash
cd ~/CMP-170HX-PROJECT/cmpunlocker
sudo ./install.sh --profile=8gb
```

> **Always pass `--profile` explicitly**, even with one card. If any other NVIDIA GPU is in
> the machine, automatic detection can read *its* memory size by mistake. Stating it removes
> the ambiguity.

This runs for **5–15 minutes** — it compiles the entire NVIDIA driver from source. Screens of
text scroll past. Compiler warnings are normal and harmless.

**What success looks like** at the end:

```
cmpunlocker install finished!
Profile: 8gb  |  1 GPU(s): 1× 8gb, 0× 10gb
IOMMU:   amd_iommu=on iommu=pt (configured)

Per-GPU expectations after unlock:
  BDF              PCI ID   Variant  Expect MiB
  0000:01:00.0     20c2     8gb      ~65536
```

Check the `Expect MiB` figure matches what Step 1 predicted (65536 for 8 GB cards, 40960 for
10 GB cards).

**Do not be alarmed by the warning at the very bottom**, but do read it:

> *This script removed the nvidia DKMS kernel modules. You will need to re-run this script
> after each kernel upgrade.*

This matters a lot later — see [Keeping it working](#keeping-it-working-important).

---

## Step 7 — Cold power cycle (not a restart)

**This must be a full power-off, not `reboot`.**

```bash
sudo shutdown -h now
```

Wait for the machine to fully power down. Wait about five seconds. Then press the power
button.

**Why it has to be this way:** the unlock happens while the GPU itself powers up. A normal
restart never removes power from the graphics card, so leftover state from the previous boot
can make the unlock silently fail. Only a genuine power-off resets the card.

---

## Step 8 — Verify

The moment of truth:

```bash
nvidia-smi
```

**What success looks like:**

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
|   0  NVIDIA CMP 170HX               On  |   00000000:01:00.0 Off |                  N/A |
| N/A   39C    P0             38W /  100W |       6MiB /  65536MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

Two things to check:

1. **`65536MiB`** (or `40960MiB` for a 10 GB card) — the memory unlock worked
2. **The name is now `NVIDIA CMP 170HX`** rather than "NVIDIA Graphics Device" — proof the
   patched driver is the one running

Now the tool's own check:

```bash
cd ~/CMP-170HX-PROJECT/cmpunlocker
sudo ./verify.sh
```

**Want to see:**

```
✓ dmesg contains SEC2_DEBUG unlock logs
✓ All 1 unlockable GPU(s) report unlocked memory
✓ All supported GPUs are negotiated at Gen2 or better
```

### Confirm the unlock registers (optional but reassuring)

```bash
sudo dmesg | grep "POST-WRITE"
```

**What you should see** — one line containing four values:

```
SEC2_DEBUG: POST-WRITE SS0=0x88888888 SS1=0x00000008 CFG1=0x02779000 LMR=0x0000020b (devId=0x20c2)
```

Compare against the table for your card:

| Value | 8 GB card (`20c2`) | 10 GB card (`2082`) | What it does |
|---|---|---|---|
| `SS0` | `0x88888888` | `0x88888888` | Removes the compute speed limit |
| `SS1` | `0x00000008` | `0x00000008` | Removes the compute speed limit |
| `CFG1` | `0x02779000` | `0x02669000` | Sets the memory capacity tier |
| `LMR` | `0x0000020B` | `0x0000028A` | Declares the total memory size |

All four matching means both the memory *and* compute unlocks landed.

### Check the PCIe speed

```bash
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max --format=csv
```

**Want to see:** `2, 2` — the card is at Gen2, up from Gen1.

The width stays at **x4**. Getting x16 requires soldering 24 tiny capacitors onto the board
and is a completely separate hardware modification.

---

## Step 9 — Prove the memory is real (recommended)

**Do not skip this.** `nvidia-smi` *reporting* 64 GB and the card actually *having* 64 GB
usable are different claims. A known-bad unlock profile reports its size correctly and then
crashes when you use the memory beyond a certain point.

This test allocates 56 GB, writes a distinct pattern into each chunk, then reads it back to
confirm nothing overlaps or wraps around. Takes about 30 seconds and barely warms the card.

Copy this whole block and paste it into the terminal:

```bash
python3 - <<'PY'
import ctypes
cuda = ctypes.CDLL('libcuda.so.1')
for f, a in [('cuMemAlloc_v2',   [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
             ('cuMemsetD8_v2',   [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_size_t]),
             ('cuMemcpyDtoH_v2', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])]:
    getattr(cuda, f).argtypes = a
cuda.cuInit(0)
dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
GiB = 2**30; CH = 4*GiB; ptrs = []
for i in range(14):                       # 14 x 4 GiB = 56 GiB
    p = ctypes.c_void_p()
    if cuda.cuMemAlloc_v2(ctypes.byref(p), CH) != 0:
        print(f"  stopped after {i*4} GiB"); break
    ptrs.append(p)
print(f"allocated {len(ptrs)*4} GiB")
bad = 0
for i, p in enumerate(ptrs):
    cuda.cuMemsetD8_v2(p, (i*17+3) & 0xFF, CH)
buf = (ctypes.c_ubyte*8)()
for i, p in enumerate(ptrs):
    exp = (i*17+3) & 0xFF
    for off in (0, CH//2, CH-8):
        cuda.cuMemcpyDtoH_v2(buf, ctypes.c_void_p(p.value+off), 8)
        if any(b != exp for b in buf):
            print(f"  MISMATCH at {i*4} GiB"); bad += 1
print("RESULT:", "PASS - all memory genuinely usable" if bad == 0 else f"FAIL - {bad} errors")
PY
```

**Want to see:**

```
allocated 56 GiB
RESULT: PASS - all memory genuinely usable
```

Then confirm no hardware faults were logged:

```bash
sudo dmesg | grep -i xid
```

**Empty output is what you want.** Any `Xid` line means a real GPU fault occurred.

If this test fails but `nvidia-smi` showed 65536 MiB, **do not use the card for real work** —
you have a card that lies about its capacity, and jobs will crash unpredictably.

---

## Step 10 — Set a power limit

First, see what your card allows:

```bash
nvidia-smi -q -d POWER | grep -E "Power Limit"
```

**Typical output:**

```
Current Power Limit    : 250.00 W
Default Power Limit    : 250.00 W
Min Power Limit        : 100.00 W
Max Power Limit        : 300.00 W
```

Pick a limit based on your cooling — this is a **thermal** decision, not a performance one:

| Your cooling | Suggested limit |
|---|---|
| No fan yet / passive only | **100 W** (the minimum) — and do not run real workloads |
| One fan | 150 W, watch temperatures closely |
| Two fans, properly ducted | 200–250 W |

Set it (replace `150` with your figure):

```bash
sudo nvidia-smi -pl 150
```

### Make it stick across restarts

The setting above is forgotten on reboot. To make it permanent, create a small service:

```bash
sudo tee /etc/systemd/system/cmp-powerlimit.service > /dev/null <<'EOF'
[Unit]
Description=CMP 170HX power limit
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -i 0 -pl 150

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now cmp-powerlimit.service
```

Edit the `150` in that block before running it if you chose a different number.

> ### ⚠️ The `-i 0` matters — a trap worth understanding
>
> Note `-i 0` in the command. That means "GPU number 0 only".
>
> **Without `-i`, `nvidia-smi -pl` applies to every NVIDIA GPU in the machine.** On the system
> this guide was written from, an old power-limit setting written for a completely different
> graphics card was silently pinning the CMP 170HX at its 100 W *minimum* — and because
> 100 W is a plausible-looking number, nothing appeared wrong. It was only found by comparing
> the applied limit against the card's own minimum.
>
> If you have more than one GPU, always use `-i` and check each card's limit individually:
> ```bash
> nvidia-smi --query-gpu=index,name,power.limit --format=csv
> ```

Confirm:

```bash
nvidia-smi --query-gpu=power.limit --format=csv
```

---

## Keeping it working (important)

### 🔴 Every kernel update breaks this

Normally Ubuntu automatically rebuilds graphics drivers when the kernel updates, using a
system called DKMS. **The unlock installer removes the NVIDIA driver from DKMS**, because
DKMS would rebuild an unpatched (locked) driver and undo your work.

The consequence: **after any update that installs a new kernel, your NVIDIA driver will stop
working entirely** until you re-run the installer.

After any `sudo apt upgrade`, check whether the kernel changed:

```bash
uname -r
```

If that number differs from before, run:

```bash
cd ~/CMP-170HX-PROJECT/cmpunlocker
sudo ./install.sh --profile=8gb
sudo shutdown -h now      # cold power cycle again
```

A useful habit is to note your kernel version before upgrading so you can tell.

### Set a reminder for yourself

```bash
echo 'echo "⚠️  CMP 170HX: re-run cmpunlocker install.sh after any kernel upgrade"' >> ~/.bashrc
```

This prints a reminder every time you open a terminal. Remove the line from `~/.bashrc` when
it gets annoying.

---

## Troubleshooting

### Scary messages that are actually fine

These appear in `dmesg` on **every successful unlock**. They look like failures. They are not:

| Message | Why it is fine |
|---|---|
| `Booter failed with non-zero error code: 0x31` | Expected during the privilege-unlocking sequence. Appears many times. |
| `kgspExecuteBooterLoad: failed to execute Booter Load: 0xffff` | Same sequence, same story. |
| `dmem.bin not found (0x59)` | That file is not published anywhere; the driver uses a built-in copy. |
| `warning: unused variable 'devId'` during install | A harmless compiler warning from the patch itself. |
| `objtool: 'naked' return found` during install | Normal for all NVIDIA drivers, unrelated to the unlock. |

**Judge success only by the final memory size**, not by these messages.

### Real problems

| Symptom | Cause and fix |
|---|---|
| `Installed driver is X, but cmpunlocker requires one of: ...` | Wrong driver version. Redo Step 4 — and make sure you did the restart in Step 5. |
| Still shows 8192 MiB after everything | Most likely you did a *restart* instead of a full *power off* in Step 7. Try again with a genuine power-down. |
| `nvidia-smi: command not found` after installing | The install failed partway. Re-run `sudo ./install.sh`, read the errors near the end. |
| `Failed to initialize NVML: Driver/library version mismatch` | Completely normal *between* Step 4 and Step 5. It means you have not restarted yet. Restart. |
| PCIe stuck at Gen1 | Check IOMMU is enabled in your BIOS (often called *VT-d*, *AMD-Vi*, or *SVM*). |
| Computer will not boot after Step 7 | At the GRUB menu press `e`, add `systemd.mask=gen2.service` to the line starting `linux`, press `Ctrl+X` to boot once. That disables the PCIe speed service. |
| `Xid 119` in dmesg | GPU startup timeout. Full power off — a restart will not clear it. |
| `Xid 31` while running a workload | Something tried to use more memory than the card really has. |
| `Xid 45` / crashes in AI workloads | In vLLM, keep `gpu_memory_utilization` at 0.90 or below. Do not assume every last megabyte is usable. |
| Card gets extremely hot / system freezes | **Stop immediately.** This is the thermal runaway warned about in Section 2. Power off, sort out cooling. |

### Getting help

Bring: your Ubuntu version, `uname -r`, `nvidia-smi` output, `sudo dmesg | grep SEC2_DEBUG`,
and the log from `cmpunlocker/logs/`. The project's Discord has an `#issue-support` channel.

---

## Undoing everything

To return the card to stock and restore a normal driver:

```bash
cd ~/CMP-170HX-PROJECT/cmpunlocker
sudo ./remove.sh --yes
sudo shutdown -h now
```

This removes the patched driver files, restores the standard NVIDIA modules and DKMS, removes
the PCIe Gen2 service, and undoes the boot settings.

**Note the script is `remove.sh`** — the README calls it `uninstall.sh`, which does not exist.

Do this if you are selling the card, moving it to another machine, or want normal automatic
driver updates back.

---

## Quick reference

Once you understand the steps, the whole procedure is:

```bash
# 1. Identify
lspci -nn | grep -i nvidia                      # note 20c2 (8GB) or 2082 (10GB)

# 2. Prerequisites
mokutil --sb-state                              # must say: disabled
sudo apt install -y build-essential linux-headers-$(uname -r) python3 curl git patch

# 3. Get tool
git clone https://github.com/amoghmunikote/cmpunlocker.git
cd cmpunlocker && cat driver/VERSION             # note supported driver versions

# 4. Driver
sudo apt install -y nvidia-driver-610-open

# 5. RESTART  <-- mandatory
sudo reboot

# 6. Unlock
sudo ./install.sh --profile=8gb

# 7. COLD POWER CYCLE  <-- must be power off, not restart
sudo shutdown -h now

# 8. Verify
nvidia-smi                                       # expect 65536MiB
sudo ./verify.sh

# 9. Power limit (match your cooling!)
sudo nvidia-smi -i 0 -pl 150
```

**The three things that catch people out:**

1. The driver version must be an exact match from `driver/VERSION`
2. You must restart between Step 4 and Step 6
3. Step 7 must be a full power-off, not a restart

---

## What this does not do

For completeness — these are permanently unavailable, so there is nothing more to try:

| | Status |
|---|---|
| PCIe x16 width | Possible, but requires soldering 24 capacitors onto the board |
| PCIe Gen3 / Gen4 | Blocked by a hardware fuse. No known method. |
| NVLink | Fuse-disabled. No known method. |
| ECC memory | Fused off. The control register is read-only. |
| Peer-to-peer GPU transfers | True P2P (the `aikitoria` patch) still attempted by the community, unresolved — this project hasn't tried it either (see `P2P-ENABLEMENT-PROJECT-PLAN.md`, Phases 0/1, not started). But you don't need it just to run tensor-parallel: **unpatched vLLM `--tensor-parallel-size 2` works fine without P2P** — tested 2026-08-13, completed cleanly at 146.29 tok/s with no hang, no NCCL error, on this project's own two cards (see `CARD-REGISTRY.md`). It's simply not a throughput win over running each card independently — expect roughly the same or a little slower than solo, not faster, since traffic falls back to staging through host RAM. |
| 80 GB on a 10 GB card | Tried and abandoned — reports the size, then fails above ~40 GB |

---

*Written from a verified working unlock on Ubuntu 24.04, kernel 7.0.0-28-generic, driver
610.43.02, cmpunlocker commit `360acd7`, on an 8 GB card (`10de:20c2`) that unlocked to
65536 MiB with PCIe Gen2 and passed a 56 GiB memory verification.*

*For what had to be downloaded from the internet and how to do all of this offline, see
[ONLINE-SOURCES.md](ONLINE-SOURCES.md).*
