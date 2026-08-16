# Transferring this project to another machine

Written 2026-08-13. Read this **before** copying the folder, then follow it on the
receiving machine.

This file is a transfer overlay only. It does not replace
[`HOWTO-UNLOCK-CMP-170HX.md`](HOWTO-UNLOCK-CMP-170HX.md), which remains the actual
step-by-step unlock procedure. This file tells you what travels, what does not, and what
to do differently because you are on new hardware.

---

## 1. What to copy

```bash
rsync -av --exclude='cmpunlocker/driver/.build/' \
  /home/user/CMP-170HX-PROJECT/ /path/to/destination/CMP-170HX-PROJECT/
```

**Always exclude `cmpunlocker/driver/.build/`.** Three independent reasons:

- It is 580 MB of regenerable compiler output — `driver/build.sh` recreates it.
- It was compiled against one specific kernel (`7.0.0-28-generic` on the source machine)
  and is useless on any other kernel. The new machine must rebuild regardless.
- It contains the only four symlinks and 9,112 of the 9,113 root-owned files in the
  project. Those are what make copies to exFAT/NTFS external drives fail with symlink
  and permission errors.

Excluding it takes the transfer from ~1.9 GB to ~1.3 GB.

For the four symlinks specifically: they are NVIDIA's standard build layout, linking
`kernel-open/nvidia/nv-kernel.o{,_binary}` and
`kernel-open/nvidia-modeset/nv-modeset-kernel.o{,_binary}` to the compiled cores under
`src/…/_out/Linux_x86_64/`. Two are absolute paths back into `/home/user/…`, so they
would be dead on arrival even on a filesystem that could store them.

### If the destination drive is exFAT, NTFS or FAT

Use a tarball instead — the drive then only ever sees one ordinary file, and ownership,
permissions and any symlinks survive intact:

```bash
sudo tar -czf /path/to/drive/CMP-170HX-PROJECT-$(date +%Y%m%d).tar.gz \
  --exclude='CMP-170HX-PROJECT/cmpunlocker/driver/.build' \
  -C /home/user CMP-170HX-PROJECT
```

### Two gotchas seen on an SMB transfer (2026-08-13)

Copying this folder in over `smb://`/gvfs (Nautilus "Connect to Server", not `rsync`/`tar`)
produced two things worth checking on arrival, neither of them data loss:

- **Every `.sh` script loses its executable bit.** Content is untouched (`git diff` showed
  `0 insertions, 0 deletions`, only a `100755` → `100644` mode change), but
  `sudo ./install.sh` etc. will fail with `Permission denied` until you restore it:
  ```bash
  find ~/CMP-170HX-PROJECT -name '*.sh' -not -path '*/.git/*' -exec chmod +x {} \;
  ```
- **Installing the offline `.deb`s throws a scary but harmless apt warning** if your home
  directory is `750` (`drwxr-x---`, the Ubuntu default): `_apt`'s sandbox user can't
  traverse into `/home/user` to read the file, so apt falls back to "unsandboxed as root"
  for that package and prints an `N:` notice. The install still succeeds — confirm with
  `dpkg -l | grep nvidia-firmware` and check `/lib/firmware/nvidia/<version>/` has non-zero
  `gsp_*.bin` files before assuming the warning means something failed.

---

## 2. What is NOT in this folder

The unlock stack is self-contained. The **inference** stack is not. Every `bench-*.sh`
and `run-*.sh` script here will fail immediately on a new machine without these:

| Needed | Size | Why |
|---|---|---|
| `/home/user/llama.cpp-portable` | 3.2 GB | 28 scripts here call `build-cuda-sm80/bin/llama-server` |
| `/home/user/ai-models/…` | 1.1 TB total | DeepSeek-V4-Flash, Qwen3.6-35B-A3B-MTP, Qwen3.6-27B, GLM-5.2 |
| `/home/user/cowboy-clue` | — | 25 refs from the Cowboy Clue benchmark runners |

⚠️ **`build-cuda-sm80` is not optional and not interchangeable.** The sibling
`build-cuda` tree targets sm_120 and will not run on these Turing GPUs (compute
capability 8.0). If you rebuild on the new machine rather than copying, you must target
sm_80.

Copy the engine alongside the project:

```bash
rsync -av /home/user/llama.cpp-portable/ /path/to/destination/llama.cpp-portable/
```

Models are the 1.1 TB question — bring a subset, or re-download on the far side.

### Third-party tools are NOT archived

`offline-archive/` contains the NVIDIA driver material and this project's own two git
bundles — nothing else. None of the community tooling catalogued in
`cmp170hx/docs/appendix/external-sources.md` is stored locally, so fetching any of it
needs network:

- `cachenetics/170tune` — installs as `/usr/local/bin/170hx-oc`. **Never installed on
  the source machine.** `cmp170hx/docs/operations/tuning.md` documents its usage
  (`170hx-oc stock`, `oc_eff`), but it is a surveyed third-party tool, not this
  project's own script. Do not go looking for it in this folder; it was never here.
  Actual power tuning in this project was done with the in-folder `power-sweep-llm.sh`
  and `power-sweep-qwen35b-fine.sh`, which use `sudo nvidia-smi -pl` directly.
- `eastmoe/CMPGPU-patch-script` (`optimize-cmp-cuda.py`) — llama.cpp FMA/intrinsics
  patcher.
- `theneocorp/cmppatcher` — binary-patching alternative approach.
- `Kepling5001/Miners` — already deleted upstream; unrecoverable if you ever want it.

---

## 3. Bring-up order on the new machine

Follow `HOWTO-UNLOCK-CMP-170HX.md` for the detail. The order, condensed:

1. **Prerequisites** (HOWTO Step 2) — Secure Boot **off**, note `uname -r`, install build
   tools, confirm the cards enumerate before changing anything.
2. **Driver** (HOWTO Step 4) — must be a version listed in `cmpunlocker/driver/VERSION`
   (610.43.02 / 610.43.03 / 610.57.04) and must be the **open** variant.
   - Only **610.43.02** has been installed and verified end-to-end against this hardware.
     Treat the other two as staged, not proven.
   - Offline: `sudo apt install ./offline-archive/nvidia-driver-610.43.02-debs/*.deb`
   - `.run` fallback: `sudo sh NVIDIA-Linux-x86_64-<ver>.run -m=kernel-open`
     (the `-m=kernel-open` flag is mandatory).
3. **Reboot** (HOWTO Step 5) — do not skip.
4. **Unlock** (HOWTO Step 6) — `cd cmpunlocker && sudo ./install.sh`
   - All four cards in this project are `10de:20c2` → 64 GB unlock → `8gb` profile.
     `install.sh` auto-classifies by PCI ID, so no `--profile` flag is needed; pass
     `--profile=8gb` only to force it.
   - By default the installer also appends `amd_iommu=on`/`intel_iommu=on` plus
     `iommu=pt` to the kernel command line, and installs the early-boot PCIe Gen2 retrain
     service. Use `--no-iommu` / `--no-gen2-service` to suppress either.
5. **Cold power cycle** (HOWTO Step 7) — a full power-off, *not* a reboot.
6. **Verify** (HOWTO Step 8) — `sudo ./verify.sh`, expect 65536 MiB per card.
7. **Memory integrity** (HOWTO Step 9) — recommended on new hardware; the source machine
   recorded 3 clean sweeps per card.
8. **Power limit** (HOWTO Step 10).

### Offline staging — do not skip

`driver/build.sh` will `curl` the source tarball from GitHub if it is not already
cached. On an offline machine, stage it first (see `ONLINE-SOURCES.md:43`):

```bash
mkdir -p cmpunlocker/driver/.build
cp offline-archive/source/open-gpu-kernel-modules-610.43.02.tar.gz \
   cmpunlocker/driver/.build/
```

Verify the archive survived the copy before you rely on it:

```bash
cd offline-archive && sha256sum -c SHA256SUMS
```

### Nothing needs hand-copying from system directories

Checked on the source machine: the only installed artifact is
`/usr/local/sbin/gen2-hammer`, byte-identical (md5 `b4d03c30705e9acf5696a0a7780e6034`)
to `cmpunlocker/tools/hammer.sh`, plus `/etc/systemd/system/gen2.service` from
`cmpunlocker/systemd/gen2.service`. `install.sh` recreates both.

References you may notice to `/opt/cmpunlocker/daemon/watchdog.py`,
`/usr/local/sbin/retrain.sh` and `cmp-gen2-retrain.sh` appear **only in removal code
paths**. The docs describe them as vestiges of an abandoned design. They do not exist on
the source machine and are not needed.

---

## 4. Card inventory

Four cards, all identical silicon — `10de:20c2` / subsys `10de:1585`, board
`900-11001-0108-000`, compute capability 8.0, VBIOS `92.00.6D.00.0A`, unlocking to
65536 MiB each:

| Serial | Record |
|---|---|
| `1322321009409` | `card-records/card_1322321009409_20260806.md` |
| `1322421002704` | `card-records/card_1322421002704_20260806.md` |
| `1322421041391` | `card-records/card_1322421041391_20260807.md` |
| `1322621127916` | `card-records/card_1322621127916_20260807.md` |

Full per-card detail — clock headroom, memory sweeps, compute figures — is in those
records and `CARD-REGISTRY.md`. Re-capture on the new machine with `capture-card.sh`.

For a new card's own Step 9 (56 GiB memory-integrity proof), note that
`verify-memory-gpu0.py` / `verify-memory-gpu1.py` hardcode `cuDeviceGet(..., 0)` /
`(..., 1)` -- they are not auto-discovering like the systemd services are. For a 3rd/4th
card, either copy one with the index changed, or parameterize it to take the GPU index as
an argument.

⚠️ **Clock offsets are per-die. Do not carry a validated offset from one card to
another, or from the old machine to the new one.** +300 is the highest upstream-validated
offset at a 1400 MHz ceiling; +325 silently corrupted memory upstream.

No VBIOS/ROM backups exist and none are needed — this unlock is driver-patch-based, not
a firmware flash.

---

## 5. Ongoing maintenance on the new machine

🔴 **Every kernel update breaks the unlock.** `install.sh` deliberately removes the
NVIDIA driver from DKMS, because DKMS would rebuild an unpatched, locked driver. After
any `apt upgrade` that installs a new kernel, the NVIDIA driver stops working entirely
until you re-run the installer:

```bash
uname -r                        # changed?
cd ~/CMP-170HX-PROJECT/cmpunlocker
sudo ./install.sh
sudo shutdown -h now            # cold power cycle again
```

Note your kernel version before upgrading so you can tell.

---

## 6. Post-transfer checklist

- [ ] `sha256sum -c SHA256SUMS` passes in `offline-archive/`
- [ ] `git -C cmpunlocker status` and `git -C cmp170hx status` both work (history intact)
- [ ] Source tarball staged into `cmpunlocker/driver/.build/` if offline
- [ ] Secure Boot off, kernel version noted
- [ ] Supported **open** driver installed
- [ ] `sudo ./install.sh` completed, then **cold power cycle**
- [ ] `sudo ./verify.sh` shows 65536 MiB on all cards
- [ ] All `.sh` scripts executable (`find . -name '*.sh' -not -perm -u+x` prints nothing —
      see the SMB gotcha above, this WILL be non-empty after an SMB/gvfs copy)
- [ ] `llama.cpp-portable/build-cuda-sm80/bin/llama-server` present and executable
- [ ] Model paths in `bench-*.sh` / `run-*.sh` updated if `ai-models` moved
- [ ] `gpu-fan-daemon.service` and `cmp-powerlimit.service` recreated if this machine wants
      persistent fan/power control (see section 7 below) -- these are NOT copied by any of
      the transfer commands above, since they live in `/etc/systemd/system/`, outside this
      folder

---

## 7. Persistent GPU services (fan control + power limit)

Two systemd services, added 2026-08-13, are **not part of the folder copy** -- they live in
`/etc/systemd/system/` and must be recreated on each new machine that wants them:

| File in this folder | Installs to | Does |
|---|---|---|
| `gpu-fan-daemon.sh` + `gpu-fan-daemon.service` | `/etc/systemd/system/gpu-fan-daemon.service` | Reads the hottest watched GPU/HBM temperature, drives chassis fans via the motherboard BMC (`ipmitool`) instead of the CPU-only stock curve. Restores BMC automatic mode on any stop/crash -- fails safe. |
| `set-power-limit.sh` + `cmp-powerlimit.service` | `/etc/systemd/system/cmp-powerlimit.service` | Applies a power limit (`POWER_LIMIT` env var, default 100 W) to every CMP 170HX found, on every boot. |

Both discover their target GPUs by **PCI device ID** (`DEVICE_IDS="20c2 2082"`), not a fixed
index list, so adding more cards on the same machine needs no edit -- just
`sudo systemctl restart gpu-fan-daemon.service cmp-powerlimit.service` (or the cold power
cycle a new-card install already requires).

Install on a new machine:
```bash
sudo cp gpu-fan-daemon.service cmp-powerlimit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-fan-daemon.service cmp-powerlimit.service
```

### Before trusting `gpu-fan-daemon.sh` on a *different* motherboard model

The fan-control half is the one piece of this project that is **not portable as-is**. It
talks to the board's BMC over IPMI, and the raw command bytes are specific to the ASPEED
chip generation, not just to "ASRock Rack" as a brand:

- **AST2500** (this board, ROMED8-2T): netfn `0x3a`, commands `0xd6`/`0xd7`/`0xd8`/`0xda`.
- **AST2600** (e.g. the AM5D4ID-2T): netfn `0x3a` cmd `0xd0` with subcommands
  (`0x11`/`0x0e`/`0x12`/`0x0f`).

Set `AST_GEN=ast2500` or `AST_GEN=ast2600` (default in the script is `ast2500`) to match.
**Do not assume either family works on a board you haven't confirmed** -- the wrong one
returns `Invalid data field in request` (harmless, but proves nothing was set) rather than a
clean error. Confirm read-only before ever calling a set command:
```bash
sudo ipmitool raw 0x3a 0xd7   # AST2500: get duty setpoints
sudo ipmitool raw 0x3a 0xda   # AST2500: get current duty
# or, for AST2600:
sudo ipmitool raw 0x3a 0xd0 0x12   # get mode
sudo ipmitool raw 0x3a 0xd0 0x0f   # get duty
```
16 clean hex bytes back (no `rsp=0xcc` error) confirms the command family is right for that
board. If neither works, check the actual chip via the board's own vendor FAQ before guessing
further raw bytes -- ASRock Rack publishes one (TSDQA-72) covering both generations, and other
vendors' BMC firmware may use a different command set entirely.
