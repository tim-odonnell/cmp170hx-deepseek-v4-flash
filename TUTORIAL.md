# cmpunlocker Tutorial — NVIDIA CMP 170HX

Step-by-step guide for when your CMP 170HX card(s) arrive. Based on the actual
scripts in `cmpunlocker/` (install.sh, remove.sh, verify.sh, driver/build.sh),
re-verified 2026-08-04 against local clone commit `f902c44` (confirmed
up to date with `origin/master`, no newer upstream commits), cross-checked
against the community wiki at
[Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx)
(55-page technical reference for this card, last synced 2026-07-31 — slightly
behind this repo's `f902c44`) and the project's own
[170th-street](https://github.com/amoghmunikote/170th-street) docs. Where the
two disagreed, I verified against the actual code in this clone — noted below.

## What this does

The CMP 170HX is physically a full GA100 die (same silicon as the A100), but
NVIDIA firmware-locks it to restrict compute (disables ~50% of SMs) and memory
geometry (reports 8GB/10GB instead of the die's real 64GB/40GB). cmpunlocker
patches the `nvidia-open` kernel modules to reconfigure the GPU during driver
boot, before those locks take effect. No physical hardware modification.

## Risks — read before powering it on

- **Bricking risk is near-zero from the software unlock itself.** It only
  writes volatile GPU registers — no VBIOS/EEPROM writes, no fuse blows. A
  cold power cycle recovers from essentially any failed unlock attempt.
- **Thermal risk is the real danger.** This is a 250W card built for a
  high-RPM server chassis' forced airflow, shipped passively cooled. Without
  active airflow arranged *before* first power-on, the GA100 die can hit
  genuine leakage-driven thermal runaway (higher temp → higher power draw →
  higher temp). One 40mm fan is not enough — plan on two. Standard FP32
  benchmarks won't stress it enough to validate cooling; use a memory-heavy
  workload.
- **Power connector**: uses an EPS 8-pin, physically similar to but distinct
  from PCIe 8-pin — pin assignments are swapped. Forcing the wrong 8-pin
  connector in **will** damage the card. Use the adapter/cable rated for EPS.
- **Warranty/legal**: NVIDIA issued a DMCA takedown against at least one fork
  of this tool in 2026. The tool itself only uses NVIDIA's own open-source
  modules plus published keys — but running unsigned kernel modules voids any
  remaining seller warranty (moot on most secondhand ex-mining cards anyway).
- **Only use the documented profiles.** An 80GB geometry was tested on 10GB
  cards and explicitly rejected as unstable above ~40GB usable. Stick to
  `8gb→64GB` / `10gb→40GB` — don't try to push past what `install.sh` offers.
- **No display output** — the 170HX can't drive a monitor. It's a pure
  compute add-in card; your existing RTX Pro 4000 stays the display GPU.

## 1. Before the card arrives — prep checklist

- **OS**: Linux x86-64 (Ubuntu/Debian/Fedora all fine)
- **Driver**: install `nvidia-open` **610.43.02 or 610.43.03** specifically —
  other versions have different boot paths and won't patch correctly
- **Kernel headers** matching your running kernel:
  `sudo apt install linux-headers-$(uname -r)` (Debian/Ubuntu) or
  `kernel-devel` (Fedora)
- **Secure Boot: disabled** in BIOS/UEFI — the patched kernel modules are
  unsigned and won't load with Secure Boot on
- **Python 3** installed (used at install time to detect card variant)
- **Build toolchain**: `patch`, `make`, `curl`, and a working kernel build
  environment (`build-essential` on Debian/Ubuntu, `gcc`/`make` group on
  Fedora) — `driver/build.sh` calls all three directly to fetch and compile
  the patched modules
- **Network access** for the first install (it pulls matching stock
  `open-gpu-kernel-modules` source to patch against)
- **Cooling arranged and powered before first boot** — see Risks above
- **EPS 8-pin power cable** (not PCIe 8-pin) rated for the card

## 2. Physical install

1. Power off, seat the CMP 170HX with EPS power connected, cooling running, power on.
2. Confirm the card is visible: `lspci -nn | grep -iE '10de:20c2|10de:2082|10de:20b0'`
   - `10de:20c2` = 8GB card (unlocks to 64GB)
   - `10de:2082` = 10GB card (unlocks to 40GB)
   - `10de:20b0` = a related GA100 variant that the tool detects but does
     **not** unlock — if this is what shows up, cmpunlocker won't help
3. Confirm `nvidia-smi` sees it and reports the stock 8GB/10GB size before
   you do anything else — that confirms the driver itself is working first.

## 3. Run the installer

From `/home/user/CMP-170HX-PROJECT/cmpunlocker`:

```bash
sudo ./install.sh
```

The installer auto-detects 8GB vs 10GB per-card via PCI ID. For a single
CMP 170HX this is fine as-is. If you'll have **more than one GPU in the
box** (e.g. a second CMP 170HX, or any other NVIDIA card alongside it),
pass the profile explicitly rather than relying on auto-detect:

```bash
sudo ./install.sh --profile=8gb    # force label, geometry is still per-PCI-ID
sudo ./install.sh --profile=10gb
```

Why: the community wiki documents a real bug class where auto-detection on
a mixed-GPU host samples whichever card `lspci`/`nvidia-smi` lists first —
e.g. a non-170HX card reporting "10GB" gets mistaken for the label source.
The actual per-card memory geometry is still chosen correctly by PCI device
ID regardless (only the stored `card_profile` metadata label is at risk), but
there's no reason to rely on that when `--profile` removes the ambiguity
entirely. After install, run `verify.sh` (step 5) and check its per-GPU table
rather than trusting the install log alone.

Other flags (`sudo ./install.sh --help` for the full list):
- `--no-iommu` — skip appending `intel_iommu=on`/`amd_iommu=on iommu=pt` to
  the kernel command line (installer does this by default, needed for PCIe
  Gen2 to negotiate correctly)
- `--no-gen2-service` — skip installing the early-boot PCIe Gen2 retrain
  service

What it does: verifies the installed driver version and kernel headers, then
**removes any conflicting NVIDIA DKMS modules** (`dkms remove nvidia/<ver>
--all` for each supported driver version — this is a new step as of the
`f902c44` commit, not mentioned in the README), builds patched
`nvidia.ko`/`nvidia-drm.ko` against your kernel headers, installs them to
`/lib/modules/$(uname -r)/updates/cmpunlocker/` (this path loads *before*
the stock modules), and stores the detected card profile in
`.../cmpunlocker/card_profile` for use on every future boot.

**Important operational consequence**: because the installer strips the DKMS
registration, a kernel upgrade will no longer auto-rebuild a working NVIDIA
module the way DKMS normally would. The installer prints this explicitly at
the end of its run: *"This script removed the nvidia DKMS kernel modules.
You will need to re-run this script after each kernel upgrade."* Put a
reminder somewhere you'll see it before running `apt upgrade` on this box.

Installer logs land in `cmpunlocker/logs/install_<timestamp>.log` if you need
to debug or paste output into a support ticket.

## 4. Cold reboot

Not a regular `reboot` — do a **full power off, then power back on**. This
matters because the unlock sequence runs during GPU power-on (GSP boot), and
a warm reboot can leave residual GPU state.

## 5. Verify the unlock

```bash
sudo ./verify.sh
```

This checks each unlockable GPU's actual memory via `nvidia-smi` against
the expected unlocked size (65536 MiB for 8GB cards, 40960 MiB for 10GB
cards), checks `dmesg` for `SEC2_DEBUG` unlock log lines, and checks
negotiated PCIe generation. Prints a per-GPU OK/STOCK/FAIL table.

The installer itself prints a fuller post-reboot checklist at the end of its
run — worth running all of these, not just `verify.sh`, on first install:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv                          # memory
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max --format=csv    # expect 2,2
sudo dmesg | grep SEC2_DEBUG                                                    # unlock logs
cat /proc/cmdline                                                               # confirm IOMMU flags applied
ls /sys/class/iommu                                                             # confirm IOMMU is actually active
sudo ./tools/service.sh verify                                                  # negotiated Gen2 check (if Gen2 service installed)
```

Memory check should show `65536 MiB` (8GB card) or `40960 MiB` (10GB card)
instead of the stock `8192`/`10240`. If PCIe Gen2 negotiation is causing boot
problems, there's a recovery kernel-command-line option:
`systemd.mask=gen2.service` to skip the Gen2 retrain service for that boot.

## 6. If something's wrong

From `docs/DEBUGGING.md`, plus extra cases documented in the community wiki's
troubleshooting page:

| Symptom | Fix |
|---|---|
| `nvidia-smi: command not found` | Installer likely failed. Re-run `sudo ./install.sh`, cold reboot. |
| `nvidia-smi` still shows stock 8192/10240 MiB | Check all PLMs show `0xffffffff`: `sudo dmesg \| grep SEC2_DEBUG`. Also check patched modules actually loaded (not the stock ones) and that initramfs was rebuilt, not just installed to `updates/`. |
| PCIe stuck at Gen1 | Confirm IOMMU passthrough mode is actually enabled (varies by distro/BIOS) |
| Booter status `0x31` during early PLM passes, or `0x59` for missing `dmem.bin` | Usually benign — these look like failures but are expected mid-sequence; only worry if the *final* boot state / memory size is wrong |
| Xid 119 in dmesg | GSP boot timeout — cold power cycle, don't just warm reboot |
| Xid 31 (MMU fault) during workloads | Something over-allocated past the unlocked memory ceiling — check the workload's memory request against the actual unlocked size |
| Xid 45/154 (CUDA context corruption), vLLM crashes | Keep `gpu_memory_utilization` ≤ 0.90 in vLLM/inference configs on unlocked cards; don't assume the full reported VRAM is safely usable to the last MB |
| Nothing works after `rmmod nvidia` + manual reload attempts | `rmmod` clears the PCI bus-master bit; a stale module unload can silently break DMA. Fully unload all four modules (`nvidia_uvm`, `nvidia_drm`, `nvidia_modeset`, `nvidia`) before reloading, or just reboot. |

If still stuck, their Discord has a `#issue-support` channel — bring your OS
version, GPU model/driver version, `sudo dmesg | grep SEC2_DEBUG` output, and
the latest install log.

## 7. Uninstalling

The README says `uninstall.sh` but the actual script in this repo is
**`remove.sh`**:

```bash
sudo ./remove.sh --yes
```

This stops the systemd service, removes the patched modules from
`/lib/modules/*/updates/cmpunlocker/`, reloads stock NVIDIA modules (brief
display flicker if you're on this GPU for display output), removes the PCIe
Gen2 helper service, and reverts the IOMMU kernel command-line changes from
its backup. Cold reboot afterward to fully finish cleanup.

## Notes for this system specifically

- This unlock only applies to the CMP 170HX cards themselves — it doesn't
  touch or interact with the RTX Pro 4000 already in this workstation.
- Confirm CUDA 12.8 pin still holds after installing `nvidia-open`
  610.43.0x for the CMP cards — check driver/CUDA compatibility before
  assuming the existing stack is unaffected.
- The 170HX has no display output and is add-in compute only, so it won't
  conflict with the RTX Pro 4000's role as display GPU — but it will need
  its own PCIe slot + EPS power + independent cooling airflow in the case.
- Since the installer disables DKMS for the NVIDIA driver on this system,
  any routine `apt upgrade` that pulls a new kernel will leave the CMP 170HX
  (and possibly the RTX Pro 4000, if they share a driver install) without a
  working module until you manually re-run `sudo ./install.sh`. Worth
  checking `uname -r` before/after kernel-upgrading this box.

## Cross-check notes

Verified against the community wiki
([Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx)),
which is a much deeper (55-page) reference than this repo's own README/docs.
Two things worth flagging:

- The wiki's install page describes the *shipping* version as **not**
  including PCIe Gen2 support and multi-GPU support as still living on an
  unmerged branch. The actual code in this clone already has both (the
  `--no-gen2-service` flag, `tools/service.sh`, and per-GPU `gpu_inventory`
  in `verify.sh`) — so this repo is ahead of what the wiki documents. I went
  with what's actually in the code you have.
- Everything else (driver version pin, cold-reboot requirement, profile
  flags, expected MiB values, PCI IDs, risk/thermal/power guidance) matched
  between the wiki and this repo's own scripts, so that part is solid.
