# Online sources & offline archive — CMP 170HX unlock

Everything that had to be fetched from the internet to complete the unlock on
**2026-08-06**, archived locally and documented here.

Two reasons this exists:

1. **Offline rebuild.** This machine is meant to work fully offline. Everything needed to
   redo the unlock from scratch — including after a kernel upgrade, which *will* break the
   driver — is now on local disk.
2. **Upstream disappearance.** NVIDIA has issued a DMCA takedown against at least one fork
   of cmpunlocker. The git bundles below carry full history for both repos, so the tooling
   survives the upstream repos going away.

Archive lives in `offline-archive/`. **348 MB.** Integrity: `offline-archive/SHA256SUMS`
(19 files).

```bash
cd /home/user/CMP-170HX-PROJECT/offline-archive && sha256sum -c SHA256SUMS
```

---

## 1. What was downloaded

### `offline-archive/source/open-gpu-kernel-modules-610.43.02.tar.gz` (27 MB)

NVIDIA's open kernel module source — the tree cmpunlocker's nine patches are applied to.
Fetched automatically by `driver/build.sh` from
`https://github.com/NVIDIA/open-gpu-kernel-modules/archive/refs/tags/610.43.02.tar.gz`.

> **This is the one that would otherwise be lost.** `cmpunlocker/.gitignore` excludes both
> `driver/.build/` and `*.tar.gz`, so it is *not* tracked in git and a fresh clone would
> re-download it. Without network access, `build.sh` fails here.

`sha256: 62fbbe29527e30be32cb38b30dfad2e94db1ca87f77a58090e563c7669857e60`

To reuse instead of re-downloading — `build.sh` looks for it at
`${SCRIPT_DIR}/.build/${SRC_NAME}.tar.gz`:

```bash
mkdir -p /home/user/CMP-170HX-PROJECT/cmpunlocker/driver/.build
cp offline-archive/source/open-gpu-kernel-modules-610.43.02.tar.gz \
   /home/user/CMP-170HX-PROJECT/cmpunlocker/driver/.build/
```

### `offline-archive/nvidia-driver-610.43.02-debs/` (320 MB, 16 packages)

The exact Ubuntu packages installed, from `noble-security/multiverse`:

```
libnvidia-cfg1-610          libnvidia-common-610       libnvidia-compute-610
libnvidia-decode-610        libnvidia-encode-610       libnvidia-extra-610
libnvidia-fbc1-610          libnvidia-gl-610           nvidia-compute-utils-610
nvidia-dkms-610-open        nvidia-driver-610-open     nvidia-firmware-610-610.43.02
nvidia-kernel-common-610    nvidia-kernel-source-610-open
nvidia-utils-610            xserver-xorg-video-nvidia-610
```

All `610.43.02-0ubuntu0.24.04.1_amd64`. Offline reinstall:

```bash
sudo apt install ./offline-archive/nvidia-driver-610.43.02-debs/*.deb
```

### `offline-archive/git-bundles/` (full history, self-contained)

| Bundle | Repo | Commit | Refs |
|---|---|---|---|
| `cmpunlocker-360acd7.bundle` | [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) | `360acd7` | 26 |
| `cmp170hx-wiki.bundle` | [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) | `879454d` (2026-07-31) | 4 |

The cmpunlocker bundle's 26 refs include the experimental branches the wiki discusses
(`Gen2`, `far`, `deced`, `debug-gen2`, `ecc`, `80`, `JTAG`, `clanker/driver-port`), not just
`master`. Restore with:

```bash
git clone offline-archive/git-bundles/cmpunlocker-360acd7.bundle cmpunlocker
```

Both verified with `git bundle verify`.

---

## 2. Information retrieved online

Facts that were **not** in the project folder and had to be looked up. Each is what actually
changed a decision.

### The driver version was the entire blocker

`cmpunlocker/driver/VERSION` accepts only **610.43.03 / 610.43.02** (upstream `master` has
since added **610.57.04**). The machine was on **595.71.05**, which `install.sh` rejects
outright. Verified online:

- 610.43.02 released May 2026; **610.43.03** released 2026-07-07, changelog "minor bug fixes
  and improvements" — [NVIDIA releases](https://github.com/NVIDIA/open-gpu-kernel-modules/releases), [GamingOnLinux](https://www.gamingonlinux.com/2026/07/nvidia-610-43-03-driver-released-for-linux-with-a-vague-changelog/)
- **The decisive find:** `nvidia-driver-610-open` **610.43.02** is packaged in Ubuntu
  `noble-security/multiverse`. No `.run` file, no manual build — plain `apt`. This turned the
  blocker into one command.

The alternative — the wiki's `clanker/driver-port` branch backporting the patches to 595 —
was rejected: the status board rates it *"Experimental. Source-verified, never boot-tested."*

### Upstream cmpunlocker had moved on

Local clone was at `f902c44`, upstream at `360acd7` (3 commits). New: JTAG support
(`74ca2ef`), 610.57.04 added to `VERSION`, and all nine patch files renamed to drop their
numeric prefixes (`0001-sec2-postbl-plm-ss-cfg.patch` → `sec2-postbl-plm-ss-cfg.patch`).
**Any doc referring to patches by number is now stale**, including `TUTORIAL.md`.

### Motherboard slot capacity

[ASRock Rack AM5D4ID2](https://www.asrockrack.com/general/productdetail.asp?Model=AM5D4ID2) —
deep mini-ITX: **1 × PCIe 5.0 x16, 2 × M.2 (PCIe 5.0 x4)**. Confirmed against live `lspci`
(root port `00:01.1` cap x16; `00:03.2` x4 holding the boot NVMe).

Consequence: **at most two GPUs**, the second necessarily on an M.2 → PCIe x4 riser. The CMP
170HX plus both RTX PRO 4000s cannot coexist in this board.

### Per-device driver registry keys

`install.sh` writes a **driver-wide** setting to `/etc/modprobe.d/cmp-pcie-gen2.conf`:

```
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x1"
```

Its purpose is to stop the RM re-clamping the link to Gen1 after each retrain — but the
scope is global, so it would also apply to any non-CMP NVIDIA card added later.

`NVreg_RegistryDwordsPerDevice` (confirmed present in this module via `modinfo`, documented
in [`nv-reg.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia/nv-reg.h))
scopes keys per PCI address:

```
NVreg_RegistryDwordsPerDevice="pci=DDDD:BB:DD.F;<key=value>;<key=value>; pci=...;<key=value>"
```

If the PCI address matches no present GPU, the following keys are **silently skipped** — so
a wrong BDF fails safe (card drops to Gen1) rather than misapplying.

> **Not community practice.** `PerDevice` appears **zero times** in the 55-page wiki, and
> every documented CMP LLM rig is an all-CMP machine where global and per-device are
> identical. Worth knowing that essentially all published CMP inference benchmarking is at
> **Gen1 x4** — Gen2 only merged to `master` on 2026-07-29. This is untested territory, not a
> settled recipe.

---

## 3. Not downloaded, and why

- **`/lib/firmware/nvidia/ga100/gsp/dmem.bin`** — dmesg reports it missing (status `0x59`)
  and falls back to a built-in payload. Expected and benign per the troubleshooting table; no
  such file is published or needed.
- **VBIOS images** — the unlock writes only volatile registers. Nothing is flashed, so no
  VBIOS was fetched. (Card reads `92.00.6D.00.0A`.)
- **Search-result noise** — GPU spec-aggregator sites, eBay listings and secondhand blog
  summaries were read but not used; several carry figures contradicted by the primary
  teardown and wiki sources. See `REFERENCES.md` for the same exclusion note.

---

## Related files

- `TUTORIAL.md` — pre-arrival walkthrough (⚠️ patch filenames now stale, see above)
- `REFERENCES.md` — research links from the 2026-08-04 planning pass
- `cmpunlocker/logs/install_20260806_113438.log` — full install log
- `cmp170hx/` — community wiki clone (also bundled)

Untouched by this pass: `cmp-170x-performance-benchmarks.md`, which is a deliberately frozen
pre-hardware prediction kept diffable against real measurements.
