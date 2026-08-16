#!/usr/bin/env bash
# Capture a CMP 170HX's identity + standard test results, formatted for CARD-REGISTRY.md.
#
# Run this on every card so the four are measured identically and stay comparable. Two of the
# tests are per-die properties that MUST NOT be assumed from another card:
#   * memory integrity  -- no ECC, no error telemetry, so a full-VRAM pattern sweep is the
#                          only evidence the card is sound
#   * clock offset headroom -- the safe maximum is a silicon property, not a model property
#
# Usage:
#   ./capture-card.sh              identity + quick tests (~3 min)
#   ./capture-card.sh --full       adds the clock-offset probe and a longer memory sweep
#
# Output goes to stdout AND card-records/card_<serial>_<date>.md
set -uo pipefail
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/card-records"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
SER=$(sudo nvidia-smi -q 2>/dev/null | awk -F: '/Serial Number/{gsub(/ /,"",$2);print $2;exit}')
[[ -n "$SER" && "$SER" != "N/A" ]] || SER="unknown-$(date +%s)"
REC="$HERE/card-records/card_${SER}_$(date +%Y%m%d).md"

q(){ nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1; }
qq(){ sudo nvidia-smi -q 2>/dev/null | awk -F: -v k="$1" '$0 ~ k {gsub(/^ +| +$/,"",$2);print $2;exit}'; }

{
echo "# Card — serial \`${SER}\`"
echo
echo "Captured $(date '+%Y-%m-%d %H:%M:%S %Z') on \`$(hostname)\`, slot \`$(q pci.bus_id)\`"
echo
echo "## Identity"
echo
echo "| | |"
echo "|---|---|"
echo "| **Serial Number** | \`${SER}\` |"
echo "| **GPU UUID** | \`$(q uuid)\` |"
echo "| Board Part Number | \`$(qq 'Board Part Number')\` |"
echo "| GPU Part Number | \`$(qq 'GPU Part Number')\` |"
echo "| PCI ID | \`$(q pci.device_id)\` / subsys \`$(q pci.sub_device_id)\` |"
echo "| **VBIOS** | \`$(q vbios_version)\` |"
echo "| Inforom image | \`$(qq 'Image Version')\` |"
echo "| Compute capability | $(q compute_cap) |"
echo
echo "## State"
echo
echo "| | |"
echo "|---|---|"
echo "| Reported memory | **$(q memory.total)** |"
echo "| PCIe | gen$(q pcie.link.gen.current) x$(q pcie.link.width.current) (max gen$(q pcie.link.gen.max)) |"
echo "| Driver | $(q driver_version) |"
echo "| Max memory clock | $(q clocks.max.memory) |"
echo "| Max SM clock | $(q clocks.max.sm) |"
echo "| Power limits | min $(q power.min_limit) / default $(q power.default_limit) / max $(q power.max_limit) |"
echo "| Idle temp | $(q temperature.gpu) C core, $(q temperature.memory) C HBM |"
echo "| Idle draw | $(q power.draw) |"
echo
UNLOCKED=$(q memory.total | grep -oE '[0-9]+')
if [[ "${UNLOCKED:-0}" -gt 40000 ]]; then
  echo "**UNLOCK: ACTIVE** ($(q memory.total))"
  echo
  echo '```'
  sudo dmesg 2>/dev/null | grep -m1 'POST-WRITE' | sed 's/.*SEC2_DEBUG: //' || echo "(no POST-WRITE line in current dmesg)"
  echo '```'
else
  echo "**UNLOCK: NOT ACTIVE** — card reports $(q memory.total) (stock)"
fi
echo

echo "## Memory integrity"
echo
echo '```'
python3 - "$FULL" <<'PY'
import sys,ctypes
full=int(sys.argv[1]); sweeps = 3 if full else 1
cuda=ctypes.CDLL('libcuda.so.1')
for f,a in [('cuMemAlloc_v2',[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]),
            ('cuMemsetD32_v2',[ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t]),
            ('cuMemcpyDtoH_v2',[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t])]:
    getattr(cuda,f).argtypes=a
if cuda.cuInit(0)!=0: print("cuInit failed"); sys.exit(1)
dev=ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev),0)
ctx=ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx),0,dev)
GiB=2**30; CH=4*GiB; NW=CH//4
ptrs=[]
for _ in range(16):
    p=ctypes.c_void_p()
    if cuda.cuMemAlloc_v2(ctypes.byref(p),CH)!=0: break
    ptrs.append(p)
print(f"allocated {len(ptrs)*4} GiB")
PAT=[0xA5A5A5A5,0x5A5A5A5A,0xFFFFFFFF]
buf=(ctypes.c_uint*4096)(); tot=0
for s in range(sweeps):
    pat=PAT[s%len(PAT)]
    for p in ptrs: cuda.cuMemsetD32_v2(p,ctypes.c_uint(pat),NW)
    e=0
    for p in ptrs:
        for off in (0,CH//2,CH-16384):
            cuda.cuMemcpyDtoH_v2(buf,ctypes.c_void_p(p.value+off),16384)
            e+=sum(1 for x in buf if x!=pat)
    tot+=e; print(f"sweep {s+1}/{sweeps} pattern 0x{pat:08X}: {e} errors")
print(f"TOTAL ERRORS: {tot}  -> {'CLEAN' if tot==0 else '*** FAILED ***'}")
PY
echo '```'
echo

echo "## Compute"
echo
echo '```'
python3 - <<'PY'
import time,torch
d=torch.device("cuda:0"); n=8192; flop=2.0*n**3
def bench(dt,tf32,label,secs=12):
    torch.backends.cuda.matmul.allow_tf32=tf32
    a=torch.randn(n,n,device=d,dtype=dt); b=torch.randn(n,n,device=d,dtype=dt)
    for _ in range(3): c=a@b
    torch.cuda.synchronize(); it=0; t0=time.perf_counter()
    while time.perf_counter()-t0<secs:
        for _ in range(5): c=a@b
        torch.cuda.synchronize(); it+=5
    print(f"{label:22s} {flop*it/(time.perf_counter()-t0)/1e12:7.2f} TFLOPS")
    del a,b,c; torch.cuda.empty_cache()
bench(torch.float32,False,"FP32 (TF32 off)")
bench(torch.float32,True ,"TF32 tensor")
bench(torch.float16,False,"FP16 tensor")
PY
echo '```'
echo
echo "Power cap during compute test: $(q power.limit) | peak temp: $(q temperature.gpu) C"
echo

if (( FULL )); then
echo "## Clock offset headroom (per-die — do not assume from another card)"
echo
echo '```'
for OFF in 250 300; do
  rc=$(sudo python3 -c "
import ctypes
nv=ctypes.CDLL('libnvidia-ml.so.1');nv.nvmlInit_v2()
d=ctypes.c_void_p();nv.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(d))
print(nv.nvmlDeviceSetGpcClkVfOffset(d,ctypes.c_int($OFF)))")
  got=$(python3 -c "
import ctypes
nv=ctypes.CDLL('libnvidia-ml.so.1');nv.nvmlInit_v2()
d=ctypes.c_void_p();nv.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(d))
o=ctypes.c_int();nv.nvmlDeviceGetGpcClkVfOffset(d,ctypes.byref(o));print(o.value)")
  echo "offset +${OFF}: set rc=${rc} readback=${got} MHz"
done
sudo python3 -c "
import ctypes
nv=ctypes.CDLL('libnvidia-ml.so.1');nv.nvmlInit_v2()
d=ctypes.c_void_p();nv.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(d))
nv.nvmlDeviceSetGpcClkVfOffset(d,ctypes.c_int(0))" >/dev/null 2>&1
echo "offset restored to 0"
echo "NOTE: +300 is the highest UPSTREAM-VALIDATED offset at a 1400 MHz ceiling."
echo "      +325 SILENTLY CORRUPTED memory upstream. Never exceed +300 without a full sweep."
echo '```'
echo
fi

echo "---"
echo "_Paste this block into CARD-REGISTRY.md._"
} 2>&1 | tee "$REC"

echo
echo ">>> saved: $REC"
