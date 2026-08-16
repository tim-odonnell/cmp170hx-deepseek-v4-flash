#!/usr/bin/env bash
# Test GPC clock VF offset on the CMP 170HX at a pinned clock ceiling.
#
# WHY THIS IS NOT A NORMAL BENCHMARK
# ----------------------------------
# This card has NO ECC and no error telemetry. The community sweep recorded
# 1400 MHz / +325 offset SILENTLY CORRUPTING MEMORY (6 errors, then 3, then 0 across three
# sweeps) and 1400/+375 taking a CUDA device fault. A run that completes is therefore NOT
# evidence the setting is safe -- the only evidence is a full-VRAM write/verify sweep coming
# back with zero mismatches, repeated. That is what this script does, and it is the reason
# it takes longer than a throughput test would.
#
# +300 at a 1400 MHz ceiling is the HIGHEST VALIDATED offset for that ceiling
# (138.5 W, 4 sweeps, 0 errors upstream). Do not raise it from here without repeating the
# sweep, and never past +325 at this ceiling.
#
# The offset is only settable through NVML (nvmlDeviceSetGpcClkVfOffset) -- nvidia-smi has
# no such option and nvidia-settings reports "GPUGraphicsClockOffset: No such attribute"
# on this GPU.
#
# Restores offset 0 and unlocks clocks on EVERY exit path.
set -uo pipefail

CEILING="${CEILING:-1400}"
OFFSET="${OFFSET:-300}"
SWEEPS="${SWEEPS:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/bench-results"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)

nvml() {  # nvml <set|get> [value]
    # SET requires root: NVML returns rc=4 (NVML_ERROR_NO_PERMISSION) as a normal user,
    # while GET works unprivileged. Run the whole helper under sudo for uniformity.
    sudo python3 - "$@" <<'PY'
import ctypes,sys
nv=ctypes.CDLL("libnvidia-ml.so.1")
if nv.nvmlInit_v2()!=0: print("NVML init failed",file=sys.stderr); sys.exit(1)
d=ctypes.c_void_p()
nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(d))
if sys.argv[1]=="set":
    rc=nv.nvmlDeviceSetGpcClkVfOffset(d, ctypes.c_int(int(sys.argv[2])))
    print(rc)
else:
    o=ctypes.c_int(); nv.nvmlDeviceGetGpcClkVfOffset(d, ctypes.byref(o)); print(o.value)
nv.nvmlShutdown()
PY
}

restore() {
    echo ""
    echo ">>> RESTORING: offset 0, clocks unlocked"
    nvml set 0 >/dev/null 2>&1
    sudo nvidia-smi -rgc >/dev/null 2>&1
    echo "    offset now: $(nvml get)   $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader)"
}
trap restore EXIT INT TERM

measure() {   # $1 = label
    python3 - "$1" "$OUT/offset_${STAMP}.csv" <<'PY'
import sys,time,statistics,subprocess,torch
label=sys.argv[1]; csv=sys.argv[2]
dev=torch.device("cuda:0")
def smi(q): return subprocess.run(["nvidia-smi",f"--query-gpu={q}","--format=csv,noheader,nounits"],
                                  capture_output=True,text=True).stdout.strip().split("\n")[0]
n=8192
a=torch.randn(n,n,device=dev,dtype=torch.bfloat16)
b=torch.randn(n,n,device=dev,dtype=torch.bfloat16)
for _ in range(3): c=a@b
torch.cuda.synchronize()
pw=[]; clk=[]
flop=2.0*n**3; iters=0; t0=time.perf_counter()
while time.perf_counter()-t0 < 20:
    for _ in range(5): c=a@b
    torch.cuda.synchronize(); iters+=5
    pw.append(float(smi("power.draw"))); clk.append(float(smi("clocks.sm")))
dt=time.perf_counter()-t0
tf=(flop*iters)/dt/1e12
p=statistics.mean(pw); k=statistics.mean(clk)
print(f"  {label:22s} BF16 {tf:7.2f} TFLOPS   {p:6.1f} W   {k:6.0f} MHz SM   {1000*tf/p:7.0f} GFLOP/W")
open(csv,"a").write(f"{label},{tf:.2f},{p:.1f},{k:.0f},{1000*tf/p:.0f}\n")
del a,b,c; torch.cuda.empty_cache()
PY
}

integrity() {  # full-VRAM write/verify -- the ONLY evidence the offset is safe
    python3 - "$1" <<'PY'
import sys,ctypes
sweeps=int(sys.argv[1])
cuda=ctypes.CDLL('libcuda.so.1')
for f,a in [('cuMemAlloc_v2',[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]),
            ('cuMemsetD32_v2',[ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t]),
            ('cuMemcpyDtoH_v2',[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t])]:
    getattr(cuda,f).argtypes=a
cuda.cuInit(0)
dev=ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev),0)
ctx=ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx),0,dev)
GiB=2**30; CH=4*GiB; NW=CH//4
ptrs=[]
for i in range(14):
    p=ctypes.c_void_p()
    if cuda.cuMemAlloc_v2(ctypes.byref(p),CH)!=0: break
    ptrs.append(p)
print(f"    allocated {len(ptrs)*4} GiB for pattern sweep")
total_err=0
PATTERNS=[0xA5A5A5A5,0x5A5A5A5A,0xFFFFFFFF,0x00000000,0xDEADBEEF]
buf=(ctypes.c_uint*4096)()
for s in range(sweeps):
    pat=PATTERNS[s % len(PATTERNS)]
    for p in ptrs: cuda.cuMemsetD32_v2(p, ctypes.c_uint(pat), NW)
    err=0
    for i,p in enumerate(ptrs):
        for off in (0, CH//4, CH//2, 3*CH//4, CH-16384):
            cuda.cuMemcpyDtoH_v2(buf, ctypes.c_void_p(p.value+off), 16384)
            err += sum(1 for x in buf if x != pat)
    total_err += err
    print(f"    sweep {s+1}/{sweeps}  pattern 0x{pat:08X}  errors: {err}")
print(f"    TOTAL ERRORS: {total_err}")
sys.exit(1 if total_err else 0)
PY
}

echo "=================================================================="
echo " GPC clock VF offset test -- ceiling ${CEILING} MHz, offset +${OFFSET}"
echo " power cap: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
echo "=================================================================="
echo "label,tflops,watts,sm_mhz,gflops_per_w" > "$OUT/offset_${STAMP}.csv"

echo ">>> pinning clock ceiling to ${CEILING} MHz"
sudo nvidia-smi -lgc "${CEILING},${CEILING}" >/dev/null 2>&1
nvml set 0 >/dev/null

echo ">>> BASELINE (ceiling ${CEILING}, offset 0)"
measure "ceil${CEILING}_off0"

echo ">>> applying offset +${OFFSET}"
rc=$(nvml set "$OFFSET"); got=$(nvml get)
echo "    nvmlDeviceSetGpcClkVfOffset rc=${rc}, readback=${got} MHz"
if [[ "$got" != "$OFFSET" ]]; then echo "    OFFSET DID NOT APPLY -- aborting"; exit 1; fi

echo ">>> MEASURING (ceiling ${CEILING}, offset +${OFFSET})"
measure "ceil${CEILING}_off${OFFSET}"

echo ""
echo ">>> MEMORY INTEGRITY SWEEP (${SWEEPS} passes) -- the safety gate"
if integrity "$SWEEPS"; then
    echo "    RESULT: CLEAN -- no corruption detected at +${OFFSET}"
else
    echo "    RESULT: *** CORRUPTION DETECTED -- DO NOT USE THIS OFFSET ***"
fi

echo ""
echo "=================================================================="
column -t -s, "$OUT/offset_${STAMP}.csv"
echo "=================================================================="
echo "upstream reference at 1400 MHz: +0 -> 198 W, +300 -> 138.5 W (4 sweeps clean), +325 -> CORRUPT"
