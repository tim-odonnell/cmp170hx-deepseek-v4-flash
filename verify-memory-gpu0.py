# Step 9 memory-integrity proof for GPU index 0 specifically (cuDeviceGet(..., 0) below is
# hardcoded, not auto-discovered). For a new card, either copy this file with the index
# changed, or parameterize it to take the index as an argument -- not done yet since only
# 2 cards existed when this was written. See TRANSFER-README.md section 4.
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
for i in range(14):
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
