import torch, time, sys
torch.cuda.init()
dev = torch.device("cuda:0")
n = 8192
a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
t_end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 25
iters = 0
while time.time() < t_end:
    c = a @ b
    iters += 1
    if iters % 50 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"done: {iters} matmuls")
