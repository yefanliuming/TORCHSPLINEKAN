#!/usr/bin/env python3
"""
Phase-3 measurement: one fwd+bwd of MSE + activation reg for a single native
layer I=O, comparing the public separate API and forward_with_regularization
with each reg_impl.  Peak allocator increment and median wall time (20 reps).

Usage: python benchmarks/bench_reg_impl.py <pkg_root> <B> <I>
"""
import sys, time, torch, torch.nn.functional as F, gc
sys.path.insert(0, sys.argv[1])
from spline_kan import KANLinear
torch.manual_seed(0)
B,I,O = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[3])
L = KANLinear(I,O,backend="cuda").cuda(); x = torch.rand(B,I,device="cuda")*2-1; yt = torch.randn(B,O,device="cuda")
def step_old():
    (F.mse_loss(L(x),yt) + L.regularization_loss(1e-2,1e-2,x=x)).backward()
def mk(impl):
    def f():
        y,r = L.forward_with_regularization(x,1e-2,1e-2,reg_impl=impl, chunk_size=256); (F.mse_loss(y,yt)+r).backward()
    return f
fns = {"separate(auto)": step_old}
if hasattr(L, "forward_with_regularization"):
    fns.update({f"fwr-{i}": mk(i) for i in ("dense","fused","checkpoint")})
for name,f in fns.items():
    for p in L.parameters(): p.grad=None
    for _ in range(3): f()
    for p in L.parameters(): p.grad=None
    gc.collect(); torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    f(); torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated()
    ts=[]
    for _ in range(20):
        for p in L.parameters(): p.grad=None
        torch.cuda.synchronize(); t=time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); print(f"{name:16s} peak+={(peak-base)/2**20:8.1f} MiB  median={ts[10]:.3f} ms")
