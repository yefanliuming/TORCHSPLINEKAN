#!/usr/bin/env python3
"""
Phase-7 measurement: local (k+1 active) vs legacy full-recurrence CUDA basis
kernels, same build, same GPU, same weights/inputs.

For every configuration: GPU kernel time of the basis kernel alone and of
the basis-gradient contraction, plus median wall / CUDA-event time and peak
VRAM increment of the native layer for inference and forward+backward.
Also reports the fixed-grid piecewise-polynomial (Horner) storage cost that
an inference-only alternative would need (computed, not implemented).

Usage: python benchmarks/bench_local_basis.py --output benchmarks/results_local_basis.json
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from spline_kan import KANLinear  # noqa: E402
from spline_kan import kan_linear as kl  # noqa: E402

C = kl._C
CONFIGS = [
    (1, 16, 16, 5), (256, 64, 64, 5), (1024, 256, 256, 5), (4096, 256, 256, 5),
    (256, 64, 64, 64), (4096, 64, 64, 64), (1024, 256, 256, 32),
]


def kernel_ms(fn, reps=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    per = {}
    for e in p.events():
        if e.device_type == DeviceType.CUDA:
            per[e.name] = per.get(e.name, 0.0) + e.time_range.elapsed_us() / reps / 1e3
    return sum(per.values()), per


def wall(fn, reps):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts, ev = [], []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        ev.append(a.elapsed_time(b))
    return {"wall_median_ms": statistics.median(ts), "wall_std_ms": statistics.stdev(ts),
            "event_median_ms": statistics.median(ev), "event_std_ms": statistics.stdev(ev)}


def peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    b = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - b) / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_HERE, "results_local_basis.json"))
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    k = 3
    rows = []
    for B, I, O, G in CONFIGS:
        torch.manual_seed(0)
        L = KANLinear(I, O, grid_size=G, backend="cuda").cuda()
        x = torch.rand(B, I, device="cuda") * 2 - 1
        R = G + k
        gH = torch.randn(B, I, R, device="cuda")
        row = {"B": B, "I": I, "O": O, "G": G, "k": k,
               "spline_weight_bytes": O * I * R * 4,
               "horner_coeff_bytes_fp32": O * I * G * (k + 1) * 4,
               "horner_note": "piecewise-polynomial form of the folded spline: G intervals x (k+1) "
                              "coefficients per edge (vs G+k B-spline coefficients); not implemented"}
        for impl, name in ((0, "recurrence"), (1, "local")):
            C.set_basis_impl(impl)
            r = {}
            r["basis_kernel_ms"], _ = kernel_ms(lambda: C.basis(x, L.grid, k))
            if impl == 1:
                r["basis_grad_kernel_ms"], _ = kernel_ms(lambda: C.basis_grad(x, L.grid, k, gH))

            def infer():
                with torch.inference_mode():
                    L(x)

            xg = x.clone().requires_grad_()

            def fb():
                for p in L.parameters():
                    p.grad = None
                xg.grad = None
                L(xg).sum().backward()

            def fb_nox():
                for p in L.parameters():
                    p.grad = None
                L(x).sum().backward()
            r["infer"] = {**wall(infer, args.reps), "peak_inc_MiB": peak(infer),
                          "gpu_kernel_ms": kernel_ms(infer)[0]}
            r["fwd_bwd_grad_x"] = {**wall(fb, args.reps), "peak_inc_MiB": peak(fb),
                                   "gpu_kernel_ms": kernel_ms(fb)[0]}
            r["fwd_bwd_no_grad_x"] = {**wall(fb_nox, args.reps), "peak_inc_MiB": peak(fb_nox),
                                      "gpu_kernel_ms": kernel_ms(fb_nox)[0]}
            row[name] = r
        C.set_basis_impl(1)
        rows.append(row)
        rc, lc = row["recurrence"], row["local"]
        print(f"B={B} I={I} O={O} G={G}: basis {rc['basis_kernel_ms']:.4f}->{lc['basis_kernel_ms']:.4f}ms | "
              f"fwd+bwd kernel {rc['fwd_bwd_grad_x']['gpu_kernel_ms']:.4f}->{lc['fwd_bwd_grad_x']['gpu_kernel_ms']:.4f}ms "
              f"event {rc['fwd_bwd_grad_x']['event_median_ms']:.4f}->{lc['fwd_bwd_grad_x']['event_median_ms']:.4f}ms "
              f"peak {rc['fwd_bwd_grad_x']['peak_inc_MiB']:.1f}->{lc['fwd_bwd_grad_x']['peak_inc_MiB']:.1f}MiB | "
              f"infer event {rc['infer']['event_median_ms']:.4f}->{lc['infer']['event_median_ms']:.4f}ms", flush=True)
    out = {"env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                   "ext": C.__file__, "command": " ".join([sys.executable] + sys.argv)},
           "results": rows}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
