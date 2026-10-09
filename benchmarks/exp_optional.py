#!/usr/bin/env python3
"""
Phase-8 optional experiments (all opt-in variants; baseline = KAN default).

A. Regression (synthetic, fixed seeds): y = sum_j sin(pi*x_j) * x_{j+1} over
   16 inputs; KAN [16, 64, 8]-shaped models (8 outputs = 8 different
   shifted targets).  For each variant: val RMSE (median over seeds),
   parameter count, train-step time, inference throughput, peak VRAM.
     baseline   KAN (native backend), FP32
     baseline-ref  KAN on the reference backend (same ops as the variants)
     mp-bf16    MixedPrecisionKANLinear: FP32 basis, BF16 GEMMs
     mp-fp16    same with FP16 GEMMs
     grouped    GroupedSplineKANLinear, 4 outputs share each spline
     lowrank    LowRankKANLinear, rank 4

B. PDE check (separate; high-order derivatives): 1D Poisson u'' = -pi^2
   sin(pi x), u(+-1)=0, [1, 16, 1] KAN, physics-informed loss with
   create_graph second derivatives.  Compared: baseline FP64, baseline
   FP32, mp-bf16 (FP32 basis, BF16 GEMMs).  Reports L2 error vs sin(pi x).

Results describe these synthetic tasks only.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from spline_kan import KAN  # noqa: E402
from spline_kan.experimental import MixedPrecisionKANLinear, build_kan  # noqa: E402


def target(x):
    outs = []
    for s in range(8):
        acc = 0
        for j in range(4):
            a, b = (2 * s + j) % 16, (2 * s + j + 5) % 16
            acc = acc + torch.sin(math.pi * x[:, a]) * x[:, b]
        outs.append(acc[:, None])
    return torch.cat(outs, 1)


def make(kind, seed):
    torch.manual_seed(seed)
    w = [16, 64, 8]
    if kind == "baseline":
        return KAN(w, backend="cuda").cuda()
    if kind == "baseline-ref":       # same maths, pure-PyTorch path like the variants
        return KAN(w, backend="reference").cuda()
    if kind in ("mp-bf16", "mp-fp16"):
        dt = torch.bfloat16 if kind == "mp-bf16" else torch.float16
        base = KAN(w, backend="reference").cuda()
        return KAN.from_layers([MixedPrecisionKANLinear.from_layer(l, dt) for l in base.layers])
    if kind == "grouped":
        return build_kan("grouped", w, group_size=4).cuda()
    if kind == "lowrank":
        return build_kan("lowrank", w, rank=4).cuda()
    raise ValueError(kind)


def med_time(fn, reps=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    b = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - b) / 2 ** 20


def regression(kinds, seeds, steps):
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(8192, 16, generator=g) * 2 - 1).cuda()
    xv = (torch.rand(8192, 16, generator=g) * 2 - 1).cuda()
    y, yv = target(x), target(xv)
    res = {}
    for kind in kinds:
        rms, rows = [], {}
        for seed in seeds:
            m = make(kind, seed)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            stable = True

            def step():
                opt.zero_grad(set_to_none=True)
                out, reg = m.forward_with_regularization(x, 1e-5, 1e-5)
                loss = ((out - y) ** 2).mean() + reg
                loss.backward()
                opt.step()
                return loss
            for _ in range(steps):
                l = step()
            if not torch.isfinite(l):
                stable = False
            with torch.no_grad():
                rms.append(((m(xv) - yv) ** 2).mean().sqrt().item())
            if seed == seeds[0]:
                def infer():
                    with torch.inference_mode():
                        m(xv)
                t_inf = med_time(infer)
                train_ms = med_time(step)
                opt.zero_grad(set_to_none=True)
                train_peak = peak(step)
                opt.zero_grad(set_to_none=True)
                infer_peak = peak(infer)
                rows = {"params": sum(p.numel() for p in m.parameters()),
                        "train_step_ms": train_ms, "train_peak_inc_MiB": train_peak,
                        "infer_ms": t_inf, "infer_samples_per_s": 8192 / (t_inf / 1e3),
                        "infer_peak_inc_MiB": infer_peak,
                        "training_api": "forward_with_regularization",
                        "training_gemm_dtype": ("bfloat16" if kind == "mp-bf16" else
                                                "float16" if kind == "mp-fp16" else "float32"),
                        "regularization_dtype": "float32",
                        "actual_layer_backends": [getattr(l, "last_backend", "reference")
                                                  for l in m.layers],
                        "optimizer_step_executed": True,
                        "memory_baseline_grads_cleared": True}
            rows.setdefault("stable", True)
            rows["stable"] = rows["stable"] and stable
        res[kind] = {**rows, "val_rmse_per_seed": rms, "val_rmse_median": statistics.median(rms)}
        print(kind, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[kind].items()}),
              flush=True)
    return res


def pde(kind, seed, steps):
    torch.manual_seed(seed)
    dt = torch.float64 if kind == "baseline-fp64" else torch.float32
    base = KAN([1, 16, 1], grid_size=10, backend="cuda").cuda().to(dt)
    m = base if kind.startswith("baseline") else KAN.from_layers(
        [MixedPrecisionKANLinear.from_layer(l, torch.bfloat16) for l in base.layers])
    xi = torch.linspace(-1, 1, 201, device="cuda", dtype=dt)[1:-1, None]
    xb = torch.tensor([[-1.0], [1.0]], device="cuda", dtype=dt)
    f = -(math.pi ** 2) * torch.sin(math.pi * xi)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x = xi.clone().requires_grad_()
        u = m(x)
        ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
        loss = ((uxx - f) ** 2).mean() + 100 * (m(xb) ** 2).mean()
        loss.backward()
        opt.step()
    xt = torch.linspace(-1, 1, 1001, device="cuda", dtype=dt)[:, None]
    with torch.no_grad():
        err = (m(xt) - torch.sin(math.pi * xt)).pow(2).mean().sqrt().item()
    return {"l2_rmse_vs_exact": err, "final_loss": loss.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_HERE, "results_optional.json"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pde-steps", type=int, default=2000)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    reg = regression(["baseline", "baseline-ref", "mp-bf16", "mp-fp16", "grouped", "lowrank"], seeds, args.steps)
    pde_res = {}
    for kind in ("baseline-fp64", "baseline-fp32", "mp-bf16"):
        runs = [pde(kind, s, args.pde_steps) for s in seeds]
        pde_res[kind] = {"runs": runs, "l2_median": statistics.median(r["l2_rmse_vs_exact"] for r in runs)}
        print("PDE", kind, pde_res[kind]["l2_median"], flush=True)
    out = {"regression": reg, "pde_poisson_1d": pde_res, "args": vars(args),
           "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                   "python_executable": sys.executable,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "source_sha256": {
                       path: hashlib.sha256(open(os.path.join(os.path.dirname(_HERE), path), "rb").read()).hexdigest()
                       for path in ("spline_kan/experimental.py", "spline_kan/fused.py", "benchmarks/exp_optional.py")},
                   "command": " ".join([sys.executable] + sys.argv)}}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
