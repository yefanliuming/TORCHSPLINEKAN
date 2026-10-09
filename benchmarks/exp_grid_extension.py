#!/usr/bin/env python3
"""
Phase-6 experiment: small-G training + grid extension vs direct large-G.

Task (fixed seeds): f(x,y) = exp(sin(pi x) + y^2)  (paper Sec. 2.4 toy),
[2, 5, 1] KAN, Adam, full-batch, 1024 train / 4096 val points.

Schedules with the same total number of optimiser steps:
  direct   : G = G_final for all steps
  extend   : G = 3 -> 5 -> 10 -> 20 (equal step share), grid extension in
             between; a NEW Adam optimiser is created after each extension
             (parameter shapes change).

Reported per schedule and seed: train / val RMSE, wall time (GPU,
synchronised), per-step time and peak VRAM per stage, parameter count per
stage, and the curve / network output error right after each extension.
Results only describe this synthetic task.
"""
import argparse
import gc
import json
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402

from spline_kan import KAN  # noqa: E402


def f(x):
    return torch.exp(torch.sin(math.pi * x[:, :1]) + x[:, 1:2] ** 2)


def rmse(m, x, y):
    with torch.no_grad():
        return ((m(x) - y) ** 2).mean().sqrt().item()


def run_stage(m, x, y, steps, lr):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    torch.cuda.synchronize()
    gc.collect()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"steps": steps, "time_s": dt, "ms_per_step": dt / steps * 1e3,
            "peak_inc_MiB": (torch.cuda.max_memory_allocated() - base) / 2 ** 20,
            "params": sum(p.numel() for p in m.parameters())}


def schedule(kind, Gs, total, x, y, xv, yv, seed, width, lr):
    torch.manual_seed(seed)
    stages = []
    if kind == "direct":
        m = KAN([2, width, 1], grid_size=Gs[-1]).cuda()
        st = run_stage(m, x, y, total, lr)
        stages.append({"G": Gs[-1], **st, "train_rmse": rmse(m, x, y), "val_rmse": rmse(m, xv, yv)})
        t_total = st["time_s"]
    else:
        m = KAN([2, width, 1], grid_size=Gs[0]).cuda()
        per = total // len(Gs)
        t_total = 0.0
        for j, G in enumerate(Gs):
            ext = {}
            if j > 0:
                t0 = time.perf_counter()
                new = m.extend_grid(G, x)
                torch.cuda.synchronize()
                ext_time = time.perf_counter() - t0
                xt = xv
                with torch.no_grad():
                    ext = {"extend_time_s": ext_time,
                           "net_max_abs_diff_at_extension": (new(xt) - m(xt)).abs().max().item(),
                           "val_rmse_before": rmse(m, xv, yv), "val_rmse_after": rmse(new, xv, yv)}
                    # curve error of every edge on its actual input
                    h, cerr = xt, 0.0
                    for lo, ln in zip(m.layers, new.layers):
                        cerr = max(cerr, (ln(h) - lo(h)).abs().max().item())
                        h = lo(h)
                    ext["layer_max_abs_diff"] = cerr
                t_total += ext_time
                m = new
            st = run_stage(m, x, y, per, lr)
            t_total += st["time_s"]
            stages.append({"G": G, **ext, **st, "train_rmse": rmse(m, x, y), "val_rmse": rmse(m, xv, yv)})
    return {"kind": kind, "seed": seed, "total_time_s": t_total,
            "final_train_rmse": rmse(m, x, y), "final_val_rmse": rmse(m, xv, yv),
            "final_params": sum(p.numel() for p in m.parameters()), "stages": stages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_HERE, "results_grid_extension.json"))
    ap.add_argument("--grids", default="3,5,10,20")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-train", type=int, default=1024)
    args = ap.parse_args()
    Gs = [int(g) for g in args.grids.split(",")]
    g = torch.Generator().manual_seed(1234)
    x = (torch.rand(args.n_train, 2, generator=g) * 2 - 1).cuda()
    xv = (torch.rand(4096, 2, generator=g) * 2 - 1).cuda()
    y, yv = f(x), f(xv)
    # warm up CUDA/cuBLAS so the first schedule is not penalised
    schedule("direct", Gs, 50, x, y, xv, yv, 99, args.width, args.lr)
    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for kind in ("direct", "extend"):
            r = schedule(kind, Gs, args.steps, x, y, xv, yv, seed, args.width, args.lr)
            runs.append(r)
            print(f"{kind:7s} seed={seed} val={r['final_val_rmse']:.2e} "
                  f"train={r['final_train_rmse']:.2e} time={r['total_time_s']:.2f}s "
                  f"params={r['final_params']}", flush=True)
    summ = {}
    for kind in ("direct", "extend"):
        rs = [r for r in runs if r["kind"] == kind]
        summ[kind] = {"val_rmse_median": statistics.median(r["final_val_rmse"] for r in rs),
                      "train_rmse_median": statistics.median(r["final_train_rmse"] for r in rs),
                      "time_s_median": statistics.median(r["total_time_s"] for r in rs)}
    out = {"task": "f=exp(sin(pi x)+y^2), [2,width,1] KAN, Adam full batch",
           "args": vars(args), "summary": summ, "runs": runs,
           "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                   "command": " ".join([sys.executable] + sys.argv)}}
    print(json.dumps(summ, indent=1))
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
