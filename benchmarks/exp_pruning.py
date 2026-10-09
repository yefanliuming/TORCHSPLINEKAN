#!/usr/bin/env python3
"""
Phase-5 experiment: sparsify -> prune -> fine-tune on a synthetic regression.

Task (fixed seeds): f(x1..x4) = exp(sin(pi*x1) + x2^2) + 0.5*x3, x4 unused.
Only 2 hidden nodes are needed in principle (paper's [2,5,1]-style example,
extended with a linear term and a distractor input).

Measured for the dense model, the pruned model (no fine-tune) and the
pruned + fine-tuned model: validation RMSE, parameter count, widths,
peak VRAM increment and median time of a training step / inference.
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


def target(x):
    return torch.exp(torch.sin(math.pi * x[:, :1]) + x[:, 1:2] ** 2) + 0.5 * x[:, 2:3]


def rmse(m, x, y):
    with torch.no_grad():
        return ((m(x) - y) ** 2).mean().sqrt().item()


def train(m, x, y, steps, lr, lam, log=None):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        out, reg = m.forward_with_regularization(x, lam, lam)
        loss = ((out - y) ** 2).mean()
        (loss + reg if lam > 0 else loss).backward()
        opt.step()
        if log is not None and s % 500 == 0:
            log.append((s, loss.item(), reg.item()))
    return m


def timing(fn, reps=50, warmup=10):
    for _ in range(warmup):
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


def profile(m, x, y):
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        out, reg = m.forward_with_regularization(x, 1e-3, 1e-3)
        (((out - y) ** 2).mean() + reg).backward()
        opt.step()

    def infer():
        with torch.inference_mode():
            m(x)
    step()
    return {"params": sum(p.numel() for p in m.parameters()), "widths": m.widths,
            "train_step_ms": timing(step), "infer_ms": timing(infer),
            "train_step_peak_inc_MiB": peak(step), "infer_peak_inc_MiB": peak(infer)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_HERE, "results_pruning.json"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--finetune", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    g = torch.Generator().manual_seed(args.seed)
    xtr = (torch.rand(4096, 4, generator=g) * 2 - 1).to(dev)
    xva = (torch.rand(4096, 4, generator=g) * 2 - 1).to(dev)
    ytr, yva = target(xtr), target(xva)

    torch.manual_seed(args.seed)
    dense = KAN([4, args.hidden, 1], grid_size=5).to(dev)
    log = []
    t0 = time.perf_counter()
    train(dense, xtr, ytr, args.steps, 1e-2, args.lam, log)
    torch.cuda.synchronize()
    t_train = time.perf_counter() - t0

    res = dense.prune(xtr, threshold=args.threshold)
    pruned = res.model
    r_pruned = rmse(pruned, xva, yva)
    ft = KAN.from_layers([l for l in pruned.layers])  # same object, fine-tuned below
    t1 = time.perf_counter()
    train(ft, xtr, ytr, args.finetune, 3e-3, 0.0)
    torch.cuda.synchronize()
    t_ft = time.perf_counter() - t1

    # reference: same dense model fine-tuned without pruning for the same steps
    torch.manual_seed(args.seed)
    dense_ft = KAN([4, args.hidden, 1], grid_size=5).to(dev)
    dense_ft.load_state_dict(dense.state_dict())
    t2 = time.perf_counter()
    train(dense_ft, xtr, ytr, args.finetune, 3e-3, 0.0)
    torch.cuda.synchronize()
    t_dft = time.perf_counter() - t2

    out = {
        "task": "f=exp(sin(pi x1)+x2^2)+0.5 x3, x4 unused; x~U[-1,1]^4; 4096 train / 4096 val",
        "args": vars(args),
        "train_log": log,
        "dense_sparsified": {"val_rmse": rmse(dense, xva, yva), "train_s": t_train,
                             **profile(dense, xtr, ytr)},
        "pruned_no_finetune": {"val_rmse": r_pruned, **res.summary()},
        "pruned_finetuned": {"val_rmse": rmse(ft, xva, yva), "finetune_s": t_ft,
                             **profile(ft, xtr, ytr)},
        "dense_finetuned_same_steps": {"val_rmse": rmse(dense_ft, xva, yva), "finetune_s": t_dft},
        "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "command": " ".join([sys.executable] + sys.argv)},
    }
    print(json.dumps({k: v for k, v in out.items() if k not in ("train_log",)}, indent=1))
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
