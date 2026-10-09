#!/usr/bin/env python3
"""
Phase-4 measurement: original vs export_inference() model.

Reports parameter count, state_dict file size, max output difference,
inference latency (inference_mode, median of repeats) and peak VRAM
increment, for the native and reference backends on the same GPU.

Usage: python benchmarks/bench_export.py --output benchmarks/results_export.json
"""
import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402

from spline_kan import KAN  # noqa: E402

CONFIGS = [
    dict(name="KAN-16-32-8", widths=[16, 32, 8], G=5, B=[1, 256, 4096]),
    dict(name="KAN-256-256-256", widths=[256, 256, 256], G=5, B=[1, 1024]),
    dict(name="KAN-64-64-G64", widths=[64, 64, 64], G=64, B=[1, 1024]),
]


def latency(fn, reps, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(ts), "std_ms": statistics.stdev(ts)}


def peak_inc(fn):
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def file_size(sd):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(sd, path)
    n = os.path.getsize(path)
    os.remove(path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_HERE, "results_export.json"))
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    out = {"env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                   "command": " ".join([sys.executable] + sys.argv)}, "results": []}
    for cfg in CONFIGS:
        for backend in ("cuda", "reference"):
            torch.manual_seed(0)
            m = KAN(cfg["widths"], grid_size=cfg["G"], backend=backend).cuda()
            with torch.no_grad():
                for l in m.layers:
                    l.spline_scaler.mul_(torch.rand_like(l.spline_scaler) + 0.5)
            m.eval()
            e = m.export_inference()
            row = {"config": cfg["name"], "backend": backend,
                   "params_original": sum(p.numel() for p in m.parameters()),
                   "params_exported": sum(p.numel() for p in e.parameters()),
                   "file_bytes_original": file_size(m.state_dict()),
                   "file_bytes_exported": file_size(e.state_dict()),
                   "per_batch": {}}
            for B in cfg["B"]:
                x = torch.rand(B, cfg["widths"][0], device="cuda") * 2 - 1
                with torch.inference_mode():
                    diff = (m(x) - e(x)).abs().max().item()
                    ref_scale = m(x).abs().max().item()
                r = {"max_abs_diff": diff, "max_abs_output": ref_scale}
                for tag, mod in (("original", m), ("exported", e)):
                    def f(mod=mod):
                        with torch.inference_mode():
                            mod(x)
                    r[tag] = {**latency(f, args.reps), "peak_increment_MiB": peak_inc(f)}
                row["per_batch"][B] = r
                print(cfg["name"], backend, B,
                      f"diff={diff:.2e} orig={r['original']['median_ms']:.4f}ms "
                      f"exp={r['exported']['median_ms']:.4f}ms "
                      f"mem {r['original']['peak_increment_MiB']:.2f}->"
                      f"{r['exported']['peak_increment_MiB']:.2f}MiB", flush=True)
            out["results"].append(row)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
