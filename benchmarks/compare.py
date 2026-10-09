#!/usr/bin/env python3
"""
Builds the before/after markdown table from two bench_kan.py JSON files.

"before" = baseline_snapshot (original code), "after" = current tree.
Separate and joint regularization APIs are reported in separate rows;
the joint API is compared against the original separate API.

Usage: python benchmarks/compare.py benchmarks/results_before.json \
           benchmarks/results_after.json > benchmarks/comparison.md
"""
import argparse
import contextlib
import io
import json
from pathlib import Path


def get(r, sc, b, *path):
    x = r["runs"].get(sc, {}).get(b)
    if not x or "wall_ms" not in x:
        return None
    for p in path:
        x = x.get(p) if isinstance(x, dict) else None
        if x is None:
            return None
    return x


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


SCENARIOS = [
    ("infer", "infer"),
    ("fwd_bwd", "fwd_bwd"),
    ("fwd_bwd_reg", "fwd_bwd_reg"),
    ("fwd_bwd_reg", "fwd_bwd_reg_fused"),
    ("grid_update", "grid_update"),
]


def print_gpu_table(B, A, backends=("gpu_ref", "gpu_ref_compile", "gpu_native")):
    bm = {r["config"]["name"]: r for r in B["results"]}
    print("| config | after scenario | backend | before ms | after ms | speedup | before MiB | after MiB | actual backend (after) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for ra in A["results"]:
        name = ra["config"]["name"]
        rb = bm.get(name)
        if rb is None:
            continue
        for before_sc, after_sc in SCENARIOS:
            for b in backends:
                tb = get(rb, before_sc, b, "cuda_event_ms", "median")
                ta = get(ra, after_sc, b, "cuda_event_ms", "median")
                if ta is None:
                    continue
                mb = get(rb, before_sc, b, "memory", "peak_increment_MiB")
                ma = get(ra, after_sc, b, "memory", "peak_increment_MiB")
                label = get(ra, after_sc, b, "actual_backend")
                sp = (tb / ta) if (tb and ta) else None
                print(f"| {name} | {after_sc} | {b} | {fmt(tb)} | {fmt(ta)} | "
                      f"{'—' if sp is None else f'{sp:.2f}x'} | {fmt(mb, 1)} | {fmt(ma, 1)} | {label} |")


def print_optional_tables(data):
    print("| variant | params | training GEMM dtype | val RMSE median (per seed) | train step ms | train peak MiB | infer ms | infer peak MiB |")
    print("|---|---:|---|---|---:|---:|---:|---:|")
    for kind, row in data["regression"].items():
        seeds = "/".join(fmt(v, 5) for v in row["val_rmse_per_seed"])
        print(f"| {kind} | {row['params']} | {row.get('training_gemm_dtype', 'unrecorded')} | "
              f"{fmt(row['val_rmse_median'], 5)} ({seeds}) | {fmt(row['train_step_ms'], 2)} | "
              f"{fmt(row['train_peak_inc_MiB'], 1)} | {fmt(row['infer_ms'])} | {fmt(row['infer_peak_inc_MiB'], 1)} |")
    print("\nPDE: 1D Poisson, 2000 Adam steps, boundary loss and second input derivatives.\n")
    print("| variant | median L2 RMSE | per-seed L2 RMSE |")
    print("|---|---:|---|")
    for kind, row in data["pde_poisson_1d"].items():
        seeds = "/".join(f"{r['l2_rmse_vs_exact']:.3e}" for r in row["runs"])
        print(f"| {kind} | {row['l2_median']:.3e} | {seeds} |")


def replace_generated_block(path, label, body):
    path = Path(path)
    start = f"<!-- BEGIN GENERATED {label} -->"
    end = f"<!-- END GENERATED {label} -->"
    content = path.read_text()
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError(f"{path}: expected one {label} block")
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before + start + "\n\n" + body.rstrip() + "\n\n" + end + after)


def main(before_p, after_p, report=None, optional_results=None):
    B = json.load(open(before_p))
    A = json.load(open(after_p))
    bm = {r["config"]["name"]: r for r in B["results"]}
    print(f"Before: `{before_p}` (ext sha {B['env']['native_ext_sha256_16']}); "
          f"After: `{after_p}` (ext sha {A['env']['native_ext_sha256_16']}); "
          f"GPU {A['env'].get('gpu')}, torch {A['env']['torch_version']}, {A['env']['dtype']}.\n")
    print("Time = median CUDA-event ms per iteration (GPU-side span of one iteration incl. launch gaps); "
          "Mem = peak allocator increment over the pre-iteration baseline (MiB). "
          "optimizer.step not executed in these rows.\n")
    print("fwd_bwd_reg = separate calls; fwd_bwd_reg_fused = joint forward_with_regularization. "
          "Both use the original separate API for the before column. No fastest-of-two selection.\n")
    print_gpu_table(B, A)
    print("\nCPU reference (wall ms, median):\n")
    print("| config | scenario | before | after |")
    print("|---|---|---:|---:|")
    for ra in A["results"]:
        rb = bm.get(ra["config"]["name"])
        if rb is None:
            continue
        for before_sc, after_sc in SCENARIOS:
            tb = get(rb, before_sc, "cpu_ref", "wall_ms", "median")
            ta = get(ra, after_sc, "cpu_ref", "wall_ms", "median")
            if ta is None:
                continue
            print(f"| {ra['config']['name']} | {after_sc} | {fmt(tb)} | {fmt(ta)} |")
    print("\nTraining-step memory breakdown (Adam, optimizer.step executed), native backend:\n")
    print("| config | scenario | side | params | grads | Adam state | peak inc | transient |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for ra in A["results"]:
        rb = bm.get(ra["config"]["name"])
        for side, r, key in (("before", rb, "fwd_bwd_reg/gpu_native"),
                             ("after", ra, "fwd_bwd_reg_fused/gpu_native"),
                             ("after (separate API)", ra, "fwd_bwd_reg/gpu_native")):
            t = (r or {}).get("train_memory", {}).get(key)
            if not t or "peak_MiB" not in t:
                continue
            print(f"| {ra['config']['name']} | {key.split('/')[0]} | {side} | {t['parameters_MiB']:.2f} | "
                  f"{t['gradients_MiB']:.2f} | {t['optimizer_state_MiB']:.2f} | "
                  f"{t['peak_increment_MiB']:.1f} | {t['transient_MiB']:.1f} |")
    if report:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_gpu_table(B, A, backends=("gpu_native",))
        replace_generated_block(report, "BENCHMARK COMPARISON", buf.getvalue())
        if optional_results:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_optional_tables(json.loads(Path(optional_results).read_text()))
            replace_generated_block(report, "OPTIONAL EXPERIMENTS", buf.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--report", help="update marked tables in the optimization report")
    parser.add_argument("--optional-results", help="regenerate the optional-experiment tables too")
    args = parser.parse_args()
    if args.optional_results and not args.report:
        parser.error("--optional-results requires --report")
    main(args.before, args.after, args.report, args.optional_results)
