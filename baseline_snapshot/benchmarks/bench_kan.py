#!/usr/bin/env python3
"""
Benchmark: torch-spline-kan reference vs CUDA backend.
Usage: python benchmarks/bench_kan.py [--output results.json]
"""
import sys
import json
import time
import argparse
sys.path.insert(0, "/home/hy/torch-spline-kan")
import torch


def get_env_info():
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["driver"] = torch.version.cuda
    return info


def time_forward(fn, x, warmup=10, reps=50):
    for _ in range(warmup):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    if x.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / reps
    else:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        elapsed_ms = (time.perf_counter() - t0) / reps * 1000
    return elapsed_ms


def time_forward_backward(fn, x, warmup=5, reps=20):
    for _ in range(warmup):
        y = fn(x.detach().requires_grad_(True))
        y.sum().backward()
    if x.is_cuda:
        torch.cuda.synchronize()

    if x.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            x_ = x.detach().requires_grad_(True)
            y = fn(x_)
            y.sum().backward()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps
    else:
        t0 = time.perf_counter()
        for _ in range(reps):
            x_ = x.detach().requires_grad_(True)
            y = fn(x_)
            y.sum().backward()
        return (time.perf_counter() - t0) / reps * 1000


def measure_memory(fn, x):
    if not x.is_cuda:
        return None
    torch.cuda.reset_peak_memory_stats()
    y = fn(x.detach())
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6  # MB


def run_benchmarks():
    from spline_kan import KANLinear

    configs = [
        {"batch": 32,   "I": 16,  "O": 16,  "G": 5,  "k": 3},
        {"batch": 256,  "I": 64,  "O": 64,  "G": 5,  "k": 3},
        {"batch": 1024, "I": 64,  "O": 64,  "G": 5,  "k": 3},
        {"batch": 256,  "I": 256, "O": 256, "G": 5,  "k": 3},
        {"batch": 256,  "I": 64,  "O": 64,  "G": 16, "k": 3},
        {"batch": 256,  "I": 64,  "O": 64,  "G": 64, "k": 3},
        {"batch": 1,    "I": 16,  "O": 16,  "G": 5,  "k": 3},
    ]

    results = []
    cuda_ok = torch.cuda.is_available()

    for cfg in configs:
        B, I, O, G, k = cfg["batch"], cfg["I"], cfg["O"], cfg["G"], cfg["k"]
        print(f"\n--- B={B}, I={I}, O={O}, G={G}, k={k} ---")
        row = {**cfg}

        # Reference (CPU)
        ref = KANLinear(I, O, grid_size=G, spline_order=k, backend="reference")
        x_cpu = torch.randn(B, I)
        fwd_ref = time_forward(ref, x_cpu)
        fwdbwd_ref = time_forward_backward(ref, x_cpu)
        row["ref_fwd_ms"] = round(fwd_ref, 3)
        row["ref_fwdbwd_ms"] = round(fwdbwd_ref, 3)
        print(f"  Reference: fwd={fwd_ref:.3f}ms, fwd+bwd={fwdbwd_ref:.3f}ms")

        # Try torch.compile of reference
        try:
            compiled_ref = torch.compile(ref)
            _ = compiled_ref(x_cpu)
            fwd_compiled = time_forward(compiled_ref, x_cpu)
            row["compiled_ref_fwd_ms"] = round(fwd_compiled, 3)
            print(f"  Compiled ref: fwd={fwd_compiled:.3f}ms")
        except Exception as e:
            row["compiled_ref_fwd_ms"] = f"error: {type(e).__name__}"
            print(f"  Compiled ref: error ({e})")

        if cuda_ok:
            try:
                cuda_layer = KANLinear(I, O, grid_size=G, spline_order=k, backend="auto").cuda()
                x_cuda = x_cpu.cuda()
                fwd_cuda = time_forward(cuda_layer, x_cuda)
                fwdbwd_cuda = time_forward_backward(cuda_layer, x_cuda)
                mem_cuda = measure_memory(cuda_layer, x_cuda)
                row["cuda_fwd_ms"] = round(fwd_cuda, 3)
                row["cuda_fwdbwd_ms"] = round(fwdbwd_cuda, 3)
                row["cuda_peak_mb"] = round(mem_cuda, 2) if mem_cuda is not None else None
                speedup = fwd_ref / fwd_cuda if fwd_cuda > 0 else 0
                row["speedup_fwd"] = round(speedup, 2)
                print(f"  CUDA: fwd={fwd_cuda:.3f}ms, fwd+bwd={fwdbwd_cuda:.3f}ms, "
                      f"mem={mem_cuda:.1f}MB, speedup={speedup:.2f}x")
            except Exception as e:
                row["cuda_error"] = str(e)
                print(f"  CUDA: error ({e})")

        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args()

    env = get_env_info()
    print("Environment:", json.dumps(env, indent=2))
    print("\nRunning benchmarks...")
    results = run_benchmarks()

    output = {"env": env, "results": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
