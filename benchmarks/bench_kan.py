#!/usr/bin/env python3
"""
Benchmark torch-spline-kan backends on identical weights, inputs and dtype.

Backends (each recorded with the backend that *actually* executed):
  cpu_ref           reference backend on CPU
  gpu_ref           reference backend on the same GPU
  gpu_ref_compile   torch.compile(reference) on the same GPU (compile time
                    reported separately from steady-state time)
  gpu_native        native C++/CUDA kernels (backend="cuda")

Scenarios:
  infer            forward under torch.inference_mode()
  fwd_bwd          forward + MSE + backward          (no optimizer.step)
  fwd_bwd_reg      forward + MSE + activation reg + backward via the public
                   API: model(x) and model.regularization_loss(x=x)
  fwd_bwd_reg_fused  same objective through forward_with_regularization()
                   when the package provides it
  grid_update      model.update_grid(x)

Timing per scenario (all repeated; median / mean / std / var / p10 / p90):
  wall_ms          host perf_counter around one iteration incl. Python
                   overhead and a final synchronize (end-to-end)
  cuda_event_ms    CUDA-event span of one iteration on the stream
  gpu_kernel_ms    sum of GPU kernel/memcpy busy time per iteration
                   (torch.profiler); excludes idle gaps from Python overhead

Memory per GPU scenario: allocator baseline before the measured iteration,
peak, peak increment, bytes still allocated afterwards, and gradient bytes.
A separate training breakdown (with optimizer.step) reports parameters,
gradients, Adam state and transient memory.

Usage:
  python benchmarks/bench_kan.py --pkg-root . --tag after \
      --output benchmarks/results_after.json
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(name="L-B1-16-G5", kind="layer", B=1, I=16, O=16, G=5, k=3),
    dict(name="L-B256-64-G5", kind="layer", B=256, I=64, O=64, G=5, k=3),
    dict(name="L-B1024-256-G5", kind="layer", B=1024, I=256, O=256, G=5, k=3),
    dict(name="L-B256-64-G64", kind="layer", B=256, I=64, O=64, G=64, k=3),
    dict(name="L-B4096-256-G5", kind="layer", B=4096, I=256, O=256, G=5, k=3, gpu_only=True),
    dict(name="KAN-16-32-8-B256", kind="kan", B=256, widths=[16, 32, 8], G=5, k=3),
    dict(name="KAN-16-32-8-B4096", kind="kan", B=4096, widths=[16, 32, 8], G=5, k=3),
]

SCENARIOS = ["infer", "fwd_bwd", "fwd_bwd_reg", "fwd_bwd_reg_fused", "grid_update"]
BACKENDS = ["cpu_ref", "gpu_ref", "gpu_ref_compile", "gpu_native"]
REG = (1e-2, 1e-2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkg-root", default=_REPO,
                   help="directory containing the spline_kan package to benchmark")
    p.add_argument("--tag", default="current")
    p.add_argument("--output", default=os.path.join(_HERE, "results_current.json"))
    p.add_argument("--configs", default="", help="comma-separated config names (default: all)")
    p.add_argument("--backends", default=",".join(BACKENDS))
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cpu-reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


ARGS = parse_args()
sys.path.insert(0, os.path.abspath(ARGS.pkg_root))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import spline_kan  # noqa: E402
from spline_kan import KAN, KANLinear  # noqa: E402
from spline_kan import kan_linear as _kl  # noqa: E402

DTYPE = getattr(torch, ARGS.dtype)
HAS_GPU = torch.cuda.is_available()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def get_env_info():
    info = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "spline_kan_file": spline_kan.__file__,
        "native_ext_loaded": bool(_kl._CUDA_EXT_LOADED),
        "native_ext_file": getattr(_kl._C, "__file__", None),
        "native_ext_sha256_16": _sha(_kl._C.__file__) if _kl._C is not None else None,
        "dtype": ARGS.dtype,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if HAS_GPU:
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True).strip()
        except Exception:
            info["driver"] = None
    return info


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return {
        "n": n,
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.stdev(xs) if n > 1 else 0.0,
        "var": statistics.variance(xs) if n > 1 else 0.0,
        "p10": xs[int(0.1 * (n - 1))],
        "p90": xs[int(0.9 * (n - 1))],
        "min": xs[0],
    }


# ---------------------------------------------------------------------------
# Model construction: one master copy -> identical weights everywhere
# ---------------------------------------------------------------------------

def build_master(cfg):
    torch.manual_seed(ARGS.seed)
    if cfg["kind"] == "layer":
        m = KANLinear(cfg["I"], cfg["O"], grid_size=cfg["G"], spline_order=cfg["k"],
                      backend="reference")
        in_f = cfg["I"]
    else:
        m = KAN(cfg["widths"], grid_size=cfg["G"], spline_order=cfg["k"], backend="reference")
        in_f = cfg["widths"][0]
    m = m.to(DTYPE)
    g = torch.Generator().manual_seed(ARGS.seed + 1)
    x = (torch.rand(cfg["B"], in_f, generator=g, dtype=DTYPE) * 2 - 1)
    out_f = cfg["O"] if cfg["kind"] == "layer" else cfg["widths"][-1]
    y = torch.randn(cfg["B"], out_f, generator=g, dtype=DTYPE)
    return m, x, y


def clone_model(cfg, master, backend, device):
    if cfg["kind"] == "layer":
        m = KANLinear(cfg["I"], cfg["O"], grid_size=cfg["G"], spline_order=cfg["k"],
                      backend=backend)
    else:
        m = KAN(cfg["widths"], grid_size=cfg["G"], spline_order=cfg["k"], backend=backend)
    m = m.to(device=device, dtype=DTYPE)
    m.load_state_dict({k: v.to(device) for k, v in master.state_dict().items()})
    return m


def layers_of(m):
    return list(m.layers) if isinstance(m, KAN) else [m]


def actual_backend(m, x):
    """Backend that really executed, from the layers themselves."""
    used = set()
    for l in layers_of(m):
        lb = getattr(l, "last_backend", None)
        if lb is None:  # baseline code has no last_backend attribute
            lb = "cuda" if (getattr(l, "_use_cuda", False) and x.is_cuda) else "reference"
        used.add(lb)
    return "+".join(sorted(used))


# ---------------------------------------------------------------------------
# Scenario iterations
# ---------------------------------------------------------------------------

def make_iteration(scenario, m, x, y, compiled=False):
    """Returns (callable, uses_grad).  Raises NotImplementedError if unsupported."""
    if scenario == "infer":
        fwd = torch.compile(m) if compiled else m

        def it():
            with torch.inference_mode():
                return fwd(x)
        return it, False

    if scenario == "fwd_bwd":
        def loss_fn():
            return F.mse_loss(m(x), y)
    elif scenario == "fwd_bwd_reg":
        def loss_fn():
            return F.mse_loss(m(x), y) + m.regularization_loss(*REG, x=x)
    elif scenario == "fwd_bwd_reg_fused":
        if not hasattr(m, "forward_with_regularization"):
            raise NotImplementedError("forward_with_regularization not provided")

        def loss_fn():
            out, reg = m.forward_with_regularization(x, *REG)
            return F.mse_loss(out, y) + reg
    elif scenario == "grid_update":
        def it():
            m.update_grid(x)
        return it, False
    else:
        raise ValueError(scenario)

    lf = torch.compile(loss_fn) if compiled else loss_fn

    def it():
        lf().backward()
    return it, True


def zero_grads(m):
    for p in m.parameters():
        p.grad = None


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def time_iterations(it, m, uses_grad, dev, reps, warmup):
    for _ in range(warmup):
        if uses_grad:
            zero_grads(m)
        it()
    sync(dev)
    wall, ev = [], []
    for _ in range(reps):
        if uses_grad:
            zero_grads(m)
        sync(dev)
        if dev == "cuda":
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            e0.record()
            it()
            e1.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ev.append(e0.elapsed_time(e1))
        else:
            t0 = time.perf_counter()
            it()
            t1 = time.perf_counter()
        wall.append((t1 - t0) * 1e3)
    out = {"wall_ms": stats(wall)}
    if ev:
        out["cuda_event_ms"] = stats(ev)
    return out


def gpu_kernel_time(it, m, uses_grad, reps):
    """Sum of device-side kernel/memcpy/memset durations per iteration (ms)."""
    per_iter = []
    for _ in range(reps):
        if uses_grad:
            zero_grads(m)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            it()
            torch.cuda.synchronize()
        us = sum(e.time_range.elapsed_us() for e in prof.events()
                 if e.device_type == DeviceType.CUDA)
        per_iter.append(us / 1e3)
    return stats(per_iter)


def tensor_bytes(ts):
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def measure_memory(it, m, uses_grad):
    if uses_grad:
        zero_grads(m)
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    it()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()
    grads = tensor_bytes([p.grad for p in m.parameters()])
    mb = 2 ** 20
    return {
        "baseline_MiB": base / mb,
        "peak_MiB": peak / mb,
        "peak_increment_MiB": (peak - base) / mb,
        "retained_after_MiB": (after - base) / mb,
        "grad_MiB": grads / mb,
        "baseline_includes": "parameters, buffers, inputs/targets, CUDA allocator state "
                             "of earlier runs in this process (grads cleared)",
    }


def train_memory_breakdown(cfg, master, x, y, backend, scenario):
    """One Adam training step (optimizer.step executed) with a memory breakdown."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a0 = torch.cuda.memory_allocated()
    m = clone_model(cfg, master, backend, "cuda")
    xg, yg = x.cuda(), y.cuda()
    a1 = torch.cuda.memory_allocated()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    it, _ = make_iteration(scenario, m, xg, yg)

    def step():
        opt.zero_grad(set_to_none=True)
        it()
        opt.step()

    step()  # materialise Adam state
    torch.cuda.synchronize()
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    params = tensor_bytes(list(m.parameters()))
    buffers = tensor_bytes(list(m.buffers()))
    grads = tensor_bytes([p.grad for p in m.parameters()])
    opt_state = sum(tensor_bytes([v for v in s.values() if torch.is_tensor(v) and v.is_cuda])
                    for s in opt.state.values())
    mb = 2 ** 20
    res = {
        "optimizer": "Adam", "optimizer_step_executed": True, "scenario": scenario,
        "parameters_MiB": params / mb,
        "buffers_MiB": buffers / mb,
        "inputs_targets_MiB": (a1 - a0 - params - buffers) / mb,
        "gradients_MiB": grads / mb,
        "optimizer_state_MiB": opt_state / mb,
        "baseline_before_step_MiB": base / mb,
        "peak_MiB": peak / mb,
        "peak_increment_MiB": (peak - base) / mb,
        "transient_MiB": (peak - base - grads) / mb,
        "note": "transient = peak increment minus gradients created by the step "
                "(activations saved for backward, workspace, temporaries)",
        "actual_backend": actual_backend(m, xg),
    }
    del m, opt, it, xg, yg
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_one(cfg, master, x, y, bname, scenario):
    dev = "cpu" if bname == "cpu_ref" else "cuda"
    backend = "cuda" if bname == "gpu_native" else "reference"
    compiled = bname == "gpu_ref_compile"
    m = clone_model(cfg, master, backend, dev)
    xd, yd = x.to(dev), y.to(dev)
    res = {"backend_label": bname, "requested_backend": backend, "device": dev,
           "torch_compile": compiled, "optimizer_step_executed": False}
    if compiled:
        torch._dynamo.reset()  # cold compile per (config, scenario)
    try:
        t_c0 = time.perf_counter()
        it, uses_grad = make_iteration(scenario, m, xd, yd, compiled=compiled)
        if compiled:
            it()  # first call triggers compilation
            sync(dev)
            res["compile_first_call_s"] = time.perf_counter() - t_c0
            res["compile_note"] = ("cold Dynamo/Inductor compile in this process; the "
                                   "on-disk Inductor/Triton cache may already be warm")
    except NotImplementedError as e:
        return {**res, "skipped": str(e)}
    except Exception as e:  # e.g. torch.compile failure
        return {**res, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    reps = ARGS.cpu_reps if dev == "cpu" else ARGS.reps
    try:
        res.update(time_iterations(it, m, uses_grad, dev, reps, ARGS.warmup))
        res["actual_backend"] = actual_backend(m, xd)
        if dev == "cuda":
            res["gpu_kernel_ms"] = gpu_kernel_time(it, m, uses_grad, min(reps, 10))
            res["memory"] = measure_memory(it, m, uses_grad)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    del m, it
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    env = get_env_info()
    print(json.dumps(env, indent=2))
    want_cfg = set(filter(None, ARGS.configs.split(",")))
    backends = [b for b in ARGS.backends.split(",") if b]
    scenarios = [s for s in ARGS.scenarios.split(",") if s]
    if not HAS_GPU:
        backends = [b for b in backends if b == "cpu_ref"]
    if not _kl._CUDA_EXT_LOADED:
        backends = [b for b in backends if b != "gpu_native"]
    results = []
    for cfg in CONFIGS:
        if want_cfg and cfg["name"] not in want_cfg:
            continue
        master, x, y = build_master(cfg)
        row = {"config": cfg, "runs": {}, "train_memory": {}}
        print(f"\n=== {cfg['name']} ===", flush=True)
        for sc in scenarios:
            row["runs"][sc] = {}
            for b in backends:
                if b == "cpu_ref" and cfg.get("gpu_only"):
                    continue
                if b == "gpu_ref_compile" and sc == "grid_update":
                    continue
                r = run_one(cfg, master, x, y, b, sc)
                row["runs"][sc][b] = r
                if "wall_ms" in r:
                    ev = r.get("cuda_event_ms", {}) or {}
                    gk = r.get("gpu_kernel_ms", {}) or {}
                    mem = r.get("memory", {}) or {}
                    print(f"  {sc:18s} {b:16s} [{r.get('actual_backend')}] "
                          f"wall={r['wall_ms']['median']:.3f}ms "
                          f"event={ev.get('median', float('nan')):.3f}ms "
                          f"kernel={gk.get('median', float('nan')):.3f}ms "
                          f"peak+={mem.get('peak_increment_MiB', float('nan')):.1f}MiB"
                          + (f" compile={r['compile_first_call_s']:.1f}s"
                             if "compile_first_call_s" in r else ""), flush=True)
                else:
                    print(f"  {sc:18s} {b:16s} {r.get('skipped') or r.get('error')}", flush=True)
        if HAS_GPU:
            for sc in ("fwd_bwd", "fwd_bwd_reg", "fwd_bwd_reg_fused"):
                if sc not in scenarios:
                    continue
                for b, be in (("gpu_ref", "reference"), ("gpu_native", "cuda")):
                    if b not in backends:
                        continue
                    try:
                        row["train_memory"][f"{sc}/{b}"] = train_memory_breakdown(
                            cfg, master, x, y, be, sc)
                    except NotImplementedError as e:
                        row["train_memory"][f"{sc}/{b}"] = {"skipped": str(e)}
        results.append(row)
    out = {"tag": ARGS.tag, "env": env, "args": vars(ARGS), "reg_coefficients": REG,
           "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.output)), exist_ok=True)
    with open(ARGS.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {ARGS.output}")


if __name__ == "__main__":
    main()
