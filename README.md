# torch-spline-kan

B-spline Kolmogorov-Arnold Networks (KAN) for PyTorch, with an optional C++/CUDA backend for accelerated inference.

## What it is

This library implements KAN layers using B-spline basis functions as described in [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). Each edge in the network learns a univariate B-spline function, giving the model interpretable and adaptable activation functions rather than fixed ones.

Two backends are provided:

- **reference**: Pure PyTorch, runs on CPU or GPU, supports higher-order autograd for PDE applications.
- **cuda**: Native C++/CUDA extension for faster forward and backward passes on GPU.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ toolkit (optional, for GPU backend)

## Install

```bash
pip install .
```

For the CUDA extension, make sure your installed PyTorch version matches the one used at build time. Pin the version to avoid ABI mismatches:

```bash
pip install torch==2.7.1
pip install . --no-build-isolation
```

To target specific GPU architectures:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install . --no-build-isolation
```

To force a CUDA build even when no GPU is detected at install time (e.g. on a CPU-only build machine targeting a GPU deployment):

```bash
FORCE_CUDA=1 pip install . --no-build-isolation
```

CPU-only build (no CUDA toolkit required):

```bash
pip install . --no-build-isolation
```

## Basic usage

```python
import torch
from spline_kan import KANLinear, KAN

# Single KAN layer: 16 inputs, 8 outputs
layer = KANLinear(16, 8, grid_size=5, spline_order=3)
x = torch.randn(32, 16)
y = layer(x)  # [32, 8]

# Multi-layer KAN network
model = KAN([16, 32, 8], grid_size=5, spline_order=3)
y = model(x)  # [32, 8]
```

## Backend selection

Pass `backend` to `KANLinear` or `KAN`:

```python
# Auto-select: CUDA if the extension is loaded, else reference (default)
layer = KANLinear(16, 8, backend="auto")

# Always use the pure-PyTorch reference backend
layer = KANLinear(16, 8, backend="reference")

# Require the CUDA extension (raises RuntimeError if not built, ValueError if
# the configuration is outside the kernel limits, RuntimeError on CPU input)
layer = KANLinear(16, 8, backend="cuda")
```

Check which backend is (or was) used:

```python
print(layer.selected_backend)      # backend used for CUDA inputs
print(layer.backend_for(x))        # backend forward() will use for this x
y = layer(x); print(layer.last_backend)   # backend that actually ran
```

`auto` uses the native kernels only for CUDA inputs and supported configurations
(`grid_size + 2*spline_order <= 71`, `in_features <= 65535`); otherwise it runs the
reference backend. Any `base_activation` is honoured on every backend: the native
kernel fuses SiLU, other activations are evaluated with PyTorch and only the spline
branch runs natively.

## Grid update

After initial training, you can redistribute the knots to better cover the data distribution your model actually sees. The interior knots follow the observed quantile/uniform mixture, preserving nonuniform spacing where the data is dense:

```python
model.update_grid(x_train)
```

**Important:** after `update_grid`, the spline coefficient coordinates change. Recreate the optimizer immediately to clear stale momentum:

```python
model.update_grid(x_train)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

The scaler parameters (`spline_scaler`) and `base_weight` are preserved across grid updates — only the knot positions and unscaled coefficients are changed.

## Higher-order differentiation

The reference backend is fully differentiable through the B-spline recurrence. This makes it suitable for physics-informed neural networks (PINNs) and other PDE applications:

```python
model = KAN([1, 16, 1], backend="reference").double()
x = torch.linspace(-1, 1, 50, requires_grad=True).unsqueeze(1)

u = model(x)
u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

pde_loss = ((u_xx - target)**2).mean()
pde_loss.backward()
```

Higher-order derivatives are exact on both backends. On the native backend the
first-order backward uses the CUDA kernels; when the backward itself is recorded
(`create_graph=True`) it switches to a differentiable ATen formulation, so u_xx and
PDE losses are correct but do not use the custom kernels for the second pass.

Use `float64` for higher-order differentiation to avoid accumulated floating-point error.

## Regularization

Both `KANLinear` and `KAN` expose a regularization loss with two modes.

**`mode="activation"` (default)** evaluates the full edge activation `phi[j,i](u) = base_weight[j,i]*sigma(u) + scaler[j,i]*spline(u)` on a batch of samples, then combines a mean-absolute-activation L1 term and an entropy term normalised globally across all edges in the layer. Pass the network input as `x`:

```python
optimizer.zero_grad()
y = model(x)
task_loss = criterion(y, target)
# x is propagated through the network; each layer evaluates its own actual input.
reg_loss = model.regularization_loss(reg_act=1e-2, reg_entropy=1e-2, x=x)
(task_loss + reg_loss).backward()
optimizer.step()
```

For a single layer, supply the sample that enters that layer:

```python
reg_loss = layer.regularization_loss(reg_act=1e-2, reg_entropy=1e-2, x=x)
```

Hidden activations are not detached: later layers' penalties propagate gradients to
earlier layers' parameters (paper eq. 22).

When you also need the output (the usual training step), compute both in one pass.
This reuses each layer's base activation and spline basis, and on CUDA evaluates
`mean_n |phi|` with a fused kernel that never stores the `[N, O, I]` edge tensor:

```python
y, reg = model.forward_with_regularization(x, reg_act=1e-2, reg_entropy=1e-2)
(criterion(y, target) + reg).backward()
```

`reg_impl` selects the evaluation (same value and gradients in all cases):
`"auto"` (fused on supported CUDA layers, else dense), `"dense"` (materialise phi),
`"fused"` (CUDA kernel, recomputes phi in backward), `"checkpoint"` (pure PyTorch,
chunks of `chunk_size` samples recomputed in backward; lower memory, slower).
`model.regularization_loss(..., x=x)` accepts the same `reg_impl`.

**`mode="coefficient"`** is the legacy mode. It uses the mean absolute scaled spline coefficient per edge, with entropy normalised per output node. It does not require `x`:

```python
reg_loss = layer.regularization_loss(reg_act=1e-2, reg_entropy=1e-2, mode="coefficient")
```

## Inference export, pruning, grid extension

```python
# Fold spline_scaler into spline_weight; returns an independent model (same function,
# O*I fewer parameters per layer). Rebuild from its checkpoint with
# KAN(widths, ..., enable_standalone_scale_spline=False).
infer_model = model.export_inference()

# Physical node pruning (paper App. M, eq. 23) on calibration samples.
res = model.prune(x_calib, threshold=1e-2)
small = res.model            # smaller tensors; res.kept / res.old_to_new map nodes
optimizer = torch.optim.Adam(small.parameters())   # new optimizer: shapes changed

# Grid extension G -> G' (paper App. L): a new model whose splines are least-squares
# fits of the current ones on each layer's actual inputs.
fine = model.extend_grid(20, x_train)
optimizer = torch.optim.Adam(fine.parameters())    # new optimizer: shapes changed
```

`update_grid` keeps its meaning (same G, knots moved in place). `extend_grid` and
`prune` return new models and leave the original untouched; their `state_dict`s have
new shapes, so load them into a model built with the new `grid_size` / widths.

## Experimental (opt-in, changes precision or capacity)

`spline_kan.experimental` provides `MixedPrecisionKANLinear` (FP32 basis, BF16/FP16
GEMMs), `GroupedSplineKANLinear` (spline coefficients shared across output groups) and
`LowRankKANLinear` (rank-q coefficient factorisation). None is used by default; see
`docs/optimization_report.md` for measured accuracy, throughput and VRAM.

Mixed-precision layers use the configured GEMM dtype in both `forward()` and
`forward_with_regularization()`, while the basis and full edge regularizer stay
FP32. `MixedPrecisionKANLinear.from_layer(layer)` copies the activation instance,
configuration, parameters, buffers, frozen parameters and train/eval modes into
an independent layer.

## Save and load

Standard PyTorch state dict, including the adaptive grid buffer:

```python
torch.save(model.state_dict(), "kan_checkpoint.pt")

model2 = KAN([16, 32, 8])
model2.load_state_dict(torch.load("kan_checkpoint.pt"))
```

## ABI compatibility

The CUDA extension is compiled against a specific PyTorch version. If you upgrade PyTorch after building, you must rebuild the extension:

```bash
pip install . --no-build-isolation --force-reinstall
```

## Known limitations

- FP16 and BF16 inputs are not supported. Inputs must be FP32 (or FP64 for gradcheck and PDE use).
- Higher-order autodiff (second derivatives and above) is exact on both backends, but the second pass is composed of ATen ops, not the custom kernels.
- Native CUDA kernels support `grid_size + 2*spline_order <= 71` and `in_features <= 65535`; `backend="cuda"` rejects larger layers, `auto` falls back to reference.
- `spline_order` must be 1, 2, or 3. Order 0 is piecewise constant and does not support gradient flow.
- `update_grid` requires at least 2 samples to estimate quantiles.
# TORCHSPLINEKAN
