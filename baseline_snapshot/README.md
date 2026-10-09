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

# Require the CUDA extension (raises RuntimeError if not built)
layer = KANLinear(16, 8, backend="cuda")
```

Check which backend is active:

```python
print(layer.selected_backend)  # "cuda" or "reference"
```

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

Use `backend="reference"` for higher-order autodiff. The CUDA backend does not guarantee support for second-order gradients through the custom kernel.

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

**`mode="coefficient"`** is the legacy mode. It uses the mean absolute scaled spline coefficient per edge, with entropy normalised per output node. It does not require `x`:

```python
reg_loss = layer.regularization_loss(reg_act=1e-2, reg_entropy=1e-2, mode="coefficient")
```

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
- Higher-order autodiff (second derivatives and above) goes through the reference backend. Even when `backend="cuda"` is selected, PDE-style double differentiation will not benefit from the CUDA kernel.
- `spline_order` must be 1, 2, or 3. Order 0 is piecewise constant and does not support gradient flow.
- `update_grid` requires at least 2 samples to estimate quantiles.
