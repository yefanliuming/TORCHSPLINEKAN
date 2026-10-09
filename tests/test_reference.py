import pytest
import torch
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spline_kan import KANLinear, KAN


def test_basis_sum_to_one():
    """B-spline basis functions should sum to ~1 inside [a,b]"""
    from spline_kan.reference_backend import compute_basis
    I, G, k = 1, 5, 3
    a, b = -1.0, 1.0
    n_knots = G + 2*k + 1
    grid = torch.zeros(I, n_knots)
    for r in range(n_knots):
        grid[0, r] = a + (r - k) * (b - a) / G

    x = torch.linspace(a + 0.01, b - 0.01, 20).unsqueeze(1)  # [20, 1]
    H = compute_basis(x, grid, k)  # [20, 1, G+k]
    basis_sum = H[:, 0, :].sum(dim=1)
    assert torch.allclose(basis_sum, torch.ones(20), atol=1e-5), f"Basis sum not 1: {basis_sum}"


def test_basis_degree_rejection():
    from spline_kan.reference_backend import compute_basis
    grid = torch.zeros(1, 12)
    x = torch.zeros(5, 1)
    with pytest.raises((ValueError, RuntimeError)):
        compute_basis(x, grid, 4)  # degree 4 not supported


def test_kan_linear_forward_shape():
    for B, I, O in [(1, 4, 4), (32, 16, 8), (100, 8, 16)]:
        layer = KANLinear(I, O, grid_size=5, spline_order=3, backend="reference")
        x = torch.randn(B, I)
        y = layer(x)
        assert y.shape == (B, O), f"Expected ({B},{O}), got {y.shape}"


def test_kan_linear_leading_dims():
    layer = KANLinear(8, 4, backend="reference")
    x = torch.randn(3, 5, 8)
    y = layer(x)
    assert y.shape == (3, 5, 4)


def test_kan_forward_shape():
    model = KAN([16, 32, 8], backend="reference")
    x = torch.randn(10, 16)
    y = model(x)
    assert y.shape == (10, 8)


def test_gradcheck_kan_linear():
    """FP64 gradcheck away from knots"""
    layer = KANLinear(4, 3, grid_size=3, spline_order=3, backend="reference").double()
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True) * 0.5

    def fn(x_):
        return layer(x_)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3), "gradcheck failed"


def test_gradgradcheck_kan_linear():
    layer = KANLinear(3, 2, grid_size=3, spline_order=3, backend="reference").double()
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True) * 0.4

    def fn(x_):
        return layer(x_)

    assert torch.autograd.gradgradcheck(fn, (x,), eps=1e-6, atol=1e-3), "gradgradcheck failed"


def test_regularization_loss_no_nan():
    layer = KANLinear(4, 4, backend="reference")
    with torch.no_grad():
        layer.spline_weight.zero_()
    x = torch.randn(20, 4)
    loss = layer.regularization_loss(x=x)
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    loss.backward()
    # base_weight is in the activation reg graph; spline_scaler is too
    assert layer.spline_scaler.grad is not None


def test_regularization_loss_gradient_flow():
    layer = KANLinear(4, 4, backend="reference")
    x = torch.randn(20, 4)
    loss = layer.regularization_loss(x=x)
    loss.backward()
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.spline_weight.grad).all()


def test_update_grid_basic():
    layer = KANLinear(4, 4, backend="reference")
    x = torch.randn(100, 4)
    layer.update_grid(x)
    assert layer.spline_weight.shape == (4, 4, 5+3)
    assert not torch.isnan(layer.spline_weight).any()


def test_update_grid_scaler_fixed():
    """Grid transfer must use UNSCALED spline_weight, not scaled."""
    layer = KANLinear(4, 4, enable_standalone_scale_spline=True, backend="reference")
    with torch.no_grad():
        layer.spline_scaler.fill_(2.0)
    scaler_before = layer.spline_scaler.clone()
    x = torch.randn(100, 4)
    layer.update_grid(x)
    assert torch.allclose(layer.spline_scaler, scaler_before), "update_grid must preserve spline_scaler"


def test_update_grid_rejects_empty():
    layer = KANLinear(4, 4, backend="reference")
    with pytest.raises((ValueError, RuntimeError)):
        layer.update_grid(torch.randn(0, 4))


def test_update_grid_leading_dims():
    layer = KANLinear(4, 4, backend="reference")
    x = torch.randn(10, 5, 4)
    layer.update_grid(x)  # should not raise


def test_state_dict_roundtrip():
    layer = KANLinear(4, 4, grid_size=5, spline_order=3, backend="reference")
    x = torch.randn(50, 4)
    layer.update_grid(x)
    sd = layer.state_dict()
    layer2 = KANLinear(4, 4, grid_size=5, spline_order=3, backend="reference")
    layer2.load_state_dict(sd)
    assert torch.allclose(layer.grid, layer2.grid)
    assert torch.allclose(layer.spline_weight, layer2.spline_weight)
    x2 = torch.randn(10, 4)
    assert torch.allclose(layer(x2), layer2(x2))


def test_noncontiguous_input():
    layer = KANLinear(4, 4, backend="reference")
    x = torch.randn(10, 8)[:, ::2]  # stride 2, noncontiguous
    assert not x.is_contiguous()
    y = layer(x)
    assert y.shape == (10, 4)


def test_pde_residual_backward():
    """Higher-order: compute u_xx residual and backprop through it."""
    model = KAN([1, 8, 1], grid_size=5, spline_order=3, backend="reference").double()
    x = torch.linspace(-0.9, 0.9, 20, dtype=torch.float64, requires_grad=True).unsqueeze(1)
    u = model(x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    target = -torch.pi**2 * torch.sin(torch.pi * x)
    loss = ((u_xx - target)**2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_optimizer_loop():
    """Small optimizer loop: loss should decrease."""
    model = KAN([4, 8, 1], backend="reference")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(32, 4)
    y_target = torch.randn(32, 1)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        y = model(x)
        loss = ((y - y_target)**2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_invalid_spline_order():
    with pytest.raises((ValueError, RuntimeError)):
        KANLinear(4, 4, spline_order=0)


def test_invalid_grid_size():
    with pytest.raises((ValueError, RuntimeError)):
        KANLinear(4, 4, grid_size=-1)
