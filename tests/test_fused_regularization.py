"""
Phase-3 tests: forward_with_regularization and the low-memory reg paths
must reproduce the dense reference (value + every gradient, incl. the
cross-layer path), keep no graph-attached state on the module, and give
real memory savings on the fused path.
"""
import gc

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spline_kan import KAN, KANLinear
from spline_kan import fused as fz
from spline_kan import kan_linear as kl

CUDA = torch.cuda.is_available() and fz.HAS_FUSED_KERNELS
needs_cuda = pytest.mark.skipif(not CUDA, reason="native fused kernels unavailable")


def _grads(model, x, impl, mu=(0.7, 1.3), chunk=37, use_fwr=True):
    for p in model.parameters():
        p.grad = None
    xx = x.clone().requires_grad_()
    if use_fwr:
        y, r = model.forward_with_regularization(xx, *mu, reg_impl=impl, chunk_size=chunk)
    else:
        y = model(xx)
        r = model.regularization_loss(*mu, x=xx)
    loss = y.square().mean() + r
    loss.backward()
    return [y.detach(), r.detach(), xx.grad] + [p.grad.clone() for p in model.parameters()]


def _assert_close(a, b, rtol):
    for u, v in zip(a, b):
        scale = v.abs().max().clamp(min=1e-30)
        assert ((u - v).abs().max() / scale).item() < rtol


IMPLS_CPU = ["dense", "checkpoint"]
IMPLS_GPU = ["dense", "checkpoint", "fused"]


@pytest.mark.parametrize("act", [nn.SiLU, nn.Identity])
@pytest.mark.parametrize("scaler", [True, False])
def test_fwr_equals_separate_calls_cpu(act, scaler):
    torch.manual_seed(0)
    m = KAN([3, 5, 2], grid_size=4, base_activation=act,
            enable_standalone_scale_spline=scaler, backend="reference").double()
    x = torch.rand(50, 3, dtype=torch.float64) * 2 - 1
    ref = _grads(m, x, None, use_fwr=False)
    for impl in IMPLS_CPU:
        _assert_close(_grads(m, x, impl), ref, 1e-12)


@needs_cuda
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("G", [3, 17, 60])
@pytest.mark.parametrize("act", [nn.SiLU, nn.Tanh])
def test_all_impls_match_dense_multilayer_gpu(k, G, act):
    torch.manual_seed(k * 100 + G)
    m = KAN([4, 6, 3], grid_size=G, spline_order=k, base_activation=act,
            backend="cuda").double().cuda()
    x = (torch.rand(129, 4, dtype=torch.float64) * 2.6 - 1.3).cuda()  # incl. out-of-domain
    ref = _grads(m, x, None, use_fwr=False)          # public separate API
    for impl in IMPLS_GPU:
        _assert_close(_grads(m, x, impl), ref, 1e-10)


@needs_cuda
def test_fused_cross_layer_gradient_present():
    """Layer-1 penalty alone must reach layer-0 params through the fused path."""
    torch.manual_seed(1)
    m = KAN([3, 4, 2], backend="cuda").double().cuda()
    x = torch.rand(40, 3, dtype=torch.float64, device="cuda") * 2 - 1
    h = m.layers[0](x)
    _, r1_fused = m.layers[1].forward_with_regularization(h, 1.0, 1.0, reg_impl="fused")
    _, r1_dense = m.layers[1].forward_with_regularization(h, 1.0, 1.0, reg_impl="dense")
    gf = torch.autograd.grad(r1_fused, list(m.layers[0].parameters()), retain_graph=True)
    gd = torch.autograd.grad(r1_dense, list(m.layers[0].parameters()))
    for a, b in zip(gf, gd):
        assert b.abs().max() > 1e-8
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-14)


@needs_cuda
def test_fused_fp32_matches_dense():
    torch.manual_seed(2)
    m = KANLinear(64, 48, backend="cuda").cuda()
    x = torch.rand(777, 64, device="cuda") * 2 - 1
    _assert_close(_grads(m, x, "fused"), _grads(m, x, "dense"), 5e-5)


@needs_cuda
def test_fused_gradcheck_and_gradgradcheck():
    torch.manual_seed(3)
    m = KANLinear(3, 2, grid_size=4, backend="cuda").double().cuda()
    x = (torch.rand(11, 3, dtype=torch.float64) * 1.8 - 0.9).cuda().requires_grad_()

    def f(t):
        y, r = m.forward_with_regularization(t, 0.5, 2.0, reg_impl="fused")
        return y.sum() + r

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-6)
    assert torch.autograd.gradgradcheck(f, (x,), eps=1e-6, atol=1e-5)


@needs_cuda
@pytest.mark.parametrize("has_base", [True, False])
def test_edge_stat_mixed_second_derivatives(has_base):
    """Checking x alone misses the bilinear input/weight mixed derivatives."""
    opts = dict(device="cuda", dtype=torch.float64, requires_grad=True)
    H = torch.tensor([[[0.3, 0.7]], [[0.6, 0.4]]], **opts)
    W = torch.tensor([[[0.5, 0.9]]], **opts)
    if has_base:
        s = torch.tensor([[0.2], [0.4]], **opts)
        bw = torch.tensor([[0.8]], **opts)
        fn, inputs = fz.edge_abs_mean, (s, H, bw, W)
        a = fn(*inputs)
        dbw = torch.autograd.grad(a.sum(), bw, create_graph=True)[0]
        mixed = torch.autograd.grad(dbw.sum(), s)[0]
        assert torch.allclose(mixed, torch.full_like(s, 0.5))
    else:
        fn = lambda h, w: fz.edge_abs_mean(None, h, None, w)
        inputs = (H, W)
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-6)
    assert torch.autograd.gradgradcheck(fn, inputs, eps=1e-6, atol=1e-6)


@needs_cuda
@pytest.mark.parametrize("scaler", [True, False])
def test_fused_multilayer_joint_hessian_matches_dense(scaler):
    """Check curvature jointly in x and every parameter, with shared ancestors."""
    torch.manual_seed(87)
    model = KAN([2, 3, 2], grid_size=4, backend="cuda",
                enable_standalone_scale_spline=scaler).double().cuda()
    x = torch.tensor([[0.19, -0.33], [0.71, 0.13]], device="cuda",
                     dtype=torch.float64, requires_grad=True)
    inputs = (x,) + tuple(model.parameters())
    directions = tuple(torch.randn_like(t) for t in inputs)
    hessians = []
    for impl in ("dense", "fused"):
        y, reg = model.forward_with_regularization(x, 0.7, 0.3, reg_impl=impl)
        grads = torch.autograd.grad(y.square().sum() + reg, inputs, create_graph=True)
        tangent = sum((g * v).sum() for g, v in zip(grads, directions))
        hessians.append(torch.autograd.grad(tangent, inputs))
    for dense, fused in zip(*hessians):
        assert torch.allclose(dense, fused, rtol=1e-9, atol=1e-10)


@needs_cuda
def test_zero_layer_finite():
    m = KANLinear(4, 3, backend="cuda").cuda()
    with torch.no_grad():
        m.base_weight.zero_(); m.spline_weight.zero_()
    x = torch.randn(20, 4, device="cuda")
    _, r = m.forward_with_regularization(x, 1.0, 1.0, reg_impl="fused")
    r.backward()
    assert abs(r.item()) < 1e-30   # entropy clamp leaves ~n*tiny*log(tiny)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_reg_impl_validation():
    m = KANLinear(4, 3, backend="reference")
    x = torch.randn(5, 4)
    with pytest.raises(ValueError, match="reg_impl"):
        m.forward_with_regularization(x, reg_impl="nope")
    with pytest.raises(RuntimeError, match="fused"):
        m.forward_with_regularization(x, reg_impl="fused")   # CPU input
    with pytest.raises(ValueError, match="chunk_size"):
        m.forward_with_regularization(x, reg_impl="checkpoint", chunk_size=0)


def test_coefficient_mode_unchanged():
    torch.manual_seed(4)
    m = KAN([4, 5, 2], backend="reference")
    x = torch.randn(6, 4)
    y, r = m.forward_with_regularization(x, 0.3, 0.6, mode="coefficient")
    assert torch.allclose(y, m(x))
    expected = sum(l._ref.regularization_loss(0.3, 0.6, mode="coefficient") for l in m.layers)
    assert torch.allclose(r, expected)


def test_no_graph_state_cached_on_module():
    """Nothing tensor-valued that holds a graph may be stored on the module."""
    m = KAN([4, 6, 2], backend="reference")
    before = {k: type(v) for k, v in m.__dict__.items()}
    x = torch.randn(30, 4, requires_grad=True)
    y, r = m.forward_with_regularization(x, 1.0, 1.0)
    (y.sum() + r).backward()
    for layer in m.layers:
        for k, v in layer.__dict__.items():
            if torch.is_tensor(v):
                assert v.grad_fn is None, k
    assert {k: type(v) for k, v in m.__dict__.items()} == before


@needs_cuda
def test_fused_peak_memory_much_lower_than_dense():
    torch.manual_seed(5)
    m = KANLinear(128, 128, backend="cuda").cuda()
    x = torch.rand(2048, 128, device="cuda") * 2 - 1
    peaks = {}
    for impl in ("dense", "fused"):
        for p in m.parameters():
            p.grad = None
        gc.collect(); torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y, r = m.forward_with_regularization(x, 1.0, 1.0, reg_impl=impl)
        (y.sum() + r).backward()
        torch.cuda.synchronize()
        peaks[impl] = torch.cuda.max_memory_allocated() - base
        del y, r
    phi_bytes = 2048 * 128 * 128 * 4
    assert peaks["dense"] > 2 * phi_bytes          # dense keeps several [N,O,I]
    assert peaks["fused"] < 0.5 * phi_bytes        # fused never stores one
