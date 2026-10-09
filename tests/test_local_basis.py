"""
Phase-7 tests: local (k+1 active functions) CUDA basis kernels against the
dense Cox-de Boor reference, incl. non-uniform grids, exact knots, the
right boundary, out-of-domain inputs and repeated knots; plus the
grad_x-skipping backward.
"""
import pytest
import torch

from spline_kan import KANLinear
from spline_kan import kan_linear as kl
from spline_kan.reference_backend import compute_basis, compute_basis_derivative

C = kl._C
CUDA = torch.cuda.is_available() and getattr(C, "HAS_FUSED_KERNELS", False)
pytestmark = pytest.mark.skipif(not CUDA, reason="native kernels unavailable")


def _grid(I, G, k, kind, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = G + 2 * k + 1
    if kind == "uniform":
        t = -1 + (torch.arange(n, dtype=torch.float64) - k) * 2 / G
        return t[None].expand(I, -1).contiguous()
    gaps = torch.rand(I, n - 1, generator=g, dtype=torch.float64) * 0.9 + 0.1
    if kind == "repeated":
        gaps[:, k + G // 2] = 0.0          # a zero-length interior span
    t = torch.cat([torch.zeros(I, 1, dtype=torch.float64), gaps.cumsum(1)], 1)
    return t - t[:, k:k + 1] - 1.0


def _points(grid, n=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = grid[:, 0].min().item(), grid[:, -1].max().item()
    span = hi - lo
    x = torch.rand(n, grid.shape[0], generator=g, dtype=torch.float64) * 1.4 * span + lo - 0.2 * span
    return torch.cat([x, grid.t()], 0)    # + every knot exactly (incl. both ends)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("G", [1, 4, 17, 64])
@pytest.mark.parametrize("kind", ["uniform", "nonuniform", "repeated"])
def test_local_basis_and_grad_match_dense(k, G, kind):
    if G + 2 * k > kl.CUDA_MAX_DEG0_BASIS or (kind == "repeated" and G < 2):
        pytest.skip("config outside kernel range")
    grid = _grid(3, G, k, kind, seed=G + k)
    x = _points(grid, seed=k)
    H = C.basis(x.cuda(), grid.cuda(), k).cpu()
    assert torch.allclose(H, compute_basis(x, grid, k), atol=1e-13, rtol=0)
    gH = torch.randn_like(H)
    gx = C.basis_grad(x.cuda(), grid.cuda(), k, gH.cuda()).cpu()
    ref = (gH * compute_basis_derivative(x, grid, k)).sum(-1)
    assert torch.allclose(gx, ref, atol=1e-10 * max(1.0, ref.abs().max().item()), rtol=0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_out_of_domain_is_zero_and_right_end_closed(k):
    grid = _grid(2, 5, k, "nonuniform", seed=3)
    x = torch.stack([grid[:, 0] - 1e-3, grid[:, -1] + 1e-3, grid[:, -1]], 0)
    H = C.basis(x.cuda(), grid.cuda(), k).cpu()
    assert (H[:2] == 0).all()
    # x == t_{N-1}: reference closes the last degree-0 span; every degree-k
    # function supported there is cut off by the knot vector, so both give 0.
    assert torch.allclose(H[2], compute_basis(x[2:], grid, k)[0])
    # x == right end of the interior domain t_{k+G}: partition of unity holds
    xe = grid[:, k + 5][None]
    He = C.basis(xe.cuda(), grid.cuda(), k).cpu()
    assert torch.allclose(He.sum(-1), torch.ones(1, 2, dtype=torch.float64), atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_and_legacy_kernel_give_same_layer_results(dtype):
    torch.manual_seed(0)
    L = KANLinear(16, 8, grid_size=9, backend="cuda").to("cuda", dtype)
    x = (torch.rand(512, 16, device="cuda", dtype=dtype) * 2.4 - 1.2)
    out = {}
    try:
        for impl in (0, 1):
            C.set_basis_impl(impl)
            for p in L.parameters():
                p.grad = None
            xx = x.clone().requires_grad_()
            y = L(xx)
            y.square().sum().backward()
            out[impl] = [y.detach(), xx.grad] + [p.grad for p in L.parameters()]
    finally:
        C.set_basis_impl(1)
    tol = 1e-12 if dtype == torch.float64 else 1e-4
    for a, b in zip(out[0], out[1]):
        assert torch.allclose(a, b, atol=tol, rtol=tol)


def test_backward_skips_unneeded_grads():
    torch.manual_seed(1)
    L = KANLinear(8, 4, backend="cuda").cuda()
    x = torch.rand(64, 8, device="cuda") * 2 - 1          # no requires_grad
    L(x).sum().backward()
    assert all(p.grad is not None for p in L.parameters())
    # frozen params, input grad only
    L2 = KANLinear(8, 4, backend="cuda").cuda().requires_grad_(False)
    xx = x.clone().requires_grad_()
    L2(xx).sum().backward()
    ref = KANLinear(8, 4, backend="reference").cuda()
    ref.load_state_dict(L2.state_dict())
    xr = x.clone().requires_grad_()
    ref(xr).sum().backward()
    assert torch.allclose(xx.grad, xr.grad, atol=1e-5)
    assert all(p.grad is None for p in L2.parameters())


def test_basis_impl_switch_validation():
    with pytest.raises(RuntimeError):
        C.set_basis_impl(7)
    assert C.get_basis_impl() == 1
