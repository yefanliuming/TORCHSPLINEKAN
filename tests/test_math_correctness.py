"""
Phase-1 regression tests: mathematical properties the original suite missed.

Each test names the defect it guards against.  All gradient checks use FP64
and sample points chosen away from spline knots (B-splines of order k are
only C^{k-1} at knots, so finite differences straddling a knot are invalid).
"""
import math

import pytest
import torch
import torch.nn as nn

from spline_kan import KAN, KANLinear
from spline_kan import kan_linear as kl
from spline_kan.reference_backend import compute_basis, curve2coeff
from torch.func import functional_call

CUDA = torch.cuda.is_available() and kl._CUDA_EXT_LOADED
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA extension not available")


def _away_from_knots(grid_row, n, lo, hi, gen=None, min_gap=0.02):
    """n points in (lo, hi) at least min_gap away from every knot."""
    out = []
    g = grid_row.double()
    while len(out) < n:
        v = torch.empty(1, dtype=torch.float64).uniform_(lo, hi, generator=gen)
        if (g - v).abs().min() > min_gap:
            out.append(v)
    return torch.cat(out)


def _nonuniform_grid(in_features, G, k, seed=0):
    """Strictly increasing, clearly non-uniform knot vectors [I, G+2k+1]."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(in_features):
        gaps = torch.rand(G + 2 * k, generator=gen, dtype=torch.float64) * 0.9 + 0.1
        gaps = gaps / gaps[k:k + G].sum() * 2.0          # interior spans [-1, 1]
        t = torch.cat([torch.zeros(1, dtype=torch.float64), gaps.cumsum(0)])
        t = t - t[k] - 1.0
        rows.append(t)
    return torch.stack(rows)


def _set_grid(layer, grid):
    with torch.no_grad():
        layer.grid.copy_(grid.to(layer.grid))
        layer._ref.grid.copy_(grid.to(layer._ref.grid))


def _clone_to(layer_src, backend, device, **kw):
    dst = KANLinear(layer_src.in_features, layer_src.out_features,
                    grid_size=layer_src.grid_size, spline_order=layer_src.spline_order,
                    backend=backend, **kw).to(device=device, dtype=layer_src.grid.dtype)
    dst.load_state_dict({k: v.to(device) for k, v in layer_src.state_dict().items()})
    return dst


# ---------------------------------------------------------------------------
# Finding 2 — KAN.regularization_loss detached hidden activations
# ---------------------------------------------------------------------------

def _direct_layer_reg(layer, x, mu1, mu2):
    """Paper eqs. 19-21 written out independently of the library."""
    act = layer._ref.base_activation
    xf = x.reshape(-1, layer.in_features)
    H = compute_basis(xf, layer.grid, layer.spline_order)            # [N,I,R]
    W = layer.spline_weight
    if layer.enable_standalone_scale_spline:
        W = W * layer.spline_scaler[..., None]
    phi = layer.base_weight[None] * act(xf)[:, None, :] + torch.einsum("nir,oir->noi", H, W)
    a = phi.abs().mean(0)                                            # |phi|_1  [O,I]
    A = a.sum()                                                      # |Phi|_1
    p = a / A
    S = -(p * p.log()).sum()
    return mu1 * A + mu2 * S


class _RegCall(nn.Module):
    """Exposes regularization_loss as forward() so functional_call can drive it."""

    def __init__(self, m, mu1, mu2):
        super().__init__()
        self.m, self.mu1, self.mu2 = m, mu1, mu2

    def forward(self, x):
        return self.m.regularization_loss(self.mu1, self.mu2, x=x)


class TestRegularizationGradients:

    def test_single_layer_matches_paper_and_fd(self):
        torch.manual_seed(0)
        layer = KANLinear(3, 4, grid_size=4, spline_order=3, backend="reference").double()
        x = torch.stack([_away_from_knots(layer.grid[i], 24, -0.95, 0.95) for i in range(3)], 1)
        mu1, mu2 = 0.7, 1.3
        got = layer.regularization_loss(mu1, mu2, x=x)
        assert torch.allclose(got, _direct_layer_reg(layer, x, mu1, mu2), rtol=1e-12, atol=1e-12)

        names = ["base_weight", "spline_weight", "spline_scaler"]

        def f(bw, sw, sc, xin):
            return functional_call(_RegCall(layer, mu1, mu2),
                                   dict(zip(["m." + n for n in names], (bw, sw, sc))), (xin,))

        inputs = tuple(getattr(layer, n).detach().clone().requires_grad_() for n in names)
        inputs += (x.clone().requires_grad_(),)
        assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)

    def test_multilayer_gradient_reaches_earlier_layers(self):
        """d(sum_l reg_l)/d theta_0 must include the path through layer-1's input."""
        torch.manual_seed(1)
        model = KAN([2, 3, 2], grid_size=3, spline_order=2, backend="reference").double()
        x = torch.stack([_away_from_knots(model.layers[0].grid[i], 16, -0.9, 0.9) for i in range(2)], 1)

        # Hidden values must also avoid layer-1 knots for FD validity.
        h = model.layers[0](x).detach()
        dist = (h[:, :, None] - model.layers[1].grid[None]).abs().min()
        assert dist > 1e-3

        w0 = model.layers[0].spline_weight
        b0 = model.layers[0].base_weight
        wrapper = _RegCall(model, 0.5, 0.25)

        def total(sw, bw):
            return functional_call(wrapper, {"m.layers.0.spline_weight": sw,
                                             "m.layers.0.base_weight": bw}, (x,))

        sw = w0.detach().clone().requires_grad_()
        bw = b0.detach().clone().requires_grad_()
        assert torch.autograd.gradcheck(total, (sw, bw), eps=1e-6, atol=1e-6, rtol=1e-5)

        # The cross-layer term is genuinely non-zero (the old no_grad version dropped it).
        reg1 = model.layers[1].regularization_loss(0.5, 0.25, x=model.layers[0](x))
        g_cross = torch.autograd.grad(reg1, w0)[0]
        assert g_cross.abs().max() > 1e-6

    def test_multilayer_value_equals_sum_of_layers_on_actual_inputs(self):
        torch.manual_seed(2)
        model = KAN([3, 5, 4, 2], grid_size=4, spline_order=3, backend="reference").double()
        x = torch.rand(40, 3, dtype=torch.float64) * 1.8 - 0.9
        expected, h = 0.0, x
        for layer in model.layers:
            expected = expected + _direct_layer_reg(layer, h, 1.0, 2.0)
            h = layer(h)
        got = model.regularization_loss(1.0, 2.0, x=x)
        assert torch.allclose(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Finding 3 — native path ignored custom base_activation
# ---------------------------------------------------------------------------

def _manual_forward(layer, x, act):
    H = compute_basis(x, layer.grid, layer.spline_order)
    W = layer.spline_weight * layer.spline_scaler[..., None]
    return act(x) @ layer.base_weight.t() + torch.einsum("bir,oir->bo", H, W)


class TestBaseActivation:

    @pytest.mark.parametrize("act_cls", [nn.Identity, nn.Tanh, nn.SiLU])
    def test_reference_executes_activation(self, act_cls):
        torch.manual_seed(3)
        layer = KANLinear(4, 3, base_activation=act_cls, backend="reference").double()
        x = torch.randn(10, 4, dtype=torch.float64) * 0.8
        assert torch.allclose(layer(x), _manual_forward(layer, x, act_cls()), atol=1e-12)

    @needs_cuda
    @pytest.mark.parametrize("backend", ["cuda", "auto"])
    @pytest.mark.parametrize("act_cls", [nn.Identity, nn.Tanh])
    def test_native_executes_activation(self, act_cls, backend):
        torch.manual_seed(4)
        ref = KANLinear(4, 3, base_activation=act_cls, backend="reference").double()
        cu = _clone_to(ref, backend, "cuda", base_activation=act_cls)
        assert cu.backend_for(torch.empty(1, device="cuda")) == "cuda"
        x = torch.randn(10, 4, dtype=torch.float64) * 0.8
        y = cu(x.cuda())
        assert cu.last_backend == "cuda"
        assert torch.allclose(y.cpu(), _manual_forward(ref, x, act_cls()), atol=1e-12)

        # first-order gradients through the custom activation
        xc = x.cuda().requires_grad_()
        assert torch.autograd.gradcheck(lambda t: cu(t), (xc,), eps=1e-6, atol=1e-6)

    @needs_cuda
    def test_identity_base_gradient_matches_reference(self):
        torch.manual_seed(5)
        ref = KANLinear(5, 2, base_activation=nn.Identity, backend="reference").double()
        cu = _clone_to(ref, "cuda", "cuda", base_activation=nn.Identity)
        x = torch.randn(12, 5, dtype=torch.float64) * 0.7
        for m, xx in ((ref, x), (cu, x.cuda())):
            m(xx).pow(2).sum().backward()
        for n, p in ref.named_parameters():
            q = dict(cu.named_parameters())[n]
            assert torch.allclose(p.grad, q.grad.cpu(), atol=1e-10), n


# ---------------------------------------------------------------------------
# Finding 4 — native double-backward silently returned zero
# ---------------------------------------------------------------------------

def _fit_cubic_layer(grid, k, device, backend):
    """Pure-spline layer (base_weight = 0) whose curve is exactly f(x)=x^3-x."""
    layer = KANLinear(1, 1, grid_size=grid.shape[1] - 2 * k - 1, spline_order=k,
                      backend=backend).double()
    _set_grid(layer, grid)
    xs = torch.linspace(-1, 1, 400, dtype=torch.float64).unsqueeze(1)
    ys = (xs ** 3 - xs).t().unsqueeze(-1)                         # [O=1,B,I=1]
    with torch.no_grad():
        layer.base_weight.zero_()
        layer.spline_scaler.fill_(1.0)
        layer.spline_weight.copy_(curve2coeff(xs, ys, layer.grid, k))
    return layer.to(device)


class TestHigherOrder:

    @pytest.mark.parametrize("device,backend", [("cpu", "reference")] +
                             ([("cuda", "cuda"), ("cuda", "reference")] if CUDA else []))
    @pytest.mark.parametrize("uniform", [True, False])
    def test_pure_spline_second_derivative_exact(self, device, backend, uniform):
        """Cubic splines reproduce x^3-x exactly, so u_xx must equal 6x."""
        k, G = 3, 6
        grid = (_nonuniform_grid(1, G, k, seed=7) if not uniform else
                (-1 + (torch.arange(G + 2 * k + 1, dtype=torch.float64) - k) * 2 / G)[None])
        layer = _fit_cubic_layer(grid, k, device, backend)
        x = _away_from_knots(grid[0], 9, -0.95, 0.95).to(device).unsqueeze(1).requires_grad_()
        u = layer(x)
        ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
        assert torch.allclose(u, x ** 3 - x, atol=1e-9)
        assert torch.allclose(ux, 3 * x ** 2 - 1, atol=1e-8)
        assert torch.allclose(uxx, 6 * x, atol=1e-7), (uxx - 6 * x).abs().max()

    @needs_cuda
    def test_native_gradgradcheck(self):
        torch.manual_seed(8)
        ref = KANLinear(3, 2, grid_size=4, spline_order=3, backend="reference").double()
        cu = _clone_to(ref, "cuda", "cuda")
        with torch.no_grad():
            cu.base_weight.zero_()     # spline-only: base branch cannot mask errors
        x = torch.stack([_away_from_knots(ref.grid[i], 5, -0.9, 0.9) for i in range(3)], 1)
        x = x.cuda().requires_grad_()
        assert torch.autograd.gradgradcheck(lambda t: cu(t), (x,), eps=1e-6, atol=1e-5)

    @needs_cuda
    def test_native_pde_loss_param_grads_match_reference(self):
        torch.manual_seed(9)
        ref = KAN([1, 6, 1], grid_size=5, backend="reference").double()
        cu = KAN([1, 6, 1], grid_size=5, backend="cuda").double().cuda()
        cu.load_state_dict({k: v.cuda() for k, v in ref.state_dict().items()})
        x0 = torch.linspace(-0.87, 0.83, 17, dtype=torch.float64).unsqueeze(1)
        grads = []
        for m, dev in ((ref, "cpu"), (cu, "cuda")):
            x = x0.to(dev).requires_grad_()
            u = m(x)
            ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
            loss = ((uxx + math.pi ** 2 * torch.sin(math.pi * x)) ** 2).mean()
            grads.append(torch.autograd.grad(loss, list(m.parameters())))
        for gr, gc in zip(*grads):
            assert torch.allclose(gr, gc.cpu(), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Non-uniform grids, k = 1, 2, 3: values and first-order gradients
# ---------------------------------------------------------------------------

class TestNonUniformGrid:

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_partition_of_unity_and_local_support(self, k):
        G = 7
        grid = _nonuniform_grid(2, G, k, seed=k)
        x = torch.stack([_away_from_knots(grid[i], 50, -1.0, 1.0) for i in range(2)], 1)
        H = compute_basis(x, grid, k)
        assert torch.allclose(H.sum(-1), torch.ones_like(x), atol=1e-12)
        assert ((H > 0).sum(-1) <= k + 1).all()
        assert (H >= 0).all()

    @needs_cuda
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_native_matches_reference(self, k):
        torch.manual_seed(10 + k)
        G = 7
        ref = KANLinear(3, 4, grid_size=G, spline_order=k, backend="reference").double()
        _set_grid(ref, _nonuniform_grid(3, G, k, seed=20 + k))
        cu = _clone_to(ref, "cuda", "cuda")
        # include points outside [-1, 1] and inside the extended knot range
        x = torch.stack([_away_from_knots(ref.grid[i], 30, -1.3, 1.3) for i in range(3)], 1)

        xr = x.clone().requires_grad_()
        xc = x.cuda().requires_grad_()
        yr, yc = ref(xr), cu(xc)
        assert torch.allclose(yr, yc.cpu(), atol=1e-12)
        w = torch.randn_like(yr)
        yr.backward(w); yc.backward(w.cuda())
        assert torch.allclose(xr.grad, xc.grad.cpu(), atol=1e-10)
        for n, p in ref.named_parameters():
            assert torch.allclose(p.grad, dict(cu.named_parameters())[n].grad.cpu(), atol=1e-10), n

        assert torch.autograd.gradcheck(lambda t: cu(t), (x[:6].cuda().requires_grad_(),),
                                        eps=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Finding 5 — fixed-size local arrays without bound checks
# ---------------------------------------------------------------------------

@needs_cuda
class TestNativeBounds:

    def test_largest_supported_grid_correct(self):
        k = 3
        G = kl.CUDA_MAX_DEG0_BASIS - 2 * k          # exactly at the limit
        torch.manual_seed(11)
        ref = KANLinear(2, 2, grid_size=G, spline_order=k, backend="reference")
        cu = _clone_to(ref, "cuda", "cuda")
        x = torch.rand(64, 2) * 2 - 1
        assert torch.allclose(ref(x), cu(x.cuda()).cpu(), atol=1e-5)

    def test_explicit_cuda_rejects_oversized_grid(self):
        G = kl.CUDA_MAX_DEG0_BASIS - 2 * 3 + 1
        with pytest.raises(ValueError, match="CUDA kernel limit"):
            KANLinear(2, 2, grid_size=G, spline_order=3, backend="cuda")

    def test_auto_falls_back_for_oversized_grid(self):
        G = 200
        torch.manual_seed(12)
        ref = KANLinear(2, 2, grid_size=G, backend="reference")
        auto = _clone_to(ref, "auto", "cuda")
        assert auto.selected_backend == "reference"
        x = torch.rand(32, 2) * 2 - 1
        assert torch.allclose(ref(x), auto(x.cuda()).cpu(), atol=1e-5)
        assert auto.last_backend == "reference"

    def test_raw_op_checks_before_launch(self):
        """The C++ entry point itself validates; a bad call must raise, not corrupt."""
        k, G = 3, 100
        x = torch.rand(8, 2, device="cuda")
        grid = torch.linspace(-2, 2, G + 2 * k + 1, device="cuda").expand(2, -1).contiguous()
        bw = torch.zeros(2, 2, device="cuda")
        sw = torch.zeros(2, 2, G + k, device="cuda")
        sc = torch.ones(2, 2, device="cuda")
        with pytest.raises(RuntimeError, match="G \\+ 2k"):
            kl._C.kan_forward(x, grid, bw, sw, sc, k, True)
        with pytest.raises(RuntimeError, match="G \\+ 2k"):
            kl._C.kan_apply(x, grid, bw, sw, sc, k, True)
        torch.cuda.synchronize()   # no asynchronous fault left behind

    def test_raw_op_shape_validation(self):
        x = torch.rand(8, 2, device="cuda")
        grid = torch.linspace(-2, 2, 12, device="cuda").expand(2, -1).contiguous()
        sw = torch.zeros(2, 2, 7, device="cuda")       # wrong R (should be 8)
        with pytest.raises(RuntimeError, match="spline_weight.size\\(2\\)"):
            kl._C.kan_forward(x, grid, torch.zeros(2, 2, device="cuda"), sw,
                              torch.ones(2, 2, device="cuda"), 3, True)


# ---------------------------------------------------------------------------
# Backend reporting
# ---------------------------------------------------------------------------

def test_auto_cpu_input_runs_reference():
    layer = KANLinear(4, 3, backend="auto")
    y = layer(torch.randn(5, 4))
    assert layer.last_backend == "reference"
    assert y.shape == (5, 3)
