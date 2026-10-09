"""Phase-6 tests: grid extension G_small -> G_large (paper App. L)."""
import pytest
import torch
import torch.nn as nn

from spline_kan import KAN, KANLinear
from spline_kan import kan_linear as kl
from spline_kan.reference_backend import compute_basis, fit_spline_coeffs

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and kl._CUDA_EXT_LOADED else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_nested_uniform_extension_is_exact(device, k):
    """G -> m*G on a uniform grid contains the old knots: spline reproduced exactly."""
    torch.manual_seed(k)
    L = KANLinear(3, 4, grid_size=4, spline_order=k).to(device).double()
    x = torch.rand(400, 3, device=device, dtype=torch.float64) * 2 - 1
    E = L.extend_grid(12, x)
    xt = torch.rand(300, 3, device=device, dtype=torch.float64) * 2 - 1
    assert torch.allclose(E(xt), L(xt), atol=1e-9)
    assert E.spline_weight.shape == (4, 3, 12 + k)
    assert E.grid.shape == (3, 12 + 2 * k + 1)
    assert torch.equal(E.grid, E._ref.grid)
    assert torch.equal(E.base_weight, L.base_weight)
    assert torch.equal(E.spline_scaler, L.spline_scaler)
    assert L.grid_size == 4 and L.spline_weight.shape == (4, 3, 4 + k)   # original untouched


def test_non_nested_extension_small_error_and_converges():
    torch.manual_seed(0)
    L = KANLinear(2, 2, grid_size=5).double()
    x = torch.rand(2000, 2, dtype=torch.float64) * 2 - 1
    xt = torch.rand(500, 2, dtype=torch.float64) * 1.98 - 0.99
    errs = [(L.extend_grid(G, x)(xt) - L(xt)).abs().max().item() for G in (7, 13, 29)]
    assert errs[0] < 1e-2 and errs[-1] < errs[0]


def test_multilayer_uses_actual_hidden_inputs():
    torch.manual_seed(1)
    m = KAN([2, 4, 1], grid_size=3, base_activation=nn.Tanh).double()
    x = torch.rand(1000, 2, dtype=torch.float64) * 2 - 1
    e = m.extend_grid(9, x)
    assert torch.allclose(e(x), m(x), atol=1e-9)
    assert isinstance(e.layers[1]._ref.base_activation, nn.Tanh)
    assert e.layers[0].grid_size == 9 and m.layers[0].grid_size == 3


def test_adaptive_knots_extension_reasonable():
    torch.manual_seed(2)
    L = KANLinear(2, 2, grid_size=4).double()
    x = torch.randn(3000, 2, dtype=torch.float64).clamp(-1, 1)
    E = L.extend_grid(10, x, adapt_knots=True)
    xt = torch.randn(400, 2, dtype=torch.float64).clamp(-0.9, 0.9)
    assert (E(xt) - L(xt)).abs().max() < 2e-2
    assert ((E.grid[:, 1:] - E.grid[:, :-1]) > 0).all()


def test_undersampled_fit_is_finite_and_bounded():
    """Fewer samples than basis functions: ridge keeps the solve stable."""
    torch.manual_seed(3)
    L = KANLinear(2, 3, grid_size=3).double()
    x = torch.rand(5, 2, dtype=torch.float64) * 0.2      # 5 samples, R'=43, tiny range
    E = L.extend_grid(40, x)
    assert torch.isfinite(E.spline_weight).all()
    assert E.spline_weight.abs().max() < 1e3 * L.spline_weight.abs().max()
    assert torch.allclose(E(x), L(x), atol=1e-4)          # interpolates the samples
    # n_dense fills the domain from the known old curve
    E2 = L.extend_grid(40, x, n_dense=2000)
    xt = torch.rand(200, 2, dtype=torch.float64) * 1.9 - 0.95
    assert (E2(xt) - L(xt)).abs().max() < 1e-6


def test_fit_spline_coeffs_rank_deficient_constant_input():
    grid = torch.linspace(-1.6, 1.6, 12, dtype=torch.float64)[None]   # G=5,k=3
    x = torch.full((50, 1), 0.3, dtype=torch.float64)                 # rank 1
    y = torch.full((1, 50, 1), 2.0, dtype=torch.float64)
    c = fit_spline_coeffs(x, y, grid, 3)
    assert torch.isfinite(c).all()
    H = compute_basis(x, grid, 3)
    assert torch.allclose((H[:, 0] * c[0, 0]).sum(-1), y[0, :, 0], atol=1e-6)
    with pytest.raises(ValueError):
        fit_spline_coeffs(x, y, grid, 3, ridge=0.0)


def test_extended_model_trains_with_new_optimizer_and_checkpoint():
    torch.manual_seed(4)
    m = KAN([2, 3, 1], grid_size=3)
    x = torch.rand(256, 2) * 2 - 1
    y = torch.sin(3 * x[:, :1]) * x[:, 1:]
    e = m.extend_grid(10, x)
    with pytest.raises(RuntimeError):
        m.load_state_dict(e.state_dict())          # shapes changed
    fresh = KAN([2, 3, 1], grid_size=10)
    fresh.load_state_dict(e.state_dict())
    assert torch.allclose(fresh(x), e(x))
    opt = torch.optim.Adam(e.parameters(), lr=1e-2)
    l0 = None
    for _ in range(50):
        opt.zero_grad()
        loss = ((e(x) - y) ** 2).mean()
        l0 = l0 if l0 is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < l0
