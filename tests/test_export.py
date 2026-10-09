"""Phase-4 tests: function-preserving inference export (scaler folding)."""
import io

import pytest
import torch
import torch.nn as nn

from spline_kan import KAN, KANLinear
from spline_kan import kan_linear as kl

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and kl._CUDA_EXT_LOADED else [])


def _trained_like(model):
    """Move scaler/base away from init so folding is non-trivial."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * 0.3)
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_export_preserves_function_grid_and_config(device, dtype):
    torch.manual_seed(1)
    m = _trained_like(KAN([5, 7, 3], grid_size=6, base_activation=nn.Tanh).to(device, dtype))
    m.update_grid(torch.randn(200, 5, device=device, dtype=dtype) ** 3)   # non-uniform knots
    sd_before = {k: v.clone() for k, v in m.state_dict().items()}
    e = m.export_inference()

    x = torch.randn(64, 5, device=device, dtype=dtype) * 1.5
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert torch.allclose(m(x), e(x), atol=tol, rtol=tol)

    # original untouched and still trainable
    for k, v in m.state_dict().items():
        assert torch.equal(v, sd_before[k]), k
    assert all(p.requires_grad for p in m.parameters())

    for lo, le in zip(m.layers, e.layers):
        assert not hasattr(le, "spline_scaler")
        assert le.enable_standalone_scale_spline is False
        assert torch.equal(lo.grid, le.grid)
        assert le.grid.device == lo.grid.device and le.grid.dtype == lo.grid.dtype
        assert isinstance(le._ref.base_activation, nn.Tanh)
        assert le._ref.base_activation is not lo._ref.base_activation
        assert le.backend == lo.backend
        assert le.spline_weight.data_ptr() != lo.spline_weight.data_ptr()
    n_m = sum(p.numel() for p in m.parameters())
    n_e = sum(p.numel() for p in e.parameters())
    assert n_m - n_e == 5 * 7 + 7 * 3
    assert not e.training and not any(p.requires_grad for p in e.parameters())


def test_export_without_scaler_is_identity_copy():
    m = KANLinear(3, 2, enable_standalone_scale_spline=False)
    e = m.export_inference()
    assert torch.equal(e.spline_weight, m.spline_weight)


@pytest.mark.parametrize("device", DEVICES)
def test_export_save_load_roundtrip(device):
    torch.manual_seed(2)
    m = _trained_like(KAN([4, 6, 2], grid_size=5).to(device))
    m.update_grid(torch.rand(100, 4, device=device) * 3 - 1)
    e = m.export_inference()
    buf = io.BytesIO()
    torch.save(e.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location=device)
    assert not any("spline_scaler" in k for k in sd)
    fresh = KAN([4, 6, 2], grid_size=5, enable_standalone_scale_spline=False).to(device)
    fresh.load_state_dict(sd)
    x = torch.randn(33, 4, device=device)
    assert torch.allclose(fresh(x), m(x), atol=1e-5)
    # loading the exported dict into a scaler-ful model must fail loudly
    with pytest.raises(RuntimeError):
        KAN([4, 6, 2], grid_size=5).to(device).load_state_dict(sd)


def test_export_requires_grad_option_finetunes():
    m = KANLinear(3, 2)
    e = m.export_inference(requires_grad=True)
    e(torch.randn(4, 3)).sum().backward()
    assert e.spline_weight.grad is not None and m.spline_weight.grad is None
