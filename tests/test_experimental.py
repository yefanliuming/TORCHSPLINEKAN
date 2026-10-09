"""Phase-8 tests: experimental variants are opt-in and mathematically consistent."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spline_kan import KAN, KANLinear
from spline_kan.experimental import (GroupedSplineKANLinear, LowRankKANLinear,
                                     MixedPrecisionKANLinear)
from spline_kan.reference_backend import compute_basis

CUDA = torch.cuda.is_available()


def test_defaults_untouched():
    m = KAN([4, 5, 2])
    assert all(type(l) is KANLinear for l in m.layers)


@pytest.mark.skipif(not CUDA, reason="bf16 GEMM test uses CUDA")
def test_mixed_precision_close_to_fp32_and_flagged():
    torch.manual_seed(0)
    L = KANLinear(32, 16, backend="reference").cuda()
    M = MixedPrecisionKANLinear.from_layer(L, torch.bfloat16)
    x = torch.rand(256, 32, device="cuda") * 2 - 1
    y, ym = L(x), M(x)
    assert ym.dtype == torch.float32
    rel = (y - ym).abs().max() / y.abs().max()
    assert 0 < rel < 3e-2            # different (lower precision) but close
    assert "bfloat16" in M.last_backend
    with pytest.raises(TypeError):
        M(x.double())


@pytest.mark.skipif(not CUDA, reason="mixed-precision GEMMs use CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("impl", ["dense", "checkpoint"])
def test_mixed_precision_training_uses_same_gemms_as_inference(dtype, impl, monkeypatch):
    torch.manual_seed(37)
    layer = MixedPrecisionKANLinear(3, 2, matmul_dtype=dtype).cuda()
    x = torch.rand(17, 3, device="cuda") * 1.8 - 0.9
    dtypes = []
    linear = F.linear

    def record_linear(input, weight, *args, **kw):
        dtypes.append((input.dtype, weight.dtype))
        return linear(input, weight, *args, **kw)

    monkeypatch.setattr(F, "linear", record_linear)
    # An outer autocast must not override the layer's requested GEMM dtype.
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    with torch.autocast("cuda", dtype=other_dtype):
        inference = layer(x)
        training, reg = layer.forward_with_regularization(
            x, 0.7, 0.3, reg_impl=impl, chunk_size=5)
        separate_reg = layer.regularization_loss(0.7, 0.3, x=x, reg_impl=impl, chunk_size=5)
    assert dtypes == [(dtype, dtype)] * 4
    assert torch.equal(training, inference)
    assert training.dtype == reg.dtype == torch.float32
    expected_reg = layer._ref.regularization_loss(0.7, 0.3, x=x)
    assert torch.allclose(reg, expected_reg, rtol=1e-6, atol=1e-7)
    assert torch.equal(reg, separate_reg)
    (training.square().mean() + reg).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in layer.parameters())


@pytest.mark.skipif(not CUDA, reason="mixed-precision GEMMs use CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mixed_precision_multilayer_joint_output_and_gradients(dtype):
    torch.manual_seed(38)
    model = KAN.from_layers([
        MixedPrecisionKANLinear(3, 4, matmul_dtype=dtype).cuda(),
        MixedPrecisionKANLinear(4, 2, matmul_dtype=dtype).cuda(),
    ])
    x = torch.rand(15, 3, device="cuda").requires_grad_()
    inputs = (x,) + tuple(model.parameters())
    joint_y, joint_reg = model.forward_with_regularization(x, 0.7, 0.3)
    joint_grads = torch.autograd.grad(joint_y.square().sum() + joint_reg, inputs)
    h, reg = x, x.new_zeros(())
    for layer in model.layers:
        reg = reg + layer.regularization_loss(0.7, 0.3, x=h)
        h = layer(h)
    separate_grads = torch.autograd.grad(h.square().sum() + reg, inputs)
    assert torch.equal(joint_y, h)
    assert torch.allclose(joint_reg, reg)
    for joint, separate in zip(joint_grads, separate_grads):
        assert torch.allclose(joint, separate, rtol=1e-6, atol=1e-6)


def test_mixed_precision_conversion_preserves_activation_and_configuration():
    layer = KANLinear(3, 2, backend="reference", base_activation=nn.LeakyReLU(0.5),
                      grid_eps=0.17, grid_range=(-2.0, 3.0), scale_noise=0.2,
                      scale_base=0.7, scale_spline=1.3)
    layer.eval()
    layer.base_weight.requires_grad_(False)
    converted = MixedPrecisionKANLinear.from_layer(layer)
    assert converted._ref.base_activation.negative_slope == 0.5
    assert converted._ref.base_activation is not layer._ref.base_activation
    for name in ("grid_eps", "grid_range", "scale_noise", "scale_base", "scale_spline"):
        assert getattr(converted._ref, name) == getattr(layer._ref, name)
    for name, value in layer.state_dict().items():
        assert torch.equal(converted.state_dict()[name], value)
        assert converted.state_dict()[name].data_ptr() != value.data_ptr()
    assert not converted.training and not converted._ref.training
    assert not converted.base_weight.requires_grad
    assert converted.spline_weight.requires_grad


def test_mixed_precision_conversion_preserves_parameterized_activation_and_modes():
    activation = nn.PReLU(num_parameters=3, init=0.6)
    layer = KANLinear(3, 2, backend="reference", base_activation=activation)
    activation.eval()  # Child mode can differ from the containing layer's mode.
    activation.weight.requires_grad_(False)
    converted = MixedPrecisionKANLinear.from_layer(layer)
    assert converted.training
    assert not converted._ref.base_activation.training
    assert converted._ref.base_activation.num_parameters == 3
    assert torch.equal(converted._ref.base_activation.weight, activation.weight)
    assert not converted._ref.base_activation.weight.requires_grad


@pytest.mark.skipif(not CUDA, reason="mixed-precision GEMMs use CUDA")
def test_mixed_precision_joint_coefficient_mode():
    layer = MixedPrecisionKANLinear(3, 2).cuda()
    x = torch.randn(5, 3, device="cuda")
    y, reg = layer.forward_with_regularization(x, mode="coefficient")
    assert torch.equal(y, layer(x))
    assert torch.equal(reg, layer.regularization_loss(mode="coefficient"))


def test_grouped_shares_coefficients():
    torch.manual_seed(1)
    L = GroupedSplineKANLinear(3, 8, n_groups=2, grid_size=4)
    W = L.effective_spline_weight() / L.spline_scaler[..., None]
    assert torch.allclose(W[0], W[2]) and torch.allclose(W[1], W[7])
    n_dense = 8 * 3 * 7
    assert L.spline_weight.numel() == 2 * 3 * 7 < n_dense
    x = torch.rand(10, 3) * 2 - 1
    H = compute_basis(x, L.grid, 3)
    ref = torch.nn.functional.silu(x) @ L.base_weight.t() + torch.einsum(
        "bir,oir->bo", H, L.effective_spline_weight())
    assert torch.allclose(L(x), ref, atol=1e-6)


def test_lowrank_factored_forward_equals_dense():
    torch.manual_seed(2)
    L = LowRankKANLinear(5, 6, rank=2, grid_size=3).double()
    x = torch.rand(20, 5, dtype=torch.float64) * 2 - 1
    H = compute_basis(x, L.grid, 3)
    dense = torch.nn.functional.silu(x) @ L.base_weight.t() + torch.einsum(
        "bir,oir->bo", H, L.effective_spline_weight())
    assert torch.allclose(L(x), dense, atol=1e-12)
    y, r = L.forward_with_regularization(x, 1.0, 1.0)
    (y.sum() + r).backward()
    assert L.U.grad is not None and L.V.grad is not None
    assert not hasattr(L, "spline_scaler")
