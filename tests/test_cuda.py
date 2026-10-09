import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spline_kan import KANLinear, KAN


def test_cuda_forward():
    layer = KANLinear(16, 32, backend="auto").cuda()
    x = torch.randn(64, 16, device="cuda")
    y = layer(x)
    assert y.device.type == "cuda"
    assert y.shape == (64, 32)


def test_cuda_reference_parity():
    """CUDA and reference backends must agree to within tolerance."""
    torch.manual_seed(42)
    layer_ref = KANLinear(8, 4, grid_size=5, spline_order=3, backend="reference")
    layer_cuda = KANLinear(8, 4, grid_size=5, spline_order=3, backend="cuda").cuda()

    layer_cuda.load_state_dict({k: v.cuda() for k, v in layer_ref.state_dict().items()})

    x = torch.randn(32, 8)
    y_ref = layer_ref(x)
    y_cuda = layer_cuda(x.cuda())

    assert torch.allclose(y_ref.cuda(), y_cuda, atol=1e-4, rtol=1e-4), \
        f"Max diff: {(y_ref.cuda() - y_cuda).abs().max()}"


def test_cuda_backward():
    layer = KANLinear(8, 4, backend="auto").cuda()
    x = torch.randn(32, 8, device="cuda", requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.base_weight.grad is not None
    assert layer.spline_weight.grad is not None


def test_cuda_gradcheck_fp64():
    layer = KANLinear(4, 3, grid_size=3, spline_order=3, backend="cuda").double().cuda()
    x = torch.randn(5, 4, dtype=torch.float64, device="cuda", requires_grad=True) * 0.5

    def fn(x_):
        return layer(x_)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)


def test_cuda_explicit_fail_no_cuda():
    """backend='cuda' on CPU tensor should raise clear error."""
    with pytest.raises((RuntimeError, ValueError)):
        layer = KANLinear(4, 4, backend="cuda")
        x = torch.randn(10, 4)  # CPU
        layer(x)


def test_cuda_fp16_rejected():
    """FP16 input should be rejected with clear error."""
    layer = KANLinear(8, 4).cuda()
    x = torch.randn(10, 8, device="cuda", dtype=torch.float16)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        layer(x)


def test_cuda_pde_residual():
    model = KAN([1, 8, 1], backend="auto").double().cuda()
    x = torch.linspace(-0.9, 0.9, 20, dtype=torch.float64, device="cuda", requires_grad=True).unsqueeze(1)
    u = model(x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    target = -(torch.pi**2) * torch.sin(torch.pi * x)
    loss = ((u_xx - target)**2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_device_movement():
    layer = KANLinear(4, 4, backend="reference")
    x = torch.randn(10, 4)
    y_cpu = layer(x)
    layer_cuda = layer.cuda()
    x_cuda = x.cuda()
    y_cuda = layer_cuda(x_cuda)
    assert torch.allclose(y_cpu.cuda(), y_cuda, atol=1e-5)
