"""
Phase-8 optional experiments.  NOT used by default; every variant is opt-in
and changes either numerical precision or model capacity.

* MixedPrecisionKANLinear  — basis location / recurrence in FP32, spline
  and base contractions (the two GEMMs) in BF16 or FP16 with FP32
  accumulation (torch.autocast semantics).  Same parameters as KANLinear.
* GroupedSplineKANLinear   — spline coefficients shared by groups of
  output channels: spline_weight [n_groups, I, R], each edge keeps its own
  base_weight and spline_scaler:
      phi[o,i](u) = bw[o,i] b(u) + s[o,i] * sum_r C[g(o), i, r] B_r(u)
  Parameter count O*I*R -> n_groups*I*R (+ O*I*2).
* LowRankKANLinear         — spline_weight[o,i,r] = sum_q U[o,q] V[q,i,r]
  (rank q along the output axis, no per-edge scaler); spline_out =
  (H @ V^T) @ U^T, so the forward never forms the dense coefficients
  (the activation regulariser still does, to evaluate every edge).

These implement only forward / autograd via the reference ops and the
activation regulariser via the dense edge statistic.
"""

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_linear import KANLinear
from .reference_backend import compute_basis, edge_abs_mean_dense, regularization_from_edge_means


class MixedPrecisionKANLinear(KANLinear):
    """KANLinear whose GEMMs run in ``matmul_dtype`` (bf16/fp16); basis stays FP32."""

    def __init__(self, *args, matmul_dtype: torch.dtype = torch.bfloat16, **kw):
        kw.setdefault("backend", "reference")
        super().__init__(*args, **kw)
        if matmul_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("matmul_dtype must be torch.bfloat16 or torch.float16")
        self.matmul_dtype = matmul_dtype

    @classmethod
    def from_layer(cls, layer: KANLinear, matmul_dtype=torch.bfloat16) -> "MixedPrecisionKANLinear":
        ref = layer._ref
        new = cls(layer.in_features, layer.out_features, grid_size=layer.grid_size,
                  spline_order=layer.spline_order,
                  scale_noise=ref.scale_noise, scale_base=ref.scale_base,
                  scale_spline=ref.scale_spline, grid_eps=ref.grid_eps,
                  grid_range=ref.grid_range,
                  enable_standalone_scale_spline=layer.enable_standalone_scale_spline,
                  base_activation=copy.deepcopy(ref.base_activation),
                  backend="reference",
                  matmul_dtype=matmul_dtype).to(layer.grid.device, layer.grid.dtype)
        new.load_state_dict(layer.state_dict())
        # Preserve independently frozen parameters and submodule train/eval
        # modes, including a custom activation with its own parameters/buffers.
        params = dict(new.named_parameters())
        for name, p in layer.named_parameters():
            params[name].requires_grad_(p.requires_grad)
        modules = dict(new.named_modules())
        for name, module in layer.named_modules():
            modules[name].training = module.training
        return new

    def _forward_parts(self, x):
        """Shared mixed-precision output; geometry and edge statistics stay FP32."""
        self.last_backend = f"reference+{str(self.matmul_dtype).split('.')[-1]}-gemm"
        shp = x.shape
        xf = x.reshape(-1, self.in_features)
        if xf.dtype != torch.float32:
            raise TypeError("MixedPrecisionKANLinear expects FP32 inputs (basis is computed in FP32)")
        # Explicit casts choose the GEMM dtype even inside an outer autocast
        # context, which must not downcast the basis or the regularizer.
        with torch.autocast(device_type=xf.device.type, enabled=False):
            H = compute_basis(xf, self.grid, self.spline_order)
            s = self._ref.base_activation(xf)
            W = self._ref.effective_spline_weight()
            B = xf.shape[0]
            md = self.matmul_dtype
            y = F.linear(H.reshape(B, -1).to(md), W.reshape(self.out_features, -1).to(md)).float()
            y = y + F.linear(s.to(md), self.base_weight.to(md)).float()
        return y.reshape(shp[:-1] + (self.out_features,)), xf, s, H, W

    def forward(self, x):
        return self._forward_parts(x)[0]

    def regularization_loss(
        self, reg_act=1.0, reg_entropy=1.0, *, x=None, mode="activation",
        reg_impl="auto", chunk_size=1024,
    ):
        device_type = self.base_weight.device.type if x is None else x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return super().regularization_loss(
                reg_act, reg_entropy, x=x, mode=mode,
                reg_impl=reg_impl, chunk_size=chunk_size)

    def forward_with_regularization(
        self, x, reg_act=1.0, reg_entropy=1.0, *, mode="activation",
        reg_impl="auto", chunk_size=1024,
    ):
        """Same mixed-precision output as forward(), with an FP32 penalty.

        Both training and inference use the configured BF16/FP16 GEMMs.
        The full edge activation regularizer retains its FP32 formulation.
        """
        if mode not in ("activation", "coefficient"):
            raise ValueError(f"mode must be 'activation' or 'coefficient', got {mode!r}")
        impl = self._resolve_reg_impl(reg_impl, x)
        y, xf, s, H, W = self._forward_parts(x)
        with torch.autocast(device_type=xf.device.type, enabled=False):
            if mode == "coefficient":
                reg = self.regularization_loss(reg_act, reg_entropy, mode=mode)
            else:
                a = self._edge_abs_mean(xf, s, H, W, impl, chunk_size)
                reg = regularization_from_edge_means(a, reg_act, reg_entropy)
        return y, reg


class _FactoredSplineLayer(nn.Module):
    """Shared plumbing for grouped / low-rank layers (reference ops only)."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 base_activation=nn.SiLU, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        G, k = grid_size, spline_order
        r = torch.arange(G + 2 * k + 1, dtype=torch.float32)
        t = grid_range[0] + (r - k) * (grid_range[1] - grid_range[0]) / G
        self.register_buffer("grid", t[None].expand(in_features, -1).contiguous())
        self.base_activation = base_activation() if isinstance(base_activation, type) else base_activation
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))

    def effective_spline_weight(self) -> torch.Tensor:   # [O, I, R]
        raise NotImplementedError

    def _spline_out(self, H):                              # H [B, I, R] -> [B, O]
        W = self.effective_spline_weight()
        return F.linear(H.reshape(H.shape[0], -1), W.reshape(self.out_features, -1))

    def forward(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.in_features)
        H = compute_basis(xf, self.grid, self.spline_order)
        y = F.linear(self.base_activation(xf), self.base_weight) + self._spline_out(H)
        return y.reshape(shp[:-1] + (self.out_features,))

    def forward_with_regularization(self, x, reg_act=1.0, reg_entropy=1.0, **_):
        xf = x.reshape(-1, self.in_features)
        s = self.base_activation(xf)
        H = compute_basis(xf, self.grid, self.spline_order)
        y = F.linear(s, self.base_weight) + self._spline_out(H)
        a = edge_abs_mean_dense(s, H, self.base_weight, self.effective_spline_weight())
        return (y.reshape(x.shape[:-1] + (self.out_features,)),
                regularization_from_edge_means(a, reg_act, reg_entropy))


class GroupedSplineKANLinear(_FactoredSplineLayer):
    def __init__(self, in_features, out_features, n_groups: int, **kw):
        super().__init__(in_features, out_features, **kw)
        if not 1 <= n_groups <= out_features:
            raise ValueError("n_groups must be in [1, out_features]")
        self.n_groups = n_groups
        R = self.grid_size + self.spline_order
        self.spline_weight = nn.Parameter((torch.rand(n_groups, in_features, R) - 0.5) * 0.2 / math.sqrt(R))
        self.register_buffer("group_of", torch.arange(out_features) % n_groups)

    def effective_spline_weight(self):
        return self.spline_weight[self.group_of] * self.spline_scaler[..., None]


class LowRankKANLinear(_FactoredSplineLayer):
    """spline_weight[o,i,r] = sum_q U[o,q] V[q,i,r]; no per-edge spline scaler
    (it would break the factorisation; U already scales each output)."""

    def __init__(self, in_features, out_features, rank: int, **kw):
        super().__init__(in_features, out_features, **kw)
        del self.spline_scaler
        R = self.grid_size + self.spline_order
        self.rank = rank
        self.U = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(rank))
        self.V = nn.Parameter((torch.rand(rank, in_features, R) - 0.5) * 0.2 / math.sqrt(R))

    def effective_spline_weight(self):
        return torch.einsum("oq,qir->oir", self.U, self.V)

    def _spline_out(self, H):
        # (H @ V^T) @ U^T : never forms the dense [O, I, R] coefficients.
        return F.linear(F.linear(H.reshape(H.shape[0], -1), self.V.reshape(self.rank, -1)), self.U)


def build_kan(kind: str, widths, **kw) -> nn.Module:
    """Sequential KAN of experimental layers.  kind in {'grouped','lowrank'}."""
    layers = []
    for a, b in zip(widths[:-1], widths[1:]):
        if kind == "grouped":
            layers.append(GroupedSplineKANLinear(a, b, n_groups=max(1, b // kw.get("group_size", 4)),
                                                 grid_size=kw.get("grid_size", 5)))
        elif kind == "lowrank":
            layers.append(LowRankKANLinear(a, b, rank=min(kw.get("rank", 4), b),
                                           grid_size=kw.get("grid_size", 5)))
        else:
            raise ValueError(kind)
    return _Seq(layers)


class _Seq(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def forward_with_regularization(self, x, reg_act=1.0, reg_entropy=1.0, **kw):
        tot = x.new_zeros(())
        for l in self.layers:
            x, r = l.forward_with_regularization(x, reg_act, reg_entropy)
            tot = tot + r
        return x, tot
