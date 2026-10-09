"""
Memory-lean building blocks for forward + activation regularisation.

The activation regulariser needs the per-edge L1 norm (paper eq. 19)

    a[o, i] = mean_n | s[n,i] * bw[o,i] + sum_r H[n,i,r] * W[o,i,r] |

with s = base_activation(x), H = B-spline basis of x, W = effective spline
weight.  The dense formulation materialises phi [N, O, I] (plus autograd
copies of it).  ``EdgeAbsMean`` computes ``a`` with a native CUDA kernel
that never stores phi; its backward recomputes phi (R+1 FMAs per element)
to obtain sign(phi).  Only s [N, I], H [N, I, R] and the weights are saved.

``NativeBasis`` evaluates H with the native kernel; its backward contracts
the incoming gradient with dB/dx in registers instead of materialising
dH/dx [N, I, R].

Both functions are exact under create_graph=True: when the backward is
itself recorded, they switch to differentiable PyTorch formulations, so
higher-order derivatives remain correct (at the dense cost).
"""

from typing import Optional

import torch

from .reference_backend import compute_basis, compute_basis_derivative, edge_abs_mean_dense

try:
    from spline_kan import _C
    HAS_FUSED_KERNELS = bool(getattr(_C, "HAS_FUSED_KERNELS", False))
except Exception:  # pragma: no cover - extension missing
    _C = None
    HAS_FUSED_KERNELS = False

_EMPTY = None


def _empty_like(t: torch.Tensor) -> torch.Tensor:
    return t.new_empty(0)


class NativeBasis(torch.autograd.Function):
    """H = basis(x) [B, I, R] using the CUDA kernel."""

    @staticmethod
    def forward(ctx, x, grid, k: int):
        ctx.k = k
        ctx.save_for_backward(x, grid)
        return _C.basis(x, grid, k)

    @staticmethod
    def backward(ctx, gH):
        x, grid = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        if torch.is_grad_enabled():
            # Double-backward requested: differentiable formulation.
            dB = compute_basis_derivative(x, grid, ctx.k)
            return (gH * dB).sum(-1), None, None
        return _C.basis_grad(x.contiguous(), grid, ctx.k, gH.contiguous()), None, None


class EdgeAbsMean(torch.autograd.Function):
    """a[o,i] = mean_n |s*bw + <H, W>|, without materialising [N, O, I]."""

    @staticmethod
    def forward(ctx, s, H, bw, W):
        has_base = bw is not None
        e = _empty_like(H)
        ctx.has_base = has_base
        ctx.save_for_backward(s if has_base else e, H, bw if has_base else e, W)
        return _C.edge_abs_mean(s if has_base else e, H, bw if has_base else e, W)

    @staticmethod
    def backward(ctx, ga):
        s, H, bw, W = ctx.saved_tensors
        has_base = ctx.has_base
        need_s, need_H, need_bw, need_W = ctx.needs_input_grad
        if torch.is_grad_enabled():
            # Use partial derivatives expressed in the ORIGINAL saved tensors.
            # Detaching them loses mixed derivatives (e.g. d(dbw)/ds).  Calling
            # autograd.grad on the rebuilt expression can also include unwanted
            # upstream paths when two inputs share an ancestor.  These formulas
            # keep those paths for double backward, without counting them twice
            # in this backward.  sign(0)=0 matches PyTorch's abs subgradient.
            phi = torch.einsum("oir,nir->noi", W, H)
            if has_base:
                phi = s.unsqueeze(1) * bw.unsqueeze(0) + phi
            g = phi.sign() * ga.unsqueeze(0) / H.shape[0]
            ds = (g * bw.unsqueeze(0)).sum(1) if need_s and has_base else None
            dH = torch.einsum("noi,oir->nir", g, W) if need_H else None
            dbw = (g * s.unsqueeze(1)).sum(0) if need_bw and has_base else None
            dW = torch.einsum("noi,nir->oir", g, H) if need_W else None
            return ds, dH, dbw, dW
        ds, dH, dbw, dW = _C.edge_abs_mean_backward(
            ga, s, H, bw, W,
            bool(need_s and has_base), bool(need_H), bool(need_bw and has_base), bool(need_W))
        return (ds if need_s and has_base else None,
                dH if need_H else None,
                dbw if need_bw and has_base else None,
                dW if need_W else None)


def native_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    return NativeBasis.apply(x, grid, k)


def edge_abs_mean(s: Optional[torch.Tensor], H: torch.Tensor,
                  bw: Optional[torch.Tensor], W: torch.Tensor) -> torch.Tensor:
    return EdgeAbsMean.apply(s, H, bw, W)


def edge_abs_mean_chunked(s: Optional[torch.Tensor], x: torch.Tensor, grid: torch.Tensor, k: int,
                          bw: Optional[torch.Tensor], W: torch.Tensor,
                          chunk_size: int) -> torch.Tensor:
    """
    Pure-PyTorch low-memory variant: activation checkpointing over sample
    chunks.  Each chunk's [n_c, O, I] phi is freed after the forward and
    recomputed (including its basis) during backward, so peak memory scales
    with chunk_size instead of N.  Supports any device and double backward.
    """
    from torch.utils.checkpoint import checkpoint

    N = x.shape[0]

    def chunk_sum(xc, sc, bw_, W_):
        Hc = compute_basis(xc, grid, k)
        return edge_abs_mean_dense(sc, Hc, bw_, W_) * xc.shape[0]

    total = None
    for n0 in range(0, N, chunk_size):
        xc = x[n0:n0 + chunk_size]
        sc = s[n0:n0 + chunk_size] if s is not None else None
        part = checkpoint(chunk_sum, xc, sc, bw, W, use_reentrant=False)
        total = part if total is None else total + part
    return total / N
