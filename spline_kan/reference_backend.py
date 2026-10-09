"""
Reference (pure PyTorch) backend for B-spline KAN layers.

Mathematical background
-----------------------
Given G grid intervals and spline order k, for each input feature i:

  Extended knot vector (G + 2k + 1 knots):
      t_r = a + (r - k) * (b - a) / G   for r = 0 .. G+2k

  R = G + k basis functions per feature.

  Cox-de Boor recurrence:
      B_{r,0}(x) = 1 if t_r <= x < t_{r+1}  (last interval closed on right)
      B_{r,d}(x) = (x - t_r)/(t_{r+d} - t_r)        * B_{r,d-1}(x)
                 + (t_{r+d+1} - x)/(t_{r+d+1}-t_{r+1}) * B_{r+1,d-1}(x)
      Zero denominators contribute zero, not NaN.

  Edge activation:
      phi_{j,i}(u) = base_weight[j,i] * activation(u)
                   + spline_scaler[j,i] * sum_r( spline_weight[j,i,r] * B_r(u) )
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_basis(
    x: torch.Tensor,
    grid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Evaluate B-spline basis functions via the Cox-de Boor recurrence.

    Parameters
    ----------
    x : Tensor [B, I]
        Input values per (sample, input-feature).
    grid : Tensor [I, G + 2*k + 1]
        Extended knot vectors, one row per input feature.
    k : int
        Spline order.  Supported: 0, 1, 2, 3.

    Returns
    -------
    Tensor [B, I, R]   R = G + k
        Evaluated basis values.

    Raises
    ------
    ValueError
        If k is not in {0, 1, 2, 3}.
    """
    if k not in (0, 1, 2, 3):
        raise ValueError(f"spline_order must be 0, 1, 2, or 3, got {k}")

    B, I = x.shape
    x_exp = x.unsqueeze(-1)        # [B, I, 1]
    t = grid.unsqueeze(0)          # [1, I, G+2k+1]

    t_l = t[:, :, :-1]            # [1, I, G+2k]
    t_r_g = t[:, :, 1:]           # [1, I, G+2k]

    # Degree-0: half-open [t_r, t_{r+1}); last interval closed on right
    basis = ((x_exp >= t_l) & (x_exp < t_r_g)).to(x.dtype)   # [B, I, G+2k]
    last_mask = ((x_exp >= t_l[:, :, -1:]) & (x_exp <= t_r_g[:, :, -1:])).to(x.dtype)
    basis = torch.cat([basis[:, :, :-1], last_mask], dim=-1)

    # Cox-de Boor recurrence for degrees 1..k
    for d in range(1, k + 1):
        n_new = basis.shape[-1] - 1   # shrinks by one per level

        t0   = t[:, :, :n_new]           # t_{r}
        t_d  = t[:, :, d:d+n_new]        # t_{r+d}
        t_1  = t[:, :, 1:1+n_new]        # t_{r+1}
        t_d1 = t[:, :, d+1:d+1+n_new]   # t_{r+d+1}

        denom_l = t_d - t0
        denom_r = t_d1 - t_1

        mask_l = denom_l.abs() > 0
        mask_r = denom_r.abs() > 0
        safe_l = torch.where(mask_l, denom_l, torch.ones_like(denom_l))
        safe_r = torch.where(mask_r, denom_r, torch.ones_like(denom_r))

        alpha = torch.where(mask_l, (x_exp - t0) / safe_l,  torch.zeros_like(denom_l))
        beta  = torch.where(mask_r, (t_d1 - x_exp) / safe_r, torch.zeros_like(denom_r))

        basis = alpha * basis[:, :, :n_new] + beta * basis[:, :, 1:n_new + 1]

    return basis  # [B, I, R] where R = G+k


def compute_basis_derivative(
    x: torch.Tensor,
    grid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    d B_{r,k}(x) / dx, built from differentiable ops (so it can itself be
    differentiated).  Same knot conventions as :func:`compute_basis`.

        dB_{r,k}/dx = k * ( B_{r,k-1} / (t_{r+k} - t_r)
                          - B_{r+1,k-1} / (t_{r+k+1} - t_{r+1}) )

    Returns [B, I, G + k].
    """
    if k < 1:
        raise ValueError("compute_basis_derivative requires spline_order >= 1")
    Hm = compute_basis(x, grid, k - 1)                       # [B, I, R+1]
    R = grid.shape[1] - k - 1
    t = grid.unsqueeze(0)
    d1 = t[:, :, k:k + R] - t[:, :, :R]
    d2 = t[:, :, k + 1:k + 1 + R] - t[:, :, 1:1 + R]
    m1, m2 = d1 != 0, d2 != 0
    term1 = torch.where(m1, Hm[:, :, :R] / torch.where(m1, d1, torch.ones_like(d1)),
                        torch.zeros_like(Hm[:, :, :R]))
    term2 = torch.where(m2, Hm[:, :, 1:R + 1] / torch.where(m2, d2, torch.ones_like(d2)),
                        torch.zeros_like(Hm[:, :, 1:R + 1]))
    return k * (term1 - term2)


def edge_abs_mean_dense(
    s: Optional[torch.Tensor],
    H: torch.Tensor,
    base_weight: Optional[torch.Tensor],
    eff_weight: torch.Tensor,
) -> torch.Tensor:
    """
    a[o, i] = mean_n | s[n,i] * base_weight[o,i] + sum_r H[n,i,r] * eff_weight[o,i,r] |

    Materialises phi [N, O, I].  Fully differentiable to any order.
    s / base_weight may be None (no base branch).
    """
    spline = torch.einsum("oir,nir->noi", eff_weight, H)   # strided (bmm over i) layout
    if base_weight is None:
        return spline.abs().mean(dim=0)
    # Contiguous operand first: TensorIterator then gives phi (and every
    # elementwise kernel in forward/backward) the contiguous [N, O, I] layout.
    phi = s.unsqueeze(1) * base_weight.unsqueeze(0) + spline
    return phi.abs().mean(dim=0)


def regularization_from_edge_means(
    a: torch.Tensor,
    regularize_activation: float,
    regularize_entropy: float,
) -> torch.Tensor:
    """
    Paper eqs. (20)-(21) from per-edge L1 norms a = |phi_{o,i}|_1:

        A = sum a,   p = a / A,   S = -sum p log p
        loss = mu1 * A + mu2 * S

    Returns zero (with finite gradients) for an entirely inactive layer.
    """
    A = a.sum()
    eps = torch.finfo(a.dtype).tiny
    A_safe = A.clamp(min=eps)
    p = a / A_safe                    # [O, I]; all-zero when A == 0
    p_safe = p.clamp(min=eps)
    H_ent = -(p_safe * p_safe.log()).sum()
    # When A == 0 the entropy gradient is zero because p is identically zero,
    # so no NaN arises from the log branch even though we clamped A.
    return regularize_activation * A + regularize_entropy * H_ent


def fit_spline_coeffs(
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    grid: torch.Tensor,
    k: int,
    ridge: float = 1e-12,
) -> torch.Tensor:
    """
    Batched, regularised least-squares fit of spline coefficients.

    For each input feature i, solves  min_c ||A_i c - y_i||^2 + lam_i*||c||^2
    with A_i = basis(x[:, i]) [B, R] via the normal equations in float64
    (Cholesky, on the samples' device), lam_i = ridge * trace(A_i^T A_i)/R.

    ``ridge`` is relative to the mean diagonal, so the conditioning of the
    solve is bounded by ~1/ridge.  The default 1e-12 leaves well-posed
    fits unbiased to ~1e-12 relative while keeping basis functions that no
    sample touches (under-sampling, rank deficiency, constant inputs) at ~0
    instead of exploding.  If Cholesky still fails (numerically singular),
    lam is increased by 1e3 steps up to ridge*1e9 before a general solve.

    Parameters / returns as :func:`curve2coeff` ([O, B, I] -> [O, I, R]).
    """
    if ridge <= 0:
        raise ValueError(f"ridge must be positive, got {ridge}")
    A = compute_basis(x_samples.double(), grid.double(), k)    # [B, I, R]
    y = y_samples.double()                                       # [O, B, I]
    R = A.shape[-1]
    AtA = torch.einsum("bir,bis->irs", A, A)                     # [I, R, R]
    Aty = torch.einsum("bir,obi->ior", A, y).transpose(1, 2)     # [I, R, O]
    scale = AtA.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-300) / R
    eye = torch.eye(R, dtype=torch.float64, device=A.device)
    lam = ridge
    for _ in range(4):
        M = AtA + (lam * scale)[:, None, None] * eye
        L, info = torch.linalg.cholesky_ex(M)
        if not bool((info != 0).any()):
            sol = torch.cholesky_solve(Aty, L)                   # [I, R, O]
            break
        lam *= 1e3
    else:
        sol = torch.linalg.lstsq(M, Aty).solution
    return sol.permute(2, 0, 1).to(dtype=x_samples.dtype)        # [O, I, R]


def curve2coeff(
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    grid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Fit B-spline coefficients to observed (x, y) pairs by least squares.

    Parameters
    ----------
    x_samples : Tensor [B, I]
    y_samples : Tensor [O, B, I]
        Target activation values per output, sample, and input feature.
    grid : Tensor [I, G+2k+1]
    k : int  spline order

    Returns
    -------
    Tensor [O, I, R]

    Notes
    -----
    Promoted to float64 on CPU for numerical stability. Handles rank-deficient
    systems via the minimum-norm least-squares solution from lstsq.
    """
    B, I = x_samples.shape
    O = y_samples.shape[0]
    basis = compute_basis(x_samples, grid, k)   # [B, I, R]
    R = basis.shape[-1]

    device = x_samples.device
    dtype_orig = x_samples.dtype

    # Promote to float64 on CPU for numerical stability
    basis_64 = basis.to(dtype=torch.float64, device="cpu")   # [B, I, R]
    y_64 = y_samples.to(dtype=torch.float64, device="cpu")   # [O, B, I]

    coeffs = torch.zeros(O, I, R, dtype=torch.float64, device="cpu")

    for i in range(I):
        A = basis_64[:, i, :]        # [B, R]
        b_mat = y_64[:, :, i].t()    # [B, O]
        result = torch.linalg.lstsq(A, b_mat)
        sol = result.solution        # [R, O]
        coeffs[:, i, :] = sol.t()    # [O, R]

    return coeffs.to(dtype=dtype_orig, device=device)


class ReferenceKANLayer(nn.Module):
    """
    Pure-PyTorch B-spline KAN layer.

    Parameters match the module-level spec.  The base activation defaults to
    SiLU; any nn.Module instance (or class) may be supplied, but only the
    reference backend is used in that case.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.grid_range = grid_range

        if isinstance(base_activation, type):
            self.base_activation: nn.Module = base_activation()
        else:
            self.base_activation = base_activation

        grid = self._build_uniform_grid(grid_range[0], grid_range[1])
        self.register_buffer("grid", grid)

        R = grid_size + spline_order

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, R))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(out_features, in_features)
            )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _build_uniform_grid(self, a: float, b: float) -> torch.Tensor:
        G = self.grid_size
        k = self.spline_order
        r = torch.arange(G + 2 * k + 1, dtype=torch.float32)
        t = a + (r - k) * (b - a) / G
        return t.unsqueeze(0).expand(self.in_features, -1).contiguous()

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(self.scale_base)

        G = self.grid_size
        k = self.spline_order
        R = G + k
        with torch.no_grad():
            noise = (torch.rand_like(self.spline_weight) - 0.5) * 2.0
            self.spline_weight.copy_(noise * self.scale_noise / math.sqrt(R))

        if self.enable_standalone_scale_spline:
            with torch.no_grad():
                self.spline_scaler.fill_(self.scale_spline)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        base_out = F.linear(self.base_activation(x_flat), self.base_weight)

        H = compute_basis(x_flat, self.grid, self.spline_order)  # [B, I, R]

        if self.enable_standalone_scale_spline:
            eff_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        else:
            eff_weight = self.spline_weight

        R = self.grid_size + self.spline_order
        H_flat = H.reshape(B, self.in_features * R)
        eff_flat = eff_weight.reshape(self.out_features, -1)
        spline_out = F.linear(H_flat, eff_flat)

        y = base_out + spline_out
        return y.reshape(orig_shape[:-1] + (self.out_features,))

    # ------------------------------------------------------------------
    # Grid update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def grid_from_samples(self, x_flat: torch.Tensor, G: int, margin: float = 0.01) -> torch.Tensor:
        """
        Knot vector [I, G+2k+1] adapted to the samples (see update_grid):
        G+1 interior points from a quantile/uniform mixture (grid_eps) and k
        outer knots per side spaced by the margin-expanded range / G.
        """
        B = x_flat.shape[0]
        k = self.spline_order
        dtype = x_flat.dtype
        device = x_flat.device

        # Margin-expanded per-feature data range, used for outer knot spacing.
        x_sorted, _ = x_flat.sort(dim=0)   # [B, I]
        x_min = x_sorted[0] - margin        # [I]
        x_max = x_sorted[-1] + margin       # [I]

        # Uniform portion: G+1 evenly-spaced points over [x_min, x_max]
        u_steps = torch.linspace(0.0, 1.0, G + 1, device=device, dtype=dtype)
        u_grid = x_min.unsqueeze(0) + u_steps.unsqueeze(-1) * (
            x_max - x_min
        ).unsqueeze(0)   # [G+1, I]

        # Quantile portion
        step_q = (1.0 - self.grid_eps) / G
        q_fracs = (
            torch.arange(G + 1, dtype=dtype, device=device) * step_q
            + self.grid_eps / 2.0
        ).clamp(0.0, 1.0)

        idx_f = q_fracs * (B - 1)
        idx_lo = idx_f.long().clamp(0, B - 1)
        idx_hi = (idx_lo + 1).clamp(0, B - 1)
        frac = (idx_f - idx_lo.to(dtype)).unsqueeze(-1)
        q_grid = (1.0 - frac) * x_sorted[idx_lo] + frac * x_sorted[idx_hi]

        # G+1 interior boundary points; all are preserved in the knot vector
        mixed = (1.0 - self.grid_eps) * q_grid + self.grid_eps * u_grid  # [G+1, I]

        # Outer knot spacing: margin-expanded range / G, always positive
        step_ext = (x_max - x_min) / G  # [I]

        # k outer knots on each side
        r_left = torch.arange(k, 0, -1, device=device, dtype=dtype)   # [k]
        left_knots = (
            mixed[0].unsqueeze(0) - r_left.unsqueeze(-1) * step_ext.unsqueeze(0)
        )  # [k, I]

        r_right = torch.arange(1, k + 1, device=device, dtype=dtype)  # [k]
        right_knots = (
            mixed[-1].unsqueeze(0) + r_right.unsqueeze(-1) * step_ext.unsqueeze(0)
        )  # [k, I]

        # Full knot vector: [G+2k+1, I] → transposed to [I, G+2k+1]
        new_grid = torch.cat([left_knots, mixed, right_knots], dim=0).t().contiguous()
        return new_grid


    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """
        Relocate knots to better cover the observed data distribution,
        then refit spline_weight to the OLD (unscaled) spline curve via
        least squares on the supplied samples.

        Grid construction
        -----------------
        1. Compute G+1 interior boundary points from a quantile/uniform
           mixture (controlled by grid_eps).  All G+1 points are preserved.
        2. Extend with k outer knots on each side, spaced by
           (x_max - x_min) / G, where x_min and x_max are the
           margin-expanded per-feature data extremes.

        The resulting G+2k+1 knot vector is strictly increasing for any
        margin > 0, including constant or repeated input features.

        The update is transactional: new grid and new coefficients are
        computed before any buffer or parameter is modified.
        spline_scaler and base_weight are never touched.
        """
        if margin <= 0:
            raise ValueError(f"margin must be positive, got {margin}")
        if not (0.0 < self.grid_eps < 1.0):
            raise ValueError(
                f"grid_eps must be in (0, 1), got {self.grid_eps}"
            )

        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        if B < 2:
            raise ValueError(
                f"update_grid requires at least 2 samples, got {B}"
            )

        k = self.spline_order

        # Capture unscaled spline curve values at the sample points.
        # y_old[o, b, i] = sum_r spline_weight[o, i, r] * H_old[b, i, r]
        H_old = compute_basis(x_flat, self.grid, k)        # [B, I, R]
        y_old = torch.einsum("oir,bir->obi", self.spline_weight, H_old)

        new_grid = self.grid_from_samples(x_flat, self.grid_size, margin)

        # Refit unscaled coefficients on the new grid (least-squares approximation)
        new_coeffs = curve2coeff(x_flat, y_old, new_grid, k)  # [O, I, R]

        # Transactional write
        self.grid.copy_(new_grid)
        self.spline_weight.copy_(new_coeffs)

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
        *,
        x: Optional[torch.Tensor] = None,
        mode: str = "activation",
    ) -> torch.Tensor:
        """
        Returns a scalar regularisation loss.

        Parameters
        ----------
        regularize_activation : float
            Weight for the L1 activation penalty term.
        regularize_entropy : float
            Weight for the entropy term.
        x : Tensor, keyword-only
            Input samples to this layer.  Required when mode="activation".
            Supports arbitrary leading batch dimensions.
        mode : {"activation", "coefficient"}, keyword-only
            "activation"  (default) — KAN paper formulation.  Evaluates the
            full edge activation phi[j,i](u) = base_weight[j,i]*sigma(u) +
            spline_scaler[j,i]*spline(u) on the supplied samples, takes the
            per-edge mean absolute value, and combines an L1 and an entropy
            term normalised across ALL edges in the layer.
            "coefficient" — legacy mode; uses mean |scaled coefficient| per
            edge normalised per output node.  x is ignored.

        Raises
        ------
        ValueError
            If mode="activation" and x is not supplied.
        """
        if mode not in ("activation", "coefficient"):
            raise ValueError(
                f"mode must be 'activation' or 'coefficient', got {mode!r}"
            )

        if mode == "activation":
            if x is None:
                raise ValueError(
                    "regularization_loss with mode='activation' requires x. "
                    "Pass the layer's input samples as x=<tensor>, or use "
                    "mode='coefficient' for the legacy coefficient-based loss."
                )
            return self._regularization_activation(
                x, regularize_activation, regularize_entropy
            )

        # mode == "coefficient" — legacy implementation
        return self._regularization_coefficient(
            regularize_activation, regularize_entropy
        )

    def _regularization_activation(
        self,
        x: torch.Tensor,
        regularize_activation: float,
        regularize_entropy: float,
    ) -> torch.Tensor:
        """
        Activation-based regularisation from the original KAN paper.

        phi[j,i](u) = base_weight[j,i]*sigma(u)
                    + scaler[j,i] * sum_r spline_weight[j,i,r]*B[i,r](u)

        a[j,i] = mean_s |phi[j,i](x[s,i])|
        A = sum_{j,i} a[j,i]
        p[j,i] = a[j,i] / A
        loss = reg_act * A + reg_entropy * (-sum_{j,i} p[j,i] log p[j,i])

        Avoids a [N, O, I, R] intermediate; uses [N, O, I] instead.
        Returns zero (with finite gradients) for an entirely inactive layer.
        """
        x_flat = x.reshape(-1, self.in_features)     # [N, I]
        sigma_x = self.base_activation(x_flat)        # [N, I]
        H = compute_basis(x_flat, self.grid, self.spline_order)  # [N, I, R]
        a = edge_abs_mean_dense(sigma_x, H, self.base_weight, self.effective_spline_weight())
        return regularization_from_edge_means(a, regularize_activation, regularize_entropy)

    def effective_spline_weight(self) -> torch.Tensor:
        """spline_weight * spline_scaler[..., None] (or spline_weight if no scaler)."""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def _regularization_coefficient(
        self,
        regularize_activation: float,
        regularize_entropy: float,
    ) -> torch.Tensor:
        """
        Legacy coefficient-based regularisation (mode='coefficient').

        Uses |S * C| as the effective coefficient magnitude, where S is
        spline_scaler (or ones if disabled) and C is spline_weight.
        Entropy is normalised per output node.  All-zero rows yield finite
        gradients.
        """
        if self.enable_standalone_scale_spline:
            magnitude = (
                self.spline_scaler.unsqueeze(-1) * self.spline_weight
            ).abs()
        else:
            magnitude = self.spline_weight.abs()

        # Per-edge mean over basis dimension -> [O, I]
        per_edge = magnitude.mean(dim=-1)

        l1_loss = per_edge.sum()

        # Entropy: normalised per output node
        total = per_edge.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p = per_edge / total
        p_safe = p.clamp(min=1e-8)
        entropy_loss = -(p_safe * p_safe.log()).sum()

        return regularize_activation * l1_loss + regularize_entropy * entropy_loss

