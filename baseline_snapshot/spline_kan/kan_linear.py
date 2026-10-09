"""
KANLinear: a single B-spline KAN layer that wraps the reference backend
and (optionally) a native CUDA extension.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional CUDA extension
# ---------------------------------------------------------------------------
_CUDA_EXT_LOADED: bool = False
_C = None

try:
    from spline_kan import _C as _ext
    _C = _ext
    _CUDA_EXT_LOADED = True
except Exception:
    pass

from .reference_backend import ReferenceKANLayer, compute_basis, curve2coeff


class KANLinear(nn.Module):
    """
    B-spline KAN layer.

    Parameters
    ----------
    in_features, out_features : int
    grid_size : int
        Number of grid intervals G.  Default 5.
    spline_order : int
        B-spline order k (0-3).  Default 3.
    scale_noise, scale_base, scale_spline : float
        Initialisation scale factors.
    enable_standalone_scale_spline : bool
        When True, a learnable per-edge scaler ``spline_scaler`` multiplies the
        spline contribution.  Default True.
    base_activation : nn.Module class or instance
        Activation applied to the base (residual) path.  Default SiLU.
    grid_eps : float
        Mixing fraction for uniform vs. quantile grid spacing in
        ``update_grid``.  Default 0.02.
    grid_range : (float, float)
        Initial knot domain.  Default (-1, 1).
    backend : str
        "auto"      — CUDA if the extension is loaded, else reference.
        "cuda"      — CUDA extension; raises RuntimeError if unavailable.
        "reference" — always use the pure-PyTorch backend.

    Attributes
    ----------
    base_weight  : nn.Parameter [out_features, in_features]
    spline_weight: nn.Parameter [out_features, in_features, grid_size + spline_order]
    spline_scaler: nn.Parameter [out_features, in_features]  (if enabled)
    grid          : buffer      [in_features, grid_size + 2*spline_order + 1]
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
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        backend: str = "auto",
    ) -> None:
        super().__init__()

        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if spline_order not in (1, 2, 3):
            raise ValueError(
                f"spline_order must be 1, 2, or 3, got {spline_order}"
            )

        if backend == "cuda":
            if not _CUDA_EXT_LOADED:
                raise RuntimeError(
                    "backend='cuda' requested but the spline_kan CUDA extension "
                    "could not be loaded.  Build the extension first or use "
                    "backend='reference'."
                )
            self._use_cuda = True
        elif backend == "reference":
            self._use_cuda = False
        elif backend == "auto":
            self._use_cuda = _CUDA_EXT_LOADED
        else:
            raise ValueError(
                f"backend must be 'auto', 'cuda', or 'reference', got {backend!r}"
            )

        # Build the reference layer — it owns all parameters and buffers.
        self._ref = ReferenceKANLayer(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        # Expose parameters and buffers as direct attributes so this module
        # participates correctly in nn.Module state_dict, to(), etc.
        self.base_weight = self._ref.base_weight
        self.spline_weight = self._ref.spline_weight
        if enable_standalone_scale_spline:
            self.spline_scaler = self._ref.spline_scaler

        # Re-register the grid buffer on this module (not only the sub-module)
        # so users can call .buffers() and get it at the top level.
        self.register_buffer("grid", self._ref.grid)

        # Store config for introspection
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def selected_backend(self) -> str:
        """Returns "cuda" or "reference"."""
        return "cuda" if self._use_cuda else "reference"

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_cuda:
            return self._cuda_forward(x)
        return self._ref(x)

    def _cuda_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError(
                "KANLinear with backend='cuda' received a CPU tensor. "
                "Move the input and model to a CUDA device, or use backend='reference'."
            )
        # Flatten leading dims, call through KANFunction for autograd, reshape.
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        has_scaler = self.enable_standalone_scale_spline
        scaler = self.spline_scaler if has_scaler else torch.empty(0, device=x.device, dtype=x.dtype)

        # Route through kan_apply (KANFunction) so backward uses native code.
        out = _C.kan_apply(
            x_flat,
            self.grid,
            self.base_weight,
            self.spline_weight,
            scaler,
            self.spline_order,
            has_scaler,
        )
        return out.reshape(orig_shape[:-1] + (self.out_features,))

    # ------------------------------------------------------------------
    # Grid update
    # ------------------------------------------------------------------

    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """
        Relocate knots to better cover the observed x distribution,
        then refit spline_weight.  See ReferenceKANLayer.update_grid for
        the full specification.
        """
        self._ref.update_grid(x, margin=margin)
        # Keep the top-level grid buffer in sync.
        with torch.no_grad():
            self.grid.copy_(self._ref.grid)

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def regularization_loss(
        self,
        reg_act: float = 1.0,
        reg_entropy: float = 1.0,
        *,
        x: Optional[torch.Tensor] = None,
        mode: str = "activation",
    ) -> torch.Tensor:
        """
        Returns a scalar regularisation loss.

        Parameters
        ----------
        reg_act : float
            Weight for the L1 activation penalty term.
        reg_entropy : float
            Weight for the entropy term.
        x : Tensor, keyword-only
            Input samples to this layer.  Required when mode="activation"
            (the default).  Supports arbitrary leading batch dimensions.
        mode : {"activation", "coefficient"}, keyword-only
            "activation"  (default) — evaluates the full edge activation on
            the supplied samples x, then computes the mean absolute value per
            edge, combining an L1 term and an entropy term normalised globally
            across all edges in the layer.
            "coefficient" — legacy mode; uses mean |scaled coefficient| per
            edge with per-output-node entropy normalisation.  x is ignored.

        Raises
        ------
        ValueError
            If mode="activation" and x is not supplied.
        """
        return self._ref.regularization_loss(
            regularize_activation=reg_act,
            regularize_entropy=reg_entropy,
            x=x,
            mode=mode,
        )
