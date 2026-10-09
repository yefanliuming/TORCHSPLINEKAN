"""
KANLinear: a single B-spline KAN layer that wraps the reference backend
and (optionally) a native CUDA extension.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from .reference_backend import (
    ReferenceKANLayer,
    compute_basis,
    curve2coeff,
    edge_abs_mean_dense,
    regularization_from_edge_means,
)
from . import fused as _fused

REG_IMPLS = ("auto", "dense", "fused", "checkpoint")

# Must match KAN_CUDA_MAX_DEG0_BASIS in csrc/kan_ops.h: the CUDA basis kernel
# holds G + 2k degree-0 values in a fixed per-thread array.
CUDA_MAX_DEG0_BASIS = int(getattr(_C, "CUDA_MAX_DEG0_BASIS", 71)) if _C is not None else 71
CUDA_MAX_IN_FEATURES = int(getattr(_C, "CUDA_MAX_IN_FEATURES", 65535)) if _C is not None else 65535


def native_config_supported(in_features: int, grid_size: int, spline_order: int) -> Tuple[bool, str]:
    """Returns (ok, reason) for whether the native CUDA kernels accept a layer shape."""
    if grid_size + 2 * spline_order > CUDA_MAX_DEG0_BASIS:
        return False, (
            f"grid_size + 2*spline_order = {grid_size + 2 * spline_order} exceeds the "
            f"CUDA kernel limit of {CUDA_MAX_DEG0_BASIS}"
        )
    if in_features > CUDA_MAX_IN_FEATURES:
        return False, (
            f"in_features = {in_features} exceeds the CUDA kernel limit of "
            f"{CUDA_MAX_IN_FEATURES}"
        )
    return True, ""


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
        "auto"      — native CUDA kernels for CUDA inputs when the extension is
                      loaded and the configuration is supported; otherwise the
                      reference backend (CPU inputs, grid too large, ...).
        "cuda"      — native CUDA kernels; raises if the extension is missing,
                      the configuration is unsupported, or the input is on CPU.
        "reference" — always use the pure-PyTorch backend.

        Any ``base_activation`` is honoured by every backend.  The native
        kernel fuses SiLU; other activations are evaluated with PyTorch and
        only the spline branch runs in the kernel.  Higher-order derivatives
        (``create_graph=True``) are exact on both backends; on the native
        backend the double-backward is assembled from differentiable ATen ops.

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

        if backend not in ("auto", "cuda", "reference"):
            raise ValueError(
                f"backend must be 'auto', 'cuda', or 'reference', got {backend!r}"
            )
        native_ok, native_reason = native_config_supported(
            in_features, grid_size, spline_order
        )
        if backend == "cuda":
            if not _CUDA_EXT_LOADED:
                raise RuntimeError(
                    "backend='cuda' requested but the spline_kan CUDA extension "
                    "could not be loaded.  Build the extension first or use "
                    "backend='reference'."
                )
            if not native_ok:
                raise ValueError(
                    f"backend='cuda' does not support this configuration: {native_reason}. "
                    "Use backend='reference' (or 'auto', which falls back automatically)."
                )
            self._use_cuda = True
        elif backend == "reference":
            self._use_cuda = False
        else:  # auto
            self._use_cuda = _CUDA_EXT_LOADED and native_ok
        self.backend = backend
        # Backend that executed the most recent forward ("cuda"/"reference").
        # Plain string, never a tensor, so it cannot keep a graph alive.
        self.last_backend: Optional[str] = None

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
        """Backend used for CUDA inputs: "cuda" (native) or "reference".

        With backend="auto", CPU inputs always run on the reference backend;
        use :meth:`backend_for` for a specific input, or ``last_backend``.
        """
        return "cuda" if self._use_cuda else "reference"

    def backend_for(self, x: torch.Tensor) -> str:
        """Backend that :meth:`forward` will use for input ``x``."""
        if not self._use_cuda:
            return "reference"
        if self.backend == "cuda" or x.is_cuda:
            return "cuda"
        return "reference"  # auto + CPU tensor

    @property
    def _native_fuses_base(self) -> bool:
        # The native kernels hard-code SiLU for the base branch.
        return type(self._ref.base_activation) is nn.SiLU

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.backend_for(x)
        self.last_backend = backend
        if backend == "cuda":
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
        empty = x_flat.new_empty(0)
        scaler = self.spline_scaler if has_scaler else empty
        fuse_base = self._native_fuses_base

        # Route through kan_apply (KANFunction) so backward uses native code.
        # An empty base_weight tells the kernel to skip the (SiLU) base branch.
        out = _C.kan_apply(
            x_flat,
            self.grid,
            self.base_weight if fuse_base else empty,
            self.spline_weight,
            scaler,
            self.spline_order,
            has_scaler,
        )
        if not fuse_base:
            out = out + F.linear(self._ref.base_activation(x_flat), self.base_weight)
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
        reg_impl: str = "auto",
        chunk_size: int = 1024,
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
        reg_impl : {"auto", "dense", "fused", "checkpoint"}, keyword-only
            How the per-edge statistic mean_n|phi| is evaluated (same maths in
            every case; see :meth:`forward_with_regularization`).
        chunk_size : int, keyword-only
            Sample chunk length for reg_impl="checkpoint".

        Raises
        ------
        ValueError
            If mode="activation" and x is not supplied.
        """
        if mode not in ("activation", "coefficient"):
            raise ValueError(
                f"mode must be 'activation' or 'coefficient', got {mode!r}"
            )
        if mode == "coefficient":
            return self._ref.regularization_loss(
                regularize_activation=reg_act,
                regularize_entropy=reg_entropy,
                mode="coefficient",
            )
        if x is None:
            # Delegate for the canonical error message.
            return self._ref.regularization_loss(reg_act, reg_entropy, x=None)
        x_flat = x.reshape(-1, self.in_features)
        s = self._ref.base_activation(x_flat)
        impl = self._resolve_reg_impl(reg_impl, x_flat)
        H = self._basis(x_flat, impl)
        a = self._edge_abs_mean(x_flat, s, H, self._ref.effective_spline_weight(),
                                impl, chunk_size)
        return regularization_from_edge_means(a, reg_act, reg_entropy)

    # ------------------------------------------------------------------
    # Fused forward + regularisation
    # ------------------------------------------------------------------

    def forward_with_regularization(
        self,
        x: torch.Tensor,
        reg_act: float = 1.0,
        reg_entropy: float = 1.0,
        *,
        mode: str = "activation",
        reg_impl: str = "auto",
        chunk_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns ``(self(x), self.regularization_loss(reg_act, reg_entropy, x=x))``
        while evaluating base_activation(x) and the B-spline basis only once.

        reg_impl selects how mean_n |phi[n,o,i]| is computed; all options give
        the same value and gradients (up to floating-point rounding) and keep
        the full graph to x and every parameter:

        "auto"       "fused" when the native CUDA kernels can run this layer on
                     this input, else "dense".
        "dense"      materialise phi [N, O, I] with PyTorch ops (reference).
        "fused"      native CUDA kernel; phi is never stored.  The backward
                     recomputes phi (R+1 FMAs per element) to get sign(phi).
                     Raises if the native kernels are unavailable.
        "checkpoint" pure PyTorch, any device: phi is built per sample chunk
                     and recomputed in backward (activation checkpointing);
                     peak memory ~ chunk_size*O*I instead of N*O*I, at the
                     cost of a second chunked forward during backward.

        Under create_graph=True (higher-order autograd) "fused" switches its
        backward to the dense differentiable formulation, so derivatives stay
        exact.  No activations are cached on the module.
        """
        if mode == "coefficient":
            return self(x), self.regularization_loss(reg_act, reg_entropy, mode="coefficient")
        if mode != "activation":
            raise ValueError(f"mode must be 'activation' or 'coefficient', got {mode!r}")
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        impl = self._resolve_reg_impl(reg_impl, x_flat)
        s = self._ref.base_activation(x_flat)
        H = self._basis(x_flat, impl)
        W = self._ref.effective_spline_weight()
        B, R = x_flat.shape[0], self.grid_size + self.spline_order
        y = F.linear(s, self.base_weight) + F.linear(
            H.reshape(B, self.in_features * R), W.reshape(self.out_features, -1))
        self.last_backend = "cuda" if impl == "fused" else "reference"
        a = self._edge_abs_mean(x_flat, s, H, W, impl, chunk_size)
        reg = regularization_from_edge_means(a, reg_act, reg_entropy)
        return y.reshape(orig_shape[:-1] + (self.out_features,)), reg

    def _resolve_reg_impl(self, reg_impl: str, x_flat: torch.Tensor) -> str:
        if reg_impl not in REG_IMPLS:
            raise ValueError(f"reg_impl must be one of {REG_IMPLS}, got {reg_impl!r}")
        native_ok = (
            _fused.HAS_FUSED_KERNELS
            and x_flat.is_cuda
            and self.backend_for(x_flat) == "cuda"
        )
        if reg_impl == "auto":
            return "fused" if native_ok else "dense"
        if reg_impl == "fused" and not native_ok:
            raise RuntimeError(
                "reg_impl='fused' needs the native CUDA kernels, a CUDA input and a "
                "layer configuration supported by backend='cuda'.  Use reg_impl="
                "'dense' or 'checkpoint' instead."
            )
        return reg_impl

    def _basis(self, x_flat: torch.Tensor, impl: str) -> Optional[torch.Tensor]:
        if impl == "fused":
            return _fused.native_basis(x_flat, self.grid, self.spline_order)
        return compute_basis(x_flat, self.grid, self.spline_order)

    def _edge_abs_mean(self, x_flat, s, H, W, impl, chunk_size):
        if impl == "fused":
            return _fused.edge_abs_mean(s, H, self.base_weight, W)
        if impl == "checkpoint":
            if chunk_size < 1:
                raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
            return _fused.edge_abs_mean_chunked(s, x_flat, self.grid, self.spline_order,
                                                self.base_weight, W, chunk_size)
        return edge_abs_mean_dense(s, H, self.base_weight, W)

    def make_like(self, in_features: Optional[int] = None, out_features: Optional[int] = None,
                  grid_size: Optional[int] = None,
                  enable_standalone_scale_spline: Optional[bool] = None) -> "KANLinear":
        """
        New layer with this layer's hyper-parameters (spline order, base
        activation (deep copy), grid_eps, grid_range, init scales, backend),
        dtype and device; optionally overriding the shape.  Parameters are
        freshly initialised; the caller copies whatever it needs.
        """
        import copy

        ref = self._ref
        return KANLinear(
            self.in_features if in_features is None else in_features,
            self.out_features if out_features is None else out_features,
            grid_size=self.grid_size if grid_size is None else grid_size,
            spline_order=self.spline_order,
            scale_noise=ref.scale_noise, scale_base=ref.scale_base,
            scale_spline=ref.scale_spline,
            enable_standalone_scale_spline=(self.enable_standalone_scale_spline
                                            if enable_standalone_scale_spline is None
                                            else enable_standalone_scale_spline),
            base_activation=copy.deepcopy(ref.base_activation),
            grid_eps=ref.grid_eps, grid_range=ref.grid_range,
            backend=self.backend,
        ).to(device=self.grid.device, dtype=self.grid.dtype)

    @torch.no_grad()
    def edge_l1(self, x: torch.Tensor) -> torch.Tensor:
        """Per-edge L1 norm |phi_{o,i}|_1 = mean_n |phi_{o,i}(x_n)|  ->  [O, I] (no grad)."""
        x_flat = x.reshape(-1, self.in_features)
        impl = self._resolve_reg_impl("auto", x_flat)
        s = self._ref.base_activation(x_flat)
        H = self._basis(x_flat, impl)
        return self._edge_abs_mean(x_flat, s, H, self._ref.effective_spline_weight(), impl, 1024)

    @torch.no_grad()
    def _copy_grid_(self, grid: torch.Tensor) -> None:
        # Module.to()/.double() re-create buffers per module, so the public
        # ``grid`` and ``_ref.grid`` are distinct tensors; keep both in sync.
        self.grid.copy_(grid)
        self._ref.grid.copy_(grid)

    # ------------------------------------------------------------------
    # Inference export
    # ------------------------------------------------------------------

    def export_inference(self, requires_grad: bool = False) -> "KANLinear":
        """
        Returns a new, independent KANLinear that computes the same function
        with ``spline_scaler`` folded into the spline coefficients:

            spline_weight_exported = spline_weight * spline_scaler[..., None]

        The exported layer has ``enable_standalone_scale_spline=False``, so it
        stores O*I fewer parameters and never rebuilds the effective weight in
        forward.  The grid (including non-uniform knots), dtype, device,
        backend choice and a deep copy of base_activation are preserved.  The
        original module is not modified and can keep training.

        The exported layer is put in eval mode and its parameters are frozen
        unless ``requires_grad=True``.  Note: fine-tuning an exported layer is
        a different parameterisation (no separate scaler), so its optimiser
        dynamics differ from the original.
        """
        ref = self._ref
        new = self.make_like(enable_standalone_scale_spline=False)
        with torch.no_grad():
            new.base_weight.copy_(self.base_weight)
            new.spline_weight.copy_(ref.effective_spline_weight())
            new._copy_grid_(self.grid)
        new.requires_grad_(requires_grad)
        return new.train(self.training and requires_grad)

    # ------------------------------------------------------------------
    # Grid extension (paper Sec. 2.4 / App. L)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extend_grid(
        self,
        new_grid_size: int,
        x: torch.Tensor,
        *,
        margin: float = 0.01,
        adapt_knots: bool = False,
        ridge: float = 1e-12,
        n_dense: int = 0,
    ) -> "KANLinear":
        """
        Returns a NEW layer with ``new_grid_size`` intervals whose spline
        coefficients are the least-squares fit (eq. 18) of this layer's
        current spline curves.  ``base_weight``, ``spline_scaler`` and the
        base activation are copied unchanged, so the new edge function is
        phi'(u) = bw*b(u) + scaler * spline_fine(u) ~= phi(u).

        Unlike :meth:`update_grid` (fixed G, moves knots in place), this
        changes parameter shapes: spline_weight [O, I, G+k] -> [O, I, G'+k]
        and grid [I, G+2k+1] -> [I, G'+2k+1].  Create a new optimiser for the
        returned layer; old state_dicts / optimiser states do not fit it.
        The global default grid_size is unaffected.

        Parameters
        ----------
        x : Tensor [..., in_features]
            Samples of this layer's actual input distribution (fit domain
            and, if adapt_knots, knot placement).
        adapt_knots : bool
            False (default): uniform knots (the paper's construction) over
            the current interior domain [t_k, t_{k+G}] of each feature,
            widened to the sample range (+margin) where samples fall outside
            it (clipped to the old knot span).  When no widening happens, the
            old grid is uniform and G' is a multiple of G, the old knots are a
            subset of the new ones: the old spline is exactly representable
            and the fit error is at round-off level.
            True: knots from the sample quantile/uniform mix (grid_eps), as
            update_grid.  Can concentrate resolution where data is, but
            clustered samples give near-coincident knots and a visibly larger
            fit error at the moment of extension (measured in
            docs/optimization_report.md).
        ridge : float
            Relative Tikhonov term of the fit; see fit_spline_coeffs.
        n_dense : int
            Additional uniformly spaced fit points per feature spanning the
            new interior domain.  Helps when ``x`` is small compared to G'
            (under-sampling): the old curve is known everywhere, so it can be
            sampled densely without extra data.
        """
        from .reference_backend import fit_spline_coeffs

        if new_grid_size < 1:
            raise ValueError(f"new_grid_size must be >= 1, got {new_grid_size}")
        x_flat = x.reshape(-1, self.in_features).to(self.grid)
        if x_flat.shape[0] < 2:
            raise ValueError("extend_grid needs at least 2 samples")
        k = self.spline_order
        new = self.make_like(grid_size=new_grid_size)
        if adapt_knots:
            new_grid = new._ref.grid_from_samples(x_flat, new_grid_size, margin)
        else:
            # Uniform knots over the old interior domain, widened to cover the
            # samples: the old spline is defined (non-zero) on the extended
            # knot span, so inputs outside the interior must stay covered.
            lo0, hi0 = self.grid[:, k], self.grid[:, k + self.grid_size]
            xmin, xmax = x_flat.amin(0), x_flat.amax(0)
            lo = torch.where(xmin < lo0, xmin - margin, lo0)
            hi = torch.where(xmax > hi0, xmax + margin, hi0)
            lo = torch.maximum(lo, self.grid[:, 0])     # old spline is 0 beyond its support
            hi = torch.minimum(hi, self.grid[:, -1])
            r = torch.arange(new_grid_size + 2 * k + 1, device=lo.device, dtype=lo.dtype) - k
            new_grid = lo[:, None] + r[None] * ((hi - lo) / new_grid_size)[:, None]
        xs = x_flat
        if n_dense > 0:
            lo, hi = new_grid[:, k], new_grid[:, k + new_grid_size]
            t = torch.linspace(0, 1, n_dense, device=lo.device, dtype=lo.dtype)
            xs = torch.cat([x_flat, (lo[None] + t[:, None] * (hi - lo)[None])], 0)
        H_old = compute_basis(xs, self.grid, k)
        y_old = torch.einsum("oir,bir->obi", self.spline_weight, H_old)   # unscaled curve
        coeffs = fit_spline_coeffs(xs, y_old, new_grid, k, ridge=ridge)
        new.base_weight.copy_(self.base_weight)
        new.spline_weight.copy_(coeffs)
        if self.enable_standalone_scale_spline:
            new.spline_scaler.copy_(self.spline_scaler)
        new._copy_grid_(new_grid)
        return new.train(self.training)
