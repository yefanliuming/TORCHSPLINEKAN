"""
KAN: stacked KANLinear layers forming a Kolmogorov-Arnold Network.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .kan_linear import KANLinear


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network built from stacked KANLinear layers.

    Parameters
    ----------
    layers_hidden : list of int
        Sequence of feature sizes.  For example [16, 32, 8] creates two
        layers: 16 -> 32 and 32 -> 8.
    grid_size : int
        Number of grid intervals G.  Default 5.
    spline_order : int
        B-spline order k (0-3).  Default 3.
    scale_noise, scale_base, scale_spline : float
        Initialisation scale factors forwarded to each KANLinear.
    enable_standalone_scale_spline : bool
        Whether each layer learns a per-edge spline scaler.  Default True.
    base_activation : nn.Module class or instance
        Activation for the base (residual) path.  Default SiLU.
    grid_eps : float
        Mixing fraction for the grid-update quantile/uniform blend.
    grid_range : (float, float)
        Initial knot domain for all layers.
    backend : str
        "auto", "cuda", or "reference" — forwarded to each KANLinear.
    """

    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1.0, 1.0),
        backend: str = "auto",
    ) -> None:
        super().__init__()

        if len(layers_hidden) < 2:
            raise ValueError(
                "layers_hidden must have at least two entries "
                "(in_features and out_features of the first layer), "
                f"got {layers_hidden}"
            )

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                KANLinear(
                    in_features=layers_hidden[i],
                    out_features=layers_hidden[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    backend=backend,
                )
                for i in range(len(layers_hidden) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """
        Update each layer's knot grid using a forward pass through the network.

        The input to each layer is the output of the previous one, so each
        layer's grid is updated with the distribution it actually sees during
        inference.
        """
        with torch.no_grad():
            for layer in self.layers:
                layer.update_grid(x, margin=margin)
                x = layer(x)

    def regularization_loss(
        self,
        reg_act: float = 1.0,
        reg_entropy: float = 1.0,
        *,
        x: torch.Tensor,
        mode: str = "activation",
        reg_impl: str = "auto",
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """
        Sum of regularisation losses over all layers (paper eq. 22).

        Parameters
        ----------
        reg_act : float
            Weight for the L1 activation penalty term.
        reg_entropy : float
            Weight for the entropy term.
        x : Tensor, keyword-only
            Network input.  Supports arbitrary leading batch dimensions.
            Each layer's regularisation is evaluated on its actual input
            (the output of the previous layer), so grids are not touched.
            Hidden activations are NOT detached: gradients of later layers'
            penalties flow back into earlier layers' parameters.
        mode : {"activation", "coefficient"}, keyword-only
            "activation" (default) — uses the full edge activation on actual
            layer inputs.  x is required.
            "coefficient" — legacy coefficient-based loss; x is unused but
            must still be supplied so that both modes share the same signature.
        reg_impl, chunk_size : keyword-only
            See :meth:`KANLinear.forward_with_regularization`.

        Notes
        -----
        This recomputes the network forward.  When the output is also needed
        (the usual training step), :meth:`forward_with_regularization`
        returns both from a single pass.
        """
        if mode == "coefficient":
            total = x.new_zeros(())
            for layer in self.layers:
                total = total + layer.regularization_loss(
                    reg_act, reg_entropy, mode="coefficient")
            return total
        _, reg = self.forward_with_regularization(
            x, reg_act, reg_entropy, mode=mode, reg_impl=reg_impl, chunk_size=chunk_size)
        return reg

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
        from one pass: each layer's base activation and spline basis are
        computed once and shared by its output and its penalty.  The hidden
        activations stay in the autograd graph.
        """
        if mode == "coefficient":
            return self(x), self.regularization_loss(
                reg_act, reg_entropy, x=x, mode="coefficient")
        total = x.new_zeros(())
        h = x
        for layer in self.layers:
            h, r = layer.forward_with_regularization(
                h, reg_act, reg_entropy, mode=mode, reg_impl=reg_impl, chunk_size=chunk_size)
            total = total + r
        return h, total

    def export_inference(self, requires_grad: bool = False) -> "KAN":
        """
        Returns an independent KAN whose layers come from
        :meth:`KANLinear.export_inference` (spline_scaler folded into the
        spline coefficients).  Same function, O*I fewer parameters per layer.
        Rebuild it from a checkpoint with
        ``KAN(widths, ..., enable_standalone_scale_spline=False)``.
        """
        new = KAN.from_layers([l.export_inference(requires_grad) for l in self.layers])
        return new.train(self.training and requires_grad)

    @classmethod
    def from_layers(cls, layers: List[KANLinear]) -> "KAN":
        """Builds a KAN around existing KANLinear layers (shapes must chain)."""
        for a, b in zip(layers[:-1], layers[1:]):
            if a.out_features != b.in_features:
                raise ValueError(
                    f"layer widths do not chain: {a.out_features} -> {b.in_features}")
        net = cls.__new__(cls)
        nn.Module.__init__(net)
        net.layers = nn.ModuleList(layers)
        return net

    @property
    def widths(self) -> List[int]:
        return [self.layers[0].in_features] + [l.out_features for l in self.layers]

    def prune(self, x_calib: torch.Tensor, threshold: float = 1e-2, **kw):
        """Physical node pruning (paper App. M, eq. 23).  See spline_kan.pruning.prune_nodes."""
        from .pruning import prune_nodes
        return prune_nodes(self, x_calib, threshold, **kw)

    @torch.no_grad()
    def extend_grid(self, new_grid_size: int, x: torch.Tensor, **kw) -> "KAN":
        """
        Returns a NEW KAN with every layer refined to ``new_grid_size``
        (see :meth:`KANLinear.extend_grid`).  Each layer is fitted on its
        actual input, i.e. the ORIGINAL network's hidden activations, so each
        refined edge approximates the corresponding original edge.  Parameter
        shapes change: build a new optimiser and save new checkpoints with the
        new grid_size.
        """
        layers, h = [], x
        for layer in self.layers:
            layers.append(layer.extend_grid(new_grid_size, h, **kw))
            h = layer(h)
        return KAN.from_layers(layers).train(self.training)
