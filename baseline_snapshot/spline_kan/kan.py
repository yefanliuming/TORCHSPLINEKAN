"""
KAN: stacked KANLinear layers forming a Kolmogorov-Arnold Network.
"""

from typing import List

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
    ) -> torch.Tensor:
        """
        Sum of regularisation losses over all layers.

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
        mode : {"activation", "coefficient"}, keyword-only
            "activation" (default) — uses the full edge activation on actual
            layer inputs.  x is required.
            "coefficient" — legacy coefficient-based loss; x is unused but
            must still be supplied so that both modes share the same signature.

        Raises
        ------
        ValueError
            If mode="activation" and x is not supplied (already enforced by
            the keyword-only requirement, but also re-raised by the layer).
        """
        total = x.new_zeros(())
        h = x
        for layer in self.layers:
            total = total + layer.regularization_loss(
                reg_act=reg_act,
                reg_entropy=reg_entropy,
                x=h,
                mode=mode,
            )
            with torch.no_grad():
                h = layer(h)
        return total
