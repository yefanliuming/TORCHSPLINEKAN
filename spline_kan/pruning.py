"""
Physical node pruning for KAN (paper Appendix M.1, eq. 23).

For hidden node i of layer l (the i-th output of layer l-1 == the i-th
input of layer l):

    I_{l,i} = max_k |phi_{l-1, i, k}|_1      (incoming: edges into node i)
    O_{l,i} = max_j |phi_{l,   j, i}|_1      (outgoing: edges out of node i)

with |phi|_1 = mean over calibration samples of |phi(x_s)| evaluated on the
*actual* layer inputs (full edge activation incl. base branch).  A node is
kept iff both scores exceed the threshold (default theta = 1e-2).

Pruned nodes are physically removed: the corresponding output row of layer
l-1 (base_weight, spline_weight, spline_scaler) and input column of layer l
(the same tensors plus its grid row) are sliced away and a smaller model is
built.  Network input and output widths are never changed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch

from .kan import KAN
from .kan_linear import KANLinear


@dataclass
class PruneResult:
    model: KAN
    kept: List[List[int]]                 # per hidden layer: kept original node indices
    old_to_new: List[Dict[int, int]]      # per hidden layer: original -> new index
    widths_before: List[int]
    widths_after: List[int]
    params_before: int
    params_after: int
    scores_in: List[torch.Tensor] = field(repr=False, default_factory=list)
    scores_out: List[torch.Tensor] = field(repr=False, default_factory=list)
    threshold: float = 1e-2

    def summary(self) -> dict:
        return {
            "widths_before": self.widths_before,
            "widths_after": self.widths_after,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "kept": self.kept,
            "threshold": self.threshold,
        }


@torch.no_grad()
def node_scores(model: KAN, x: torch.Tensor):
    """Returns (incoming, outgoing) score lists, one [width] tensor per hidden layer."""
    edge = []           # per layer: [O, I] mean |phi| on that layer's actual inputs
    h = x
    for layer in model.layers:
        edge.append(layer.edge_l1(h))
        h = layer(h)
    s_in, s_out = [], []
    for l in range(1, len(model.layers)):
        s_in.append(edge[l - 1].amax(dim=1))     # edges into hidden node: rows of layer l-1
        s_out.append(edge[l].amax(dim=0))        # edges out of hidden node: cols of layer l
    return s_in, s_out


def _slice_layer(layer: KANLinear, out_idx: torch.Tensor, in_idx: torch.Tensor) -> KANLinear:
    new = layer.make_like(in_features=len(in_idx), out_features=len(out_idx))
    with torch.no_grad():
        new.base_weight.copy_(layer.base_weight[out_idx][:, in_idx])
        new.spline_weight.copy_(layer.spline_weight[out_idx][:, in_idx])
        if layer.enable_standalone_scale_spline:
            new.spline_scaler.copy_(layer.spline_scaler[out_idx][:, in_idx])
        new._copy_grid_(layer.grid[in_idx])
    new.requires_grad_(any(p.requires_grad for p in layer.parameters()))
    return new.train(layer.training)


def prune_nodes(
    model: KAN,
    x_calib: Optional[torch.Tensor] = None,
    threshold: float = 1e-2,
    *,
    keep: Optional[Sequence[Sequence[int]]] = None,
    min_keep: int = 1,
) -> PruneResult:
    """
    Builds a physically smaller KAN without unimportant hidden nodes.

    Parameters
    ----------
    model : KAN
        Not modified.
    x_calib : Tensor [N, in_features]
        Calibration samples used for the edge L1 norms.  Required unless
        ``keep`` is given.
    threshold : float
        theta in eq. 23: a node is kept iff incoming AND outgoing score > theta.
    keep : list of index lists, optional
        Explicit kept indices per hidden layer (overrides scoring).
    min_keep : int
        Keep at least this many nodes per hidden layer (highest
        min(in, out) score), so the network never disconnects.

    Returns
    -------
    PruneResult with the new model (independent parameters; the optimiser
    and any checkpoint of the original no longer match its shapes, so
    create a new optimiser), index mappings, widths and parameter counts.
    """
    n_hidden = len(model.layers) - 1
    s_in: List[torch.Tensor] = []
    s_out: List[torch.Tensor] = []
    if keep is None:
        if x_calib is None:
            raise ValueError("prune_nodes needs x_calib (or explicit keep indices)")
        s_in, s_out = node_scores(model, x_calib)
        keep = []
        for si, so in zip(s_in, s_out):
            mask = (si > threshold) & (so > threshold)
            idx = mask.nonzero().flatten()
            if len(idx) < min_keep:
                idx = torch.minimum(si, so).topk(min(min_keep, len(si))).indices.sort().values
            keep.append(idx.tolist())
    if len(keep) != n_hidden:
        raise ValueError(f"keep must list {n_hidden} hidden layers, got {len(keep)}")

    dev = model.layers[0].grid.device
    widths = model.widths
    idx_all = [torch.arange(widths[0], device=dev)]
    for l, k in enumerate(keep):
        k = sorted(set(int(i) for i in k))
        if not k:
            raise ValueError(f"hidden layer {l + 1} would have no nodes")
        if k[-1] >= widths[l + 1] or k[0] < 0:
            raise ValueError(f"keep indices out of range for hidden layer {l + 1}")
        idx_all.append(torch.tensor(k, device=dev))
    idx_all.append(torch.arange(widths[-1], device=dev))

    new_layers = [_slice_layer(layer, idx_all[l + 1], idx_all[l])
                  for l, layer in enumerate(model.layers)]
    new = KAN.from_layers(new_layers).train(model.training)
    kept = [t.tolist() for t in idx_all[1:-1]]
    return PruneResult(
        model=new,
        kept=kept,
        old_to_new=[{o: n for n, o in enumerate(k)} for k in kept],
        widths_before=widths,
        widths_after=new.widths,
        params_before=sum(p.numel() for p in model.parameters()),
        params_after=sum(p.numel() for p in new.parameters()),
        scores_in=s_in,
        scores_out=s_out,
        threshold=threshold,
    )
