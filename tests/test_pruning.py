"""Phase-5 tests: physical node pruning (paper App. M eq. 23)."""
import io

import pytest
import torch
import torch.nn as nn

from spline_kan import KAN, prune_nodes
from spline_kan import kan_linear as kl

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and kl._CUDA_EXT_LOADED else [])


def _kill_nodes(model, layer_idx, nodes, side):
    """Zero all edges into (side='in') or out of (side='out') hidden nodes."""
    with torch.no_grad():
        if side == "in":
            l = model.layers[layer_idx - 1]
            l.base_weight[nodes] = 0; l.spline_weight[nodes] = 0
        else:
            l = model.layers[layer_idx]
            l.base_weight[:, nodes] = 0; l.spline_weight[:, nodes] = 0


@pytest.mark.parametrize("device", DEVICES)
def test_known_redundant_nodes_removed(device):
    torch.manual_seed(0)
    m = KAN([3, 6, 5, 2], grid_size=4).to(device).double()
    with torch.no_grad():                      # make live edges clearly important
        for l in m.layers:
            l.base_weight.mul_(10)
    _kill_nodes(m, 1, [1, 4], "out")     # hidden-1 nodes 1,4: no outgoing edges
    _kill_nodes(m, 2, [0], "in")         # hidden-2 node 0: no incoming edges
    _kill_nodes(m, 2, [3], "out")        # hidden-2 node 3: no outgoing edges
    x = torch.rand(300, 3, device=device, dtype=torch.float64) * 2 - 1
    res = prune_nodes(m, x, threshold=1e-2)
    assert res.kept == [[0, 2, 3, 5], [1, 2, 4]]
    assert res.widths_before == [3, 6, 5, 2] and res.widths_after == [3, 4, 3, 2]
    p = res.model
    assert p.layers[0].base_weight.shape == (4, 3)
    assert p.layers[1].spline_weight.shape == (3, 4, 4 + 3)
    assert p.layers[1].grid.shape == (4, 4 + 2 * 3 + 1)
    assert p.layers[2].base_weight.shape == (2, 3)
    assert torch.equal(p.layers[1].grid, m.layers[1].grid[[0, 2, 3, 5]])
    assert torch.equal(p.layers[1].grid, p.layers[1]._ref.grid)
    assert torch.equal(p.layers[2].spline_weight, m.layers[2].spline_weight[:, [1, 2, 4]])
    assert res.params_after < res.params_before
    # hidden-2 node 0 has zero incoming edges, so its value is constant 0 and
    # its outgoing edges output the constant phi(0); removing it shifts the
    # output by exactly that constant.
    l2 = m.layers[2]
    with torch.no_grad():
        z = torch.zeros(1, 5, device=device, dtype=torch.float64)
        only0 = l2.make_like()
        only0.load_state_dict(l2.state_dict())
        keep = torch.zeros(5, dtype=torch.bool, device=device); keep[0] = True
        only0.base_weight[:, ~keep] = 0; only0.spline_weight[:, ~keep] = 0
        shift = only0(z)
    xt = torch.rand(50, 3, device=device, dtype=torch.float64) * 2 - 1
    assert torch.allclose(res.model(xt) + shift, m(xt), atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_pruning_dead_outgoing_nodes_is_exact(device):
    torch.manual_seed(1)
    m = KAN([4, 7, 3], grid_size=5, base_activation=nn.Tanh).to(device).double()
    _kill_nodes(m, 1, [0, 3, 6], "out")
    x = torch.rand(200, 4, device=device, dtype=torch.float64) * 2 - 1
    res = m.prune(x)
    assert res.kept == [[1, 2, 4, 5]]
    assert torch.allclose(res.model(x), m(x), atol=1e-12)
    assert isinstance(res.model.layers[0]._ref.base_activation, nn.Tanh)
    expected = sum(p.numel() for p in m.parameters()) - 3 * (4 * (1 + 8 + 1) + 3 * (1 + 8 + 1))
    assert res.params_after == expected


def test_scores_follow_eq23():
    torch.manual_seed(2)
    m = KAN([2, 3, 2], grid_size=3).double()
    x = torch.rand(64, 2, dtype=torch.float64) * 2 - 1
    res = prune_nodes(m, x, threshold=0.0)
    # recompute from full edge activations
    from spline_kan.reference_backend import compute_basis

    def edge(layer, h):
        H = compute_basis(h, layer.grid, layer.spline_order)
        W = layer.spline_weight * layer.spline_scaler[..., None]
        phi = layer.base_weight * torch.nn.functional.silu(h)[:, None] + torch.einsum("nir,oir->noi", H, W)
        return phi.abs().mean(0)
    e0, e1 = edge(m.layers[0], x), edge(m.layers[1], m.layers[0](x))
    assert torch.allclose(res.scores_in[0], e0.amax(1))
    assert torch.allclose(res.scores_out[0], e1.amax(0))


def test_min_keep_and_explicit_keep():
    m = KAN([2, 4, 1], grid_size=3)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    x = torch.rand(20, 2)
    res = prune_nodes(m, x, min_keep=1)
    assert res.widths_after == [2, 1, 1]
    res2 = prune_nodes(m, keep=[[0, 3]])
    assert res2.widths_after == [2, 2, 1] and res2.old_to_new == [{0: 0, 3: 1}]
    with pytest.raises(ValueError):
        prune_nodes(m, keep=[[]])


@pytest.mark.parametrize("device", DEVICES)
def test_finetune_after_prune_with_new_optimizer_and_checkpoint(device):
    torch.manual_seed(3)
    m = KAN([3, 8, 1], grid_size=4).to(device)
    _kill_nodes(m, 1, [2, 5], "out")
    x = torch.rand(256, 3, device=device) * 2 - 1
    y = (x[:, :1] * x[:, 1:2]).sin()
    res = m.prune(x)
    p = res.model
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)     # new optimizer for new shapes
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out, reg = p.forward_with_regularization(x, 1e-4, 1e-4)
        loss = ((out - y) ** 2).mean()
        (loss + reg).backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    buf = io.BytesIO()
    torch.save({"model": p.state_dict(), "opt": opt.state_dict(), "widths": p.widths}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location=device)
    q = KAN(ck["widths"], grid_size=4).to(device)
    q.load_state_dict(ck["model"])
    opt2 = torch.optim.Adam(q.parameters(), lr=1e-2)
    opt2.load_state_dict(ck["opt"])
    assert torch.allclose(q(x), p(x))
    with pytest.raises(RuntimeError):          # original-shape model cannot load it
        KAN([3, 8, 1], grid_size=4).to(device).load_state_dict(ck["model"])
