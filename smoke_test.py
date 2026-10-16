"""Execution check for the full-view KIGWN package."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:  # Allow ``python smoke_test.py`` as well as -m.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from final_kigwn_minimal import (
    build_final_kigwn,
    build_rbf_support,
    build_topology_support,
    kigwn_total_loss,
    load_model_config,
)
from final_kigwn_minimal.graph_components import DiffusionConv


def _check_article_graph_semantics(n_nodes: int) -> None:
    directed_probe = build_topology_support(n_nodes, [(0, 1)])
    assert directed_probe[0, 1] == 1.0
    assert directed_probe[1, 0] == 0.0
    assert torch.count_nonzero(torch.diagonal(directed_probe)) == 0

    knn_probe = build_rbf_support(
        torch.tensor([[-100.0, 0.0], [0.0, 0.0], [100.0, 0.0]], dtype=torch.float64),
        max_radius=150.0,
        max_neighbors=1,
        min_neighbors=1,
    )
    assert torch.equal(knn_probe, knn_probe.t())
    assert torch.count_nonzero(knn_probe[1]) == 2

    diffusion_probe = DiffusionConv(c_in=1, c_out=1, k=1, n_supports=1)
    assert diffusion_probe.mlp.in_channels == 2
    with torch.no_grad():
        diffusion_probe.mlp.weight.zero_()
        diffusion_probe.mlp.bias.zero_()
        diffusion_probe.mlp.weight[0, 0, 0, 0] = 1.0
    probe_x = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]])
    assert torch.equal(diffusion_probe(probe_x, [torch.zeros(2, 2)]), torch.zeros_like(probe_x))

    probe_coords = torch.tensor(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=torch.float64
    )
    offset = torch.tensor([428089.72, 4421368.74], dtype=torch.float64)
    assert torch.allclose(
        build_rbf_support(probe_coords),
        build_rbf_support(probe_coords + offset),
        atol=1e-6,
        rtol=0.0,
    )
    try:
        build_rbf_support(probe_coords.float() + offset.float())
    except ValueError as error:
        assert "float64 input" in str(error)
    else:
        raise AssertionError("Large, pre-rounded float32 coordinates must be rejected.")


def main() -> None:
    config = load_model_config()
    torch.manual_seed(0)
    n_nodes = config["n_nodes"]
    try:
        build_final_kigwn()
    except ValueError as error:
        assert "head_min_per_node" in str(error)
    else:
        raise AssertionError("Physics-enabled KIGWN must require physical node metadata.")
    model = build_final_kigwn(
        head_min_per_node=torch.zeros(n_nodes),
        head_range_per_node=torch.ones(n_nodes),
        invert_elev_per_node=torch.zeros(n_nodes),
        max_depth_per_node=torch.ones(n_nodes),
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable_parameters == 434_627
    model.eval()
    x = torch.randn(1, config["in_channels"], n_nodes, config["window_steps"])

    topology = build_topology_support(
        n_nodes, [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    )
    _check_article_graph_semantics(n_nodes)
    coords = torch.rand(n_nodes, 2) * 2000.0
    rbf = build_rbf_support(coords)
    assert topology.shape == (n_nodes, n_nodes)
    assert rbf.shape == (n_nodes, n_nodes)

    with torch.no_grad():
        outputs = model(x, [topology, rbf])
    assert outputs["pred_head"].shape == (1, 1, n_nodes)
    assert outputs["overflow_with_prob"].shape == (1, 1, n_nodes)

    terms = kigwn_total_loss(
        outputs,
        torch.zeros(1, 1, n_nodes),
        torch.zeros(1, 1, n_nodes),
        torch.zeros(1, 1, n_nodes),
    )
    assert torch.isfinite(terms["total"])

    print("KIGWN smoke test passed")
    print(f"pred_head={tuple(outputs['pred_head'].shape)}")
    print(f"overflow_with_prob={tuple(outputs['overflow_with_prob'].shape)}")
    print(f"trainable_parameters={trainable_parameters}")
    print(f"loss={float(terms['total']):.6f}")


if __name__ == "__main__":
    main()
