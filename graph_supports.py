"""Static graph supports for the topology and spatial-proximity views.

The spatial-proximity support is an undirected RBF kernel truncated at a
maximum distance, with a fixed number of nearest neighbours and a
median-distance kernel scale.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def build_topology_support(
    n_nodes: int,
    conduits: Sequence[tuple[int, int]],
    *,
    symmetrize: bool = False,
    self_loops: bool = False,
) -> Tensor:
    """Build the conduit-connectivity adjacency from a list of node pairs.

    Args:
        n_nodes: Number of drainage nodes.
        conduits: Conduit endpoints as ``(upstream, downstream)`` index pairs.
        symmetrize: Add the reverse of every conduit. Defaults to ``False`` so
            the conduit direction is preserved.
        self_loops: Include the identity diagonal. Defaults to ``False``.
    """

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    support = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    for upstream, downstream in conduits:
        if not (0 <= upstream < n_nodes and 0 <= downstream < n_nodes):
            raise ValueError(f"Conduit ({upstream}, {downstream}) is out of range for {n_nodes} nodes.")
        support[upstream, downstream] = 1.0
        if symmetrize:
            support[downstream, upstream] = 1.0
    if self_loops:
        support.fill_diagonal_(1.0)
    return support


def _neighbour_mask(
    dist: Tensor,
    max_radius: float,
    max_neighbors: int,
    min_neighbors: int,
) -> Tensor:
    """Select each node's neighbours, then union selections symmetrically."""

    n_nodes = dist.shape[0]
    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    for node in range(n_nodes):
        row = dist[node].clone()
        row[node] = float("inf")
        within = torch.nonzero(row <= max_radius, as_tuple=False).flatten()
        if within.numel() > 0:
            if within.numel() > max_neighbors:
                within = within[torch.argsort(row[within])[:max_neighbors]]
        else:
            keep = min(min_neighbors, n_nodes - 1)
            within = torch.argsort(row)[:keep]
        mask[node, within] = True
    return mask | mask.t()


def build_rbf_support(
    coordinates: Tensor | Sequence[Sequence[float]],
    *,
    max_radius: float = 1800.0,
    max_neighbors: int = 10,
    min_neighbors: int = 3,
) -> Tensor:
    """Build the undirected RBF spatial-proximity support.

    At most ``max_neighbors`` nearest nodes are selected per node inside
    ``max_radius``. The resulting graph is undirected, so the mask is
    symmetrized and a node's final degree can exceed ``max_neighbors`` when
    other nodes select it.

    Args:
        coordinates: Projected node coordinates, shape ``[n_nodes, 2]``. Use
            float64 for tensors/arrays with large projected-coordinate offsets.
        max_radius: Distance cutoff in metres; pairs beyond it get zero weight.
        max_neighbors: Maximum neighbours selected per node before symmetrization.
        min_neighbors: Nearest neighbours kept when a node has no neighbour
            inside the cutoff, so that no node is left isolated.
    """

    if isinstance(coordinates, Tensor):
        raw_coords = coordinates
    elif hasattr(coordinates, "dtype"):
        raw_coords = torch.as_tensor(coordinates)
    else:
        raw_coords = torch.as_tensor(coordinates, dtype=torch.float64)
    if raw_coords.ndim != 2 or raw_coords.shape[1] != 2 or raw_coords.shape[0] < 2:
        raise ValueError("coordinates must have shape [n_nodes, 2] with at least two nodes.")
    if raw_coords.dtype == torch.bool or raw_coords.is_complex():
        raise TypeError("coordinates must use a real numeric dtype.")
    if raw_coords.dtype.is_floating_point and raw_coords.dtype != torch.float64:
        resolution = torch.finfo(raw_coords.dtype).eps * raw_coords.detach().abs().max().item()
        if resolution > 0.1:
            raise ValueError(
                "Large projected coordinates require float64 input; converting an already-rounded "
                "float32 tensor cannot recover its spatial precision."
            )
    coords = raw_coords.to(dtype=torch.float64)
    if not torch.isfinite(coords).all():
        raise ValueError("coordinates must contain only finite values.")
    # Center projected coordinates before pairwise distances. This avoids
    # float cancellation when coordinates have large map-coordinate offsets.
    coords = coords - coords.mean(dim=0, keepdim=True)
    dist = torch.cdist(coords, coords)
    mask = _neighbour_mask(dist, max_radius, max_neighbors, min_neighbors)

    neighbour_distances = dist[torch.triu(mask, diagonal=1)]
    if neighbour_distances.numel() == 0:
        raise ValueError("No neighbour pairs were selected; check the coordinates.")
    sigma = torch.quantile(neighbour_distances, 0.5).clamp(min=1e-6)

    # The exponent is clamped so that a selected pair cannot decay to exactly
    # zero in float32. Without the clamp, a fallback neighbour kept only to
    # avoid isolation would still receive a zero weight.
    exponent = -dist.pow(2) / (2.0 * sigma.pow(2))
    weights = torch.exp(exponent.clamp(min=-60.0))
    return torch.where(mask, weights, torch.zeros_like(weights)).to(dtype=torch.float32)


__all__ = ["build_rbf_support", "build_topology_support"]
