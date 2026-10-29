"""Static graph supports for the topology and spatial-proximity views.

``build_topology_support`` returns the binary conduit adjacency of the drainage
network. ``build_rbf_support`` implements the spatial-proximity kernel of the
manuscript (Eq. 7): a Gaussian kernel on projected Euclidean distance, scaled by
the median neighbour distance and truncated at ``max_radius``.

A node without any neighbour inside ``max_radius`` would contribute an all-zero
row to the row-normalised diffusion operator, so its own state would stop
propagating through that support. ``min_neighbors`` therefore keeps that node's
nearest neighbours even when they lie outside the radius. This is a deviation
from Eq. (7) as printed; pass ``min_neighbors=0`` to reproduce the strict radius
cutoff.
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

    The matrix is binary and unnormalised; :class:`DiffusionConv` applies the
    row normalisation of Eqs (4)-(6), so an entry ``(i, j)`` means "a conduit
    runs from ``i`` to ``j``".

    Args:
        n_nodes: Number of drainage nodes.
        conduits: Conduit endpoints as ``(upstream, downstream)`` index pairs.
            Repeated pairs are idempotent.
        symmetrize: Add the reverse of every conduit. Defaults to ``False``
            because the conduit direction is physically meaningful and is
            already used in both directions: the diffusion layer normalises
            both ``A`` and ``A.T``.
        self_loops: Include the identity diagonal. Defaults to ``False``; each
            GWN block keeps the node's own state through its residual
            connection, so self loops are not needed for self-information.

    Returns:
        ``[n_nodes, n_nodes]`` float32 tensor with entries in ``{0, 1}``.
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
) -> tuple[Tensor, Tensor]:
    """Select each node's neighbours and return both the raw and unioned masks.

    A node keeps the closest ``max_neighbors`` of its neighbours inside
    ``max_radius``. When it has none inside the radius, it keeps its
    ``min_neighbors`` closest nodes regardless of distance, so that a sparse
    corner of the network is not left isolated.

    Returns:
        ``(selected, mask)``: ``selected[i, j]`` says node ``i`` picked ``j``;
        ``mask = selected | selected.T`` is the undirected support.
    """

    n_nodes = dist.shape[0]
    selected = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
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
        selected[node, within] = True
    return selected, selected | selected.t()


def build_rbf_support(
    coordinates: Tensor | Sequence[Sequence[float]],
    *,
    max_radius: float = 1800.0,
    max_neighbors: int = 10,
    min_neighbors: int = 3,
) -> Tensor:
    """Build the spatial-proximity support of the manuscript (Eq. 7).

    For every selected pair the weight is ``exp(-d**2 / (2 * sigma**2))``, with
    ``sigma`` the median distance over the per-node neighbour selections, taken
    *before* the selections are unioned: each node contributes one distance per
    neighbour it picked, so a pair selected from both sides contributes twice.
    Unselected pairs get zero. Selection proceeds as follows:

    1. each node keeps the closest ``max_neighbors`` of its neighbours inside
       ``max_radius``;
    2. a node without any neighbour inside the radius keeps its
       ``min_neighbors`` closest nodes whatever their distance;
    3. the per-node selections are unioned, so the graph is undirected and a
       node's final degree can exceed ``max_neighbors`` when other nodes select
       it.

    Step 2 is a deviation from Eq. (7) as printed; set ``min_neighbors=0`` for
    the strict radius cutoff.

    Args:
        coordinates: Projected node coordinates, shape ``[n_nodes, 2]``. Use
            float64 for tensors/arrays with large projected-coordinate offsets:
            the kernel is meaningless when the input grid step exceeds 0.1 m,
            and precision already lost to float32 rounding cannot be recovered.
        max_radius: Distance cutoff in metres for neighbour selection.
        max_neighbors: Cap on the neighbours selected per node before
            symmetrisation.
        min_neighbors: Nearest nodes kept when a node has no neighbour inside
            ``max_radius``. ``0`` disables the fallback.

    Returns:
        ``[n_nodes, n_nodes]`` float32 tensor with a zero diagonal.
    """

    if max_radius <= 0:
        raise ValueError("max_radius must be positive.")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1.")
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be non-negative.")

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
        # 0.1 m is an order of magnitude below the spatial resolution that
        # matters for a kernel truncated at 1800 m.
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
    selected, mask = _neighbour_mask(dist, max_radius, max_neighbors, min_neighbors)

    neighbour_distances = dist[selected]
    if neighbour_distances.numel() == 0:
        raise ValueError("No neighbour pairs were selected; check the coordinates.")
    sigma = torch.quantile(neighbour_distances, 0.5).clamp(min=1e-6)

    # The exponent is clamped so that a selected pair cannot underflow to
    # exactly zero in float32. Without the clamp, nodes kept only by the
    # isolation fallback would receive zero weights whenever their distance
    # makes exp(-d^2 / 2 sigma^2) vanish, and an all-zero row loses the node.
    exponent = -dist.pow(2) / (2.0 * sigma.pow(2))
    weights = torch.exp(exponent.clamp(min=-60.0))
    return torch.where(mask, weights, torch.zeros_like(weights)).to(dtype=torch.float32)


__all__ = ["build_rbf_support", "build_topology_support"]
