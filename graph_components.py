"""Graph supports used by the full-view KIGWN model.

Holds the data-driven adjacency (Eq. 8) and the bidirectional multi-support
diffusion layer (Eqs. 4-6 and 9).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class AdaptiveAdjacency(nn.Module):
    """Data-driven graph support, ``A_data = softmax(relu(E1 E2.T), dim=1)`` (Eq. 8).

    ``E1`` and ``E2`` are learnable ``[n_nodes, rank]`` node embeddings. The
    softmax runs along ``dim=1`` (targets), so every row already sums to one and
    the row normalisation applied later by :class:`DiffusionConv` is a no-op for
    this support.

    The ``0.1`` initialisation scale keeps ``relu(E1 E2.T)`` close to zero, so
    the softmax starts close to uniform: training begins from an almost fully
    connected, nearly homogeneous adaptive support that the data then sharpens.
    """

    def __init__(self, n_nodes: int, rank: int = 16) -> None:
        super().__init__()
        if n_nodes <= 0 or rank <= 0:
            raise ValueError("n_nodes and rank must be positive.")
        self.e1 = nn.Parameter(torch.randn(n_nodes, rank) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, rank) * 0.1)

    def forward(self) -> Tensor:
        adjacency = torch.relu(self.e1 @ self.e2.t())
        adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.softmax(adjacency, dim=1)


def _row_random_walk(adjacency: Tensor) -> Tensor:
    """Row normalisation ``RW(.)`` of Eqs (4)-(6): divide each row by its sum."""

    degree = torch.clamp(adjacency.sum(dim=1, keepdim=True), min=1e-6)
    return adjacency / degree


class DiffusionConv(nn.Module):
    """Bidirectional K-step diffusion over every supplied graph support.

    For each support ``A`` the layer builds the forward operator ``P_f = RW(A)``
    and the reverse operator ``P_b = RW(A.T)`` and applies ``K`` successive
    propagations. Only the propagated states are concatenated (the 0-step input
    is not), in support-major order and, within a support, as
    ``[P_f^1, P_b^1, ..., P_f^K, P_b^K]``. A 1x1 convolution then maps the
    ``2 * K * n_supports * c_in`` channels to ``c_out``.

    ``A[i, j]`` is read as "information flows from ``i`` to ``j``", i.e.
    ``P_f`` aggregates along the conduit direction and ``P_b`` against it.
    """

    def __init__(self, c_in: int, c_out: int, k: int, n_supports: int) -> None:
        super().__init__()
        if c_in <= 0 or c_out <= 0:
            raise ValueError("c_in and c_out must be positive.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        if n_supports < 1:
            raise ValueError("n_supports must be at least 1.")
        self.c_in = c_in
        self.k = k
        self.n_supports = n_supports
        self.mlp = nn.Conv2d((2 * k) * n_supports * c_in, c_out, kernel_size=1)

    @staticmethod
    def _node_conv(x: Tensor, adjacency: Tensor) -> Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, adjacency)

    def forward(self, x: Tensor, supports: Sequence[Tensor]) -> Tensor:
        if x.shape[1] != self.c_in:
            raise ValueError(f"Expected {self.c_in} input channels, got {x.shape[1]}.")
        if len(supports) != self.n_supports:
            raise ValueError(
                f"Expected {self.n_supports} graph supports, got {len(supports)}."
            )
        outputs = []
        for adjacency in supports:
            forward_rw = _row_random_walk(adjacency)
            backward_rw = _row_random_walk(adjacency.t())
            forward_state, backward_state = x, x
            for _ in range(self.k):
                forward_state = self._node_conv(forward_state, forward_rw)
                backward_state = self._node_conv(backward_state, backward_rw)
                outputs.extend((forward_state, backward_state))
        return self.mlp(torch.cat(outputs, dim=1))


__all__ = ["AdaptiveAdjacency", "DiffusionConv"]
