"""Graph supports used by the full-view KIGWN model.

Holds the adaptive adjacency and the bidirectional diffusion layer.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class AdaptiveAdjacency(nn.Module):
    """Learnable data-driven graph support used by the adaptive view."""

    def __init__(self, n_nodes: int, rank: int = 16) -> None:
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n_nodes, rank) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, rank) * 0.1)

    def forward(self) -> Tensor:
        adjacency = torch.relu(self.e1 @ self.e2.t())
        adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.softmax(adjacency, dim=1)


def _row_random_walk(adjacency: Tensor) -> Tensor:
    degree = torch.clamp(adjacency.sum(dim=1, keepdim=True), min=1e-6)
    return adjacency / degree


class DiffusionConv(nn.Module):
    """Bidirectional K-step diffusion over every supplied graph support."""

    def __init__(self, c_in: int, c_out: int, k: int, n_supports: int) -> None:
        super().__init__()
        self.k = k
        self.mlp = nn.Conv2d((2 * k) * n_supports * c_in, c_out, kernel_size=1)

    @staticmethod
    def _node_conv(x: Tensor, adjacency: Tensor) -> Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, adjacency)

    def forward(self, x: Tensor, supports: Sequence[Tensor]) -> Tensor:
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
