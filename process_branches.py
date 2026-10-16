"""Process-informed and hurdle-guided branches used by KIGWN."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def physics_features(head_phys: Tensor, invert_elev: Tensor, max_depth: Tensor) -> Tensor:
    """Build the four head-exceedance features used by the physics branch."""

    eps = 1e-6
    head = torch.nan_to_num(head_phys, nan=0.0, posinf=1e4, neginf=-1e4)
    invert = torch.nan_to_num(invert_elev, nan=0.0, posinf=1e4, neginf=-1e4)
    depth = torch.clamp(
        torch.nan_to_num(max_depth, nan=1.0, posinf=10.0, neginf=1.0),
        min=eps,
    )
    fill_ratio = torch.clamp(
        torch.nan_to_num((head - invert) / depth, nan=0.0, posinf=10.0, neginf=-10.0),
        min=0.0,
        max=10.0,
    )
    exceedance = F.relu(fill_ratio - 1.0)
    sqrt_eps = torch.sqrt(exceedance.new_tensor(eps))
    return torch.nan_to_num(
        torch.cat(
            (
                fill_ratio,
                F.relu(1.0 - fill_ratio),
                torch.sqrt(exceedance + eps) - sqrt_eps,
                torch.pow(exceedance, 1.5),
            ),
            dim=1,
        ),
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )


def build_physics_embedder(n_features: int, embed_dim: int) -> nn.Sequential:
    """Create the 1x1 projection for the physics-derived features."""

    return nn.Sequential(
        nn.Conv2d(n_features, embed_dim, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        nn.ReLU(),
    )


def build_hurdle_branch(overflow_in: int, c_skip: int) -> nn.Sequential:
    """Create the occurrence-probability branch."""

    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(overflow_in, c_skip, kernel_size=1),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(c_skip, 1, kernel_size=1),
        nn.Sigmoid(),
    )


def build_overflow_branch(overflow_in: int, c_skip: int) -> nn.Sequential:
    """Create the raw overflow-magnitude regression branch."""

    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(overflow_in, c_skip, kernel_size=1),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(c_skip, 1, kernel_size=1),
    )


__all__ = [
    "build_hurdle_branch",
    "build_overflow_branch",
    "build_physics_embedder",
    "physics_features",
]
