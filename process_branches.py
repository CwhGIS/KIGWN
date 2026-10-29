"""Process-informed and hurdle-guided branches used by KIGWN.

``physics_features`` turns the predicted hydraulic head into the four
head-derived descriptors of Section 3.5, ``build_physics_embedder`` projects
them into the 32-dimensional latent space, and the two branch builders return
``P_flood`` and ``Q_raw`` of Eqs (10), (11) and (14). The projected descriptors
are concatenated with the backbone representation, so **both** branches -- the
occurrence branch and the magnitude branch -- see them; the two branch builders
therefore take the same ``overflow_in`` width.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def physics_features(head_phys: Tensor, invert_elev: Tensor, max_depth: Tensor) -> Tensor:
    """Head-derived physical descriptors injected into the overflow branch.

    All three inputs are physical values in metres, broadcast to the same
    shape. With node depth ``D``, invert elevation ``z`` and (de-normalised)
    predicted head ``H``, the four channels are returned in this order, which
    matches ``physics_features`` in ``config.json``:

    1. ``fill_ratio = clip((H - z) / D, 0, 10)`` -- filling state of the node;
    2. ``safety_margin = relu(1 - fill_ratio)`` -- distance to the rim in units of
       depth, saturating at ``1`` once the head is a depth or more below it;
    3. ``orifice_exceedance_sqrt = sqrt(e + eps) - sqrt(eps)`` -- the
       orifice-inspired basis of Eq. (13);
    4. ``weir_exceedance_power_1_5 = e ** 1.5`` -- the weir-inspired basis of
       Eq. (12),

    where ``e = relu(fill_ratio - 1) = max(H - (z + D), 0) / D`` is the
    **depth-normalised** rim exceedance. Eqs (12)-(13) are written for the
    dimensional exceedance ``dh = max(H - (z + D), 0)`` in metres; the
    implementation divides by the node depth before applying the two basis
    functions, so the descriptors are dimensionless.

    ``clip`` saturates nodes more than ten depths above the invert and floors
    dry nodes (``H < z``) at zero, so a negative depth below the invert is not
    represented. ``eps`` (1e-6) keeps the derivative of the square root finite
    where ``e = 0``.
    """

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
    """Project the physical descriptors into the latent space of Section 3.5.

    Two 1x1 convolutions with ReLU; ``embed_dim`` is 32 in the shipped config.
    The output is concatenated with the backbone representation and the result
    is fed to both overflow branches, not only to the magnitude branch.
    """

    return nn.Sequential(
        nn.Conv2d(n_features, embed_dim, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        nn.ReLU(),
    )


def build_hurdle_branch(overflow_in: int, c_skip: int) -> nn.Sequential:
    """Occurrence-probability branch producing ``P_flood`` in ``(0, 1)``.

    A 1x1 convolutional trunk with dropout (0.1) followed by a sigmoid. The
    continuous probability is kept during training and thresholded at 0.5 for
    inference only.
    """

    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(overflow_in, c_skip, kernel_size=1),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(c_skip, 1, kernel_size=1),
        nn.Sigmoid(),
    )


def build_overflow_branch(overflow_in: int, c_skip: int) -> nn.Sequential:
    """Overflow-magnitude branch producing ``Q_raw``.

    "Raw" means the estimate is not yet multiplied by ``P_flood`` (Eq. 14); the
    branch itself has no sigmoid, and its output lives in the normalised
    overflow target space, not in m3/s. It shares the trunk design of the
    occurrence branch, without the final sigmoid.
    """

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
