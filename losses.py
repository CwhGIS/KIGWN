"""Training loss terms for the three KIGWN prediction tasks.

Overflow magnitude and hydraulic head use mean squared error, overflow
occurrence uses focal loss, and the three terms carry equal weight.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def overflow_regression_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on the overflow magnitude."""

    return F.mse_loss(prediction, target)


def head_regression_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on the hydraulic head."""

    return F.mse_loss(prediction, target)


def occurrence_focal_loss(
    probability: Tensor,
    target: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Binary focal loss on the overflow occurrence probability.

    Args:
        probability: Occurrence probability, already squashed to [0, 1].
        target: Binary occurrence label in {0, 1}.
        gamma: Focusing parameter that down-weights well-classified samples.
        alpha: Weight for the positive (overflow) class.
    """

    eps = 1e-7
    p = probability.clamp(eps, 1.0 - eps)
    ce = F.binary_cross_entropy(p, target, reduction="none")
    p_t = p * target + (1.0 - p) * (1.0 - target)
    loss = ce * (1.0 - p_t).pow(gamma)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * loss).mean()


def kigwn_total_loss(
    outputs: Mapping[str, Tensor],
    overflow_target: Tensor,
    head_target: Tensor,
    occurrence_target: Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> dict[str, Tensor]:
    """Assemble the three equally weighted task loss terms.

    Args:
        outputs: Forward outputs holding ``overflow_mag``, ``pred_head``, and
            ``prob_flood``.
        overflow_target: Target for the probability-weighted overflow output.
        head_target: Target hydraulic head.
        occurrence_target: Binary overflow occurrence label.

    Returns:
        Dictionary with ``total``, ``overflow``, ``head``, and ``cls`` entries.
    """

    overflow = overflow_regression_loss(outputs["overflow_with_prob"], overflow_target)
    head = head_regression_loss(outputs["pred_head"], head_target)
    cls = occurrence_focal_loss(outputs["prob_flood"], occurrence_target, gamma, alpha)
    return {"total": overflow + head + cls, "overflow": overflow, "head": head, "cls": cls}


__all__ = [
    "head_regression_loss",
    "kigwn_total_loss",
    "occurrence_focal_loss",
    "overflow_regression_loss",
]
