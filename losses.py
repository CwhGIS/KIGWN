"""Training loss terms, Eq. (15) of the manuscript.

``L_total = w_head * L_head + w_cls * L_cls + w_overflow * L_overflow``. The
weights default to one, the equal-weight form of Eq. (15).

* ``L_overflow`` is the MSE of the magnitude branch ``Q_raw`` restricted to
  overflow-positive cells. Overflow is zero-inflated, so averaging over all
  cells would let the zeros dominate and the magnitude branch would learn to
  predict zero. The threshold comes from the occurrence label, i.e. the same
  cells that contribute to ``L_cls`` as positives.
* ``L_head`` is the MSE between the predicted and observed hydraulic head.
* ``L_cls`` is the binary focal loss (``alpha=0.25``, ``gamma=2.0``) on
  ``P_flood``.

Prediction and target are compared in the normalised target space described in
``config.json``; the package does not apply the inverse transform.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def overflow_regression_loss(
    prediction: Tensor,
    target: Tensor,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    """Mean squared error on the overflow magnitude at overflow-positive cells.

    Args:
        prediction: ``Q_raw`` in normalised target space.
        target: Normalised overflow target.
        positive_mask: Boolean mask of the cells where overflow occurs. Defaults
            to ``target > 0``, which holds because the normalised target of a
            dry cell is exactly zero under the log1p / min-max convention.
    """

    if positive_mask is None:
        positive_mask = target > 0
    else:
        positive_mask = positive_mask.to(dtype=torch.bool).expand_as(target)
    if not bool(positive_mask.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[positive_mask], target[positive_mask])


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
        probability: Occurrence probability, already squashed to [0, 1]. It is
            clamped away from the endpoints so that the log stays finite.
        target: Binary occurrence label in {0, 1}.
        gamma: Focusing parameter that down-weights well-classified samples.
        alpha: Weight for the positive (overflow) class.
    """

    eps = 1e-2
    # The training code clamped the probability to [1e-2, 1 - 1e-2] rather than
    # to the smallest representable margin; matching it keeps the reported runs
    # reproducible at saturated probabilities.
    p = probability.clamp(eps, 1.0 - eps)
    ce = F.binary_cross_entropy(p, target, reduction="none")
    p_t = p * target + (1.0 - p) * (1.0 - target)
    loss = ce * (1.0 - p_t).pow(gamma)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * loss).mean()


def kigwn_total_loss(
    outputs: Mapping[str, Optional[Tensor]],
    overflow_target: Tensor,
    head_target: Tensor,
    occurrence_target: Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    task_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Tensor]:
    """Assemble the three task losses of Eq. (15).

    ``L_overflow`` uses ``overflow_mag`` (``Q_raw``), not the gated output
    ``overflow_with_prob``: the gate is trained separately by ``L_cls``, and
    weighting the regression term by a probability that starts near a constant
    only rescales it.

    Args:
        outputs: Forward outputs holding ``overflow_mag``, ``pred_head`` and
            ``prob_flood``. ``prob_flood`` is ``None`` when the model was built
            with ``use_hurdle=False``, in which case this three-term loss does
            not apply.
        overflow_target: Normalised overflow target.
        head_target: Target hydraulic head, normalised like ``pred_head``.
        occurrence_target: Binary overflow occurrence label.
        gamma: Focusing parameter of the occurrence focal loss.
        alpha: Positive-class weight of the occurrence focal loss.
        task_weights: Weights for ``(head, cls, overflow)``, in the order used
            by the training loop. The default is the equal-weight Eq. (15).

    Returns:
        Dictionary with ``total`` (weighted) and the unweighted ``overflow``,
        ``head`` and ``cls`` terms.

    Raises:
        ValueError: if the hurdle outputs are missing or ``task_weights`` does
            not contain three values.
    """

    magnitude = outputs.get("overflow_mag")
    probability = outputs.get("prob_flood")
    if magnitude is None or probability is None:
        raise ValueError(
            "kigwn_total_loss requires the hurdle outputs 'overflow_mag' and "
            "'prob_flood'; build the model with use_hurdle=True or use the "
            "individual loss terms."
        )
    if len(task_weights) != 3:
        raise ValueError("task_weights must contain three values: (head, cls, overflow).")
    w_head, w_cls, w_overflow = (float(weight) for weight in task_weights)
    overflow = overflow_regression_loss(magnitude, overflow_target, occurrence_target > 0)
    head = head_regression_loss(outputs["pred_head"], head_target)
    cls = occurrence_focal_loss(probability, occurrence_target, gamma, alpha)
    total = w_head * head + w_cls * cls + w_overflow * overflow
    return {"total": total, "overflow": overflow, "head": head, "cls": cls}


__all__ = [
    "head_regression_loss",
    "kigwn_total_loss",
    "occurrence_focal_loss",
    "overflow_regression_loss",
]
