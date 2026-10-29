"""Full-view KIGWN model assembly.

This file defines the forward model. Data loading, preprocessing, the training
loop, and checkpoint paths are handled outside the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from .graph_components import AdaptiveAdjacency
    from .gwn_block import GWBlock
    from .model_config import load_model_config, normalize_model_config
    from .process_branches import (
        build_hurdle_branch,
        build_overflow_branch,
        build_physics_embedder,
        physics_features,
    )
except ImportError:  # Direct import when this directory is on sys.path.
    from graph_components import AdaptiveAdjacency
    from gwn_block import GWBlock
    from model_config import load_model_config, normalize_model_config
    from process_branches import (
        build_hurdle_branch,
        build_overflow_branch,
        build_physics_embedder,
        physics_features,
    )

_ARCHITECTURE_ARGUMENTS = (
    "n_nodes",
    "in_channels",
    "c_res",
    "c_skip",
    "diffusion_steps",
    "dilations",
    "use_adaptive",
    "use_physics_embedding",
    "use_hurdle",
    "n_static_supports",
    "physics_embed_dim",
    "adaptive_rank",
    "window_steps",
)

_PHYSICS_METADATA_ARGUMENTS = (
    "head_min_per_node",
    "head_range_per_node",
    "invert_elev_per_node",
    "max_depth_per_node",
    "outfall_mask_per_node",
)

_MODEL_ARGUMENTS = frozenset(_ARCHITECTURE_ARGUMENTS) | frozenset(_PHYSICS_METADATA_ARGUMENTS)


class KIGWNFullView(nn.Module):
    """Full-view KIGWN forward model (topology, spatial-proximity, data-driven).

    Input:
        x: ``[batch, in_channels, nodes, window_steps]``. The nine channels are
            listed in ``config.json``: rainfall, cumulative rainfall, area,
            imperviousness, width, slope, invert elevation, maximum depth and
            node type, each normalised as recorded there.
        static_supports: ``[topology_support, rbf_support]``, each
            ``[nodes, nodes]``. The data-driven support (Eq. 8) is built inside
            the model. Supports are cast to the model's device and dtype; they
            must be finite.

    Output, all reduced to the last input step except ``shared``:

    * ``pred_head`` ``[B, 1, N]``: min-max normalised head, not metres;
    * ``prob_flood`` ``[B, 1, N]``: ``P_flood`` in ``[0, 1]``, or ``None``;
    * ``overflow_mag`` ``[B, 1, N]``: ``Q_raw`` in the normalised overflow space,
      zeroed at outfall nodes;
    * ``overflow_with_prob`` ``[B, 1, N]``: ``P_flood * Q_raw`` (Eq. 14 as printed);
    * ``overflow_mask`` ``[B, 1, N]``: ``1.0`` where ``P_flood >= 0.5``;
    * ``overflow_hard`` ``[B, 1, N]``: ``overflow_mask * Q_raw``, the gated output
      behind the reported overflow metrics;
    * ``head_exceedance`` ``[B, 1, N]``: metres above the node rim, from the
      unclipped prediction;
    * ``shared`` ``[B, c_skip, N, T]``: backbone features, not sliced.

    ``pred_head`` is the direct output of the head predictor: the model predicts
    the normalised target and does not apply the inverse transform. Recover
    metres with ``pred_head * head_range_per_node + head_min_per_node``. The
    overflow outputs are likewise in the log1p-min-max normalised overflow
    space; those statistics are not part of the model. ``head_exceedance`` is
    the only output in physical units.
    """

    def __init__(
        self,
        n_nodes: int = 392,
        in_channels: int = 9,
        c_res: int = 64,
        c_skip: int = 128,
        diffusion_steps: int = 2,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        use_adaptive: bool = True,
        use_physics_embedding: bool = True,
        use_hurdle: bool = True,
        n_static_supports: int = 2,
        head_min_per_node: Optional[Sequence[float]] = None,
        head_range_per_node: Optional[Sequence[float]] = None,
        invert_elev_per_node: Optional[Sequence[float]] = None,
        max_depth_per_node: Optional[Sequence[float]] = None,
        outfall_mask_per_node: Optional[Sequence[float]] = None,
        physics_embed_dim: int = 32,
        adaptive_rank: int = 16,
        window_steps: int = 6,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations must contain at least one positive integer.")
        if any(type(value) is not int or value <= 0 for value in dilations):
            raise ValueError("dilations must contain only positive integers.")
        if use_physics_embedding:
            required_physics_inputs = {
                "head_min_per_node": head_min_per_node,
                "head_range_per_node": head_range_per_node,
                "invert_elev_per_node": invert_elev_per_node,
                "max_depth_per_node": max_depth_per_node,
            }
            missing = [name for name, value in required_physics_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    "Physics embedding requires per-node head statistics and hydraulic metadata: "
                    + ", ".join(missing)
                )
        self.n = n_nodes
        self.use_adp = use_adaptive
        self.use_phys_emb = use_physics_embedding
        self.use_hurdle = use_hurdle
        self.n_static_supports = n_static_supports
        self.physics_embed_dim = physics_embed_dim
        self.in_channels = in_channels
        self.window_steps = window_steps
        self.adp = AdaptiveAdjacency(n_nodes, rank=adaptive_rank) if use_adaptive else None

        support_count = n_static_supports + int(use_adaptive)
        self.input_proj = nn.Conv2d(in_channels, c_res, kernel_size=1)
        self.blocks = nn.ModuleList(
            GWBlock(c_res, c_res, c_skip, diffusion_steps, support_count, dilation)
            for dilation in dilations
        )
        self.out_shared = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(c_skip, c_skip, kernel_size=1),
            nn.ReLU(),
        )
        self.head_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(c_skip, c_skip, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(c_skip, 1, kernel_size=1),
        )

        # The descriptor count is read from the feature builder itself so that
        # the embedder and physics_features() cannot drift apart.
        with torch.no_grad():
            n_physics_features = physics_features(
                torch.zeros(1, 1, n_nodes, 1),
                torch.zeros(1, 1, n_nodes, 1),
                torch.ones(1, 1, n_nodes, 1),
            ).shape[1]
        self.physics_embedder = (
            build_physics_embedder(n_physics_features, physics_embed_dim)
            if use_physics_embedding
            else None
        )

        overflow_in = c_skip + (physics_embed_dim if use_physics_embedding else 0)
        self.gate_branch = build_hurdle_branch(overflow_in, c_skip) if use_hurdle else None
        self.overflow_branch = build_overflow_branch(overflow_in, c_skip)
        self._register_node_buffers(
            head_min_per_node,
            head_range_per_node,
            invert_elev_per_node,
            max_depth_per_node,
            outfall_mask_per_node,
        )
        # ``head_exceedance`` is de-normalised from these buffers even when the
        # physics embedding is off, so the validation is not conditional on it.
        self._validate_physics_statistics()

    def _register_node_buffers(
        self,
        head_min: Optional[Sequence[float]],
        head_range: Optional[Sequence[float]],
        invert_elev: Optional[Sequence[float]],
        max_depth: Optional[Sequence[float]],
        outfall_mask: Optional[Sequence[float]],
    ) -> None:
        """Store the per-node metadata as float32 buffers, in metres where relevant.

        ``head_min``/``head_range`` must be the min-max statistics of the head
        targets used during training, in the same node order as ``x`` and the
        graph supports; they are the de-normalisation applied before the
        physics descriptors and ``head_exceedance`` are computed.

        ``outfall_mask_per_node`` marks the outfall nodes (``OutfallType == 1``
        in the training metadata, eight nodes on this network). The forward
        pass zeroes the overflow magnitude there before the gate is applied,
        matching the training code: outfalls discharge out of the network and
        cannot overflow. The buffer keeps its checkpoint name so that
        ``load_state_dict`` maps it directly.
        """

        values = {
            "head_min_per_node": torch.zeros(self.n) if head_min is None else torch.as_tensor(head_min),
            "head_range_per_node": torch.ones(self.n) if head_range is None else torch.as_tensor(head_range),
            "invert_elev_per_node": torch.zeros(self.n) if invert_elev is None else torch.as_tensor(invert_elev),
            "max_depth_per_node": torch.zeros(self.n) if max_depth is None else torch.as_tensor(max_depth),
            "outfall_mask_per_node": torch.zeros(self.n) if outfall_mask is None else torch.as_tensor(outfall_mask),
        }
        for name, value in values.items():
            value = value.detach().to(dtype=torch.float32).reshape(-1)
            if value.numel() != self.n:
                raise ValueError(f"{name} length mismatch: expected {self.n}, got {value.numel()}")
            self.register_buffer(name, value)

    def _validate_physics_statistics(self) -> None:
        """Reject metadata that would make the head de-normalisation degenerate."""

        for name in _PHYSICS_METADATA_ARGUMENTS:
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must contain only finite values.")
        if (self.head_range_per_node <= 0).any():
            raise ValueError(
                "head_range_per_node must be positive: it is the min-max range of the "
                "training head and is used to de-normalise the predicted head."
            )
        if (self.max_depth_per_node < 0).any():
            raise ValueError(
                "max_depth_per_node must be non-negative: it divides the head increment in "
                "physics_features(). Outfall nodes carry MaxDepth = 0, where the descriptor "
                "clamps the divisor to eps, so zero must stay valid."
            )

    def _prepare_supports(self, static_supports: Sequence[Tensor]) -> list[Tensor]:
        if len(static_supports) != self.n_static_supports:
            raise ValueError(
                f"Expected {self.n_static_supports} static supports "
                f"(topology and spatial proximity for the full view), got {len(static_supports)}"
            )
        reference = self.input_proj.weight
        supports = []
        for index, support in enumerate(static_supports):
            if not isinstance(support, Tensor):
                raise TypeError(f"Static support {index} must be a torch.Tensor.")
            if support.ndim != 2 or tuple(support.shape) != (self.n, self.n):
                raise ValueError(
                    f"Static support {index} must have shape ({self.n}, {self.n}), "
                    f"got {tuple(support.shape)}"
                )
            support = support.to(device=reference.device, dtype=reference.dtype)
            if not torch.isfinite(support).all():
                raise ValueError(f"Static support {index} contains non-finite values.")
            supports.append(support)
        if self.use_adp and self.adp is not None:
            supports.append(self.adp())
        return supports

    def _run_backbone(self, x: Tensor, supports: Sequence[Tensor]) -> Tensor:
        hidden = self.input_proj(x)
        skip_total = None
        for block in self.blocks:
            hidden, skip = block(hidden, supports)
            # Defensive guard for ill-conditioned supports: one non-finite value
            # would otherwise turn the whole batch into NaN. The 1.0 stand-in is
            # an arbitrary finite sentinel, not a calibrated activation bound.
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
            skip = torch.nan_to_num(skip, nan=0.0, posinf=1.0, neginf=-1.0)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("No residual blocks produced skip connections.")
        return torch.nan_to_num(self.out_shared(skip_total), nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_physics_context(self, shared: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build the head prediction, the overflow-branch input and the exceedance.

        The physics descriptors are derived from the *predicted* head
        (Section 3.5), not from observations. They see the predicted normalised
        head clipped to ``[0, 1]``, so the descriptors stay inside the range
        covered by the training statistics. ``head_exceedance`` is computed from
        the unclipped head, so it reports the prediction in metres as made.

        The returned ``combined`` tensor -- backbone features concatenated with
        the projected descriptors -- is the input of *both* overflow branches,
        so the occurrence probability also depends on the head descriptors.
        """

        norm_head = torch.nan_to_num(self.head_predictor(shared), nan=0.0, posinf=1.0, neginf=-1.0)
        head_range = self.head_range_per_node.view(1, 1, self.n, 1)
        head_min = self.head_min_per_node.view(1, 1, self.n, 1)
        invert_elev = self.invert_elev_per_node.view(1, 1, self.n, 1)
        max_depth = self.max_depth_per_node.view(1, 1, self.n, 1)
        head_exceedance = norm_head * head_range + head_min - (invert_elev + max_depth)
        if self.use_phys_emb and self.physics_embedder is not None:
            # The descriptors are inputs to the overflow branch, not supervision:
            # detaching keeps the overflow loss from training the head predictor
            # through this hand-designed, non-invertible transform.
            phys_head = torch.clamp(norm_head, 0.0, 1.0) * head_range + head_min
            physics = physics_features(phys_head.detach(), invert_elev, max_depth)
            combined = torch.cat((shared, self.physics_embedder(physics)), dim=1)
        else:
            combined = shared
        return norm_head, combined, head_exceedance

    def _build_overflow_outputs(
        self,
        combined: Tensor,
    ) -> tuple[Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Return ``(prob_flood, overflow_mag, overflow_with_prob, overflow_mask, overflow_hard)``.

        All entries except ``overflow_mag`` are ``None`` when the model was
        built with ``use_hurdle=False``.
        """

        if self.use_hurdle and self.gate_branch is not None:
            prob_flood = torch.nan_to_num(self.gate_branch(combined), nan=0.0, posinf=1.0, neginf=0.0)
        else:
            prob_flood = None
        overflow_mag = torch.nan_to_num(self.overflow_branch(combined), nan=0.0, posinf=1.0, neginf=0.0)
        # Outfalls discharge out of the network and never overflow; the training
        # code zeroed their magnitude before the gate was applied, so the same
        # mask must be applied here or those eight nodes invent overflow.
        outfall_keep = (self.outfall_mask_per_node != 1).float().view(1, 1, self.n, 1)
        overflow_mag = overflow_mag * outfall_keep
        if prob_flood is not None:
            # Eq. (14) as printed is the continuous product. The reported
            # overflow metrics were computed on the thresholded output, which
            # is returned separately as ``overflow_hard``. Neither enters the
            # objective: L_overflow regresses Q_raw directly.
            overflow_with_prob = prob_flood * overflow_mag
            overflow_mask = (prob_flood >= 0.5).float()
            overflow_hard = overflow_mask * overflow_mag
        else:
            overflow_with_prob = None
            overflow_mask = None
            overflow_hard = None
        return prob_flood, overflow_mag, overflow_with_prob, overflow_mask, overflow_hard

    def forward(self, x: Tensor, static_supports: Sequence[Tensor]) -> dict[str, Optional[Tensor]]:
        """Run one forward pass; see the class docstring for the output contract."""

        if not isinstance(x, Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point torch.Tensor.")
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, N, T], got {tuple(x.shape)}")
        if x.dtype != self.input_proj.weight.dtype:
            raise ValueError(
                f"Expected x of dtype {self.input_proj.weight.dtype}, got {x.dtype}."
            )
        expected_shape = (self.in_channels, self.n, self.window_steps)
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected x trailing shape {expected_shape}, "
                f"got {tuple(x.shape[1:])}"
            )
        supports = self._prepare_supports(static_supports)
        shared = self._run_backbone(x, supports)
        norm_head, combined, head_exceedance = self._build_physics_context(shared)
        prob_flood, overflow_mag, overflow_with_prob, overflow_mask, overflow_hard = (
            self._build_overflow_outputs(combined)
        )
        return {
            "pred_head": norm_head[..., -1],
            "prob_flood": prob_flood[..., -1] if prob_flood is not None else None,
            "overflow_mag": overflow_mag[..., -1],
            "overflow_with_prob": overflow_with_prob[..., -1] if overflow_with_prob is not None else None,
            "overflow_mask": overflow_mask[..., -1] if overflow_mask is not None else None,
            "overflow_hard": overflow_hard[..., -1] if overflow_hard is not None else None,
            "head_exceedance": head_exceedance[..., -1],
            "shared": shared,
        }


def build_final_kigwn(
    config_path: str | Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> KIGWNFullView:
    """Build KIGWN from the shipped config, an explicit file, or a mapping.

    Args:
        config_path: Path to a JSON config. The packaged ``config.json`` is used
            when both ``config_path`` and ``config`` are omitted.
        config: Config mapping, either the full document or its ``architecture``
            section only. Mutually exclusive with ``config_path``.
        **overrides: Constructor arguments applied on top of the config. The
            per-node physics metadata (``head_min_per_node``,
            ``head_range_per_node``, ``invert_elev_per_node``,
            ``max_depth_per_node``) is mandatory while
            ``use_physics_embedding`` is enabled.

    Raises:
        ValueError: for an unsupported or inconsistent config, an unknown
            argument, or missing physics metadata.
    """

    if config_path is not None and config is not None:
        raise ValueError("Pass either config_path or config, not both.")
    values = load_model_config(config_path) if config is None else normalize_model_config(config)
    values.update(overrides)
    unknown = sorted(set(values) - _MODEL_ARGUMENTS)
    if unknown:
        raise ValueError(f"Unknown KIGWN arguments: {', '.join(unknown)}")
    architecture = {name: values[name] for name in _ARCHITECTURE_ARGUMENTS}
    values.update(normalize_model_config(architecture))
    return KIGWNFullView(**values)


__all__ = ["KIGWNFullView", "build_final_kigwn"]
