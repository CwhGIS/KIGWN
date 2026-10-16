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


class KIGWNFullView(nn.Module):
    """Full-view KIGWN forward model.

    Input:
        x: [batch, channels, nodes, window].
        static_supports: [topology_support, rbf_support], each [nodes, nodes].
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

        with torch.no_grad():
            n_physics_features = self._physics_features(
                torch.zeros(1, 1, n_nodes, 1),
                torch.zeros(1, 1, n_nodes, 1),
                torch.ones(1, 1, n_nodes, 1),
            ).shape[1]
        self.physics_embedder = build_physics_embedder(n_physics_features, physics_embed_dim)

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

    def _register_node_buffers(
        self,
        head_min: Optional[Sequence[float]],
        head_range: Optional[Sequence[float]],
        invert_elev: Optional[Sequence[float]],
        max_depth: Optional[Sequence[float]],
        outfall_mask: Optional[Sequence[float]],
    ) -> None:
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

    @staticmethod
    def _physics_features(head_phys: Tensor, invert_elev: Tensor, max_depth: Tensor) -> Tensor:
        return physics_features(head_phys, invert_elev, max_depth)

    def _prepare_supports(self, static_supports: Sequence[Tensor]) -> list[Tensor]:
        if len(static_supports) != self.n_static_supports:
            raise ValueError(
                f"Expected {self.n_static_supports} static supports "
                f"(topology and RBF for full-view), got {len(static_supports)}"
            )
        supports = list(static_supports)
        for index, support in enumerate(supports):
            if support.ndim != 2 or tuple(support.shape) != (self.n, self.n):
                raise ValueError(
                    f"Static support {index} must have shape ({self.n}, {self.n}), "
                    f"got {tuple(support.shape)}"
                )
        if self.use_adp and self.adp is not None:
            supports.append(self.adp())
        return supports

    def _run_backbone(self, x: Tensor, supports: Sequence[Tensor]) -> Tensor:
        hidden = self.input_proj(x)
        skip_total = None
        for block in self.blocks:
            hidden, skip = block(hidden, supports)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
            skip = torch.nan_to_num(skip, nan=0.0, posinf=1.0, neginf=-1.0)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("No residual blocks produced skip connections.")
        return torch.nan_to_num(self.out_shared(skip_total), nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_physics_context(self, shared: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        norm_head = torch.nan_to_num(self.head_predictor(shared), nan=0.0, posinf=1.0, neginf=-1.0)
        norm_head_for_phys = torch.clamp(
            torch.nan_to_num(norm_head, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        head_range = self.head_range_per_node.view(1, 1, self.n, 1)
        head_min = self.head_min_per_node.view(1, 1, self.n, 1)
        invert_elev = self.invert_elev_per_node.view(1, 1, self.n, 1)
        max_depth = self.max_depth_per_node.view(1, 1, self.n, 1)
        phys_head = norm_head_for_phys * head_range + head_min
        head_exceedance = phys_head - (invert_elev + max_depth)
        if self.use_phys_emb:
            physics = self._physics_features(phys_head.detach(), invert_elev, max_depth)
            combined = torch.cat((shared, self.physics_embedder(physics)), dim=1)
        else:
            combined = shared
        return norm_head, combined, head_exceedance

    def _build_overflow_outputs(
        self,
        combined: Tensor,
    ) -> tuple[Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if self.use_hurdle and self.gate_branch is not None:
            prob_flood = torch.nan_to_num(self.gate_branch(combined), nan=0.0, posinf=1.0, neginf=0.0)
        else:
            prob_flood = None
        overflow_mag = torch.nan_to_num(self.overflow_branch(combined), nan=0.0, posinf=1.0, neginf=0.0)
        if prob_flood is not None:
            # Keep the continuous probability so that magnitude training stays
            # differentiable. The binary occurrence decision is reported
            # separately.
            overflow_with_prob = prob_flood * overflow_mag
            overflow_mask = (prob_flood >= 0.5).float()
            overflow_hard = (prob_flood >= 0.5).float() * overflow_mag
        else:
            overflow_with_prob = None
            overflow_mask = None
            overflow_hard = None
        return prob_flood, overflow_mag, overflow_with_prob, overflow_mask, overflow_hard

    def forward(self, x: Tensor, static_supports: Sequence[Tensor]) -> dict[str, Optional[Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, N, T], got {tuple(x.shape)}")
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
    """Build KIGWN from the shipped config or an explicit config file."""

    if config_path is not None and config is not None:
        raise ValueError("Pass either config_path or config, not both.")
    values = load_model_config(config_path) if config is None else normalize_model_config(config)
    values.update(overrides)
    architecture = {key: values[key] for key in values if key in values and key in {
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
    }}
    values.update(normalize_model_config(architecture))
    return KIGWNFullView(**values)


__all__ = ["KIGWNFullView", "build_final_kigwn"]
