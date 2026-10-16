"""Gated temporal and graph-diffusion block for KIGWN."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .graph_components import DiffusionConv
except ImportError:  # Direct import when this directory is on sys.path.
    from graph_components import DiffusionConv


class GWBlock(nn.Module):
    """One gated temporal convolution plus multi-support graph diffusion."""

    def __init__(
        self,
        c_in: int,
        c_res: int,
        c_skip: int,
        k: int,
        n_supports: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.tconv = nn.Conv2d(c_in, 2 * c_res, kernel_size=(1, 3), dilation=(1, dilation))
        self.dconv = DiffusionConv(c_res, c_res, k, n_supports)
        self.res_proj = nn.Conv2d(c_res, c_in, kernel_size=1)
        self.skip_proj = nn.Conv2d(c_res, c_skip, kernel_size=1)

    def forward(self, x: Tensor, supports: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        padding = (3 - 1) * self.dilation
        temporal = self.tconv(F.pad(x, (padding, 0, 0, 0)))
        tanh_part, gate_part = torch.chunk(temporal, 2, dim=1)
        hidden = torch.tanh(tanh_part) * torch.sigmoid(gate_part)
        hidden = self.dconv(hidden, supports)
        residual = torch.nan_to_num(self.res_proj(hidden), nan=0.0, posinf=1.0, neginf=-1.0)
        skip = torch.nan_to_num(self.skip_proj(hidden), nan=0.0, posinf=1.0, neginf=-1.0)
        return x + residual, skip


__all__ = ["GWBlock"]
