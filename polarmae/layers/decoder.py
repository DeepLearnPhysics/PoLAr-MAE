from typing import Literal

import torch
import torch.nn as nn
from polarmae.layers.transformer import TransformerOutput, make_transformer

__all__ = ['TransformerDecoder']

class TransformerDecoder(nn.Module):
    """Just a wrapper around the transformer."""
    def __init__(
            self,
            arch: Literal['vit_tiny', 'vit_small', 'vit_base'] = 'vit_small',
            transformer_kwargs: dict = {},
    ):
        super().__init__()
        self.transformer = make_transformer(
            arch_name=arch,
            **transformer_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        y: torch.Tensor | None = None,          # X-attn for e.g. PCP-MAE
        pos_y: torch.Tensor | None = None,      # X-attn for e.g. PCP-MAE
        y_mask: torch.Tensor | None = None,     # X-attn for e.g. PCP-MAE
        rpb: torch.Tensor | None = None,       # X-attn for e.g. PCP-MAE
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ) -> TransformerOutput:
        """Call transformer forward"""
        return self.transformer.forward(
            x, pos_x, x_mask, y, pos_y, y_mask, rpb, return_hidden_states, return_attentions, return_ffns
        )