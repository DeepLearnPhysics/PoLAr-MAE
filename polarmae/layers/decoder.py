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
            header: nn.Module | None = None,
    ):
        super().__init__()
        self.transformer = make_transformer(
            arch_name=arch,
            **transformer_kwargs,
        )
        self.header = header # not used currently

    def forward(
        self,
        q: torch.Tensor,
        pos_q: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,          # X-attn for e.g. PCP-MAE
        pos_kv: torch.Tensor | None = None,      # X-attn for e.g. PCP-MAE
        kv_mask: torch.Tensor | None = None,     # X-attn for e.g. PCP-MAE
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ) -> TransformerOutput:
        """Call transformer forward"""
        return self.transformer.forward(
            q, pos_q, q_mask, kv, pos_kv, kv_mask, return_hidden_states, return_attentions, return_ffns
        )