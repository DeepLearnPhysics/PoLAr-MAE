from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from polarmae.layers.attention import prepare_attn_mask
from polarmae.layers.block import Block
from polarmae.layers.masking import MaskedDropPath, MaskedLayerNorm

__all__ = [
    'Transformer',
    'make_transformer',
    'vit_tiny',
    'vit_small',
    'vit_base',
]

class Identity(nn.Module):
    def forward(self, x, mask=None): return x

@dataclass
class TransformerOutput:
    last_hidden_state: torch.Tensor  # (B, T, C)
    hidden_states: Optional[List[torch.Tensor]] = None  # [(B, T, C)]
    attentions: Optional[List[torch.Tensor]] = None  # [(B, H, T)]
    ffns: Optional[List[torch.Tensor]] = None  # [(B, T, C)]

class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate: float | List[float] = 0.0,
        drop_path_uniform: bool = False,
        add_pos_at_every_layer=False,
        postnorm=True,
        use_kv=False,
        # deprecated
        use_flash_self_attn=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    use_kv=use_kv,
                )
                for i in range(depth)
            ]
        )

        # output norm
        self.norm = MaskedLayerNorm(embed_dim) if postnorm else Identity()

        self.add_pos_at_every_layer = add_pos_at_every_layer

        self.apply(self._init_weights)

    def _zero_drop_path(self):
        for module in self.modules():
            if isinstance(module, MaskedDropPath):
                module.drop_prob = 0.0
            elif isinstance(module, nn.Dropout):
                module.p = 0.0

    def _init_weights(self, m):
        ''' ViT weight initialization '''
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        q: torch.Tensor,
        pos_q: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,          # X-attn
        pos_kv: torch.Tensor | None = None,      # X-attn
        kv_mask: torch.Tensor | None = None,     # X-attn
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ) -> TransformerOutput:
        """
        If memory is provided, the blocks will perform cross-attention:
        Q from x, K/V from cat[x,y].
        If memory is None, self-attention is performed.
        """
        qkv_attn_mask = prepare_attn_mask(q, q_mask, kv, kv_mask)

        hidden_states = [] if return_hidden_states else None
        attentions = [] if return_attentions else None
        ffns = [] if return_ffns else None

        kv = kv + pos_kv if kv is not None else kv
        if not self.add_pos_at_every_layer:
            q = q + pos_q

        for block in self.blocks:
            if self.add_pos_at_every_layer:
                q = q + pos_q
            q, attn = block(
                q,
                q_mask,
                qkv_attn_mask,
                kv,
                kv_mask,
            )
            if return_hidden_states:
                assert hidden_states is not None
                hidden_states.append(q)
            if return_attentions:
                assert attentions is not None
                attentions.append(attn)

        q = self.norm(q, q_mask)

        return TransformerOutput(q, hidden_states, attentions, ffns)

def make_transformer(
    arch_name: Literal['vit_tiny', 'vit_small', 'vit_base'],
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float | List[float] = 0.0,
    drop_path_uniform: bool = False,
    add_pos_at_every_layer: bool = False,
    postnorm: bool = True,
    use_kv: bool = False,
    prompt_tuning: bool = False,
    **kwargs,
) -> Transformer:
    name = arch_name + ("_prompted" if prompt_tuning else "")
    transformer_config = dict(
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        drop_path_uniform=drop_path_uniform,
        add_pos_at_every_layer=add_pos_at_every_layer,
        postnorm=postnorm,
        use_kv=use_kv,
    )
    transformer_config.update(kwargs)
    return globals()[name](**transformer_config)

def vit_tiny(**kwargs) -> Transformer:
    transformer_config = dict(
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
    )
    transformer_config.update(kwargs)
    return Transformer(
        **transformer_config,
    )

def vit_small(**kwargs) -> Transformer:
    transformer_config = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
    )
    transformer_config.update(kwargs)
    return Transformer(
        **transformer_config,
    )

def vit_base(**kwargs) -> Transformer:
    transformer_config = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
    )
    transformer_config.update(kwargs)
    return Transformer(
        **transformer_config,
    )