from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from polarmae.layers.attention import prepare_attn_mask
from polarmae.layers.block import Block
from polarmae.layers.masking import MaskedDropPath, MaskedLayerNorm, MaskedRMSNorm
from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

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
        use_flash_attn=True,
        norm_layer=MaskedLayerNorm,
        prefix_tuning=False,
        prefix_dim=128,
        prefix_hidden_dim=128,
        prefix_num_tokens=0,
        # deprecated
        use_flash_self_attn=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if isinstance(norm_layer, str):
            norm_layer = globals()[norm_layer]

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
                    use_flash_attn=use_flash_attn,
                    use_kv=use_kv,
                    norm_layer=norm_layer,
                    # deprecated
                    use_flash_self_attn=use_flash_self_attn,
                )
                for i in range(depth)
            ]
        )

        # output norm
        self.norm = norm_layer(embed_dim) if postnorm else Identity()
        log.info(f'postnorm: {postnorm}')
        log.info(f'norm_layer: {norm_layer}')
        self.add_pos_at_every_layer = add_pos_at_every_layer

        # prefix-tuning
        self.prefix_tuning = prefix_tuning
        self.prefix_num_tokens = prefix_num_tokens
        if prefix_tuning:
            log.info("🔥  Initializing prefix tokens.")
            # Initialize a small parameter matrix (1, N, k)
            self.p_theta_prime = nn.Parameter(torch.zeros(1, prefix_num_tokens, prefix_dim))
            nn.init.trunc_normal_(self.p_theta_prime, std=0.02)

            # MLP to transform from (1, N, k) to (1, N, L*2*C)
            # Where L is the number of layers, 2 is for K and V, and C is embed_dim
            self.prefix_mlp = nn.Sequential(
                nn.Linear(prefix_dim, prefix_hidden_dim),
                nn.GELU(),
                nn.Linear(prefix_hidden_dim, prefix_hidden_dim),
                nn.GELU(),
                nn.Linear(prefix_hidden_dim, depth * embed_dim * 2),
            )

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

    def get_prefix_tokens(self):
        """
        Process prefix tokens through MLP and reshape for use in attention.
        
        Returns:
            Tensor of shape (depth, 2, B, prefix_num_tokens, embed_dim) where the second
            dimension represents K and V prefix tokens for each layer
        """
        if not self.prefix_tuning or self.prefix_num_tokens == 0:
            return None
            
        # Run through MLP
        batch_size = 1  # We use the same prefix for all batch items
        prefix_tokens = self.p_theta_prime  # (1, N, prefix_dim)
        
        # Transform to (1, N, L*2*C)
        prefix_tokens = self.prefix_mlp(prefix_tokens)
        
        # Reshape to (depth, 2, 1, prefix_num_tokens, embed_dim)
        prefix_tokens = prefix_tokens.view(
            batch_size, 
            self.prefix_num_tokens, 
            len(self.blocks), 
            2 * self.embed_dim
        ).permute(2, 0, 1, 3)
        
        # Reshape to split K and V dimensions
        prefix_tokens = prefix_tokens.reshape(
            len(self.blocks),    # depth
            batch_size,          # B
            self.prefix_num_tokens, 
            2,                   # K and V
            self.embed_dim
        ).permute(0, 3, 1, 2, 4)  # (depth, 2, B, prefix_num_tokens, embed_dim)
        
        return prefix_tokens

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
        final_norm: bool = True,
    ) -> TransformerOutput:
        """
        If memory is provided, the blocks will perform cross-attention:
        Q from x, K/V from cat[x,y].
        If memory is None, self-attention is performed.
        """
        # Get prefix tokens if using prefix tuning (for self-attention only)
        prefix_tokens = None
        if self.prefix_tuning and kv is None:  # Only for self-attention
            prefix_tokens = self.get_prefix_tokens()
            
        # Calculate padding for attention masks if using prefix tuning
        pad = self.prefix_num_tokens if self.prefix_tuning and kv is None else 0
        qkv_attn_mask = prepare_attn_mask(q, q_mask, kv, kv_mask, pad=pad)

        hidden_states = [] if return_hidden_states else None
        attentions = [] if return_attentions else None
        ffns = [] if return_ffns else None

        kv = kv + pos_kv if kv is not None else kv
        if not self.add_pos_at_every_layer:
            q = q + pos_q

        for i, block in enumerate(self.blocks):
            if self.add_pos_at_every_layer:
                q = q + pos_q
                
            # Pass appropriate prefix tokens for this layer if doing self-attention with prefix tuning
            current_prefix = None
            if prefix_tokens is not None:
                current_prefix = prefix_tokens[i]  # Get the prefix tokens for this layer
                
            q, attn = block(
                q,
                q_mask,
                qkv_attn_mask,
                kv,
                kv_mask,
                prefix_k_v=current_prefix,
            )
            if return_hidden_states:
                assert hidden_states is not None
                hidden_states.append(q)
            if return_attentions:
                assert attentions is not None
                attentions.append(attn)

        if final_norm:
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