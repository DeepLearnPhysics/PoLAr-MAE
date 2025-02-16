import torch
import torch.nn as nn
from polarmae.layers.attention import Attention
from polarmae.layers.masking import MaskedDropPath, MaskedLayerNorm
from polarmae.layers.mlp import Mlp


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=MaskedLayerNorm,
        use_kv=False,
        # deprecated
        use_flash_self_attn=False,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = MaskedDropPath(drop_path) if drop_path > 0.0 else Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.norm1_kv = norm_layer(dim) if use_kv else None
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # MLP BLOCK
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        q,
        q_mask,
        qkv_attn_mask,
        kv=None,
        kv_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) where B is the batch size, N is the sequence length, and C is the feature dimension.
            x_attn_mask (torch.Tensor, optional): Attention mask for the input tensor x.
            x_mask (torch.Tensor, optional): Mask for the input tensor x.
            y (torch.Tensor, optional): Optional second input tensor of shape (B, M, C) where M is the sequence length for the second input.
            y_attn_mask (torch.Tensor, optional): Attention mask for the input tensor y.
            y_mask (torch.Tensor, optional): Mask for the input tensor y.
            rpb (torch.Tensor, optional): Relative position bias tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Output tensor after processing x.
                - y (torch.Tensor): Output tensor after processing y (if y is provided, otherwise None).
                - attn (torch.Tensor): Attention weights.
        """

        # if kv is None:
        #     _q, _, attn = self.attn(self.norm1(q, q_mask), q_attn_mask, rpb=rpb)
        #     q = q + self.drop_path(_q, q_mask)
        #     ffn = self.mlp(self.norm2(q, q_mask))
        #     if q_mask is not None:
        #         ffn = ffn * q_mask.unsqueeze(-1)
        #     q = q + self.drop_path(ffn, q_mask)
        #     return q, attn
        if self.norm1_kv is not None:
            assert kv is not None, "kv must be provided if use_kv is True"

        _q, attn = self.attn(
            q=self.norm1(q, q_mask), 
            qkv_attn_mask=qkv_attn_mask, 
            kv=self.norm1_kv(kv, kv_mask) if kv is not None else None, 
        )
        q = q + self.drop_path(_q, q_mask)

        ffn_q = self.mlp(self.norm2(q, q_mask))

        if q_mask is not None:
            ffn_q = ffn_q * q_mask.unsqueeze(-1)

        q = q + self.drop_path(ffn_q, q_mask)

        return q, attn