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
        use_flash_self_attn=False,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = MaskedDropPath(drop_path) if drop_path > 0.0 else Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_self_attn=use_flash_self_attn,
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
        x,
        x_attn_mask=None,
        x_mask=None,
        y=None,
        y_attn_mask=None,
        y_mask=None,
        rpb=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if y is None:
            _x, _, attn = self.attn(self.norm1(x, x_mask), x_attn_mask, rpb=rpb)
            x = x + self.drop_path(_x, x_mask)
            ffn = self.mlp(self.norm2(x, x_mask))
            if x_mask is not None:
                ffn = ffn * x_mask.unsqueeze(-1)
            x = x + self.drop_path(ffn, x_mask)
            y = None
            return x, y, attn

        _x, _y, attn = self.attn(
            x=self.norm1(x, x_mask), 
            x_attn_mask=x_attn_mask, 
            y=self.norm1(y, y_mask), 
            y_attn_mask=y_attn_mask,
            rpb=rpb,
        )
        x = x + self.drop_path(_x, x_mask)
        y = y + self.drop_path(_y, y_mask)

        ffn_x = self.mlp(self.norm2(x, x_mask))
        ffn_y = self.mlp(self.norm2(y, y_mask))

        if x_mask is not None:
            ffn_x = ffn_x * x_mask.unsqueeze(-1)
        if y_mask is not None:
            ffn_y = ffn_y * y_mask.unsqueeze(-1)

        x = x + self.drop_path(ffn_x, x_mask)
        y = y + self.drop_path(ffn_y, y_mask)

        return x, y, attn