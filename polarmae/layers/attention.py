from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "prepare_attn_masks"]

class Attention(nn.Module):
    """Attention with optional cross-attention"""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        # deprecated
        use_flash_self_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        # We will compute Q, K, V differently for cross-attention
        # so let's just keep a single linear projection for Q, K, V each.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def q_proj(self, x):
        return F.linear(x, self.qkv.weight[:self.dim], self.qkv.bias[:self.dim])

    def k_proj(self, x):
        return F.linear(x, self.qkv.weight[self.dim:2*self.dim], self.qkv.bias[self.dim:2*self.dim])

    def v_proj(self, x):
        return F.linear(x, self.qkv.weight[2*self.dim:], self.qkv.bias[2*self.dim:])
    
    def self_attention(self, x, x_attn_mask=None):
        # Reshape Q, K, V
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (B, H, 3, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D), (B, H, N, D), (B, H, N, D)

        # Prepare attn_mask
        # if x_attn_mask is not None:
        #     # Adjust attn_mask dimensions to (B * H, N, M)
        #     attn_mask = x_attn_mask.squeeze(1).repeat_interleave(
        #         self.num_heads, dim=0
        #     )

        # Reshape for scaled_dot_product_attention
        # q = q.reshape(B * self.num_heads, N, self.head_dim)
        # k = k.reshape(B * self.num_heads, N, self.head_dim)
        # v = v.reshape(B * self.num_heads, N, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=x_attn_mask, dropout_p=self.attn_drop.p
        )

        # Reshape back
        attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim)
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        _attn = None

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, _attn
    
    # def self_plus_cross_attention(self, x, y, x_attn_mask=None, y_attn_mask=None, rpb=None):
    #     B, N, C = x.shape
    #     L = y.shape[1]

    #     # run both x & y thru qkv. could be quicker but
    #     x = torch.cat([x, y], dim=1)
    #     qkv = self.qkv(x).reshape(B, N+L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]


    #     """ ------------------------------------------------------------------------ """
    #     """                      Cross attention of y on x and y                     """
    #     """ ------------------------------------------------------------------------ """
    #     attn = (q[:, :, N:]) @ k[:, :, :].transpose(-2, -1) * self.scale
    #     if x_attn_mask is not None:
    #         attn = attn + torch.cat([x_attn_mask, y_attn_mask], dim=-1)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     y = (attn @ v).transpose(1, 2).reshape(B, L, C)
    #     y = self.proj(y)
    #     y = self.proj_drop(y)

    #     """ ------------------------------------------------------------------------ """
    #     """                         Self attention of x on x                         """
    #     """ ------------------------------------------------------------------------ """
    #     attn = (q[:, :, :N]) @ k[:, :, :N].transpose(-2, -1) * self.scale
    #     if rpb is not None:
    #         attn = attn + rpb
    #     if x_attn_mask is not None:
    #         attn = attn + x_attn_mask
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v[:, :, :N]).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     _attn = None
    #     return x, y, _attn


    def cross_attention(self, q, kv, attn_mask=None):
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        B,Nq,C = q.shape
        B,Nv,C = v.shape
        q = q.reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p
        )
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        _attn = None # no intermediates srry
        return x, _attn


    def forward(self, q, qkv_attn_mask=None, kv=None):
        """
        Args:
            q (torch.Tensor): Queries of shape (B, N, C).
            y (torch.Tensor, optional): If provided, should be of shape (B, M, C) and used for K, V.
                                         Otherwise, K, V come from x (self-attention).
            x_attn_mask (torch.Tensor, optional): Attention mask for x.
            y_attn_mask (torch.Tensor, optional): Attention mask for y.
            rpb (torch.Tensor, optional): Relative position bias.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensors after attention.
        """
        if kv is None: # Regular self-attention
            x, _attn = self.self_attention(q, qkv_attn_mask)
        else: # Self attention on q, cross attention of q&kv on kv
            x, _attn = self.cross_attention(q, kv, qkv_attn_mask)
        return x, _attn


def _create_attn_mask(q_mask: torch.Tensor, kv_mask: torch.Tensor=None) -> torch.Tensor:
    """
    Create an additive attention mask from a binary mask.

    The input mask is a binary tensor where 1s indicate positions to attend to, and 0s
    indicate positions to ignore. The output is an attention mask where positions to
    ignore are set to a large negative value, and positions to attend to are set to 0.

    Args:
        mask (torch.Tensor): A binary mask tensor of shape (B, N), where B is the batch size
                                and N is the sequence length. 1s indicate positions to attend to,
                                and 0s indicate positions to ignore.
        dtype (torch.dtype): The desired data type of the attention mask.

    Returns:
        torch.Tensor: An attention mask tensor of shape (B, 1, N, N), where positions to ignore
                        are set to a large negative value (-1e9), making them effectively ignored
                        during attention computation.
    """
    if kv_mask is None:
        kv_mask = q_mask

    attn_mask = (
        q_mask.unsqueeze(1).unsqueeze(3) & kv_mask.unsqueeze(1).unsqueeze(2)
    )
    return attn_mask
    
def prepare_attn_mask(
        q: torch.Tensor,
        q_mask: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    B, Nq, C = q.shape

    # make sure masks are of the correct shape
    if q_mask is not None:
        assert q_mask.shape == (B, Nq), "q_mask must be of shape (B, Nq)"
    if kv_mask is not None:
        Bv, Nv, Cv = kv.shape
        assert kv_mask is not None, "kv_mask is required for cross-attention"
        assert B == Bv, "Batch size must match between q and key_value"
        assert kv_mask.shape == (B, Nv), "kv_mask must be (B, Nv)"

    return _create_attn_mask(q_mask, kv_mask)