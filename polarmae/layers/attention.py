from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "prepare_attn_masks"]

class Attention(nn.Module):
    """Attention with optional cross-attention and support for cu_seqlens"""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=True,
        # deprecated
        use_flash_self_attn=True,
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

        self.use_flash_attn = use_flash_attn

    def q_proj(self, x):
        return F.linear(x, self.qkv.weight[:self.dim], self.qkv.bias[:self.dim] if self.qkv.bias is not None else None)

    def k_proj(self, x):
        return F.linear(x, self.qkv.weight[self.dim:2*self.dim], self.qkv.bias[self.dim:2*self.dim] if self.qkv.bias is not None else None)

    def v_proj(self, x):
        return F.linear(x, self.qkv.weight[2*self.dim:], self.qkv.bias[2*self.dim:] if self.qkv.bias is not None else None)
    
    def self_attention(self, x, x_attn_mask=None, rpb=None, prefix_k_v=None):
        # Reshape Q, K, V
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D), (B, H, N, D), (B, H, N, D)
        
        # Handle prefix tokens for K and V if provided
        if prefix_k_v is not None:
            # prefix_k_v has shape (2, B', prefix_num_tokens, embed_dim)
            # where the first dimension is [k, v]
            prefix_k, prefix_v = prefix_k_v[0], prefix_k_v[1]  # (B', prefix_N, C)
            prefix_num_tokens = prefix_k.shape[1]
            
            # Expand batch dimension if needed
            if prefix_k.shape[0] == 1 and B > 1:
                prefix_k = prefix_k.expand(B, -1, -1)
                prefix_v = prefix_v.expand(B, -1, -1)
            
            # Reshape to match attention head format
            prefix_k = prefix_k.reshape(B, prefix_num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            prefix_v = prefix_v.reshape(B, prefix_num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Concatenate prefix tokens to k and v
            k = torch.cat([prefix_k, k], dim=2)  # (B, H, prefix_N + N, D)
            v = torch.cat([prefix_v, v], dim=2)  # (B, H, prefix_N + N, D)

        # Check if the batch size or sequence length is too large
        if self.use_flash_attn:# and B * N < 200000:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=x_attn_mask, dropout_p=self.attn_drop.p
            )

            # Reshape back
            attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim)
            x = attn_output.transpose(1, 2).reshape(B, N, C)
            _attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale 
            if rpb is not None:
                attn = attn + rpb
            if x_attn_mask is not None:
                attn = attn + (~x_attn_mask).to(attn.dtype).masked_fill(~x_attn_mask, -1e9)
            attn = attn.softmax(dim=-1)
            _attn = attn
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, _attn
    
    def cross_attention(self, q, kv, attn_mask=None):
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        B,Nq,C = q.shape
        B,Nv,C = v.shape
        q = q.reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash_attn and B < 200000: # crashes on huge batch sizes
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p
            )
        else: # no flash attn :-(
            qk = torch.einsum("b h n d, b h m d -> b h n m", q, k)
            qk = qk * self.scale
            if attn_mask is not None:
                qk = qk + attn_mask
            qk = qk.softmax(dim=-1)
            qk = self.attn_drop(qk)
            x = torch.einsum("b h n m, b h m d -> b h n d", qk, v)


        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        _attn = None # no intermediates srry
        return x, _attn


    def forward(self, q, qkv_attn_mask=None, kv=None, prefix_k_v=None):
        """
        Args:
            q (torch.Tensor): Queries of shape (B, N, C) or (sum(seqlens), C) if using cu_seqlens.
            qkv_attn_mask (torch.Tensor, optional): Attention mask.
            kv (torch.Tensor, optional): If provided, should be for K, V. Otherwise, K, V come from x (self-attention).
            prefix_k_v (tuple, optional): Tuple of (prefix_k, prefix_v) tensors to append to K and V for prefix tuning.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors after attention.
        """
        if kv is None:  # Regular self-attention
            x, _attn = self.self_attention(q, qkv_attn_mask, prefix_k_v=prefix_k_v)
        else:  # Cross attention
            x, _attn = self.cross_attention(q, kv, qkv_attn_mask)
                
        return x, _attn


def _create_attn_mask(q_mask: torch.Tensor, kv_mask: torch.Tensor | None = None, pad: int = 0) -> torch.Tensor:
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
        pad (int): The number of padding tokens to prepend to the key/value mask.

    Returns:
        torch.Tensor: An attention mask tensor of shape (B, 1, N, N), where positions to ignore
                        are set to a large negative value (-1e9), making them effectively ignored
                        during attention computation.
    """
    if kv_mask is None:
        kv_mask = q_mask
    if pad > 0:
        kv_mask = torch.cat([torch.ones(kv_mask.shape[0], pad, dtype=kv_mask.dtype, device=kv_mask.device), kv_mask], dim=1)

    attn_mask = (
        q_mask.unsqueeze(1).unsqueeze(3) & kv_mask.unsqueeze(1).unsqueeze(2)
    )
    return attn_mask
    

def prepare_attn_mask(
        q: torch.Tensor,
        q_mask: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
        pad: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Prepare attention mask for transformer.

    Args:
        q: Query tensor (B, N, C)
        q_mask: Query mask (B, N)
        kv: Key/value tensor (B, M, C)
        kv_mask: Key/value mask (B, M)

    Returns:
        attn_mask: Attention mask (B, 1, N, N) or (B, 1, N, N+M)
    """
    B, Nq, C = q.shape

    # make sure masks are of the correct shape
    if q_mask is not None:
        assert q_mask.shape == (B, Nq), "q_mask must be of shape (B, Nq)"
    if kv_mask is not None:
        Bv, Nv, Cv = kv.shape
        assert kv_mask is not None, "kv_mask is required for cross-attention"
        assert B == Bv, "Batch size must match between q and key_value"
        assert kv_mask.shape == (B, Nv), "kv_mask must be (B, Nv)"

    return _create_attn_mask(q_mask, kv_mask, pad)