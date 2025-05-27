import torch
import torch.nn as nn
from polarmae.layers.attention import Attention, prepare_attn_mask
from polarmae.layers.grouping import masked_gather, fill_empty_indices
from polarmae.layers.block import Block
from typing import Optional, Tuple

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GroupLocalAttention(nn.Module):
    """
    Local group attention.

    Given a set of points and indices correspond to groups of points, we
    perform attention among points within each group, and then average them
    across groups (if points are found in more than 1 group). 

    Notably, in the forward
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        fill_with_original_points: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.fill_with_original_points = fill_with_original_points
        self.attn = Attention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            use_flash_attn=use_flash_attn,
        )

        self.layer_scale = LayerScale(embed_dim)
        # self.block = Block(
        #     dim=embed_dim,
        #     num_heads=num_heads,
        #     mlp_ratio=4.0,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop_rate, 
        #     drop=proj_drop_rate,
        #     drop_path=drop_rate,
        #     use_flash_attn=use_flash_attn,
        #     use_layer_scale=True,
        # )
        
    def group_attention(
        self,
        upscaled_feats,
        grouping_idx,
        grouping_point_mask,
        groups_shape=None,
        context_tokens=None,
    ):
        if context_tokens is not None:
            assert context_tokens.ndim == 3, f"context_tokens must be 3D, got {context_tokens.ndim}"
            assert context_tokens.shape[0] == upscaled_feats.shape[0], f"context_tokens and upscaled_feats must have the same batch size, got {context_tokens.shape[0]} and {upscaled_feats.shape[0]}"
            assert context_tokens.shape[2] == upscaled_feats.shape[2], f"context_tokens and upscaled_feats must have the same embedding dimension, got {context_tokens.shape[2]} and {upscaled_feats.shape[2]}"
        if groups_shape is None:
            groups_shape = grouping_idx.shape


        # sort point features into local groups
        x_groups = masked_gather(upscaled_feats, fill_empty_indices(grouping_idx)) # [B, N_groups, N_max, embed_dim]

        # truncate to the actual number of groups
        x_groups = x_groups[ :, :groups_shape[1], :groups_shape[2], :]  # [B, G, N_max, C]

        # add context tokens to the front of each group
        if context_tokens is not None:
            B, G, N_max, C = x_groups.shape
            N_context = context_tokens.shape[1]
            x_groups = torch.cat([x_groups, context_tokens.unsqueeze(1).expand(-1, G, -1, -1)], dim=2)
            grouping_point_mask = torch.cat([
                torch.ones(B, G, N_context, device=grouping_point_mask.device, dtype=grouping_point_mask.dtype),
                grouping_point_mask,
            ], dim=2)

        # prepare attention masks
        x_groups_reshaped = x_groups.reshape(-1, *x_groups.shape[-2:]) # [B*G, N_max, C]
        point_mask_reshaped = grouping_point_mask.reshape(
            -1, grouping_point_mask.shape[-1]
        ) # [B*G, N_max]
        attn_mask = prepare_attn_mask(
            x_groups_reshaped, point_mask_reshaped
        ) # [B*G, 1, N_max, N_max]

        # apply attention among points within each group
        x_groups_attn, attn = self.attn(
            q=x_groups_reshaped,
            qkv_attn_mask=attn_mask,
        ) # [B*G, N_max, C]

        # reshape back to the original shape
        x_groups_attn = x_groups_attn.reshape(x_groups.shape) # [B, G, N_max, C]

        # remove context tokens if they were added
        if context_tokens is not None:
            x_groups_attn = x_groups_attn[:, :, N_context:, :]
            grouping_point_mask = grouping_point_mask[:, :, N_context:]

        return x_groups_attn, attn

    def average_groups(
        self,
        attended_groups: torch.Tensor,
        grouping_idx: torch.Tensor,
        points: torch.Tensor,
        grouping_point_mask: torch.Tensor,
    ) -> torch.Tensor:
        # prepare indices and groups:
        # - fill empty indices with 0
        # - flatten indices
        # - flatten groups
        B, N_max, C = points.shape
        filled_indices = fill_empty_indices(grouping_idx)
        filled_indices_flat = filled_indices.view(B, -1)  # (B, G*K)
        filled_indices_flat[filled_indices_flat.eq(-1)] = 0
        updated_grouped_flat = (
            attended_groups * grouping_point_mask.unsqueeze(-1)
        ).view(B, -1, C)  # (B, G*K, C)

        # perform averaging of grouped points. we do this by
        # adding all instances of each point in each group to the accumulator,
        # and then dividing by the number of instances of each point.
        # if fill_with_original_points is True, we set ungrouped points (i.e.
        # points with indices not found in `filled_indices_flat`) to the original points themselves.

        # initialize accumulator
        accumulator = torch.zeros(B, N_max, C, device=points.device)  # (B, N_max, C)
        # add all instances of each point in each group to the accumulator
        accumulator = accumulator.scatter_add(
            dim=1,
            index=filled_indices_flat.unsqueeze(-1).expand(-1, -1, C),
            src=updated_grouped_flat,
        )  # (B, N_max, C)

        # count the number of points in each group
        count = torch.zeros(
            B, N_max, 1, device=points.device, dtype=points.dtype
        )  # (B, N_max, 1)
        count = count.scatter_add(
            dim=1,
            index=filled_indices_flat.unsqueeze(-1).expand(-1, -1, 1),
            src=grouping_point_mask.unsqueeze(-1).view(B, -1, 1).float(),
        )  # (B, N_max, 1)

        # set points with count=0 in accumulator to the points themselves
        if self.fill_with_original_points:
            accumulator[count.squeeze().eq(0)] = points[count.squeeze().eq(0)]

        # clamp count to 1.0 to avoid div by zero
        count = torch.clamp(count, min=1.0)

        # get average of accumulated features per point
        averaged_updates = accumulator / count  # (B, N_max, C)
        return averaged_updates

    def forward(
        self,
        upscaled_feats: torch.Tensor,
        grouping_idx: torch.Tensor,
        grouping_point_mask: torch.Tensor,
        groups_shape: Optional[torch.Tensor] = None,
        context_tokens: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        # upscaled_feats: [B, N_points, embed_dim]
        # grouping_idx: [B, G, N_max] - outputted from grouping module, used to cast an event into groups
        # grouping_point_mask: [B, G, N_max] - outputted from grouping module, used to cast an event into groups
        # average the attention across groups
        x_groups_attn, attn = self.group_attention(
            upscaled_feats,
            grouping_idx,
            grouping_point_mask,
            groups_shape,
            context_tokens,
        )
        average_groups = self.average_groups(
            x_groups_attn,
            grouping_idx,
            upscaled_feats,
            grouping_point_mask,
        )

        upscaled_feats = upscaled_feats + self.layer_scale(average_groups)

        if return_attn:
            return upscaled_feats, attn
        else:
            return upscaled_feats

