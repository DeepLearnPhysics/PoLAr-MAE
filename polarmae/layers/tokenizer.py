from math import sqrt
from typing import Literal, Tuple

import torch
import torch.nn as nn
from polarmae.layers.grouping import PointcloudGrouping
from polarmae.layers.pointnet import MaskedMiniPointNet

__all__ = [
    'PointcloudTokenizer',
    'make_tokenizer',
    'vits5_tokenizer',
    'vits25_tokenizer',
    'vitb5_tokenizer',
    'vitb25_tokenizer',
    'vits2p5_tokenizer',
    'vitb2p5_tokenizer',
    'vitt2p5_tokenizer',
]

class PointcloudTokenizer(nn.Module):
    def __init__(
        self,
        num_init_groups: int,
        context_length: int,
        group_max_points: int,
        group_radius: float | None,
        group_upscale_points: int | None,
        overlap_factor: float | None,
        token_dim: int,
        num_channels: int,
        reduction_method: str = 'energy',
        use_relative_features: bool = False,
        normalize_group_centers: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grouping = PointcloudGrouping(
            num_groups=num_init_groups,
            group_max_points=group_max_points,
            group_radius=group_radius,
            group_upscale_points=group_upscale_points,
            overlap_factor=overlap_factor,
            context_length=context_length,
            reduction_method=reduction_method,
            use_relative_features=use_relative_features,
            normalize_group_centers=normalize_group_centers,
        )

        self.embedding = MaskedMiniPointNet(num_channels, token_dim)

    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        endpoints: torch.Tensor | None = None,
        return_point_info: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, num_channels)
        # lengths: (B,)
        group: torch.Tensor
        group_center: torch.Tensor
        tokens: torch.Tensor
        lengths: torch.Tensor
        semantic_id_groups: torch.Tensor | None

        grouping_out = self.grouping(
            points, lengths, semantic_id, endpoints)
        
        groups = grouping_out['groups']
        point_mask = grouping_out['point_mask']
        group_center = grouping_out['group_centers']
        embedding_mask = grouping_out['embedding_mask']
        semantic_id_groups = grouping_out['semantic_id_groups']
        endpoints_groups = grouping_out['endpoints_groups']
        idx = grouping_out['idx']
        
        # just embed nonzero groups
        out = self.embedding(groups[embedding_mask], point_mask[embedding_mask].unsqueeze(1))
        tokens = torch.zeros(
            groups.shape[0],groups.shape[1],
            self.token_dim,
            device=out.device,
            dtype=out.dtype,
        )
        tokens[embedding_mask] = out

        if return_point_info:
            return (
                tokens,
                group_center,
                embedding_mask,
                semantic_id_groups,
                endpoints_groups,
                groups,
                point_mask,
                idx,
            )
        else:
            return (
                tokens,
                group_center,
                embedding_mask,
                semantic_id_groups,
                endpoints_groups,
                idx,
            )

    @staticmethod
    def extract_model_checkpoint(path: str):
        checkpoint = torch.load(path, weights_only=True)
        return {k.replace("embed.", "embedding."):v for k,v in checkpoint["state_dict"].items() if k.startswith("embed.")}

def make_tokenizer(
    arch_name: Literal['vit_tiny', 'vit_small', 'vit_base'],
    num_channels: int,
    voxel_size: int | float,
    **kwargs,
) -> PointcloudTokenizer:
    compact_arch_name = arch_name.replace("_","")[:4]

    if int(voxel_size) == voxel_size:
        name = f"{compact_arch_name}{int(voxel_size)}_tokenizer"
    else:
        name = f"{compact_arch_name}{str(voxel_size).replace('.', 'p')}_tokenizer"

    return globals()[name](num_channels=num_channels, **kwargs)

def _2p5voxel_tokenizer(num_channels=4, embed_dim=384, **kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-S/2.5 voxel tokenizer
    """
    config = dict(
        num_init_groups=2048,
        context_length=1024,
        group_max_points=16,
        group_radius=2.5 / (768 * sqrt(3) / 2), # voxel_radius * scaling_constant
        group_upscale_points=64,
        overlap_factor=0.73,
        reduction_method='fps',
    )
    config.update(kwargs)
    return PointcloudTokenizer(
        token_dim=embed_dim,
        num_channels=num_channels,
        **config,
    )

def _5voxel_tokenizer(num_channels=4,embed_dim=384,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-S/5 voxel tokenizer
    """
    config = dict(
        num_init_groups=2048,
        context_length=512,
        group_max_points=32,
        group_radius=5 / (768 * sqrt(3) / 2), # voxel_radius * scaling_constant
        group_upscale_points=256,
        overlap_factor=0.72,
        reduction_method='fps',
    )
    config.update(kwargs)
    return PointcloudTokenizer(
        token_dim=embed_dim,
        num_channels=num_channels,
        **config,
    )

def _25voxel_tokenizer(num_channels=4,embed_dim=384,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-S/25 voxel tokenizer
    """
    config = dict(
        num_init_groups=256,
        context_length=128,
        group_max_points=128,
        group_radius=25 / (768 * sqrt(3) / 2), # voxel_radius * scaling_constant
        group_upscale_points=2048,
        overlap_factor=0.72,
        reduction_method='fps',
        use_relative_features=False,
        normalize_group_centers=True,
    )
    config.update(kwargs)
    return PointcloudTokenizer(
        num_channels=num_channels,
        token_dim=embed_dim,
        **config,
    )

def vits2p5_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-S/2.5 voxel tokenizer
    """
    return _2p5voxel_tokenizer(
        num_channels=num_channels,
        embed_dim=384,
        **kwargs,
    )

def vitt2p5_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-T/2.5 voxel tokenizer
    """
    return _2p5voxel_tokenizer(
        num_channels=num_channels,
        embed_dim=192,
        **kwargs,
    )

def vitt5_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-T/5 voxel tokenizer
    """
    return _5voxel_tokenizer(
        num_channels=num_channels,
        embed_dim=192,
        **kwargs,
    )

def vitt25_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-T/25 voxel tokenizer
    """
    return _25voxel_tokenizer(
        num_channels=num_channels,
        embed_dim=192,
        **kwargs,
    )

def vits5_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-S/5 voxel tokenizer
    """
    return _5voxel_tokenizer(
        num_channels=num_channels,
        embed_dim=384,
        **kwargs,
    )

def vits25_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-S/25 voxel tokenizer
    """
    return _25voxel_tokenizer(
        num_channels=num_channels,
        **kwargs,
    )

def vitb5_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-B/5 voxel tokenizer
    """
    return _5voxel_tokenizer(
        num_channels=num_channels,
        embed_dim=768,
        **kwargs,
    )

def vitb25_tokenizer(num_channels=4,**kwargs) -> PointcloudTokenizer:
    """
    LArNET ViT-B/25 voxel tokenizer
    """
    return _25voxel_tokenizer(
        num_channels=num_channels,
        **kwargs,
    )
