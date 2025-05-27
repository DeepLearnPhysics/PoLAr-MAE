from math import sqrt
from typing import Literal, Tuple

import torch
import torch.nn as nn
from polarmae.layers.grouping import PointcloudGrouping
from polarmae.layers.pointnet import MaskedMiniPointNet
from polarmae.layers.masking import VariablePointcloudMasking
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
        masking_ratio: float = 0.6,
        masking_type: Literal['rand'] = 'rand',
        use_fps_seed: bool = True,
        rescale_by_group_radius: bool | float = True,
    ) -> None:
        super().__init__()

        def try_type(x, _type):
            if isinstance(x, bool):
                return x
            elif x is None:
                return None
            else:
                return _type(x)

        self.token_dim = token_dim
        self.grouping = PointcloudGrouping(
            num_groups=try_type(num_init_groups, int),
            group_max_points=try_type(group_max_points, int),
            group_radius=try_type(group_radius, float),
            group_upscale_points=try_type(group_upscale_points, int),
            overlap_factor=try_type(overlap_factor, float),
            context_length=try_type(context_length, int),
            reduction_method=reduction_method,
            use_relative_features=use_relative_features,
            normalize_group_centers=normalize_group_centers,
            use_fps_seed=use_fps_seed,
            rescale_by_group_radius=try_type(rescale_by_group_radius, float),
        )

        self.embedding = MaskedMiniPointNet(num_channels, token_dim)

        self.masking = VariablePointcloudMasking(
            ratio=masking_ratio, type=masking_type
        )


    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        endpoints: torch.Tensor | None = None,
        augmented_points: torch.Tensor | None = None,
        encode_all: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, num_channels)
        # lengths: (B,)
        tokens: torch.Tensor
        lengths: torch.Tensor

        grouping_out = self.grouping(
            points, lengths, semantic_id, endpoints, augmented_points)
        
        masked_indices, masked_mask, unmasked_indices, unmasked_mask, unmasked_groups, unmasked_point_mask = [None] * 6
        if encode_all:
            groups = grouping_out['groups']
            point_mask = grouping_out['point_mask']
            emb_mask = grouping_out['embedding_mask']

            augmented_groups = grouping_out['augmented_groups']
        else:
            masked_indices, masked_mask, unmasked_indices, unmasked_mask = self.masking(grouping_out['embedding_mask'].sum(-1))

            gather = lambda x, idx: torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))
            large_gather = lambda x, idx: torch.gather(x, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]))

            unmasked_groups = large_gather(
                grouping_out["groups"], unmasked_indices
            )
            unmasked_point_mask = gather(grouping_out['point_mask'], unmasked_indices)

            groups = unmasked_groups
            point_mask = unmasked_point_mask
            emb_mask = unmasked_mask 

            augmented_groups = grouping_out['augmented_groups']
            if augmented_groups is not None:
                augmented_groups = large_gather(augmented_groups, unmasked_indices)

        # just embed nonzero visible groups (no need to encode the others !)
        with torch.amp.autocast_mode.autocast(device_type=groups.device.type, dtype=torch.float32):
            flattened_tokens = self.embedding(
                groups[emb_mask], point_mask[emb_mask].unsqueeze(1)
            )

            tokens = torch.zeros(
                groups.shape[0],
                groups.shape[1],
                self.token_dim,
                device=flattened_tokens.device,
                dtype=flattened_tokens.dtype,
            )
            tokens[emb_mask] = flattened_tokens


        # create ANOTHER set of visible tokens -- augmented by some transformation
        augmented_tokens = None
        if augmented_groups is not None:
            with torch.amp.autocast(device_type=augmented_groups.device.type, dtype=torch.float32): 
                flattened_augmented_tokens = self.embedding(
                    augmented_groups[emb_mask], point_mask[emb_mask].unsqueeze(1)
                )
                augmented_tokens = torch.zeros(
                    augmented_groups.shape[0],
                    augmented_groups.shape[1],
                    self.token_dim,
                    device=flattened_augmented_tokens.device,
                    dtype=flattened_augmented_tokens.dtype,
                )
                augmented_tokens[emb_mask] = flattened_augmented_tokens


        grouping_out["tokens"] = tokens
        grouping_out["masked_indices"] = masked_indices
        grouping_out["masked_mask"] = masked_mask
        grouping_out["unmasked_indices"] = unmasked_indices
        grouping_out["unmasked_mask"] = unmasked_mask
        grouping_out["unmasked_groups"] = unmasked_groups
        grouping_out["unmasked_point_mask"] = unmasked_point_mask
        grouping_out["augmented_tokens"] = augmented_tokens
        return grouping_out

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
        group_max_points=24,
        group_radius=2.5 / (768 * sqrt(3) / 2), # voxel_radius * scaling_constant
        group_upscale_points=64,
        overlap_factor=0.75,
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
