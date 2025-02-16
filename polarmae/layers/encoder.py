from math import ceil
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from polarmae.layers.masking import VariablePointcloudMasking, masked_layer_norm
from polarmae.layers.pos_embed import LearnedPositionalEncoder
from polarmae.layers.rpb import RelativePositionalBias3D
from polarmae.layers.grouping import fill_empty_indices
from polarmae.layers.tokenizer import make_tokenizer
from polarmae.layers.transformer import TransformerOutput, make_transformer

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_channels: int = 4,
            arch: Literal['vit_tiny', 'vit_small', 'vit_base'] = 'vit_small',
            masking_ratio: float = 0.6,
            masking_type: Literal['rand', 'fps+nms'] = 'rand',
            voxel_size: float = 5,
            tokenizer_kwargs: dict = {},
            transformer_kwargs: dict = {},
            apply_relative_position_bias: bool = False,
        ):
        super().__init__()

        self.transformer = make_transformer(
            arch_name=arch,
            use_kv=False,
            **transformer_kwargs,
        )
        self.num_channels = num_channels
        self.tokenizer = make_tokenizer(
            arch_name=arch,
            num_channels=num_channels,
            voxel_size=voxel_size,
            **tokenizer_kwargs,
        )

        self.masking = VariablePointcloudMasking(
            ratio=masking_ratio, type=masking_type
        )

        self.embed_dim = self.transformer.embed_dim
        self.pos_embed = LearnedPositionalEncoder(
            num_channels=num_channels,
            embed_dim=self.embed_dim,
            use_relative_features=tokenizer_kwargs.get('use_relative_features', False),
        )

        self.relative_position_bias = None
        if apply_relative_position_bias:
            normalized_voxel_size = self.tokenizer.grouping.group_radius
            num_heads = self.transformer.blocks[0].attn.num_heads
            self.relative_position_bias = RelativePositionalBias3D(
                num_bins=int(ceil(1/normalized_voxel_size)),
                bin_size=normalized_voxel_size,
                num_heads=num_heads,
            )

    def prepare_tokens_with_masks(
            self,
            points: torch.Tensor,
            lengths: torch.Tensor,
            ids: Optional[torch.Tensor] = None,
            endpoints: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.prepare_tokens(points, lengths, ids, endpoints)
        masked_indices, masked_mask, unmasked_indices, unmasked_mask = self.masking(out['emb_mask'].sum(-1))
        gather = lambda x, idx: torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        large_gather = lambda x, idx: torch.gather(x, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]))

        # masked_indices = fill_empty_indices(masked_indices)
        # unmasked_indices = fill_empty_indices(unmasked_indices)

        # out['masked_indices'] = masked_indices
        out['masked_mask'] = masked_mask
        out['masked_tokens'] = gather(out['x'], masked_indices)
        out['masked_centers'] = gather(out['centers'], masked_indices)
        out['masked_pos_embed'] = self.pos_embed(out['masked_centers'])
        out['masked_groups'] = large_gather(out['groups'], masked_indices)
        out['masked_groups_point_mask'] = gather(out['point_mask'], masked_indices) * out['masked_mask'].unsqueeze(-1)

        # out['unmasked_indices'] = unmasked_indices
        out['unmasked_mask'] = unmasked_mask
        out['unmasked_tokens'] = gather(out['x'], unmasked_indices)
        out['unmasked_centers'] = gather(out['centers'], unmasked_indices)
        out['unmasked_pos_embed'] = self.pos_embed(out['unmasked_centers'])
        # out['unmasked_groups'] = large_gather(out['groups'], unmasked_indices)
        # out['unmasked_groups_point_mask'] = gather(out['point_mask'], unmasked_indices) * out['unmasked_mask'].unsqueeze(-1)
        return out
    
    def prepare_tokens(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
        endpoints: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x, centers, emb_mask, id_groups, endpoints_groups, groups, point_mask, idx = (
            self.tokenizer(
                points[..., : self.num_channels],
                lengths,
                ids,
                endpoints,
                return_point_info=True,
            )
        )
        # pos_embed = self.pos_embed(centers)
        rpb = (
            self.relative_position_bias(centers)
            if self.relative_position_bias is not None
            else None
        )
        out = {
            "x": x,
            "centers": centers,
            "emb_mask": emb_mask,
            "id_groups": id_groups,
            "endpoints_groups": endpoints_groups,
            "groups": groups,
            "point_mask": point_mask,
            # "pos_embed": pos_embed,
            "rpb": rpb,
            "grouping_idx": idx,
        }
        return out

    def combine_intermediate_layers(
        self,
        output: TransformerOutput,
        mask: Optional[torch.Tensor] = None,
        layers: List[int] = [0],
    ) -> torch.Tensor:
        hidden_states = [
            masked_layer_norm(output.hidden_states[i], output.hidden_states[i].shape[-1], mask)
            for i in layers
        ]
        return torch.stack(hidden_states, dim=0).mean(0)

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