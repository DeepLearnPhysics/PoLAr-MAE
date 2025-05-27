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
from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

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
            **transformer_kwargs,
            use_kv=False,
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

    def flash_attention(self, use_flash: bool = True):
        for block in self.transformer.blocks:
            block.attn.use_flash_attn = use_flash

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        # unfreeze the prefix tokens if prefix tuning is enabled
        if self.transformer.prefix_tuning:
            log.info("🔥  Unfreezing prefix tokens.")
            self.transformer.p_theta_prime.requires_grad = True
            for param in self.transformer.prefix_mlp.parameters():
                param.requires_grad = True

    def prepare_tokens_with_masks(
            self,
            points: torch.Tensor,
            lengths: torch.Tensor,
            ids: Optional[torch.Tensor] = None,
            endpoints: Optional[torch.Tensor] = None,
            augmented_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.prepare_tokens(points, lengths, ids, endpoints, augmented_points, encode_all=False)

        gather = lambda x, idx: torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        large_gather = lambda x, idx: torch.gather(x, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]))

        # masked_indices = fill_empty_indices(masked_indices)
        # unmasked_indices = fill_empty_indices(unmasked_indices)

        out["augmented_pos_embed"] = None
        if augmented_points is not None:
            out['augmented_pos_embed'] = self.pos_embed(gather(out['augmented_centers'], out['unmasked_indices']))

        # out['masked_tokens'] = gather(out['x'], out['masked_indices'])
        # out['masked_tokens'] = out['x']
        out['masked_centers'] = gather(out['centers'], out['masked_indices'])
        out['masked_pos_embed'] = self.pos_embed(out['masked_centers'])
        out['masked_groups'] = large_gather(out['groups'], out['masked_indices'])
        out['masked_groups_point_mask'] = gather(out['point_mask'], out['masked_indices']) * out['masked_mask'].unsqueeze(-1)

        out['unmasked_tokens'] = out['x']
        out['unmasked_centers'] = gather(out['centers'], out['unmasked_indices'])
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
        augmented_points: Optional[torch.Tensor] = None,
        encode_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        grouping_out = (
            self.tokenizer(
                points[..., : self.num_channels],
                lengths,
                ids,
                endpoints,
                augmented_points,
                encode_all=encode_all,
            )
        )

        pos_embed = (
            self.pos_embed(grouping_out["centers"]) if encode_all is not None else None
        )

        rpb = (
            self.relative_position_bias(grouping_out["centers"])
            if self.relative_position_bias is not None
            else None
        )
        out = {
            "x": grouping_out["tokens"],
            "augmented_tokens": grouping_out["augmented_tokens"],
            "centers": grouping_out["centers"],
            "emb_mask": grouping_out["embedding_mask"],
            "id_groups": grouping_out["semantic_id_groups"],
            "endpoints_groups": grouping_out["endpoints_groups"],
            "groups": grouping_out["groups"],
            "augmented_groups": grouping_out["augmented_groups"],
            "augmented_centers": grouping_out["augmented_centers"],
            "point_mask": grouping_out["point_mask"],
            "pos_embed": pos_embed,
            "rpb": rpb,
            "grouping_idx": grouping_out["idx"],
            "masked_indices": grouping_out["masked_indices"],
            "masked_mask": grouping_out["masked_mask"],
            "unmasked_indices": grouping_out["unmasked_indices"],
            "unmasked_mask": grouping_out["unmasked_mask"],
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
        final_norm: bool = True,
    ) -> TransformerOutput:
        """Call transformer forward"""
        return self.transformer.forward(
            q, pos_q, q_mask, kv, pos_kv, kv_mask, return_hidden_states, return_attentions, return_ffns, final_norm
        )