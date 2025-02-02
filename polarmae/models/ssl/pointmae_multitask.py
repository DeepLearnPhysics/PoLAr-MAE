from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Literal

import pytorch_lightning as pl
import torch
from polarmae.models.ssl.base import SSLModel
from polarmae.layers.encoder import TransformerEncoder
from polarmae.layers.decoder import TransformerDecoder
from polarmae.layers.pointnet import MaskedMiniPointNet
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from polarmae.utils.pylogger import RankedLogger
import torch.nn.functional as F


log = RankedLogger(__name__, rank_zero_only=True)

class PointMAE_Multitask(SSLModel):
    def __init__(self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        learning_rate: float = 1e-3,
        mae_prediction: Literal['full', 'pos'] = 'full',
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        lr_scheduler_stepping: str = 'step',
        freeze_last_layer_iters: int = -1,
        train_transformations: List[str] = ['center_and_scale', 'rotate'],
        val_transformations: List[str] = ["center_and_scale"],
        transformation_center: torch.Tensor | List[float] | float = torch.tensor([768, 768, 768]) / 2,
        transformation_scale_factor: torch.Tensor | List[float] | float = 1 / (768 * sqrt(3) / 2),
        transformation_rotate_dims: Optional[List[int]] = [0,1,2],
        transformation_rotate_degs: Optional[float] = None,
        svm_validation: Dict[str, pl.LightningDataModule] = {},
        svm_validation_C=0.005,  # C=0.012 copied from Point-M2AE code
        svm_validation_max_tokens: int = 7500,
        fix_estimated_stepping_batches: Optional[int] = None,  # multi GPU bug fix
        loss_weights: Dict[str, float] = {'ae': 0.0, 'chamfer': 1.0, 'energy': 1.0},
        ):

        super().__init__()
        self.save_hyperparameters()
        super().configure_transformations()
        self.hparams.svm_validation = svm_validation

        self.encoder = encoder
        self.decoder = decoder

        if mae_prediction == 'full':
            log.info("⚙️  MAE prediction: full patch reconstruction")
            self.mae_channels = 4
        elif mae_prediction == 'pos':
            log.info("⚙️  MAE prediction: position prediction")
            self.mae_channels = 3

        self.increase_dim = nn.Conv1d(
            encoder.embed_dim,
            self.mae_channels * encoder.tokenizer.grouping.group_max_points,
            1,
        )
        init_std = 0.02
        self.mask_token = nn.Parameter(torch.zeros(encoder.embed_dim))
        nn.init.trunc_normal_(
            self.mask_token, mean=0, std=init_std, a=-init_std, b=init_std
        )

        self.do_ae = loss_weights.get('ae', 0) > 0


        # Energy that takes in encoder.embed_dim*2 and outputs 32x1 (32 pt, 1 energy channel)
        self.embed_dim = encoder.embed_dim
        self.equivariant_patch_encoder = MaskedMiniPointNet(
            channels=3,
            feature_dim=self.embed_dim,
            equivariant=True
        )

        self.energy_decoder = nn.Conv1d(
            2 * self.embed_dim,
            1 * encoder.tokenizer.grouping.group_max_points,
            1,
        )  # 2*embed_dim (1 embed_dim for encoded positions, 1 for regressed masked tokens)

    def compute_loss(self, points: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        # encode toks
        out = self.encoder.prepare_tokens_with_masks(points, lengths)
        masked, unmasked = out['masked'], out['unmasked']

        # run visible tokens through encoder
        encoder_output = self.encoder.transformer(out['x'], out['pos_embed'], unmasked).last_hidden_state

        # corrupt embeddings with masked tokens
        corrupted_embeddings = (
            encoder_output * unmasked.unsqueeze(-1) + 
            self.mask_token * masked.unsqueeze(-1)
        )
        decoder_output = self.decoder.transformer(corrupted_embeddings, out['pos_embed'], out['emb_mask']).last_hidden_state
        masked_output = decoder_output[masked]

        # full patch reconstruction task
        upscaled = self.increase_dim(masked_output.transpose(0, 1)).transpose(0, 1)
        upscaled = upscaled.reshape(upscaled.shape[0], -1, self.mae_channels)

        masked_groups = out['groups'][masked]
        point_lengths = out['point_mask'][masked].sum(-1)

        chamfer_loss, _ = chamfer_distance(
            upscaled.float(),
            masked_groups.float()[..., :self.mae_channels],
            x_lengths=point_lengths,
            y_lengths=point_lengths,
        )

        # energy infilling task
        masked_point_mask = out['point_mask'][masked].unsqueeze(1) # (B*G, 1, S)
        equivariant_patch_encoder_output = self.equivariant_patch_encoder(
            masked_groups[..., :3], masked_point_mask
        )
        # concatenate output with encoded masked tokens
        decoder_input = torch.cat(
            [equivariant_patch_encoder_output, masked_output], dim=1
        )

        # decode energy from embeddings
        decoded_energy = self.energy_decoder(
            decoder_input.transpose(0, 1)
        ).transpose(0, 1)
        
        energy_loss = F.mse_loss(
            decoded_energy[masked_point_mask.squeeze(1)], masked_groups[masked_point_mask.squeeze(1)][..., -1]
        )

        ae_loss = 0
        if self.do_ae:
            upscaled_unmasked = self.increase_dim(encoder_output.transpose(0, 1)).transpose(0, 1)
            upscaled_unmasked = upscaled_unmasked.reshape(upscaled_unmasked.shape[0], -1, self.encoder.num_channels)
            unmasked_groups = out['groups'][unmasked]
            unmasked_point_lengths = out['point_mask'][unmasked].sum(-1)
            ae_loss, _ = chamfer_distance(
                upscaled_unmasked.float(),
                unmasked_groups.float(),
                x_lengths=unmasked_point_lengths,
                y_lengths=unmasked_point_lengths,
            )
        return {'chamfer': chamfer_loss, 'ae': ae_loss, 'energy': energy_loss}

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        points = batch['points']
        lengths = batch['lengths']
        points = self.train_transformations(points)
        loss_dict = self.compute_loss(points, lengths)
        self.log_losses(loss_dict, prefix='loss/train_')
        loss = sum(loss_dict[k] * self.hparams.loss_weights.get(k, 1.0) for k in loss_dict.keys())
        self.log('loss/train', loss, sync_dist=True, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        points = batch['points']
        lengths = batch['lengths']
        points = self.val_transformations(points)
        loss_dict = self.compute_loss(points, lengths)
        self.log_losses(loss_dict, prefix='loss/val_')
        loss = sum(loss_dict[k] * self.hparams.loss_weights.get(k, 1.0) for k in loss_dict.keys())
        self.log('loss/val', loss, sync_dist=True, on_epoch=True, on_step=False)
        return loss