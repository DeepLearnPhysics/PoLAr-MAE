import copy
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision, F1Score

from polarmae.eval.segmentation import compute_shape_ious
from polarmae.layers.encoder import TransformerEncoder
from polarmae.layers.decoder import TransformerDecoder
from polarmae.layers.feature_upsampling import PointNetFeatureUpsampling
from polarmae.layers.masking import masked_max, masked_mean
from polarmae.layers.seg_head import SegmentationHead
from polarmae.loss import SoftmaxFocalLoss
from polarmae.models.finetune.base import FinetuneModel
from polarmae.utils.pylogger import RankedLogger
from math import sqrt

log = RankedLogger(__name__, rank_zero_only=True)


class PartSegmentation(FinetuneModel):
    def __init__(
        self,
        encoder: TransformerEncoder,
        seg_decoder: Optional[TransformerDecoder],
        num_classes: int,
        seg_head_fetch_layers: List[int] = [3, 7, 11],
        seg_head_dim: int = 512,
        seg_head_dropout: float = 0.5,
        apply_local_attention: bool = False,
        condition_global_features: bool = False,
        use_pos_enc_for_upsampling: bool = False,
        # LR/optimizer
        learning_rate: float = 1e-3,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        lr_scheduler_stepping: str = "step",
        # Training
        train_transformations: List[str] = ["center_and_scale", "rotate"],
        val_transformations: List[str] = ["center_and_scale"],
        transformation_center: torch.Tensor | List[float] | float = torch.tensor([768, 768, 768]) / 2,
        transformation_scale_factor: torch.Tensor | List[float] | float = 1 /(768 * sqrt(3) / 2),
        transformation_rotate_dims: List[int] = [0, 1, 2],
        transformation_rotate_degs: Optional[int] = None,
        encoder_freeze: bool = False,
        loss_func: Literal["nll", "focal", "fancy"] = "nll",
        # Checkpoints
        pretrained_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        super().configure_transformations()

        self.encoder = encoder
        self.seg_decoder = seg_decoder

        self.loss_func = loss_func
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.encoder_freeze = encoder_freeze

        self.condition_global_features = condition_global_features

        upsampling_dim: int = self.encoder.embed_dim  # type: ignore
        self.upsampler = PointNetFeatureUpsampling(
            in_channel=upsampling_dim, mlp=[upsampling_dim, upsampling_dim]
        )  # type: ignore

        self.seg_head = SegmentationHead(
            self.encoder.embed_dim if (self.seg_decoder is None and self.condition_global_features) else 0,
            0,  # event-wide label embedding -- 0 for polarmae!
            upsampling_dim,
            self.hparams.seg_head_dim,
            self.hparams.seg_head_dropout,
            num_classes,
        )
        if self.seg_decoder is not None:
            log.info(
                "Using decoder, so not aggregating token features (i.e. `seg_head_fetch_layers`) for use in seg head."
            )


    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

        assert self.trainer.datamodule.num_seg_classes == self.hparams.num_classes, f"num_seg_classes {self.trainer.datamodule.num_seg_classes} must match num_classes {self.hparams.num_classes} given to model constructor"
        self.category_to_seg_classes = self.trainer.datamodule.category_to_seg_classes
        self.seg_class_to_category = self.trainer.datamodule.seg_class_to_category

        metric_kwargs = {
            'num_classes': self.hparams.num_classes,
            'ignore_index': -1,
            'compute_on_cpu': False,
            'sync_on_compute': False,
            'dist_sync_on_step': True,
            'zero_division': 0,
        }

        self.train_acc = Accuracy('multiclass', **metric_kwargs)
        self.train_macc = Accuracy('multiclass', **metric_kwargs, average="macro")
        self.train_precision = Precision("multiclass", **metric_kwargs)
        self.train_mprecision = Precision("multiclass", **metric_kwargs, average="macro")

        self.val_acc = Accuracy('multiclass', **metric_kwargs)
        self.val_macc = Accuracy('multiclass', **metric_kwargs, average="macro")
        self.val_precision = Precision("multiclass", **metric_kwargs)
        self.val_mprecision = Precision("multiclass", **metric_kwargs, average="macro")

        self.val_f1_score = F1Score("multiclass", **metric_kwargs)
        self.val_f1_score_m = F1Score("multiclass", **metric_kwargs, average="macro")

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")
                logger.experiment.define_metric("val_ins_miou", summary="last,max")
                logger.experiment.define_metric("val_cat_miou", summary="last,max")
                logger.experiment.define_metric("val_precision", summary="last,max")
                logger.experiment.define_metric("val_mprecision", summary="last,max")
                logger.experiment.define_metric("val_f1_score", summary="last,max")
                logger.experiment.define_metric("val_f1_score_m", summary="last,max")

        """ ------------------------------------------------------------------------ """
        """                                  losses                                  """
        """ ------------------------------------------------------------------------ """
        # instantiate loss function
        if self.hparams.loss_func == "nll":
            self.loss_func = nn.NLLLoss(weight=self.trainer.datamodule.class_weights, reduction='mean', ignore_index=-1)
        elif self.hparams.loss_func in ["focal", "fancy"]:
            self.loss_func = SoftmaxFocalLoss(weight=self.trainer.datamodule.class_weights, reduction='mean', ignore_index=-1, gamma=2)
        else:
            raise ValueError(f"Unknown loss function: {self.hparams.loss_func}")

        """ ------------------------------------------------------------------------ """
        """                                  checkpoints                             """
        """ ------------------------------------------------------------------------ """
        if self.hparams.pretrained_ckpt_path is not None:
            self.load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)
            log.info('🔥  Loaded pretrained checkpoint.')
        else:
            log.info('🔥  No pretrained checkpoint loaded. Training from scratch??')

        """ ------------------------------------------------------------------------ """
        """                                  freezing                                 """
        """ ------------------------------------------------------------------------ """
        if self.hparams.encoder_freeze:
            self.encoder.requires_grad_(False)
            log.info('🔥  Performing linear probing.')
        else:
            log.info('🔥  Not freezing encoder.')


    def forward(
            self, 
            points: torch.Tensor,
            lengths: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            class_slice: Optional[slice] = None,
            return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # run encoder as usual
        out = self.encoder.prepare_tokens_with_masks(points, lengths, ids=labels)
        output = self.encoder(
            out["x"],
            out["pos_embed"],
            out["emb_mask"],
            return_hidden_states=self.seg_decoder is None,
        )
        batch_lengths = out['emb_mask'].sum(dim=1)

        if self.seg_decoder is not None:
            output = self.seg_decoder(output.last_hidden_state, out['pos_embed'], out['emb_mask'])
            token_features = output.last_hidden_state
        else:
            # fetch intermediate layers & get averaged token features
            token_features = self.encoder.fetch_intermediate_layers(
                output,
                out['emb_mask'],
                self.hparams.seg_head_fetch_layers,
            ) # (B, T, C)
            assert token_features.shape[1] == out['x'].shape[1], "token_features and tokens must have the same number of tokens!"

            if self.condition_global_features:
                # get global features
                token_features_max = masked_max(token_features, out['emb_mask'])  # (B, C)
                token_features_mean = masked_mean(token_features, out['emb_mask'])  # (B, C)

                global_feature = torch.cat(
                    [token_features_max, token_features_mean], dim=-1
                )  # (B, 2*C')

        # upsample token features to point features a la PointNet++
        point_mask = torch.arange(lengths.max(), device=lengths.device).expand(
            len(lengths), -1
        ) < lengths.unsqueeze(-1)

        x, idx = self.upsampler(
            points[..., :3],
            out['centers'][:, :, :3],
            points[..., :3],
            token_features,
            lengths,
            batch_lengths,
            point_mask,
        )  # (B, N, 384)

        if self.seg_decoder is None and self.condition_global_features:
            B, N, C = points.shape
            global_feature = global_feature.reshape(B, -1) # (B, 2*C')
            x = torch.cat(
                [x, global_feature.unsqueeze(-1).expand(-1, -1, N).transpose(1, 2)], dim=-1
            )  # (B, N, 2*C')

        x = self.seg_head(x.transpose(1, 2), point_mask).transpose(1, 2) # (B, N, num_classes)

        if class_slice is not None:
            x = x[..., class_slice] # (B, N, num_classes_to_keep)

        return {
            'x': x if return_logits else F.log_softmax(x, dim=-1),
            'idx': idx,
            'point_mask': point_mask,
            'id_groups': out['id_groups'],
            'id_pred': torch.max(x, dim=-1).indices,
        }

    def compute_loss(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        class_mask: Optional[slice] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        out = self.forward(
            points,
            lengths,
            labels,
            class_mask,
            return_logits=self.hparams.loss_func == 'fancy',
        )

        loss = self.loss_func(
            out['x'][out['point_mask']],
            labels.squeeze(-1)[out['point_mask']]
        )

        loss_dict = {
            self.hparams.loss_func: loss,
        }
        output_dict = { 
            'logits': out['x'],
            'pred': out['id_pred'],
            'labels': labels,
        }

        return loss_dict, output_dict

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict, output_dict = self.compute_loss(
            self.train_transformations(batch['points']),
            batch['lengths'],
            batch['semantic_id'],
        )
        bsz = batch['points'].shape[0]
        self.log_losses(loss_dict, prefix='train_', batch_size=bsz)
        pred, labels = output_dict['pred'], output_dict['labels'].squeeze(-1)
        for metric in ['macc', 'mprecision', 'acc', 'precision']:
            self.log(f'train_{metric}', getattr(self, f'train_{metric}')(pred, labels).to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)
        loss = loss_dict[self.hparams.loss_func]
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict, output_dict = self.compute_loss(
            self.val_transformations(batch['points']),
            batch['lengths'],
            batch['semantic_id'],
        )
        bsz = batch['points'].shape[0]
        self.val_bsz = bsz
        self.log_losses(loss_dict, prefix='val_', batch_size=bsz)
        pred, labels = output_dict['pred'], output_dict['labels'].squeeze(-1)
        for metric in ['macc', 'mprecision', 'acc', 'precision', 'f1_score', 'f1_score_m']:
            self.log(f'val_{metric}', getattr(self, f'val_{metric}')(pred, labels).to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)

        ious = compute_shape_ious(
            output_dict['logits'],
            output_dict['labels'],
            batch['lengths'],
            self.category_to_seg_classes,
            self.seg_class_to_category,
        )
        self.ious.append(ious)

    def on_validation_epoch_start(self) -> None:
        self.ious = []

    def on_validation_epoch_end(self) -> None:
        shape_mious = {cat: [] for cat in self.category_to_seg_classes.keys()}

        for d in self.ious:
            for k, v in d.items():
                shape_mious[k] = shape_mious[k] + v

        all_shape_mious = torch.stack([miou for mious in shape_mious.values() for miou in mious])
        cat_mious = {k: torch.stack(v).mean() for k, v in shape_mious.items() if len(v) > 0}

        # instance (total) mIoU
        self.log("val_ins_miou", all_shape_mious.mean().to('cuda'), sync_dist=True, batch_size=self.val_bsz)
        # mIoU averaged over categories
        self.log("val_cat_miou", torch.stack(list(cat_mious.values())).mean().to('cuda'), sync_dist=True, batch_size=self.val_bsz)
        for cat in sorted(cat_mious.keys()):
            self.log(f"val_cat_miou_{cat}", cat_mious[cat].to('cuda'), sync_dist=True, batch_size=self.val_bsz)
