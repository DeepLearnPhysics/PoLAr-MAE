from typing import Any, Dict, List, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision, F1Score

from polarmae.eval.segmentation import compute_shape_ious
from polarmae.eval.visualization import colored_pointcloud_batch
from polarmae.layers.encoder import TransformerEncoder
from polarmae.layers.decoder import TransformerDecoder
from polarmae.layers.feature_upsampling import PointNetFeatureUpsampling
from polarmae.layers.masking import masked_max, masked_mean
from polarmae.layers.seg_head import SegmentationHead
from polarmae.layers.pointnet import MaskedMiniPointNet
from polarmae.loss import SoftmaxFocalLoss, DiceLoss, TotalVariationLoss
from polarmae.models.finetune.base import FinetuneModel
from polarmae.utils.pylogger import RankedLogger
from polarmae.layers.seg_head import IntermediateFusion
from polarmae.layers.group_local_attention import GroupLocalAttention
from math import sqrt
import wandb

log = RankedLogger(__name__, rank_zero_only=True)


class SemanticSegmentationEqPN(FinetuneModel):
    def __init__(
        self,
        encoder: TransformerEncoder,
        num_classes: int,
        class_temps: Optional[List[float]] = None,
        seg_head_fetch_layers: List[int] = [3, 7, 11],
        seg_head_combination_method: Literal['concat', 'mean'] = 'mean',
        seg_head_dim: int = 384,
        seg_head_dropout: float = 0.5,
        condition_global_features: bool = False,
        apply_encoder_postnorm: bool = True,
        # New parameters
        use_token_local_attention: bool = False,
        token_local_attention_heads: int = 8,
        token_local_attention_attn_drop_rate: float = 0.0,
        token_local_attention_drop_rate: float = 0.0,
        # LR/optimizer
        learning_rate: float = 1e-3,
        lr_scheduler_type: Literal['cosine', 'linear', 'wsd'] = 'cosine',
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
        start_lr_decay: bool = False,
        # deprecated args used in past training runs
        use_pos_enc_for_upsampling: bool = False,
        apply_local_attention: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        super().configure_transformations()

        self.encoder = encoder

        self.loss_func = loss_func
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.encoder_freeze = encoder_freeze

        self.condition_global_features = condition_global_features
        self.class_temps = class_temps
        if self.class_temps is not None:
            assert len(self.class_temps) == num_classes, "class_temps must have the same number of elements as num_classes"
            self.class_temps = nn.Parameter(torch.tensor(self.class_temps, device=self.device), requires_grad=False)
            log.info(f"🌡️  Using class temps: tau = {self.class_temps} for {num_classes} classes")


        self.equivariant_patch_encoder = MaskedMiniPointNet(
            channels=4,
            feature_dim=encoder.embed_dim // 4,
            equivariant=True,
            hidden_dim1=128,
            hidden_dim2=128,
        )
        self.seg_head = nn.Linear(
            encoder.embed_dim + encoder.embed_dim // 4,  # 384 + 64
            num_classes * encoder.tokenizer.grouping.group_max_points,
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
        self.train_f1_score = F1Score("multiclass", **metric_kwargs)
        self.train_f1_score_m = F1Score("multiclass", **metric_kwargs, average="macro")
        self.train_f1_score_perclass = F1Score("multiclass", **metric_kwargs, average=None)
        self.train_acc_perclass = Accuracy('multiclass', **metric_kwargs, average=None)
        self.train_precision_perclass = Precision("multiclass", **metric_kwargs, average=None)

        self.val_acc = Accuracy('multiclass', **metric_kwargs)
        self.val_macc = Accuracy('multiclass', **metric_kwargs, average="macro")
        self.val_precision = Precision("multiclass", **metric_kwargs)
        self.val_mprecision = Precision("multiclass", **metric_kwargs, average="macro")

        self.val_f1_score = F1Score("multiclass", **metric_kwargs)
        self.val_f1_score_m = F1Score("multiclass", **metric_kwargs, average="macro")
        self.val_f1_score_perclass = F1Score("multiclass", **metric_kwargs, average=None)
        self.val_acc_perclass = Accuracy('multiclass', **metric_kwargs, average=None)
        self.val_precision_perclass = Precision("multiclass", **metric_kwargs, average=None)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                # logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")
                logger.experiment.define_metric("val_ins_miou", summary="last,max")
                logger.experiment.define_metric("val_cat_miou", summary="last,max")
                logger.experiment.define_metric("val_precision", summary="last,max")
                logger.experiment.define_metric("val_mprecision", summary="last,max")
                logger.experiment.define_metric("val_f1_score", summary="last,max")
                logger.experiment.define_metric("val_f1_score_m", summary="last,max")
                for cls_idx in range(self.hparams.num_classes):
                    cls_name = self.seg_class_to_category.get(cls_idx, f"class_{cls_idx}")
                    logger.experiment.define_metric(f"val_f1_score_{cls_name}", summary="last,max")
                    logger.experiment.define_metric(f"val_acc_{cls_name}", summary="last,max")
                    logger.experiment.define_metric(f"val_precision_{cls_name}", summary="last,max")

        """ ------------------------------------------------------------------------ """
        """                                  losses                                  """
        """ ------------------------------------------------------------------------ """
        if self.hparams.loss_func == "nll":
            self.loss_func = nn.NLLLoss(
                weight=self.trainer.datamodule.class_weights,
                reduction="mean",
                ignore_index=-1,
            )
        elif self.hparams.loss_func == 'focal':
            self.loss_func = self.focal_loss = SoftmaxFocalLoss(
                weight=self.trainer.datamodule.class_weights,
                reduction="mean",
                ignore_index=-1,
                gamma=2,
            )
        elif self.hparams.loss_func == 'fancy':
            self.dice_loss = DiceLoss(
                smooth=1,
                ignore_index=-1,
            )
            self.focal_loss = SoftmaxFocalLoss(
                weight=self.trainer.datamodule.class_weights,
                reduction="mean",
                ignore_index=-1,
                gamma=2,
            )
            self.loss_func = lambda logits, labels: 0.05 * self.dice_loss(logits, labels) + self.focal_loss(logits, labels)
        else:
            raise ValueError(f"Unknown loss function: {self.hparams.loss_func}")

        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.trainer.datamodule.class_weights,
            reduction="mean",
            ignore_index=-1,
        )

        group_radius = self.encoder.tokenizer.grouping.group_radius
        self.tv_loss = TotalVariationLoss(
            radius=2 * group_radius,
            K=10,
            reduction="mean",
            apply_to_argmax=True,
        )

        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.trainer.datamodule.class_weights,
            reduction="mean",
            ignore_index=-1,
        )

        """ ------------------------------------------------------------------------ """
        if self.hparams.pretrained_ckpt_path is not None:
            super().load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)
            log.info('🔥  Loaded pretrained checkpoint.')
        else:
            log.info('🔥  No pretrained checkpoint loaded. Training from scratch??')

        self.encoder.freeze(self.hparams.encoder_freeze)
        log.info(f'🔥  {"Freezing" if self.hparams.encoder_freeze else "Unfreezing"} encoder.')

    def forward(
            self, 
            points: torch.Tensor,
            lengths: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            class_slice: Optional[slice] = None,
            return_logprobs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # run encoder as usual
        assert len(points.shape) == 3, f"points must be of shape (B, N, C), got {points.shape}"
        assert points.shape[1] == lengths.max(), f"points and lengths must have the same number of points, got {points.shape[1]} and {lengths.max()}"
        assert len(lengths.shape) == 1, f"lengths must be of shape (B), got {lengths.shape}"
        if labels is not None:
            assert labels.shape[:2] == points.shape[:2], f"labels must be of shape (B, N) and same as points, got {labels.shape} but points.shape is {points.shape}"

        out = self.encoder.prepare_tokens(points, lengths, ids=labels)
        output = self.encoder(
            out["x"],
            out["pos_embed"],
            out["emb_mask"],
            return_hidden_states=self.hparams.seg_head_fetch_layers != [],
            final_norm=self.hparams.apply_encoder_postnorm,
        )

        token_features = output.last_hidden_state # (B, T, C)
        if self.hparams.seg_head_fetch_layers != []: # fetch intermediate layers & get averaged token features
            token_features = self.combine_intermediate_layers(
                output.hidden_states,
                out['emb_mask'],
                self.hparams.seg_head_fetch_layers,
            ) # (B, T, C)

        # goal:
        # - run equivariant patch encoder on each group
        # - concatenate equivariant patch encoder output with token features for each group
        # - run logit decoder to get logits for each individual point in each group (B, G, N_group_max, num_classes)
        # - average logits over groups using GroupLocalAttention.average_groups()
        groups = out["groups"]  # (B, G, N_group_max, 4)
        point_mask = torch.arange(lengths.max(), device=lengths.device).expand(
            len(lengths), -1
        ) < lengths.unsqueeze(-1)  # (B, N_event_max)
        groups_point_mask = out["groups_point_mask"]  # (B, G, N_group_max)
        with torch.amp.autocast(device_type=token_features.device.type, dtype=torch.float32):
            # do same thing as in prepare_tokens() of tokenizer
            flattened_structure_tokens = self.equivariant_patch_encoder(
                groups[out['emb_mask']], 
                groups_point_mask[out['emb_mask']].unsqueeze(1)
            ) # (-1, C // 4)
            structure_tokens = torch.zeros(
                groups.shape[0],
                groups.shape[1],
                self.equivariant_patch_encoder.feature_dim,
                device=flattened_structure_tokens.device,
                dtype=flattened_structure_tokens.dtype,
            )
            structure_tokens[out['emb_mask']] = flattened_structure_tokens

            # now we have structure tokens (B, G, C // 4) and token features (B, T, C). let's concatenate them
            decoder_input = torch.cat(
                [structure_tokens, token_features], dim=-1
            ) # (B, G, C + C // 4)

            # now we have decoder input (B*G, N_group_max, 384 + 96). let's run the decoder
            decoded_seg_logits = self.seg_head(
                decoder_input
            ) # (B, G, N_group_max*num_classes)
            decoded_seg_logits = decoded_seg_logits.reshape(
                decoded_seg_logits.shape[0],
                decoded_seg_logits.shape[1],
                self.encoder.tokenizer.grouping.group_max_points,
                self.hparams.num_classes,
            ) # (B, G, N_group_max, num_classes)

            # now, we average over groups using GroupLocalAttention.average_groups()
            x = GroupLocalAttention.average_groups(
                decoded_seg_logits,
                out["grouping_idx"],
                points,
                groups_point_mask,
                fill_with_original_points=True,
            ) # (B, N_event_max, num_classes) <-- logits for each point in an event!

        return {
            'x': F.log_softmax(x, dim=-1) if return_logprobs else x,
            'logits': x,
            # 'idx': idx,
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
        return_logprobs: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        out = self.forward(
            points,
            lengths,
            labels,
            class_mask,
            return_logprobs=self.hparams.loss_func == 'nll' or return_logprobs,
        )
        output_dict = { 
            'logits': out['logits'],
            'pred': out['id_pred'],
            'labels': labels,
        }

        loss = self.loss_func(out['x'][out['point_mask']], labels.squeeze(-1)[out['point_mask']]) # (B,)
        with torch.no_grad():
            ce_loss = self.ce_loss(out['x'][out['point_mask']], labels.squeeze(-1)[out['point_mask']]) # (B,)
            tv_loss = self.tv_loss(points, lengths, out['x'])
        loss_dict = {self.hparams.loss_func: loss, 'ce': ce_loss, 'tv': tv_loss}
        return loss_dict, output_dict

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict, output_dict = self.compute_loss(
            self.train_transformations(batch['points']),
            batch['lengths'],
            batch['semantic_id'],
            return_logprobs=self.hparams.loss_func == 'nll',
        )
        super().log_losses(loss_dict, prefix='train', batch_size=batch['points'].shape[0])
        self.log_metrics(output_dict['pred'], output_dict['labels'].squeeze(-1), loss_dict, prefix='train', batch_size=batch['points'].shape[0])
        return loss_dict[self.hparams.loss_func]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict, output_dict = self.compute_loss(
            self.val_transformations(batch['points']),
            batch['lengths'],
            batch['semantic_id'],
            return_logprobs=self.hparams.loss_func == 'nll',
        )
        super().log_losses(loss_dict, prefix='val', batch_size=batch['points'].shape[0])
        self.log_pointcloud(output_dict, batch, batch_idx)
        self.log_metrics(output_dict['pred'], output_dict['labels'].squeeze(-1), loss_dict, prefix='val', batch_size=batch['points'].shape[0])
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

    def log_metrics(self, pred: torch.Tensor, labels: torch.Tensor, loss_dict: Dict[str, torch.Tensor], prefix: str = 'train') -> None:
        assert prefix in ['train', 'val'], "prefix must be either 'train' or 'val'"

        # macro metrics
        for metric in ['macc', 'mprecision', 'acc', 'precision', 'f1_score', 'f1_score_m']:
            self.log(f'{prefix}_{metric}', getattr(self, f'{prefix}{metric}')(pred, labels).to('cuda'), on_epoch=True, sync_dist=True, batch_size=self.val_bsz, prog_bar=prefix=='val')
        ce_loss = loss_dict['ce']
        self.log(f'{prefix}_ce', ce_loss.to('cuda'), on_epoch=True, sync_dist=True, batch_size=self.val_bsz)

        # per-class metrics
        f1_perclass = getattr(self, f'{prefix}_f1_score_perclass')(pred, labels)
        acc_perclass = getattr(self, f'{prefix}_acc_perclass')(pred, labels)
        precision_perclass = getattr(self, f'{prefix}_precision_perclass')(pred, labels)
        for cls_idx in range(self.hparams.num_classes):
            if cls_idx < len(f1_perclass):
                cls_name = self.seg_class_to_category.get(cls_idx, f"class_{cls_idx}")
                self.log(f'{prefix}_f1_score_{cls_name}', f1_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=self.val_bsz, prog_bar=prefix=='val')
                self.log(f'{prefix}_acc_{cls_name}', acc_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=self.val_bsz)
                self.log(f'{prefix}_precision_{cls_name}', precision_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=self.val_bsz)

    def log_pointcloud(self, output_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if batch_idx != 0:
            return # only log first batch
        for i in range(5):
            truth, pred, pred_weighted = colored_pointcloud_batch(output_dict, batch, batch_idx=i)
            self.logger.experiment.log({
                f"preds_{i}": [
                    wandb.Object3D(truth, caption=f"Batch {batch_idx}, Instance {i}"),
                    wandb.Object3D(pred, caption=f"Batch {batch_idx}, Instance {i}"),
                    wandb.Object3D(pred_weighted, caption=f"Batch {batch_idx}, Instance {i}"),
                ]
            })
