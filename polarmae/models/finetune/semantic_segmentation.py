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
from polarmae.loss import SoftmaxFocalLoss, DiceLoss, TotalVariationLoss
from polarmae.models.finetune.base import FinetuneModel
from polarmae.utils.pylogger import RankedLogger
from polarmae.layers.group_local_attention import GroupLocalAttention
from math import sqrt
import wandb

log = RankedLogger(__name__, rank_zero_only=True)


class SemanticSegmentation(FinetuneModel):
    def __init__(
        self,
        encoder: TransformerEncoder,
        seg_decoder: Optional[TransformerDecoder],
        num_classes: int,
        class_temps: Optional[List[float]] = None,
        seg_head_fetch_layers: List[int] = [3, 7, 11],
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
        lr_scheduler_type: Literal['cosine', 'linear'] = 'cosine',
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
        # deprecated args used in past training runs
        use_pos_enc_for_upsampling: bool = False,
        apply_local_attention: bool = False,
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
        self.class_temps = class_temps
        if self.class_temps is not None:
            assert len(self.class_temps) == num_classes, "class_temps must have the same number of elements as num_classes"
            self.class_temps = nn.Parameter(torch.tensor(self.class_temps, device=self.device), requires_grad=False)
            log.info(f"🌡️  Using class temps: tau = {self.class_temps} for {num_classes} classes")

        if use_token_local_attention:
            self.gla = GroupLocalAttention(
                embed_dim=(
                    encoder.transformer.embed_dim // 6
                    if not condition_global_features
                    else encoder.transformer.embed_dim // 6
                ),
                num_heads=token_local_attention_heads,
                proj_drop_rate=token_local_attention_drop_rate,
                attn_drop_rate=token_local_attention_attn_drop_rate,
                qkv_bias=True,
                use_flash_attn=True,  # not good for yuge batch sizes
            )
            self.gla_downcast = nn.Linear(
                encoder.transformer.embed_dim, encoder.transformer.embed_dim // 6
            )
            # self.gla_upcast = nn.Linear(
            #     encoder.transformer.embed_dim // 6, encoder.transformer.embed_dim
            # )
            self.upsampler = PointNetFeatureUpsampling(
                in_channel=encoder.transformer.embed_dim // 6,
                mlp=[encoder.transformer.embed_dim // 6],
                # K=5,
            )
        else:
            self.upsampler = PointNetFeatureUpsampling(
                in_channel=encoder.transformer.embed_dim,
                mlp=[encoder.transformer.embed_dim, encoder.transformer.embed_dim],
                # K=5,
            )

        self.seg_head = SegmentationHead(
            self.encoder.embed_dim // 6 if (self.seg_decoder is None and self.condition_global_features and not use_token_local_attention) else 0,
            0,  # event-wide label embedding -- 0 for polarmae!
            encoder.transformer.embed_dim // 6,
            seg_head_dim,
            seg_head_dropout,
            num_classes,
        )
        if self.seg_decoder is not None:
            log.info(
                "Using decoder, so not aggregating token features (i.e. `seg_head_fetch_layers`) for use in seg head."
            )

        if self.hparams.loss_func == "nll":
            self.loss_func = nn.NLLLoss(
                weight=torch.ones(self.hparams.num_classes, device=self.device),
                reduction="mean",
                ignore_index=-1,
            )
        elif self.hparams.loss_func == 'focal':
            self.loss_func = self.focal_loss = SoftmaxFocalLoss(
                weight=torch.ones(self.hparams.num_classes, device=self.device),
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
                weight=torch.ones(self.hparams.num_classes, device=self.device),
                reduction="mean",
                ignore_index=-1,
                gamma=2,
            )
            self.loss_func = lambda logits, labels: 0.05 * self.dice_loss(logits, labels) + self.focal_loss(logits, labels)
        else:
            raise ValueError(f"Unknown loss function: {self.hparams.loss_func}")

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.ones(self.hparams.num_classes, device=self.device),
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

        for loss in ["dice_loss", "focal_loss", "loss_func"]:
            if hasattr(self, loss) and hasattr(getattr(self, loss), 'weight') and getattr(self, loss).weight is not None:
                getattr(self, loss).weight.copy_(self.trainer.datamodule.class_weights)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.ones(self.hparams.num_classes, device=self.device),
            reduction="mean",
            ignore_index=-1,
        )
        self.ce_loss.weight.copy_(self.trainer.datamodule.class_weights)

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
            self.encoder.freeze()
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

        out = self.encoder.prepare_tokens(points, lengths, ids=labels)
        output = self.encoder(
            out["x"],
            out["pos_embed"],
            out["emb_mask"],
            return_hidden_states=self.seg_decoder is None,
            final_norm=self.hparams.apply_encoder_postnorm,
        )
        batch_lengths = out['emb_mask'].sum(dim=1)

        if self.seg_decoder is not None:
            output = self.seg_decoder(output.last_hidden_state, out['pos_embed'], out['emb_mask'])
            token_features = output.last_hidden_state
        else:
            # fetch intermediate layers & get averaged token features
            if len(self.hparams.seg_head_fetch_layers) == 0:
                token_features = output.last_hidden_state
            else:
                token_features = self.encoder.combine_intermediate_layers(
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

        # Create point mask
        point_mask = torch.arange(lengths.max(), device=lengths.device).expand(
            len(lengths), -1
        ) < lengths.unsqueeze(-1)

        if self.hparams.use_token_local_attention:
            # Use encoder's position embedding for points
            # point_positions = self.encoder.pos_embed(points)
            upscaled_feats, idx = self.upsampler(
                points[..., :3],
                out["centers"][:, :, :3],
                points[..., :3],
                self.gla_downcast(token_features),
                lengths,
                batch_lengths,
                point_mask,
            ) # (B, N, C // 6)

            # Apply TokenLocalAttention
            if self.condition_global_features:
                B, N, C = token_features.shape
                context_features = self.gla_downcast(global_feature.reshape(-1, C)).reshape(B, 2, -1) # (B, 2, C // 6)
            else:
                context_features = None
            point_features = self.gla(
                upscaled_feats=upscaled_feats,
                grouping_idx=out["grouping_idx"],
                grouping_point_mask=out["point_mask"],
                context_tokens=context_features,
            ) # (B, N, C // 6)

            # upcast token features
            # point_features = self.gla_upcast(point_features) # (B, N, C)
            x = point_features
        elif self.seg_decoder is None and self.condition_global_features:
            B, N, C = points.shape
            global_feature = global_feature.reshape(B, -1) # (B, 2*C')
            x = torch.cat(
                [x, global_feature.unsqueeze(-1).expand(-1, -1, N).transpose(1, 2)], dim=-1
            )  # (B, N, 2*C')

        x = self.seg_head(x.transpose(1, 2), point_mask).transpose(1, 2) # (B, N, num_classes)

        if class_slice is not None:
            x = x[..., class_slice] # (B, N, num_classes_to_keep)

        if self.class_temps is not None:
            x = x / self.class_temps[None, None, :]

        return {
            'x': x if return_logits else F.log_softmax(x, dim=-1),
            'logits': x if return_logits else None,
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
        return_logits: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        out = self.forward(
            points,
            lengths,
            labels,
            class_mask,
            return_logits=self.hparams.loss_func == 'fancy' or return_logits,
        )

        loss = self.loss_func(
            out['x'][out['point_mask']],
            labels.squeeze(-1)[out['point_mask']]
        )
        with torch.no_grad():
            ce_loss = self.ce_loss(
                out['x'][out['point_mask']],
                labels.squeeze(-1)[out['point_mask']]
            )

        loss_dict = {
            self.hparams.loss_func: loss,
            'ce': ce_loss,
            'tv': self.tv_loss(points, lengths, out['x']),
        }
        output_dict = { 
            'logits': out['logits'],
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
        for metric in ['macc', 'mprecision', 'acc', 'precision', 'f1_score', 'f1_score_m']:
            self.log(f'train_{metric}', getattr(self, f'train_{metric}')(pred, labels).to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)
        ce_loss = loss_dict['ce']

        # Compute per-class metrics during training (optional, can be slower)
        if self.current_epoch % 5 == 0:  # Compute every 5 epochs to avoid slowing down training
            f1_perclass = self.train_f1_score_perclass(pred, labels)
            acc_perclass = self.train_acc_perclass(pred, labels)
            precision_perclass = self.train_precision_perclass(pred, labels)
            
            # Log per-class metrics
            for cls_idx in range(self.hparams.num_classes):
                if cls_idx < len(f1_perclass):
                    cls_name = self.seg_class_to_category.get(cls_idx, f"class_{cls_idx}")
                    self.log(f'train_f1_score_{cls_name}', f1_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)
                    self.log(f'train_acc_{cls_name}', acc_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)
                    self.log(f'train_precision_{cls_name}', precision_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)

        self.log('train_ce', ce_loss.to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)
        loss = loss_dict[self.hparams.loss_func]# + 1e-3 * loss_dict['tv']
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict, output_dict = self.compute_loss(
            self.val_transformations(batch['points']),
            batch['lengths'],
            batch['semantic_id'],
            return_logits=True,
        )

        # visualize if batch_idx == 0. let's hope that the dataloader doesn't shuffle the data between epochs
        if batch_idx == 0:
            for i in range(5):
                truth, pred, pred_weighted = colored_pointcloud_batch(output_dict, batch, batch_idx=i)
                truth[..., 3:6] *= 255
                pred[..., 3:6] *= 255
                pred_weighted[..., 3:6] *= 255
                self.logger.experiment.log({
                    f"preds_{i}": [wandb.Object3D(truth, caption=f"Batch {batch_idx}, Instance {i}"), wandb.Object3D(pred, caption=f"Batch {batch_idx}, Instance {i}"), wandb.Object3D(pred_weighted, caption=f"Batch {batch_idx}, Instance {i}")]
                })

        bsz = batch['points'].shape[0]
        self.val_bsz = bsz
        self.log_losses(loss_dict, prefix='val_', batch_size=bsz)
        pred, labels = output_dict['pred'], output_dict['labels'].squeeze(-1)
        for metric in ['macc', 'mprecision', 'acc', 'precision', 'f1_score', 'f1_score_m']:
            self.log(f'val_{metric}', getattr(self, f'val_{metric}')(pred, labels).to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz, prog_bar=True)
        ce_loss = loss_dict['ce']
        self.log('val_ce', ce_loss.to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)

        # Compute per-class metrics
        f1_perclass = self.val_f1_score_perclass(pred, labels)
        acc_perclass = self.val_acc_perclass(pred, labels)
        precision_perclass = self.val_precision_perclass(pred, labels)
        
        # Log per-class metrics
        for cls_idx in range(self.hparams.num_classes):
            if cls_idx < len(f1_perclass):
                cls_name = self.seg_class_to_category.get(cls_idx, f"class_{cls_idx}")
                self.log(f'val_f1_score_{cls_name}', f1_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz, prog_bar=True)
                self.log(f'val_acc_{cls_name}', acc_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)
                self.log(f'val_precision_{cls_name}', precision_perclass[cls_idx].to('cuda'), on_epoch=True, sync_dist=True, batch_size=bsz)

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

        if hasattr(self, 'token_local_attention') and self.token_local_attention is not None:
            self.logger.experiment.log({"val_localattn_gamma": wandb.Histogram(self.token_local_attention.ls.gamma.detach().cpu().numpy())})

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
