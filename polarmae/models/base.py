from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from polarmae.layers.encoder import TransformerEncoder, TransformerOutput
from polarmae.utils import transforms
from polarmae.utils.checkpoint import extract_model_checkpoint
from polarmae.utils.scheduler import LinearWarmupCosineAnnealingLR
from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ['BaseModel']

class BaseModel(pl.LightningModule):
    """
    Base class for all models.

    Does some housekeeping that is common to all models. This includes:
    - setting up point cloud augmentations
    - setting up optimizers and learning rate scheduler
    - loading pretrained checkpoints
    - logging of a loss dictionary
    """
    def __init__(self):
        super().__init__()

    def configure_transformations(self):
        """
        Configure transformations for training and validation.
        """
        if not hasattr(self.hparams, 'train_transformations'):
            raise ValueError("train_transformations not set")
        if not hasattr(self.hparams, 'val_transformations'):
            raise ValueError("val_transformations not set")

        train_transformations = self.hparams.train_transformations
        val_transformations = self.hparams.val_transformations
        def build_transformation(name: str) -> transforms.Transform:
            if name == "center_and_scale":
                return transforms.PointcloudCenterAndNormalize(
                    center=self.hparams.transformation_center,
                    scale_factor=self.hparams.transformation_scale_factor
                )
            elif name == "scale":
                return transforms.PointcloudScaling(
                    min=self.hparams.transformation_scale_min, max=self.hparams.transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=self.hparams.transformation_rotate_dims, deg=self.hparams.transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(self.hparams.transformation_translate)
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    self.hparams.transformation_height_normalize_dim
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

    def configure_optimizers(self):
        assert self.trainer is not None

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # type: ignore
        )

        warmup_epochs = self.hparams.lr_scheduler_linear_warmup_epochs
        max_epochs = self.trainer.max_epochs
        if self.hparams.lr_scheduler_stepping == 'step': # iters probably given
            steps_per_epoch = self.trainer.num_training_batches
            max_epochs = self.trainer.max_steps if self.trainer.max_steps is not None else self.trainer.max_epochs * steps_per_epoch

        for name, val in zip(['max_iters', 'warmup_iters'], [max_epochs, warmup_epochs]):
            log.info(f'{name}: {val}')

        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=warmup_epochs, # iters if step, epochs otherwise
            max_epochs=max_epochs,       # iters if step, epochs otherwise
            warmup_start_lr=self.hparams.lr_scheduler_linear_warmup_start_lr,  # type: ignore
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,  # type: ignore
        )

        if self.hparams.lr_scheduler_stepping == 'step':
            sched = {
                "scheduler": sched,
                "interval": "step",
            }

        return [opt], [sched]

    def load_pretrained_checkpoint(self, path: str) -> None:
            log.info(f"Loading pretrained checkpoint from '{path}'.")

            checkpoint = extract_model_checkpoint(path)

            missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
            log.warning(f"Missing keys: {missing_keys}")
            log.warning(f"Unexpected keys: {unexpected_keys}")

    def log_losses(self,
                   loss_dict: Dict[str, torch.Tensor],
                   prefix: Optional[str] = "",
                   postfix: Optional[str] = "",
                   **kwargs
                   ) -> None:
        for name in loss_dict.keys():
            l = loss_dict[name]
            if hasattr(self.hparams, 'loss_weights'):
                l *= self.hparams.loss_weights.get(name, 1.0)
            if l == 0:
                continue
            full_name = f'{prefix}{name}{postfix}'
            self.log(full_name, l, sync_dist=True, on_step='train' in full_name, on_epoch=True, **kwargs)

    def forward(
            self,
            points: torch.Tensor,
            lengths: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            endpoints: Optional[torch.Tensor] = None,
    ) -> TransformerOutput:
        assert hasattr(self, 'encoder')
        self.encoder: TransformerEncoder
        out = self.encoder.prepare_tokens_with_masks(points, lengths, ids=labels, endpoints=endpoints)
        output = self.encoder.transformer(out['x'], out['pos_embed'], out['emb_mask'])
        return output