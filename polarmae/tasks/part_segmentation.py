from omegaconf import OmegaConf
import pytorch_lightning
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

OmegaConf.register_new_resolver("eval", eval)

import torch
from polarmae.datasets import PILArNetDataModule
from polarmae.models.finetune.part_segmentation import PartSegmentation

if __name__ == "__main__":
    cli = LightningCLI(
        PartSegmentation,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": "bf16-mixed",
            "log_every_n_steps": 10,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True),
                ModelCheckpoint(
                    monitor="val_cat_miou",
                    mode="max",
                    filename="{epoch}-{step}-{val_cat_miou:.4f}",
                ),
            ],
        },
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=123,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )

