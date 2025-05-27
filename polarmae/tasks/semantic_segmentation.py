import polarmae.utils.pytorch_monkey_patch

from omegaconf import OmegaConf
import pytorch_lightning
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

OmegaConf.register_new_resolver("eval", eval)

import torch
torch.set_float32_matmul_precision("high")

from polarmae.datasets import PILArNetDataModule
from polarmae.models.finetune import SemanticSegmentation

if __name__ == "__main__":
    cli = LightningCLI(
        SemanticSegmentation,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": "bf16-mixed",
            "log_every_n_steps": 10,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True, save_last=True),
                ModelCheckpoint(
                    monitor="val_f1_score_m",
                    mode="max",
                    filename="{epoch}-{step}-f1{val_f1_score_m:.4f}",
                ),
                ModelCheckpoint(
                    monitor="val_ce",
                    mode="min",
                    filename="{epoch}-{step}-ce{val_ce:.4f}",
                ),
            ],
        },
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=123,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )

