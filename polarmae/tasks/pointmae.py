import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

OmegaConf.register_new_resolver("eval", eval)

from polarmae.datasets import PILArNetDataModule
from polarmae.models.ssl.pointmae import PointMAE

import pytorch_lightning

if __name__ == "__main__":
    cli = LightningCLI(
        PointMAE,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 4,
            "precision": "bf16-mixed",
            "max_epochs": 800,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 200,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True),
                ModelCheckpoint(
                    filename="{epoch}-{step}-{val:.3f}",
                    monitor="loss/val",
                ),
                ModelCheckpoint(
                    monitor="svm_val_acc_larnet",
                    mode="max",
                    filename="{epoch}-{step}-{svm_val_acc_larnet:.4f}",
                ),
                # ModelCheckpoint(
                #     save_top_k=4,
                #     monitor="epoch", # checked every `check_val_every_n_epoch` epochss
                #     mode="max",
                #     filename="{epoch}-{step}-intermediate",
                # ),
            ],
            # 'profiler': 'advanced'
        },
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=123,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )

