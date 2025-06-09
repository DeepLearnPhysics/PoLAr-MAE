import torch
from omegaconf import OmegaConf
import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
import torch
import os

torch.set_float32_matmul_precision("high")
getenv = lambda x: os.environ[x]
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("getenv", getenv)

from polarmae.datasets import PILArNetDataModule
from polarmae.models.ssl import PoLArMAE

if __name__ == "__main__":
    cli = LightningCLI(
        PoLArMAE,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 4,
            "precision": "32",
            "max_epochs": -1,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 200,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(every_n_train_steps=10000, save_last=True),
                ModelCheckpoint(
                    filename="{epoch}-{step}-{val:.3f}",
                    monitor="loss/val",
                ),
                ModelCheckpoint(
                    monitor="svm_val_class_f1_larnet_postnorm_macro",
                    mode="max",
                    filename="{epoch}-{step}-{svm_val_class_f1_larnet_postnorm_macro:.4f}",
                ),
                ModelCheckpoint(
                    monitor="svm_val_class_f1_larnet_no_postnorm_macro",
                    mode="max",
                    filename="{epoch}-{step}-{svm_val_class_f1_larnet_no_postnorm_macro:.4f}",
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

