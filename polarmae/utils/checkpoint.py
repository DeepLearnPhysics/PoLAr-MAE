import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
import importlib
from typing import Type
from polarmae.utils.pylogger import RankedLogger
import pytorch_lightning as pl

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ['extract_model_checkpoint', 'load_finetune_checkpoint']

def extract_model_checkpoint(path: str):
    checkpoint = torch.load(path)

    if "state_dict" in checkpoint:
        # lightning checkpoint
        checkpoint = {
            k.replace("student.", "encoder.").replace(
                "pos_embedding.", "positional_encoding."
            ): v
            for k, v in checkpoint["state_dict"].items()
        }
        for k in list(checkpoint.keys()):
            if k.startswith("teacher."):
                del checkpoint[k]
            elif k.startswith("regressor."):
                del checkpoint[k]
            elif k.startswith("decoder."):
                del checkpoint[k]
            elif k == "mask_token":
                del checkpoint[k]
            elif k.startswith('diffusion'):
                del checkpoint[k]
            elif k.startswith('var_scheduler.'):
                del checkpoint[k]
            elif k.startswith('time_encoding.'):
                del checkpoint[k]
    elif "base_model" in checkpoint:
        # Point-MAE or Point-BERT or DiffPMAE
        if (
            "transformer_q.cls_token" in checkpoint["base_model"]
        ):  # Point-BERT pretrained
            checkpoint["base_model"] = {
                k.replace("transformer_q.", "module."): v
                for k, v in checkpoint["base_model"].items()
            }
            for k in list(checkpoint["base_model"].keys()):
                if not k.startswith("module."):
                    del checkpoint["base_model"][k]

        checkpoint = {
            k
            # Point-MAE
            .replace("module.MAE_encoder.blocks.", "encoder.")
            .replace("module.MAE_encoder.norm.", "encoder.norm.")
            .replace("module.MAE_encoder.pos_embed.", "positional_encoding.")
            .replace("module.MAE_encoder.encoder.", "tokenizer.embedding.")
            # Point-BERT
            .replace("module.encoder.", "tokenizer.embedding.")
            .replace("module.reduce_dim.", "tokenizer_to_encoder.")
            .replace("module.blocks.", "encoder.")
            .replace("module.norm.", "encoder.norm.")
            .replace("module.pos_embed.", "positional_encoding.")
            .replace("module.cls_token", "cls_token")
            .replace("module.cls_pos", "cls_pos")
            .replace("module.cls_head_finetune", "cls_head")
            # finally
            .replace("module.", ""): v
            for k, v in checkpoint["base_model"].items()
        }
        for k in list(checkpoint.keys()):
            if k.startswith("MAE_decoder."):
                del checkpoint[k]
            elif k == "mask_token":
                del checkpoint[k]
            elif k.startswith("decoder_pos_embed."):
                del checkpoint[k]
            elif k.startswith("increase_dim."):
                del checkpoint[k]
            elif k in ["cls_token", "cls_pos"]:
                checkpoint[k] = checkpoint[k].squeeze(0).squeeze(0)
            elif k.startswith("lm_head."):
                del checkpoint[k]
            elif k.startswith('diffusion'):
                del checkpoint[k]
            elif k.startswith('var'):
                del checkpoint[k]
    else:
        raise RuntimeError("Unknown checkpoint format")

    return checkpoint


def _recursive_resolve(cfg):
    """
    Recursively traverse cfg and instantiate any dictionary that
    contains a "class_path" key.
    """
    # If it's a DictConfig or a plain dict, process its items.
    if isinstance(cfg, (dict, DictConfig)):
        # First, recursively process children.
        instantiated = {}
        for key, value in cfg.items():
            instantiated[key] = _recursive_resolve(value)

        # If this dictionary is meant to be instantiated, do so.
        if "class_path" in instantiated:
            # Optionally, you can merge extra keys outside of "init_args" here if needed.
            init_args = instantiated.get("init_args", {})
            # Make sure the init_args themselves have been recursively instantiated.
            if isinstance(init_args, (dict, DictConfig)):
                init_args = _recursive_resolve(init_args)

            # Dynamically import and instantiate.
            class_path = instantiated["class_path"]
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(**init_args)
        else:
            return instantiated

    # If it's a list, apply the function to each element.
    elif isinstance(cfg, list):
        return [_recursive_resolve(item) for item in cfg]

    elif isinstance(cfg, ListConfig):
        return list(cfg)

    # Otherwise, return the value as is.
    else:
        return cfg

def parse_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))  # Use with caution!
    return _recursive_resolve(cfg)

def load_finetune_checkpoint(
    cls,
    checkpoint_path: str,
    data_path: str = None,
    pretrained_ckpt_path: str = None,
    load_state_dict: bool = True,
):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    hparams = _recursive_resolve(hparams)
    hparams.pop("_instantiator")

    datamodule_hparams = ckpt["datamodule_hyper_parameters"]
    datamodule_cls_path = datamodule_hparams["_class_path"]
    datamodule_name, class_name = datamodule_cls_path.rsplit(".", 1)
    
    # fix for old checkpoints
    if datamodule_name == 'larnet.datasets':
        datamodule_name = 'polarmae.datasets'
    if class_name == 'LArNetDataModule':
        class_name = 'PILArNetDataModule'

    datamodule = importlib.import_module(datamodule_name)
    datamodule_cls = getattr(datamodule, class_name)

    datamodule_hparams.pop("_class_path")
    datamodule_hparams.pop("_instantiator")

    if data_path is not None:
        datamodule_hparams["data_path"] = data_path
    else:
        log.warning(
            f"No data path provided, using data path {datamodule_hparams['data_path']}"
        )

    datamodule = datamodule_cls(**datamodule_hparams)
    datamodule.setup()
    model = cls(**hparams)
    trainer = pl.Trainer()
    model.trainer = trainer
    trainer.datamodule = datamodule

    model.setup()

    if load_state_dict:
        log.info(
            "Loading state dict. There should be no missing or unexpected keys below here."
        )
        missed, unexpected = model.load_state_dict(ckpt["state_dict"])
        log.info(f"Missed: {missed}")
        log.info(f"Unexpected: {unexpected}")
    return model
