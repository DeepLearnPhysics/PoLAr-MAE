import pytorch_lightning as pl
import os
import shutil
import tempfile
from typing import Optional, List
from polarmae.utils.scheduler import WarmupStableDecay
from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class TriggerDecayOnPlateau(pl.Callback):
    """
    Trigger decay on plateau of one or more metrics.

    Only works with the WarmupStableDecay scheduler.
    
    Args:
        monitor: metric or list of metrics to monitor
        min_delta: minimum change in the monitored quantity to qualify as an improvement
        patience: number of epochs with no improvement after which decay will be triggered
        mode: 'min' or 'max'
        ema_decay: decay factor for EMA smoothing. default = 1.0 
    """
    def __init__(self, monitor : str | List[str] ="val_loss", min_delta=0.0, patience=2, mode='min', ema_decay=1):
        assert mode in ['min', 'max'], "mode must be either 'min' or 'max'"
        if isinstance(monitor, str):
            monitor = [monitor]
        self.monitor = monitor
        self.min_delta, self.patience, self.mode = min_delta, patience, mode
        self.best, self.wait, self.stop_step = None, 0, None
        self.ema_decay = ema_decay
        
    def setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: Optional[str]):
        if not isinstance(trainer.lr_schedulers()[0]["scheduler"], WarmupStableDecay):
            raise ValueError("Scheduler must be a WarmupStableDecay scheduler to use the TriggerDecayOnPlateau callback")

        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = trainer.strategy.broadcast(dirpath)
        self.dirpath = dirpath

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        if self.stop_step is not None:         # decay already triggered
            return
        
        # Get all monitored metrics
        metrics = []
        for monitor_name in self.monitor:
            metric = trainer.callback_metrics.get(monitor_name)
            if metric is None:
                log.warning(f"Metric {monitor_name} not found in callback_metrics. Skipping epoch.")
                return
            metrics.append(metric)
        
        # Apply EMA smoothing to the metrics
        if self.best is None:
            self.best = metrics
            self.wait = 0
        else:
            # Update EMA of each metric and check for improvement
            ema_metrics = [self.ema_decay * self.best[i] + (1 - self.ema_decay) * metric 
                          for i, metric in enumerate(metrics)]
            
            # Only reset wait if ALL metrics improved
            if self.__all_metrics_improved(ema_metrics):
                self.wait = 0
            else:
                self.wait += 1
            
            self.best = ema_metrics

        if self.wait >= self.patience:         # plateau => begin decay
            log.info(f"Plateau reached for {self.monitor} with patience {self.patience}. Beginning decay.")
            sched = trainer.lr_schedulers()[0]["scheduler"]
            sched.begin_decay()                # jump to decay branch

            # tell trainer how many steps remain until we want to stop
            self.stop_step = trainer.global_step + sched.nd
            trainer.fit_loop.max_steps = self.stop_step   # Lightning stops at this step

            # save checkpoint of the model
            filepath = os.path.join(self.dirpath, f"wsd_{trainer.global_step}.ckpt")
            log.info(f"Saving checkpoint of the model at {trainer.global_step} steps to {filepath}.")
            trainer.save_checkpoint(filepath)
    
    def __all_metrics_improved(self, ema_metrics):
        """Check if all metrics have improved beyond min_delta threshold."""
        if self.mode == 'min':
            return all(ema < best - self.min_delta for ema, best in zip(ema_metrics, self.best))
        else:  # mode == 'max'
            return all(ema > best + self.min_delta for ema, best in zip(ema_metrics, self.best))

    def __resolve_ckpt_dir(self, trainer: 'pl.Trainer'):
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")
        return ckpt_path

    def state_dict(self):
        return {
            "dirpath": self.dirpath,
            "best": self.best,
            "wait": self.wait,
            "stop_step": self.stop_step,
            "monitor": self.monitor,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "mode": self.mode,
        }

    def load_state_dict(self, state_dict):
        self.dirpath = state_dict["dirpath"]
        self.best = state_dict["best"]
        self.wait = state_dict["wait"]
        self.stop_step = state_dict["stop_step"]
        self.monitor = state_dict["monitor"]
        self.min_delta = state_dict["min_delta"]
        self.patience = state_dict["patience"]
        self.mode = state_dict["mode"]

class CopyCodebaseCallback(pl.Callback):
    """
    Copy specific folders of the codebase to the save directory.
    
    Uses the same directory resolution logic as checkpoints but saves to a 'code' folder.
    The codebase is copied at the beginning of training to preserve the exact code state.
    Only copies the polarmae, configs, extensions, and scripts folders.
    
    Args:
        source_dir: Path to the root directory of the codebase to copy. If None, uses the current working directory.
    """
    def __init__(self):
        import polarmae
        self.source_dir = polarmae.__path__[0]
        self.folders_to_copy = ['polarmae', 'configs', 'extensions', 'scripts']
        self.code_dir = None
        
    def setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: Optional[str]):
        if stage == 'fit':
            # Resolve the code directory path using the same logic as checkpoints
            code_dir = self.__resolve_code_dir(trainer)
            code_dir = trainer.strategy.broadcast(code_dir)
            self.code_dir = code_dir
            
            # Create the code directory if it doesn't exist
            os.makedirs(self.code_dir, exist_ok=True)
            
            # Copy the codebase
            self._copy_codebase()
            
    def _copy_codebase(self):
        """Copy specific folders to the code directory."""
        if self.code_dir is None:
            log.warning("Code directory not set. Skipping codebase copy.")
            return
            
        log.info(f"Copying folders {self.folders_to_copy} from {self.source_dir} to {self.code_dir}")
        
        for folder_name in self.folders_to_copy:
            src_folder = os.path.join(self.source_dir, folder_name)
            dest_folder = os.path.join(self.code_dir, folder_name)
            
            if os.path.exists(src_folder) and os.path.isdir(src_folder):
                try:
                    # Copy the entire folder tree
                    shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
                    log.info(f"Successfully copied {folder_name} folder")
                except Exception as e:
                    log.warning(f"Failed to copy {folder_name} folder: {e}")
            else:
                log.warning(f"Folder {folder_name} not found in {self.source_dir}")
        
        log.info(f"Codebase folders copied successfully to {self.code_dir}")
    
    def __resolve_code_dir(self, trainer: 'pl.Trainer'):
        """Resolve the code directory path using the same logic as checkpoint directory."""
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            code_path = os.path.join(save_dir, str(name), version, "code")
        else:
            code_path = os.path.join(trainer.default_root_dir, "code")
        return code_path


