from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm

from polarmae.models.base import BaseModel
from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class SSLModel(BaseModel):
    """
    Base class for SSL models.
    """
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

        log.info(f"🍗  Setup datamodule for SVM validation.")
        for logger in self.loggers:
            dataset_name = 'larnet' # for now :) TODO! REMOVE
            if isinstance(logger, WandbLogger):
                logger.experiment.define_metric(
                    f"svm_train_acc_{dataset_name}", summary="last,max"
                )
                logger.experiment.define_metric(
                    f"svm_val_acc_{dataset_name}", summary="last,max"
                )
                if getattr(self.hparams, 'watch_grad', False):
                    logger.watch(self, log_graph=False)

    def validate(self, postnorm=False):
        # Lightning controls the `training` and `grad_enabled` state. Don't want to mess with it, but make sure it's correct.
        assert not self.training
        assert not torch.is_grad_enabled()

        datamodule = self.trainer.datamodule
        max_tokens: int = self.hparams.svm_validation_max_tokens  # type: ignore
        def xy(dataloader):
            x_list = []
            label_list = []

            total = (
                int(25 * (max_tokens / 15000) * (100 / self.trainer.datamodule.hparams.svm_batch_size))
                if max_tokens is not None
                else None
            )
            num_labels = datamodule.num_seg_classes
            
            # Collect all data first for class balancing
            all_x = []
            all_y = []
            
            for i, batch in enumerate(dataloader):
                data = batch['points'].cuda()
                data = self.val_transformations(data)
                lengths = batch['lengths'].cuda()
                labels_batch = batch['semantic_id'].cuda()
                with torch.no_grad():
                    out = self.encoder.prepare_tokens(data, lengths, ids=labels_batch)
                    x = self.encoder.transformer(out['x'], out['pos_embed'], out['emb_mask'], final_norm=postnorm).last_hidden_state.reshape(-1, self.encoder.embed_dim)
                    semantic_ids = out['id_groups'].reshape(-1, out['id_groups'].shape[2])

                    # Vectorized computation to replace the loop
                    N = semantic_ids.shape[0]  # Number of groups
                    D = semantic_ids.shape[1]  # Number of semantic IDs per group

                    group_indices = torch.arange(N, device=semantic_ids.device).unsqueeze(1).expand(-1, D)  # Shape: (N, D)
                    semantic_ids_flat = semantic_ids.reshape(-1)
                    group_indices_flat = group_indices.reshape(-1)
                    valid_mask = semantic_ids_flat != -1
                    semantic_ids_valid = semantic_ids_flat[valid_mask]  # Shape: (K,)
                    group_indices_valid = group_indices_flat[valid_mask]  # Shape: (K,)
                    counts = torch.zeros((N, num_labels), dtype=torch.int64, device=semantic_ids.device)
                    counts.index_add_(0, group_indices_valid, torch.nn.functional.one_hot(semantic_ids_valid, num_classes=num_labels).to(torch.int64))
                    # y = counts.argmax(dim=1)  # Shape: (N,)
                    y = (counts>0).long()  # Shape: (N, num_labels)
                    mask_flat = out['emb_mask'].reshape(-1)
                    x = x[mask_flat]
                    y = y[mask_flat]
                    all_x.append(x.cpu())
                    all_y.append(y.cpu())
                    if total is not None and i >= total:
                        break

            # Concatenate all data
            all_x = torch.cat(all_x, dim=0)
            all_y = torch.cat(all_y, dim=0)
            
            # Balance classes - find groups for each semantic ID
            semantic_ids_numpy = all_y.numpy()  # Shape: (N, num_labels)
            
            # Find which semantic IDs are present (exclude classes with no samples)
            class_present = (semantic_ids_numpy.sum(axis=0) > 0)
            present_classes = torch.where(torch.from_numpy(class_present))[0].numpy()
            
            if len(present_classes) == 0:
                # Fallback if no classes found
                x = all_x[:max_tokens] if max_tokens else all_x
                y = all_y[:max_tokens] if max_tokens else all_y
                return x, y
            
            # For each group, find which semantic IDs are present
            N = semantic_ids_numpy.shape[0]
            groups_per_class = {class_id: [] for class_id in present_classes}
            
            for i in range(N):
                group_classes = torch.where(torch.from_numpy(semantic_ids_numpy[i, :]))[0].numpy()
                # Add this group index to each class that appears in it
                for class_id in group_classes:
                    if class_id in groups_per_class:
                        groups_per_class[class_id].append(i)
            
            # Find minimum number of groups available for any class
            min_groups_per_class = min(len(groups) for groups in groups_per_class.values() if len(groups) > 0)
            
            # If we have a max_tokens limit, adjust min_groups_per_class accordingly
            if max_tokens:
                max_groups_per_class = max_tokens // len(present_classes)
                min_groups_per_class = min(min_groups_per_class, max_groups_per_class)
            
            # Sample equal number of groups for each class
            balanced_indices = set()
            torch.manual_seed(42)  # For reproducibility
            
            for class_id, group_indices in groups_per_class.items():
                if len(group_indices) > 0:
                    # Randomly sample min_groups_per_class indices for this class
                    if len(group_indices) >= min_groups_per_class:
                        sampled_indices = torch.randperm(len(group_indices))[:min_groups_per_class]
                        sampled_indices = [group_indices[idx] for idx in sampled_indices]
                    else:
                        sampled_indices = group_indices
                    balanced_indices.update(sampled_indices)
            
            balanced_indices = sorted(list(balanced_indices))
            
            # Create balanced subsets
            x = all_x[balanced_indices]
            y = all_y[balanced_indices]
            
            # log.info(f"Class balancing: {len(balanced_indices)} groups selected, {min_groups_per_class} per class")
            
            return x, y

        x_train, y_train = xy(datamodule.svm_train_dataloader())  # type: ignore
        x_val, y_val = xy(datamodule.svm_val_dataloader())  # type: ignore

        # PCA down to 128 dimensions
        # pca = PCA(n_components=128)
        # x_train = pca.fit_transform(x_train)
        # x_val = pca.transform(x_val)

        # Grid search over C values
        C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0]
        best_val_f1 = -1
        best_C = None
        best_results = None
        
        for svm_C in tqdm(C_values):
            svm = OneVsRestClassifier(LinearSVC(C=svm_C, class_weight='balanced', random_state=0, max_iter=5000), n_jobs=y_train.shape[1])
            svm.fit(x_train, y_train)  # type: ignore
            train_acc: float = svm.score(x_train, y_train)  # type: ignore
            val_acc: float = svm.score(x_val, y_val)  # type: ignore
            
            train_report = classification_report(y_train, svm.predict(x_train), output_dict=True, zero_division=torch.nan)
            val_report = classification_report(y_val, svm.predict(x_val), output_dict=True, zero_division=torch.nan)

            train_class_scores = {datamodule.seg_class_to_category[int(label)]: metrics['f1-score'] for label, metrics in train_report.items() if label.isdigit()}
            val_class_scores = {datamodule.seg_class_to_category[int(label)]: metrics['f1-score'] for label, metrics in val_report.items() if label.isdigit()}

            train_class_scores['macro'] = train_report['macro avg']['f1-score']
            val_class_scores['macro'] = val_report['macro avg']['f1-score']
            
            val_f1 = val_class_scores['macro']
            
            # Track the best validation F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_C = svm_C
                
                best_results = (train_acc, val_acc, train_class_scores, val_class_scores, best_C)
            
            # Log results for this C value if desired (optional)
            if hasattr(self, 'log'):
                suffix = "_postnorm" if postnorm else "_no_postnorm"
                self.log(f"svm_val_f1_C_{svm_C}{suffix}", val_f1, sync_dist=True, on_epoch=True, on_step=False)
        log.info(f"Best SVM C value: {best_C} (val_f1: {best_val_f1:.4f})")
        
        return best_results

    def on_validation_epoch_end(self) -> None:
        assert not self.training
        assert not torch.is_grad_enabled()

        svm_train_acc, svm_val_acc, train_class_scores, val_class_scores, best_C = self.validate(postnorm=False)
        batch_size = self.trainer.datamodule.hparams.svm_batch_size
        self.log("svm_train_acc_no_postnorm", svm_train_acc, sync_dist=True, batch_size=batch_size)
        self.log("svm_val_acc_no_postnorm", svm_val_acc, sync_dist=True, batch_size=batch_size)
        self.log("svm_best_C_no_postnorm", best_C, sync_dist=True, batch_size=batch_size)
        for label, score in train_class_scores.items():
            self.log(f"svm_train_class_f1_larnet_no_postnorm_{label}", score, sync_dist=True, batch_size=batch_size)
        for label, score in val_class_scores.items():
            self.log(f"svm_val_class_f1_larnet_no_postnorm_{label}", score, sync_dist=True, batch_size=batch_size)

        if self.encoder.transformer.norm.__class__.__name__ != "Identity":
            svm_train_acc, svm_val_acc, train_class_scores, val_class_scores, best_C = self.validate(postnorm=True)
            batch_size = self.trainer.datamodule.hparams.svm_batch_size
            self.log("svm_train_acc_postnorm", svm_train_acc, sync_dist=True, batch_size=batch_size)
            self.log("svm_val_acc_postnorm", svm_val_acc, sync_dist=True, batch_size=batch_size)
            self.log("svm_best_C_postnorm", best_C, sync_dist=True, batch_size=batch_size)
            for label, score in train_class_scores.items():
                self.log(f"svm_train_class_f1_larnet_postnorm_{label}", score, sync_dist=True, batch_size=batch_size)
            for label, score in val_class_scores.items():
                self.log(f"svm_val_class_f1_larnet_postnorm_{label}", score, sync_dist=True, batch_size=batch_size)
