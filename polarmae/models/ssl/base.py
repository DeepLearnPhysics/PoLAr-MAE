from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

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

        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            datamodule.setup("fit")
            log.info(f"🍗  Setup {dataset_name} datamodule for SVM validation.")
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.define_metric(
                        f"svm_train_acc_{dataset_name}", summary="last,max"
                    )
                    logger.experiment.define_metric(
                        f"svm_val_acc_{dataset_name}", summary="last,max"
                    )

    def validate(self, datamodule: pl.LightningDataModule):
        # Lightning controls the `training` and `grad_enabled` state. Don't want to mess with it, but make sure it's correct.
        assert not self.training
        assert not torch.is_grad_enabled()

        max_tokens: int = self.hparams.svm_validation_max_tokens  # type: ignore
        def xy(dataloader):
            x_list = []
            label_list = []

            total = (
                max_tokens
                // (
                    self.encoder.tokenizer.grouping.group_max_points
                    * dataloader.batch_size
                )
                if max_tokens is not None
                else None
            )
            num_labels = datamodule.num_seg_classes

            for i, batch in enumerate(dataloader):
                data = batch['points'].cuda()
                lengths = batch['lengths'].cuda()
                labels_batch = batch['semantic_id'].cuda()
                with torch.no_grad():
                    out = self.encoder.prepare_tokens_with_masks(data, lengths, ids=labels_batch)
                    x = self.encoder.transformer(out['x'], out['pos_embed'], out['emb_mask']).last_hidden_state.reshape(-1, self.encoder.embed_dim)
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
                    y = (counts>0).long() # Shape: (N, num_labels)
                    mask_flat = out['emb_mask'].reshape(-1)
                    x = x[mask_flat]
                    y = y[mask_flat]
                    x_list.append(x.cpu())
                    label_list.append(y.cpu())
                    if total is not None and i >= total:
                        break

            x = torch.cat(x_list, dim=0)[:max_tokens]
            y = torch.cat(label_list, dim=0)[:max_tokens]
            return x, y

        x_train, y_train = xy(datamodule.train_dataloader())  # type: ignore
        x_val, y_val = xy(datamodule.val_dataloader())  # type: ignore

        # PCA down to 128 dimensions
        pca = PCA(n_components=128)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)

        svm_C: float = self.hparams.svm_validation_C  # type: ignore
        # svm = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=0, n_jobs=4)
        # n_estimators = 5
        # svm = BaggingClassifier(SVC(
        #     kernel='linear', probability=False, class_weight='balanced', random_state=0), 
        #     max_samples=1.0 / n_estimators, n_estimators=n_estimators, random_state=0,
        #     n_jobs=n_estimators)
        svm = OneVsRestClassifier(LinearSVC(C=svm_C, class_weight='balanced', random_state=0), n_jobs=y_train.shape[1])
        svm.fit(x_train, y_train)  # type: ignore
        train_acc: float = svm.score(x_train, y_train)  # type: ignore
        val_acc: float = svm.score(x_val, y_val)  # type: ignore
        train_report = classification_report(y_train, svm.predict(x_train), output_dict=True)
        val_report = classification_report(y_val, svm.predict(x_val), output_dict=True)

        train_class_scores = {datamodule.seg_class_to_category[int(label)]: metrics['f1-score'] for label, metrics in train_report.items() if label.isdigit()}
        val_class_scores = {datamodule.seg_class_to_category[int(label)]: metrics['f1-score'] for label, metrics in val_report.items() if label.isdigit()}
        return train_acc, val_acc, train_class_scores, val_class_scores

    def on_validation_epoch_end(self) -> None:
        assert not self.training
        assert not torch.is_grad_enabled()
        # return
        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            svm_train_acc, svm_val_acc, train_class_scores, val_class_scores = self.validate(datamodule)
            self.log(f"svm_train_acc_{dataset_name}", svm_train_acc, sync_dist=True)
            self.log(f"svm_val_acc_{dataset_name}", svm_val_acc, sync_dist=True)
            for label, score in train_class_scores.items():
                self.log(f"svm_train_class_score_{dataset_name}_{label}", score, sync_dist=True)
            for label, score in val_class_scores.items():
                self.log(f"svm_val_class_score_{dataset_name}_{label}", score, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # This is a bit of a hack. We want to avoid saving the datasets in the svm_validation dict,
        # as this would store the entire dataset inside the checkpoint, blowing it up to multiple GBs.
        checkpoint["hyper_parameters"]["svm_validation"] = {}