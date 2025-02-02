import os
from glob import glob
from typing import Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from polarmae.utils.pylogger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

log = RankedLogger(__name__, rank_zero_only=True)

class PILArNet(Dataset):
    def __init__(
        self,
        data_path: str,
        emin: float = 1.0e-6,
        emax: float = 20.0,
        energy_threshold: float = 0.13,
        remove_low_energy_scatters: bool = False,
        maxlen: int = -1,
        min_points: int = 1024,
        return_semantic_id: bool = True,
        return_cluster_id: bool = True,
    ):
        self.data_path = data_path
        self.h5_files = glob(data_path)
        assert len(self.h5_files) > 0, f"No h5 files found in {data_path}"
        self.emin = emin
        self.emax = emax
        self.energy_threshold = energy_threshold
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.min_points = min_points
        self.return_semantic_id = return_semantic_id
        self.return_cluster_id = return_cluster_id

        self.maxlen = maxlen
        self.initted = False

        log.info(f"{self.emin=}, {self.emax=}, {self.energy_threshold=}, {self.remove_low_energy_scatters=}")

        self.lengths = []

        self._build_index()
        self.h5data = []

    def __len__(self):
        if self.maxlen > 0:
            return min(self.maxlen, self.cumulative_lengths[-1])
        return self.cumulative_lengths[-1]

    def _build_index(self):
        log.info("Building index")
        self.cumulative_lengths = []
        indices = []
        for h5_file in self.h5_files:
            try:
                npoints = np.load(h5_file.replace(".h5", "_points.npy"))
                index = np.argwhere(npoints >= self.min_points).flatten()
            except FileNotFoundError as e: # just use em all
                log.info(f"File {h5_file} not found, using all points")
                with h5py.File(h5_file, 'r', libver='latest', swmr=True) as h5_file:
                    index = np.arange(h5_file['point'].shape[0])
            finally:
                self.cumulative_lengths.append(index.shape[0])
                indices.append(index)
        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        self.indices = indices
        log.info(f"{self.cumulative_lengths[-1]} point clouds were loaded")
        log.info(f"{len(self.h5_files)} files were loaded")

    def h5py_worker_init(self):
        self.h5data = []
        for h5_file in self.h5_files:
            self.h5data.append(h5py.File(h5_file, mode="r", libver="latest", swmr=True))
        self.initted = True

    def transform_energy(self, pc):
        """tranforms energy to logarithmic scale on [-1,1]"""
        threshold_mask = None
        if self.energy_threshold > 0.0:
            threshold_mask = pc[:, 3] > self.energy_threshold
            self.emin = self.energy_threshold
        pc[:, 3] = log_transform(pc[:, 3], self.emax, self.emin)
        return pc, threshold_mask
    
    def __getitem__(self, idx):
        if not self.initted:
            self.h5py_worker_init()

        # find which h5 file and index of the file the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        h5_file = self.h5data[h5_idx]
        idx = idx - self.cumulative_lengths[h5_idx]
        idx = self.indices[h5_idx][idx]

        # load the point cloud
        data = h5_file["point"][idx].reshape(-1, 8)[:, [0,1,2,3,5]] # (x,y,z,e,t)
        cluster_size, semantic_id = h5_file["cluster"][idx].reshape(-1, 5)[:, [0, -1]].T

        # remove low energy scatters (always first particle)
        if self.remove_low_energy_scatters:
            data = data[cluster_size[0] :]
            semantic_id, cluster_size = semantic_id[1:], cluster_size[1:]

        # compute semantic and instance (cluster) ids for each point
        data_semantic_id = np.repeat(semantic_id, cluster_size)
        cluster_id = np.repeat(np.arange(len(cluster_size)), cluster_size)


        # log transform energy to [-1,1]
        data, threshold_mask = self.transform_energy(data)

        # remove points below specified energy threshold
        if threshold_mask is not None:
            data = data[threshold_mask]
            data_semantic_id = data_semantic_id[threshold_mask]
            cluster_id = cluster_id[threshold_mask]

        data = torch.from_numpy(data[:,:4]).float()
        data_semantic_id = torch.from_numpy(data_semantic_id).unsqueeze(1).long()
        cluster_id = torch.from_numpy(cluster_id).unsqueeze(1).long()

        output = dict(
            points=data,
            semantic_id=data_semantic_id if self.return_semantic_id else None,
            cluster_id=cluster_id if self.return_cluster_id else None,
        )

        return output

    def __del__(self):
        if self.initted:
            for h5_file in self.h5data:
                h5_file.close()

    @staticmethod
    def init_worker_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.h5py_worker_init()

    @staticmethod
    def collate_fn(batch):
        """
        Pad all data to the same length.
        """

        data = [item['points'] for item in batch]
        semantic_id = [item['semantic_id'] for item in batch]
        cluster_id = [item['cluster_id'] for item in batch]

        lengths = torch.tensor(
            [points.size(0) for points in data], dtype=torch.long
        )  # Shape: (B,)
        padded_points = pad_sequence(data, batch_first=True)  # Shape: (B, N_max, 4)

        # optional data
        padded_semantic_id = None
        if semantic_id[0] is not None:
            padded_semantic_id = pad_sequence(
                semantic_id, batch_first=True, padding_value=-1
            )  # Shape: (B, N_max)

        padded_cluster_id = None
        if cluster_id[0] is not None:
            padded_cluster_id = pad_sequence(cluster_id, batch_first=True, padding_value=-1) # Shape: (B, N_max)

        out = dict(
            points=padded_points,
            lengths=lengths,
            semantic_id=padded_semantic_id,
            cluster_id=padded_cluster_id,
        )
        return out


class PILArNetDataModule(pl.LightningDataModule):
    _class_weights = None
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_kwargs: dict = {},
        test_dataset_kwargs: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.persistent_workers = True if num_workers > 0 else False

        # from datasets/ShapeNetPart.py
        self._category_to_seg_classes = {
            "shower": [0],
            "track": [1],
            "michel": [2],
            "delta": [3],
            "low energy deposit": [4],
        }
        # inverse mapping
        self._seg_class_to_category = {}
        for cat in self._category_to_seg_classes.keys():
            for cls in self._category_to_seg_classes[cat]:
                self._seg_class_to_category[cls] = cat

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PILArNet(self.hparams.data_path, **self.hparams.dataset_kwargs)
        val_dir = self.hparams.data_path.replace("train", "val")

        dataset_kwargs = self.hparams.dataset_kwargs.copy()
        dataset_kwargs.update(self.hparams.test_dataset_kwargs)
        self.val_dataset = PILArNet(val_dir, **dataset_kwargs)

        if self.train_dataset.remove_low_energy_scatters:
            self._category_to_seg_classes.pop("low energy deposit")
            self._seg_class_to_category.pop(4)

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=PILArNet.collate_fn,
            worker_init_fn=PILArNet.init_worker_fn,
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=PILArNet.collate_fn,
            worker_init_fn=PILArNet.init_worker_fn,
        )

    @property
    def category_to_seg_classes(self):
        return self._category_to_seg_classes

    @property
    def seg_class_to_category(self):
        return self._seg_class_to_category

    @property
    def num_seg_classes(self):
        return len(self._category_to_seg_classes)

    @property
    def class_weights(self):
        """
        inverse class weights, in the same order as category_to_seg_classes.
        used for loss weighting in semantic segmentation task.

        values taken from Figure 8 in Appendix A of the PoLArMAE paper.
        """
        if not hasattr(self, 'train_dataset'):
            self.setup()

        class_counts = torch.tensor([1926651899.0, 2038240940.0, 34083197.0, 92015482.0, 1145363125.0])

        if self.train_dataset.remove_low_energy_scatters:
            class_counts = class_counts[:-1]

        return class_counts.sum() / class_counts


def log_transform(x, xmax=1, eps=1e-7):
    # [eps, xmax] -> [-1,1]
    y0 = np.log10(eps)
    y1 = np.log10(eps + xmax)
    return 2 * (np.log10(x + eps) - y0) / (y1 - y0) - 1


def inv_log_transform(x, xmax=1, eps=1e-7):
    # [-1,1] -> [eps, xmax]
    y0 = np.log10(eps)
    y1 = np.log10(xmax + eps)
    x = (x + 1) / 2
    return 10 ** (x * (y1 - y0) + y0) - eps