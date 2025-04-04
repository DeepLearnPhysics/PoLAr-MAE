import torch
from polarmae.layers.masking import SyncMaskedBatchNorm1d

# set pytorch_lightning.plugins to use this TorchSyncBatchNorm instead of the default one
torch.nn.SyncBatchNorm = SyncMaskedBatchNorm1d