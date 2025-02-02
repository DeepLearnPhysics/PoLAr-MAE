import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
from typing import Optional
from torch import Tensor

__all__ = ['SoftmaxFocalLoss', 'DiceLoss']

class SoftmaxFocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        gamma: float = 2,
        ignore_index: int = -1,
    ):
        super().__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        flattened_logits = logits.reshape(-1, logits.shape[-1])
        flattened_labels = labels.view(-1)

        p_t = flattened_logits.softmax(dim=-1)
        ce_loss = F.cross_entropy(
            flattened_logits,
            flattened_labels,
            reduction="none",
            ignore_index=self.ignore_index,
        )  # -log(p_t)

        alpha_t = self.weight
        # alpha_t = labels.ne(-1).sum() / labels[labels.ne(-1)].bincount()
        loss = (
            alpha_t[flattened_labels]
            * ((1 - p_t[torch.arange(p_t.shape[0]), flattened_labels]) ** self.gamma)
            * ce_loss
        )

        if self.reduction == "mean":
            loss = loss.sum() / labels.ne(self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        return loss
    
def focal_loss(inputs, targets, pos_weight=None, alpha=0.25, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=pos_weight)
    pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()
