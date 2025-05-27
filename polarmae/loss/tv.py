import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from torch import Tensor
from pytorch3d.ops import ball_query
from polarmae.layers.grouping import masked_gather, fill_empty_indices

__all__ = ["TotalVariationLoss"]

class TotalVariationLoss(nn.Module):
    def __init__(self, radius: float, K: int, reduction: str = "mean", apply_to_argmax: bool = False):
        super().__init__()
        self.radius = radius
        self.K = K
        self.reduction = reduction
        self.apply_to_argmax = apply_to_argmax

    def forward(
        self,
        points: torch.Tensor, # [B, N, >=3]
        lengths: Optional[torch.Tensor] = None, # [B, N]
        logits: torch.Tensor = None, # [B, N, n_classes]
        ) -> torch.Tensor:
        """
        Compute the total variation loss for a point cloud.
        """

        N, P, D = points.shape

        if lengths is None:
            # create a dummy lengths tensor with shape (N,) and
            # all entries = P
            lengths = torch.full(
                (N,), fill_value=P, dtype=torch.int32, device=points.device
            )

        _, idx, _ = ball_query(
            p1=points,
            p2=points,
            radius=self.radius,
            lengths1=lengths,
            lengths2=lengths,
            K=self.K,
            return_nn=False,
        )

        # masked gather the logits
        lengths = idx.ne(-1).sum(-1)
        B,G,K = idx.shape
        mask = torch.arange(K, device=idx.device).expand(B, G, -1) < lengths.unsqueeze(-1)
        logit_nn = masked_gather(logits, fill_empty_indices(idx)) # [B, N, K, n_classes]
        logit_nn = logit_nn * mask.unsqueeze(-1) # [B, N, K, n_classes]

        if self.apply_to_argmax:
            # Get argmax indices (which class is predicted)
            pred_class_indices = logits.argmax(dim=-1)  # [B, N]
            
            # Create indices for gathering
            batch_indices = torch.arange(logits.shape[0], device=logits.device).view(-1, 1).expand(-1, logits.shape[1])
            point_indices = torch.arange(logits.shape[1], device=logits.device).view(1, -1).expand(logits.shape[0], -1)
            
            # Extract only the logit value for the predicted class for each point
            # This maintains differentiability while focusing only on the predicted class
            selected_logits = logits[batch_indices, point_indices, pred_class_indices]  # [B, N]
            
            # Similarly for neighbors
            # First get the predicted class for each original point (not the neighbors' own predictions)
            pred_class_expanded = pred_class_indices.unsqueeze(2).expand(-1, -1, logit_nn.shape[2])  # [B, N, K]
            
            # Create batch and point indices for logit_nn gathering
            batch_nn_indices = torch.arange(logits.shape[0], device=logits.device).view(-1, 1, 1).expand(-1, logits.shape[1], logit_nn.shape[2])
            point_nn_indices = torch.arange(logits.shape[1], device=logits.device).view(1, -1, 1).expand(logits.shape[0], -1, logit_nn.shape[2])
            neighbor_indices = torch.arange(logit_nn.shape[2], device=logits.device).view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1)
            
            # Extract neighbor logits for the same class that was predicted at the central point
            selected_neighbor_logits = logit_nn[batch_nn_indices, point_nn_indices, neighbor_indices, pred_class_expanded]  # [B, N, K]
            
            # Compute TV loss only on the selected logits
            selected_logits_expanded = selected_logits.unsqueeze(2).expand(-1, -1, logit_nn.shape[2])  # [B, N, K]
            tv_loss = torch.abs(selected_neighbor_logits - selected_logits_expanded)  # [B, N, K]
        else:
            tv_loss = torch.abs(logit_nn - logits.unsqueeze(2))  # [B, N, K, n_classes]
            tv_loss = tv_loss.mean(dim=-1)  # [B, N, K]

        tv_loss = tv_loss.sum(dim=-1) / (lengths.clamp(min=1)) # [B, N]
        tv_loss = tv_loss[lengths.gt(0)] # [B, N]

        if self.reduction == "mean":
            return tv_loss.mean()
        elif self.reduction == "sum":
            return tv_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
