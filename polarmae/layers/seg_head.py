import torch
import torch.nn as nn
import torch.nn.functional as F
from polarmae.layers.masking import MaskedBatchNorm1d, masked_layer_norm
from typing import List


class SegmentationHead(nn.Module):
    """Takes in latent points and outputs segmentation logits for each point"""
    def __init__(
        self,
        in_channels: int,
        seg_head_dim: int,
        seg_head_dropout: float,
        num_classes: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            seg_head_dim,
            1,
            bias=False,
        )
        self.bn1 = MaskedBatchNorm1d(seg_head_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(seg_head_dropout)

        self.conv2 = nn.Conv1d(seg_head_dim, seg_head_dim // 2, 1, bias=False)
        self.bn2 = MaskedBatchNorm1d(seg_head_dim // 2)
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(seg_head_dropout)  # Uncomment if needed

        self.conv3 = nn.Conv1d(seg_head_dim // 2, num_classes, 1)

    def forward(self, x, point_mask=None):
        """
        x: Input tensor of shape [B, C, N], where N is the maximum number of points.
        point_mask: Boolean tensor of shape [B, N], where True indicates valid points.
        """ 
        # Ensure point_mask has the correct shape and type
        if point_mask is None:
            point_mask = torch.ones_like(x[..., 0], dtype=torch.bool)
        mask = point_mask.unsqueeze(1).float()  # [B, 1, N]

        # Apply first layer
        x = self.conv1(x)
        x = self.bn1(x, mask)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Apply second layer
        x = self.conv2(x)
        x = self.bn2(x, mask)
        x = self.relu2(x)
        # x = self.dropout2(x)  # Uncomment if dropout is needed

        # Final convolution layer (no batch norm or activation)
        x = self.conv3(x)

        return x

class IntermediateFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, hidden_states: List[torch.Tensor], mask=None, layers: List[int] = []) -> torch.Tensor:
        hidden_states = [
            masked_layer_norm(hidden_states[i], hidden_states[i].shape[-1], mask)
            for i in layers
        ]
        return self.linear(torch.cat(hidden_states, dim=-1))
