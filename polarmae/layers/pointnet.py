import math
from typing import Tuple

import torch
import torch.nn as nn
from polarmae.layers.masking import MaskedBatchNorm1d


class MaskedMiniPointNet(nn.Module):
    def __init__(
        self,
        channels: int,
        feature_dim: int,
        equivariant: bool = False,
    ):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 128, 1, bias=False),
            MaskedBatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            MaskedBatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feature_dim, 1),
        )

        self.equivariant = equivariant
        if self.equivariant:
            self.position_encoder = PointOrderEncoder(256)

    def forward(self, points, mask) -> torch.Tensor:
        # points: (B, N, C)
        # mask: (B, 1, N)
        # pos: (B, N, 256)
        reshape = points.ndim == 4
        if reshape:
            B,G,S,C = points.shape
            points = points.reshape(B*G, S, C)
            mask = mask.reshape(B*G, 1, S)

        feature = points.transpose(2, 1)  # (B, C, N)

        for layer in self.first_conv:
            if isinstance(layer, MaskedBatchNorm1d):
                feature = layer(feature, mask)
            else:
                feature = layer(feature)

        if self.equivariant:
            feature = feature + self.position_encoder(points).transpose(2,1)

        # (B, 256, N) --> (B, 256, 1)
        feature_global = torch.max(feature, dim=2, keepdim=True).values  # (B, 256, 1)
        # concating global features to each point features
        feature = torch.cat(
            [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        )  # (B, 512, N)

        for layer in self.second_conv:
            if isinstance(layer, MaskedBatchNorm1d):
                feature = layer(feature, mask)
            else:
                feature = layer(feature)

        # (B, feature_dim, N) --> (B, feature_dim)
        feature_global = torch.max(feature, dim=2).values  # (B, feature_dim)

        if reshape:
            feature_global = feature_global.reshape(B,G, -1)
        return feature_global


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for the diffusion process.

    Input:
    Timestep: the current timestep, in range [1, ..., T]
    """
    def __init__(self, dim):
        super().__init__()
        self.emb_dim = dim

    def forward(self, ts):
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=ts.device) * -emb)
        emb = ts[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PointOrderEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, points):
        # points: (B, N, C)
        inp = torch.arange(points.shape[1], device=points.device) # (N)
        temb = self.time_embed(inp) # (N, dim)
        temb = temb.unsqueeze(0) # (1, N, dim)
        return temb