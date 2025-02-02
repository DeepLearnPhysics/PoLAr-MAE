# Modified from https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/utils/pointnet_util.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from polarmae.layers.grouping import masked_gather
from polarmae.layers.masking import MaskedBatchNorm1d
from pytorch3d.ops import knn_gather, knn_points

__all__ = ["PointNetFeatureUpsampling"]

class PointNetFeatureUpsampling(nn.Module):
    def __init__(self, in_channel, mlp, K=5, pos_enc_module=None):
        super().__init__()
        self.K = K
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_enc_module = pos_enc_module


        last_channel = in_channel
        if self.pos_enc_module is None:
            last_channel = in_channel + 3

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(MaskedBatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2, point_lens, embedding_lens, point_mask):
        """
        Input:
            xyz1: input points position data, [B, N_max, C]
            xyz2: sampled input points position data, [B, S_max, C]
            points1: input points data, [B, N_max, D1]
            points2: input points data, [B, S_max, D2]
            point_mask: [B, N_max] boolean tensor indicating valid points in xyz1
            embedding_mask: [B, S_max] boolean tensor indicating valid points in xyz2
        Return:
            new_points: upsampled points data, [B, N_max, D']
        """
        B, N_max, C = xyz1.shape
        _, S_max, _ = xyz2.shape

        lengths1 = point_lens  # [B]
        lengths2 = embedding_lens  # [B]

        # K = torch.min(lengths2) # number of nearest neighbors
        # Find K nearest neighbors in xyz2 for each point in xyz1
        dists, idx, _ = knn_points(
            xyz1[..., :3], xyz2, lengths1=lengths1, lengths2=lengths2, K=self.K, return_sorted=False
        ) # [B, N_max, K], [B, N_max, K]

        # Avoid division by zero[]
        dist_recip = 1.0 / (dists + torch.finfo(dists.dtype).eps)
        norm = dist_recip.sum(dim=2, keepdim=True)  # [B, N_max, 1]
        weight = dist_recip / norm  # [B, N_max, K]

        # Gather features from points2 at indices idx
        # interpolated_points = knn_gather(
        #     points2, idx, lengths=lengths2
        # )  # [B, N_max, K, D2]
        interpolated_points = masked_gather(points2, idx)

        # Compute weighted sum
        interpolated_points = (interpolated_points * weight.unsqueeze(-1)).sum(
            dim=2
        )  # [B, N_max, D2]

        if points1 is not None and len(self.mlp_convs) > 0:
            if self.pos_enc_module is not None:
                new_points = interpolated_points + self.pos_enc_module(points1)
            else:
                new_points = torch.cat(
                    [points1, interpolated_points], dim=-1
                )  # [B, N_max, D1 + D2]
        else:
            new_points = interpolated_points  # [B, N_max, D2]

        # Transpose for convolution: [B, D', N_max]
        new_points = new_points.transpose(1, 2)

        # Mask for BatchNorm1d
        point_mask = point_mask.unsqueeze(1)  # [B, 1, N_max]

        # Apply MLP with masked batch normalization
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = conv(new_points)
            new_points = bn(new_points, point_mask)
            new_points = F.relu(new_points)

        new_points = new_points.transpose(1, 2)  # [B, N_max, D']

        return new_points, idx