# -*- coding: utf-8 -*-
# @Author: Sam Young
# @Date:   2024-12-31 10:34:00
# @Email: youngsam@stanford.edu

from pytorch3d.ops import ball_query
from . import _ext
import torch

def cnms(centroids, radius, overlap_factor, K=None, lengths=None):
    """
    Perform Centrality-Based Non-Maximum Suppression (C-NMS).

    This function is designed to reduce the number of centroids in a point cloud by 
    applying centrality-based non-maximum suppression. It identifies overlapping 
    spheres around each centroid and retains only the most central ones based on a 
    specified overlap factor and radius. The function uses a greedy reduction 
    algorithm to efficiently determine which centroids to keep, ensuring that the 
    retained centroids are as central as possible within their respective 
    neighborhoods.

    This is essentially the same as patchifying a point cloud using voxelization, just
    without the aliasing that comes from voxelization.

    Args:
        centroids: Tensor of shape (N, P, 3) containing the coordinates of centroids.
        overlap_factor: Factor to determine the query radius for overlap checking. (0.7 = 70% overlap of diameters)
        radius: Radius of the spheres (fixed).
        K: Number of points to consider for overlap checking. Default is P (all centroids). Consider this to be a
        best guess for the number of points that may be within the overlap radius.
        lengths: Tensor of shape (N,) containing the number of points in each cloud. Default is P for all clouds.

    Returns:
        centroids: Tensor of shape (N, P, 3) containing the coordinates of retained centroids.
        lengths: Tensor of shape (N,) containing the number of points in each cloud.
    """
    N, P, D = centroids.shape

    if lengths is None:
        # create a dummy lengths tensor with shape (N,) and
        # all entries = P
        lengths = torch.full(
            (N,),
            fill_value=P,
            dtype=torch.int64,
            device=centroids.device
        )


    query_radius = 2 * radius * overlap_factor
    _, idx, _ = ball_query(
        p1=centroids,
        p2=centroids,
        K=P if K is None else K,
        radius=query_radius,
        lengths1=lengths,
        lengths2=lengths,
        return_nn=False,
    )

    overlap_counts = (~idx.eq(-1)).sum(-1)
    _, sorted_indices = overlap_counts.sort(
        dim=-1, descending=True
    )  # (N, P)


    # find retained points
    ignore_idx = -1
    retain = torch.zeros((N, P), device=centroids.device).bool()
    retain = greedy_reduction(sorted_indices, idx, lengths, ignore_idx, retain)

    # reindex group centers to be retained points first, then discarded points after so we can use lengths1
    idx = torch.argsort((~retain).long(), dim=1)  # shape [G, K]
    centroids = centroids.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D)) # (N, P, D)
    lengths = retain.sum(dim=1)
    centroids = centroids[:,:lengths.max()]

    return centroids, lengths

def greedy_reduction(sorted_indices, idx, lengths, ignore_idx, out=None):
    N, P = sorted_indices.shape
    if out is None:
        out = torch.zeros((N, P), device=sorted_indices.device).bool()
    return _ext.greedy_reduction(sorted_indices, idx, lengths, ignore_idx, out)
