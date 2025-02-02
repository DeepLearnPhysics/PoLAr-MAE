from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from cnms import cnms
from pytorch3d import _C
from pytorch3d.ops import ball_query, knn_points


@torch.no_grad()
def masked_mean(group, point_mask):
    valid_elements = point_mask.sum(-1).float().clamp(min=1)
    return (group * point_mask.unsqueeze(-1)).sum(-2) / valid_elements.unsqueeze(-1)


# @torch.no_grad()
def fill_empty_indices(idx: torch.Tensor) -> torch.Tensor:
    """
    replaces all empty indices (-1) with the first index from its group
    """
    K = idx.shape[-1]
    mask = idx == -1
    first_idx = idx[..., 0].unsqueeze(-1).expand(*([-1]*len(idx.shape[:-1])), K)
    idx[mask] = first_idx[mask]  # replace -1 index with first index
    return idx

# @torch.no_grad()
def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """
    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    if points.ndim == 3:
        N, P, D = points.shape
    elif points.ndim == 4:
        N, G, P, D = points.shape
    else:
        raise ValueError(f"points format is not supported {points.shape}")

    # Replace -1 values with 0 before expanding
    idx = idx.clone()
    idx[idx.eq(-1)] = 0

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]

        if points.ndim == 3:
            # Match dimensions for points and indices
            idx_expanded = idx[..., None].expand(-1, -1, -1, D)
            points = points[:, :, None, :].expand(-1, -1, K, -1)
        elif points.ndim == 4:
            # Match dimensions for points and indices
            idx_expanded = idx[:, :, :, None, None].expand(-1, -1, -1, P, D)
            points = points[:, :, None, :, :].expand(-1, -1, K, -1, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    elif idx.ndim == 3:
        K = idx.shape[2]

    else:
        raise ValueError(f"idx format is not supported {idx.shape}")

    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)

    # SY 10/7/24: this takes a while and doesn't seem to be necessary because
    # we don't use the invalid indices for anything
    # Replace padded values
    # selected_points[idx_expanded_mask] = 0.0
    return selected_points

# modified from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_farthest_points.html
@torch.no_grad()
def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.dtype != torch.int64:
            lengths = lengths.to(torch.int64)
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Check dtypes are correct and convert if necessary
    if not (points.dtype == torch.float32):
        points = points.to(torch.float32)
    if not (lengths.dtype == torch.int64):
        lengths = lengths.to(torch.int64)
    if not (K.dtype == torch.int64):
        K = K.to(torch.int64)

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start_point:
        rand_uniform = torch.rand(N, device=lengths.device)
        scaled_rand = rand_uniform * lengths.float()
        start_idxs = scaled_rand.long()

    with torch.no_grad():
        # pyre-fixme[16]: `pytorch3d_._C` has no attribute `sample_farthest_points`.
        idx = _C.sample_farthest_points(points[:, :, :3], lengths, K, start_idxs)
    sampled_points = masked_gather(points, idx)
    return sampled_points, idx

@torch.no_grad()
def select_topk_by_energy(
    points: torch.Tensor,
    idx: torch.Tensor,
    K: int,
    energies_idx: int = 3,
) -> torch.Tensor:
    """
    Select the top K indices based on energies for each group.

    Args:
        points: Tensor of shape (B, N, C) containing point cloud data.
        idx: Tensor of shape (B, G, K_original) containing indices from ball_query.
        K: Desired number of top points to select per group.
        energies_idx: Index in `points` where the energy value is stored.

    Returns:
        topk_idx: Tensor of shape (B, G, K) containing indices of the top K energies per group.
    """
    B, G, K_original = idx.shape
    invalid_idx_mask = idx == -1

    # Clamp idx to handle negative indices for gathering
    idx_clamped = idx.clamp(min=0)  # Shape: (B, G, K_original)

    # Extract energies from points
    points_energies = points[..., energies_idx]  # Shape: (B, N)

    # Expand points_energies to match idx_clamped's shape for gathering
    points_energies_expanded = points_energies.unsqueeze(1).expand(
        -1, G, -1
    )  # Shape: (B, G, N)

    # Gather energies using idx_clamped
    energies = torch.gather(
        points_energies_expanded, dim=2, index=idx_clamped
    )  # Shape: (B, G, K_original)

    # Set energies of invalid indices to -infinity
    energies[invalid_idx_mask] = -float("inf")

    # Select top K energies
    topk_energies, topk_indices = energies.topk(K, dim=2)  # Shapes: (B, G, K)

    # Gather corresponding indices
    topk_idx = torch.gather(idx, 2, topk_indices)  # Shape: (B, G, K)

    # Set invalid indices back to -1
    mask_valid = topk_energies != -float("inf")  # Shape: (B, G, K)
    topk_idx[~mask_valid] = -1

    return topk_idx

@torch.no_grad()
def select_topk_by_fps(points: torch.Tensor, idx: torch.Tensor, K: int) -> torch.Tensor:
    """
    Args:
        points: Tensor of shape (B, N, C) containing point cloud data.
        idx: Tensor of shape (B, G, K_original) containing indices from ball_query.
        K: Desired number of points to select per group.
    """
    B, N, C = points.shape
    B, G, K_original = idx.shape

    # 1. index points by idx
    points_grouped = masked_gather(points, idx)  # (B, G, K, C)
    # 2. reshape points to (B*G, K, C)
    points_grouped = points_grouped.view(B*G, K_original, C)  # (B*G, K, C)
    # 3. run fps on the reshaped points
    _, idx_fps = sample_farthest_points(
        points_grouped,
        lengths=(~idx.eq(-1)).sum(2).view(B*G),
        K=K,
    )  # (B*G, K)
    # 4. reshape the fps indices back to to (B, G, K)
    idx_fps = idx_fps.view(B, G, K)

    invalid_idx_mask = idx_fps == -1
    idx_fps = idx_fps.clamp(min=0)
    idx_fps = torch.gather(idx, 2, idx_fps)
    idx_fps[invalid_idx_mask] = -1

    # 5. return the fps indices
    return idx_fps

class PointcloudGrouping(nn.Module):
    """
    Groups points in a point cloud into clusters based on specified criteria.

    This module supports grouping using different strategies determined by the reduction_method,
    such as "energy" which uses an energy-based approach, or "fps" which employs farthest point
    sampling. It also provides options for using relative features, normalizing group centers,
    and rescaling by group radius. An optional overlap factor further refines the grouping.

    - If group_radius is not provided, then we use a kNN to get the group points.
    - If group_radius is provided, then we use a ball query to get the group points.
    - If overlap_factor is provided, then we use treat num_groups as num_seed_groups and
      use cnms to get the final group centers.
    - If overlap_factor is not provided, then we just use the seed points as the group centers
      for vanilla FPS+{kNN, ball query}.

    Args:
        num_groups (int): Number of groups (clusters) to form.
        group_max_points (int): Maximum number of points allowed in each group.
        group_radius (Optional[float]): Radius used to determine the neighborhood for grouping.
        group_upscale_points (Optional[int]): Upscaling parameter used for kNN/ball query to get most of the points/group,
                                              even if there are more than group_max_points in the radius.
        overlap_factor (Optional[float]): Factor to control the allowed overlap between groups.
        context_length (Optional[int]): Limit on the number of groups to retain after grouping.
        reduction_method (Literal["energy", "fps"]): Reduction method for grouping from group_upscale_points to group_max_points; either using energy-based
            scoring ("energy") or farthest point sampling ("fps").
        use_relative_features (bool): Flag indicating whether to subtract off the 'group center' for the channels,
                                      in addition to the x,y,z group center.
        normalize_group_centers (bool): Flag to make each group's center the mean of the points in the group,
                                       instead of just one of the points.
        rescale_by_group_radius (bool): Flag to determine if groups should be rescaled by the group_radius, as in PointNeXT.
    """
    def __init__(
        self,
        num_groups: int,
        group_max_points: int,
        group_radius: Optional[float] = None,
        group_upscale_points: Optional[int] = None,
        overlap_factor: Optional[float] = None,
        context_length: Optional[int] = None,
        reduction_method: Literal["energy", "fps"] = "energy",
        use_relative_features: bool = False,
        normalize_group_centers: bool = False,
        rescale_by_group_radius: bool = True,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.group_radius = group_radius
        self.overlap_factor = overlap_factor
        self.context_length = context_length
        self.reduction_method = reduction_method
        self.group_max_points = group_max_points
        self.group_upscale_points = group_upscale_points
        self.use_relative_features = use_relative_features
        self.normalize_group_centers = normalize_group_centers
        self.rescale_by_group_radius = rescale_by_group_radius


    @torch.no_grad()
    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        endpoints: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points: (B, N, C)
        # lengths: (B,)

        # sample farthest points (either seed points or legit the centers)
        possible_centers, idx = sample_farthest_points(
            points[...,:3].float(),
            K=self.num_groups,
            lengths=lengths,
            random_start_point=True,
        )  # (B, G, 3)

        # if we have an overlap factor, run cnms to get the final group centers
        if self.overlap_factor is not None:
            # run cnms
            group_centers, lengths1 = cnms(
                possible_centers,
                overlap_factor=self.overlap_factor,
                radius=self.group_radius,
                K=self.num_groups,
                lengths=idx.ne(-1).sum(-1),
            )  # (B, G, 3), (B,)
            group_centers = group_centers[:,:self.context_length]
            lengths1 = lengths1.clamp_max(self.context_length)
        else:
            # if no overlap factor, just use the seed points as the group centers
            group_centers = possible_centers
            lengths1 = None

        semantic_id_groups = None
        endpoints_groups = None

        if self.group_upscale_points is None:
            self.group_upscale_points = self.group_max_points

        # if no group radius, use knn to get the group points
        if self.group_radius is None:
            # KNN
            _, idx, _ = knn_points(
                group_centers[:, :, :3].float(),
                points[:, :, :3].float(),
                lengths1=lengths1,
                lengths2=lengths,
                K=self.group_upscale_points,
                return_sorted=False,
                return_nn=False,
            )  # (B, G, K_big)
        else: # otherwise, use ball query to get the group points
            # Ball query
            _, idx, _ = ball_query(
                group_centers[:, :, :3].float(),
                points[:, :, :3].float(),
                K=self.group_upscale_points,
                radius=self.group_radius,
                lengths1=lengths1,
                lengths2=lengths,
                return_nn=False,
            )  # idx: (B, G, K_big)

        # Energy-based reduction of the group points (K_big --> K by taking top K energies)
        if self.reduction_method == 'energy':
            idx = select_topk_by_energy(
                points=points,
                idx=idx,
                K=self.group_max_points,
                energies_idx=3,  # Assuming energy is at index 3
            )  # idx: (B, G, K)
        elif self.reduction_method == 'fps':
            # farthest point sampling reduction of the group points (K_big --> K by farthest point sampling)
            idx = select_topk_by_fps(
                points=points,
                idx=idx,
                K=self.group_max_points,
            )  # idx: (B, G, K)

        # Gather semantic ids
        if semantic_id is not None:
            semantic_id_groups = masked_gather(
                    semantic_id, idx
            )  # (B, G, K, 1)
            semantic_id_groups[idx.eq(-1)] = -1

        if endpoints is not None:
            endpoints_groups = masked_gather(
                endpoints, idx
            )  # (B, G, K, 6)
            endpoints_groups[idx.eq(-1)] = -1

        # Create point mask with shape (B, G, K)
        point_lengths = (~idx.eq(-1)).sum(2)  # (B, G)
        groups = masked_gather(points, fill_empty_indices(idx))  # (B, G, K, C)
        B,G,K = idx.shape
        T = min(self.context_length, G)
        point_mask = torch.arange(K, device=idx.device).expand(B, T, -1) < point_lengths[:, :T].unsqueeze(-1)  # (B, G, K)

        # Create embedding mask (i.e. which groups/embeddings to ignore in transformer)
        B, G, K, C = groups.shape
        group_lengths = (~idx.eq(-1)).all(2).sum(1) # (B,)
        embedding_mask = torch.arange(G, device=points.device).repeat(B, 1) < group_lengths.unsqueeze(1)

        # G (max groups) --> T (context length)
        # we are implicitly assuming that the number of non-padded groups is less than the context length. if 
        # this is not the case, some valid groups will be ignored, which is terrible. be careful!
        K = self.group_max_points
        groups = groups[:, :T, :K] # (B, G, K, C) --> (B, T, K, C)
        point_mask = point_mask[:, :, :K] # (B, G, K) --> (B, T, K)
        group_centers = group_centers[:, :T] # (B, G, 3) --> (B, T, 3)
        embedding_mask = embedding_mask[:, :T] # (B, G) --> (B, T)
        if semantic_id_groups is not None:
            semantic_id_groups = semantic_id_groups[:, :T] # (B, G, K) --> (B, T, K)
        if endpoints_groups is not None:
            endpoints_groups = endpoints_groups[:, :T] # (B, G, K, 6) --> (B, T, K, 6)


        # normalize the group centers to the mean of the points in the group
        # instead of just one of the points.
        if self.normalize_group_centers:
            group_centers = masked_mean(groups, point_mask)

        # Normalize group coordinates
        if self.use_relative_features:
            groups = groups - group_centers.unsqueeze(2).expand(-1,-1,K,-1)
        else:
            groups[:, :, :, :3] = groups[:, :, :, :3] - group_centers[..., :3].unsqueeze(2).expand(-1,-1,K,-1)

        if self.group_radius is not None and self.rescale_by_group_radius:
            groups[:, :, :, :3] = (
                groups[:, :, :, :3] / self.group_radius
            )  # proposed by PointNeXT to make relative coordinates less small

        # zero out groups/tokens that are padded. not necessary but helpful for debugging
        groups *= embedding_mask.unsqueeze(-1).unsqueeze(-1)

        out = dict(
            groups=groups, # (B, T, K, C)
            group_centers=group_centers, # (B, T, 3)
            embedding_mask=embedding_mask, # (B, T)
            point_mask=point_mask, # (B, T, K)
            semantic_id_groups=semantic_id_groups, # (B, T, K)
            endpoints_groups=endpoints_groups, # (B, T, K, 6)
            idx=idx, # (B, T, K)
        )
        return out