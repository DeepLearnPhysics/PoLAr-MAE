from typing import Optional

import torch
import torch.nn as nn
from torch import nn


class VariablePointcloudMasking(nn.Module):
    def __init__(self, ratio: float, type: str, overlap_factor: Optional[float] = None, group_radius: Optional[float] = None):
        super().__init__()
        self.ratio = ratio
        self.overlap_factor = overlap_factor
        self.group_radius = group_radius

        if type == "rand":
            self.forward = self._mask_center_rand
        elif type == "block":
            raise NotImplementedError('Block masking is not implemented for variable group masking')
        elif type == "fps+nms":
            self.forward = self._mask_center_fps_nms
        else:
            raise ValueError(f"No such masking type: {type}")

    @torch.no_grad()
    def _mask_center_rand(
        self, centers: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # centers: (B, G, C)
        # Create a mask for valid positions (positions within lengths)
        B, G, _ = centers.shape
        device = centers.device
        valid_positions_mask = torch.arange(G, device=device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)  # Shape: (B, G)
        if self.ratio == 0:
            masked = torch.zeros(centers.shape[:2], device=centers.device, dtype=torch.bool)
            not_masked = torch.zeros_like(masked)
            not_masked[valid_positions_mask] = True
            return masked, not_masked

        # Generate random scores
        random_scores = torch.rand(B, G, device=device)

        # Set random_scores for invalid positions to infinity so they are sorted to the end
        random_scores[~valid_positions_mask] = float("inf")

        # Sort the random scores to simulate random permutations
        sorted_scores, sorted_indices = torch.sort(random_scores, dim=1)

        # Compute the number of tokens to mask per batch
        num_mask = (self.ratio * lengths).int()  # Shape: (B,)

        # Create indices for batch and sequence positions
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, G)
        seq_indices = torch.arange(G, device=device).unsqueeze(0).expand(B, G)

        # Create a mask indicating which positions should be masked
        mask = seq_indices < num_mask.unsqueeze(1)
        mask = mask & valid_positions_mask  # Ensure we only mask valid positions

        # Initialize masked and not_masked tensors
        masked = torch.zeros(B, G, device=device, dtype=torch.bool)
        not_masked = torch.zeros(B, G, device=device, dtype=torch.bool)

        # Assign masked and not_masked positions using advanced indexing
        masked[batch_indices, sorted_indices] = mask
        not_masked[batch_indices, sorted_indices] = (~mask) & valid_positions_mask

        return masked, not_masked  # (B, G)

# https://github.com/allenai/allennlp/blob/main/allennlp/modules/masked_layer_norm.py
class MaskedLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, size: int, gamma0: float = 1.0) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size

    def forward(self, tensor: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        broadcast_mask = mask.unsqueeze(-1)
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt(
            (masked_centered * masked_centered).sum() / num_elements
            + tiny_value_of_dtype(tensor.dtype)
        )
        return (
            self.gamma
            * (tensor - mean)
            / (std + tiny_value_of_dtype(tensor.dtype))
            + self.beta
        )

# https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py
def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double or dtype == torch.bfloat16:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

# from MaskedLayerNorm
def masked_layer_norm(input, normalized_shape, mask, gamma=1.0, beta=0.0):
    """
    Applies layer normalization to the input tensor while ignoring padded tokens.

    Parameters:
    - input (Tensor): Input tensor of shape (B, T, C)
    - normalized_shape (int): Input shape from an expected input of size
    - mask (Tensor): Mask tensor of shape (B, T) where valid tokens are 1 and padded tokens are 0
    - gamma (float, optional): Scaling factor for the normalized tensor (default: 1.0)
    - beta (float, optional): Bias factor for the normalized tensor (default: 0.0)

    Returns:
    - Tensor: The normalized tensor with the same shape as input
    """
    broadcast_mask = mask.unsqueeze(-1)
    num_elements = broadcast_mask.sum() * normalized_shape
    mean = (input * broadcast_mask).sum() / num_elements
    masked_centered = (input - mean) * broadcast_mask
    std = torch.sqrt(
        (masked_centered * masked_centered).sum() / num_elements
        + tiny_value_of_dtype(input.dtype)
    )
    return (
        gamma * (input - mean) / (std + tiny_value_of_dtype(input.dtype)) + beta
    ) * broadcast_mask


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
def masked_drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True, mask: Optional[torch.Tensor] = None):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    """Drop paths (Stochastic Depth) per sample or per token, handling padded sequences.

    Args:
        x: Input tensor of shape (B, T, C)
        drop_prob: Probability of dropping a path.
        training: Whether the model is in training mode.
        scale_by_keep: Whether to scale outputs by the keep probability.
        mask: Optional padding mask of shape (B, T), where True indicates valid tokens.

    Returns:
        Tensor with paths dropped according to the specified drop probability, unaffected padded positions.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    # Generate random tensor for dropping paths
    # Shape: (B, T, 1)
    random_tensor = x.new_empty((x.shape[0], x.shape[1], 1)).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    # Apply the random tensor to x
    x = x * random_tensor

    # If mask is provided, ensure that padded positions remain zero
    if mask is not None:
        mask = mask.unsqueeze(-1).to(x.dtype)  # Shape: (B, T, 1)
        x = x * mask

    return x

# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
class MaskedDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MaskedDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return masked_drop_path(x, self.drop_prob, self.training, self.scale_by_keep, mask)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class MaskedBatchNorm1d(nn.Module):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, **factory_kwargs))  # Gamma
            self.bias = nn.Parameter(torch.zeros(num_features, **factory_kwargs))  # Beta
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.eps = eps
        self.momentum = momentum

        # Running stats
        self.register_buffer("running_mean", torch.zeros(num_features, **factory_kwargs))
        self.register_buffer("running_var", torch.ones(num_features, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, mask=None):
        # x: (B, C, L)
        # mask: (B, 1, L)
        if mask is None:
            mask = torch.ones_like(x[:, 0, :], device=x.device)
        B, C, L = x.size()

        # Ensure mask has the correct shape and type
        # mask: (B, 1, L), dtype=torch.float32
        mask = mask.float()

        # Compute the total number of valid elements (scalar)
        valid_elements = mask.sum()  # Scalar

        # Avoid division by zero
        valid_elements = valid_elements.clamp(min=1)

        # Compute the mean over valid elements
        # Sum over batch and length dimensions
        sum_x = (x * mask).sum(dim=(0, 2))  # Shape: (C,)
        mean = sum_x / valid_elements  # Shape: (C,)

        # Center the inputs
        x = x - mean.view(1, C, 1)

        # Compute the variance over valid elements
        var = ((x * mask) ** 2).sum(dim=(0, 2)) / valid_elements  # Shape: (C,)

        # Update running statistics
        if self.training:
            with torch.no_grad():
                momentum = self.momentum
                self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
                self.running_var = (1 - momentum) * self.running_var + momentum * var
        else:
            x = x + mean.view(1, C, 1)
            # Use running stats during evaluation
            mean = self.running_mean
            var = self.running_var

            # Recompute x_centered with updated mean
            x = x - mean.view(1, C, 1)

        # Normalize
        x = (
            x / torch.sqrt(var + self.eps).view(1, C, 1)
        ) * mask  # Multiply by mask to zero out padded positions

        # Apply affine transformation if enabled
        if self.affine:
            x = x * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)

        return x

def masked_mean(group, point_mask):
    valid_elements = point_mask.sum(-1).float().clamp(min=1)
    return (group * point_mask.unsqueeze(-1)).sum(-2) / valid_elements.unsqueeze(-1)

def masked_max(group, point_mask, max_val=1e10):
    return group.masked_fill(~point_mask.unsqueeze(-1), -max_val).max(-2).values
