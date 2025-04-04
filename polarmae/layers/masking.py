from typing import Optional

import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.function import Function


def masked_mean(group, point_mask):
    valid_elements = point_mask.sum(-1).float().clamp(min=1)
    return (group * point_mask.unsqueeze(-1)).sum(-2) / valid_elements.unsqueeze(-1)


def masked_max(group, point_mask, max_val=1e10):
    return group.masked_fill(~point_mask.unsqueeze(-1), -max_val).max(-2).values


class VariablePointcloudMasking(nn.Module):
    def __init__(self, ratio: float, type: str, overlap_factor: Optional[float] = None, group_radius: Optional[float] = None):
        super().__init__()
        self.ratio = ratio
        self.overlap_factor = overlap_factor
        self.group_radius = group_radius

        if type == "rand":
            self.forward = self._mask_center_rand
        elif type == "rand_reindex":
            self.forward = self._mask_center_rand_reindex
        elif type == "block":
            raise NotImplementedError('Block masking is not implemented for variable group masking')
        elif type == "fps+nms":
            self.forward = self._mask_center_fps_nms
        else:
            raise ValueError(f"No such masking type: {type}")

    # @torch.no_grad()
    # def _mask_center_rand(
    #     self, lengths: torch.Tensor
    # ) -> torch.Tensor:
    #     # centers: (B, G, C)
    #     # Create a mask for valid positions (positions within lengths)
    #     B, = lengths.shape
    #     G = lengths.max()
    #     device = lengths.device
    #     valid_positions_mask = torch.arange(G, device=device).unsqueeze(
    #         0
    #     ) < lengths.unsqueeze(1)  # Shape: (B, G)
    #     if self.ratio == 0:
    #         masked = torch.zeros(B, G, device=device, dtype=torch.bool)
    #         not_masked = torch.zeros_like(masked)
    #         not_masked[valid_positions_mask] = True
    #         return masked, not_masked

    #     # Generate random scores
    #     random_scores = torch.rand(B, G, device=device)

    #     # Set random_scores for invalid positions to infinity so they are sorted to the end
    #     random_scores[~valid_positions_mask] = float("inf")

    #     # Sort the random scores to simulate random permutations
    #     sorted_scores, sorted_indices = torch.sort(random_scores, dim=1)

    #     # Compute the number of tokens to mask per batch
    #     num_mask = (self.ratio * lengths).int()  # Shape: (B,)

    #     # Create indices for batch and sequence positions
    #     batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, G)
    #     seq_indices = torch.arange(G, device=device).unsqueeze(0).expand(B, G)

    #     # Create a mask indicating which positions should be masked
    #     mask = seq_indices < num_mask.unsqueeze(1)
    #     mask = mask & valid_positions_mask  # Ensure we only mask valid positions

    #     # Initialize masked and not_masked tensors
    #     masked = torch.zeros(B, G, device=device, dtype=torch.bool)
    #     not_masked = torch.zeros(B, G, device=device, dtype=torch.bool)

    #     # Assign masked and not_masked positions using advanced indexing
    #     masked[batch_indices, sorted_indices] = mask
    #     not_masked[batch_indices, sorted_indices] = (~mask) & valid_positions_mask

    #     return masked, not_masked  # (B, G)
    

    def _mask_center_rand(self, lengths: torch.Tensor):
        """
        Given centers of shape (B, G, C) and lengths (B,) indicating how many positions
        in each batch are valid, this function computes a random masking (with ratio self.ratio)
        and returns, for each batch:
        - masked_indices: (B, max_mask) long tensor of positions (in random order) for masked tokens,
                            padded with -1's beyond each batch's valid count.
        - masked_attn: (B, max_mask) bool tensor with True for real masked tokens.
        - unmasked_indices: (B, max_unmask) long tensor of positions for unmasked tokens,
                            padded with -1's similarly.
        - unmasked_attn: (B, max_unmask) bool tensor with True for real unmasked tokens.
        
        To extract token features from centers, you can do:
        masked_tokens = torch.gather(
            centers,
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, centers.size(-1))
        )
        unmasked_tokens = torch.gather(
            centers,
            1,
            unmasked_indices.unsqueeze(-1).expand(-1, -1, centers.size(-1))
        )
        
        (Note: For batches with self.ratio==0, masked tokens will be empty.)
        """
        B, = lengths.shape
        G = lengths.max()
        device = lengths.device

        # Compute valid positions per batch (positions [0, lengths[b]) are valid)
        valid_positions = torch.arange(G, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, G)

        # --- Handle the trivial case where no tokens are masked ---
        if self.ratio == 0:
            max_valid = lengths.max().item()
            all_indices = torch.arange(G, device=device).unsqueeze(0).expand(B, G)
            unmasked_indices = all_indices[:, :max_valid].clone()
            unmasked_attn = torch.arange(max_valid, device=device).unsqueeze(0).expand(B, max_valid) < lengths.unsqueeze(1)
            # Set unused unmasked indices to -1
            unmasked_indices[~unmasked_attn] = -1

            # No masked tokens at all (empty tensors)
            masked_indices = torch.empty(B, 0, device=device, dtype=torch.long)
            masked_attn = torch.empty(B, 0, device=device, dtype=torch.bool)
            return masked_indices, masked_attn, unmasked_indices, unmasked_attn

        # --- Compute a random permutation over valid tokens ---
        random_scores = torch.rand(B, G, device=device)
        # Push invalid positions (>= lengths[b]) to the end
        random_scores[~valid_positions] = float("inf")
        # Sorted indices so that valid positions come first in random order.
        _, sorted_indices = torch.sort(random_scores, dim=1)  # sorted_indices: (B, G)

        # --- Decide how many tokens to mask per batch ---
        num_mask = (self.ratio * lengths).to(torch.long)  # (B,)
        max_mask = num_mask.max()  # maximum masked tokens across batch

        # For each batch, the first num_mask[b] indices in sorted_indices are the masked tokens.
        masked_indices = sorted_indices[:, :max_mask].clone()  # (B, max_mask)
        # Build an attention mask: each row is True for positions < num_mask[b] and False otherwise.
        masked_mask = torch.arange(max_mask, device=device).unsqueeze(0).expand(B, max_mask) < num_mask.unsqueeze(1)
        # Set unused masked indices to -1
        # masked_indices[~masked_mask] = -1

        # --- Compute unmasked indices ---
        num_unmask = lengths - num_mask  # (B,)
        max_unmask = num_unmask.max()

        # Create an index offset for each batch: for row b, we want indices from num_mask[b] to num_mask[b]+num_unmask[b]-1.
        idx_range = torch.arange(max_unmask, device=device).unsqueeze(0).expand(B, max_unmask)  # (B, max_unmask)
        # Add the per-batch starting index (num_mask) so that for batch b we select sorted_indices[b, num_mask[b] + i].
        unmask_select_idx = num_mask.unsqueeze(1) + idx_range  # (B, max_unmask)
        # (Clamp not strictly necessary because idx_range is built from num_unmask, but just in case)
        unmask_select_idx = unmask_select_idx.clamp(max=G - 1)
        unmasked_indices = torch.gather(sorted_indices, 1, unmask_select_idx).clone()  # (B, max_unmask)
        unmasked_mask = idx_range < num_unmask.unsqueeze(1)
        # Set unused unmasked indices to -1
        # unmasked_indices[~unmasked_mask] = -1

        return masked_indices, masked_mask, unmasked_indices, unmasked_mask



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
    


class SyncMaskedBatchNormFunction(Function):
    @staticmethod
    def forward(
        self,
        input,
        mask,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        process_group,
        world_size,
    ):
        # Make input contiguous if not already
        if not input.is_contiguous():
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        B, C, L = input.size()

        # Ensure mask has correct shape and type
        if mask is None:
            mask = torch.ones_like(input[:, 0, :], device=input.device)
        mask = mask.float()

        # Compute the total number of valid elements per channel
        valid_elements = mask.sum()  # Scalar
        valid_elements = valid_elements.clamp(min=1)

        if valid_elements == 1 and world_size < 2:
            raise ValueError(
                f"Expected more than 1 value per channel when training, got {valid_elements}"
            )

        if input.numel() > 0:
            # Compute local mean and variance
            sum_x = (input * mask.unsqueeze(1)).sum(dim=(0, 2))  # Shape: (C,)
            mean = sum_x / valid_elements  # Shape: (C,)

            # Center the inputs
            centered_input = input - mean.view(1, C, 1)

            # Compute the variance over valid elements
            var_x = ((centered_input * mask.unsqueeze(1)) ** 2).sum(
                dim=(0, 2)
            ) / valid_elements  # Shape: (C,)
            invstd = torch.rsqrt(var_x + eps)

            # Prepare data for all-gather
            count = torch.full(
                (1,),
                valid_elements,
                dtype=mean.dtype,
                device=mean.device,
            )

            # Combine statistics for synchronization
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            # For empty input, set stats and count to zero
            combined = torch.zeros(2 * C + 1, dtype=input.dtype, device=input.device)

        # All-gather statistics from all processes
        if process_group._get_backend_name() != "gloo":
            # Use optimized gather for NCCL backend
            combined_size = combined.numel()
            combined_flat = torch.empty(
                1,
                combined_size * world_size,
                dtype=combined.dtype,
                device=combined.device,
            )
            dist.all_gather_into_tensor(
                combined_flat, combined, process_group, async_op=False
            )
            combined = torch.reshape(combined_flat, (world_size, combined_size))
        else:
            # For gloo backend
            combined_list = [torch.empty_like(combined) for _ in range(world_size)]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)

        # Split the combined tensor back into mean, invstd, and count
        mean_all, invstd_all, count_all = torch.split(combined, C, dim=1)

        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            # Remove stats from empty inputs
            mask_valid = count_all.squeeze(-1) >= 1
            count_all = count_all[mask_valid]
            mean_all = mean_all[mask_valid]
            invstd_all = invstd_all[mask_valid]

        # Calculate global mean & invstd
        if count_all.numel() > 0:
            total_count = count_all.sum()
            global_mean = (mean_all * count_all).sum(dim=0) / total_count

            # Calculate global variance
            var_term = count_all * (1.0 / invstd_all**2 + (mean_all - global_mean) ** 2)
            global_var = var_term.sum(dim=0) / total_count
            global_invstd = torch.rsqrt(global_var + eps)
        else:
            global_mean = torch.zeros_like(mean)
            global_invstd = torch.ones_like(invstd)

        # Update running statistics
        if self.training and running_mean is not None:
            running_mean.mul_(1 - momentum).add_(global_mean * momentum)
            running_var.mul_(1 - momentum).add_((1.0 / global_invstd**2) * momentum)

        # Save for backward
        self.save_for_backward(
            input,
            mask,
            weight,
            global_mean,
            global_invstd,
            valid_elements.to(torch.int32),
        )
        self.process_group = process_group

        # Normalize
        if input.numel() > 0:
            # Adjust input using global statistics
            normalized = centered_input * global_invstd.view(1, C, 1)
            normalized = normalized * mask.unsqueeze(1)

            # Apply weight and bias if provided
            if weight is not None and bias is not None:
                normalized = normalized * weight.view(1, C, 1) + bias.view(1, C, 1)

            return normalized
        else:
            return torch.empty_like(input)

    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        saved_input, mask, weight, mean, invstd, count_tensor = self.saved_tensors
        process_group = self.process_group

        grad_input = grad_weight = grad_bias = grad_mask = None
        B, C, L = saved_input.size()

        if saved_input.numel() > 0:
            # Apply mask to the gradient
            masked_grad = grad_output * mask.unsqueeze(1)

            # Calculate grad_bias (sum over batch and length dimensions)
            if self.needs_input_grad[3]:  # bias
                grad_bias = masked_grad.sum(dim=(0, 2))

            # Calculate grad_weight
            centered_input = saved_input - mean.view(1, C, 1)
            if self.needs_input_grad[2]:  # weight
                grad_weight = (masked_grad * centered_input * invstd.view(1, C, 1)).sum(
                    dim=(0, 2)
                )

            if self.needs_input_grad[0]:  # input
                # Calculate local statistics for gradient
                N = count_tensor.item()

                # Calculate sum_dy and sum_dy_xmu for gradient computation
                sum_dy = masked_grad.sum(dim=(0, 2))
                sum_dy_xmu = (masked_grad * centered_input).sum(dim=(0, 2))

                # Synchronize these stats across processes
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                dist.all_reduce(
                    combined, dist.ReduceOp.SUM, process_group, async_op=False
                )
                sum_dy, sum_dy_xmu = torch.split(combined, C)

                # Compute input gradient
                mean_dy = sum_dy / N
                mean_dy_xmu = sum_dy_xmu / N

                # Gradient formula: https://kevinzakka.github.io/2016/09/14/batch_normalization/
                grad_input = mask.unsqueeze(1) * (
                    invstd.view(1, C, 1)
                    * (
                        masked_grad
                        - mean_dy.view(1, C, 1)
                        - centered_input
                        * invstd.view(1, C, 1)
                        * mean_dy_xmu.view(1, C, 1)
                        / N
                    )
                )

                if weight is not None:
                    grad_input = grad_input * weight.view(1, C, 1)

            if self.needs_input_grad[1]:  # mask
                # Gradient for the mask
                # This is complex as mask affects both forward and statistics calculation
                # Simplified approximation (might not be theoretically perfect)
                if weight is not None:
                    grad_mask = (
                        grad_output
                        * weight.view(1, C, 1)
                        * centered_input
                        * invstd.view(1, C, 1)
                    ).sum(dim=1)
                else:
                    grad_mask = (
                        grad_output * centered_input * invstd.view(1, C, 1)
                    ).sum(dim=1)
        else:
            # If input is empty, still need to participate in the collective communication
            if self.needs_input_grad[0]:
                combined = torch.zeros(
                    2 * C, dtype=saved_input.dtype, device=saved_input.device
                )
                dist.all_reduce(
                    combined, dist.ReduceOp.SUM, process_group, async_op=False
                )

        return (
            grad_input,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SyncMaskedBatchNorm1d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        process_group=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SyncMaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.process_group = process_group

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_var", torch.ones(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long, **factory_kwargs),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x, mask=None):
        # x: (B, C, L)
        # mask: (B, 1, L) or (B, L)

        # Handle mask dimensions
        if mask is not None and mask.dim() == 2:
            # If mask is (B, L), unsqueeze to (B, 1, L)
            mask = mask.unsqueeze(1)
        elif mask is not None and mask.dim() > 2:
            # Make sure it's the right shape
            if mask.size(1) != 1:
                mask = mask[:, 0:1, :]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # Update running stats during training
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # exponential moving average
                    exponential_average_factor = self.momentum

        # Determine if we need to use batch statistics
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # If not tracking running stats, set them to None for the forward pass
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Determine if synchronization is needed
        need_sync = (
            bn_training
            and self.training
            and dist.is_available()
            and dist.is_initialized()
        )

        if need_sync:
            # Check if input is on GPU (required for synchronization)
            if x.device.type not in ["cuda", torch._C._get_privateuse1_backend_name()]:
                raise ValueError(
                    "SyncMaskedBatchNorm expected input tensor to be on GPU or "
                    f"{torch._C._get_privateuse1_backend_name()}"
                )

            # Set process group
            process_group = dist.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = dist.get_world_size(process_group)
            need_sync = world_size > 1

        # Fallback to non-sync if synchronization is not needed
        if not need_sync:
            # Perform normal masked batch normalization
            B, C, L = x.size()

            # Create default mask if not provided
            if mask is None:
                mask = torch.ones_like(x[:, 0, :], device=x.device)
            mask = mask.float()

            # Sum valid elements
            valid_elements = mask.sum().clamp(min=1)

            if bn_training:
                # Compute mean over valid elements
                sum_x = (x * mask.unsqueeze(1)).sum(dim=(0, 2))  # Shape: (C,)
                mean = sum_x / valid_elements  # Shape: (C,)

                # Center the inputs
                centered_x = x - mean.view(1, C, 1)

                # Compute variance over valid elements
                var = ((centered_x * mask.unsqueeze(1)) ** 2).sum(
                    dim=(0, 2)
                ) / valid_elements  # Shape: (C,)

                # Update running stats if needed
                if self.track_running_stats and self.training:
                    self.running_mean.mul_(1 - exponential_average_factor).add_(
                        mean * exponential_average_factor
                    )
                    self.running_var.mul_(1 - exponential_average_factor).add_(
                        var * exponential_average_factor
                    )
            else:
                # Use running stats
                mean = self.running_mean
                var = self.running_var
                centered_x = x - mean.view(1, C, 1)

            # Normalize using computed/saved statistics
            x_norm = centered_x / torch.sqrt(var.view(1, C, 1) + self.eps)
            x_norm = x_norm * mask.unsqueeze(
                1
            )  # Apply mask to zero out padded positions

            # Apply affine transform if enabled
            if self.affine:
                x_norm = x_norm * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)

            return x_norm
        else:
            # Use synchronized batch normalization
            return SyncMaskedBatchNormFunction.apply(
                x,
                mask,
                self.weight if self.affine else None,
                self.bias if self.affine else None,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,
                world_size,
            )

    @staticmethod
    def convert_sync_masked_batchnorm(module, process_group=None):
        """
        Converts MaskedBatchNorm1d layers in the model to SyncMaskedBatchNorm1d.

        Args:
            module (nn.Module): Module containing MaskedBatchNorm1d layers
            process_group (optional): Process group for synchronization

        Returns:
            The module with converted batch norm layers
        """
        print(f"Converting MaskedBatchNorm1d to SyncMaskedBatchNorm1d!")
        module_output = module
        if isinstance(module, MaskedBatchNorm1d):
            module_output = SyncMaskedBatchNorm1d(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                hasattr(module, "running_mean"),  # True if tracking stats
                process_group,
                module.weight.device if module.affine else None,
                module.weight.dtype if module.affine else None,
            )

            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias

            if hasattr(module, "running_mean"):
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var

            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(
                name,
                SyncMaskedBatchNorm1d.convert_sync_masked_batchnorm(
                    child, process_group
                ),
            )

        del module
        return module_output
