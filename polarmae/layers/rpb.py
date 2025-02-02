import torch
import torch.nn as nn


class RelativePositionalBias3D(nn.Module):
    def __init__(self, num_bins, bin_size, num_heads):
        """
        num_bins: number of bins in each direction (total range = [-num_bins, num_bins])
        bin_size: size of each bin
        num_heads: number of attention heads
        """
        super().__init__()
        self.num_bins = num_bins
        self.bin_size = bin_size
        self.num_heads = num_heads

        # bias table dimensions: [num_heads, (2*num_bins+1)^3]
        self.bias_table = nn.Parameter(
            torch.zeros(num_heads, (2 * num_bins + 1) ** 3)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)  # initialize biases

    def quantize_relative_positions(self, relative_positions):
        """
        Quantize relative positions into bins.
        relative_positions: Tensor of shape [num_pairs, 3]
        """
        quantized = torch.round(relative_positions / self.bin_size).long()
        quantized = torch.clamp(quantized, -self.num_bins, self.num_bins)
        return quantized

    def relative_position_to_index(self, quantized_positions):
        """
        Convert quantized relative positions into bias table indices.
        quantized_positions: Tensor of shape [num_pairs, 3]
        """
        x, y, z = quantized_positions[:, 0], quantized_positions[:, 1], quantized_positions[:, 2]
        index = (x + self.num_bins) * (2 * self.num_bins + 1) ** 2 + \
                (y + self.num_bins) * (2 * self.num_bins + 1) + \
                (z + self.num_bins)
        return index

    def forward(self, token_centers, return_indices=False):
        """
        Compute relative positional bias for tokens.
        token_centers: Tensor of shape [B, num_tokens, 3]
        """
        B, num_tokens = token_centers.size(0), token_centers.size(1)
        
        # compute pairwise relative positions
        relative_positions = token_centers.unsqueeze(2) - token_centers.unsqueeze(1)  # [B, num_tokens, num_tokens, 3]
        relative_positions = relative_positions.view(-1, 3)  # [B*num_tokens^2, 3]

        # quantize and get table indices
        quantized_positions = self.quantize_relative_positions(relative_positions)
        indices = self.relative_position_to_index(quantized_positions)  # [num_tokens^2]

        # retrieve biases
        biases = self.bias_table[:, indices]  # [num_heads, B*num_tokens^2]

        # reshape biases to [B, num_heads, num_tokens, num_tokens]
        biases = biases.view(self.num_heads, B, num_tokens, num_tokens).permute(1, 0, 2, 3)

        if return_indices:
            return biases, indices
        return biases