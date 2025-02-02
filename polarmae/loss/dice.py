
import torch
import torch.nn.functional as F
import torch.nn as nn

__all__ = ['DiceLoss']

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Computes the Dice Loss for multi-class segmentation or classification tasks.

        Args:
            inputs: Tensor of shape (N, C, ...) where C = number of classes.
                    This tensor contains raw scores (logits) for each class.
            targets: Tensor of shape (N, ...) with integer values in the range [0, C-1].
                     This tensor contains the ground truth class indices.

        Returns:
            dice_loss: Scalar tensor representing the Dice Loss.
        """
        # Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)  # Shape: (N, C, ...)

        # Ensure targets have the same spatial dimensions as inputs
        if inputs.dim() > 2:
            targets = targets.unsqueeze(1)  # Shape: (N, 1, ...)
            targets = targets.expand(
                -1, inputs.size(2), *([-1] * (inputs.dim() - 2))
            )  # Shape: (N, C, ...)

        # Flatten inputs and targets to compute per-pixel/per-element Dice coefficients
        inputs = inputs.contiguous().view(
            inputs.size(0), inputs.size(1), -1
        )  # Shape: (N, C, D)
        targets = targets.contiguous().view(targets.size(0), -1)  # Shape: (N, D)

        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.clamp(0), num_classes)  # Shape: (N, D, C)
        targets_one_hot = targets_one_hot.permute(0, 2, 1).float()  # Shape: (N, C, D)

        # Handle ignore_index if specified
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index  # Shape: (N, D)
            valid_mask = valid_mask.unsqueeze(1).expand_as(inputs)  # Shape: (N, C, D)
            inputs = inputs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(2)  # Shape: (N, C)
        inputs_sum = inputs.sum(2)  # Shape: (N, C)
        targets_sum = targets_one_hot.sum(2)  # Shape: (N, C)

        dice_coeff = (2.0 * intersection + self.smooth) / (
            inputs_sum + targets_sum + self.smooth
        )  # Shape: (N, C)
        dice_loss = 1 - dice_coeff  # Shape: (N, C)

        # Mean over classes and batch
        dice_loss = dice_loss.mean()

        return dice_loss
