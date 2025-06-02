import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pointcloud_predictions(points, true_labels, pred_labels, pred_probs, class_names=None, figsize=(15, 5)):
    """
    Plot pointcloud with true labels, predicted labels, and probability-weighted predictions.
    
    Args:
        points (torch.Tensor): Pointcloud coordinates [N, 3]
        true_labels (torch.Tensor): Ground truth labels [N]
        pred_labels (torch.Tensor): Predicted labels [N]
        pred_probs (torch.Tensor): Prediction probabilities [N, num_classes]
        class_names (list, optional): List of class names for legend
        figsize (tuple, optional): Figure size
    """
    # Convert tensors to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu().numpy()
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()
    

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    palette = ["#79B5A4", "#F5D69B", "#185890", "#BA4A09", "#eeb5ff"]
    label_to_color = {i: palette[i] for i in range(len(palette))}

    # Plot true labels
    for i in range(len(palette)):
        ax1.scatter(
            points[true_labels == i, 0],
            points[true_labels == i, 1],
            points[true_labels == i, 2],
            c=label_to_color[i],
            s=10,
            alpha=1,
        )
    ax1.set_title("True Labels")
    ax1.legend(frameon=False)

    # Plot predicted labels
    for i in range(len(palette)):
        curr_points = points[true_labels == i, :]
        curr_pred = pred_labels[true_labels == i]

        ax2.scatter(
            curr_points[curr_pred == i, 0],
            curr_points[curr_pred == i, 1],
            curr_points[curr_pred == i, 2],
            c=label_to_color[i],
            s=10,
            alpha=1,
        )
    ax2.set_title("Predicted Labels")
    ax2.legend(frameon=False)

    palette_rgb = []
    for color in palette:
        color = color.lstrip('#')
        rgb = np.array([int(color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
        palette_rgb.append(rgb)

    point_colors = []
    for i in range(points.shape[0]):
        weighted_color = np.zeros(3)
        for c in range(len(palette)):
            weighted_color += palette_rgb[c] * pred_probs[i, c]
        point_colors.append(weighted_color)

    ax3.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=point_colors,
        s=10,
        alpha=1,
    )
    ax3.set_title("Predicted Labels Weighted by Logits")

    # Add a legend with the pure colors
    if class_names is None:
        class_names = list(range(len(palette)))
    for c in range(len(palette)):
        ax3.scatter([], [], [], c=palette[c], label=class_names[c])
    ax3.legend(frameon=False)
    plt.tight_layout()

    return fig

def visualize_model_output(points, true_labels, pred_labels, pred_probs, class_names=None, figsize=(15, 5)):
    """
    Visualize model predictions for a pointcloud.
    
    Args:
        points (torch.Tensor): Pointcloud coordinates [N, 3]
        true_labels (torch.Tensor): Ground truth labels [N]
        pred_labels (torch.Tensor): Predicted labels [N]
        pred_probs (torch.Tensor): Prediction probabilities [N, num_classes]
        class_names (list, optional): List of class names for legend
    
    Returns:
        fig, axes: The figure and axes objects for further customization
    """
    return plot_pointcloud_predictions(
        points,
        true_labels,
        pred_labels,
        pred_probs,
        class_names=class_names,
        figsize=figsize
    )

def visualize_batch_item(model_output, batch, batch_idx=0, class_names=None, figsize=(15, 5)):
    """
    Visualize model predictions for a specific item in a batch.
    
    Args:
        model_output (dict): Output from model containing 'id_pred' and 'x'
        batch (dict): Batch dictionary containing 'points', 'lengths', 'semantic_id'
        batch_idx (int): Index of batch item to visualize
        class_names (list, optional): List of class names for legend
    
    Returns:
        fig, axes: The figure and axes objects for further customization
    """
    # Extract data from the batch and model output
    points = batch['points']
    lengths = batch['lengths']
    true_labels = batch['semantic_id'].squeeze(-1)
    
    pred_labels = model_output['id_pred']
    logits = model_output['x']
    pred_probs = logits.softmax(dim=-1)
    
    # Get the specific batch item
    length = lengths[batch_idx].item() if isinstance(lengths, torch.Tensor) else lengths[batch_idx]
    item_points = points[batch_idx, :length]
    item_true_labels = true_labels[batch_idx, :length]
    item_pred_labels = pred_labels[batch_idx, :length]
    item_pred_probs = pred_probs[batch_idx, :length]
    
    return visualize_model_output(
        item_points,
        item_true_labels,
        item_pred_labels,
        item_pred_probs,
        class_names=class_names,
        figsize=figsize
    ) 

def colored_pointcloud_predictions(
    points, true_labels, pred_labels, pred_probs
):
    """
    Plot pointcloud with true labels, predicted labels, and probability-weighted predictions.

    Args:
        points (torch.Tensor): Pointcloud coordinates [N, 3]
        true_labels (torch.Tensor): Ground truth labels [N]
        pred_labels (torch.Tensor): Predicted labels [N]
        pred_probs (torch.Tensor): Prediction probabilities [N, num_classes]
    """
    # Convert tensors to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu().numpy()
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()

    palette = ["#79B5A4", "#F5D69B", "#185890", "#BA4A09", "#eeb5ff"]
    palette_rgb = []
    for color in palette:
        color = color.lstrip("#")
        rgb = np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)]).astype(np.float32)
        palette_rgb.append(rgb)

    # Plot true labels
    truth_points = []
    for i in range(len(palette)):
        curr_points = points[true_labels == i, :]
        curr_colors = np.repeat([palette_rgb[i]], curr_points.shape[0], axis=0)
        if curr_points.shape[0] > 0:
            truth_points.append(np.concatenate([curr_points[...,:3], curr_colors], axis=-1))
    truth_points = np.concatenate(truth_points, axis=0) # (N, 6)

    pred_points = []
    for i in range(len(palette)):
        curr_points = points[pred_labels == i, :]
        curr_colors = np.repeat([palette_rgb[i]], curr_points.shape[0], axis=0)
        if curr_points.shape[0] > 0:
            pred_points.append(np.concatenate([curr_points[...,:3], curr_colors], axis=-1))
    pred_points = np.concatenate(pred_points, axis=0) # (N, 6)

    point_colors = []
    for i in range(points.shape[0]):
        weighted_color = np.zeros(3)
        for c in range(len(palette)):
            weighted_color += palette_rgb[c] * pred_probs[i, c]
        point_colors.append(weighted_color)

    pred_weighted_points = np.concatenate([points[...,:3], point_colors], axis=-1) # (N, 6)

    return truth_points, pred_points, pred_weighted_points


def colored_pointcloud_batch(model_output, batch, batch_idx=0):
    """
    Visualize model predictions for a specific item in a batch.
    """
    points = batch["points"]
    lengths = batch["lengths"]
    true_labels = batch["semantic_id"].squeeze(-1)

    pred_labels = model_output["pred"]
    logits = model_output["logits"]
    pred_probs = logits.softmax(dim=-1)

    length = (
        lengths[batch_idx].item()
        if isinstance(lengths, torch.Tensor)
        else lengths[batch_idx]
    )
    item_points = points[batch_idx, :length]
    item_true_labels = true_labels[batch_idx, :length]
    item_pred_labels = pred_labels[batch_idx, :length]
    item_pred_probs = pred_probs[batch_idx, :length]

    return colored_pointcloud_predictions(
        item_points, item_true_labels, item_pred_labels, item_pred_probs
    )