from typing import Dict, List

import torch


def compute_shape_ious(
    log_probabilities: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    category_to_seg_classes: Dict[str, int],
    seg_class_to_category: Dict[int, str],
) -> Dict[str, List[torch.Tensor]]:
    # log_probabilities: (B, N, N_cls) in -inf..<0
    # labels:            (B, N) in 0..<N_cls
    # returns:           { category: List of IoUs for each instance }

    shape_ious: Dict[str, List[torch.Tensor]] = {
        cat: [] for cat in category_to_seg_classes.keys()
    }

    for i in range(log_probabilities.shape[0]):
        if lengths[i] == 0:
            continue
        curr_logprobs = log_probabilities[i][: lengths[i]]
        curr_seg_labels = labels[i][: lengths[i]]
        seg_preds = torch.argmax(curr_logprobs, dim=1)  # (N,)

        # Iterate over each segmentation class
        for c in seg_class_to_category.keys():
            # Create masks for the ground truth and predictions for class c
            gt_mask = curr_seg_labels == c
            pred_mask = seg_preds == c

            # If the class is absent in both, skip to avoid inflating mIoU
            if gt_mask.sum() == 0 and pred_mask.sum() == 0:
                continue

            intersection = (gt_mask & pred_mask).sum().float()
            union = (gt_mask | pred_mask).sum().float()
            iou = intersection / union if union > 0 else float("nan")

            shape_ious[seg_class_to_category[c]].append(iou)
    return shape_ious


# def compute_shape_ious(
#         log_probabilities: torch.Tensor,
#         labels: torch.Tensor, lengths: torch.Tensor,
#         category_to_seg_classes: Dict[str, int],
#         seg_class_to_category: Dict[int, str]
#     ) -> Dict[str, List[torch.Tensor]]:
#     # log_probabilities: (B, N, N_cls) in -inf..<0
#     # labels:            (B, N) in 0..<N_cls
#     # returns:           { category: List of IoUs for each instance }

#     shape_ious: Dict[str, List[torch.Tensor]] = {
#         cat: [] for cat in category_to_seg_classes.keys()
#     }

#     for i in range(log_probabilities.shape[0]):
#         if lengths[i] == 0:
#             continue
#         curr_logprobs = log_probabilities[i][: lengths[i]]
#         curr_seg_labels = labels[i][: lengths[i]]
#         seg_preds = torch.argmax(curr_logprobs, dim=1)  # (N,)

#         # Iterate over each segmentation class
#         for c in seg_class_to_category.keys():
#             # Create masks for the ground truth and predictions for class c
#             gt_mask = (curr_seg_labels == c)
#             pred_mask = (seg_preds == c)
            
#             # If the class is absent in both, skip to avoid inflating mIoU
#             if gt_mask.sum() == 0 and pred_mask.sum() == 0:
#                 continue

#             intersection = (gt_mask & pred_mask).sum().float()
#             union = (gt_mask | pred_mask).sum().float()
#             iou = intersection / union if union > 0 else float('nan')

#             shape_ious[seg_class_to_category[c]].append(iou)
#     return shape_ious
